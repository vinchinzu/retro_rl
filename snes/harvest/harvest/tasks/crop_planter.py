"""
Crop planting task — CropWaterTask composer + dual-FSM (detect/plant/water/refill).

Remaining mixins:
  - ``crop_geometry`` — plot/water geometry (pond/refill stands in water_refill)
  - ``crop_establish`` — detect / plot lifecycle + hoe + plant
  - ``crop_water_ops`` — water-step, residual recovery, center/act/verify/tool_switch
  - ``crop_refill`` — can refill / pond access / corridor thrash mixin
  - ``crop_navigate`` — multi-phase navigate / stuck recovery
  - ``pond_*`` — corridor charges, hop densify, policy
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set, Tuple

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from harvest.core.ram_catalog import read_ram_value
from harvest.core.carry import (
    ADDR_TOOL_BACKPACK,
    SEED_ITEM,
    carry_pair_items,
    seed_in_carry_pair as seed_item_in_carry_pair,
    watering_can_in_carry_pair,
)
from harvest.core.tile_catalog import ADDR_INPUT_LOCK
from harvest.tasks.nav import Pathfinder, Navigator, get_tile_at, make_action
from harvest.tasks.farm_ops import TileScanner, ToolManager
from harvest.tasks.water_refill import REFILL_PREFERRED_WATER_TILES, crop_completion_status
from harvest.tasks.pond_hop import PondCorridorController

# Public re-exports (stable import path for tests / day-plan / scripts).
from harvest.tasks.crop_geometry import (  # noqa: F401
    hoe_plan,
    water_plan,
    hoe_action_sequence,
    water_action_sequence,
    center_water_all,
    plant_action_sequence,
    refill_action_sequence,
    is_bad_refill_stand,
    is_main_pond_stand,
    refill_stand_band,
    edge_water_tile_id,
    refill_edge_sort_key,
    pond_access_blocking_fences,
    find_pond_edges,
    nearest_pond_edge,
    plot_tiles,
    count_tilled,
    count_needs_water,
    is_crop_tile,
    is_dry_crop_tile,
    is_watered_crop_tile,
    crop_pickup_stage,
    is_mature_crop_tile,
    tile_is_watered,
    tile_needs_watering,
    count_crop_survival,
    tile_can_be_water_target,
    is_rainy_weather,
    _count_plot_tiles,
    _refine_center,
    detect_plots,
    _count_crop_tiles,
    _merge_plot_centers,
    detect_crop_resume_plots,
    _water_target_tiles,
    _preferred_outward_faces,
    _water_step_variants,
    build_water_steps,
    FRESH_TILLED,
    DRIED_TILLED,
    WATERED_TILLED,
    UNTILLED,
    TILLABLE_TILES,
    PLANTABLE_TILES,
    WATER_TILES,
    REFILL_WATER_TILES,
    BAD_REFILL_STAND_BOUNDS,
    REFILL_BAND_POND,
    REFILL_BAND_SOUTH,
    REFILL_BAND_NORTH,
    REFILL_BAND_MID,
    REFILL_BAND_BAD,
    MAIN_POND_STAND_BOUNDS,
    HOE_PLAN,
    WATER_PLAN_CENTER,
    WATER_PLAN,
    CROP_TILE_RANGE,
    DRY_CROP_TILES,
    WET_CROP_TILES,
    MATURE_CROP_TILES,
    PLOT_TILES,
    UNRIPE_DRY_CROP_TILES,
    WATERABLE_TILES,
    DEFAULT_CROP_BOUNDS,
    ADDR_WATER_LEVEL,
    WATER_LEVEL_MAX,
    WATER_REFILL_THRESHOLD,
    ADDR_WEATHER,
    ADDR_WEATHER_FLAGS,
    RAINY_WEATHER_CODES,
    RAINY_WEATHER_FLAG_MASK,
)


class CropState(str, Enum):
    DETECT = "detect"
    NAVIGATE = "navigate"
    CENTER = "center"
    ACT = "act"
    VERIFY = "verify"
    TOOL_SWITCH = "tool_switch"
    FENCE_OPEN = "fence_open"
    DONE = "done"


class PlotPhase(str, Enum):
    PLANT = "plant"
    HOE = "hoe"
    WATER = "water"
    REFILL = "refill"
    STAGE_POND = "stage_pond"
    OPEN_POND = "open_pond"


# Membership sets for position / timeout policy (not thrash bands).
ON_APPROACH_PHASES = frozenset(
    {
        PlotPhase.PLANT,
        PlotPhase.WATER,
        PlotPhase.HOE,
        PlotPhase.STAGE_POND,
    }
)
# Soft-timeout owners for pond access (fence_open is CropState, not a phase).
POND_ACCESS_PHASES = frozenset(
    {
        PlotPhase.OPEN_POND,
        PlotPhase.STAGE_POND,
    }
)

WORK_MODE_FULL = "full"
WORK_MODE_ESTABLISH = "establish"
WORK_MODE_WATER = "water"

VALID_WORK_MODES = frozenset({WORK_MODE_FULL, WORK_MODE_ESTABLISH, WORK_MODE_WATER})

# Remaining mixins import CropState from this module; define enums first.
from harvest.tasks.crop_establish import CropEstablishMixin  # noqa: E402
from harvest.tasks.crop_water_ops import CropWaterOpsMixin  # noqa: E402
from harvest.tasks.crop_refill import CropRefillMixin  # noqa: E402
from harvest.tasks.crop_navigate import CropNavigateMixin  # noqa: E402

# Re-export carry helpers under the historical crop_planter names.
__all__ = [
    "CropWaterTask",
    "CropState",
    "PlotPhase",
    "DEFAULT_CROP_BOUNDS",
    "ADDR_TOOL_BACKPACK",
    "SEED_ITEM",
    "carry_pair_items",
    "seed_item_in_carry_pair",
    "watering_can_in_carry_pair",
]


@dataclass
class CropWaterTask(
    CropEstablishMixin,
    CropWaterOpsMixin,
    CropRefillMixin,
    CropNavigateMixin,
    Task,
):
    """Detect crop plots, plant seeds on tilled tiles, water all crops.

    Follows the GrassPlantTask state machine pattern:
      detect -> navigate -> center -> act -> verify -> tool_switch

    ``work_mode`` splits the two-slot plant vs water ceremony:
      - establish: hoe + plant only (day-plan plant pass with seeds+hoe)
      - water: water existing plots only (day-plan can pass)
      - full: plant then water in one run (legacy / manual crop mode)

    Fixes vs v1:
      - Planting: explicit tile position check (must be ON center tile)
      - Watering: waters all 8 tiles blindly, tracks per-plot 8/8
      - Refill: RAM-based (reads actual water level at 0x0926), verifies success
      - Center detection: refined with offset search to fix alignment
    """

    name: str = "crop_water"
    seed_type: str = "potato"
    work_mode: str = WORK_MODE_FULL
    bounds: Tuple[int, int, int, int] = DEFAULT_CROP_BOUNDS
    max_steps_per_target: int = 1200
    stasis_repath: int = 180
    max_failures: int = 50
    refill_bounds: Optional[Tuple[int, int, int, int]] = None
    skip_water_tiles: Set[Tuple[int, int]] = field(default_factory=set)
    debug: bool = False
    debug_interval: int = 300

    # Internal components
    _scanner: TileScanner = field(default_factory=TileScanner, init=False)
    _pathfinder: Pathfinder = field(init=False)
    _navigator: Navigator = field(init=False)
    _tool_mgr: ToolManager = field(default_factory=ToolManager, init=False)

    # Plot list
    _plots: List[Tuple[int, int]] = field(default_factory=list, init=False)
    _plot_index: int = field(default=0, init=False)
    _pass_number: int = field(default=1, init=False)  # 1=first pass, 2=verification pass

    # State machine
    _state: CropState = field(default=CropState.DETECT, init=False)
    _action_queue: deque = field(default_factory=deque, init=False)
    _steps_on_target: int = field(default=0, init=False)
    _total_steps: int = field(default=0, init=False)
    _failures: int = field(default=0, init=False)
    _failed_tiles: Set[Tuple[int, int]] = field(default_factory=set, init=False)

    # Per-plot phase tracking
    _plot_phase: PlotPhase = field(default=PlotPhase.PLANT, init=False)
    _water_steps: List[Tuple[Tuple[int, int], Tuple[int, int], str]] = field(default_factory=list, init=False)
    _water_index: int = field(default=0, init=False)
    _plot_watered: int = field(default=0, init=False)   # per-plot water count
    _plot_skipped: int = field(default=0, init=False)   # per-plot skip count
    _allow_unknown_water_tiles: bool = field(default=False, init=False)
    _allow_crop_walkable: bool = field(default=False, init=False)
    _target_tile: Optional[Tuple[int, int]] = field(default=None, init=False)
    _approach_tile: Optional[Tuple[int, int]] = field(default=None, init=False)
    _face_direction: Optional[str] = field(default=None, init=False)

    # Refill state
    _resume_water_index: int = field(default=0, init=False)
    _refill_pond_tile: Optional[Tuple[int, int]] = field(default=None, init=False)
    _refill_pond_face: Optional[str] = field(default=None, init=False)
    _refill_level_before: int = field(default=0, init=False)  # water level before refill attempt
    _refill_search_level: int = field(default=-1, init=False)  # water level when refill search started
    _bad_refill_tiles: Set[Tuple[int, int]] = field(default_factory=set, init=False)  # tiles that didn't work
    _refill_exhausted: bool = field(default=False, init=False)  # no more refill sources available
    _fence_subtask: Optional[Task] = field(default=None, init=False)
    _fence_open_attempts: int = field(default=0, init=False)
    _refill_nav_failures: int = field(default=0, init=False)
    _refill_multihop: bool = field(default=False, init=False)
    _refill_best_dist: int = field(default=999, init=False)
    _pending_multihop_after_drop: bool = field(default=False, init=False)
    # Pond corridor thrash / scripted-charge state (rr-ds3 extraction).
    _corridor: PondCorridorController = field(
        default_factory=PondCorridorController, init=False
    )

    # Water verification
    _pre_water_level: int = field(default=-1, init=False)  # water level before watering action
    _last_water_level_before: int = field(default=-1, init=False)
    _last_water_tile_before: int = field(default=-1, init=False)
    _water_verify_retries: int = field(default=0, init=False)

    # Counters
    planted_count: int = field(default=0, init=False)
    watered_count: int = field(default=0, init=False)
    skipped_water: int = field(default=0, init=False)
    refill_count: int = field(default=0, init=False)
    # Acceptance tracking — harden SUCCESS so false greens do not pollute journals.
    _dry_crop_tiles_at_start: int = field(default=0, init=False)
    _had_seed_stock_at_start: bool = field(default=False, init=False)
    _acceptance_snapped: bool = field(default=False, init=False)
    # Planned centers that failed hoe/path — avoid infinite redetect loops.
    _rejected_plan_centers: Set[Tuple[int, int]] = field(default_factory=set, init=False)

    def __post_init__(self):
        self._pathfinder = Pathfinder(self._scanner)
        self._navigator = Navigator(self._pathfinder)
        mode = (self.work_mode or WORK_MODE_FULL).strip().lower()
        if mode not in VALID_WORK_MODES:
            raise ValueError(
                f"CropWaterTask.work_mode must be one of "
                f"{sorted(VALID_WORK_MODES)!r}; "
                f"got {self.work_mode!r}"
            )
        self.work_mode = mode

    @property
    def _is_establish_only(self) -> bool:
        return self.work_mode == WORK_MODE_ESTABLISH

    @property
    def _is_water_only(self) -> bool:
        return self.work_mode == WORK_MODE_WATER

    @staticmethod
    def _water_level(ram: np.ndarray) -> int:
        """Read watering can fill level (0 = empty, 20 = full).

        Prefer ``read_ram_value(..., "watering_can")`` so live emu RAM uses the
        WRAM mirror offset. Fall back to fixed ADDR_WATER_LEVEL for tiny test
        buffers that may not resolve through the catalog path.
        """
        try:
            return int(read_ram_value(ram, "watering_can"))
        except Exception:
            pass
        if ADDR_WATER_LEVEL < len(ram):
            return int(ram[ADDR_WATER_LEVEL])
        return 0

    def reset(self, world: WorldState) -> None:
        if os.getenv("CROP_DEBUG", "").lower() in ("1", "true", "yes"):
            self.debug = True
        self._state = CropState.DETECT
        self._plots = []
        self._plot_index = 0
        self._pass_number = 1
        self._plot_phase = PlotPhase.PLANT
        self._water_steps = []
        self._water_index = 0
        self._target_tile = None
        self._approach_tile = None
        self._face_direction = None
        self._action_queue.clear()
        self._steps_on_target = 0
        self._total_steps = 0
        self._failures = 0
        self._failed_tiles.clear()
        self._plot_watered = 0
        self._plot_skipped = 0
        self._allow_unknown_water_tiles = False
        self._allow_crop_walkable = False
        self._refill_pond_tile = None
        self._refill_pond_face = None
        self._refill_level_before = 0
        self._refill_search_level = -1
        self._bad_refill_tiles = set()
        self._refill_exhausted = False
        self._refill_nav_failures = 0
        self._refill_multihop = False
        self._refill_best_dist = 999
        self._pending_multihop_after_drop = False
        self._pending_gap_reseat = False
        self._corridor.reset()
        self._water_north_returns = 0
        self._water_crop_walk_recoveries = 0
        self._water_step_retries = 0
        self._gap_backed = False
        self._fence_subtask = None
        self._fence_open_attempts = 0
        self._pond_staged = False
        self._pending_fence_open = False
        self._pre_water_level = -1
        self._last_water_level_before = -1
        self._last_water_tile_before = -1
        self._water_verify_retries = 0
        self._resume_water_index = 0
        self.planted_count = 0
        self.watered_count = 0
        self.skipped_water = 0
        self.refill_count = 0
        self._dry_crop_tiles_at_start = 0
        self._had_seed_stock_at_start = False
        self._acceptance_snapped = False
        self._rejected_plan_centers = set()
        self._clear_crop_walkable()
        self._navigator.update(world.ram)
        self._tool_mgr.update(world.ram)

    def resume_after_hotswap(self, world: WorldState) -> None:
        """Re-scan live crop/refill state after manual control changes it."""
        self._navigator.update(world.ram)
        self._tool_mgr.update(world.ram)
        self._action_queue.clear()
        self._navigator.path = []
        self._navigator.stasis = 0
        self._pathfinder.temp_blocked.clear()
        self._clear_crop_walkable()
        self._state = CropState.DETECT
        self._plots = []
        self._plot_index = 0
        self._pass_number = 1
        self._plot_phase = PlotPhase.PLANT
        self._water_steps = []
        self._water_index = 0
        self._target_tile = None
        self._approach_tile = None
        self._face_direction = None
        self._total_steps = 0
        self._steps_on_target = 0
        self._failures = 0
        self._failed_tiles.clear()
        self._plot_watered = 0
        self._plot_skipped = 0
        self._allow_unknown_water_tiles = False
        self._allow_crop_walkable = False
        self._refill_pond_tile = None
        self._refill_pond_face = None
        self._refill_level_before = self._water_level(world.ram)
        self._refill_search_level = -1
        self._bad_refill_tiles = set()
        self._refill_exhausted = False
        self._refill_nav_failures = 0
        self._refill_multihop = False
        self._refill_best_dist = 999
        self._pending_multihop_after_drop = False
        self._pending_gap_reseat = False
        self._corridor.reset()
        self._water_north_returns = 0
        self._water_crop_walk_recoveries = 0
        self._gap_backed = False
        self._fence_subtask = None
        self._fence_open_attempts = 0
        self._pond_staged = False
        self._pending_fence_open = False
        self._pre_water_level = -1
        self._last_water_level_before = -1
        self._last_water_tile_before = -1
        self._water_verify_retries = 0
        self._resume_water_index = 0
        self.planted_count = 0
        self.watered_count = 0
        self.skipped_water = 0
        self.refill_count = 0
        self._dry_crop_tiles_at_start = 0
        self._had_seed_stock_at_start = False
        self._acceptance_snapped = False
        self._rejected_plan_centers = set()
        print(f"[CROP] Hot-swap resume: re-scan crops/refill state can={self._water_level(world.ram)}")

    def can_start(self, world: WorldState) -> bool:
        return True

    @property
    def phase_text(self) -> str:
        return f"{self._plot_phase}:{self._state}"

    @property
    def progress_text(self) -> str:
        s = f"plot={self._plot_index + 1}/{len(self._plots)} planted={self.planted_count} watered={self.watered_count}"
        if self.skipped_water:
            s += f" skip={self.skipped_water}"
        if self.refill_count:
            s += f" refills={self.refill_count}"
        if self._failures:
            s += f" fail={self._failures}"
        return s

    def _count_dry_crop_tiles(self, ram: np.ndarray) -> int:
        """Count dry crop / waterable tiles in task bounds (for acceptance)."""
        x0, y0, x1, y1 = self.bounds
        n = 0
        for ty in range(y0, y1 + 1):
            for tx in range(x0, x1 + 1):
                tid = get_tile_at(ram, tx, ty)
                if tile_needs_watering(tid):
                    n += 1
        return n

    def _snapshot_start_acceptance(self, ram: np.ndarray) -> None:
        """Capture dry-tile / seed-stock facts once at first detect."""
        if self._acceptance_snapped:
            return
        self._dry_crop_tiles_at_start = self._count_dry_crop_tiles(ram)
        self._had_seed_stock_at_start = self._has_plantable_seed_stock(ram)
        self._acceptance_snapped = True

    def _terminal_result(self, *, rain: bool = False) -> TaskResult:
        """Map plant/water counters to SUCCESS / no_work SUCCESS / FAILURE."""
        status, reason = crop_completion_status(
            work_mode=self.work_mode,
            planted=self.planted_count,
            watered=self.watered_count,
            dry_at_start=self._dry_crop_tiles_at_start,
            refill_exhausted=self._refill_exhausted,
            had_seed_stock=self._had_seed_stock_at_start,
            rain=rain,
        )
        extra = ""
        if self.skipped_water:
            extra += f" skipped={self.skipped_water}"
        if self.refill_count:
            extra += f" refills={self.refill_count}"
        if self._pass_number:
            extra += f" passes={self._pass_number}"
        full_reason = reason + extra
        print(f"[CROP] Complete ({status}): {full_reason}")
        if status == "failure":
            return TaskResult(status=TaskStatus.FAILURE, reason=full_reason)
        return TaskResult(status=TaskStatus.SUCCESS, reason=full_reason)

    def _has_plantable_seed_stock(self, ram: np.ndarray) -> bool:
        """True when seeds are in hand or counted in inventory for this crop."""
        if seed_item_in_carry_pair(ram, self.seed_type):
            return True
        try:
            from harvest.planner.day_plan_status import ram_seed_count

            return int(ram_seed_count(ram, self.seed_type)) > 0
        except Exception:
            return False

    def _plan_bounds_around(
        self,
        anchor: Tuple[int, int],
        radius: int = 12,
    ) -> Tuple[int, int, int, int]:
        """Clamp planning to a neighborhood around ``anchor`` inside task bounds."""
        x_min, y_min, x_max, y_max = self.bounds
        ax, ay = anchor
        return (
            max(x_min, ax - radius),
            max(y_min, ay - radius),
            min(x_max, ax + radius),
            min(y_max, ay + radius),
        )

    def _plan_bounds_near_player(self, start: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Clamp planning to a viewport-reachable neighborhood around the player."""
        return self._plan_bounds_around(start, radius=12)

    def step(self, world: WorldState) -> TaskResult:
        self._navigator.update(world.ram)
        self._tool_mgr.update(world.ram)
        self._total_steps += 1
        self._steps_on_target += 1

        if (
            self._total_steps == 1
            and is_rainy_weather(world.ram)
            and not self._is_water_only
            and not seed_item_in_carry_pair(world.ram, self.seed_type)
        ):
            wanted = SEED_ITEM.get(self.seed_type, SEED_ITEM["potato"])
            # Rain waters existing crops; without seeds there is no plant work either.
            # Still run detect in case established plots need nothing — but if no
            # seeds and rain, short-circuit so day plan can finish.
            # Water-only mode still scans (rain already watered; detect will no-op).
            print(f"[CROP] Rain and seed tool 0x{wanted:02X} not in carry pair; no crop work needed")
            self._snapshot_start_acceptance(world.ram)
            return self._terminal_result(rain=True)

        # Do not fail early when the watering can is out of the 2-slot carry pair.
        # Day plan often leaves seeds in-hand after ENSURE_CROP_SEEDS; we still
        # need to hoe/plant, then cycle to the can for watering.

        if self.debug and self._total_steps % self.debug_interval == 0:
            cur = self._navigator.current_tile
            print(f"[CROP] step={self._total_steps} phase={self._plot_phase} state={self._state} "
                  f"pos={cur} plot={self._plot_index}/{len(self._plots)} "
                  f"planted={self.planted_count} watered={self.watered_count} can={self._water_level(world.ram)}")

        # Timeout per target. Multi-hop refill gets a longer budget (corridor
        # from west pocket is 15–25 tiles + fence open overhead). Fence-open /
        # stage_pond own their own subtask budgets — do not abort them via
        # crop per-target timeout (that was resetting to detect mid-clear).
        if self._plot_phase in POND_ACCESS_PHASES:
            # Soft-cap fence thrash. Only early-bail when gap is open AND hands
            # are empty — otherwise we interrupt mid-carry before local_drop
            # (ROM: gap opens on lift, then 900f timeout left the bot stuck
            # carrying on the gap tile).
            carrying = self._player_carrying(world.ram)
            gap_open = self._pond_corridor_gap_open(world.ram)
            fence_budget = (
                900
                if gap_open and not carrying
                else max(self.max_steps_per_target * 3, 4000)
            )
            if self._steps_on_target > fence_budget:
                print(
                    f"[CROP] Fence/stage soft-timeout phase={self._plot_phase} "
                    f"budget={fence_budget}; forcing multi-hop or refill search"
                )
                self._fence_subtask = None
                self._steps_on_target = 0
                # Drop carried post first — multi-hop while carrying soft-locks
                # south-through-gap at the cleared fence tile.
                if self._ensure_hands_empty_for_refill(world.ram):
                    self._pending_multihop_after_drop = True
                    self._plot_phase = PlotPhase.REFILL
                    self._state = CropState.NAVIGATE
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        action=ActionResult(self._action_queue.popleft()),
                    )
                if self._pond_corridor_gap_open(world.ram) or self._fence_open_attempts > 0:
                    if self._commit_multihop_main_pond(
                        world.ram, self._water_level(world.ram)
                    ):
                        return TaskResult(
                            status=TaskStatus.RUNNING,
                            action=ActionResult(make_action()),
                        )
                self._plot_phase = PlotPhase.WATER
                self._start_refill(world.ram)
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action()),
                )
            # Fall through to normal step handling without target timeout.
            pass
        refill_budget = (
            max(self.max_steps_per_target * 3, 3600)
            if self._plot_phase == PlotPhase.REFILL
            else self.max_steps_per_target
        )
        if (
            self._plot_phase not in POND_ACCESS_PHASES
            and self._steps_on_target > refill_budget
            and self._target_tile is not None
        ):
            self._failed_tiles.add(self._target_tile)
            self._failures += 1
            self._action_queue.clear()
            if self._plot_phase == PlotPhase.WATER:
                if self._reprioritize_water_step(world.ram, reason="timeout"):
                    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
                if self._try_residual_crop_walk_recovery(world.ram):
                    return TaskResult(
                        status=TaskStatus.RUNNING, action=ActionResult(make_action())
                    )
                self.skipped_water += 1
                self._plot_skipped += 1
                print(f"[CROP] SKIP water tile {self._water_index + 1}/{len(self._water_steps)} (timeout) target={self._target_tile}")
                self._advance_water_step(world.ram)
            elif self._plot_phase == PlotPhase.HOE:
                print(
                    f"[CROP] SKIP hoe tile {self._water_index + 1}/{len(self._water_steps)} "
                    f"(timeout) target={self._target_tile}"
                )
                self._advance_hoe_step(world.ram)
            elif self._plot_phase == PlotPhase.REFILL:
                player = self._navigator.current_tile
                print(
                    f"[CROP] Refill timed out at {player} "
                    f"stand={self._refill_pond_tile} best_dist="
                    f"{getattr(self, '_refill_best_dist', '?')}"
                )
                # Densify thrash: scripted charge before more multihop.
                pond = self._refill_pond_tile
                pond_ok = pond is None or (pond[0] >= 30 and pond[1] >= 30)
                if (
                    player[1] <= 31
                    and 18 <= player[0] <= 32
                    and pond_ok
                    and getattr(self._corridor, "east_south_charges", 0) < 6
                ):
                    self._queue_east_south_corridor_charge(player)
                    self._steps_on_target = 0
                    self._corridor.refill_densify_stalls = 0
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        action=ActionResult(make_action()),
                    )
                if (
                    player[1] >= 32
                    and player[0] <= 31
                    and pond_ok
                    and getattr(self._corridor, "south_lip_charges", 0) < 12
                ):
                    self._queue_west_south_lip_charge(player)
                    self._steps_on_target = 0
                    self._corridor.refill_densify_stalls = 0
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        action=ActionResult(make_action()),
                    )
                # Soft: try multi-hop re-commit once more before blacklisting.
                if (
                    getattr(self, "_refill_multihop", False)
                    and getattr(self, "_refill_nav_failures", 0) < 6
                    and (
                        self._pond_corridor_gap_open(world.ram)
                        or self._fence_open_attempts > 0
                    )
                ):
                    self._refill_nav_failures = getattr(self, "_refill_nav_failures", 0) + 1
                    self._steps_on_target = 0
                    if self._commit_multihop_main_pond(
                        world.ram, self._water_level(world.ram)
                    ):
                        return TaskResult(
                            status=TaskStatus.RUNNING,
                            action=ActionResult(make_action()),
                        )
                if self._refill_pond_tile and not is_main_pond_stand(
                    self._refill_pond_tile
                ):
                    self._bad_refill_tiles.add(self._refill_pond_tile)
                # Navigate back to current water step
                self._plot_phase = PlotPhase.WATER
                self._refill_multihop = False
                self._set_water_walkable()
                if self._water_index < len(self._water_steps):
                    target, stand, face = self._water_steps[self._water_index]
                    self._target_tile = target
                    self._approach_tile = stand
                    self._face_direction = face
                else:
                    center = self._plots[self._plot_index]
                    self._target_tile = center
                    self._approach_tile = center
                self._state = CropState.NAVIGATE
                self._navigator.path = []
            elif self._plot_phase == PlotPhase.PLANT:
                center = self._plots[self._plot_index] if self._plot_index < len(self._plots) else None
                if center is not None:
                    self._rejected_plan_centers.add(center)
                print(f"[CROP] Plant timeout at {center}; skipping plot")
                self._advance_plot(world.ram)
            else:
                self._target_tile = None
                self._state = CropState.DETECT
            if self._failures >= self.max_failures:
                return TaskResult(status=TaskStatus.FAILURE, reason="too many target timeouts")

        # Drain action queue
        if self._action_queue:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(self._action_queue.popleft()))

        # Dialog dismissal
        input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
        if input_lock != 1:
            action = make_action(a=True) if self._total_steps % 2 == 0 else make_action()
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action), reason="dialog")

        # Check if all plots done
        if self._state == CropState.DONE:
            return self._terminal_result()

        # Outer CropState dispatch. FENCE_OPEN takes WorldState (subtask);
        # other arms take RAM only.
        if self._state == CropState.FENCE_OPEN:
            result = self._handle_fence_open(world)
            if result is not None:
                return result

        handlers = {
            CropState.DETECT: self._handle_detect,
            CropState.NAVIGATE: self._handle_navigate,
            CropState.CENTER: self._handle_center,
            CropState.ACT: self._handle_act,
            CropState.VERIFY: self._handle_verify,
            CropState.TOOL_SWITCH: self._handle_tool_switch,
        }

        handler = handlers.get(self._state)
        if handler:
            result = handler(world.ram)
            if result is not None:
                return result

        if self._action_queue:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(self._action_queue.popleft()))

        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

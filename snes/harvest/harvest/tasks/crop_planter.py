"""
Crop planting task — thin-ish composer for detect/plant/water/refill.

Extracted arms (rr-ds3):
  - ``crop_geometry`` — pure plot/water geometry
  - ``crop_fsm`` — CropState / PlotPhase / work_mode constants
  - ``crop_establish`` — hoe + plant phase mixin
  - ``crop_water_ops`` — water-step + residual recovery mixin
  - ``crop_refill`` — can refill / pond access / corridor thrash mixin
  - ``crop_navigate`` — multi-phase navigate / stuck recovery
  - ``pond_*`` — corridor charges, hop densify, policy
"""

from __future__ import annotations

import os
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from harvest.core.ram_catalog import field_spec, read_ram_u8, read_ram_u16, read_ram_value
from harvest.core.carry import (
    ADDR_TOOL_BACKPACK,
    SEED_ITEM,
    carry_pair_items,
    seed_in_carry_pair as seed_item_in_carry_pair,
    watering_can_in_carry_pair,
)
from harvest.core.tile_catalog import (
    Tool,
    ADDR_MAP,
    ADDR_TOOL,
    ADDR_INPUT_LOCK,
)
from harvest.tasks.nav import (
    Point,
    Pathfinder,
    Navigator,
    make_action,
    get_tile_at,
    tile_dist,
    TILE_SIZE,
    MAP_WIDTH,
    VIEWPORT_HOP_TILES,
    WALKABLE_TILES,
)
from harvest.tasks.farm_clearer import (
    TileScanner,
    ToolManager,
    use_tool,
    use_tool_facing,
    cycle_tool,
)
from harvest.tasks.water_refill import (
    REFILL_NONFILL_WATER_TILES,
    REFILL_PREFERRED_WATER_TILES,
    corridor_needs_fence_open,
    crop_completion_status,
    order_preferred_edges,
    select_main_pond_refill,
    select_staging_stand,
)
from harvest.tasks.pond_corridor import (
    KIND_ACT_AT_STAND,
    KIND_ARM_F0_AND_LIP,
    KIND_COMMIT_MULTIHOP_MAYBE_ACT_OR_REFILL,
    KIND_COMMIT_MULTIHOP_OR_REFILL,
    KIND_QUEUE_EAST_SOUTH,
    KIND_QUEUE_GAP_SOUTH,
    KIND_QUEUE_WEST_SOUTH_LIP,
    KIND_TRY_MULTIHOP_CONTINUE,
    KIND_TRY_MULTIHOP_MAYBE_ACT_CONTINUE,
    PRIMARY_POND_FACE,
    PRIMARY_POND_STAND,
    CorridorNavDecision,
    PondCorridorController,
    build_east_south_corridor_charge,
    build_gap_south_fallback,
    build_west_south_lip_charge,
    compute_refill_hop_goal,
    decide_after_east_south_charge,
    decide_after_gap_reseat,
    decide_after_multihop_drop,
    decide_after_south_lip_charge,
    pond_corridor_gap_open as pond_corridor_gap_is_open,
)
from harvest.tasks.crop_geometry import (
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

from harvest.tasks.crop_fsm import (
    CropState,
    PlotPhase,
    ON_APPROACH_PHASES,
    POND_ACCESS_PHASES,
    WORK_MODE_FULL,
    WORK_MODE_ESTABLISH,
    WORK_MODE_WATER,
    VALID_WORK_MODES,
)
from harvest.tasks.crop_establish import CropEstablishMixin
from harvest.tasks.crop_water_ops import CropWaterOpsMixin
from harvest.tasks.crop_refill import CropRefillMixin
from harvest.tasks.crop_navigate import CropNavigateMixin

@dataclass
class CropWaterTask(CropEstablishMixin, CropWaterOpsMixin, CropRefillMixin, CropNavigateMixin, Task):
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

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

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
        """Clamp planning to a viewport-reachable neighborhood around the player.

        Full-farm plans often pick distant centers that BFS cannot reach through
        stale off-screen tiles. Keep new plots within ~12 tiles of the player
        and inside the task bounds.
        """
        return self._plan_bounds_around(start, radius=12)

    def _handle_detect(self, ram: np.ndarray) -> Optional[TaskResult]:
        """Scan for crop plots."""
        self._snapshot_start_acceptance(ram)
        resume_plots = detect_crop_resume_plots(ram, self.bounds)
        if resume_plots:
            supplemental = detect_plots(ram, self.bounds)
            self._plots = _merge_plot_centers(resume_plots, supplemental)
        else:
            self._plots = detect_plots(ram, self.bounds)
        if not self._plots and self._is_water_only and self._dry_crop_tiles_at_start > 0:
            # Partial plant (west-pocket tilled=2 / 1–3 crop tiles) misses the
            # default resume min_count=4. Water keep-alive must still find those
            # dry singles/pairs or every day ends water fail: dry_crops=N watered=0
            # without attempting can/refill (rr-5in continuous residual).
            sparse = detect_crop_resume_plots(ram, self.bounds, min_count=1)
            if sparse:
                print(
                    f"[CROP] Sparse water plots (min_count=1): {sparse} "
                    f"dry={self._dry_crop_tiles_at_start}"
                )
                self._plots = sparse
            else:
                singles = self._singleton_dry_crop_centers(ram)
                if singles:
                    print(
                        f"[CROP] Singleton dry water targets: {singles} "
                        f"dry={self._dry_crop_tiles_at_start}"
                    )
                    self._plots = singles
        if not self._plots:
            # Virgin soil: plan + hoe + plant instead of silently succeeding.
            # Water-only pass never opens new plots (no seeds/hoe in carry).
            can_plant = (
                not self._is_water_only
                and self._has_plantable_seed_stock(ram)
            )
            if self._pass_number == 1 and can_plant:
                planned = self._plan_new_plot_centers(ram)
                if planned:
                    self._plots = planned
                else:
                    print("[CROP] No plots detected and no plantable plan")
                    return self._terminal_result()
            elif self._pass_number == 1:
                return self._terminal_result()
            else:
                self._state = CropState.DONE
                return None
        current_tile = self._navigator.current_tile
        self._plots.sort(key=lambda center: (tile_dist(current_tile, center), center[1], center[0]))
        self._plot_index = 0
        pass_label = f"(pass {self._pass_number})" if self._pass_number > 1 else ""
        print(
            f"[CROP] Detected {len(self._plots)} plots: {self._plots} "
            f"mode={self.work_mode} {pass_label}"
        )
        self._start_plot(ram)
        return None

    def _start_plot(self, ram: np.ndarray):
        """Begin processing the current plot."""
        if self._plot_index >= len(self._plots):
            return
        center = self._plots[self._plot_index]
        self._set_crop_walkable()  # allow pathfinding through crop tiles
        tilled = count_tilled(ram, center)
        crop_tiles = _count_crop_tiles(ram, center[0], center[1])

        # Water-only: never hoe/plant; water established crops or skip the plot.
        if self._is_water_only:
            if crop_tiles > 0:
                self._begin_water_phase(ram, allow_unknown_tiles=False)
            else:
                print(
                    f"[CROP] Plot {self._plot_index + 1}/{len(self._plots)} "
                    f"center=({center[0]},{center[1]}) water-only with no crops; skip"
                )
                self._advance_plot(ram)
            return

        # Seed bag plants a 3x3 from the center notch. Plant once enough ring
        # tiles are hoed (full 8 is ideal; partial still uses the bag).
        if crop_tiles == 0 and tilled >= 4:
            self._plot_phase = PlotPhase.PLANT
            self._target_tile = center
            self._approach_tile = center  # stand ON center to plant
            self._face_direction = "down"
            self._set_crop_walkable()
            self._state = CropState.NAVIGATE
            self._navigator.path = []  # force re-path
            self._steps_on_target = 0
            print(f"[CROP] Plot {self._plot_index + 1}/{len(self._plots)} center=({center[0]},{center[1]}) phase=PLANT tilled={tilled}")
        elif crop_tiles == 0 and tilled < 4:
            # Untilled soil: hoe the 8 ring tiles, then plant from center.
            self._begin_hoe_phase(ram)
        else:
            if crop_tiles > 0 and tilled > 0:
                print(
                    f"[CROP] Plot {self._plot_index + 1}/{len(self._plots)} "
                    f"has {crop_tiles} crop tiles and {tilled} open tilled tiles; skip seeding partial plot"
                )
            if self._is_establish_only:
                # Plant pass leaves watering for the later can pass.
                print(
                    f"[CROP] Plot {self._plot_index + 1}/{len(self._plots)} "
                    f"already established; establish-only skips water"
                )
                self._advance_plot(ram)
            else:
                self._begin_water_phase(ram, allow_unknown_tiles=False)

    def _advance_plot(self, ram: np.ndarray):
        """Move to the next plot, or trigger a re-scan pass, or finish."""
        self._clear_crop_walkable()
        self._plot_index += 1
        if self._plot_index >= len(self._plots):
            # Water: re-scan when tiles were skipped.
            # Establish: one retry pass so a rejected center can fall back to
            # another nearby tillable plot (rejected centers are remembered).
            can_retry_establish = (
                self._is_establish_only
                and self._pass_number < 2
                and self.planted_count == 0
                and bool(self._rejected_plan_centers)
            )
            can_retry_water = (
                not self._is_establish_only
                and self._pass_number < 3
                and self.skipped_water > 0
            )
            if can_retry_establish or can_retry_water:
                prev_skip = self.skipped_water
                self._pass_number += 1
                self._state = CropState.DETECT
                self._pathfinder.temp_blocked.clear()
                self._refill_exhausted = False
                if can_retry_establish:
                    print(
                        f"[CROP] Establish pass {self._pass_number - 1} planted=0; "
                        f"retry with rejected={sorted(self._rejected_plan_centers)}"
                    )
                else:
                    print(
                        f"[CROP] Pass {self._pass_number - 1} complete ({prev_skip} skipped), "
                        f"starting pass {self._pass_number}..."
                    )
            else:
                self._state = CropState.DONE
        else:
            self._start_plot(ram)

    def _handle_center(self, ram: np.ndarray) -> Optional[TaskResult]:
        if self._approach_tile is None:
            self._state = CropState.DETECT
            return None
        tol = 1 if self._plot_phase in ON_APPROACH_PHASES else 2
        center_action = self._navigator.center_on_tile(self._approach_tile, tolerance=tol)
        if center_action is None:
            if self._plot_phase == PlotPhase.STAGE_POND:
                # Staged — hand off to fence clear (skip re-stage).
                fences = pond_access_blocking_fences(ram)
                print(
                    f"[CROP] Pond stage reached at {self._navigator.current_tile}; "
                    f"starting fence clear (wall n={len(fences)})"
                )
                if fences and self._try_open_pond_access(ram, fences, skip_stage=True):
                    return None
                self._plot_phase = PlotPhase.WATER
                self._start_refill(ram)
                return None
            self._state = CropState.ACT
        else:
            self._action_queue.append(center_action)
        return None

    def _handle_act(self, ram: np.ndarray) -> Optional[TaskResult]:
        if self._action_queue:
            return None

        # Wait for input lock to clear and player to settle
        input_lock = int(ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(ram) else 0
        if input_lock != 1 or self._navigator.stasis < 6:
            return None

        # Position check: must be on the correct tile
        player = self._navigator.current_tile
        if self._plot_phase in ON_APPROACH_PHASES:
            # Must be ON approach tile for plant/water/hoe
            if player != self._approach_tile:
                print(f"[CROP] {self._plot_phase.upper()} pos mismatch: at ({player[0]},{player[1]}) need ({self._approach_tile[0]},{self._approach_tile[1]}), re-navigate")
                self._state = CropState.NAVIGATE
                self._navigator.path = []
                return None
        else:
            # Refill: on or adjacent to approach tile
            if tile_dist(player, self._approach_tile) > 1:
                self._state = CropState.NAVIGATE
                self._navigator.path = []
                return None

        # Re-center drift correction
        tol = 1 if self._plot_phase in ON_APPROACH_PHASES else 2
        center_action = self._navigator.center_on_tile(self._approach_tile, tolerance=tol)
        if center_action is not None:
            self._action_queue.append(center_action)
            return None

        if self._plot_phase == PlotPhase.STAGE_POND:
            fences = pond_access_blocking_fences(ram)
            if fences and self._try_open_pond_access(ram, fences, skip_stage=True):
                return None
            self._plot_phase = PlotPhase.WATER
            self._start_refill(ram)
            return None

        act_handlers = {
            PlotPhase.PLANT: self._act_plant,
            PlotPhase.HOE: self._act_hoe,
            PlotPhase.WATER: self._act_water,
            PlotPhase.REFILL: self._act_refill,
        }
        handler = act_handlers.get(self._plot_phase)
        if handler is not None:
            return handler(ram)
        return None

    def _handle_verify(self, ram: np.ndarray) -> Optional[TaskResult]:
        if self._action_queue:
            return None

        input_lock = int(ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(ram) else 0
        if input_lock != 1:
            return None

        if self._plot_phase == PlotPhase.HOE:
            self._advance_hoe_step(ram)
            return None

        if self._plot_phase == PlotPhase.PLANT:
            center = self._plots[self._plot_index]
            tilled_remaining = count_tilled(ram, center)
            # Don't retry - plant action fires once per plot.  If position
            # and tool were correct (checked in _handle_act), seeds were used.
            # Tile data may lag behind the animation; retrying wastes seeds.
            self.planted_count += 1
            if tilled_remaining == 0:
                print(f"[CROP] PLANT OK plot {self._plot_index + 1} planted={self.planted_count}")
            else:
                print(f"[CROP] PLANT OK plot {self._plot_index + 1} planted={self.planted_count} ({tilled_remaining} tiles still updating)")
            if self._is_establish_only:
                # Day-plan plant pass: seeds+hoe only; water after can re-fetch.
                self._advance_plot(ram)
            else:
                self._begin_water_phase(ram, allow_unknown_tiles=True)

        elif self._plot_phase == PlotPhase.WATER:
            if self._water_index >= len(self._water_steps):
                self._advance_water_step(ram)
                return None

            target = self._water_steps[self._water_index][0]
            lvl_after = self._water_level(ram)
            tid_after = get_tile_at(ram, target[0], target[1])
            used_water = self._last_water_level_before >= 0 and lvl_after < self._last_water_level_before
            tile_watered = tile_is_watered(tid_after)

            if used_water or tile_watered:
                self._plot_watered += 1
                self._water_verify_retries = 0
                self._advance_water_step(ram)
            else:
                self._water_verify_retries += 1
                if self._water_verify_retries >= 2:
                    self.skipped_water += 1
                    self._plot_skipped += 1
                    print(
                        f"[CROP] SKIP water tile {self._water_index + 1}/{len(self._water_steps)} "
                        f"target={target} tid=0x{tid_after:02X} can={lvl_after} (verify failed)"
                    )
                    self._water_verify_retries = 0
                    self._advance_water_step(ram)
                else:
                    print(
                        f"[CROP] RETRY water tile {self._water_index + 1}/{len(self._water_steps)} "
                        f"target={target} tid=0x{tid_after:02X} can={lvl_after}"
                    )
                    self._state = CropState.CENTER

        elif self._plot_phase == PlotPhase.REFILL:
            lvl_after = self._water_level(ram)
            if lvl_after > self._refill_level_before:
                # Refill succeeded — navigate back to current water step
                self.refill_count += 1
                print(f"[CROP] REFILL OK can={lvl_after} (was {self._refill_level_before}) refills={self.refill_count}")
                self._pre_water_level = lvl_after  # reset for plot-level verification
                self._refill_search_level = -1  # reset search tracking
                self._plot_phase = PlotPhase.WATER
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
                self._steps_on_target = 0
            else:
                # Refill failed — mark tile and neighbors as bad, try another
                bad = self._refill_pond_tile
                self._bad_refill_tiles.add(bad)
                # If water was CONSUMED (level decreased), this area is actively
                # harmful — mark a 2-tile radius as bad to skip nearby tiles
                if lvl_after < self._refill_level_before:
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            self._bad_refill_tiles.add((bad[0] + dx, bad[1] + dy))
                    print(f"[CROP] REFILL FAILED at ({bad[0]},{bad[1]}) can={lvl_after} (was {self._refill_level_before}), "
                          f"water consumed! blacklisted neighborhood, trying next")
                else:
                    print(f"[CROP] REFILL FAILED at ({bad[0]},{bad[1]}) can={lvl_after} (was {self._refill_level_before}), trying next")
                self._plot_phase = PlotPhase.WATER
                self._start_refill(ram)  # try another water edge

        if self._state == CropState.DONE:
            return self._terminal_result()

        return None

    def _handle_tool_switch(self, ram: np.ndarray) -> Optional[TaskResult]:
        """Cycle tools to find the needed one."""
        if self._plot_phase == PlotPhase.PLANT:
            wanted = SEED_ITEM.get(self.seed_type, SEED_ITEM["potato"])
        elif self._plot_phase == PlotPhase.HOE:
            wanted = int(Tool.HOE)
        else:
            wanted = int(Tool.WATERING_CAN)

        self._tool_mgr.update(ram)
        current = self._tool_mgr.current

        if current == wanted:
            if self.debug:
                print(f"[CROP] Found tool 0x{wanted:02X}")
            self._state = CropState.CENTER
            return None

        self._tool_mgr.record()

        if self._tool_mgr.cycle_complete():
            if self._plot_phase == PlotPhase.PLANT:
                center = self._plots[self._plot_index] if self._plot_index < len(self._plots) else None
                print(f"[CROP] Seed tool 0x{wanted:02X} not found, skipping plant plot at {center}")
                self._advance_plot(ram)
                return None
            if self._plot_phase == PlotPhase.HOE:
                center = self._plots[self._plot_index] if self._plot_index < len(self._plots) else None
                print(f"[CROP] Hoe 0x{wanted:02X} not found, skipping establish plot at {center}")
                self._advance_plot(ram)
                return None
            # Watering can missing after plant is common (only 2 carry slots).
            # Report partial success so the day plan can re-fetch the can and
            # run a second CROP_WATER pass instead of aborting the whole day.
            if self.planted_count > 0 or self.watered_count > 0:
                msg = (
                    f"planted={self.planted_count} watered={self.watered_count} "
                    f"refills={self.refill_count}; tool 0x{wanted:02X} not in carry pair"
                )
                print(f"[CROP] Partial complete: {msg}")
                return TaskResult(status=TaskStatus.SUCCESS, reason=msg)
            # Water-only pass without the can: keep the explicit reason so the
            # day plan can recover via ENSURE_WATERING_CAN / recovery task.
            if self._is_water_only and wanted == int(Tool.WATERING_CAN):
                print("[CROP] Watering can not in carry pair (water-only pass)")
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason="watering can not in carry pair",
                )
            print(f"[CROP] Tool 0x{wanted:02X} not found in inventory")
            return TaskResult(status=TaskStatus.FAILURE, reason=f"tool 0x{wanted:02X} not in inventory")

        self._action_queue.extend(cycle_tool())
        return None

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

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

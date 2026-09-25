"""
Crop planting task — one ``CropWaterTask``.

``step`` runs pocket / crop skills for establish, water, and full.
Plot geometry stays in ``crop_geometry``. Refill policy that is not a skill
yet is plain functions in ``crop_refill`` / ``crop_refill_verify`` /
``crop_navigate`` / ``crop_water_ops``, called by this task (including the
pre-armed navigate and corridor phases).
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
from harvest.tasks.nav import TILE_SIZE, Pathfinder, Navigator, get_tile_at, make_action, tile_dist
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
class CropWaterTask(Task):
    """Plant and water crops by running crop skills. One task, no mixins.

    ``work_mode``:
      - establish: ``farm_pocket_plant_skill`` (hoe + plant, no water)
      - water: detected plots via ``water_until_wet_skill``; empty can calls
        refill functions (``refill_bounds``, not the pocket fill skill)
      - full: establish skill, then the water skills

    Refill corridor / pre-armed navigate still uses the plain policy functions.
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
    # Skill sequence for establish / water / full. Refill phases do not use it.
    _child: Optional[Task] = field(default=None, init=False)
    _armed_water_tiles: int = field(default=0, init=False)

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
        self._water_steps_deferred = 0
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
        self._child = None
        self._armed_water_tiles = 0
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
        self._water_steps_deferred = 0
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
        self._child = None
        self._armed_water_tiles = 0
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

    def _use_legacy_dispatch(self) -> bool:
        """Refill corridor and pre-armed navigate/act still use policy functions.

        A fresh DETECT tick composes skills instead. That refill phase machine
        is not a second task; folding the corridor charges into a skill would
        be one.
        """
        if self._plot_phase in (
            PlotPhase.REFILL,
            PlotPhase.STAGE_POND,
            PlotPhase.OPEN_POND,
        ):
            return True
        if self._state == CropState.FENCE_OPEN:
            return True
        return self._child is None and self._state != CropState.DETECT

    def _detect_work_plots(self, ram: np.ndarray) -> List[Tuple[int, int]]:
        resume_plots = detect_crop_resume_plots(ram, self.bounds)
        if resume_plots:
            plots = _merge_plot_centers(resume_plots, detect_plots(ram, self.bounds))
        else:
            plots = detect_plots(ram, self.bounds)
        if not plots and self._is_water_only and self._dry_crop_tiles_at_start > 0:
            plots = detect_crop_resume_plots(ram, self.bounds, min_count=1)
        if not plots:
            return []
        current = self._navigator.current_tile
        return sorted(plots, key=lambda center: (tile_dist(current, center), center[1], center[0]))

    def _plant_skill(self, ram: np.ndarray) -> Optional[Task]:
        from harvest.tasks.skills import farm_pocket_plant_skill, sequence_skills

        centers = self._plan_new_plot_centers(ram)
        if not centers:
            return None
        self._plots = list(centers)
        self._plot_index = 0
        skills = [
            farm_pocket_plant_skill(
                seed_type=self.seed_type,
                center=center,
                ram=ram,
                include_water=False,
            )
            for center in centers
        ]
        if len(skills) == 1:
            return skills[0]
        return sequence_skills("crop_establish", *skills)

    def _water_skill(self, ram: np.ndarray) -> Optional[Task]:
        from harvest.tasks.crop_skills import water_until_wet_skill
        from harvest.tasks.skills import NavSkill, sequence_skills

        plots = self._detect_work_plots(ram)
        if not plots:
            return None
        if self._is_water_only or not self._plots:
            self._plots = list(plots)
        else:
            for plot in plots:
                if plot not in self._plots:
                    self._plots.append(plot)
        self._plot_index = 0
        start = self._navigator.current_tile
        timeout = max(int(self.max_steps_per_target), 1)
        skills: List[Task] = []
        for center in plots:
            steps = build_water_steps(
                ram,
                center,
                allow_crop_walkable=False,
                start_tile=start,
                skip_tiles=set(self.skip_water_tiles),
            )
            for index, (target, stand, face) in enumerate(steps):
                skills.append(
                    NavSkill(
                        name=f"nav_water_{center[0]}_{center[1]}_{index}",
                        target_px=(stand[0] * TILE_SIZE + 8, stand[1] * TILE_SIZE + 8),
                        radius=7,
                        soft_radius=7,
                        timeout=timeout,
                        require_tilemap=0x00,
                    )
                )
                skills.append(
                    water_until_wet_skill(
                        target_tile=target,
                        face=face,
                        timeout=timeout,
                    )
                )
        self._armed_water_tiles = sum(
            1 for skill in skills if str(getattr(skill, "name", "")).startswith("water")
        )
        if not skills:
            return None
        return sequence_skills("crop_water", *skills)

    def _compose_mode_skill(self, ram: np.ndarray) -> Task | TaskResult:
        self._snapshot_start_acceptance(ram)
        if self._is_water_only:
            skill = self._water_skill(ram)
            if skill is None:
                return self._terminal_result()
            self._plot_phase = PlotPhase.WATER
            return skill
        if self._is_establish_only:
            if not self._has_plantable_seed_stock(ram):
                return self._terminal_result()
            skill = self._plant_skill(ram)
            if skill is None:
                return self._terminal_result()
            self._plot_phase = PlotPhase.HOE
            return skill
        skills: List[Task] = []
        if self._has_plantable_seed_stock(ram):
            plant = self._plant_skill(ram)
            if plant is not None:
                skills.append(plant)
        water = self._water_skill(ram)
        if water is not None:
            skills.append(water)
        if not skills:
            return self._terminal_result()
        # Plant runs first when both are armed; water-only full mode stays on water.
        self._plot_phase = (
            PlotPhase.WATER if len(skills) == 1 and water is not None else PlotPhase.HOE
        )
        if len(skills) == 1:
            return skills[0]
        from harvest.tasks.skills import sequence_skills

        return sequence_skills("crop_full", *skills)

    def _sync_phase_from_child(self) -> None:
        current = getattr(self._child, "current_task", None)
        name = str(getattr(current, "name", "") or "")
        if name.startswith("water") or name.startswith("nav_water"):
            self._plot_phase = PlotPhase.WATER
        elif "hoe" in name:
            self._plot_phase = PlotPhase.HOE
        elif "plant" in name or name.startswith("select_carry"):
            self._plot_phase = PlotPhase.PLANT

    def _skill_needs_refill(self, ram: np.ndarray) -> bool:
        if self._is_establish_only or self._refill_exhausted:
            return False
        if self._water_level(ram) >= 1:
            return False
        current = getattr(self._child, "current_task", None)
        name = str(getattr(current, "name", "") or getattr(self._child, "name", "") or "")
        return name.startswith("water") or name.startswith("nav_water") or name == "crop_water"

    def _credit_skill_success(self) -> None:
        if (
            not self._is_water_only
            and self.planted_count == 0
            and self._plots
            and (self._is_establish_only or self._had_seed_stock_at_start)
        ):
            self.planted_count = len(self._plots)
        if (
            not self._is_establish_only
            and self._armed_water_tiles
            and self.watered_count == 0
        ):
            self.watered_count = self._armed_water_tiles

    def _step_mode_skill(self, world: WorldState) -> TaskResult:
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
            print(
                f"[CROP] Rain and seed tool 0x{wanted:02X} not in carry pair; "
                "no crop work needed"
            )
            self._snapshot_start_acceptance(world.ram)
            return self._terminal_result(rain=True)

        if self._child is None:
            built = self._compose_mode_skill(world.ram)
            if isinstance(built, TaskResult):
                return built
            self._child = built
            self._child.reset(world)
            self._state = CropState.ACT
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action()),
                reason=f"crop skill {self.work_mode}",
            )

        if self._skill_needs_refill(world.ram):
            self._start_refill(world.ram)
            if self._use_legacy_dispatch():
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action()),
                    reason="refill",
                )

        result = self._child.step(world)
        self._sync_phase_from_child()
        if result.status == TaskStatus.RUNNING:
            return result
        if result.status == TaskStatus.SUCCESS:
            self._credit_skill_success()
            self._state = CropState.DONE
            return self._terminal_result()
        reason = result.reason or ""
        if "watering can not in carry pair" in reason and self._is_water_only:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason="watering can not in carry pair",
            )
        self._failures += 1
        print(f"[CROP] Skill failed mode={self.work_mode}: {reason}")
        return self._terminal_result()

    def step(self, world: WorldState) -> TaskResult:
        if self._state == CropState.DONE and self._child is None:
            return self._terminal_result()
        if self._use_legacy_dispatch():
            return self._dispatch_legacy(world)
        return self._step_mode_skill(world)


def _bind_crop_policy(cls: type) -> None:
    """Install refill / navigate / establish functions as CropWaterTask methods."""
    import harvest.tasks.crop_establish as establish
    import harvest.tasks.crop_navigate as navigate
    import harvest.tasks.crop_refill as refill
    import harvest.tasks.crop_refill_verify as verify
    import harvest.tasks.crop_water_ops as water_ops

    for mod in (establish, water_ops, refill, verify, navigate):
        for name, obj in vars(mod).items():
            if not name.startswith("_") or not callable(obj):
                continue
            if getattr(obj, "__module__", None) != mod.__name__:
                continue
            setattr(cls, name, obj)


_bind_crop_policy(CropWaterTask)

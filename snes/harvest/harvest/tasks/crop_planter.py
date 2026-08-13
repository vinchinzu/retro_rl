"""
Crop planting task — thin composer for detect/plant/water/refill.

Extracted arms (rr-ds3):
  - ``crop_geometry`` — pure plot/water geometry + public constants
  - ``crop_fsm`` — CropState / PlotPhase / work_mode constants
  - ``crop_establish`` — hoe + plant phase mixin
  - ``crop_water_ops`` — water-step + residual recovery mixin
  - ``crop_refill`` — can refill / pond access / corridor thrash mixin
  - ``crop_navigate`` — multi-phase navigate / stuck recovery
  - ``crop_detect`` — detect / start_plot / advance_plot
  - ``crop_act_verify`` — center / act / verify / tool_switch
  - ``crop_step`` — main step loop + target timeouts
  - ``pond_*`` — corridor charges, hop densify, policy
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np

from retro_harness import Task, WorldState

from harvest.core.ram_catalog import read_ram_value
from harvest.core.carry import (
    ADDR_TOOL_BACKPACK,
    SEED_ITEM,
    carry_pair_items,
    seed_in_carry_pair as seed_item_in_carry_pair,
    watering_can_in_carry_pair,
)
from harvest.tasks.nav import Pathfinder, Navigator
from harvest.tasks.farm_clearer import TileScanner, ToolManager
from harvest.tasks.water_refill import REFILL_PREFERRED_WATER_TILES
from harvest.tasks.pond_corridor import PondCorridorController

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
from harvest.tasks.crop_fsm import (
    CropState,
    PlotPhase,
    WORK_MODE_FULL,
    WORK_MODE_ESTABLISH,
    WORK_MODE_WATER,
    VALID_WORK_MODES,
)
from harvest.tasks.crop_establish import CropEstablishMixin
from harvest.tasks.crop_water_ops import CropWaterOpsMixin
from harvest.tasks.crop_refill import CropRefillMixin
from harvest.tasks.crop_navigate import CropNavigateMixin
from harvest.tasks.crop_detect import CropDetectMixin
from harvest.tasks.crop_act_verify import CropActVerifyMixin
from harvest.tasks.crop_step import CropStepMixin

# Re-export carry helpers under the historical crop_planter names.
__all__ = [
    "CropWaterTask",
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
    CropDetectMixin,
    CropActVerifyMixin,
    CropStepMixin,
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

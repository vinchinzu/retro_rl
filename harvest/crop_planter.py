"""
Crop planting utilities - hoe patterns, watering paths, pond refill.

Composable building blocks for potato/turnip/corn/tomato farming.
Works with Navigator/Pathfinder for BFS navigation between steps.

3x3 Plot Layout (relative to center):
    (-1,-1) (0,-1) (1,-1)
    (-1, 0) [CEN ] (1, 0)
    (-1, 1) (0, 1) (1, 1)

Workflow:
    1. hoe_plan()  — hoe 8 tiles around center, leave center as walk-in point
    2. nav to center, plant seeds (covers 3x3 tilled area)
    3. water_plan() — water all crops from outside perimeter (can't walk on grown crops)
    4. pond_refill_plan() — find pond edge, refill watering can

Extracted from recorded gameplay (potato_plant.json, 4574 frames).
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Set, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import numpy as np

from farm_clearer import (
    Tool,
    Point,
    TileScanner,
    Pathfinder,
    Navigator,
    ToolManager,
    make_action,
    use_tool,
    use_tool_facing,
    cycle_tool,
    get_tile_at,
    tile_dist,
    TILE_SIZE,
    MAP_WIDTH,
    ADDR_MAP,
    ADDR_TOOL,
    ADDR_INPUT_LOCK,
    WALKABLE_TILES,
)

# ── seed item IDs (in tool/item slot 0x0921) ────────────────────────

SEED_ITEM: Dict[str, int] = {
    "corn": 0x05,       # yellow seed
    "tomato": 0x06,     # red seed
    "potato": 0x07,     # brown seed
    "turnip": 0x08,     # white seed
    "grass": 0x0C,
}

# ── tile IDs ─────────────────────────────────────────────────────────

FRESH_TILLED = 0x07
DRIED_TILLED = 0x02
UNTILLED = 0x01
TILLABLE_TILES = {UNTILLED, DRIED_TILLED}
PLANTABLE_TILES = {FRESH_TILLED}

# Pond/water tiles — stand adjacent to these, face them, use watering can
WATER_TILES = frozenset({
    0xA6,                               # pond edge
    0xF0, 0xF1, 0xF2,                  # water
    0xF7, 0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD,
})

# Actual water tiles for refilling — excludes 0xA6 (pond border/decorative)
REFILL_WATER_TILES = frozenset({
    0xF0, 0xF1, 0xF2,
    0xF7, 0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD,
})

# ── 3x3 hoe pattern ─────────────────────────────────────────────────
# Hoe 8 tiles around center.  Center stays untilled — that's where
# the player stands to plant seeds.
#
# Each entry: (target_dx, target_dy, stand_dx, stand_dy, face_dir)
# stand = where the player walks to; face = direction to face before Y.
#
# Order: clockwise from top-right, extracted from recording.
# (right column facing left → top row facing down → left column facing
#  right → bottom center facing up)

HOE_PLAN = [
    # right column: stand 1 tile right of target, face left
    ( 1, -1,   2, -1,  "left"),
    ( 1,  0,   2,  0,  "left"),
    ( 1,  1,   2,  1,  "left"),
    # top row remainder: stand 1 tile above target, face down
    ( 0, -1,   0, -2,  "down"),
    (-1, -1,  -1, -2,  "down"),
    # left column: stand 1 tile left of target, face right
    (-1,  0,  -2,  0,  "right"),
    (-1,  1,  -2,  1,  "right"),
    # bottom center: stand 1 tile below, face up
    ( 0,  1,   0,  2,  "up"),
]

# ── 3x3 water pattern ───────────────────────────────────────────────
# Water from INSIDE the plot using the center notch.
# Freshly planted crops are walkable for the first few days,
# so the bot can step onto adjacent crop tiles for corner watering.
#
# Phase 1: From center, water 4 cardinal neighbors
# Phase 2: Step up to top-middle, water 2 top corners
# Phase 3: Step down through center to bottom-middle, water 2 bottom corners
# Phase 4: Return to center

# Inner watering plan: stand inside the plot (center + adjacent crop tiles).
# Freshly planted crops are walkable for the first few days.
# Each entry: (target_dx, target_dy, stand_dx, stand_dy, face_dir)
WATER_PLAN_CENTER = [
    # From center: water 4 cardinal neighbors
    ( 0, -1,   0,  0,  "up"),      # top
    ( 1,  0,   0,  0,  "right"),   # right
    ( 0,  1,   0,  0,  "down"),    # bottom
    (-1,  0,   0,  0,  "left"),    # left
    # Center tile: water from top-middle facing down
    ( 0,  0,   0, -1,  "down"),    # center
    # Corners: alternate between right-middle and left-middle stands
    # so no two consecutive steps share a stand position (avoids drift)
    ( 1, -1,   1,  0,  "up"),      # top-right    from right-middle
    (-1, -1,  -1,  0,  "up"),      # top-left     from left-middle
    ( 1,  1,   1,  0,  "down"),    # bottom-right from right-middle
    (-1,  1,  -1,  0,  "down"),    # bottom-left  from left-middle
]

# Legacy plan for external watering (kept for reference)
WATER_PLAN = [
    ( 1, -1,   1, -2,  "down"),
    ( 0, -1,   0, -2,  "down"),
    (-1, -1,  -1, -2,  "down"),
    (-1,  0,  -2,  0,  "right"),
    (-1,  1,  -2,  1,  "right"),
    ( 0,  1,   0,  2,  "up"),
    ( 1,  1,   1,  2,  "up"),
    ( 1,  0,   2,  0,  "left"),
]


# ── helper functions ─────────────────────────────────────────────────

def hoe_plan(center: Tuple[int, int]) -> List[Tuple[Tuple[int, int], Tuple[int, int], str]]:
    """Return absolute (target_tile, stand_tile, face_dir) for hoeing a 3x3 plot.

    >>> plan = hoe_plan((35, 20))
    >>> target, stand, face = plan[0]
    """
    cx, cy = center
    return [
        ((cx + tdx, cy + tdy), (cx + sdx, cy + sdy), face)
        for tdx, tdy, sdx, sdy, face in HOE_PLAN
    ]


def water_plan(center: Tuple[int, int]) -> List[Tuple[Tuple[int, int], Tuple[int, int], str]]:
    """Return absolute (target_tile, stand_tile, face_dir) for watering a 3x3 plot."""
    cx, cy = center
    return [
        ((cx + tdx, cy + tdy), (cx + sdx, cy + sdy), face)
        for tdx, tdy, sdx, sdy, face in WATER_PLAN
    ]


def hoe_action_sequence(face_dir: str) -> List[np.ndarray]:
    """Action frames to hoe one tile: face 4f → settle 6f → Y 20f → cooldown 30f."""
    actions: List[np.ndarray] = []
    actions.extend([make_action(**{face_dir: True}) for _ in range(4)])
    actions.extend([make_action() for _ in range(6)])
    actions.extend(use_tool(frames=20, cooldown=30))
    return actions


def water_action_sequence(face_dir: str, cooldown: int = 25, face_frames: int = 4) -> List[np.ndarray]:
    """Action frames to water one tile: face → settle → use_tool_facing → cooldown.

    Uses use_tool_facing which does a 1-frame re-face then Y without combining
    direction+Y (avoids accidental movement).
    """
    actions: List[np.ndarray] = []
    # Pre-face to establish direction
    actions.extend([make_action(**{face_dir: True}) for _ in range(face_frames)])
    # Settle (stop movement before tool use)
    actions.extend([make_action() for _ in range(4)])
    # Tool use with facing stabilization
    actions.extend(use_tool_facing(face_dir, frames=15, cooldown=cooldown))
    return actions


def center_water_all() -> List[np.ndarray]:
    """Full action sequence to water all 8 tiles of a 3x3 plot from inside.

    Player must start at the plot center (the untilled notch).
    Freshly planted crops are walkable for the first few days,
    allowing the bot to step onto adjacent crop tiles.

    Sequence:
      1. From center: face up/right/down/left and water 4 cardinal tiles
      2. Step up 1 tile: face left and right to water 2 top corners
      3. Step down 2 tiles (through center): face left/right for 2 bottom corners
      4. Step back up to center

    Total: 8 waters + 3 movements.  ~700 frames.
    """
    cd = 45  # cooldown per water action (enough for RAM update)
    actions: List[np.ndarray] = []

    # Phase 1: Water 4 cardinal neighbors from center
    for face_dir in ("up", "right", "down", "left"):
        actions.extend(water_action_sequence(face_dir, cooldown=cd))

    # Phase 2: Step up 1 tile onto top-middle crop tile
    actions.extend([make_action(up=True) for _ in range(12)])
    actions.extend([make_action() for _ in range(10)])  # settle
    # Water top-left and top-right corners
    actions.extend(water_action_sequence("left", cooldown=cd))
    actions.extend(water_action_sequence("right", cooldown=cd))

    # Phase 3: Step down 2 tiles to bottom-middle crop tile
    actions.extend([make_action(down=True) for _ in range(28)])
    actions.extend([make_action() for _ in range(10)])  # settle
    # Water bottom-left and bottom-right corners
    actions.extend(water_action_sequence("left", cooldown=cd))
    actions.extend(water_action_sequence("right", cooldown=cd))

    # Phase 4: Return to center
    actions.extend([make_action(up=True) for _ in range(12)])
    actions.extend([make_action() for _ in range(10)])  # settle

    return actions


def plant_action_sequence() -> List[np.ndarray]:
    """Action frames to plant seeds: face down → Y → long cooldown (planting anim ~150f)."""
    actions: List[np.ndarray] = []
    actions.extend([make_action(down=True) for _ in range(4)])
    actions.extend([make_action() for _ in range(4)])
    actions.extend(use_tool(frames=15, cooldown=45))
    return actions


def refill_action_sequence(face_dir: str, face_frames: int = 2) -> List[np.ndarray]:
    """Action frames to refill watering can at pond edge.

    face_frames: fewer frames = less drift away from the water tile.
    """
    actions: List[np.ndarray] = []
    actions.extend([make_action(**{face_dir: True}) for _ in range(face_frames)])
    settle = max(1, 8 - face_frames)
    actions.extend([make_action() for _ in range(settle)])
    actions.extend(use_tool(frames=15, cooldown=45))
    return actions


def find_pond_edges(
    ram: np.ndarray,
    bounds: Tuple[int, int, int, int] = (3, 3, 62, 60),
    water_tiles: Optional[frozenset] = None,
) -> List[Tuple[Tuple[int, int], str]]:
    """Find walkable tiles adjacent to water, suitable for watering can refill.

    Returns list of (tile, face_dir) where tile is walkable and face_dir
    points toward adjacent water.

    water_tiles: set of tile IDs to consider as water.  Defaults to WATER_TILES
        (includes A6 pond border).  Pass REFILL_WATER_TILES for actual water only.
    """
    from farm_clearer import WALKABLE_TILES

    if water_tiles is None:
        water_tiles = WATER_TILES

    x_min, y_min, x_max, y_max = bounds
    results = []
    directions = [
        (0, -1, "up"), (0, 1, "down"), (-1, 0, "left"), (1, 0, "right"),
    ]
    for ty in range(y_min, y_max + 1):
        for tx in range(x_min, x_max + 1):
            tid = get_tile_at(ram, tx, ty)
            if tid not in WALKABLE_TILES:
                continue
            for dx, dy, face in directions:
                nx, ny = tx + dx, ty + dy
                if 0 <= nx < MAP_WIDTH and 0 <= ny < MAP_WIDTH:
                    ntid = get_tile_at(ram, nx, ny)
                    if ntid in water_tiles:
                        results.append(((tx, ty), face))
                        break  # one per walkable tile
    return results


def nearest_pond_edge(
    ram: np.ndarray,
    player_tile: Tuple[int, int],
    bounds: Tuple[int, int, int, int] = (3, 3, 62, 60),
) -> Optional[Tuple[Tuple[int, int], str]]:
    """Find the closest pond edge tile to the player.

    Returns (tile, face_dir) or None.
    """
    edges = find_pond_edges(ram, bounds)
    if not edges:
        return None
    px, py = player_tile
    best = None
    best_dist = float("inf")
    for tile, face in edges:
        d = abs(tile[0] - px) + abs(tile[1] - py)
        if d < best_dist:
            best_dist = d
            best = (tile, face)
    return best


def plot_tiles(center: Tuple[int, int], include_center: bool = False) -> List[Tuple[int, int]]:
    """All tile coordinates in a 3x3 plot."""
    cx, cy = center
    tiles = []
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if not include_center and dx == 0 and dy == 0:
                continue
            tiles.append((cx + dx, cy + dy))
    return tiles


def count_tilled(ram: np.ndarray, center: Tuple[int, int]) -> int:
    """Count how many tiles in the 3x3 are tilled (ready for planting)."""
    count = 0
    for tx, ty in plot_tiles(center, include_center=True):
        if get_tile_at(ram, tx, ty) in PLANTABLE_TILES:
            count += 1
    return count


def count_needs_water(ram: np.ndarray, center: Tuple[int, int], crop_tiles: Set[int]) -> int:
    """Count planted tiles that need watering (tile ID indicates dry crop)."""
    count = 0
    for tx, ty in plot_tiles(center, include_center=True):
        if get_tile_at(ram, tx, ty) in crop_tiles:
            count += 1
    return count


# ── crop detection & watering task ─────────────────────────────────

from dataclasses import dataclass, field
from collections import deque

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

# Crop growth tile ranges
CROP_TILE_RANGE = range(0x1E, 0x70)  # all crop growth stages

# Tiles indicating a crop plot exists (tilled + crop stages)
PLOT_TILES = {FRESH_TILLED, 0x08} | set(CROP_TILE_RANGE)

# Tiles worth watering (crop stages + freshly tilled)
WATERABLE_TILES = {FRESH_TILLED} | set(CROP_TILE_RANGE)

# Seed data key names in data.json (for env.data.set_value)
SEED_DATA_KEY: Dict[str, str] = {
    "potato": "potato_seeds",
    "turnip": "turnip_seeds",
    "corn": "corn_seeds",
    "tomato": "tomato_seeds",
}

DEFAULT_CROP_BOUNDS = (3, 3, 62, 60)

# Watering can fill level (RAM address 0x0926, max 20, decreases by 1 per use)
ADDR_WATER_LEVEL = 0x0926
WATER_LEVEL_MAX = 20
WATER_REFILL_THRESHOLD = 1  # refill when level drops to this



def _count_plot_tiles(ram: np.ndarray, cx: int, cy: int) -> int:
    """Count PLOT_TILES in the 8 surrounding tiles of a candidate center."""
    count = 0
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            tid = get_tile_at(ram, cx + dx, cy + dy)
            if tid in PLOT_TILES:
                count += 1
    return count


def _refine_center(ram: np.ndarray, cx: int, cy: int) -> Tuple[int, int]:
    """Refine a candidate center by testing offsets [-1, 0, +1] in x and y.

    Picks the offset that maximizes the PLOT_TILES count in the 3x3 area.
    Breaks ties toward the original position.
    """
    best = (cx, cy)
    best_count = _count_plot_tiles(ram, cx, cy)
    for oy in range(-1, 2):
        for ox in range(-1, 2):
            if ox == 0 and oy == 0:
                continue
            nx, ny = cx + ox, cy + oy
            c = _count_plot_tiles(ram, nx, ny)
            if c > best_count:
                best_count = c
                best = (nx, ny)
    return best


def detect_plots(
    ram: np.ndarray,
    bounds: Tuple[int, int, int, int] = DEFAULT_CROP_BOUNDS,
) -> List[Tuple[int, int]]:
    """Auto-detect 3x3 crop plot centers by scanning for tile clusters.

    A tile is a plot center if >= 5 of 8 surrounding tiles are in PLOT_TILES.
    After detection, each center is refined by testing nearby offsets to find
    the true centroid (fixes off-by-one alignment near edges).
    De-duplicates overlapping centers (within 2 tiles of each other).
    """
    x_min, y_min, x_max, y_max = bounds
    candidates = []
    for cy in range(y_min + 1, y_max):
        for cx in range(x_min + 1, x_max):
            count = _count_plot_tiles(ram, cx, cy)
            if count >= 5:
                candidates.append((cx, cy))

    # De-duplicate: keep first center when two are within 2 tiles
    kept: List[Tuple[int, int]] = []
    for c in candidates:
        too_close = False
        for k in kept:
            if abs(c[0] - k[0]) <= 2 and abs(c[1] - k[1]) <= 2:
                too_close = True
                break
        if not too_close:
            kept.append(c)

    # Refine each kept center to find the true centroid
    refined = []
    for cx, cy in kept:
        rx, ry = _refine_center(ram, cx, cy)
        if (rx, ry) != (cx, cy):
            print(f"[CROP] Refined center ({cx},{cy}) -> ({rx},{ry})")
        refined.append((rx, ry))
    return refined


# ── CropWaterTask ──────────────────────────────────────────────────

@dataclass
class CropWaterTask(Task):
    """Detect crop plots, plant seeds on tilled tiles, water all crops.

    Follows the GrassPlantTask state machine pattern:
      detect -> navigate -> center -> act -> verify -> tool_switch

    Fixes vs v1:
      - Planting: explicit tile position check (must be ON center tile)
      - Watering: waters all 8 tiles blindly, tracks per-plot 8/8
      - Refill: RAM-based (reads actual water level at 0x0926), verifies success
      - Center detection: refined with offset search to fix alignment
    """

    name: str = "crop_water"
    seed_type: str = "potato"
    bounds: Tuple[int, int, int, int] = DEFAULT_CROP_BOUNDS
    max_steps_per_target: int = 1200
    stasis_repath: int = 180
    max_failures: int = 50
    refill_bounds: Optional[Tuple[int, int, int, int]] = None
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
    _state: str = field(default="detect", init=False)
    _action_queue: deque = field(default_factory=deque, init=False)
    _steps_on_target: int = field(default=0, init=False)
    _total_steps: int = field(default=0, init=False)
    _failures: int = field(default=0, init=False)
    _failed_tiles: Set[Tuple[int, int]] = field(default_factory=set, init=False)

    # Per-plot phase tracking
    _plot_phase: str = field(default="plant", init=False)  # "plant", "water", "refill"
    _water_steps: List[Tuple[Tuple[int, int], Tuple[int, int], str]] = field(default_factory=list, init=False)
    _water_index: int = field(default=0, init=False)
    _plot_watered: int = field(default=0, init=False)   # per-plot water count
    _plot_skipped: int = field(default=0, init=False)   # per-plot skip count
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

    # Water verification
    _pre_water_level: int = field(default=-1, init=False)  # water level before watering action

    # Counters
    planted_count: int = field(default=0, init=False)
    watered_count: int = field(default=0, init=False)
    skipped_water: int = field(default=0, init=False)
    refill_count: int = field(default=0, init=False)

    def __post_init__(self):
        self._pathfinder = Pathfinder(self._scanner)
        self._navigator = Navigator(self._pathfinder)

    @staticmethod
    def _water_level(ram: np.ndarray) -> int:
        """Read watering can fill level from RAM (0 = empty, 20 = full)."""
        if ADDR_WATER_LEVEL < len(ram):
            return int(ram[ADDR_WATER_LEVEL])
        return 0

    def reset(self, world: WorldState) -> None:
        if os.getenv("CROP_DEBUG", "").lower() in ("1", "true", "yes"):
            self.debug = True
        self._state = "detect"
        self._plots = []
        self._plot_index = 0
        self._pass_number = 1
        self._plot_phase = "plant"
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
        self._refill_pond_tile = None
        self._refill_pond_face = None
        self._refill_level_before = 0
        self._refill_search_level = -1
        self._bad_refill_tiles = set()
        self._refill_exhausted = False
        self._pre_water_level = -1
        self._resume_water_index = 0
        self.planted_count = 0
        self.watered_count = 0
        self.skipped_water = 0
        self.refill_count = 0
        self._clear_crop_walkable()
        self._navigator.update(world.ram)
        self._tool_mgr.update(world.ram)

    def can_start(self, world: WorldState) -> bool:
        return True

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _handle_detect(self, ram: np.ndarray) -> Optional[TaskResult]:
        """Scan for crop plots."""
        self._plots = detect_plots(ram, self.bounds)
        if not self._plots:
            if self._pass_number == 1:
                return TaskResult(status=TaskStatus.SUCCESS, reason="no plots detected")
            else:
                self._state = "done"
                return None
        self._plot_index = 0
        pass_label = f"(pass {self._pass_number})" if self._pass_number > 1 else ""
        print(f"[CROP] Detected {len(self._plots)} plots: {self._plots} {pass_label}")
        self._start_plot(ram)
        return None

    def _start_plot(self, ram: np.ndarray):
        """Begin processing the current plot."""
        if self._plot_index >= len(self._plots):
            return
        center = self._plots[self._plot_index]
        self._set_crop_walkable()  # allow pathfinding through crop tiles
        tilled = count_tilled(ram, center)
        if tilled > 0:
            self._plot_phase = "plant"
            self._target_tile = center
            self._approach_tile = center  # stand ON center to plant
            self._state = "navigate"
            self._navigator.path = []  # force re-path
            self._steps_on_target = 0
            print(f"[CROP] Plot {self._plot_index + 1}/{len(self._plots)} center=({center[0]},{center[1]}) phase=PLANT tilled={tilled}")
        else:
            # Skip to water phase
            self._begin_water_phase(ram)

    def _begin_water_phase(self, ram: np.ndarray):
        """Set up per-tile watering for current plot using WATER_PLAN_CENTER."""
        if self._plot_index >= len(self._plots):
            return
        center = self._plots[self._plot_index]
        cx, cy = center
        self._plot_phase = "water"
        self._plot_watered = 0
        self._plot_skipped = 0
        self._set_crop_walkable()  # allow pathfinding through plot

        # Build per-tile water steps from center
        self._water_steps = [
            ((cx + tdx, cy + tdy), (cx + sdx, cy + sdy), face)
            for tdx, tdy, sdx, sdy, face in WATER_PLAN_CENTER
        ]
        self._water_index = 0

        water_lvl = self._water_level(ram)
        self._pre_water_level = water_lvl  # plot-level: track starting level
        print(f"[CROP] Plot {self._plot_index + 1}/{len(self._plots)} center=({cx},{cy}) phase=WATER can={water_lvl}")

        # Navigate to first stand position
        target, stand, face = self._water_steps[0]
        self._target_tile = target
        self._approach_tile = stand
        self._face_direction = face
        self._state = "navigate"
        self._navigator.path = []
        self._steps_on_target = 0

    @staticmethod
    def _face_from_approach(approach: Tuple[int, int], target: Tuple[int, int]) -> str:
        """Derive face direction from stand tile toward target tile."""
        dx = target[0] - approach[0]
        dy = target[1] - approach[1]
        if abs(dx) >= abs(dy):
            return "right" if dx > 0 else "left"
        return "down" if dy > 0 else "up"

    def _set_crop_walkable(self):
        """Mark current plot's 3x3 tiles as walkable on the pathfinder.

        Freshly planted crops are walkable in-game for the first few days.
        Sets pathfinder.extra_walkable so both find_path and follow_path work.
        """
        self._pathfinder.extra_walkable.clear()
        if self._plot_index < len(self._plots):
            center = self._plots[self._plot_index]
            self._pathfinder.extra_walkable = set(plot_tiles(center, include_center=True))

    def _clear_crop_walkable(self):
        """Remove crop walkable overrides from pathfinder."""
        self._pathfinder.extra_walkable.clear()

    def _advance_plot(self, ram: np.ndarray):
        """Move to the next plot, or trigger a re-scan pass, or finish."""
        self._clear_crop_walkable()
        self._plot_index += 1
        if self._plot_index >= len(self._plots):
            if self._pass_number < 3 and self.skipped_water > 0:
                prev_skip = self.skipped_water
                self._pass_number += 1
                self._state = "detect"
                self._pathfinder.temp_blocked.clear()
                self._refill_exhausted = False
                print(f"[CROP] Pass {self._pass_number - 1} complete ({prev_skip} skipped), "
                      f"starting pass {self._pass_number}...")
            else:
                self._state = "done"
        else:
            self._start_plot(ram)

    def _advance_water_step(self, ram: np.ndarray):
        """Move to the next water step, or finish the plot."""
        self._water_index += 1
        if self._water_index >= len(self._water_steps):
            # All tiles attempted — plot-level verification
            center = self._plots[self._plot_index]
            cx, cy = center
            lvl = self._water_level(ram)
            water_used = max(0, self._pre_water_level - lvl) if self._pre_water_level >= 0 else 0
            actual_watered = min(water_used, self._plot_watered)
            actual_skipped = self._plot_watered - actual_watered + self._plot_skipped

            self.watered_count += actual_watered
            self.skipped_water += actual_skipped

            tile_ids = []
            for dy in range(-1, 2):
                row = []
                for dx in range(-1, 2):
                    tid = get_tile_at(ram, cx + dx, cy + dy)
                    row.append(f"0x{tid:02X}")
                tile_ids.append(" ".join(row))
            print(f"[CROP] Plot {self._plot_index + 1}/{len(self._plots)} WATER DONE: "
                  f"{actual_watered}/{len(self._water_steps)} watered (used {water_used} water, can={lvl})")
            print(f"[CROP]   3x3 tiles: [{tile_ids[0]}] [{tile_ids[1]}] [{tile_ids[2]}]")
            if actual_skipped > 0:
                print(f"[CROP] WARNING: Plot {self._plot_index + 1} incomplete ({actual_skipped} skipped)")
            self._pre_water_level = -1
            self._advance_plot(ram)
        else:
            # Navigate to next stand position
            target, stand, face = self._water_steps[self._water_index]
            self._target_tile = target
            self._approach_tile = stand
            self._face_direction = face
            self._state = "navigate"
            self._navigator.path = []
            self._steps_on_target = 0

    def _start_refill(self, ram: np.ndarray):
        """Navigate to nearest water source to refill watering can.

        Searches the full farm bounds for walkable tiles adjacent to ACTUAL
        water (F0+ tiles, not A6 pond borders).  Sorts by distance, path-verifies.
        Excludes tiles that previously failed to refill.
        After refill, navigates back to plot center to resume watering.
        """
        current_lvl = self._water_level(ram)

        # Track when refill search starts; detect water leaking during search
        if self._refill_search_level < 0:
            self._refill_search_level = current_lvl
        elif current_lvl < self._refill_search_level:
            # Water leaked during failed refill attempts — the area is harmful.
            # Blacklist all previously-tried tiles + neighborhoods.
            leaked = self._refill_search_level - current_lvl
            print(f"[CROP] Refill search leaked {leaked} water (was {self._refill_search_level}, now {current_lvl})")
            for bad in list(self._bad_refill_tiles):
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        self._bad_refill_tiles.add((bad[0] + dx, bad[1] + dy))
            self._refill_search_level = current_lvl  # reset for next attempts

        edges = find_pond_edges(ram, self.refill_bounds or self.bounds, water_tiles=REFILL_WATER_TILES)
        if self._bad_refill_tiles:
            edges = [(t, f) for t, f in edges if t not in self._bad_refill_tiles]
        if not edges:
            self._refill_exhausted = True
            remaining = len(self._water_steps) - self._water_index
            print(f"[CROP] No water found for refill, skipping {remaining} remaining tiles")
            self.skipped_water += remaining
            self._plot_skipped += remaining
            self._water_index = len(self._water_steps)
            self._advance_water_step(ram)
            return

        player = self._navigator.current_tile
        edges.sort(key=lambda e: abs(e[0][0] - player[0]) + abs(e[0][1] - player[1]))

        chosen = None
        for tile, face in edges[:10]:
            path = self._pathfinder.find_path(ram, player, tile)
            if path is not None:
                chosen = (tile, face)
                break

        if chosen is None:
            self._refill_exhausted = True
            remaining = len(self._water_steps) - self._water_index
            print(f"[CROP] No reachable water edge (checked {min(len(edges), 10)}/{len(edges)}), skipping {remaining} tiles")
            self.skipped_water += remaining
            self._plot_skipped += remaining
            self._water_index = len(self._water_steps)
            self._advance_water_step(ram)
            return

        self._refill_pond_tile, self._refill_pond_face = chosen
        self._refill_level_before = current_lvl
        self._clear_crop_walkable()  # refill pathing shouldn't use crop override
        self._plot_phase = "refill"
        self._target_tile = self._refill_pond_tile
        self._approach_tile = self._refill_pond_tile
        self._face_direction = self._refill_pond_face
        self._state = "navigate"
        self._navigator.path = []
        self._steps_on_target = 0
        dist = abs(chosen[0][0] - player[0]) + abs(chosen[0][1] - player[1])
        print(f"[CROP] Refill at ({chosen[0][0]},{chosen[0][1]}) facing {chosen[1]} dist={dist} can={current_lvl}")

    def _handle_navigate(self, ram: np.ndarray) -> Optional[TaskResult]:
        if self._target_tile is None or self._approach_tile is None:
            self._state = "detect"
            return None

        # Arrived?
        if self._navigator.current_tile == self._approach_tile:
            self._state = "center"
            return None

        # Stuck recovery
        if self._navigator.stasis > self.stasis_repath and self._navigator.path:
            self._pathfinder.temp_blocked.add(self._navigator.path[0])
            path = self._pathfinder.find_path(ram, self._navigator.current_tile, self._approach_tile)
            if path:
                self._navigator.path = path
                self._navigator.stasis = 0
            else:
                self._failures += 1
                self._failed_tiles.add(self._target_tile)
                if self._plot_phase == "water":
                    self.skipped_water += 1
                    self._plot_skipped += 1
                    print(f"[CROP] SKIP water tile {self._water_index + 1}/{len(self._water_steps)} (stuck nav) target={self._target_tile}")
                    self._advance_water_step(ram)
                elif self._plot_phase == "refill":
                    print("[CROP] Can't reach pond, skipping refill")
                    self._refill_exhausted = True
                    self._plot_phase = "water"
                    self._set_crop_walkable()
                    if self._water_index < len(self._water_steps):
                        target, stand, face = self._water_steps[self._water_index]
                        self._target_tile = target
                        self._approach_tile = stand
                        self._face_direction = face
                    else:
                        center = self._plots[self._plot_index]
                        self._target_tile = center
                        self._approach_tile = center
                    self._state = "navigate"
                    self._navigator.path = []
                else:
                    self._state = "detect"
                if self._failures >= self.max_failures:
                    return TaskResult(status=TaskStatus.FAILURE, reason="too many nav failures")
                return None

        # Try to path if no current path
        if not self._navigator.path:
            path = self._pathfinder.find_path(ram, self._navigator.current_tile, self._approach_tile)
            if path:
                self._navigator.path = path
                self._navigator.stasis = 0
            else:
                self._failures += 1
                if self._plot_phase == "water":
                    self.skipped_water += 1
                    self._plot_skipped += 1
                    print(f"[CROP] SKIP water tile {self._water_index + 1}/{len(self._water_steps)} (no path) target={self._target_tile}")
                    self._advance_water_step(ram)
                elif self._plot_phase == "refill":
                    print("[CROP] No path to pond, skipping refill")
                    self._refill_exhausted = True
                    self._plot_phase = "water"
                    self._set_crop_walkable()
                    if self._water_index < len(self._water_steps):
                        target, stand, face = self._water_steps[self._water_index]
                        self._target_tile = target
                        self._approach_tile = stand
                        self._face_direction = face
                    else:
                        center = self._plots[self._plot_index]
                        self._target_tile = center
                        self._approach_tile = center
                    self._state = "navigate"
                    self._navigator.path = []
                else:
                    self._state = "detect"
                return None

        action = self._navigator.follow_path(ram)
        if action is not None:
            self._action_queue.append(action)
        return None

    def _handle_center(self, ram: np.ndarray) -> Optional[TaskResult]:
        if self._approach_tile is None:
            self._state = "detect"
            return None
        tol = 1 if self._plot_phase in ("plant", "water") else 2
        center_action = self._navigator.center_on_tile(self._approach_tile, tolerance=tol)
        if center_action is None:
            self._state = "act"
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
        if self._plot_phase in ("plant", "water"):
            # Must be ON center tile for planting and watering (uses center notch)
            if player != self._approach_tile:
                print(f"[CROP] {self._plot_phase.upper()} pos mismatch: at ({player[0]},{player[1]}) need ({self._approach_tile[0]},{self._approach_tile[1]}), re-navigate")
                self._state = "navigate"
                self._navigator.path = []
                return None
        else:
            # Refill: on or adjacent to approach tile
            if tile_dist(player, self._approach_tile) > 1:
                self._state = "navigate"
                self._navigator.path = []
                return None

        # Re-center drift correction
        tol = 1 if self._plot_phase in ("plant", "water") else 2
        center_action = self._navigator.center_on_tile(self._approach_tile, tolerance=tol)
        if center_action is not None:
            self._action_queue.append(center_action)
            return None

        if self._plot_phase == "plant":
            return self._act_plant(ram)
        elif self._plot_phase == "water":
            return self._act_water(ram)
        elif self._plot_phase == "refill":
            return self._act_refill(ram)
        return None

    def _act_plant(self, ram: np.ndarray) -> Optional[TaskResult]:
        """Plant seeds at current plot center."""
        seed_item = SEED_ITEM.get(self.seed_type, SEED_ITEM["potato"])
        if self._tool_mgr.current != seed_item:
            self._tool_mgr.start_search()
            self._state = "tool_switch"
            return None

        center = self._plots[self._plot_index]
        player = self._navigator.current_tile
        # Debug: dump 3x3 tile IDs around center
        cx, cy = center
        tile_ids = []
        for dy in range(-1, 2):
            row = []
            for dx in range(-1, 2):
                tid = get_tile_at(ram, cx + dx, cy + dy)
                row.append(f"0x{tid:02X}")
            tile_ids.append(" ".join(row))
        print(f"[CROP] PLANT at ({cx},{cy}) player=({player[0]},{player[1]}) seed=0x{seed_item:02X}")
        print(f"[CROP]   3x3 tiles: [{tile_ids[0]}] [{tile_ids[1]}] [{tile_ids[2]}]")

        # Face → settle → Y → long cooldown.  Plant animation takes ~150f
        # so use 90f cooldown to ensure tile data updates before verify.
        self._action_queue.extend([make_action(down=True) for _ in range(4)])  # face down
        self._action_queue.extend([make_action() for _ in range(6)])           # settle
        self._action_queue.extend(use_tool(frames=20, cooldown=90))            # Y + long cooldown
        self._state = "verify"
        return None

    def _act_water(self, ram: np.ndarray) -> Optional[TaskResult]:
        """Water current tile using navigator-precise positioning."""
        if self._tool_mgr.current != Tool.WATERING_CAN:
            self._tool_mgr.start_search()
            self._state = "tool_switch"
            return None

        # Skip tiles that don't need watering (dried tilled, untilled, etc.)
        if self._water_index < len(self._water_steps):
            target = self._water_steps[self._water_index][0]
            tid = get_tile_at(ram, target[0], target[1])
            if tid not in WATERABLE_TILES:
                print(f"[CROP] SKIP water tile {self._water_index + 1}/{len(self._water_steps)} target={target} "
                      f"tid=0x{tid:02X} (not waterable)")
                self._advance_water_step(ram)
                return None

        water_lvl = self._water_level(ram)

        # Count only waterable remaining tiles for refill check
        waterable_remaining = 0
        for i in range(self._water_index, len(self._water_steps)):
            t = self._water_steps[i][0]
            if get_tile_at(ram, t[0], t[1]) in WATERABLE_TILES:
                waterable_remaining += 1

        if water_lvl < waterable_remaining and not self._refill_exhausted:
            print(f"[CROP] Water level={water_lvl} < {waterable_remaining} waterable remaining, need refill")
            self._start_refill(ram)
            return None

        if water_lvl < 1 and self._refill_exhausted:
            # Empty and can't refill — skip remaining tiles
            remaining = len(self._water_steps) - self._water_index
            print(f"[CROP] Empty can, no refill, skipping {remaining} remaining tiles")
            self.skipped_water += remaining
            self._plot_skipped += remaining
            self._water_index = len(self._water_steps)
            self._advance_water_step(ram)
            return None

        face = self._face_direction or "down"

        if self.debug or self._water_index == 0:
            target = self._water_steps[self._water_index][0] if self._water_index < len(self._water_steps) else None
            print(f"[CROP] WATER tile {self._water_index + 1}/{len(self._water_steps)} target={target} face={face} can={water_lvl}")

        self._action_queue.extend(water_action_sequence(face, cooldown=60, face_frames=1))
        self._state = "verify"
        return None

    def _act_refill(self, ram: np.ndarray) -> Optional[TaskResult]:
        """Refill watering can at pond."""
        if self._tool_mgr.current != Tool.WATERING_CAN:
            self._tool_mgr.start_search()
            self._state = "tool_switch"
            return None

        face = self._refill_pond_face or "down"
        # Record level right before action (not during _start_refill which is pre-navigation)
        self._refill_level_before = self._water_level(ram)
        print(f"[CROP] REFILL facing {face} can={self._refill_level_before}")

        self._action_queue.extend(refill_action_sequence(face))
        self._state = "verify"
        return None

    def _handle_verify(self, ram: np.ndarray) -> Optional[TaskResult]:
        if self._action_queue:
            return None

        input_lock = int(ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(ram) else 0
        if input_lock != 1:
            return None

        if self._plot_phase == "plant":
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
            self._begin_water_phase(ram)

        elif self._plot_phase == "water":
            # Per-tile: just advance. Plot-level verification in _advance_water_step
            # (RAM water level update is delayed past the per-tile cooldown window)
            self._plot_watered += 1
            self._advance_water_step(ram)

        elif self._plot_phase == "refill":
            lvl_after = self._water_level(ram)
            if lvl_after > self._refill_level_before:
                # Refill succeeded — navigate back to current water step
                self.refill_count += 1
                print(f"[CROP] REFILL OK can={lvl_after} (was {self._refill_level_before}) refills={self.refill_count}")
                self._pre_water_level = lvl_after  # reset for plot-level verification
                self._refill_search_level = -1  # reset search tracking
                self._plot_phase = "water"
                self._set_crop_walkable()  # re-enable crop walkable for return path
                if self._water_index < len(self._water_steps):
                    target, stand, face = self._water_steps[self._water_index]
                    self._target_tile = target
                    self._approach_tile = stand
                    self._face_direction = face
                else:
                    center = self._plots[self._plot_index]
                    self._target_tile = center
                    self._approach_tile = center
                self._state = "navigate"
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
                self._plot_phase = "water"
                self._start_refill(ram)  # try another water edge

        if self._state == "done":
            msg = f"planted={self.planted_count} watered={self.watered_count} refills={self.refill_count}"
            if self.skipped_water:
                msg += f" skipped={self.skipped_water}"
            msg += f" passes={self._pass_number}"
            print(f"[CROP] Complete: {msg}")
            return TaskResult(status=TaskStatus.SUCCESS, reason=msg)

        return None

    def _handle_tool_switch(self, ram: np.ndarray) -> Optional[TaskResult]:
        """Cycle tools to find the needed one."""
        if self._plot_phase == "plant":
            wanted = SEED_ITEM.get(self.seed_type, SEED_ITEM["potato"])
        else:
            wanted = Tool.WATERING_CAN

        self._tool_mgr.update(ram)
        current = self._tool_mgr.current

        if current == wanted:
            if self.debug:
                print(f"[CROP] Found tool 0x{wanted:02X}")
            self._state = "center"
            return None

        self._tool_mgr.record()

        if self._tool_mgr.cycle_complete():
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

        if self.debug and self._total_steps % self.debug_interval == 0:
            cur = self._navigator.current_tile
            print(f"[CROP] step={self._total_steps} phase={self._plot_phase} state={self._state} "
                  f"pos={cur} plot={self._plot_index}/{len(self._plots)} "
                  f"planted={self.planted_count} watered={self.watered_count} can={self._water_level(world.ram)}")

        # Timeout per target
        if self._steps_on_target > self.max_steps_per_target and self._target_tile is not None:
            self._failed_tiles.add(self._target_tile)
            self._failures += 1
            self._action_queue.clear()
            if self._plot_phase == "water":
                self.skipped_water += 1
                self._plot_skipped += 1
                print(f"[CROP] SKIP water tile {self._water_index + 1}/{len(self._water_steps)} (timeout) target={self._target_tile}")
                self._advance_water_step(world.ram)
            elif self._plot_phase == "refill":
                print("[CROP] Refill timed out, marking bad")
                if self._refill_pond_tile:
                    self._bad_refill_tiles.add(self._refill_pond_tile)
                # Navigate back to current water step
                self._plot_phase = "water"
                self._set_crop_walkable()
                if self._water_index < len(self._water_steps):
                    target, stand, face = self._water_steps[self._water_index]
                    self._target_tile = target
                    self._approach_tile = stand
                    self._face_direction = face
                else:
                    center = self._plots[self._plot_index]
                    self._target_tile = center
                    self._approach_tile = center
                self._state = "navigate"
                self._navigator.path = []
            else:
                self._target_tile = None
                self._state = "detect"
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
        if self._state == "done":
            msg = f"planted={self.planted_count} watered={self.watered_count} refills={self.refill_count}"
            return TaskResult(status=TaskStatus.SUCCESS, reason=msg)

        # State dispatch
        handlers = {
            "detect": self._handle_detect,
            "navigate": self._handle_navigate,
            "center": self._handle_center,
            "act": self._handle_act,
            "verify": self._handle_verify,
            "tool_switch": self._handle_tool_switch,
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

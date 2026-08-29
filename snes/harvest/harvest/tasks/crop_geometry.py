"""Plot geometry, tile classification, and water-step builders for crops.

Pond/refill stand ranking lives in water_refill (re-exported here).
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Set, Tuple

import numpy as np

from harvest.core.ram_catalog import field_spec, read_ram_u16
from harvest.tasks.nav import (
    make_action,
    get_tile_at,
    tile_dist,
    MAP_WIDTH,
    WALKABLE_TILES,
)
from harvest.tasks.farm_ops import use_tool, use_tool_facing
from harvest.tasks.water_refill import (  # noqa: F401
    BAD_REFILL_STAND_BOUNDS,
    MAIN_POND_STAND_BOUNDS,
    REFILL_BAND_BAD,
    REFILL_BAND_MID,
    REFILL_BAND_NORTH,
    REFILL_BAND_POND,
    REFILL_BAND_SOUTH,
    REFILL_WATER_TILES,
    WATER_TILES,
    edge_water_tile_id,
    find_pond_edges,
    is_bad_refill_stand,
    is_main_pond_stand,
    nearest_pond_edge,
    pond_access_blocking_fences,
    refill_edge_sort_key,
    refill_stand_band,
)

# ── tile IDs ─────────────────────────────────────────────────────────

FRESH_TILLED = 0x07
DRIED_TILLED = 0x02
WATERED_TILLED = 0x08
UNTILLED = 0x01
TILLABLE_TILES = {UNTILLED, DRIED_TILLED}
PLANTABLE_TILES = {FRESH_TILLED}

# 3x3 hoe: (target_dx, target_dy, stand_dx, stand_dy, face_dir).
# Center stays untilled — plant stand. Clockwise from top-right.
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

# Inner watering: (target_dx, target_dy, stand_dx, stand_dy, face_dir).
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


def water_action_sequence(face_dir: str, cooldown: int = 18, face_frames: int = 3) -> List[np.ndarray]:
    """Face, settle, then use_tool_facing (no combined direction+Y)."""
    actions: List[np.ndarray] = []
    actions.extend([make_action(**{face_dir: True}) for _ in range(face_frames)])
    actions.extend([make_action() for _ in range(4)])
    actions.extend(use_tool_facing(face_dir, frames=15, cooldown=cooldown))
    return actions


def center_water_all() -> List[np.ndarray]:
    """Water all 8 tiles of a 3x3 from the untilled center notch (~700f)."""
    cd = 32  # cooldown per water action (enough for RAM update)
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
    """Action frames to plant seeds: face down → Y → cooldown for plant anim."""
    actions: List[np.ndarray] = []
    actions.extend([make_action(down=True) for _ in range(3)])
    actions.extend([make_action() for _ in range(2)])
    actions.extend(use_tool(frames=12, cooldown=36))
    return actions


def refill_action_sequence(face_dir: str, face_frames: int = 2) -> List[np.ndarray]:
    """Action frames to refill watering can at pond edge.

    face_frames: fewer frames = less drift away from the water tile.
    Cooldown must cover ToolAnimationWateringCan writing can=0x14 after Y.
    """
    actions: List[np.ndarray] = []
    actions.extend([make_action(**{face_dir: True}) for _ in range(face_frames)])
    settle = max(1, 8 - face_frames)
    actions.extend([make_action() for _ in range(settle)])
    # 45f was tight for animation write; 75f leaves room for can=0x14.
    actions.extend(use_tool(frames=15, cooldown=75))
    return actions


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
        if tile_needs_watering(get_tile_at(ram, tx, ty)):
            count += 1
    return count


# ── crop detection & watering task ─────────────────────────────────

# Crop growth tile ranges
CROP_TILE_RANGE = range(0x1E, 0x70)  # all crop growth stages
DRY_CROP_TILES = {tile_id for tile_id in CROP_TILE_RANGE if tile_id % 2 == 0}
WET_CROP_TILES = set(CROP_TILE_RANGE) - DRY_CROP_TILES
MATURE_CROP_TILES = {
    0x38, 0x39,  # tomato
    0x52, 0x53,  # corn
    0x60, 0x61,  # potato
    0x6E, 0x6F,  # turnip
}

# Tiles indicating a crop plot exists (tilled + crop stages)
PLOT_TILES = {FRESH_TILLED, WATERED_TILLED} | set(CROP_TILE_RANGE)

# Tiles worth watering right now. Established crop states appear to use
# dry/wet pairs where the dry state is even and the watered state is the next
# odd tile ID, e.g. 0x54->0x55 and 0x58->0x59.  Fully grown crops should be
# harvested, not watered, and raw tilled soil is only waterable immediately
# after this task plants seeds.
UNRIPE_DRY_CROP_TILES = {
    tile_id for tile_id in DRY_CROP_TILES
    if tile_id not in MATURE_CROP_TILES
}
WATERABLE_TILES = set(UNRIPE_DRY_CROP_TILES)

DEFAULT_CROP_BOUNDS = (2, 3, 62, 60)

# Watering can fill level (RAM address 0x0926, max 20, decreases by 1 per use)
ADDR_WATER_LEVEL = 0x0926
WATER_LEVEL_MAX = 20
WATER_REFILL_THRESHOLD = 1  # refill when level drops to this
ADDR_WEATHER = field_spec("weather").address
ADDR_WEATHER_FLAGS = 0x0196
RAINY_WEATHER_CODES = frozenset({1, 2, 3})
RAINY_WEATHER_FLAG_MASK = 0x0002 | 0x0008 | 0x0010


def is_crop_tile(tile_id: int) -> bool:
    return tile_id in CROP_TILE_RANGE


def is_dry_crop_tile(tile_id: int) -> bool:
    return tile_id in DRY_CROP_TILES


def is_watered_crop_tile(tile_id: int) -> bool:
    return tile_id in WET_CROP_TILES


def crop_pickup_stage(tile_id: int) -> int:
    """Normalize dry/wet crop tile pairs to one growth stage."""
    return tile_id if tile_id % 2 == 0 else tile_id - 1


def is_mature_crop_tile(tile_id: int) -> bool:
    return tile_id in MATURE_CROP_TILES


def tile_is_watered(tile_id: int) -> bool:
    return tile_id == WATERED_TILLED or is_watered_crop_tile(tile_id)


def tile_needs_watering(tile_id: int, *, include_fresh_tilled: bool = False) -> bool:
    if is_mature_crop_tile(tile_id):
        return False
    if include_fresh_tilled and tile_id == FRESH_TILLED:
        return True
    return tile_id in UNRIPE_DRY_CROP_TILES


def count_crop_survival(
    ram: np.ndarray,
    bounds: Tuple[int, int, int, int] = DEFAULT_CROP_BOUNDS,
) -> dict:
    """Count live crop tiles in bounds for multi-day keep-alive acceptance.

    Returns keys: crop, mature, dry, wet, samples (up to 12 (x,y,tid) tuples).
    Only meaningful when the farm metatile map is loaded (tilemap farm).
    """
    x0, y0, x1, y1 = bounds
    crop = mature = dry = wet = 0
    samples: List[Tuple[int, int, int]] = []
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            tid = int(get_tile_at(ram, tx, ty))
            if not is_crop_tile(tid):
                continue
            crop += 1
            if is_mature_crop_tile(tid):
                mature += 1
            elif tile_needs_watering(tid):
                dry += 1
            elif is_watered_crop_tile(tid):
                wet += 1
            if len(samples) < 12:
                samples.append((tx, ty, tid))
    return {
        "crop": crop,
        "mature": mature,
        "dry": dry,
        "wet": wet,
        "samples": samples,
    }


def tile_can_be_water_target(
    tile_id: int,
    allow_unknown: bool = False,
    *,
    include_fresh_tilled: bool = False,
) -> bool:
    if tile_is_watered(tile_id):
        return False
    if is_mature_crop_tile(tile_id):
        return False
    if tile_needs_watering(tile_id, include_fresh_tilled=include_fresh_tilled):
        return True
    return allow_unknown and is_crop_tile(tile_id) and not is_mature_crop_tile(tile_id)


def is_rainy_weather(ram: np.ndarray) -> bool:
    flags = read_ram_u16(ram, ADDR_WEATHER_FLAGS, live_offset=False)
    return bool(flags & RAINY_WEATHER_FLAG_MASK)


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


def _count_crop_tiles(ram: np.ndarray, cx: int, cy: int) -> int:
    """Count crop-stage tiles in the 8 surrounding tiles of a candidate center."""
    count = 0
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            if is_crop_tile(get_tile_at(ram, cx + dx, cy + dy)):
                count += 1
    return count


def _merge_plot_centers(
    primary: List[Tuple[int, int]],
    secondary: List[Tuple[int, int]],
    suppress_radius: int = 2,
) -> List[Tuple[int, int]]:
    merged = list(primary)
    for cx, cy in secondary:
        too_close = any(
            max(abs(cx - mx), abs(cy - my)) <= suppress_radius
            for mx, my in merged
        )
        if not too_close:
            merged.append((cx, cy))
    return merged


def detect_crop_resume_plots(
    ram: np.ndarray,
    bounds: Tuple[int, int, int, int] = DEFAULT_CROP_BOUNDS,
    min_count: int = 4,
    suppress_radius: int = 2,
) -> List[Tuple[int, int]]:
    """Detect established crop plots using crop tiles only.

    Resume states with existing crops can have partial watering and degraded
    shapes. This detector prefers local maxima of actual crop-stage tiles, then
    callers can merge in supplemental tilled-only centers if needed.
    """
    x_min, y_min, x_max, y_max = bounds
    seen: Set[Tuple[int, int]] = set()
    centers: List[Tuple[int, int]] = []

    for cy in range(y_min, y_max + 1):
        for cx in range(x_min, x_max + 1):
            if (cx, cy) in seen or not is_crop_tile(get_tile_at(ram, cx, cy)):
                continue

            queue = deque([(cx, cy)])
            seen.add((cx, cy))
            component: List[Tuple[int, int]] = []

            while queue:
                tx, ty = queue.popleft()
                component.append((tx, ty))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = tx + dx, ty + dy
                        if not (x_min <= nx <= x_max and y_min <= ny <= y_max):
                            continue
                        if (nx, ny) in seen or not is_crop_tile(get_tile_at(ram, nx, ny)):
                            continue
                        seen.add((nx, ny))
                        queue.append((nx, ny))

            if len(component) < min_count:
                continue

            xs = [tx for tx, _ in component]
            ys = [ty for _, ty in component]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            if max_x - min_x > 2 or max_y - min_y > 2:
                continue

            center = ((min_x + max_x) // 2, (min_y + max_y) // 2)
            too_close = any(
                max(abs(center[0] - sx), abs(center[1] - sy)) <= suppress_radius
                for sx, sy in centers
            )
            if not too_close:
                centers.append(center)

    return centers


def _water_target_tiles(
    ram: np.ndarray,
    center: Tuple[int, int],
    allow_unknown_tiles: bool,
    skip_tiles: Optional[Set[Tuple[int, int]]] = None,
    include_fresh_tilled: bool = False,
) -> List[Tuple[int, int]]:
    """Return concrete plot cells that still need water."""
    targets: List[Tuple[int, int]] = []
    skip_tiles = skip_tiles or set()
    for ty in range(center[1] - 1, center[1] + 2):
        for tx in range(center[0] - 1, center[0] + 2):
            if (tx, ty) in skip_tiles:
                continue
            tile_id = get_tile_at(ram, tx, ty)
            if tile_can_be_water_target(
                tile_id,
                allow_unknown=allow_unknown_tiles,
                include_fresh_tilled=include_fresh_tilled,
            ):
                targets.append((tx, ty))
    return targets


def _preferred_outward_faces(center: Tuple[int, int], target: Tuple[int, int]) -> List[str]:
    dx = target[0] - center[0]
    dy = target[1] - center[1]
    order: List[str] = []
    if dy < 0:
        order.append("down")
    elif dy > 0:
        order.append("up")
    if dx < 0:
        order.append("right")
    elif dx > 0:
        order.append("left")
    for face in ("up", "down", "left", "right"):
        if face not in order:
            order.append(face)
    return order


def _water_step_variants(
    ram: np.ndarray,
    center: Tuple[int, int],
    target: Tuple[int, int],
    *,
    allow_crop_walkable: bool = False,
) -> List[Tuple[Tuple[int, int], str]]:
    """Return all valid adjacent stand tiles for watering a specific target."""
    plot_set = set(plot_tiles(center, include_center=True))
    face_to_delta = {
        "up": (0, 1),
        "down": (0, -1),
        "left": (1, 0),
        "right": (-1, 0),
    }
    variants: List[Tuple[Tuple[int, int], str]] = []
    for face in _preferred_outward_faces(center, target):
        dx, dy = face_to_delta[face]
        stand = (target[0] + dx, target[1] + dy)
        sx, sy = stand
        if not (0 <= sx < MAP_WIDTH and 0 <= sy < MAP_WIDTH):
            continue

        stand_tid = get_tile_at(ram, sx, sy)
        stand_is_walkable = stand_tid in WALKABLE_TILES or (
            allow_crop_walkable and stand in plot_set
        )
        if stand_is_walkable:
            variants.append((stand, face))
    return variants


def build_water_steps(
    ram: np.ndarray,
    center: Tuple[int, int],
    allow_crop_walkable: bool = False,
    allow_unknown_tiles: bool = False,
    include_fresh_tilled: bool = False,
    start_tile: Optional[Tuple[int, int]] = None,
    skip_tiles: Optional[Set[Tuple[int, int]]] = None,
) -> List[Tuple[Tuple[int, int], Tuple[int, int], str]]:
    """Build per-cell watering targets with explicit adjacent stand tiles.

    For established crops this keeps stands on the notch/perimeter instead of
    trying to route through crop tiles. For fresh post-plant plots, callers can
    allow temporary walkability inside the 3x3.
    """
    targets = _water_target_tiles(
        ram,
        center,
        allow_unknown_tiles=allow_unknown_tiles,
        skip_tiles=skip_tiles,
        include_fresh_tilled=include_fresh_tilled,
    )
    steps: List[Tuple[Tuple[int, int], Tuple[int, int], str]] = []

    for target in targets:
        variants = _water_step_variants(
            ram,
            center,
            target,
            allow_crop_walkable=False,
        )
        if not variants and allow_crop_walkable:
            variants = _water_step_variants(
                ram,
                center,
                target,
                allow_crop_walkable=True,
            )
        if variants:
            stand, face = variants[0]
            steps.append((target, stand, face))

    if start_tile is not None:
        steps.sort(key=lambda step: (tile_dist(start_tile, step[1]), tile_dist(start_tile, step[0])))

    return steps



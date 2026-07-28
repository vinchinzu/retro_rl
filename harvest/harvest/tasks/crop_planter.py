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


import numpy as np

from harvest.core.ram_catalog import field_spec, read_ram_u8, read_ram_u16

from harvest.tasks.farm_clearer import (
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

ADDR_TOOL_BACKPACK = 0x0923

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
WATERED_TILLED = 0x08
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


def carry_pair_items(ram: np.ndarray) -> Set[int]:
    items: Set[int] = set()
    if ADDR_TOOL < len(ram):
        items.add(int(ram[ADDR_TOOL]))
    if ADDR_TOOL_BACKPACK < len(ram):
        items.add(int(ram[ADDR_TOOL_BACKPACK]))
    return items


def watering_can_in_carry_pair(ram: np.ndarray) -> bool:
    return int(Tool.WATERING_CAN) in carry_pair_items(ram)


def seed_item_in_carry_pair(ram: np.ndarray, seed_type: str = "potato") -> bool:
    seed_item = SEED_ITEM.get(seed_type, SEED_ITEM["potato"])
    return seed_item in carry_pair_items(ram)

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


def water_action_sequence(face_dir: str, cooldown: int = 18, face_frames: int = 3) -> List[np.ndarray]:
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
    from harvest.tasks.farm_clearer import WALKABLE_TILES

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
        if tile_needs_watering(get_tile_at(ram, tx, ty)):
            count += 1
    return count


# ── crop detection & watering task ─────────────────────────────────

from dataclasses import dataclass, field
from collections import deque

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

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

# Seed data key names in data.json (for env.data.set_value)
SEED_DATA_KEY: Dict[str, str] = {
    "potato": "potato_seeds",
    "turnip": "turnip_seeds",
    "corn": "corn_seeds",
    "tomato": "tomato_seeds",
}

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
    # Planned centers that failed hoe/path — avoid infinite redetect loops.
    _rejected_plan_centers: Set[Tuple[int, int]] = field(default_factory=set, init=False)

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
        self._allow_unknown_water_tiles = False
        self._allow_crop_walkable = False
        self._refill_pond_tile = None
        self._refill_pond_face = None
        self._refill_level_before = 0
        self._refill_search_level = -1
        self._bad_refill_tiles = set()
        self._refill_exhausted = False
        self._pre_water_level = -1
        self._last_water_level_before = -1
        self._last_water_tile_before = -1
        self._water_verify_retries = 0
        self._resume_water_index = 0
        self.planted_count = 0
        self.watered_count = 0
        self.skipped_water = 0
        self.refill_count = 0
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
        self._pre_water_level = -1
        self._last_water_level_before = -1
        self._last_water_tile_before = -1
        self._water_verify_retries = 0
        self._resume_water_index = 0
        self.planted_count = 0
        self.watered_count = 0
        self.skipped_water = 0
        self.refill_count = 0
        self._rejected_plan_centers = set()
        print(f"[CROP] Hot-swap resume: re-scan crops/refill state can={self._water_level(world.ram)}")

    def can_start(self, world: WorldState) -> bool:
        return True

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

    def _plan_bounds_near_player(self, start: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Clamp planning to a viewport-reachable neighborhood around the player.

        Full-farm plans often pick distant centers that BFS cannot reach through
        stale off-screen tiles. Keep new plots within ~12 tiles of the player
        and inside the task bounds.
        """
        x_min, y_min, x_max, y_max = self.bounds
        px, py = start
        radius = 12
        return (
            max(x_min, px - radius),
            max(y_min, py - radius),
            min(x_max, px + radius),
            min(y_max, py + radius),
        )

    def _plan_new_plot_centers(self, ram: np.ndarray) -> List[Tuple[int, int]]:
        """Use crop_planner to place new 3x3 plots on tillable soil."""
        try:
            from harvest.planner.crop_planner import CropPlanningConfig, plan_crop_field
            from harvest.planner.day_plan_status import read_world_date
        except Exception as exc:
            print(f"[CROP] Crop planner unavailable: {exc}")
            return []

        season, day = read_world_date(ram)
        start = self._navigator.current_tile
        local_bounds = self._plan_bounds_near_player(start)
        config = CropPlanningConfig(
            season=int(season),
            day=int(day),
            seed_type=self.seed_type,
            max_seed_bags=1,
            bounds=local_bounds,
            start_tile=start,
            # Strongly prefer nearby plots over slightly higher remote scores.
            route_weight=40,
        )
        plan = plan_crop_field(ram, config)
        centers = [
            plot.center
            for plot in plan.plots
            if plot.center not in self._rejected_plan_centers
        ]
        if not centers and local_bounds != self.bounds:
            # Fall back to full bounds once, still skipping rejected centers.
            config = CropPlanningConfig(
                season=int(season),
                day=int(day),
                seed_type=self.seed_type,
                max_seed_bags=3,
                bounds=self.bounds,
                start_tile=start,
                route_weight=40,
            )
            plan = plan_crop_field(ram, config)
            centers = [
                plot.center
                for plot in plan.plots
                if plot.center not in self._rejected_plan_centers
            ][:1]
        if centers:
            print(
                f"[CROP] Planned {len(centers)} new {plan.crop_name} plot(s) "
                f"layout={plan.layout_name} bounds={local_bounds}: {centers}"
            )
        else:
            print("[CROP] Crop planner found no placeable plots")
        return centers

    def _handle_detect(self, ram: np.ndarray) -> Optional[TaskResult]:
        """Scan for crop plots."""
        resume_plots = detect_crop_resume_plots(ram, self.bounds)
        if resume_plots:
            supplemental = detect_plots(ram, self.bounds)
            self._plots = _merge_plot_centers(resume_plots, supplemental)
        else:
            self._plots = detect_plots(ram, self.bounds)
        if not self._plots:
            # Virgin soil: plan + hoe + plant instead of silently succeeding.
            can_plant = self._has_plantable_seed_stock(ram)
            if self._pass_number == 1 and can_plant:
                planned = self._plan_new_plot_centers(ram)
                if planned:
                    self._plots = planned
                else:
                    print("[CROP] No plots detected and no plantable plan")
                    return TaskResult(status=TaskStatus.SUCCESS, reason="no plots detected")
            elif self._pass_number == 1:
                print("[CROP] No plots detected (no plantable seed stock)")
                return TaskResult(status=TaskStatus.SUCCESS, reason="no plots detected")
            else:
                self._state = "done"
                return None
        current_tile = self._navigator.current_tile
        self._plots.sort(key=lambda center: (tile_dist(current_tile, center), center[1], center[0]))
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
        crop_tiles = _count_crop_tiles(ram, center[0], center[1])
        # Seed bag plants a 3x3 from the center notch. Plant once enough ring
        # tiles are hoed (full 8 is ideal; partial still uses the bag).
        if crop_tiles == 0 and tilled >= 4:
            self._plot_phase = "plant"
            self._target_tile = center
            self._approach_tile = center  # stand ON center to plant
            self._face_direction = "down"
            self._set_crop_walkable()
            self._state = "navigate"
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
            # Skip to water phase
            self._begin_water_phase(ram, allow_unknown_tiles=False)

    def _begin_hoe_phase(self, ram: np.ndarray) -> None:
        """Hoe untilled ring tiles for the current planned plot center."""
        center = self._plots[self._plot_index]
        cx, cy = center
        self._plot_phase = "hoe"
        self._water_steps = []
        self._water_index = 0
        for target, stand, face in hoe_plan(center):
            tid = get_tile_at(ram, target[0], target[1])
            if tid in TILLABLE_TILES or tid == DRIED_TILLED or tid == UNTILLED:
                self._water_steps.append((target, stand, face))
            elif tid not in {FRESH_TILLED, WATERED_TILLED}:
                # Unknown/blocked — still try if soil-like low IDs.
                if tid in {0x00, 0x01, 0x02}:
                    self._water_steps.append((target, stand, face))
        print(
            f"[CROP] Plot {self._plot_index + 1}/{len(self._plots)} "
            f"center=({cx},{cy}) phase=HOE steps={len(self._water_steps)}"
        )
        if not self._water_steps:
            # Nothing to hoe; try plant or water.
            tilled = count_tilled(ram, center)
            if tilled >= 4:
                self._plot_phase = "plant"
                self._target_tile = center
                self._approach_tile = center
                self._state = "navigate"
                self._navigator.path = []
                self._steps_on_target = 0
                print(f"[CROP] HOE skipped; planting with tilled={tilled}")
                return
            print(f"[CROP] HOE found no tillable tiles at ({cx},{cy}); skipping plot")
            self._rejected_plan_centers.add(center)
            self._advance_plot(ram)
            return
        target, stand, face = self._water_steps[0]
        self._target_tile = target
        self._approach_tile = stand
        self._face_direction = face
        self._clear_crop_walkable()
        self._state = "navigate"
        self._navigator.path = []
        self._steps_on_target = 0

    def _advance_hoe_step(self, ram: np.ndarray) -> None:
        self._water_index += 1
        self._steps_on_target = 0
        self._navigator.path = []
        if self._water_index >= len(self._water_steps):
            center = self._plots[self._plot_index]
            tilled = count_tilled(ram, center)
            print(f"[CROP] HOE complete plot {self._plot_index + 1} tilled={tilled}")
            if tilled < 4:
                # No reachable till work — reject this planned center and move on.
                self._rejected_plan_centers.add(center)
                print(f"[CROP] Rejecting planned center {center} after failed hoe")
                self._advance_plot(ram)
                return
            self._plot_phase = "plant"
            self._target_tile = center
            self._approach_tile = center
            self._face_direction = "down"
            self._state = "navigate"
            self._navigator.path = []
            return
        target, stand, face = self._water_steps[self._water_index]
        self._target_tile = target
        self._approach_tile = stand
        self._face_direction = face
        self._state = "navigate"

    def _begin_water_phase(self, ram: np.ndarray, allow_unknown_tiles: bool = False):
        """Set up per-tile watering for current plot using WATER_PLAN_CENTER."""
        if self._plot_index >= len(self._plots):
            return
        center = self._plots[self._plot_index]
        cx, cy = center
        if is_rainy_weather(ram):
            print(
                f"[CROP] Plot {self._plot_index + 1}/{len(self._plots)} "
                f"center=({cx},{cy}) rain; skipping manual watering"
            )
            self._advance_plot(ram)
            return

        self._plot_phase = "water"
        self._plot_watered = 0
        self._plot_skipped = 0
        self._allow_unknown_water_tiles = allow_unknown_tiles
        self._allow_crop_walkable = allow_unknown_tiles
        self._water_verify_retries = 0
        self._last_water_level_before = -1
        self._last_water_tile_before = -1
        # Build concrete per-tile watering steps. Resume states keep stands on
        # the notch/perimeter when possible, but allow inside-plot recovery.
        self._water_steps = build_water_steps(
            ram,
            center,
            allow_crop_walkable=self._allow_crop_walkable,
            allow_unknown_tiles=allow_unknown_tiles,
            include_fresh_tilled=allow_unknown_tiles,
            start_tile=self._navigator.current_tile,
            skip_tiles=set(self.skip_water_tiles),
        )
        self._water_index = 0

        water_lvl = self._water_level(ram)
        self._pre_water_level = water_lvl  # plot-level: track starting level
        mode = "unknown-ok" if allow_unknown_tiles else "dry-only"
        print(f"[CROP] Plot {self._plot_index + 1}/{len(self._plots)} center=({cx},{cy}) phase=WATER can={water_lvl} mode={mode} steps={len(self._water_steps)}")

        if not self._water_steps:
            print(f"[CROP] Plot {self._plot_index + 1}/{len(self._plots)} has no reachable water targets")
            self._advance_plot(ram)
            return

        self._reorder_remaining_water_steps(ram)
        # Navigate to first stand position
        target, stand, face = self._water_steps[0]
        self._target_tile = target
        self._approach_tile = stand
        self._face_direction = face
        self._set_water_walkable()
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

    def _current_plot_tiles(self) -> Set[Tuple[int, int]]:
        if self._plot_index >= len(self._plots):
            return set()
        return set(plot_tiles(self._plots[self._plot_index], include_center=True))

    def _set_water_walkable(self) -> None:
        """Allow only the current crop-tile stand, not the full 3x3 plot."""
        self._pathfinder.extra_walkable.clear()
        if self._plot_phase != "water" or self._approach_tile is None:
            return
        if self._allow_crop_walkable and self._approach_tile in self._current_plot_tiles():
            self._pathfinder.extra_walkable.add(self._approach_tile)

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
        self._water_verify_retries = 0
        self._last_water_level_before = -1
        self._last_water_tile_before = -1
        self._water_index += 1
        if self._water_index >= len(self._water_steps):
            # All tiles attempted — plot-level verification
            center = self._plots[self._plot_index]
            cx, cy = center
            lvl = self._water_level(ram)
            water_used = max(0, self._pre_water_level - lvl) if self._pre_water_level >= 0 else 0
            actual_watered = self._plot_watered
            actual_skipped = self._plot_skipped

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
            self._reorder_remaining_water_steps(ram)
            # Navigate to next stand position
            target, stand, face = self._water_steps[self._water_index]
            self._target_tile = target
            self._approach_tile = stand
            self._face_direction = face
            self._set_water_walkable()
            self._state = "navigate"
            self._navigator.path = []
            self._steps_on_target = 0

    def _best_water_variant(
        self,
        ram: np.ndarray,
        target: Tuple[int, int],
        current_tile: Tuple[int, int],
    ) -> Optional[Tuple[Tuple[int, int], str, int]]:
        """Pick the best currently-reachable adjacent stand for a water target."""
        if self._plot_index >= len(self._plots):
            return None

        center = self._plots[self._plot_index]
        variants = _water_step_variants(
            ram,
            center,
            target,
            allow_crop_walkable=self._allow_crop_walkable,
        )
        plot_set = self._current_plot_tiles()
        best: Optional[Tuple[Tuple[int, int], str, int]] = None
        for stand, face in variants:
            walkable_override = {stand} if self._allow_crop_walkable and stand in plot_set else None
            path = self._pathfinder.find_path(ram, current_tile, stand, walkable_override=walkable_override)
            if path is None:
                continue
            # Strongly prefer perimeter/notch stands over standing on crops.
            score = tile_dist(current_tile, stand)
            if stand in plot_set:
                score += 32
            candidate = (stand, face, score)
            if best is None or candidate[2] < best[2]:
                best = candidate
        return best

    def _retarget_current_water_step(self, ram: np.ndarray) -> bool:
        if self._plot_phase != "water" or self._target_tile is None:
            return False
        best = self._best_water_variant(ram, self._target_tile, self._navigator.current_tile)
        if best is None:
            return False
        stand, face, _score = best
        changed = stand != self._approach_tile or face != self._face_direction
        self._approach_tile = stand
        self._face_direction = face
        if self._water_index < len(self._water_steps):
            self._water_steps[self._water_index] = (self._target_tile, stand, face)
        self._set_water_walkable()
        return changed

    def _reorder_remaining_water_steps(self, ram: np.ndarray) -> bool:
        if self._plot_phase != "water" or self._water_index >= len(self._water_steps):
            return False

        current_tile = self._navigator.current_tile
        prefix = self._water_steps[:self._water_index]
        remaining_scored = []
        for offset, (target, _stand, _face) in enumerate(self._water_steps[self._water_index:]):
            best = self._best_water_variant(ram, target, current_tile)
            if best is None:
                continue
            stand, face, score = best
            remaining_scored.append(((score, offset), (target, stand, face)))

        if not remaining_scored:
            return False

        reordered = [step for _score, step in sorted(remaining_scored, key=lambda item: item[0])]
        changed = reordered != self._water_steps[self._water_index:]
        self._water_steps = prefix + reordered
        self._target_tile, self._approach_tile, self._face_direction = self._water_steps[self._water_index]
        return changed

    def _reprioritize_water_step(self, ram: np.ndarray, *, reason: str) -> bool:
        if not self._reorder_remaining_water_steps(ram):
            return False
        print(
            f"[CROP] REORDER water tiles ({reason}) "
            f"target={self._target_tile} stand={self._approach_tile} face={self._face_direction}"
        )
        self._state = "navigate"
        self._navigator.path = []
        self._steps_on_target = 0
        return True

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

        if self._plot_phase == "water":
            if not self._navigator.path and self._retarget_current_water_step(ram):
                self._navigator.path = []
                self._navigator.stasis = 0
            self._set_water_walkable()

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
                    if self._reprioritize_water_step(ram, reason="stuck nav"):
                        return None
                    self.skipped_water += 1
                    self._plot_skipped += 1
                    print(f"[CROP] SKIP water tile {self._water_index + 1}/{len(self._water_steps)} (stuck nav) target={self._target_tile}")
                    self._advance_water_step(ram)
                elif self._plot_phase == "hoe":
                    print(
                        f"[CROP] SKIP hoe tile {self._water_index + 1}/{len(self._water_steps)} "
                        f"(stuck nav) target={self._target_tile}"
                    )
                    self._advance_hoe_step(ram)
                elif self._plot_phase == "plant":
                    center = self._plots[self._plot_index] if self._plot_index < len(self._plots) else None
                    tilled = count_tilled(ram, center) if center else 0
                    if center is not None and tilled >= 4:
                        # Close enough: attempt plant even if exact center nav struggled.
                        print(f"[CROP] Plant nav stuck at center {center}; forcing plant attempt tilled={tilled}")
                        self._state = "act"
                    else:
                        if center is not None:
                            self._rejected_plan_centers.add(center)
                        print(f"[CROP] Plant nav stuck; skipping plot {center}")
                        self._advance_plot(ram)
                elif self._plot_phase == "refill":
                    print("[CROP] Can't reach pond, skipping refill")
                    self._refill_exhausted = True
                    self._plot_phase = "water"
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
                    if self._reprioritize_water_step(ram, reason="no path"):
                        return None
                    self.skipped_water += 1
                    self._plot_skipped += 1
                    print(f"[CROP] SKIP water tile {self._water_index + 1}/{len(self._water_steps)} (no path) target={self._target_tile}")
                    self._advance_water_step(ram)
                elif self._plot_phase == "hoe":
                    print(
                        f"[CROP] SKIP hoe tile {self._water_index + 1}/{len(self._water_steps)} "
                        f"(no path) target={self._target_tile}"
                    )
                    self._advance_hoe_step(ram)
                elif self._plot_phase == "plant":
                    center = self._plots[self._plot_index] if self._plot_index < len(self._plots) else None
                    tilled = count_tilled(ram, center) if center else 0
                    if center is not None and tilled >= 4 and tile_dist(self._navigator.current_tile, center) <= 1:
                        print(f"[CROP] Plant path missing but adjacent to {center}; forcing plant")
                        self._approach_tile = self._navigator.current_tile
                        self._state = "act"
                    elif center is not None and tilled >= 4:
                        print(f"[CROP] No path to plant center {center}; retrying with crop walkable")
                        self._set_crop_walkable()
                        self._navigator.path = []
                        self._steps_on_target = 0
                    else:
                        if center is not None:
                            self._rejected_plan_centers.add(center)
                        print(f"[CROP] No path to plant center {center}; skipping plot")
                        self._advance_plot(ram)
                elif self._plot_phase == "refill":
                    print("[CROP] No path to pond, skipping refill")
                    self._refill_exhausted = True
                    self._plot_phase = "water"
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
                    self._state = "navigate"
                    self._navigator.path = []
                else:
                    self._state = "detect"
                return None

        action = self._navigator.follow_path(ram)
        if action is not None:
            if self._plot_phase == "water":
                action = action.copy()
                if (
                    self._navigator.current_tile in self._current_plot_tiles()
                    or tile_dist(self._navigator.current_tile, self._approach_tile) <= 1
                ):
                    action[0] = 0  # slow down only once we're threading the plot edge
            self._action_queue.append(action)
        return None

    def _handle_center(self, ram: np.ndarray) -> Optional[TaskResult]:
        if self._approach_tile is None:
            self._state = "detect"
            return None
        tol = 1 if self._plot_phase in ("plant", "water", "hoe") else 2
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
        if self._plot_phase in ("plant", "water", "hoe"):
            # Must be ON approach tile for plant/water/hoe
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
        tol = 1 if self._plot_phase in ("plant", "water", "hoe") else 2
        center_action = self._navigator.center_on_tile(self._approach_tile, tolerance=tol)
        if center_action is not None:
            self._action_queue.append(center_action)
            return None

        if self._plot_phase == "plant":
            return self._act_plant(ram)
        elif self._plot_phase == "hoe":
            return self._act_hoe(ram)
        elif self._plot_phase == "water":
            return self._act_water(ram)
        elif self._plot_phase == "refill":
            return self._act_refill(ram)
        return None

    def _act_hoe(self, ram: np.ndarray) -> Optional[TaskResult]:
        """Hoe one untilled ring tile for the current planned plot."""
        if self._tool_mgr.current != int(Tool.HOE):
            self._tool_mgr.start_search()
            self._state = "tool_switch"
            return None
        face = self._face_direction or "down"
        target = self._target_tile
        tid = get_tile_at(ram, target[0], target[1]) if target else 0xFF
        print(
            f"[CROP] HOE tile {self._water_index + 1}/{len(self._water_steps)} "
            f"target={target} face={face} tid=0x{tid:02X}"
        )
        self._action_queue.extend(hoe_action_sequence(face))
        self._state = "verify"
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
            if target in self.skip_water_tiles or not tile_can_be_water_target(
                tid,
                allow_unknown=self._allow_unknown_water_tiles,
                include_fresh_tilled=self._allow_unknown_water_tiles,
            ):
                print(f"[CROP] SKIP water tile {self._water_index + 1}/{len(self._water_steps)} target={target} "
                      f"tid=0x{tid:02X} (not a dry target)")
                self._advance_water_step(ram)
                return None

        water_lvl = self._water_level(ram)

        # Count only waterable remaining tiles for refill check
        waterable_remaining = 0
        for i in range(self._water_index, len(self._water_steps)):
            t = self._water_steps[i][0]
            if t in self.skip_water_tiles:
                continue
            if tile_can_be_water_target(
                get_tile_at(ram, t[0], t[1]),
                allow_unknown=self._allow_unknown_water_tiles,
                include_fresh_tilled=self._allow_unknown_water_tiles,
            ):
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

        self._last_water_level_before = water_lvl
        self._last_water_tile_before = tid
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

        if self._plot_phase == "hoe":
            self._advance_hoe_step(ram)
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
            self._begin_water_phase(ram, allow_unknown_tiles=True)

        elif self._plot_phase == "water":
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
                    self._state = "center"

        elif self._plot_phase == "refill":
            lvl_after = self._water_level(ram)
            if lvl_after > self._refill_level_before:
                # Refill succeeded — navigate back to current water step
                self.refill_count += 1
                print(f"[CROP] REFILL OK can={lvl_after} (was {self._refill_level_before}) refills={self.refill_count}")
                self._pre_water_level = lvl_after  # reset for plot-level verification
                self._refill_search_level = -1  # reset search tracking
                self._plot_phase = "water"
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
        elif self._plot_phase == "hoe":
            wanted = int(Tool.HOE)
        else:
            wanted = int(Tool.WATERING_CAN)

        self._tool_mgr.update(ram)
        current = self._tool_mgr.current

        if current == wanted:
            if self.debug:
                print(f"[CROP] Found tool 0x{wanted:02X}")
            self._state = "center"
            return None

        self._tool_mgr.record()

        if self._tool_mgr.cycle_complete():
            if self._plot_phase == "plant":
                center = self._plots[self._plot_index] if self._plot_index < len(self._plots) else None
                print(f"[CROP] Seed tool 0x{wanted:02X} not found, skipping plant plot at {center}")
                self._advance_plot(ram)
                return None
            if self._plot_phase == "hoe":
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

        if self._total_steps == 1 and is_rainy_weather(world.ram) and not seed_item_in_carry_pair(world.ram, self.seed_type):
            wanted = SEED_ITEM.get(self.seed_type, SEED_ITEM["potato"])
            # Rain waters existing crops; without seeds there is no plant work either.
            # Still run detect in case established plots need nothing — but if no
            # seeds and rain, short-circuit so day plan can finish.
            print(f"[CROP] Rain and seed tool 0x{wanted:02X} not in carry pair; no crop work needed")
            return TaskResult(status=TaskStatus.SUCCESS, reason=f"rain; seed tool 0x{wanted:02X} not in carry pair")

        # Do not fail early when the watering can is out of the 2-slot carry pair.
        # Day plan often leaves seeds in-hand after ENSURE_CROP_SEEDS; we still
        # need to hoe/plant, then cycle to the can for watering.

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
                if self._reprioritize_water_step(world.ram, reason="timeout"):
                    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
                self.skipped_water += 1
                self._plot_skipped += 1
                print(f"[CROP] SKIP water tile {self._water_index + 1}/{len(self._water_steps)} (timeout) target={self._target_tile}")
                self._advance_water_step(world.ram)
            elif self._plot_phase == "hoe":
                print(
                    f"[CROP] SKIP hoe tile {self._water_index + 1}/{len(self._water_steps)} "
                    f"(timeout) target={self._target_tile}"
                )
                self._advance_hoe_step(world.ram)
            elif self._plot_phase == "refill":
                print("[CROP] Refill timed out, marking bad")
                if self._refill_pond_tile:
                    self._bad_refill_tiles.add(self._refill_pond_tile)
                # Navigate back to current water step
                self._plot_phase = "water"
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
                self._state = "navigate"
                self._navigator.path = []
            elif self._plot_phase == "plant":
                center = self._plots[self._plot_index] if self._plot_index < len(self._plots) else None
                if center is not None:
                    self._rejected_plan_centers.add(center)
                print(f"[CROP] Plant timeout at {center}; skipping plot")
                self._advance_plot(world.ram)
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

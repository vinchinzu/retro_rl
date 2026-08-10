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

from harvest.core.ram_catalog import field_spec, read_ram_u8, read_ram_u16, read_ram_value

from harvest.core.carry import (
    ADDR_TOOL_BACKPACK,
    SEED_ITEM,
    carry_pair_items,
    seed_in_carry_pair as seed_item_in_carry_pair,
    watering_can_in_carry_pair,
)
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
    VIEWPORT_HOP_TILES,
    WALKABLE_TILES,
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

# Actual water tiles for refilling — excludes 0xA6 (pond border/decorative).
# Keep F1/F8/etc. for search (property table may map some); sort prefers
# REFILL_PREFERRED_WATER_TILES first.
REFILL_WATER_TILES = frozenset({
    0xF0, 0xF1, 0xF2,
    0xF7, 0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD,
})

# REFILL_PREFERRED_WATER_TILES / REFILL_NONFILL_WATER_TILES live in water_refill
# (CheckToolSuccess F0/F9–FD fill; F1/F2/F7/F8 never fill). Re-exported above.

# Shipping-bin F2 pocket (x~8-9,y~29-30) does NOT refill the can. Pre-blacklist
# stand tiles in this rectangle so path search never prefers them.
# Bounds: (x_min, y_min, x_max, y_max) inclusive stand coordinates.
BAD_REFILL_STAND_BOUNDS = (6, 27, 12, 33)
# Score bands for refill stand preference (lower = better; secondary to
# preferred-water-id rank in refill_edge_sort_key).
# Main F0 pond (mid-farm) is the verified fill stand and is closest to fields
# once the y=31 fence corridor is open — rank it first among preferred water.
REFILL_BAND_POND = 0    # main F0 pond stands (y 28–36, x 28–36)
REFILL_BAND_SOUTH = 1   # y >= 45: south stream FC / SE FD
REFILL_BAND_NORTH = 2   # y <= 25: north spur F9 / east FA (preferred) or F1/F8 (non-fill)
REFILL_BAND_MID = 3     # other mid-farm (east FB, etc.)
REFILL_BAND_BAD = 4     # known-bad shipping pocket (should be filtered out)
# Main pond stand bbox for band classification (stand tiles, not water cells).
MAIN_POND_STAND_BOUNDS = (28, 28, 36, 36)

# Face direction → delta into the adjacent water cell.
_REFILL_FACE_DELTA = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


# carry_pair_items / watering_can_in_carry_pair / seed_item_in_carry_pair
# live in harvest.core.carry (re-exported above).

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
    Cooldown must cover ToolAnimationWateringCan writing can=0x14 after Y.
    """
    actions: List[np.ndarray] = []
    actions.extend([make_action(**{face_dir: True}) for _ in range(face_frames)])
    settle = max(1, 8 - face_frames)
    actions.extend([make_action() for _ in range(settle)])
    # 45f was tight for animation write; 75f leaves room for can=0x14.
    actions.extend(use_tool(frames=15, cooldown=75))
    return actions


def is_bad_refill_stand(tile: Tuple[int, int]) -> bool:
    """True if stand is in the shipping-bin F2 pocket that never refills."""
    x, y = tile
    x0, y0, x1, y1 = BAD_REFILL_STAND_BOUNDS
    return x0 <= x <= x1 and y0 <= y <= y1


def is_main_pond_stand(tile: Tuple[int, int]) -> bool:
    """True if stand is on the main F0 pond lip (verified fill stands)."""
    x, y = tile
    x0, y0, x1, y1 = MAIN_POND_STAND_BOUNDS
    return x0 <= x <= x1 and y0 <= y <= y1


def refill_stand_band(tile: Tuple[int, int]) -> int:
    """Preference band for a refill stand (lower is better)."""
    if is_bad_refill_stand(tile):
        return REFILL_BAND_BAD
    if is_main_pond_stand(tile):
        return REFILL_BAND_POND
    y = tile[1]
    if y >= 45:
        return REFILL_BAND_SOUTH
    if y <= 25:
        return REFILL_BAND_NORTH
    return REFILL_BAND_MID


def edge_water_tile_id(
    ram: np.ndarray,
    tile: Tuple[int, int],
    face: str,
) -> int:
    """Tilemap id of the water cell a stand faces, or -1 if out of bounds."""
    dx, dy = _REFILL_FACE_DELTA.get(face, (0, 0))
    nx, ny = tile[0] + dx, tile[1] + dy
    if 0 <= nx < MAP_WIDTH and 0 <= ny < MAP_WIDTH:
        return int(get_tile_at(ram, nx, ny))
    return -1


def refill_edge_sort_key(
    edge: Tuple[Tuple[int, int], str],
    player: Tuple[int, int],
    water_tid: int = -1,
) -> Tuple[int, int, int]:
    """Sort key: preferred CheckToolSuccess water → band → Manhattan dist.

    Preferred water ids (F0/F9–FD) rank before non-fill. Band is secondary
    (main pond → south → north → mid → bad). Distance breaks ties.
    """
    tile, _face = edge
    preferred = 0 if water_tid in REFILL_PREFERRED_WATER_TILES else 1
    px, py = player
    dist = abs(tile[0] - px) + abs(tile[1] - py)
    return (preferred, refill_stand_band(tile), dist)


def pond_access_blocking_fences(
    ram: np.ndarray,
    *,
    fence_row: Optional[int] = None,
    x_range: Optional[Tuple[int, int]] = None,
) -> List[Tuple[int, int]]:
    """Fence tiles on the y=31 wall that cut west field off from the main pond.

    ROM-mapped early-spring layout: solid 0x05 on y=31, x=11–29. Clearing any
    one opens full BFS from the west plant pocket to F0 pond stands.
    """
    try:
        from harvest.maps.map_config import (
            FARM_POND_ACCESS_FENCE_ROW,
            FARM_POND_ACCESS_FENCE_X_RANGE,
        )
        from harvest.core.tile_catalog import FENCE
    except Exception:
        FARM_POND_ACCESS_FENCE_ROW = 31
        FARM_POND_ACCESS_FENCE_X_RANGE = (11, 29)
        FENCE = 0x05

    row = FARM_POND_ACCESS_FENCE_ROW if fence_row is None else fence_row
    x0, x1 = FARM_POND_ACCESS_FENCE_X_RANGE if x_range is None else x_range
    found: List[Tuple[int, int]] = []
    for tx in range(x0, x1 + 1):
        if get_tile_at(ram, tx, row) == FENCE:
            found.append((tx, row))
    return found


def find_pond_edges(
    ram: np.ndarray,
    bounds: Tuple[int, int, int, int] = (3, 3, 62, 60),
    water_tiles: Optional[frozenset] = None,
    *,
    exclude_bad_stands: bool = False,
) -> List[Tuple[Tuple[int, int], str]]:
    """Find walkable tiles adjacent to water, suitable for watering can refill.

    Returns list of (tile, face_dir) where tile is walkable and face_dir
    points toward adjacent water. Use edge_water_tile_id(ram, tile, face)
    for the adjacent water tilemap id (preferred-vs-fallback sort).

    water_tiles: set of tile IDs to consider as water.  Defaults to WATER_TILES
        (includes A6 pond border).  Pass REFILL_WATER_TILES for actual water only.
    exclude_bad_stands: drop shipping-bin F2 pocket stands (never refill).
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
            if exclude_bad_stands and is_bad_refill_stand((tx, ty)):
                continue
            tid = get_tile_at(ram, tx, ty)
            if tid not in WALKABLE_TILES:
                continue
            # Prefer a face toward CheckToolSuccess-valid water when several
            # water neighbors exist (e.g. corner stand next to F8 and FC).
            best_face: Optional[str] = None
            best_rank = 2  # 0=preferred, 1=other refill water
            for dx, dy, face in directions:
                nx, ny = tx + dx, ty + dy
                if 0 <= nx < MAP_WIDTH and 0 <= ny < MAP_WIDTH:
                    ntid = get_tile_at(ram, nx, ny)
                    if ntid in water_tiles:
                        rank = 0 if ntid in REFILL_PREFERRED_WATER_TILES else 1
                        if best_face is None or rank < best_rank:
                            best_face = face
                            best_rank = rank
                            if rank == 0:
                                break
            if best_face is not None:
                results.append(((tx, ty), best_face))
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


# ── CropWaterTask ──────────────────────────────────────────────────

# CropWaterTask work modes — day plan plant pass vs water pass only hold two
# carry slots, so establish (hoe+seed) and water (can) must not share a run.
WORK_MODE_FULL = "full"          # hoe/plant then water (legacy single pass)
WORK_MODE_ESTABLISH = "establish"  # hoe + plant only; no watering can required
WORK_MODE_WATER = "water"        # water existing crops only; no new plant plan


@dataclass
class CropWaterTask(Task):
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
    _fence_subtask: Optional[object] = field(default=None, init=False)
    _fence_open_attempts: int = field(default=0, init=False)
    _refill_nav_failures: int = field(default=0, init=False)
    _refill_multihop: bool = field(default=False, init=False)
    _refill_best_dist: int = field(default=999, init=False)
    _pending_multihop_after_drop: bool = field(default=False, init=False)

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
        if mode not in {WORK_MODE_FULL, WORK_MODE_ESTABLISH, WORK_MODE_WATER}:
            mode = WORK_MODE_FULL
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
        self._refill_nav_failures = 0
        self._refill_multihop = False
        self._refill_best_dist = 999
        self._pending_multihop_after_drop = False
        self._pending_gap_reseat = False
        self._pending_gap_charge = False
        self._pending_south_lip_charge = False
        self._east_south_charges = 0
        self._water_north_returns = 0
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
        self._refill_nav_failures = 0
        self._refill_multihop = False
        self._refill_best_dist = 999
        self._pending_multihop_after_drop = False
        self._pending_gap_reseat = False
        self._pending_gap_charge = False
        self._pending_south_lip_charge = False
        self._east_south_charges = 0
        self._water_north_returns = 0
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

    def _plan_new_plot_centers(self, ram: np.ndarray) -> List[Tuple[int, int]]:
        """Use crop_planner to place new 3x3 plots on tillable soil.

        Prefer the early-spring field anchor (crop_planner.DEFAULT_START_TILE)
        so we do not plant near shipping/south stream when NAV lands there.
        Fall back to player-local then full-farm bounds.
        """
        try:
            from harvest.planner.crop_planner import (
                DEFAULT_START_TILE,
                CropPlanningConfig,
                plan_crop_field,
            )
            from harvest.planner.day_plan_status import read_world_date
        except Exception as exc:
            print(f"[CROP] Crop planner unavailable: {exc}")
            return []

        season, day = read_world_date(ram)
        start = self._navigator.current_tile
        preferred = DEFAULT_START_TILE
        # Prefer player-local first so BFS can reach hoe stands after NAV_CROP.
        # Preferred-field / full-farm plans often pick east of the x=32 fence
        # (e.g. 35,27) which is unreachable from the early-spring west pocket.
        attempts: List[Tuple[str, Tuple[int, int, int, int], int]] = [
            ("player_local", self._plan_bounds_near_player(start), 1),
            ("preferred_field", self._plan_bounds_around(preferred, radius=14), 1),
            ("full_farm", self.bounds, 1),
        ]
        plan = None
        centers: List[Tuple[int, int]] = []
        used_label = ""
        used_bounds = attempts[0][1]
        for label, bounds, max_bags in attempts:
            config = CropPlanningConfig(
                season=int(season),
                day=int(day),
                seed_type=self.seed_type,
                max_seed_bags=max_bags,
                bounds=bounds,
                start_tile=start,
                # Strongly prefer nearby plots over slightly higher remote scores.
                route_weight=40,
            )
            plan = plan_crop_field(ram, config)
            centers = [
                plot.center
                for plot in plan.plots
                if plot.center not in self._rejected_plan_centers
            ][:1]
            if centers:
                used_label = label
                used_bounds = bounds
                break
        # Planner access checks are strict (watering stands) and full-farm
        # scores often pick east/south of the early-spring fence pocket
        # (unreachable via viewport BFS). Prefer a nearby tillable 3x3 the hoe
        # can actually reach.
        fallback = self._fallback_local_till_center(ram, start)
        if fallback is not None:
            if not centers:
                print(
                    f"[CROP] Planner empty; fallback till center {fallback} "
                    f"near player {start}"
                )
                return [fallback]
            planned = centers[0]
            planned_dist = abs(planned[0] - start[0]) + abs(planned[1] - start[1])
            fallback_dist = abs(fallback[0] - start[0]) + abs(fallback[1] - start[1])
            if planned_dist > 12 and fallback_dist + 4 < planned_dist:
                print(
                    f"[CROP] Prefer fallback till {fallback} (dist={fallback_dist}) "
                    f"over planner {planned} (dist={planned_dist}, zone={used_label})"
                )
                return [fallback]
        elif centers:
            # No nearby till fallback: drop unreachable remote planner centers.
            planned = centers[0]
            planned_dist = abs(planned[0] - start[0]) + abs(planned[1] - start[1])
            if planned_dist > 12:
                print(
                    f"[CROP] Drop remote planner center {planned} "
                    f"(dist={planned_dist}); no local fallback"
                )
                centers = []
        if centers and plan is not None:
            print(
                f"[CROP] Planned {len(centers)} new {plan.crop_name} plot(s) "
                f"layout={plan.layout_name} zone={used_label} bounds={used_bounds}: "
                f"{centers}"
            )
        else:
            print("[CROP] Crop planner found no placeable plots")
        return centers

    def _fallback_local_till_center(
        self,
        ram: np.ndarray,
        start: Tuple[int, int],
    ) -> Optional[Tuple[int, int]]:
        """Pick a nearby 3x3 of untilled soil when the formal planner finds none.

        Early spring west pocket has open dirt the planner rejects (missing
        watering-access stands). Hoe+plant still works if we stand on the
        center notch. Only accept centers reachable via a short hop path so we
        do not plant south/east of the live-map fence pocket.
        """
        px, py = start
        best: Optional[Tuple[int, int]] = None
        best_key: Optional[Tuple[int, int, int]] = None
        for cy in range(max(2, py - 8), min(62, py + 9)):
            for cx in range(max(2, px - 8), min(62, px + 9)):
                center = (cx, cy)
                if center in self._rejected_plan_centers:
                    continue
                tillable = 0
                hard_block = 0
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        tx, ty = cx + dx, cy + dy
                        tid = get_tile_at(ram, tx, ty)
                        if tid in TILLABLE_TILES or tid in {
                            0x00,
                            0x01,
                            0x02,
                            FRESH_TILLED,
                            WATERED_TILLED,
                        }:
                            tillable += 1
                        elif tid in WALKABLE_TILES:
                            # path tile inside plot — can still hoe around
                            tillable += 1
                        else:
                            hard_block += 1
                # Allow a rock/debris in the notch (seen at 12,25) if enough soil.
                if tillable < 6 or hard_block > 2:
                    continue
                # Prefer centers we can path to, or at least path to a hoe stand.
                stand_ok = False
                for _target, stand, _face in hoe_plan(center):
                    if stand == start:
                        stand_ok = True
                        break
                    stand_path = self._pathfinder.find_path(
                        ram, start, stand, max_steps=12
                    )
                    if stand_path and stand_path[-1] == stand:
                        stand_ok = True
                        break
                if not stand_ok:
                    # Center path is enough when stands fail only due to hop cap.
                    if start != center:
                        path = self._pathfinder.find_path(
                            ram, start, center, max_steps=12
                        )
                        if not path or path[-1] != center:
                            continue
                dist = abs(cx - px) + abs(cy - py)
                key = (dist, -tillable, cy, cx)
                if best_key is None or key < best_key:
                    best_key = key
                    best = center
        return best

    def _handle_detect(self, ram: np.ndarray) -> Optional[TaskResult]:
        """Scan for crop plots."""
        self._snapshot_start_acceptance(ram)
        resume_plots = detect_crop_resume_plots(ram, self.bounds)
        if resume_plots:
            supplemental = detect_plots(ram, self.bounds)
            self._plots = _merge_plot_centers(resume_plots, supplemental)
        else:
            self._plots = detect_plots(ram, self.bounds)
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
                self._state = "done"
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
            if self._is_establish_only:
                # Plant pass leaves watering for the later can pass.
                print(
                    f"[CROP] Plot {self._plot_index + 1}/{len(self._plots)} "
                    f"already established; establish-only skips water"
                )
                self._advance_plot(ram)
            else:
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
            if tilled < 2:
                # No reachable till work — reject this planned center and move on.
                self._rejected_plan_centers.add(center)
                print(f"[CROP] Rejecting planned center {center} after failed hoe")
                self._advance_plot(ram)
                return
            if tilled < 4:
                print(
                    f"[CROP] Partial hoe tilled={tilled}; still attempting plant "
                    f"(seed bag covers tilled tiles)"
                )
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
                self._state = "detect"
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

    def _select_preferred_refill_edge(
        self,
        ram: np.ndarray,
        player: Tuple[int, int],
    ) -> Optional[Tuple[Tuple[int, int], str, int, str]]:
        """Pick a pathable preferred fill stand (F0/F9–FD).

        Returns (stand, face, water_id, path_mode) or None. Prefers full BFS,
        then viewport-hop. Does not start fence-open — caller decides that.
        """
        edges = find_pond_edges(
            ram,
            self.refill_bounds or self.bounds,
            water_tiles=REFILL_PREFERRED_WATER_TILES,
            exclude_bad_stands=True,
        )
        if self._bad_refill_tiles:
            edges = [(t, f) for t, f in edges if t not in self._bad_refill_tiles]
        edges = [(t, f) for t, f in edges if not is_bad_refill_stand(t)]
        edges = [
            (t, f)
            for t, f in edges
            if edge_water_tile_id(ram, t, f) in REFILL_PREFERRED_WATER_TILES
        ]
        if not edges:
            return None

        edges = order_preferred_edges(
            edges,
            player,
            water_id_for=lambda t, f: edge_water_tile_id(ram, t, f),
        )
        check_n = min(len(edges), 40)
        hop_fallback: Optional[Tuple[Tuple[int, int], str, int]] = None
        for tile, face in edges[:check_n]:
            water_id = edge_water_tile_id(ram, tile, face)
            full = self._pathfinder.find_path(ram, player, tile)
            if full is not None:
                return (tile, face, water_id, "full")
            # Viewport-near only: hop must actually reach the stand tile.
            # Partial hops toward a fenced-off pond/stream are not reachability.
            if hop_fallback is None:
                hop = self._pathfinder.find_path(
                    ram, player, tile, max_steps=VIEWPORT_HOP_TILES
                )
                if hop is not None and (not hop or hop[-1] == tile):
                    hop_fallback = (tile, face, water_id)
        if hop_fallback is not None:
            tile, face, water_id = hop_fallback
            return (tile, face, water_id, "hop")
        return None

    def _start_refill(self, ram: np.ndarray):
        """Navigate to a CheckToolSuccess fill stand to refill the watering can.

        Order (Clean, no RAM poke):
          1) Named main-pond corridor when pathable
          2) Any other preferred edge (F9 north / FC south / …) while pathable
             — critical: do this *before* fence-open so west-pocket empty-can
             can fill at F9 without burning the day on y=31 fence toss stalls
          3) Stage + open y=31 fence only when no preferred water is reachable
          4) Exhaust
        Non-fill stream IDs F1/F2/F7/F8 are never chosen.
        """
        current_lvl = self._water_level(ram)

        # Track when refill search starts; detect water leaking during search
        if self._refill_search_level < 0:
            self._refill_search_level = current_lvl
        elif current_lvl < self._refill_search_level:
            leaked = self._refill_search_level - current_lvl
            print(f"[CROP] Refill search leaked {leaked} water (was {self._refill_search_level}, now {current_lvl})")
            for bad in list(self._bad_refill_tiles):
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        self._bad_refill_tiles.add((bad[0] + dx, bad[1] + dy))
            self._refill_search_level = current_lvl

        player = self._navigator.current_tile

        def _full_path(start: Tuple[int, int], goal: Tuple[int, int]):
            return self._pathfinder.find_path(ram, start, goal)

        def _reachable_path(start: Tuple[int, int], goal: Tuple[int, int]):
            """True reachability only — not a partial hop toward a blocked goal.

            Viewport hop is used later for navigation after a stand is chosen.
            Using hop as the select_main_pond path oracle falsely treats the
            main pond as pathable when the y=31 wall is still up (hop ends
            north of the wall but is non-None).
            """
            full = self._pathfinder.find_path(ram, start, goal)
            if full is not None:
                return full
            # Goal inside live viewport: hop that actually reaches the stand.
            hop = self._pathfinder.find_path(
                ram, start, goal, max_steps=VIEWPORT_HOP_TILES
            )
            if hop is not None and (not hop or hop[-1] == goal):
                return hop
            return None

        # 1) Named main-pond corridor (FARM_MAIN_POND_STANDS) — only when the
        # stand still faces CheckToolSuccess-valid water on the live map.
        def _pond_fill_ok(stand: Tuple[int, int], face: str) -> bool:
            return edge_water_tile_id(ram, stand, face) in REFILL_PREFERRED_WATER_TILES

        pond = select_main_pond_refill(
            player, _reachable_path, bad_stands=self._bad_refill_tiles
        )
        if pond is not None and not _pond_fill_ok(pond.stand, pond.face):
            pond = None

        if pond is not None:
            wid = edge_water_tile_id(ram, pond.stand, pond.face)
            self._commit_refill_nav(
                ram,
                pond.stand,
                pond.face,
                current_lvl,
                source=pond.source,
                water_id=wid if wid >= 0 else 0xF0,
            )
            return

        blocking = pond_access_blocking_fences(ram)

        # 2) Preferred-edge search BEFORE fence-open. North F9 spur is often
        # pathable from the west plant pocket without clearing y=31; south FC
        # after partial clear too. Fence toss used to starve this path.
        chosen = self._select_preferred_refill_edge(ram, player)
        if chosen is not None:
            tile, face, water_id, path_mode = chosen
            if path_mode == "hop":
                print(
                    f"[CROP] Refill using hop path to "
                    f"({tile[0]},{tile[1]}) water=0x{water_id:02X}"
                )
            self._commit_refill_nav(
                ram,
                tile,
                face,
                current_lvl,
                source="preferred_edge",
                water_id=water_id,
            )
            return

        # 2b) Multi-hop preferred edges (esp. north F9) when full/exact-hop
        # reachability fails under viewport. Dry fixture has F9 at ~(26,12)
        # but full BFS from west pocket is None — without this we burn the day
        # on y=31 fence toss that cannot south-transit empty-handed.
        if self._commit_multihop_preferred_edge(ram, current_lvl):
            return

        # 3) Corridor gap already open but full BFS still fails under viewport
        # staleness — multi-hop to F0 *before* spending another fence-open.
        # ROM trap: after local_drop gap at ~(25,30), preferred edges and full
        # pond BFS are still false; a second FenceClearLoopTask burns the day.
        if self._pond_corridor_gap_open(ram) or self._fence_open_attempts > 0:
            if self._commit_multihop_main_pond(ram, current_lvl):
                return

        # 4) Wall still sealed — open y=31 corridor for main pond.
        if blocking and corridor_needs_fence_open(
            player,
            _full_path,
            blocking_fences=blocking,
            bad_stands=self._bad_refill_tiles,
        ):
            if self._try_open_pond_access(ram, list(blocking)):
                return
        elif blocking and self._try_open_pond_access(ram, list(blocking)):
            return

        # 5) Exhaust — nothing preferred pathable and fence open declined.
        self._refill_exhausted = True
        remaining = len(self._water_steps) - self._water_index
        print(
            f"[CROP] No reachable preferred water edge"
            f"{f' (fences={len(blocking)})' if blocking else ''}"
            f", skipping {remaining} tiles"
        )
        self.skipped_water += remaining
        self._plot_skipped += remaining
        self._water_index = len(self._water_steps)
        self._advance_water_step(ram)

    def _player_carrying(self, ram: np.ndarray) -> bool:
        """True when player is carrying a liftable (fence/bush/rock)."""
        try:
            from harvest.tasks.fence_flow import ACTION_CARRYING_BIT, ADDR_PLAYER_STATE
        except Exception:
            ACTION_CARRYING_BIT = 0x02
            ADDR_PLAYER_STATE = 0xD2
        if ADDR_PLAYER_STATE >= len(ram):
            return False
        return bool(int(ram[ADDR_PLAYER_STATE]) & ACTION_CARRYING_BIT)

    def _queue_local_drop(self) -> None:
        """Queue a multi-face local drop so multi-hop can walk after fence open."""
        self._action_queue.clear()
        for face in ("down", "left", "right", "up"):
            self._action_queue.extend([make_action(**{face: True}) for _ in range(6)])
            self._action_queue.extend(
                [make_action(**{face: True, "a": True}) for _ in range(12)]
            )
            self._action_queue.extend([make_action() for _ in range(8)])
        print("[CROP] Queued local drop (carrying blocks pond multi-hop)")

    def _queue_east_south_corridor_charge(
        self,
        player: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Scripted east-along-y30 then south — ROM-verified gap bypass.

        Empty-handed south through y=31 gap soft-blocks on (13,31). From the
        north lip, holding RIGHT to x≥28 then DOWN reaches y≥32 (probe K).
        BFS densify alone sticks at ~(25,30) soft-block.
        """
        if player is None:
            player = self._navigator.current_tile
        self._action_queue.clear()
        # Nudge off gap row if needed.
        if player[1] >= 31:
            self._action_queue.extend([make_action(up=True) for _ in range(20)])
            self._action_queue.extend([make_action() for _ in range(6)])
        # East run past the soft-block band at x≈25.
        need_right = max(0, 29 - player[0])
        self._action_queue.extend(
            [make_action(right=True, b=True) for _ in range(24 * max(need_right, 6))]
        )
        self._action_queue.extend([make_action() for _ in range(8)])
        # South through / past fence row end into pond band.
        self._action_queue.extend(
            [make_action(down=True, b=True) for _ in range(160)]
        )
        self._action_queue.extend([make_action() for _ in range(12)])
        self._pending_gap_charge = True
        self._east_south_charges = getattr(self, "_east_south_charges", 0) + 1
        print(
            f"[CROP] Queue east→south corridor charge from {player} "
            f"(bypass y=31 gap soft-block) n={self._east_south_charges}"
        )

    def _queue_west_south_lip_charge(
        self,
        player: Optional[Tuple[int, int]] = None,
    ) -> None:
        """From (28,32) soft-block band: west then south corridor to F0 stand.

        ROM: RIGHT/DOWN from (28,32) are dead; LEFT works. Waypoints
        (24,34)→(29,36)→(32,36)→(32,34) then refill_action_sequence fills.
        """
        if player is None:
            player = self._navigator.current_tile
        self._action_queue.clear()
        # West off the soft-block
        self._action_queue.extend(
            [make_action(left=True, b=True) for _ in range(100)]
        )
        self._action_queue.extend([make_action() for _ in range(6)])
        # South
        self._action_queue.extend(
            [make_action(down=True, b=True) for _ in range(100)]
        )
        self._action_queue.extend([make_action() for _ in range(6)])
        # East along y≈35–36 toward pond south lip
        self._action_queue.extend(
            [make_action(right=True, b=True) for _ in range(160)]
        )
        self._action_queue.extend([make_action() for _ in range(6)])
        # North onto (32,34)
        self._action_queue.extend(
            [make_action(up=True, b=True) for _ in range(60)]
        )
        self._action_queue.extend([make_action() for _ in range(8)])
        # Nudge right if short of x=32
        self._action_queue.extend(
            [make_action(right=True) for _ in range(20)]
        )
        self._action_queue.extend([make_action() for _ in range(8)])
        self._pending_south_lip_charge = True
        print(
            f"[CROP] Queue west→south-lip charge from {player} toward F0 stand"
        )

    def _ensure_hands_empty_for_refill(self, ram: np.ndarray) -> bool:
        """If carrying, queue drop and return True (caller should wait)."""
        if not self._player_carrying(ram):
            return False
        if self._action_queue:
            return True
        self._queue_local_drop()
        return True

    def _pond_corridor_gap_open(self, ram: np.ndarray) -> bool:
        """True when the y=31 wall has a usable gap (not sealed, not unknown).

        Partial wall (some posts remain, ≥1 missing) is the common post-
        FenceClearLoopTask state. A completely empty fence row is only treated
        as open after we actually ran fence-open — otherwise blank unit maps
        with 0 fences would falsely multi-hop-commit to F0 stands.
        """
        try:
            from harvest.maps.map_config import FARM_POND_ACCESS_FENCE_X_RANGE

            x0, x1 = FARM_POND_ACCESS_FENCE_X_RANGE
            full = (x1 - x0) + 1
        except Exception:
            full = 19
        n = len(pond_access_blocking_fences(ram))
        if 0 < n < full:
            return True
        if n == 0 and getattr(self, "_fence_open_attempts", 0) > 0:
            return True
        return False

    def _commit_multihop_preferred_edge(
        self,
        ram: np.ndarray,
        current_lvl: int,
    ) -> bool:
        """Multi-hop to a preferred fill edge when exact path is viewport-false.

        Prioritize north-band F9 (and other preferred water north of y=25) so
        empty-can refill from the west plant pocket does not require the y=31
        fence corridor. Returns True if a stand was committed.
        """
        player = self._navigator.current_tile
        edges = find_pond_edges(
            ram,
            self.refill_bounds or self.bounds,
            water_tiles=REFILL_PREFERRED_WATER_TILES,
            exclude_bad_stands=True,
        )
        if self._bad_refill_tiles:
            edges = [(t, f) for t, f in edges if t not in self._bad_refill_tiles]
        edges = [(t, f) for t, f in edges if not is_bad_refill_stand(t)]
        edges = [
            (t, f)
            for t, f in edges
            if edge_water_tile_id(ram, t, f) in REFILL_PREFERRED_WATER_TILES
        ]
        if not edges:
            return False

        # Prefer F9/near-pocket north stands (x≤35, y≤20) before distant FA/FB.
        def sort_key(edge: Tuple[Tuple[int, int], str]) -> Tuple[int, int, int]:
            tile, face = edge
            wid = edge_water_tile_id(ram, tile, face)
            # 0=F9 near, 1=other north preferred, 2=south FC, 3=far east FA/FB
            if wid == 0xF9 or (tile[1] <= 20 and tile[0] <= 35):
                rank = 0
            elif tile[1] <= 25:
                rank = 1
            elif tile[1] >= 45:
                rank = 2
            else:
                rank = 3
            dist = abs(tile[0] - player[0]) + abs(tile[1] - player[1])
            return (rank, dist, wid if wid >= 0 else 999)

        edges = sorted(edges, key=sort_key)
        hop_budget = VIEWPORT_HOP_TILES + 5
        # ROM 2026-08-10: north F9 is sealed from the west plant pocket by the
        # y=13–14 fence bar. Manhattan-improving hops to ~(21,23) are false
        # positives — multihop thrash never reaches F9. Only commit when a full
        # path exists or a hop *nearly arrives* (end within 3 of stand).
        for tile, face in edges[:24]:
            if is_main_pond_stand(tile):
                continue
            wid = edge_water_tile_id(ram, tile, face)
            full = self._pathfinder.find_path(ram, player, tile)
            if full is not None:
                self._commit_refill_nav(
                    ram,
                    tile,
                    face,
                    current_lvl,
                    source="preferred_edge_multihop",
                    water_id=wid,
                    multihop=len(full) > VIEWPORT_HOP_TILES,
                )
                return True
            hop = self._pathfinder.find_path(
                ram, player, tile, max_steps=hop_budget
            )
            if hop is None:
                continue
            end = hop[-1] if hop else player
            # Require a hop that nearly arrives. Manhattan "improved" hops to
            # sealed F9/FA/FB islands thrash forever (dry-fixture false positive).
            nearly = end == tile or tile_dist(end, tile) <= 3
            if not nearly:
                continue
            print(
                f"[CROP] Multi-hop preferred edge ({tile[0]},{tile[1]}) "
                f"face={face} water=0x{wid:02X} end={end} nearly=True"
            )
            self._commit_refill_nav(
                ram,
                tile,
                face,
                current_lvl,
                source="preferred_edge_multihop",
                water_id=wid,
                multihop=True,
            )
            return True
        return False

    def _commit_multihop_main_pond(
        self,
        ram: np.ndarray,
        current_lvl: int,
    ) -> bool:
        """Commit to a main-pond F0 stand for multi-hop navigate after gap open.

        True full-path reachability is often false under live viewport even
        when the y=31 wall has a gap — partial hops + densified north-lip
        waypoints close the distance. Prefer north-lip stands when the player
        is still on y≤30 (post-fence-open stall ~tile 25,30).
        """
        player = self._navigator.current_tile
        try:
            from harvest.maps.map_config import farm_pond_refill_stands
            stands = list(farm_pond_refill_stands())
        except Exception:
            stands = [((32, 34), "up"), ((33, 30), "down"), ((32, 30), "down")]

        candidates = [
            (s, f)
            for s, f in stands
            if s not in self._bad_refill_tiles and not is_bad_refill_stand(s)
        ]
        if not candidates:
            candidates = [(s, f) for s, f in stands if not is_bad_refill_stand(s)]
        if not candidates:
            return False

        # Drop stands that are non-walkable on the live map (dry fixture often
        # has 0x05 fence residue on north-lip (32,30)/(34,30)).
        walkable_cands: List[Tuple[Tuple[int, int], str]] = []
        for s, f in candidates:
            tid = int(get_tile_at(ram, s[0], s[1]))
            if tid in WALKABLE_TILES or tid in (0xA0, 0xA1, 0xA8, 0x01, 0x02, 0x07):
                walkable_cands.append((s, f))
            elif s == player:
                walkable_cands.append((s, f))
        if walkable_cands:
            candidates = walkable_cands

        # After gap open: prefer south-lip stands (32/33,34). North-lip crawl
        # soft-blocks at ~(25,30) (can't walk north around 0xFF).
        south = [(s, f) for s, f in candidates if s[1] >= 33]
        if south:
            candidates = south

        candidates.sort(
            key=lambda sf: abs(sf[0][0] - player[0]) + abs(sf[0][1] - player[1])
        )
        stand, face = candidates[0]
        wid = edge_water_tile_id(ram, stand, face)
        # Stale viewport often returns 0 / dirt far from pond; still commit.
        if wid in REFILL_NONFILL_WATER_TILES:
            for alt_stand, alt_face in candidates[1:]:
                alt_wid = edge_water_tile_id(ram, alt_stand, alt_face)
                if alt_wid not in REFILL_NONFILL_WATER_TILES:
                    stand, face, wid = alt_stand, alt_face, alt_wid
                    break

        water_id = wid if wid in REFILL_PREFERRED_WATER_TILES else 0xF0
        self._commit_refill_nav(
            ram,
            stand,
            face,
            current_lvl,
            source="main_pond_multihop",
            water_id=water_id,
            multihop=True,
        )
        return True

    def _refill_hop_goal(
        self,
        ram: np.ndarray,
        player: Tuple[int, int],
        ultimate: Tuple[int, int],
    ) -> Tuple[int, int]:
        """Densify multi-hop refill: nearest corridor waypoint that closes dist.

        Without intermediates, hop-toward F0 from ~(25,30) stalls when the
        next live walkable cell is not strictly closer in raw BFS (viewport
        dirt IDs / fence residue). Named north-lip chain keeps progress.

        Monotonic: never pick a waypoint farther from the ultimate stand than
        the best distance already achieved — that was the (24,30)→(15,30)
        thrash on the dry fixture.
        """
        dist_u = tile_dist(player, ultimate)
        hop_budget = VIEWPORT_HOP_TILES + 3
        best_seen = getattr(self, "_refill_best_dist", dist_u)

        # Only accept the ultimate when a *true* short path exists.
        full = self._pathfinder.find_path(ram, player, ultimate)
        if full is not None and len(full) <= hop_budget:
            return ultimate

        try:
            from harvest.maps.map_config import (
                FARM_POND_MULTIHOP_WAYPOINTS,
                FARM_POND_POST_GAP_CORRIDOR,
            )
            post_gap = FARM_POND_POST_GAP_CORRIDOR
            chain = FARM_POND_MULTIHOP_WAYPOINTS
        except Exception:
            post_gap = (
                (13, 32),
                (16, 32),
                (20, 32),
                (24, 32),
                (28, 32),
                (32, 34),
            )
            chain = post_gap

        # ROM trap: after east→south wall cross, player often lands (28,32)
        # where RIGHT/DOWN soft-block. BFS wants (29,32)→pond and thrash.
        # Route WEST then south corridor: (24,32)→(24,35)→(29,36)→(32,34).
        if (
            player[1] >= 32
            and ultimate[1] >= 33
            and player[0] <= 30
            and tile_dist(player, ultimate) > 1
        ):
            south_lip_crumbs: List[Tuple[int, int]] = [
                (24, 32),
                (24, 33),
                (24, 34),
                (24, 35),
                (26, 35),
                (28, 35),
                (29, 36),
                (30, 36),
                (32, 36),
                (32, 35),
                (32, 34),
                (33, 34),
            ]
            # If on the soft-block tile, force west first.
            if player[0] >= 27 and player[1] <= 33:
                for wp in ((24, 32), (24, 33), (22, 32), (20, 32)):
                    if wp == player:
                        continue
                    hop = self._pathfinder.find_path(
                        ram, player, wp, max_steps=hop_budget + 2
                    )
                    if hop is not None:
                        print(
                            f"[CROP] Pond soft-block (28,32) band: west first "
                            f"{player} → {wp}"
                        )
                        return wp
                return (24, 32)
            for wp in south_lip_crumbs:
                if wp == player:
                    continue
                d_to_wp = tile_dist(player, wp)
                if d_to_wp > hop_budget + 4:
                    continue
                # Prefer crumbs that improve toward ultimate without re-entering
                # the (28,32) east-lock.
                if wp[0] >= 28 and wp[1] <= 33 and player[0] <= 27:
                    continue
                hop = self._pathfinder.find_path(
                    ram, player, wp, max_steps=hop_budget + 4
                )
                if hop is None:
                    continue
                end = hop[-1]
                if end == player:
                    continue
                if tile_dist(end, ultimate) < dist_u or end[1] > player[1]:
                    print(
                        f"[CROP] South-lip densify {player} → {wp} (end={end})"
                    )
                    return wp

        # ROM trap: multi-hop to main-pond south lip from north of y=31.
        # Empty-handed south through a y=31 gap soft-blocks on (13,31) y≈505
        # (BFS invents (12,32) path that game physics rejects). NEVER densify
        # south through the gap empty-handed.
        #
        # ROM-verified routes after gap open (Y1_Test_Crops_Planted_Dry):
        #   1) Carry-south while holding a post (FenceClearLoop corridor_only)
        #   2) East-crawl on y=30 to x≥28 then pure south (empty OK)
        # Prefer (2) for post-drop multi-hop; never charge gap from y≤31 empty.
        if player[1] <= 31 and ultimate[1] >= 32:
            # East-crawl corridor: y=30 lip → x≥28 → y=32 → pond south lip.
            east_crumbs: List[Tuple[int, int]] = [
                (min(player[0] + 4, 28), min(player[1], 30)),
                (20, min(player[1], 30)),
                (24, 30),
                (26, 30),
                (28, 30),
                (28, 32),
                (30, 32),
                (30, 33),
                (32, 33),
                (32, 34),
            ]
            # On the gap row y=31: step north/east off the soft-block tile first.
            if player[1] == 31:
                for wp in (
                    (min(player[0] + 2, 28), 30),
                    (player[0], 30),
                    (player[0] + 1, 30),
                    (player[0] - 1, 30),
                    (20, 30),
                    (24, 30),
                    (28, 30),
                ):
                    if wp == player or wp[0] < 0 or wp[1] < 0:
                        continue
                    hop = self._pathfinder.find_path(
                        ram, player, wp, max_steps=hop_budget + 2
                    )
                    if hop is not None:
                        print(
                            f"[CROP] Gap soft-block escape {player} → {wp} "
                            f"(east-crawl, never south through gap)"
                        )
                        return wp
                return (min(player[0] + 3, 28), 30)

            for wp in east_crumbs:
                if wp == player:
                    continue
                # Only accept crumbs that improve toward ultimate or push east/south.
                if wp[0] < player[0] and wp[1] <= player[1]:
                    continue
                d_to_wp = tile_dist(player, wp)
                if d_to_wp > hop_budget + 4:
                    continue
                hop = self._pathfinder.find_path(
                    ram, player, wp, max_steps=hop_budget + 4
                )
                if hop is None:
                    continue
                end = hop[-1] if hop else player
                # Reject hops that only walk onto the gap tile (y=31, x≈12–16).
                if end[1] == 31 and end[0] <= 18:
                    continue
                if end[1] >= 32 or end[0] > player[0] or tile_dist(end, ultimate) < dist_u:
                    print(
                        f"[CROP] East-crawl densify {player} → {wp} "
                        f"(end={end}, ultimate={ultimate})"
                    )
                    return wp

            # Fallback: walk east on current lip (never south through gap).
            for wx in (28, 26, 24, 22, 20, 18):
                wp = (wx, min(player[1], 30))
                if wp == player or wp[0] <= player[0]:
                    continue
                hop = self._pathfinder.find_path(
                    ram, player, wp, max_steps=hop_budget + 2
                )
                if hop is not None:
                    return wp
            return (min(player[0] + 4, 28), min(player[1], 30))

        # North F9 multi-hop only when already north of the y=13 fence bar
        # (player y≤16) or east of it (x≥20). From west plant pocket F9 is
        # sealed — do not densify into potato/y=13 thrash.
        if ultimate[1] <= 20 and (player[1] <= 16 or player[0] >= 20):
            north_crumbs: List[Tuple[int, int]] = [
                (20, 16),
                (22, 14),
                (24, 13),
                (25, 13),
                (player[0], max(ultimate[1], player[1] - 4)),
                (min(ultimate[0], player[0] + 3), max(ultimate[1], player[1] - 3)),
                (ultimate[0] - 1, ultimate[1]),
                ultimate,
            ]
            for wp in north_crumbs:
                if wp == player or wp[0] < 0 or wp[1] < 0:
                    continue
                hop = self._pathfinder.find_path(
                    ram, player, wp, max_steps=hop_budget + 2
                )
                if hop is None:
                    continue
                end = hop[-1]
                if end[1] < player[1] or tile_dist(end, ultimate) < dist_u:
                    return wp

        # Default: next breadcrumb closer to ultimate (south corridor first).
        best: Optional[Tuple[int, int]] = None
        best_goal_dist = dist_u
        best_wp_dist = 999
        progress_cap = min(dist_u, best_seen + 1)
        for wp in chain:
            if wp == player:
                continue
            d_to_goal = tile_dist(wp, ultimate)
            if d_to_goal >= progress_cap:
                continue
            d_to_wp = tile_dist(player, wp)
            if d_to_wp > hop_budget or d_to_wp < 1:
                continue
            hop = self._pathfinder.find_path(
                ram, player, wp, max_steps=hop_budget
            )
            if hop is None:
                continue
            end = hop[-1] if hop else player
            if tile_dist(end, ultimate) >= dist_u:
                continue
            if (
                best is None
                or d_to_wp < best_wp_dist
                or (d_to_wp == best_wp_dist and d_to_goal < best_goal_dist)
            ):
                best = wp
                best_goal_dist = d_to_goal
                best_wp_dist = d_to_wp

        if best is not None:
            return best
        return ultimate

    def _commit_refill_nav(
        self,
        ram: np.ndarray,
        stand: Tuple[int, int],
        face: str,
        current_lvl: int,
        *,
        source: str,
        water_id: int = 0xF0,
        multihop: bool = False,
    ) -> None:
        """Begin navigate/act for a chosen refill stand."""
        player = self._navigator.current_tile
        self._refill_pond_tile = stand
        self._refill_pond_face = face
        self._refill_level_before = current_lvl
        self._clear_crop_walkable()
        self._plot_phase = "refill"
        self._target_tile = stand
        self._approach_tile = stand
        self._face_direction = face
        self._state = "navigate"
        self._navigator.path = []
        self._steps_on_target = 0
        dist = abs(stand[0] - player[0]) + abs(stand[1] - player[1])
        # Multi-hop when far or after fence-gap (viewport cannot full-BFS).
        self._refill_multihop = bool(multihop) or dist > VIEWPORT_HOP_TILES
        self._refill_best_dist = dist
        band = refill_stand_band(stand)
        print(
            f"[CROP] Refill at ({stand[0]},{stand[1]}) facing {face} "
            f"water=0x{water_id:02X} source={source} dist={dist} band={band} "
            f"can={current_lvl} multihop={self._refill_multihop}"
        )

    def _pond_access_staging_tiles(self) -> Tuple[Tuple[int, int], ...]:
        try:
            from harvest.maps.map_config import FARM_POND_ACCESS_STAGING_TILES
            return FARM_POND_ACCESS_STAGING_TILES
        except Exception:
            return (
                (11, 29),
                (12, 29),
                (10, 28),
                (11, 28),
                (15, 29),
                (18, 30),
                (20, 30),
            )

    def _try_stage_pond_access(self, ram: np.ndarray) -> bool:
        """Nav to a free stand north of the fence wall before clearing fences.

        ROM trap: after planting in the west pocket the bot often stands on
        (13,27) where pure-south movement soft-blocks (tile IDs still look
        walkable). Staging west/left first makes FenceClearLoopTask pathable.
        """
        if getattr(self, "_pond_staged", False):
            return False
        player = self._navigator.current_tile
        staging_tiles = self._pond_access_staging_tiles()

        def _hop(start: Tuple[int, int], goal: Tuple[int, int]):
            return self._find_nav_path(ram, start, goal)

        target = select_staging_stand(player, _hop, staging_tiles=staging_tiles)
        if target is None:
            return False
        if target.stand == player:
            self._pond_staged = True
            return False

        self._pond_staged = True
        self._pending_fence_open = True
        self._plot_phase = "stage_pond"
        self._target_tile = target.stand
        self._approach_tile = target.stand
        self._face_direction = target.face
        self._state = "navigate"
        self._navigator.path = []
        self._steps_on_target = 0
        print(
            f"[CROP] Staging pond access via ({target.stand[0]},{target.stand[1]}) "
            f"from {player} (named corridor)"
        )
        return True

    def _try_open_pond_access(
        self,
        ram: np.ndarray,
        fences: List[Tuple[int, int]],
        *,
        skip_stage: bool = False,
    ) -> bool:
        """Start a limited fence-clear subtask to open the y=31 pond corridor.

        Returns True if a subtask/nav was started (caller should wait). Fence
        toss targets the main pond lip, so success both opens the path and
        parks the player at a fill stand.
        """
        if getattr(self, "_fence_open_attempts", 0) >= 2:
            return False
        if not fences:
            return False

        # Stage out of the plant pocket first — otherwise FenceClearLoopTask
        # plans a pure-south path that game physics never accepts.
        if not skip_stage and self._try_stage_pond_access(ram):
            return True

        try:
            from harvest.tasks.fence_flow import FenceClearLoopTask
        except Exception as exc:
            print(f"[CROP] Fence open unavailable: {exc}")
            return False

        self._fence_open_attempts = getattr(self, "_fence_open_attempts", 0) + 1
        self._pending_fence_open = False
        # Prefer fences nearest the player on the access row.
        player = self._navigator.current_tile
        fences_sorted = sorted(
            fences,
            key=lambda t: abs(t[0] - player[0]) + abs(t[1] - player[1]),
        )
        print(
            f"[CROP] Opening pond access: corridor_only clear 2 fences "
            f"(nearest={fences_sorted[0]}, wall n={len(fences)}, from={player})"
        )
        # corridor_only: local-drop (no pond toss thrash). Clear 2 adjacent
        # fences — single-tile gap soft-blocks south transit empty-handed;
        # two-wide gap is walkable on the dry fixture.
        task = FenceClearLoopTask(
            max_fences=2,
            max_steps_per_fence=1600,
            corridor_only=True,
        )
        # Lightweight world for reset
        from types import SimpleNamespace
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task.reset(world)
        self._fence_subtask = task
        self._plot_phase = "open_pond"
        self._state = "fence_open"
        self._navigator.path = []
        # Fence subtask must not inherit stage/water steps_on_target budget.
        self._steps_on_target = 0
        return True

    def _handle_fence_open(self, world: WorldState) -> Optional[TaskResult]:
        """Drive FenceClearLoopTask until corridor opens, then resume refill."""
        task = self._fence_subtask
        if task is None:
            self._state = "detect"
            self._plot_phase = "water"
            return None

        result = task.step(world)
        status = result.status
        if status == TaskStatus.RUNNING:
            return result
        if status == TaskStatus.SUCCESS:
            cleared = getattr(task, "cleared_count", 0)
            player = self._navigator.current_tile
            print(
                f"[CROP] Pond access open (cleared={cleared} fences) "
                f"at {player}; multi-hop F0"
            )
            self._fence_subtask = None
            # Carry-south success: already south of wall with/without post.
            if player[1] >= 32:
                if self._ensure_hands_empty_for_refill(world.ram):
                    self._pending_multihop_after_drop = True
                    self._plot_phase = "refill"
                    self._state = "navigate"
                    return None
                lvl = self._water_level(world.ram)
                if self._commit_multihop_main_pond(world.ram, lvl):
                    return None
            # Still north: drop any post, then scripted east→south corridor
            # (ROM: empty gap charge soft-blocks; pure right from ~x13 y30 to
            # x≥28 then down crosses). Densify alone sticks at ~(25,30).
            if self._ensure_hands_empty_for_refill(world.ram):
                self._pending_multihop_after_drop = True
                self._plot_phase = "refill"
                self._state = "navigate"
                return None
            self._queue_east_south_corridor_charge(player)
            self._plot_phase = "refill"
            self._state = "navigate"
            return None
        # Failure / blocked — try multi-hop if gap partial, else full search.
        print(f"[CROP] Fence open failed: {result.reason}; retrying refill search")
        self._fence_subtask = None
        if self._ensure_hands_empty_for_refill(world.ram):
            self._pending_multihop_after_drop = True
            self._plot_phase = "refill"
            self._state = "navigate"
            return None
        lvl = self._water_level(world.ram)
        if self._pond_corridor_gap_open(world.ram):
            if self._commit_multihop_main_pond(world.ram, lvl):
                return None
        self._plot_phase = "water"
        self._start_refill(world.ram)
        return None

    def _find_nav_path(
        self,
        ram: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """Pathfind with viewport hop fallback for distant crop targets.

        Live farm tiles go stale outside the loaded viewport. Without
        ``max_steps``, ``find_path`` returns None for goals ~12+ tiles away
        and establish/water immediately skip every hoe stand. Hop toward the
        goal so multi-hop navigation can close the gap.
        """
        # Refill stands are often 15–25 tiles from west plots after fence open;
        # allow a longer hop so multi-hop can keep closing without false exhaust.
        # Densify intermediate goals so hop-toward does not thrash at ~(25,30).
        hop = VIEWPORT_HOP_TILES + 3 if self._plot_phase == "refill" else VIEWPORT_HOP_TILES
        nav_goal = goal
        if self._plot_phase == "refill" and getattr(self, "_refill_multihop", False):
            nav_goal = self._refill_hop_goal(ram, start, goal)
            if nav_goal != goal:
                print(
                    f"[CROP] Refill densify hop {start} → {nav_goal} "
                    f"(ultimate={goal})"
                )
        path = self._pathfinder.find_path(
            ram,
            start,
            nav_goal,
            max_steps=hop,
        )
        # Reject regressive truncated hops (long west-around routes).
        if (
            path
            and self._plot_phase == "refill"
            and getattr(self, "_refill_multihop", False)
        ):
            end = path[-1]
            if tile_dist(end, goal) >= tile_dist(start, goal) and end != goal:
                # Try an explicit densify target once more.
                alt = self._refill_hop_goal(ram, start, goal)
                if alt != nav_goal and alt != start:
                    alt_path = self._pathfinder.find_path(
                        ram, start, alt, max_steps=hop
                    )
                    if alt_path and tile_dist(alt_path[-1], goal) < tile_dist(
                        start, goal
                    ):
                        print(
                            f"[CROP] Refill reject regressive hop end={end}; "
                            f"using densify {alt}"
                        )
                        return alt_path
                return None
        return path

    def _recover_refill_nav(self, ram: np.ndarray, *, reason: str) -> None:
        """Recover from mid-refill path loss without hard-exhausting once.

        Multi-hop pond approaches often lose full BFS mid-route when the
        viewport rolls; blacklisting the stand + reselecting (or snapping when
        already adjacent to a corridor stand) is enough to continue. Only after
        several soft fails do we mark refill exhausted.
        """
        player = self._navigator.current_tile
        try:
            from harvest.maps.map_config import farm_pond_refill_stands
            stands = farm_pond_refill_stands()
        except Exception:
            stands = (((32, 34), "up"), ((33, 30), "down"))

        for stand, face in stands:
            if tile_dist(player, stand) <= 1:
                print(
                    f"[CROP] Refill snap to nearby stand {stand} face={face} "
                    f"({reason})"
                )
                self._refill_pond_tile = stand
                self._refill_pond_face = face
                self._target_tile = stand
                self._approach_tile = stand
                self._face_direction = face
                self._state = "act"
                self._navigator.path = []
                self._steps_on_target = 0
                self._refill_multihop = False
                return

        # Track multi-hop progress: if we closed distance, soft-retry without
        # blacklisting the ultimate F0 stand (viewport hop thrash otherwise).
        ultimate = self._refill_pond_tile or self._approach_tile
        if ultimate is not None:
            cur_dist = tile_dist(player, ultimate)
            best = getattr(self, "_refill_best_dist", 999)
            if cur_dist < best:
                self._refill_best_dist = cur_dist
                print(
                    f"[CROP] Refill multi-hop progress dist {best}→{cur_dist} "
                    f"at {player} ({reason}); repath"
                )
                self._state = "navigate"
                self._navigator.path = []
                self._navigator.stasis = 0
                self._pathfinder.temp_blocked.clear()
                return

        # Multi-hop densify soft retry before burning a failure slot.
        if getattr(self, "_refill_multihop", False) and ultimate is not None:
            hop_goal = self._refill_hop_goal(ram, player, ultimate)
            if hop_goal != ultimate and hop_goal != player:
                path = self._pathfinder.find_path(
                    ram,
                    player,
                    hop_goal,
                    max_steps=VIEWPORT_HOP_TILES + 3,
                )
                if path:
                    print(
                        f"[CROP] Refill densify recover → {hop_goal} "
                        f"from {player} ({reason})"
                    )
                    self._navigator.path = path
                    self._navigator.stasis = 0
                    self._state = "navigate"
                    return

        # Only blacklist non-progress failures; keep main pond stands longer.
        if self._refill_pond_tile is not None and not is_main_pond_stand(
            self._refill_pond_tile
        ):
            self._bad_refill_tiles.add(self._refill_pond_tile)
        elif self._refill_pond_tile is not None:
            # Main pond: only blacklist after repeated no-progress stalls.
            fails = getattr(self, "_refill_nav_failures", 0)
            if fails >= 3:
                self._bad_refill_tiles.add(self._refill_pond_tile)

        self._refill_nav_failures = getattr(self, "_refill_nav_failures", 0) + 1
        if self._refill_nav_failures >= 8:
            print(
                f"[CROP] Refill nav failed {self._refill_nav_failures}x "
                f"({reason}); exhausting"
            )
            self._refill_exhausted = True
            self._refill_multihop = False
            self._plot_phase = "water"
            self._set_water_walkable()
            if self._water_index < len(self._water_steps):
                target, stand, face = self._water_steps[self._water_index]
                self._target_tile = target
                self._approach_tile = stand
                self._face_direction = face
            elif self._plots and self._plot_index < len(self._plots):
                center = self._plots[self._plot_index]
                self._target_tile = center
                self._approach_tile = center
            self._state = "navigate"
            self._navigator.path = []
            return

        print(
            f"[CROP] Refill repath {self._refill_nav_failures}/8 after {reason} "
            f"(stand={self._refill_pond_tile} pos={player})"
        )
        # Prefer re-commit multi-hop to nearest remaining stand over full search
        # which may re-enter fence-open when wall residue remains.
        if self._pond_corridor_gap_open(ram) or self._fence_open_attempts > 0:
            if self._commit_multihop_main_pond(ram, self._water_level(ram)):
                return
        self._start_refill(ram)

    def _handle_navigate(self, ram: np.ndarray) -> Optional[TaskResult]:
        # Finish local drop before multi-hop after fence open.
        if getattr(self, "_pending_multihop_after_drop", False):
            if self._player_carrying(ram):
                if not self._action_queue:
                    self._queue_local_drop()
                return None  # step() drains queue
            self._pending_multihop_after_drop = False
            player = self._navigator.current_tile
            if player[1] >= 32:
                print(f"[CROP] Hands empty south of wall at {player}; multi-hop F0")
                if self._commit_multihop_main_pond(ram, self._water_level(ram)):
                    return None
                self._plot_phase = "water"
                self._start_refill(ram)
                return None
            # Still north of wall — scripted east→south (not densify thrash).
            print(
                f"[CROP] Hands empty north of wall at {player}; "
                f"east→south corridor charge"
            )
            self._queue_east_south_corridor_charge(player)
            return None

        # Drain N/E nudge after gap drop, then multi-hop.
        if getattr(self, "_pending_gap_reseat", False):
            if self._action_queue:
                return None
            self._pending_gap_reseat = False
            player = self._navigator.current_tile
            if player[1] < 32:
                self._queue_east_south_corridor_charge(player)
                return None
            print(f"[CROP] Gap nudge done at {player}; multi-hop F0")
            if self._commit_multihop_main_pond(ram, self._water_level(ram)):
                return None
            self._plot_phase = "water"
            self._start_refill(ram)
            return None

        # East→south corridor charge (or legacy gap-charge) completion.
        if getattr(self, "_pending_gap_charge", False):
            if self._action_queue:
                return None
            self._pending_gap_charge = False
            player = self._navigator.current_tile
            print(
                f"[CROP] Corridor charge done at {player}; multi-hop F0 "
                f"(y={'ok' if player[1] >= 32 else 'still_north'})"
            )
            # If still stuck on the ~(25,30) soft-block, one more hard charge.
            if (
                player[1] <= 31
                and player[0] < 28
                and getattr(self, "_east_south_charges", 0) < 2
            ):
                print("[CROP] Still north/west; re-queue east→south charge")
                self._queue_east_south_corridor_charge(player)
                return None
            # Landed on (28,32) soft-block band: script west then south-lip.
            if player[1] >= 32 and player[0] >= 27 and player[0] <= 30:
                print(
                    f"[CROP] South-of-wall soft-block band at {player}; "
                    f"queue west→south-lip to F0"
                )
                self._queue_west_south_lip_charge(player)
                return None
            if self._commit_multihop_main_pond(ram, self._water_level(ram)):
                return None
            self._plot_phase = "water"
            self._start_refill(ram)
            return None

        # West→south-lip charge completion (after (28,32) soft-block).
        if getattr(self, "_pending_south_lip_charge", False):
            if self._action_queue:
                return None
            self._pending_south_lip_charge = False
            player = self._navigator.current_tile
            print(f"[CROP] South-lip charge done at {player}; multi-hop/act F0")
            if self._commit_multihop_main_pond(ram, self._water_level(ram)):
                # If already on/near stand, snap to act.
                if (
                    self._refill_pond_tile is not None
                    and tile_dist(player, self._refill_pond_tile) <= 1
                ):
                    self._approach_tile = self._refill_pond_tile
                    self._target_tile = self._refill_pond_tile
                    self._state = "act"
                    self._navigator.path = []
                return None
            self._plot_phase = "water"
            self._start_refill(ram)
            return None

        # Mid-refill: if we somehow started carrying, drop before walking.
        if (
            self._plot_phase == "refill"
            and self._player_carrying(ram)
            and not self._action_queue
        ):
            self._queue_local_drop()
            return None

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
        # Refill multi-hop: adjacent to stand is enough to act (center tolerance).
        if (
            self._plot_phase == "refill"
            and self._approach_tile is not None
            and tile_dist(self._navigator.current_tile, self._approach_tile) <= 1
        ):
            self._state = "center"
            return None

        # Track multi-hop progress toward the ultimate refill stand.
        if self._plot_phase == "refill" and self._approach_tile is not None:
            d = tile_dist(self._navigator.current_tile, self._approach_tile)
            if d < getattr(self, "_refill_best_dist", 999):
                self._refill_best_dist = d
                # Progress resets the per-target timeout so multi-hop F0 can
                # cross ~20 tiles without false "Refill timed out".
                self._steps_on_target = 0
                self._pathfinder.temp_blocked.clear()

        # Stuck recovery
        if self._navigator.stasis > self.stasis_repath and self._navigator.path:
            # Refill multi-hop: do not permanently block the next east cell —
            # stasis on the north lip often means animation lock, not a wall.
            if not (
                self._plot_phase == "refill"
                and getattr(self, "_refill_multihop", False)
            ):
                self._pathfinder.temp_blocked.add(self._navigator.path[0])
            else:
                self._pathfinder.temp_blocked.clear()
            path = self._find_nav_path(ram, self._navigator.current_tile, self._approach_tile)
            if path:
                self._navigator.path = path
                self._navigator.stasis = 0
            else:
                self._failures += 1
                self._failed_tiles.add(self._target_tile)
                if self._plot_phase == "water":
                    # ROM: post-F0 return north through y=31 gap soft-blocks
                    # pure-up from ~(13,33). Nudge right then up (lands y≤30).
                    player = self._navigator.current_tile
                    if (
                        player[1] >= 32
                        and self._approach_tile is not None
                        and self._approach_tile[1] <= 30
                        and not self._action_queue
                        and getattr(self, "_water_north_returns", 0) < 2
                    ):
                        self._water_north_returns = (
                            getattr(self, "_water_north_returns", 0) + 1
                        )
                        self._action_queue.extend(
                            [make_action(right=True, b=True) for _ in range(40)]
                        )
                        self._action_queue.extend(
                            [make_action(up=True, b=True) for _ in range(160)]
                        )
                        self._action_queue.extend([make_action() for _ in range(8)])
                        self._navigator.stasis = 0
                        self._navigator.path = []
                        print(
                            f"[CROP] Water return north charge from {player} "
                            f"(n={self._water_north_returns})"
                        )
                        return None
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
                    self._recover_refill_nav(ram, reason="stuck nav")
                elif self._plot_phase == "stage_pond":
                    print("[CROP] Stage pond stuck; trying fence open from here")
                    fences = pond_access_blocking_fences(ram)
                    if fences and self._try_open_pond_access(ram, fences, skip_stage=True):
                        return None
                    self._plot_phase = "water"
                    self._start_refill(ram)
                else:
                    self._state = "detect"
                if self._failures >= self.max_failures:
                    return TaskResult(status=TaskStatus.FAILURE, reason="too many nav failures")
                return None

        # Try to path if no current path
        if not self._navigator.path:
            path = self._find_nav_path(ram, self._navigator.current_tile, self._approach_tile)
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
                    self._recover_refill_nav(ram, reason="no path")
                elif self._plot_phase == "stage_pond":
                    print("[CROP] No path to pond stage; trying fence open from here")
                    fences = pond_access_blocking_fences(ram)
                    if fences and self._try_open_pond_access(ram, fences, skip_stage=True):
                        return None
                    self._plot_phase = "water"
                    self._start_refill(ram)
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
        tol = 1 if self._plot_phase in ("plant", "water", "hoe", "stage_pond") else 2
        center_action = self._navigator.center_on_tile(self._approach_tile, tolerance=tol)
        if center_action is None:
            if self._plot_phase == "stage_pond":
                # Staged — hand off to fence clear (skip re-stage).
                fences = pond_access_blocking_fences(ram)
                print(
                    f"[CROP] Pond stage reached at {self._navigator.current_tile}; "
                    f"starting fence clear (wall n={len(fences)})"
                )
                if fences and self._try_open_pond_access(ram, fences, skip_stage=True):
                    return None
                self._plot_phase = "water"
                self._start_refill(ram)
                return None
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
        if self._plot_phase in ("plant", "water", "hoe", "stage_pond"):
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
        tol = 1 if self._plot_phase in ("plant", "water", "hoe", "stage_pond") else 2
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
        elif self._plot_phase == "stage_pond":
            fences = pond_access_blocking_fences(ram)
            if fences and self._try_open_pond_access(ram, fences, skip_stage=True):
                return None
            self._plot_phase = "water"
            self._start_refill(ram)
            return None
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

        # Empty can: always refill before attempting water (ToolUsed early-outs at 0).
        if water_lvl < 1 and not self._refill_exhausted:
            print(f"[CROP] Empty can (level={water_lvl}), need refill before watering")
            self._start_refill(ram)
            return None

        # Count only waterable remaining tiles for partial-can refill check
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
            if self._is_establish_only:
                # Day-plan plant pass: seeds+hoe only; water after can re-fetch.
                self._advance_plot(ram)
            else:
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
            return self._terminal_result()

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
        if self._plot_phase in ("open_pond", "stage_pond", "fence_open"):
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
                    self._plot_phase = "refill"
                    self._state = "navigate"
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
                self._plot_phase = "water"
                self._start_refill(world.ram)
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action()),
                )
            # Fall through to normal step handling without target timeout.
            pass
        refill_budget = (
            max(self.max_steps_per_target * 3, 3600)
            if self._plot_phase == "refill"
            else self.max_steps_per_target
        )
        if (
            self._plot_phase not in ("open_pond", "stage_pond", "fence_open")
            and self._steps_on_target > refill_budget
            and self._target_tile is not None
        ):
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
                print(
                    f"[CROP] Refill timed out at {self._navigator.current_tile} "
                    f"stand={self._refill_pond_tile} best_dist="
                    f"{getattr(self, '_refill_best_dist', '?')}"
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
                self._plot_phase = "water"
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
            return self._terminal_result()

        # State dispatch
        if self._state == "fence_open":
            result = self._handle_fence_open(world)
            if result is not None:
                return result

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

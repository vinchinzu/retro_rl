"""Watering-can refill selection for the main F0 pond corridor.

ROM fact (CheckToolSuccess / ToolAnimationWateringCan): farm fill only when
the tile-in-front property is F0/F9–FD (can → 0x14). Prefer the named
``map_config`` main-pond stands over generic water-edge search so west-pocket
plants open one corridor instead of ranking non-fill streams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from harvest.maps.map_config import (
    FARM_MAIN_POND_STANDS,
    FARM_POND_ACCESS_STAGING_TILES,
    FARM_POND_REFILL_CORRIDOR,
    farm_pond_refill_stands,
    player_in_west_plant_pocket,
)
from harvest.tasks.nav import MAP_WIDTH, WALKABLE_TILES, get_tile_at

PathFn = Callable[
    [Tuple[int, int], Tuple[int, int]],
    Optional[Sequence[Tuple[int, int]]],
]


# Preferred CheckToolSuccess fill properties (raw map IDs that usually map).
REFILL_PREFERRED_WATER_TILES = frozenset({0xF0, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD})
REFILL_NONFILL_WATER_TILES = frozenset({0xF1, 0xF2, 0xF7, 0xF8})

# Pond/water tiles — stand adjacent, face them, use watering can.
WATER_TILES = frozenset({
    0xA6,  # pond edge
    0xF0, 0xF1, 0xF2,
    0xF7, 0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD,
})
# Actual refill water — excludes 0xA6 (decorative pond border).
REFILL_WATER_TILES = REFILL_PREFERRED_WATER_TILES | REFILL_NONFILL_WATER_TILES

# Shipping-bin F2 pocket (x~8-9, y~29-30) never fills. Inclusive stand bbox.
BAD_REFILL_STAND_BOUNDS = (6, 27, 12, 33)
# Lower band = better; secondary to preferred-water-id in refill_edge_sort_key.
REFILL_BAND_POND = 0    # main F0 pond stands (y 28–36, x 28–36)
REFILL_BAND_SOUTH = 1   # y >= 45: south stream FC / SE FD
REFILL_BAND_NORTH = 2   # y <= 25: north spur F9 / east FA
REFILL_BAND_MID = 3     # other mid-farm (east FB, etc.)
REFILL_BAND_BAD = 4     # known-bad shipping pocket
MAIN_POND_STAND_BOUNDS = (28, 28, 36, 36)

_REFILL_FACE_DELTA = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


@dataclass(frozen=True)
class RefillTarget:
    """Stand tile + face for a can refill attempt."""

    stand: Tuple[int, int]
    face: str
    source: str  # "main_pond_corridor" | "preferred_edge" | "staging"
    pathable: bool = True


def select_main_pond_refill(
    player: Tuple[int, int],
    find_path: PathFn,
    *,
    bad_stands: Optional[set] = None,
) -> Optional[RefillTarget]:
    """Pick the nearest pathable stand from FARM_MAIN_POND_STANDS.

    ``find_path(start, goal)`` should return a full path list ending at goal,
    or None if unreachable. After a y=31 fence gap open the player is often on
    the *north* lip — preferring a fixed south-lip order then times out walking
    around the pond body.
    """
    blocked = bad_stands or set()
    hits: List[Tuple[int, int, RefillTarget]] = []
    for rank, (stand, face) in enumerate(farm_pond_refill_stands()):
        if stand in blocked:
            continue
        path = find_path(player, stand)
        if path is not None:
            dist = abs(stand[0] - player[0]) + abs(stand[1] - player[1])
            hits.append(
                (
                    dist,
                    rank,
                    RefillTarget(
                        stand=stand,
                        face=face,
                        source="main_pond_corridor",
                        pathable=True,
                    ),
                )
            )
    if not hits:
        return None
    hits.sort(key=lambda row: (row[0], row[1]))
    return hits[0][2]


def select_staging_stand(
    player: Tuple[int, int],
    find_path: PathFn,
    *,
    staging_tiles: Sequence[Tuple[int, int]] = FARM_POND_ACCESS_STAGING_TILES,
) -> Optional[RefillTarget]:
    """Nearest pathable staging tile north of the y=31 fence wall."""
    if player in staging_tiles:
        return RefillTarget(
            stand=player,
            face="down",
            source="staging",
            pathable=True,
        )
    if not player_in_west_plant_pocket(player):
        return None
    ordered = sorted(
        staging_tiles,
        key=lambda t: abs(t[0] - player[0]) + abs(t[1] - player[1]),
    )
    for stage in ordered:
        path = find_path(player, stage)
        if path is not None:
            return RefillTarget(
                stand=stage,
                face="down",
                source="staging",
                pathable=True,
            )
    return None


def corridor_needs_fence_open(
    player: Tuple[int, int],
    find_path: PathFn,
    *,
    blocking_fences: Sequence[Tuple[int, int]],
    bad_stands: Optional[set] = None,
) -> bool:
    """True when main pond is unreachable and the y=31 fence wall is up."""
    if not blocking_fences:
        return False
    if select_main_pond_refill(player, find_path, bad_stands=bad_stands) is not None:
        return False
    # Only force fence-open from the west plant pocket (or just north of wall).
    if player_in_west_plant_pocket(player) or player[1] <= 31:
        return True
    return False


def order_preferred_edges(
    edges: Sequence[Tuple[Tuple[int, int], str]],
    player: Tuple[int, int],
    *,
    water_id_for: Callable[[Tuple[int, int], str], int],
    preferred_ids: frozenset = REFILL_PREFERRED_WATER_TILES,
) -> List[Tuple[Tuple[int, int], str]]:
    """Sort water edges: preferred fill id → main-pond stand rank → distance.

    Main-pond stands from map_config always sort before other preferred water.
    """
    pond_rank = {
        stand: idx for idx, (stand, _face) in enumerate(FARM_MAIN_POND_STANDS)
    }

    def key(edge: Tuple[Tuple[int, int], str]) -> Tuple[int, int, int, int]:
        tile, face = edge
        wid = water_id_for(tile, face)
        pref = 0 if wid in preferred_ids else 1
        if tile in pond_rank:
            band = 0
            rank = pond_rank[tile]
        else:
            band = 1
            rank = 99
        dist = abs(tile[0] - player[0]) + abs(tile[1] - player[1])
        return (pref, band, rank, dist)

    return sorted(edges, key=key)


def is_no_work_reason(reason: Optional[str]) -> bool:
    """True when a crop task SUCCESS is intentional no-op (not crop work)."""
    if not reason:
        return False
    r = reason.strip().lower()
    return r.startswith("no_work") or r.startswith("no_work:")


def crop_completion_status(
    *,
    work_mode: str,
    planted: int,
    watered: int,
    dry_at_start: int,
    refill_exhausted: bool,
    had_seed_stock: bool,
    rain: bool = False,
) -> Tuple[str, str]:
    """Return (status, reason) for CropWaterTask terminal outcome.

    status is ``success`` | ``no_work`` | ``failure``.
    Callers map ``no_work`` → TaskStatus.SUCCESS with a no_work reason so the
    day plan can journal it separately from real crop work.
    """
    mode = (work_mode or "full").strip().lower()
    if rain and mode != "water" and planted == 0 and watered == 0:
        return ("no_work", "no_work: rain; no plant/water work")

    if mode == "water":
        if watered > 0:
            return (
                "success",
                f"planted={planted} watered={watered}",
            )
        if dry_at_start <= 0:
            return ("no_work", "no_work: water-only; no dry crop tiles")
        if refill_exhausted:
            return (
                "failure",
                f"water fail: refill exhausted with dry_crops={dry_at_start}",
            )
        return (
            "failure",
            f"water fail: dry_crops={dry_at_start} watered=0",
        )

    if mode == "establish":
        if planted > 0:
            return ("success", f"planted={planted} watered={watered}")
        if not had_seed_stock:
            return ("no_work", "no_work: no plantable seed stock")
        return ("failure", "establish fail: planted=0 with seed stock")

    # full mode
    if planted > 0 or watered > 0:
        return ("success", f"planted={planted} watered={watered}")
    if dry_at_start > 0 and refill_exhausted:
        return (
            "failure",
            f"crop fail: refill exhausted with dry_crops={dry_at_start}",
        )
    if dry_at_start > 0:
        return ("failure", f"crop fail: dry_crops={dry_at_start} no progress")
    if not had_seed_stock:
        return ("no_work", "no_work: no plots and no plantable seed stock")
    return ("no_work", "no_work: no plots detected")


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
    """Sort key: preferred CheckToolSuccess water → band → Manhattan dist."""
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
    """Fence tiles on the y=31 wall that cut west field off from the main pond."""
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
    """Walkable tiles adjacent to water, as (tile, face_dir) toward water."""
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
    """Closest pond-edge (tile, face_dir), or None."""
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


__all__ = [
    "BAD_REFILL_STAND_BOUNDS",
    "FARM_POND_REFILL_CORRIDOR",
    "MAIN_POND_STAND_BOUNDS",
    "REFILL_BAND_BAD",
    "REFILL_BAND_MID",
    "REFILL_BAND_NORTH",
    "REFILL_BAND_POND",
    "REFILL_BAND_SOUTH",
    "REFILL_NONFILL_WATER_TILES",
    "REFILL_PREFERRED_WATER_TILES",
    "REFILL_WATER_TILES",
    "WATER_TILES",
    "RefillTarget",
    "corridor_needs_fence_open",
    "crop_completion_status",
    "edge_water_tile_id",
    "find_pond_edges",
    "is_bad_refill_stand",
    "is_main_pond_stand",
    "is_no_work_reason",
    "nearest_pond_edge",
    "order_preferred_edges",
    "pond_access_blocking_fences",
    "refill_edge_sort_key",
    "refill_stand_band",
    "select_main_pond_refill",
    "select_staging_stand",
]

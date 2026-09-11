"""L3 occupancy seeds (no emulator).

Unknown cells stay free until a live miss blocks them (OccupancyWalker).
Screenshot 16px tiles over-blocked dest (l3_dest_0x5b_occ: 9735 blocked,
9 misses, timeout at (120,181)). 0x6b seeds only the documented north-door
strand. L3 dest hops spec-declare BLOCK_TILES from cart-WRAM ``$6530`` as
1px Link-coord rectangles (y shifted by LINK_FOOT_OFFSET). 0x5c diamonds
and the 0x5d center plus sit on the walk. That is not a 16px occupancy
grid: BFS stays 1px; inferred misses may still be forgotten, spec cells
stay.
"""

from __future__ import annotations

from typing import Any

from zelda_i.dungeon.tilemap import (
    BLOCK_TILES,
    CELL_PX,
    LINK_FOOT_OFFSET,
    find_cells,
    has_room_tile_map,
)
from zelda_i.level3.geometry import NORTH_DOOR_X, ROOM_6B_STRAND_Y
from zelda_i.walk.physics import DEFAULT_BOUNDS, OccupancyGrid

__all__ = ["link_block_pixels", "room_6b_grid", "seed_block_cells"]

_DOOR_COL_HALF = 8


def room_6b_grid() -> OccupancyGrid:
    """Fresh 0x6b seed. Callers may mutate ``blocked`` on a miss."""
    xmin, xmax, ymin, _ymax = DEFAULT_BOUNDS
    blocked: set[tuple[int, int]] = set()
    x0 = max(xmin, NORTH_DOOR_X - _DOOR_COL_HALF)
    x1 = min(xmax, NORTH_DOOR_X + _DOOR_COL_HALF)
    for y in range(ymin, ROOM_6B_STRAND_Y + 1):
        for x in range(x0, x1 + 1):
            blocked.add((x, y))
    return OccupancyGrid(blocked=blocked)


def link_block_pixels(cx: int, cy: int) -> frozenset[tuple[int, int]]:
    """1px Link-coord occupancy of a 16x16 tilemap cell origin ``(cx, cy)``."""
    y0 = cy - LINK_FOOT_OFFSET
    return frozenset(
        (x, y)
        for x in range(cx, cx + CELL_PX)
        for y in range(y0, y0 + CELL_PX)
    )


def seed_block_cells(grid: OccupancyGrid, ram: Any) -> int:
    """Spec-declare interior BLOCK_TILES. Not inferred; never forgotten.

    Returns the number of pixels added. No-ops when ``ram`` has no cart WRAM.
    """
    if not has_room_tile_map(ram):
        return 0
    before = len(grid.blocked)
    for cx, cy in find_cells(ram, BLOCK_TILES):
        grid.blocked.update(link_block_pixels(cx, cy))
    return len(grid.blocked) - before

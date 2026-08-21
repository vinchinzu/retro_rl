"""Manji 0x6b occupancy seed (no emulator).

Unknown cells stay free until a live miss blocks them (OccupancyWalker).
Screenshot 16px tiles over-blocked dest (l3_dest_0x5b_occ: 9735 blocked,
9 misses, timeout at (120,181)). Seed only the documented strand:
y<=100 on the north-door column. Door cell (120,93) is not 1px-passable.
"""

from __future__ import annotations

from zelda_i.level3_geometry import NORTH_DOOR_X, ROOM_6B_STRAND_Y
from zelda_i.walk_physics import DEFAULT_BOUNDS, OccupancyGrid

__all__ = ["room_6b_grid"]

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

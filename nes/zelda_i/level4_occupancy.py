"""Level 4 mode-9 0x60 occupancy seed (no emulator).

Unknown cells stay free until a live miss blocks them (OccupancyWalker).
Seed only solids from continuous leftover samples:

- west-aisle RIGHT at y=53..157 (v6/v7, v12 y=68/65, v16 y=157)
- south-corridor UP at x=80..168 (v8 x=80..144, v13 x=152, v14 x=160, v15 x=168)
- exit stairs x>=176 (v3/v5)

Island cardinals from the west aisle and south corridor are live-blocked.
"""

from __future__ import annotations

from zelda_i.level4_dungeon import LADDER_60_PICKUP_XY
from zelda_i.walk_physics import OccupancyGrid

__all__ = [
    "ROOM_60_BOUNDS",
    "ROOM_60_EXIT_X",
    "ROOM_60_ISLAND_XY",
    "ROOM_60_NORTH_STRIP_Y",
    "ROOM_60_SPAWN_XY",
    "ROOM_60_WAYPOINTS",
    "ROOM_60_WEST_AISLE_X",
    "room_60_grid",
]

# Spawn leftover after stairs settle; y=69 is already north of DEFAULT ymin=77.
ROOM_60_SPAWN_XY = (48, 69)
ROOM_60_WEST_AISLE_X = 48
ROOM_60_NORTH_STRIP_Y = 68
ROOM_60_ISLAND_XY = LADDER_60_PICKUP_XY
ROOM_60_EXIT_X = 176
ROOM_60_SE_X = 168
# v15: south-corridor UP solid at x=168,189. Try the 7px band y=158 between
# live west-brick y<=157 and south-water y>=165.
ROOM_60_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (48, 158),
    (136, 158),
    ROOM_60_ISLAND_XY,
)
# Include the north strip (y=53..68) that default dungeon bounds drop.
ROOM_60_BOUNDS: tuple[int, int, int, int] = (32, 216, 53, 205)

# Live v6/v7 + v12: RIGHT from x=48 is solid through the north wall face.
_WEST_BRICK_X0, _WEST_BRICK_X1 = 49, 90
_WEST_BRICK_Y0, _WEST_BRICK_Y1 = 53, 157
# Live v8/v13/v14/v15: south corridor UP is solid through x=168 (v15 leftover).
# x=169..175 is the same stairs tile west of exit 176.
_SOUTH_WATER_X0, _SOUTH_WATER_X1 = 80, 175
_SOUTH_WATER_Y0, _SOUTH_WATER_Y1 = 158, 180


def room_60_grid() -> OccupancyGrid:
    """Fresh 0x60 seed. Callers may mutate ``blocked`` on a miss."""
    xmin, xmax, ymin, ymax = ROOM_60_BOUNDS
    blocked: set[tuple[int, int]] = set()
    for y in range(_WEST_BRICK_Y0, _WEST_BRICK_Y1 + 1):
        for x in range(_WEST_BRICK_X0, _WEST_BRICK_X1 + 1):
            blocked.add((x, y))
    for y in range(_SOUTH_WATER_Y0, _SOUTH_WATER_Y1 + 1):
        for x in range(_SOUTH_WATER_X0, _SOUTH_WATER_X1 + 1):
            blocked.add((x, y))
    for y in range(ymin, ymax + 1):
        for x in range(ROOM_60_EXIT_X, xmax + 1):
            blocked.add((x, y))
    return OccupancyGrid(
        blocked=blocked, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
    )

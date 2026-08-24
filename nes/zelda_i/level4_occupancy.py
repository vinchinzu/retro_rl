"""Level 4 mode-9 0x60 occupancy seed (no emulator).

Unknown cells stay free until a live miss blocks them (OccupancyWalker).
Seed only solids from continuous leftover samples:

- west-aisle RIGHT at y=53..157 (v6/v7, v12 y=68/65, v16 y=157)
- south-corridor UP at x=80..168 (v8 x=80..144, v13 x=152, v14 x=160, v15 x=168)
- exit stairs x>=176 (v3/v5)

Island cardinals from the west aisle and south corridor are live-blocked.

Isolated ``l4_tib8_stepladder`` is not a spine path. Live BFS used hold=4 /
q=4 with Keese, then ``em.set_state(goal_state)``. Token replay (v3/v4)
dumps south. Occupancy 1px 4-connected has no spawn→island path. v17/v18
at (48,161) leftover x=48 — the y=158 gap is not an east walk. v19 is
RIGHT+UP at the south-corridor SW water corner (80,189), not v8 cardinal
UP at x=80-144 and not v13-v15 stairs-column UP.
"""

from __future__ import annotations

from zelda_i.level4_dungeon import LADDER_60_PICKUP_XY
from zelda_i.walk_physics import OccupancyGrid

__all__ = [
    "ROOM_60_BOUNDS",
    "ROOM_60_CLIP_BUDGET",
    "ROOM_60_CLIP_BUTTONS",
    "ROOM_60_CLIP_OPEN_X",
    "ROOM_60_CLIP_STAND",
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
# v18 leftover (48,159): RIGHT at the notch is solid. Approach from the
# south corridor (v13 walked y=189) and clip the water SW corner.
ROOM_60_CLIP_STAND = (80, 189)
ROOM_60_CLIP_BUTTONS: tuple[str, str] = ("RIGHT", "UP")
ROOM_60_CLIP_OPEN_X = 90  # east of west-brick → occupancy-free interior
ROOM_60_CLIP_BUDGET = 96
ROOM_60_WAYPOINTS: tuple[tuple[int, int], ...] = (
    ROOM_60_CLIP_STAND,
    ROOM_60_ISLAND_XY,
)
# Include the north strip (y=53..68) that default dungeon bounds drop.
ROOM_60_BOUNDS: tuple[int, int, int, int] = (32, 216, 53, 205)

# Live v6/v7 + v12 + v18: RIGHT from x=48 is solid through y=161 (no y=158 gap).
_WEST_BRICK_X0, _WEST_BRICK_X1 = 49, 90
_WEST_BRICK_Y0, _WEST_BRICK_Y1 = 53, 161
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

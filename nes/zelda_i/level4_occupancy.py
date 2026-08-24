"""Level 4 occupancy seeds (no emulator). 0x60 east-dock + 0x20 H-water + dark 0x21.

Unknown cells stay free until a live miss blocks them (OccupancyWalker).
Solids from leftover samples plus screenshot floor:

- west-aisle RIGHT at y=53..157 (v6/v7, v12 y=68/65, v16 y=157)
- south-corridor UP at x=80..168 (v8 x=80..144, v13 x=152, v14 x=160, v15 x=168)
- exit stairs x>=176 **and** y>=189 (v3/v5). East grey dock is x>=168, y<189.

v26 leftover (48,65) LEFT+UP into north brick. Screenshot spawn/v15/v26 plus
checkpoint geom: grey east dock is the walkway. South-water x1=175 over-blocked
that column (v15 UP at x=168 is stairs; x=175 UP reaches y=151). Causeway
west from dock north ~(176,151) to island (136,141) was never seeded. Isolated
BFS is still not a spine path.

Exit reverse (v34 leftover (136,141)): RIGHT at y=141 to x=175, DOWN the dock,
LEFT along y=189 (never x>=176), UP the west aisle to spawn stairs.
"""

from __future__ import annotations

from zelda_i.level4_dungeon import (
    BOMB_21_NORTH_STAND,
    LADDER_60_PICKUP_XY,
    MAP_21_PICKUP_XY,
    RIGHT_20_STAND,
)
from zelda_i.walk_physics import OccupancyGrid

__all__ = [
    "ROOM_20_BOUNDS",
    "ROOM_20_CLIP_BUDGET",
    "ROOM_20_DOOR_Y_MAX",
    "ROOM_20_EAST_XY",
    "ROOM_20_NORTH_EAST_XY",
    "ROOM_20_NORTH_XY",
    "ROOM_20_SOUTH_EAST_XY",
    "ROOM_20_SOUTH_XY",
    "ROOM_20_SOUTH_Y_MAX",
    "ROOM_20_SPAWN_XY",
    "ROOM_20_WAYPOINTS",
    "ROOM_21_ALCOVE_Y",
    "ROOM_21_ALCOVE_Y_TOL",
    "ROOM_21_BOMB_CORRIDOR_Y",
    "ROOM_21_BOMB_EAST_XY",
    "ROOM_21_BOMB_STAND_XY",
    "ROOM_21_BOMB_WAYPOINTS",
    "ROOM_21_BOUNDS",
    "ROOM_21_CLIP_BUDGET",
    "ROOM_21_CLIP_X",
    "ROOM_21_CORRIDOR_XY",
    "ROOM_21_EAST_XY",
    "ROOM_21_INLAND_XY",
    "ROOM_21_NORTH_STRIP_Y",
    "ROOM_21_SE_X",
    "ROOM_21_SE_Y",
    "ROOM_21_SOUTH_XY",
    "ROOM_21_PICKUP_XY",
    "ROOM_21_SPAWN_XY",
    "ROOM_21_WAYPOINTS",
    "ROOM_60_BOUNDS",
    "ROOM_60_CAUSWAY_XY",
    "ROOM_60_CLIP_BUDGET",
    "ROOM_60_DOCK_MOUTH_X_MIN",
    "ROOM_60_DOCK_MOUTH_XY",
    "ROOM_60_DOCK_NORTH_XY",
    "ROOM_60_EXIT_WAYPOINTS",
    "ROOM_60_EXIT_X",
    "ROOM_60_ISLAND_XY",
    "ROOM_60_NORTH_STRIP_Y",
    "ROOM_60_SOUTH_XY",
    "ROOM_60_SPAWN_XY",
    "ROOM_60_WAYPOINTS",
    "ROOM_60_WEST_AISLE_X",
    "room_20_grid",
    "room_21_grid",
    "room_60_grid",
]

# Spawn leftover after stairs settle; y=69 is already north of DEFAULT ymin=77.
ROOM_60_SPAWN_XY = (48, 69)
ROOM_60_WEST_AISLE_X = 48
ROOM_60_NORTH_STRIP_Y = 68
ROOM_60_ISLAND_XY = LADDER_60_PICKUP_XY
ROOM_60_EXIT_X = 176
ROOM_60_SOUTH_XY = (48, 189)
# Live geom: UP at x=175 y=189 walks the east grey dock to ~(176,151).
ROOM_60_DOCK_MOUTH_XY = (175, 189)
ROOM_60_DOCK_MOUTH_X_MIN = 175
ROOM_60_DOCK_NORTH_XY = (176, 151)
ROOM_60_CLIP_BUDGET = 96
ROOM_60_WAYPOINTS: tuple[tuple[int, int], ...] = (
    ROOM_60_SOUTH_XY,
    ROOM_60_DOCK_MOUTH_XY,
    ROOM_60_DOCK_NORTH_XY,
    ROOM_60_ISLAND_XY,
)
# Reverse inbound dock. Stay on island y=141; DOWN at x=175 not x>=176.
ROOM_60_CAUSWAY_XY = (ROOM_60_DOCK_MOUTH_X_MIN, ROOM_60_ISLAND_XY[1])
ROOM_60_EXIT_WAYPOINTS: tuple[tuple[int, int], ...] = (
    ROOM_60_CAUSWAY_XY,
    ROOM_60_DOCK_MOUTH_XY,
    ROOM_60_SOUTH_XY,
    ROOM_60_SPAWN_XY,
)
# Include the north strip (y=53..68) that default dungeon bounds drop.
ROOM_60_BOUNDS: tuple[int, int, int, int] = (32, 216, 53, 205)

# Live v6/v7 + v12 + v18: RIGHT from x=48 is solid through y=161 (no y=158 gap).
_WEST_BRICK_X0, _WEST_BRICK_X1 = 49, 90
_WEST_BRICK_Y0, _WEST_BRICK_Y1 = 53, 161
# Live v8/v13/v14/v15: south corridor UP is solid through x=168.
# y=181..188 is the same water/stairs lip (1px occupancy gap was fake).
# x>=175 is the east grey dock (geom x=175 UP from y=189).
_SOUTH_WATER_X0, _SOUTH_WATER_X1 = 80, 174
_SOUTH_WATER_Y0, _SOUTH_WATER_Y1 = 158, 188
# Live v20: DOWN from (84,189) y unchanged. Seed the south brick at leftover x.
_SOUTH_WALL_X0, _SOUTH_WALL_X1 = 84, 88
_SOUTH_WALL_Y0 = 190


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
    for y in range(_SOUTH_WALL_Y0, ymax + 1):
        for x in range(_SOUTH_WALL_X0, _SOUTH_WALL_X1 + 1):
            blocked.add((x, y))
    # Exit stairs mouth only (v3/v5). East dock x>=176 y<189 stays free.
    for y in range(189, ymax + 1):
        for x in range(ROOM_60_EXIT_X, xmax + 1):
            blocked.add((x, y))
    return OccupancyGrid(
        blocked=blocked, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
    )


# 0x20 leftover (120,205) after KEY-UP. Screenshot H-water (v1–v3 PNGs):
# 16px tiles. v1 RIGHT at (120,141) is the H-bar; v2 (120,133) still on it;
# v3 RIGHT at (120,205) is the south door frame. South gold y>=192 (water
# ends y=191). v20 DOWN at x=200 is solid at y=109 (16px spine, not PNG gold).
ROOM_20_SPAWN_XY = (120, 205)
ROOM_20_SOUTH_XY = (120, 192)
# v5 exact y=192 yo-yo (stall=0). Gold strip is y=192–204; stay off door lip.
ROOM_20_SOUTH_Y_MAX = 200
# v7 RIGHT at y=199 in the door column is solid. v8 y>192 UP yo-yoed at 193.
ROOM_20_DOOR_Y_MAX = 196
ROOM_20_EAST_XY = RIGHT_20_STAND
ROOM_20_SOUTH_EAST_XY = (ROOM_20_EAST_XY[0], ROOM_20_SOUTH_XY[1])
# v1/v2 RIGHT at y=141/133 is still on the H. Empty-room PNG: y=88–110 gold
# north of the top arms (y=112–127). Ladder-cross the H-bar (v1 reached 141).
ROOM_20_NORTH_XY = (120, 96)
# v13 leftover: RIGHT at y=96 is the east wall (x=208 unreachable).
ROOM_20_NORTH_EAST_XY = (200, ROOM_20_NORTH_XY[1])
ROOM_20_CLIP_BUDGET = 96
# v20: north-around to (200,96) then DOWN solid at (200,109). East door
# is a RIGHT+DOWN clip, not 4-connected occupancy.
ROOM_20_WAYPOINTS: tuple[tuple[int, int], ...] = (
    ROOM_20_NORTH_XY,
    ROOM_20_NORTH_EAST_XY,
    ROOM_20_EAST_XY,
)
ROOM_20_BOUNDS: tuple[int, int, int, int] = (40, 216, 77, 205)

_H20_SPINE_Y0, _H20_SPINE_Y1 = 112, 191
_H20_LEFT_SPINE_X0, _H20_LEFT_SPINE_X1 = 48, 63
# v20 live: DOWN at x=200 hits y=109. 16px spine, PNG gold is the tile edge.
_H20_RIGHT_SPINE_X0, _H20_RIGHT_SPINE_X1 = 192, 207
_H20_RIGHT_SPINE_Y0 = 110
_H20_LEFT_ARM_X1 = 95
_H20_RIGHT_ARM_X0 = 160
_H20_TOP_ARM_Y0, _H20_TOP_ARM_Y1 = 112, 127
_H20_BOT_ARM_Y0, _H20_BOT_ARM_Y1 = 176, 191
_H20_BAR_X0, _H20_BAR_X1 = 64, 175
_H20_BAR_Y0, _H20_BAR_Y1 = 144, 159
# v3: L/R at y=205 is door-frame solid. Keep the N-S door column.
_H20_DOOR_Y0 = 193
_H20_DOOR_X0, _H20_DOOR_X1 = 112, 128


def _block_rect(
    blocked: set[tuple[int, int]], x0: int, x1: int, y0: int, y1: int
) -> None:
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            blocked.add((x, y))


def room_20_grid() -> OccupancyGrid:
    """Fresh 0x20 H-water seed. Callers may mutate ``blocked`` on a miss."""
    xmin, xmax, ymin, ymax = ROOM_20_BOUNDS
    blocked: set[tuple[int, int]] = set()
    _block_rect(
        blocked,
        _H20_LEFT_SPINE_X0,
        _H20_LEFT_SPINE_X1,
        _H20_SPINE_Y0,
        _H20_SPINE_Y1,
    )
    _block_rect(
        blocked,
        _H20_RIGHT_SPINE_X0,
        _H20_RIGHT_SPINE_X1,
        _H20_RIGHT_SPINE_Y0,
        _H20_SPINE_Y1,
    )
    _block_rect(
        blocked,
        _H20_LEFT_SPINE_X1 + 1,
        _H20_LEFT_ARM_X1,
        _H20_TOP_ARM_Y0,
        _H20_TOP_ARM_Y1,
    )
    _block_rect(
        blocked,
        _H20_LEFT_SPINE_X1 + 1,
        _H20_LEFT_ARM_X1,
        _H20_BOT_ARM_Y0,
        _H20_BOT_ARM_Y1,
    )
    _block_rect(
        blocked,
        _H20_RIGHT_ARM_X0,
        _H20_RIGHT_SPINE_X0 - 1,
        _H20_TOP_ARM_Y0,
        _H20_TOP_ARM_Y1,
    )
    _block_rect(
        blocked,
        _H20_RIGHT_ARM_X0,
        _H20_RIGHT_SPINE_X0 - 1,
        _H20_BOT_ARM_Y0,
        _H20_BOT_ARM_Y1,
    )
    _block_rect(blocked, _H20_BAR_X0, _H20_BAR_X1, _H20_BAR_Y0, _H20_BAR_Y1)
    # v13: RIGHT at (200,96) is the east wall. Corridor x=208 only y≈128–175.
    _block_rect(blocked, 201, xmax, ymin, 127)
    # v1 UP at x=120 crossed the 1-tile H-bar with ADDR_LADDER.
    for y in range(_H20_BAR_Y0, _H20_BAR_Y1 + 1):
        for x in range(112, 129):
            blocked.discard((x, y))
    for y in range(_H20_DOOR_Y0, ymax):
        for x in range(xmin, xmax + 1):
            if x < _H20_DOOR_X0 or x > _H20_DOOR_X1:
                blocked.add((x, y))
    # v3 leftover: RIGHT at y=205 is the south lip (door column included).
    for x in range(xmin, xmax + 1):
        blocked.add((x, ymax))
    return OccupancyGrid(
        blocked=blocked, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
    )


# Dark 0x21 leftover (16,141). PNG is black (no candle). Seed playfield +
# west-door leftover. Interior unknown=free until a live miss. Isolated
# leftover PNGs are also dark — maze gold vs floor is not visible east of
# the west door, so do not paint x=49 as the whole room.
# Pickup ~(208,181) from isolated ADDR_MAP. Isolated MAP_21_SAMPLE_PATH
# is state-BFS, not this tape.
# v1 leftover (48,141): inland RIGHT works; UP is maze.
# v2 leftover (48,141): cardinal RIGHT also solid.
# v3 leftover (48,141): RIGHT+UP clip no-ops (two-wall NE corner).
# v4 leftover (48,141): DOWN also solid. N/E/S boxed; only west returns.
# v5 leftover (48,141): RIGHT+DOWN clip no-ops. (48,141) is a west-door
# pocket — turn UP at x=32 before overshooting.
# v6 leftover (48,117): UP at x=32 works; RIGHT along y=117 is maze.
# v7 leftover (48,101): RIGHT at y=96-101 still the x=49 wall.
# v8 leftover (32,93): UP is north wall of the west column.
# v9 leftover (32,100): RIGHT+UP yo-yos (timeout stall=0). East sealed.
# v10 leftover (48,173): DOWN at x=32 works; RIGHT at y=173 still x=49 wall.
# v11 leftover (48,189): RIGHT at south band still x=49 wall.
# v12 leftover (32,189): RIGHT+DOWN clip at SE corner.
# v13 leftover (48,125): spawn RIGHT+UP reaches (40,125) then 0x31 x>=40
# off-band exit cardinal-RIGHTs into the wall. y=125 is new; still x=49.
# v14 leftover (48,109): hold RIGHT+UP to x=48 lands (48,93) north strip;
# PATH DOWN to (48,109) then cardinal RIGHT is still x=49.
# v15 2/2: RIGHT+DOWN from (48,93) clips east; ADDR_MAP|0x08 at (208,181)
# in 297f (v15b same). Cardinal RIGHT at y=109 is still the wall; the clip
# is two-button, not 4-connected occupancy. Mid-wall seed stays.
ROOM_21_SPAWN_XY = (16, 141)
ROOM_21_ALCOVE_Y = 141
ROOM_21_ALCOVE_Y_TOL = 8
ROOM_21_CLIP_X = 48
ROOM_21_NORTH_STRIP_Y = 113
ROOM_21_SE_X = 64
ROOM_21_SE_Y = 125
# v14 PATH target; still the vestibule east face, not maze inland.
ROOM_21_INLAND_XY = (48, 109)
ROOM_21_CORRIDOR_XY = (32, 189)
ROOM_21_EAST_XY = (208, 189)
ROOM_21_SOUTH_XY = (48, 181)
ROOM_21_PICKUP_XY = MAP_21_PICKUP_XY
ROOM_21_CLIP_BUDGET = 96
ROOM_21_WAYPOINTS: tuple[tuple[int, int], ...] = (
    ROOM_21_EAST_XY,
    ROOM_21_PICKUP_XY,
)
# Reverse of v15 inbound: spawn RIGHT+UP lands y=93; bomb v1 LEFT at
# (192,109) is a 16px pillar. North-around y=93 then LEFT, not y=109.
ROOM_21_BOMB_CORRIDOR_Y = 93
ROOM_21_BOMB_EAST_XY = (ROOM_21_PICKUP_XY[0], ROOM_21_BOMB_CORRIDOR_Y)
ROOM_21_BOMB_STAND_XY = BOMB_21_NORTH_STAND
ROOM_21_BOMB_WAYPOINTS: tuple[tuple[int, int], ...] = (
    ROOM_21_BOMB_EAST_XY,
    ROOM_21_BOMB_STAND_XY,
)
ROOM_21_BOUNDS: tuple[int, int, int, int] = (16, 216, 77, 205)
# v1: UP at (48,141). 16px tile north of the west door-row.
_H21_WEST_WALL_X0, _H21_WEST_WALL_X1 = 40, 55
_H21_WEST_WALL_Y0, _H21_WEST_WALL_Y1 = 125, 140
# v2: RIGHT at (48,141) is the east face of that pocket.
_H21_WEST_WALL_EAST_X = 49
# v4: DOWN at (48,141).
_H21_WEST_WALL_SOUTH_Y = 142
# v6: RIGHT at (48,117) is maze. 16px tile east of the x=32 north column.
_H21_MID_WALL_X0, _H21_MID_WALL_X1 = 49, 63
_H21_MID_WALL_Y0, _H21_MID_WALL_Y1 = 96, 173
# bomb v1: LEFT at (192,109) solid. 16px pillar on the y=109 band.
_H21_PILLAR_X0, _H21_PILLAR_X1 = 176, 191
_H21_PILLAR_Y0, _H21_PILLAR_Y1 = 96, 111


def room_21_grid() -> OccupancyGrid:
    """Fresh 0x21 seed. Dark leftover + v1 UP miss."""
    xmin, xmax, ymin, ymax = ROOM_21_BOUNDS
    blocked: set[tuple[int, int]] = set()
    _block_rect(
        blocked,
        _H21_WEST_WALL_X0,
        _H21_WEST_WALL_X1,
        _H21_WEST_WALL_Y0,
        _H21_WEST_WALL_Y1,
    )
    for y in range(_H21_WEST_WALL_Y0, 142):
        blocked.add((_H21_WEST_WALL_EAST_X, y))
    for x in range(_H21_WEST_WALL_X0, _H21_WEST_WALL_X1 + 1):
        blocked.add((x, _H21_WEST_WALL_SOUTH_Y))
    _block_rect(
        blocked,
        _H21_MID_WALL_X0,
        _H21_MID_WALL_X1,
        _H21_MID_WALL_Y0,
        _H21_MID_WALL_Y1,
    )
    _block_rect(
        blocked,
        _H21_PILLAR_X0,
        _H21_PILLAR_X1,
        _H21_PILLAR_Y0,
        _H21_PILLAR_Y1,
    )
    return OccupancyGrid(
        blocked=blocked, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
    )

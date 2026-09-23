"""Read-only reader for the live room tile map in cart WRAM.

`env.get_ram()` returns 10240 bytes: index `0..0x7FF` is CPU `$0000-$07FF`
and index `0x800..` is cart WRAM `$6000-$7FFF`. The tile map the engine
collides against for the **current** room lives at `$6530`, column-major,
32 columns x 22 rows of 8x8 tiles (stride 22).

Screen coords: tile `(col, row)` covers `x = col*8`, `y = 64 + row*8`.
A 16x16 room cell is a 2x2 tile quad; a dungeon room interior is the 12x7
grid of cells at `x = 32..208`, `y = 96..192`.

Object coords are screen coords (a `0x68` block at `(192, 144)` is the quad
at cols 24-25 / rows 10-11). Link's stored `y` sits `LINK_FOOT_OFFSET` px
above his colliding row, which is why UP from `(144, 141)` parks at `y=117`
rather than `y=125`.

This module never writes. It is the measurement tool that replaced the
direction-sensitive `$049E` `colliding_tile` sweeps (which report the tile
Link is walking *into*, not the tile he is standing on).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ADDR_ROOM_TILE_MAP",
    "BLOCK_TILES",
    "BOMB_HOLE_TILES",
    "CELL_PX",
    "DOORWAY_VOID_TILE",
    "DOORWAY_ART_TILES",
    "DOOR_LEAF_TILES",
    "FLOOR_TILES",
    "INTERIOR_X",
    "INTERIOR_Y",
    "LINK_FOOT_OFFSET",
    "LINK_WALKABLE_TILES",
    "LOCKED_DOOR_TILES",
    "PLAYFIELD_TOP_Y",
    "RING_CELLS",
    "SHUTTER_TILES",
    "STAIR_TILES",
    "TILE_COLS",
    "TILE_PX",
    "TILE_ROWS",
    "WRAM_BASE",
    "WRAM_RAM_OFFSET",
    "ascii_room",
    "blocked_link_cells",
    "PLAYFIELD_X",
    "PLAYFIELD_Y",
    "cell_tiles",
    "door_cells",
    "find_cells",
    "has_room_tile_map",
    "link_cell",
    "read_room_tiles",
    "stair_cells",
    "tile_at",
    "tile_at_screen",
    "ADDR_FIRST_UNWALKABLE",
    "OW_LATTICE_X",
    "OW_LATTICE_Y",
    "OW_WALKABLE_EXTRA",
    "ow_walkable_nodes",
]

WRAM_BASE = 0x6000
WRAM_RAM_OFFSET = 0x0800  # ram[WRAM_RAM_OFFSET] == $6000
ADDR_ROOM_TILE_MAP = 0x6530
TILE_COLS = 32
TILE_ROWS = 22
TILE_PX = 8
CELL_PX = 16
PLAYFIELD_TOP_Y = 64
# Link's stored y is this many px above the top of the row he collides with.
LINK_FOOT_OFFSET = 11
# Dungeon room interior, in 16x16 cell origins (screen coords).
INTERIOR_X = tuple(range(32, 209, CELL_PX))
INTERIOR_Y = tuple(range(96, 193, CELL_PX))
# Whole playfield, one ring wider than the interior: the four door cells sit
# at x=16 / x=224 and y=80 / y=208.
PLAYFIELD_X = tuple(range(16, 225, CELL_PX))
PLAYFIELD_Y = tuple(range(80, 209, CELL_PX))
# The one-cell wall ring around the interior. Every doorway of every kind is
# on it, and nothing else is: the sweep found doorway art at eight ring
# positions and zero interior ones.
RING_CELLS = tuple(
    (x, y)
    for y in PLAYFIELD_Y
    for x in PLAYFIELD_X
    if x not in INTERIOR_X or y not in INTERIOR_Y
)

FLOOR_TILES = frozenset({0x74, 0x75, 0x76, 0x77})
BLOCK_TILES = frozenset({0xB0, 0xB1, 0xB2, 0xB3})
STAIR_TILES = frozenset({0x70, 0x71, 0x72, 0x73})
# Doorway art, measured off all 259 dungeon fixtures in
# ``custom_integrations`` (2026-09-14 sweep). There is no "door tile" for an
# *open* doorway: its mouth is plain ``FLOOR_TILES`` set into the wall ring,
# so it is already walkable under ``FLOOR_TILES`` alone.
#
# A bombed-open wall is different art: one tile pair sitting in the wall ring
# itself, with ``DOORWAY_VOID_TILE`` beyond it. One pair per direction, and
# they appear at exactly eight fixed ring positions across the whole sweep --
# never inside a room -- so grading them walkable cannot open an interior
# cell. ``DOOR_TILES`` used to hold the west pair alone under a name that
# claimed to cover all four, which graded every north, south and east bombed
# passage SOLID (29 of the 259 rooms have one; L4 ``0x10``, L5 ``0x07``,
# L9 ``0x03``/``0x05`` are east, L8 ``0x3e`` is north).
BOMB_HOLE_TILES = frozenset({0x8C, 0x8D, 0x8E, 0x8F, 0x90, 0x91, 0x92, 0x93})
# A *closed* door leaf. Locked doors are direction-specific art in map order
# N/S/W/E; shutters have one vertical pair and one horizontal pair. Both are
# solid while shut and become an open mouth (floor) when they open, so
# neither belongs in LINK_WALKABLE_TILES -- they are listed to name what the
# walker is refusing, not to change it.
LOCKED_DOOR_TILES = frozenset(range(0x98, 0xA8))
SHUTTER_TILES = frozenset(range(0xA8, 0xB0))
DOOR_LEAF_TILES = LOCKED_DOOR_TILES | SHUTTER_TILES
# Every tile that is doorway art rather than floor or wall.
DOORWAY_ART_TILES = DOOR_LEAF_TILES | BOMB_HOLE_TILES
# The black void past a doorway mouth. Never walkable: the scroll is a held
# direction at the mouth, not a BFS goal one cell further out.
DOORWAY_VOID_TILE = 0x24
# What Link may stand on. Stairs and bombed passages are floor he walks over,
# not geometry: a CheckWarp staircase is the *destination* of a route, and a
# bomb hole at x=16/232 or y=80/216 is the only way out of the rooms that
# have one. Grading them SOLID walls off exactly the cells ROUTE_BOUNDS
# exists to include, and every stair a room is routed to.
LINK_WALKABLE_TILES = FLOOR_TILES | STAIR_TILES | BOMB_HOLE_TILES

_MAP_LEN = TILE_COLS * TILE_ROWS
_MIN_RAM = WRAM_RAM_OFFSET + ADDR_ROOM_TILE_MAP - WRAM_BASE + _MAP_LEN


def has_room_tile_map(ram: np.ndarray) -> bool:
    """True when ``ram`` carries the cart-WRAM window holding the map."""
    return int(len(ram)) >= _MIN_RAM


def _require(ram: np.ndarray) -> None:
    if not has_room_tile_map(ram):
        raise ValueError(
            f"ram of {len(ram)} bytes has no cart WRAM window; "
            f"need >= {_MIN_RAM} (use env.get_ram())"
        )


def read_room_tiles(ram: np.ndarray) -> np.ndarray:
    """Current room's tile map as a ``(TILE_ROWS, TILE_COLS)`` uint8 array."""
    _require(ram)
    start = WRAM_RAM_OFFSET + ADDR_ROOM_TILE_MAP - WRAM_BASE
    flat = np.asarray(ram[start : start + _MAP_LEN], dtype=np.uint8)
    # Stored column-major with stride TILE_ROWS.
    return flat.reshape(TILE_COLS, TILE_ROWS).T.copy()


def tile_at(ram: np.ndarray, col: int, row: int) -> int:
    """One 8x8 tile id by map index."""
    _require(ram)
    if not (0 <= col < TILE_COLS and 0 <= row < TILE_ROWS):
        raise IndexError(f"tile ({col}, {row}) outside {TILE_COLS}x{TILE_ROWS}")
    idx = WRAM_RAM_OFFSET + ADDR_ROOM_TILE_MAP - WRAM_BASE + col * TILE_ROWS + row
    return int(ram[idx])


def tile_at_screen(ram: np.ndarray, x: int, y: int) -> int:
    """One 8x8 tile id by screen pixel."""
    return tile_at(ram, int(x) // TILE_PX, (int(y) - PLAYFIELD_TOP_Y) // TILE_PX)


def cell_tiles(ram: np.ndarray, x: int, y: int) -> tuple[int, int, int, int]:
    """The 2x2 tile quad of the 16x16 cell whose origin is ``(x, y)``."""
    return (
        tile_at_screen(ram, x, y),
        tile_at_screen(ram, x + TILE_PX, y),
        tile_at_screen(ram, x, y + TILE_PX),
        tile_at_screen(ram, x + TILE_PX, y + TILE_PX),
    )


def find_cells(
    ram: np.ndarray, kinds: frozenset[int]
) -> tuple[tuple[int, int], ...]:
    """Interior 16x16 cell origins whose whole quad is in ``kinds``."""
    return tuple(
        (x, y)
        for y in INTERIOR_Y
        for x in INTERIOR_X
        if all(t in kinds for t in cell_tiles(ram, x, y))
    )


def stair_cells(ram: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Interior cell origins that are a CheckWarp staircase."""
    return find_cells(ram, STAIR_TILES)


def door_cells(ram: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Ring cell origins Link can walk *out* through.

    Walking into one leaves the room, so path legs that hug the west or east
    wall must avoid the door row (L7 `0x0D`: west bomb hole at `(16, 144)`).

    Two kinds qualify, and they are the two kinds the walker can stand on:
    an open doorway, whose mouth is floor set into the wall ring, and a
    bombed-open wall (`BOMB_HOLE_TILES`). A *closed* leaf is not a door cell
    -- it is solid, the walker already refuses it, and walking into it does
    not leave the room. A north or south doorway spans two ring cells, so it
    reports both.

    Only the ring is searched. The interior is floor by definition, so
    scanning the whole playfield -- which is what this did while its tile set
    was the west bomb hole alone and so never matched inside -- would report
    every open cell in the room.
    """
    walk_out = FLOOR_TILES | BOMB_HOLE_TILES
    return tuple(
        (x, y)
        for (x, y) in RING_CELLS
        if any(t in walk_out for t in cell_tiles(ram, x, y))
    )


def link_cell(link_x: int, link_y: int) -> tuple[int, int]:
    """The 16x16 cell origin Link's feet occupy, in screen coords."""
    y = int(link_y) + LINK_FOOT_OFFSET
    return (int(link_x) // CELL_PX * CELL_PX, y // CELL_PX * CELL_PX)


def ascii_room(ram: np.ndarray) -> str:
    """One line per interior cell row: ``.`` floor ``#`` block ``S`` stairs."""
    lines = ["room " + " ".join(f"{x:3d}" for x in INTERIOR_X)]
    for y in INTERIOR_Y:
        marks = []
        for x in INTERIOR_X:
            quad = cell_tiles(ram, x, y)
            if all(t in FLOOR_TILES for t in quad):
                marks.append("  .")
            elif all(t in BLOCK_TILES for t in quad):
                marks.append("  #")
            elif all(t in STAIR_TILES for t in quad):
                marks.append("  S")
            elif any(t in DOORWAY_ART_TILES for t in quad):
                marks.append("  D")
            else:
                marks.append("  ?")
        lines.append(f"y{y:03d} " + " ".join(marks))
    return "\n".join(lines)


def blocked_link_cells(
    ram: np.ndarray,
    bounds: tuple[int, int, int, int],
    *,
    walkable: frozenset[int] = LINK_WALKABLE_TILES,
) -> frozenset[tuple[int, int]]:
    """Occupancy blocks for ``bounds``, measured from the live tile map.

    Occupancy is keyed on Link's stored ``(x, y)`` while his feet collide
    ``LINK_FOOT_OFFSET`` px lower, so cell ``(x, y)`` is solid exactly when
    the tile under ``(x, y + LINK_FOOT_OFFSET)`` is not walkable.

    ``walkable`` defaults to ``LINK_WALKABLE_TILES`` — floor plus stairs plus
    door mouths — not bare ``FLOOR_TILES``. Grading a staircase or a door as
    solid walls off the cells a route is aimed *at*.

    This replaces hand-written ``occupancy_blocked`` range boxes, which are
    written from a screenshot and drift: the L1 ``0x23`` list walled 84 cells
    of real floor, and the walker burned 2438 frames learning and forgetting
    them (3994 misses, 1335 forgets) instead of reaching the key.
    """
    _require(ram)
    xmin, xmax, ymin, ymax = (int(v) for v in bounds)
    if xmin > xmax or ymin > ymax:
        return frozenset()
    tiles = read_room_tiles(ram)
    solid = ~np.isin(tiles, np.fromiter(walkable, dtype=np.uint8))
    xs = np.arange(xmin, xmax + 1)
    ys = np.arange(ymin, ymax + 1)
    cols = np.clip(xs // TILE_PX, 0, TILE_COLS - 1)
    rows = np.clip(
        (ys + LINK_FOOT_OFFSET - PLAYFIELD_TOP_Y) // TILE_PX, 0, TILE_ROWS - 1
    )
    hits = solid[np.ix_(rows, cols)]
    yy, xx = np.nonzero(hits)
    return frozenset(zip(xs[xx].tolist(), ys[yy].tolist()))


# ------------------------------------------------ overworld lattice ---
# ``Z_07.asm`` ``GetCollidingTileMoving``: Link collides on his feet row
# (``y + $0B``) and, moving vertically, on the column at ``x + 8`` as well.
# He only turns on the 8 px grid (x % 8 == 0 to go vertical, y % 8 == 5 to go
# horizontal; measured on ``OW_39``: off-grid, a perpendicular press first
# slides him to the nearest grid line). So a walkable overworld position is a
# lattice node whose two feet tiles pass the ROM test, and every straight run
# between two walkable neighbours is one the ROM lets him walk.
ADDR_FIRST_UNWALKABLE = 0x034A  # ObjectFirstUnwalkableTile; $89 on the OW
# ``WalkableTiles``: OW tiles past the threshold that ``GetCollidableTile``
# rewrites to $26 before the test.
OW_WALKABLE_EXTRA = frozenset({0x8D, 0x91, 0x9C, 0xAC, 0xAD, 0xCC, 0xD2, 0xD5, 0xDF})
OW_LATTICE_X = tuple(range(0, 241, TILE_PX))
OW_LATTICE_Y = tuple(range(61, 222, TILE_PX))  # $3D (north edge) .. $DD (south)


def ow_walkable_nodes(ram: np.ndarray, *, overworld: bool = True) -> frozenset[tuple[int, int]]:
    """Lattice nodes ``(x, y)`` Link can stand on, from ``$6530``.

    The same collision runs in a dungeon with ``$034A`` at ``$78``; only the
    ``WalkableTiles`` rewrite is overworld-only (``CurLevel`` skips it).
    """
    _require(ram)
    tiles = read_room_tiles(ram)
    first = int(ram[ADDR_FIRST_UNWALKABLE]) or (0x89 if overworld else 0x78)
    ok = tiles < first
    if overworld:
        ok = ok | np.isin(tiles, np.fromiter(OW_WALKABLE_EXTRA, dtype=np.uint8))
    nodes: set[tuple[int, int]] = set()
    for y in OW_LATTICE_Y:
        row = (y + LINK_FOOT_OFFSET - PLAYFIELD_TOP_Y) // TILE_PX
        if not 0 <= row < TILE_ROWS:
            continue
        for x in OW_LATTICE_X:
            col = x // TILE_PX
            if ok[row, col] and (col + 1 >= TILE_COLS or ok[row, col + 1]):
                if overworld or (_in_dungeon_lane(x, y) and not _head_in_water(tiles, x, y)):
                    nodes.add((x, y))
    return frozenset(nodes)


# Dungeon interior on the lattice; outside it only the door lanes are floor.
# The door mouth's tiles read walkable a row either side of the lane, and a
# route along y=133 into the 0x74 west wall pressed LEFT for 3000 frames.
_DUNGEON_X = (32, 208)
_DUNGEON_Y = (85, 189)
_DOOR_LANE_X = 120
_DOOR_LANE_Y = 141


# Dungeon moat/water. The ROM collision reads Link's feet row only, but a
# body over water with the stepladder owned deploys it, and on the ladder
# sideways input is locked (L5 0x26: LEFT at (48,181) forever, head in the
# moat's bottom row).
DUNGEON_WATER_TILES = frozenset({0xF4})
LINK_HEAD_OFFSET = 3


def _head_in_water(tiles: np.ndarray, x: int, y: int) -> bool:
    row = (y + LINK_HEAD_OFFSET - PLAYFIELD_TOP_Y) // TILE_PX
    if not 0 <= row < TILE_ROWS:
        return False
    col = x // TILE_PX
    cols = (col, col + 1) if col + 1 < TILE_COLS else (col,)
    return any(int(tiles[row, c]) in DUNGEON_WATER_TILES for c in cols)


def _in_dungeon_lane(x: int, y: int) -> bool:
    inside_x = _DUNGEON_X[0] <= x <= _DUNGEON_X[1]
    inside_y = _DUNGEON_Y[0] <= y <= _DUNGEON_Y[1]
    if inside_x and inside_y:
        return True
    if not inside_x and not inside_y:
        return False
    return y == _DOOR_LANE_Y if not inside_x else x == _DOOR_LANE_X

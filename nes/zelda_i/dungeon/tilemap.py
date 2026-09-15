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
    "CELL_PX",
    "DOOR_TILES",
    "FLOOR_TILES",
    "INTERIOR_X",
    "INTERIOR_Y",
    "LINK_FOOT_OFFSET",
    "LINK_WALKABLE_TILES",
    "PLAYFIELD_TOP_Y",
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

FLOOR_TILES = frozenset({0x74, 0x75, 0x76, 0x77})
BLOCK_TILES = frozenset({0xB0, 0xB1, 0xB2, 0xB3})
STAIR_TILES = frozenset({0x70, 0x71, 0x72, 0x73})
DOOR_TILES = frozenset({0x90, 0x91})
# What Link may stand on. Stairs and door mouths are floor he walks over,
# not geometry: a CheckWarp staircase is the *destination* of a route, and
# the four door cells at x=16/224 and y=80/208 are the only way in or out
# of a room. Grading them SOLID walls off exactly the cells ROUTE_BOUNDS
# exists to include, and every stair a room is routed to.
LINK_WALKABLE_TILES = FLOOR_TILES | STAIR_TILES | DOOR_TILES

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
    """Playfield cell origins holding a doorway; these sit outside the interior.

    Walking into one leaves the room, so path legs that hug the west or east
    wall must avoid the door row (L7 `0x0D`: west door at `(16, 144)`).
    """
    return tuple(
        (x, y)
        for y in PLAYFIELD_Y
        for x in PLAYFIELD_X
        if any(t in DOOR_TILES for t in cell_tiles(ram, x, y))
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
            elif any(t in DOOR_TILES for t in quad):
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

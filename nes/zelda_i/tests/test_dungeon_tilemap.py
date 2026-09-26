"""Cart-WRAM `$6530` room tile map reader (read-only).

Geometry is calibrated against the live L7 play `0x0D` pin: the room is a
ring whose only crossings of the `y=112` / `y=176` solid bands are `x=32`
and `x=208`, and the `block_stairs` push reveals a staircase at `(208,96)`.
"""

from __future__ import annotations

import numpy as np
import pytest

from zelda_i.dungeon.tilemap import (
    ADDR_ROOM_TILE_MAP,
    BLOCK_TILES,
    BOMB_HOLE_TILES,
    DOOR_LEAF_TILES,
    FLOOR_TILES,
    INTERIOR_X,
    INTERIOR_Y,
    LINK_FOOT_OFFSET,
    LINK_WALKABLE_TILES,
    LOCKED_DOOR_TILES,
    PLAYFIELD_X,
    PLAYFIELD_Y,
    RING_CELLS,
    SHUTTER_TILES,
    STAIR_TILES,
    TILE_COLS,
    TILE_ROWS,
    WRAM_BASE,
    WRAM_RAM_OFFSET,
    ascii_room,
    cell_tiles,
    door_cells,
    find_cells,
    has_room_tile_map,
    link_cell,
    read_room_tiles,
    room_door_types,
    stair_cells,
    tile_at,
    tile_at_screen,
)
from zelda_i.tests.ram_helpers import room_tile_ram

_FLOOR_QUAD = (0x74, 0x76, 0x75, 0x77)
_BLOCK_QUAD = (0xB0, 0xB2, 0xB1, 0xB3)
_STAIR_QUAD = (0x70, 0x72, 0x71, 0x73)
# The measured L7 0x0D west bombed-open wall: the hole pair sits in the
# wall column itself, with the void beyond it.
_WEST_HOLE_QUAD = (0x90, 0x24, 0x91, 0x24)


def _blank_ram() -> np.ndarray:
    return np.zeros(10240, dtype=np.uint8)


def _set_cell(ram: np.ndarray, x: int, y: int, quad) -> None:
    """Write a 2x2 tile quad at the 16x16 cell whose origin is (x, y)."""
    base = WRAM_RAM_OFFSET + ADDR_ROOM_TILE_MAP - WRAM_BASE
    col, row = x // 8, (y - 64) // 8
    for dc, dr, value in (
        (0, 0, quad[0]), (1, 0, quad[1]), (0, 1, quad[2]), (1, 1, quad[3])
    ):
        ram[base + (col + dc) * TILE_ROWS + (row + dr)] = value


def _room_0d_ram(*, pushed: bool) -> np.ndarray:
    """The measured L7 0x0D map, before or after the RIGHT push."""
    ram = _blank_ram()
    for y in INTERIOR_Y:
        for x in INTERIOR_X:
            _set_cell(ram, x, y, _FLOOR_QUAD)
    for y in (112, 176):
        for x in INTERIOR_X:
            if x not in (32, 208):
                _set_cell(ram, x, y, _BLOCK_QUAD)
    for y in (128, 144, 160):
        _set_cell(ram, 192, y, _BLOCK_QUAD)
    _set_cell(ram, 16, 144, _WEST_HOLE_QUAD)
    if pushed:
        _set_cell(ram, 192, 144, _FLOOR_QUAD)
        _set_cell(ram, 208, 144, _BLOCK_QUAD)
        _set_cell(ram, 208, 96, _STAIR_QUAD)
    return ram


def test_tile_classes_do_not_overlap() -> None:
    assert not FLOOR_TILES & BLOCK_TILES
    assert not FLOOR_TILES & STAIR_TILES
    assert not BLOCK_TILES & STAIR_TILES
    assert not BOMB_HOLE_TILES & FLOOR_TILES
    assert not DOOR_LEAF_TILES & FLOOR_TILES
    assert not DOOR_LEAF_TILES & BOMB_HOLE_TILES
    assert not LOCKED_DOOR_TILES & SHUTTER_TILES
    # A closed leaf is geometry: the walker must refuse it.
    assert not DOOR_LEAF_TILES & LINK_WALKABLE_TILES
    assert BOMB_HOLE_TILES <= LINK_WALKABLE_TILES


def test_room_door_types_decode_the_level_block() -> None:
    # Live L6 bytes (C8d pins): 0x79 A=0x22 B=0xa3, 0x28 A=0x1e B=0x32.
    ram = _blank_ram()
    wram = WRAM_RAM_OFFSET - WRAM_BASE
    for room, a, b in ((0x79, 0x22, 0xA3), (0x28, 0x1E, 0x32)):
        ram[wram + 0x687E + room] = a
        ram[wram + 0x68FE + room] = b
    assert room_door_types(ram, 0x79) == {"N": "wall", "S": "open", "W": "key", "E": "open"}
    assert room_door_types(ram, 0x28) == {"N": "open", "S": "shutter", "W": "wall", "E": "bomb"}


def test_short_ram_has_no_tile_map() -> None:
    short = np.zeros(0x800, dtype=np.uint8)
    assert not has_room_tile_map(short)
    with pytest.raises(ValueError):
        read_room_tiles(short)


def test_read_room_tiles_is_column_major() -> None:
    ram = _blank_ram()
    base = WRAM_RAM_OFFSET + ADDR_ROOM_TILE_MAP - WRAM_BASE
    ram[base + 3 * TILE_ROWS + 7] = 0x5A
    grid = read_room_tiles(ram)
    assert grid.shape == (TILE_ROWS, TILE_COLS)
    assert grid[7][3] == 0x5A
    assert tile_at(ram, 3, 7) == 0x5A
    assert tile_at_screen(ram, 3 * 8, 64 + 7 * 8) == 0x5A


def test_tile_index_out_of_range_raises() -> None:
    ram = _blank_ram()
    with pytest.raises(IndexError):
        tile_at(ram, TILE_COLS, 0)
    with pytest.raises(IndexError):
        tile_at(ram, 0, TILE_ROWS)


def test_link_cell_uses_the_foot_offset() -> None:
    assert LINK_FOOT_OFFSET == 11
    for link_y, cell_y in (
        (93, 96), (109, 112), (125, 128), (141, 144),
        (157, 160), (173, 176), (189, 192),
    ):
        assert link_cell(32, link_y) == (32, cell_y)
    # The pin (63,149) stands one row below the y=144 door row.
    assert link_cell(63, 149) == (48, 160)
    # UP from (144,141) parks at y=117, still the y=128 row.
    assert link_cell(144, 117) == (144, 128)


def test_room_0d_ring_has_only_two_column_crossings() -> None:
    ram = _room_0d_ram(pushed=False)
    blocked = set(find_cells(ram, BLOCK_TILES))
    for x in INTERIOR_X:
        crosses_bands = all(
            (x, y) not in blocked for y in (112, 176)
        )
        assert crosses_bands is (x in (32, 208)), x
    # x=192 is never a corridor: it is walled at every middle row.
    for y in (128, 144, 160):
        assert (192, y) in blocked


def test_room_0d_stairs_appear_only_after_the_push() -> None:
    before = _room_0d_ram(pushed=False)
    after = _room_0d_ram(pushed=True)
    assert stair_cells(before) == ()
    assert stair_cells(after) == ((208, 96),)
    assert cell_tiles(after, 208, 96) == _STAIR_QUAD
    # The block really slides one tile RIGHT; (192,144) opens up.
    assert (208, 144) in find_cells(after, BLOCK_TILES)
    assert (192, 144) not in find_cells(after, BLOCK_TILES)


def test_door_cells_find_the_west_door_outside_the_interior() -> None:
    ram = _room_0d_ram(pushed=False)
    assert door_cells(ram) == ((16, 144),)
    assert (16, 144) not in find_cells(ram, FLOOR_TILES)


# --- Measured doorway art (2026-09-14 sweep of all 259 dungeon fixtures) ---
#
# Every captured map below is a live ``$6530`` window, not a hand-built one.
# The sweep found doorway art at exactly eight ring positions and none inside
# a room, and the ids are identical in every level.


def test_ring_is_the_border_and_never_the_interior() -> None:
    for x, y in RING_CELLS:
        assert x in PLAYFIELD_X and y in PLAYFIELD_Y
        assert x not in INTERIOR_X or y not in INTERIOR_Y
    for y in INTERIOR_Y:
        for x in INTERIOR_X:
            assert (x, y) not in RING_CELLS


def test_an_open_doorway_mouth_is_plain_floor() -> None:
    """There is no tile id for an open door: the mouth is floor in the wall.

    L1 ``0x44`` has both side doors open (``$0668 = 0x02`` plus the east
    mouth). Neither cell holds door art, which is why any walkable set that
    contains ``FLOOR_TILES`` -- the fight grid included -- can already stand
    on them.
    """
    ram = room_tile_ram("0x44", level=1)
    assert cell_tiles(ram, 16, 144) == (0x24, 0x76, 0x24, 0x77)
    assert cell_tiles(ram, 224, 144) == (0x74, 0x24, 0x75, 0x24)
    for quad in (cell_tiles(ram, 16, 144), cell_tiles(ram, 224, 144)):
        assert not set(quad) & (DOOR_LEAF_TILES | BOMB_HOLE_TILES)
        assert set(quad) & FLOOR_TILES
    assert door_cells(ram) == ((16, 144), (224, 144))


def test_a_closed_leaf_is_solid_and_is_not_a_door_cell() -> None:
    """A shut door is geometry, not an exit.

    L1 ``0x23`` is shut on the west (locked, ``0xA0``-``0xA3``); L5 ``0x65``
    is shut on the north (shutter, ``0xA8``-``0xAB``). Walking into either
    does not leave the room, so neither is reported -- while the open south
    mouth of ``0x23``, which the old west-hole-only tile set could not see at
    all, is.
    """
    room_23 = room_tile_ram("0x23", level=1)
    assert cell_tiles(room_23, 16, 144) == (0xA0, 0xA2, 0xA1, 0xA3)
    assert set(cell_tiles(room_23, 16, 144)) <= LOCKED_DOOR_TILES
    assert door_cells(room_23) == ((112, 208), (128, 208))

    room_65 = room_tile_ram("0x65", level=5)
    assert set(cell_tiles(room_65, 112, 80)) - {0x79, 0x7A} <= SHUTTER_TILES
    assert (112, 80) not in door_cells(room_65)


def test_every_bombed_wall_direction_is_walkable() -> None:
    """The bug the old ``DOOR_TILES`` held: three of four holes graded solid.

    ``{0x90, 0x91}`` is the *west* pair alone. North (``0x8C``/``0x8D``),
    south (``0x8E``/``0x8F``) and east (``0x92``/``0x93``) are different ids,
    so a route aimed out through one of those had no path at all.
    """
    cases = (
        (7, "0x0d", (16, 144), (0x90, 0x91)),
        (5, "0x65", (224, 144), (0x92, 0x93)),
        (8, "0x3e", (112, 80), (0x8C, 0x8D)),
        (9, "0x10", (112, 208), (0x8E, 0x8F)),
    )
    for level, room, cell, pair in cases:
        ram = room_tile_ram(room, level=level)
        found = {
            t
            for x, y in RING_CELLS
            for t in cell_tiles(ram, x, y)
            if t in BOMB_HOLE_TILES
        }
        assert set(pair) <= found, (level, room)
        assert set(pair) <= LINK_WALKABLE_TILES
        assert cell in door_cells(ram), (level, room)


def test_hole_tiles_never_appear_inside_a_room() -> None:
    """Why grading them walkable cannot open an interior cell."""
    for level, room in ((7, "0x0d"), (5, "0x65"), (8, "0x3e"), (9, "0x10"),
                        (1, "0x44"), (1, "0x23")):
        ram = room_tile_ram(room, level=level)
        for y in INTERIOR_Y:
            for x in INTERIOR_X:
                assert not set(cell_tiles(ram, x, y)) & BOMB_HOLE_TILES


def test_ascii_room_renders_the_ring() -> None:
    lines = ascii_room(_room_0d_ram(pushed=True)).splitlines()
    assert lines[0].split()[1:] == [str(x) for x in INTERIOR_X]
    body = [line.split()[1:] for line in lines[1:]]
    assert body[0] == ["."] * 11 + ["S"]
    assert body[1] == ["."] + ["#"] * 10 + ["."]
    assert body[3] == ["."] * 11 + ["#"]
    assert body[6] == ["."] * 12

"""L4 occupancy seeds (no emulator). Dest cells stay open; solids stay blocked."""

from __future__ import annotations

from zelda_i.level4.dungeon import LADDER_60_PICKUP_XY, MAP_21_PICKUP_XY, RIGHT_20_STAND
from zelda_i.level4.occupancy import (
    ROOM_40_LEFTOVER_XY,
    ROOM_40_PICKUP_XY,
    room_20_grid,
    room_21_grid,
    room_40_grid,
    room_60_grid,
)
from zelda_i.walk.physics import follow_path


def test_dest_cells_not_overblocked() -> None:
    assert room_60_grid().passable(*LADDER_60_PICKUP_XY)
    assert room_20_grid().passable(*RIGHT_20_STAND)
    assert room_21_grid().passable(*MAP_21_PICKUP_XY)
    assert room_40_grid().passable(*ROOM_40_PICKUP_XY)


def test_documented_solids_are_blocked() -> None:
    assert not room_60_grid().passable(49, 133)
    assert not room_20_grid().passable(160, 150)
    assert not room_21_grid().passable(48, 140)
    grid = room_40_grid()
    assert not grid.passable(128, 148)
    assert not grid.passable(129, 149)
    assert not grid.passable(119, 149)


def test_room40_pocket_bfs_leaves_south() -> None:
    grid = room_40_grid()
    path = grid.shortest_path(ROOM_40_LEFTOVER_XY, ROOM_40_PICKUP_XY)
    assert path is not None
    assert follow_path(path, ROOM_40_LEFTOVER_XY) == "DOWN"


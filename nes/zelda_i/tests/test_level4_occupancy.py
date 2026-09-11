"""L4 occupancy seeds (no emulator). Dest cells stay open; solids stay blocked."""

from __future__ import annotations

from zelda_i.level4.dungeon import LADDER_60_PICKUP_XY, MAP_21_PICKUP_XY, RIGHT_20_STAND
from zelda_i.level4.occupancy import (
    ROOM_13_HC_XY,
    ROOM_31_CLEAR_XY,
    ROOM_31_EAST_XY,
    ROOM_31_SPAWN_XY,
    ROOM_13_NORTH_DOOR_XY,
    ROOM_13_SOUTH_XY,
    ROOM_13_SPAWN_XY,
    ROOM_40_LEFTOVER_XY,
    ROOM_40_PICKUP_XY,
    occupancy_dir,
    room_13_grid,
    room_31_grid,
    room_20_grid,
    room_21_grid,
    room_40_grid,
    room_60_grid,
)
from zelda_i.walk.physics import OccupancyWalker, WALK_DELTA, follow_path, predicted_xy


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


def test_room13_dest_cells_stay_open() -> None:
    grid = room_13_grid()
    assert grid.passable(*ROOM_13_SPAWN_XY)
    assert grid.passable(*ROOM_13_SOUTH_XY)
    assert grid.passable(*ROOM_13_HC_XY)
    assert grid.passable(*ROOM_13_NORTH_DOOR_XY)
    for x in range(40, 201, 8):
        assert grid.passable(x, ROOM_13_SOUTH_XY[1])


def test_room13_miss_blocks_ahead_and_replans() -> None:
    walker = OccupancyWalker(grid=room_13_grid())
    start = (80, 141)
    first = occupancy_dir(walker, start, ROOM_13_SOUTH_XY)
    assert first in WALK_DELTA
    second = occupancy_dir(walker, start, ROOM_13_SOUTH_XY)
    assert walker.misses == 1
    blocked = predicted_xy(*start, first)
    assert blocked in walker.grid.blocked
    assert second in WALK_DELTA
    path = walker.grid.shortest_path(start, ROOM_13_SOUTH_XY)
    assert path is not None
    assert blocked not in path
    assert walker.grid.passable(*ROOM_13_SOUTH_XY)
    assert walker.grid.passable(*ROOM_13_NORTH_DOOR_XY)


def test_room31_dest_cells_stay_open() -> None:
    grid = room_31_grid()
    assert grid.passable(*ROOM_31_SPAWN_XY)
    assert grid.passable(*ROOM_31_EAST_XY)
    assert grid.passable(80, 173)
    assert grid.passable(80, 109)
    assert not grid.passable(*ROOM_31_CLEAR_XY)
    path = grid.shortest_path((80, 173), ROOM_31_SPAWN_XY)
    assert path is not None
    assert ROOM_31_CLEAR_XY not in path


def test_room31_water_pocket_stands() -> None:
    walker = OccupancyWalker(grid=room_31_grid())
    assert occupancy_dir(
        walker, ROOM_31_CLEAR_XY, ROOM_31_EAST_XY, sticky=True
    ) is None
    assert walker.grid.passable(*ROOM_31_EAST_XY)
    assert walker.grid.passable(*ROOM_31_SPAWN_XY)


def test_room13_no_path_stands() -> None:
    grid = room_13_grid()
    sx, sy = 80, 141
    for dx, dy in WALK_DELTA.values():
        grid.blocked.add((sx + dx, sy + dy))
    walker = OccupancyWalker(grid=grid)
    assert occupancy_dir(walker, (sx, sy), ROOM_13_SOUTH_XY) is None
    assert occupancy_dir(walker, (sx, sy), ROOM_13_SOUTH_XY) is None
    assert walker.grid.passable(*ROOM_13_SOUTH_XY)


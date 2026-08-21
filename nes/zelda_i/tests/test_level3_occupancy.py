"""0x6b occupancy seed (no emulator)."""

from __future__ import annotations

import numpy as np

from zelda_i.level3_dungeon import ROOM_L3_NORTH_ZOLS as ROOM_6B
from zelda_i.level3_geometry import NORTH_DOOR_X, ROOM_6B_BAND_Y, ROOM_6B_STRAND_Y
from retro_harness.nes import nes_action
from zelda_i.level3_occupancy import room_6b_grid
from zelda_i.level3_path import Level3NorthExit6bController
from zelda_i.ram import (
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.walk_physics import OccupancyWalker


def _ram(*, x: int, y: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 3
    ram[ADDR_SCREEN] = ROOM_6B
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    return ram


def test_room_6b_grid_paths_south_to_north_band() -> None:
    grid = room_6b_grid()
    band = (NORTH_DOOR_X, ROOM_6B_BAND_Y)
    # Live dest combat/exit stands (l3_dest_0x5b / _occ).
    for start in ((120, 181), (100, 181), (144, 165)):
        path = grid.shortest_path(start, band)
        assert path is not None
        assert path[0] == start
        assert path[-1] == band
        assert all(
            not (abs(x - NORTH_DOOR_X) <= 8 and y <= ROOM_6B_STRAND_Y)
            for x, y in path
        )


def test_room_6b_strand_cells_are_blocked() -> None:
    grid = room_6b_grid()
    assert not grid.passable(NORTH_DOOR_X, ROOM_6B_STRAND_Y)
    assert not grid.passable(NORTH_DOOR_X, 93)
    assert grid.passable(NORTH_DOOR_X, ROOM_6B_BAND_Y)
    assert grid.passable(NORTH_DOOR_X, 141)


def test_controller_starts_from_6b_seed() -> None:
    ctrl = Level3NorthExit6bController()
    assert ctrl.grid.blocked
    assert not ctrl.grid.passable(NORTH_DOOR_X, ROOM_6B_STRAND_Y)
    assert ctrl.grid.passable(NORTH_DOOR_X, ROOM_6B_BAND_Y)


def test_miss_blocks_ahead_and_replans() -> None:
    ctrl = Level3NorthExit6bController()
    start = read_snapshot(_ram(x=120, y=141))
    first = ctrl.step(start)
    assert first.reason == "north6b_path"
    assert ctrl.walker.last_dir == "UP"
    second = ctrl.step(start)
    assert ctrl.misses == 1
    assert (120, 140) in ctrl.grid.blocked
    assert second.reason == "north6b_path"
    assert ctrl.walker.last_dir in {"LEFT", "RIGHT"}
    path = ctrl.grid.shortest_path((120, 141), (NORTH_DOOR_X, ROOM_6B_BAND_Y))
    assert path is not None
    assert (120, 140) not in path


def test_south_mouth_is_not_occupancy_graded() -> None:
    """Combat can leave Link on the south door; do not miss-block inland UP."""
    ctrl = Level3NorthExit6bController()
    door = ctrl.step(read_snapshot(_ram(x=120, y=181)))
    assert door.reason == "north6b_leave_mouth_x"
    assert list(door.action) == list(nes_action("LEFT"))
    assert ctrl.misses == 0
    assert ctrl.walker.last_dir is None
    inland = ctrl.step(read_snapshot(_ram(x=100, y=181)))
    assert inland.reason == "north6b_leave_mouth"
    assert list(inland.action) == list(nes_action("UP"))
    assert ctrl.misses == 0


def test_no_path_walker_stands() -> None:
    grid = room_6b_grid()
    for x in range(grid.xmin, grid.xmax + 1):
        grid.blocked.add((x, 120))
    walker = OccupancyWalker(grid=grid, goal=(NORTH_DOOR_X, ROOM_6B_BAND_Y))
    start = (120, 181)
    walker.observe(start)
    assert walker.next_dir(start) is None
    assert walker.last_dir is None

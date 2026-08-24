"""0x60 occupancy seed (no emulator). East-dock waypoints, not live BFS."""

from __future__ import annotations

import numpy as np

from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level4_dungeon import LADDER_60_PICKUP_XY, ROOM_L4_STEPLADDER
from zelda_i.level4_occupancy import (
    ROOM_60_CAUSWAY_XY,
    ROOM_60_DOCK_MOUTH_XY,
    ROOM_60_DOCK_NORTH_XY,
    ROOM_60_EXIT_WAYPOINTS,
    ROOM_60_EXIT_X,
    ROOM_60_ISLAND_XY,
    ROOM_60_NORTH_STRIP_Y,
    ROOM_60_SOUTH_XY,
    ROOM_60_SPAWN_XY,
    ROOM_60_WAYPOINTS,
    ROOM_60_WEST_AISLE_X,
    room_60_grid,
)
from zelda_i.level4_exit60 import make_exit60_controller
from zelda_i.level4_stepladder import make_stepladder_controller
from zelda_i.ram import (
    ADDR_LADDER,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.walk_physics import OccupancyWalker


def test_room_60_grid_blocks_live_solids() -> None:
    grid = room_60_grid()
    assert not grid.passable(49, 133)
    assert not grid.passable(49, 69)
    assert not grid.passable(49, ROOM_60_NORTH_STRIP_Y)
    assert not grid.passable(49, 159)
    assert grid.passable(136, 189)
    assert not grid.passable(176, 189)
    assert grid.passable(176, 151)
    assert grid.passable(175, 188)
    assert not grid.passable(168, 188)
    assert grid.passable(*ROOM_60_SPAWN_XY)
    assert grid.passable(*ROOM_60_ISLAND_XY)
    assert grid.passable(168, 189)
    assert grid.passable(*ROOM_60_DOCK_MOUTH_XY)
    assert not grid.passable(84, 190)
    assert not grid.passable(88, 190)


def test_room_60_spawn_to_island_path() -> None:
    grid = room_60_grid()
    goal = LADDER_60_PICKUP_XY
    path = grid.shortest_path(ROOM_60_SPAWN_XY, goal)
    assert path is not None
    assert path[0] == ROOM_60_SPAWN_XY
    assert path[-1] == goal
    xs = {p[0] for p in path}
    assert max(xs) >= 174


def test_room_60_waypoints_are_east_dock() -> None:
    assert ROOM_60_WAYPOINTS[0] == ROOM_60_SOUTH_XY
    assert ROOM_60_SOUTH_XY == (48, 189)
    assert ROOM_60_WAYPOINTS[1] == ROOM_60_DOCK_MOUTH_XY
    assert ROOM_60_DOCK_MOUTH_XY == (175, 189)
    assert ROOM_60_WAYPOINTS[2] == ROOM_60_DOCK_NORTH_XY
    assert ROOM_60_WAYPOINTS[-1] == LADDER_60_PICKUP_XY
    assert ROOM_60_WEST_AISLE_X == 48


def test_room_60_walker_from_spawn_goes_south() -> None:
    walker = OccupancyWalker(grid=room_60_grid(), goal=LADDER_60_PICKUP_XY)
    assert walker.next_dir(ROOM_60_SPAWN_XY) == "DOWN"
    assert walker.next_dir(ROOM_60_SOUTH_XY) == "RIGHT"


def test_stepladder_path_walks_east_dock() -> None:
    from zelda_i.level4_stepladder import StepladderPhase

    ctl = make_stepladder_controller(clear_first=False)
    ctl.phase = StepladderPhase.PATH
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = 9
    ram[ADDR_LEVEL] = 4
    ram[ADDR_SCREEN] = ROOM_L4_STEPLADDER
    ram[ADDR_LINK_X] = 48
    ram[ADDR_LINK_Y] = 69
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "join_dock_y"
    assert list(act.action) == list(nes_action("DOWN"))
    ram[ADDR_LINK_Y] = 189
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "join_dock_x"
    assert list(act.action) == list(nes_action("RIGHT"))
    ram[ADDR_LINK_X] = 171
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "join_dock_x"
    ram[ADDR_LINK_X] = 174
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "join_dock_x"
    ram[ADDR_LINK_X] = 175
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "join_dock_y"
    assert list(act.action) == list(nes_action("UP"))
    ram[ADDR_LINK_X] = 176
    ram[ADDR_LINK_Y] = 157
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "join_dock_y"
    assert list(act.action) == list(nes_action("UP"))
    ram[ADDR_LINK_Y] = 149
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("UP"))
    ram[ADDR_LINK_Y] = 141
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "join_dock_x"
    assert list(act.action) == list(nes_action("LEFT"))
    ram[ADDR_LINK_X] = 141
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "join_dock_x"
    ram[ADDR_LINK_X] = 138
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "join_dock_x"
    ram[ADDR_LINK_X] = 136
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "at_pedestal"
    ctl2 = make_stepladder_controller(clear_first=False)
    ctl2.phase = StepladderPhase.PATH
    ctl2._last_xy = (48, 189)
    ctl2._stall = 96
    ram[ADDR_LINK_X] = 48
    ram[ADDR_LINK_Y] = 189
    act = ctl2.step(read_snapshot(ram))
    assert ctl2.phase is StepladderPhase.FAILED
    assert act.reason.startswith("dock_solid_48_189")


def test_room_60_exit_waypoints_reverse_dock() -> None:
    assert ROOM_60_CAUSWAY_XY == (175, 141)
    assert ROOM_60_EXIT_WAYPOINTS == (
        ROOM_60_CAUSWAY_XY,
        ROOM_60_DOCK_MOUTH_XY,
        ROOM_60_SOUTH_XY,
        ROOM_60_SPAWN_XY,
    )
    grid = room_60_grid()
    assert grid.passable(*ROOM_60_CAUSWAY_XY)
    assert (ROOM_60_EXIT_X, 189) not in ROOM_60_EXIT_WAYPOINTS
    assert not grid.passable(ROOM_60_EXIT_X, 189)
    for start, goal in zip(ROOM_60_EXIT_WAYPOINTS, ROOM_60_EXIT_WAYPOINTS[1:]):
        path = grid.shortest_path(start, goal)
        assert path is not None
        assert (ROOM_60_EXIT_X, 189) not in path
    island_to_dock = grid.shortest_path(ROOM_60_ISLAND_XY, ROOM_60_CAUSWAY_XY)
    assert island_to_dock is not None
    assert {y for _, y in island_to_dock} == {141}


def test_room_60_walker_from_island_goes_east() -> None:
    walker = OccupancyWalker(grid=room_60_grid(), goal=ROOM_60_CAUSWAY_XY)
    assert walker.next_dir(ROOM_60_ISLAND_XY) == "RIGHT"
    assert walker.next_dir(ROOM_60_CAUSWAY_XY, ROOM_60_DOCK_MOUTH_XY) == "DOWN"
    walker2 = OccupancyWalker(grid=room_60_grid(), goal=ROOM_60_SOUTH_XY)
    assert walker2.next_dir(ROOM_60_DOCK_MOUTH_XY) == "LEFT"


def test_exit60_path_walks_reverse_dock() -> None:
    from zelda_i.level4_dungeon import ROOM_L4_EAST_32
    from zelda_i.level4_exit60 import Exit60Phase
    from zelda_i.level4_stepladder import POST_LADDER_ITEM_SETTLE

    ctl = make_exit60_controller()
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = 9
    ram[ADDR_LEVEL] = 4
    ram[ADDR_SCREEN] = ROOM_L4_STEPLADDER
    ram[ADDR_LINK_X] = 136
    ram[ADDR_LINK_Y] = 141
    ram[ADDR_LADDER] = 1
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "item_freeze"
    assert list(act.action) == list(nes_idle_action())
    ctl.phase_frames = POST_LADDER_ITEM_SETTLE
    act = ctl.step(read_snapshot(ram))
    assert ctl.phase is Exit60Phase.PATH
    assert act.reason == "join_exit"
    assert list(act.action) == list(nes_action("RIGHT"))
    ram[ADDR_LINK_X] = 175
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "join_exit"
    assert list(act.action) == list(nes_action("DOWN"))
    ram[ADDR_LINK_X] = 176
    ram[ADDR_LINK_Y] = 173
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("DOWN"))
    ram[ADDR_LINK_Y] = 189
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("LEFT"))
    ram[ADDR_LINK_X] = 176
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("LEFT"))
    ram[ADDR_LINK_X] = 48
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("UP"))
    ram[ADDR_LINK_Y] = 69
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "enter_stairs_up"
    assert list(act.action) == list(nes_action("UP"))
    ram[ADDR_SCREEN] = ROOM_L4_EAST_32
    ram[ADDR_MODE] = PLAY_MODE
    act = ctl.step(read_snapshot(ram))
    assert ctl.success
    assert act.reason == "done"
    report = ctl.report()
    assert "bfs" not in report
    assert report["segment"] == "level4_exit_0x60"
    ctl2 = make_exit60_controller()
    ctl2.phase = Exit60Phase.PATH
    ctl2._last_xy = (136, 141)
    ctl2._stall = 96
    ram[ADDR_MODE] = 9
    ram[ADDR_SCREEN] = ROOM_L4_STEPLADDER
    ram[ADDR_LINK_X] = 136
    ram[ADDR_LINK_Y] = 141
    act = ctl2.step(read_snapshot(ram))
    assert ctl2.phase is Exit60Phase.FAILED
    assert act.reason.startswith("exit_solid_136_141")

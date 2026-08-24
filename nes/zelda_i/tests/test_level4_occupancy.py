"""0x60 occupancy seed (no emulator). North-strip LEFT+UP, not live BFS."""

from __future__ import annotations

import numpy as np

from retro_harness.nes import nes_action
from zelda_i.level4_dungeon import LADDER_60_PICKUP_XY, ROOM_L4_STEPLADDER
from zelda_i.level4_occupancy import (
    ROOM_60_CLIP_BUTTONS,
    ROOM_60_CLIP_STAND,
    ROOM_60_ISLAND_XY,
    ROOM_60_NORTH_STRIP_Y,
    ROOM_60_SPAWN_XY,
    ROOM_60_WAYPOINTS,
    ROOM_60_WEST_AISLE_X,
    room_60_grid,
)
from zelda_i.level4_stepladder import make_stepladder_controller
from zelda_i.ram import (
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
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
    assert not grid.passable(176, 141)
    assert grid.passable(*ROOM_60_SPAWN_XY)
    assert grid.passable(*ROOM_60_ISLAND_XY)
    assert grid.passable(168, 189)
    assert grid.passable(*ROOM_60_CLIP_STAND)
    assert not grid.passable(84, 190)
    assert not grid.passable(88, 190)


def test_room_60_no_cardinal_path_after_v16() -> None:
    grid = room_60_grid()
    goal = LADDER_60_PICKUP_XY
    for start in (
        (48, 133),
        ROOM_60_SPAWN_XY,
        (48, 68),
        (48, 65),
        (48, 71),
        (48, 130),
        (48, 157),
        (48, 161),
        (48, 165),
        ROOM_60_CLIP_STAND,
        (48, 159),
        (48, 189),
        (80, 189),
        (84, 189),
        (88, 189),
        (136, 189),
        (160, 189),
        (168, 189),
    ):
        assert grid.shortest_path(start, goal) is None, start


def test_room_60_waypoints_are_north_leftup() -> None:
    assert ROOM_60_WAYPOINTS[0] == ROOM_60_CLIP_STAND
    assert ROOM_60_CLIP_STAND == (48, 68)
    assert ROOM_60_WAYPOINTS[-1] == LADDER_60_PICKUP_XY
    assert ROOM_60_CLIP_BUTTONS == ("LEFT", "UP")


def test_room_60_walker_from_spawn_stands() -> None:
    walker = OccupancyWalker(grid=room_60_grid(), goal=LADDER_60_PICKUP_XY)
    assert walker.next_dir(ROOM_60_SPAWN_XY) is None
    assert walker.next_dir(ROOM_60_CLIP_STAND) is None


def test_stepladder_path_clips_north_leftup() -> None:
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
    assert act.reason == "clip_leftup68"
    assert list(act.action) == list(nes_action("LEFT", "UP"))
    ram[ADDR_LINK_Y] = 133
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "join_clip_y"
    ctl2 = make_stepladder_controller(clear_first=False)
    ctl2.phase = StepladderPhase.PATH
    ctl2._last_xy = (48, 68)
    ctl2._stall = 96
    ram[ADDR_LINK_X] = 48
    ram[ADDR_LINK_Y] = 68
    act = ctl2.step(read_snapshot(ram))
    assert ctl2.phase is StepladderPhase.FAILED
    assert act.reason.startswith("leftup68_solid_48_68")

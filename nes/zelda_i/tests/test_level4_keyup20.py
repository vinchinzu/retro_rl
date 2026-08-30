"""0x31 maze-west: SW pocket leftover peels to the west aisle. No emulator."""

from __future__ import annotations

import numpy as np

from zelda_i.level4.keyup20 import (
    Maze31WestPhase,
    make_maze_31_west_controller,
)
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


def _pose(x: int, y: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 4
    ram[ADDR_SCREEN] = 0x31
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_LADDER] = 1
    return ram


def _inland(path_index: int = 3) -> object:
    ctrl = make_maze_31_west_controller()
    ctrl.phase = Maze31WestPhase.INLAND
    ctrl.path_index = path_index
    return ctrl


def test_sw_pocket_40_165_peels_right_not_up() -> None:
    """Compose leftover (40,165): UP is the door-frame south face."""
    ctrl = _inland(3)
    act = ctrl.step(read_snapshot(_pose(40, 165)))
    assert ctrl.phase is Maze31WestPhase.INLAND
    assert act.reason == "west_aisle_peel"
    assert "UP" not in act.reason


def test_west_aisle_south_aligns_y_to_door() -> None:
    ctrl = _inland(3)
    act = ctrl.step(read_snapshot(_pose(48, 165)))
    assert ctrl.phase is Maze31WestPhase.INLAND
    assert act.reason == "west_door_align_y"


def test_door_band_from_aisle_goes_left() -> None:
    ctrl = _inland(3)
    act = ctrl.step(read_snapshot(_pose(48, 141)))
    assert ctrl.phase is Maze31WestPhase.INLAND
    assert act.reason == "west_door_left"


def test_alcove_32_149_aligns_y_not_left() -> None:
    """l4_maze_west_pocket leftover: LEFT at y=149 is the door-frame lip."""
    ctrl = _inland(3)
    act = ctrl.step(read_snapshot(_pose(32, 149)))
    assert ctrl.phase is Maze31WestPhase.INLAND
    assert act.reason == "west_door_align_y"


def test_north_strip_still_left_to_inland() -> None:
    """Historical CLIP leftover (160,113) must keep LEFT toward (80,109)."""
    ctrl = _inland(0)
    act = ctrl.step(read_snapshot(_pose(160, 113)))
    assert ctrl.phase is Maze31WestPhase.INLAND
    assert act.reason == "join_maze_west"

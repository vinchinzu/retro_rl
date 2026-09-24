"""0x31 maze-west: SW pocket leftover peels to the west aisle. No emulator."""

from __future__ import annotations

import numpy as np
import pytest

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


@pytest.mark.parametrize(
    "path_index,x,y,reason",
    [
        pytest.param(3, 48, 165, "west_door_align_y", id="west_aisle_south_aligns_y_to_door"),
        pytest.param(3, 48, 141, "west_door_left", id="door_band_from_aisle_goes_left"),
    ],
)
def test_west_door_band_reasons(path_index: int, x: int, y: int, reason: str) -> None:
    ctrl = _inland(path_index)
    act = ctrl.step(read_snapshot(_pose(x, y)))
    assert ctrl.phase is Maze31WestPhase.INLAND
    assert act.reason == reason


@pytest.mark.parametrize(
    "path_index,x,y,reason",
    [
        pytest.param(
            3, 32, 149, "west_door_align_y",
            id="alcove_32_149_aligns_y_not_left",
        ),
        pytest.param(
            0, 160, 113, "join_maze_west",
            id="north_strip_still_left_to_inland",
        ),
    ],
)
def test_west_aisle_leftover_reasons(path_index: int, x: int, y: int, reason: str) -> None:
    """l4_maze_west_pocket leftovers: (32,149) door-frame lip aligns y not left;
    historical CLIP leftover (160,113) keeps LEFT toward (80,109)."""
    ctrl = _inland(path_index)
    act = ctrl.step(read_snapshot(_pose(x, y)))
    assert ctrl.phase is Maze31WestPhase.INLAND
    assert act.reason == reason


def test_knocked_off_strip_on_east_column_climbs_again() -> None:
    """A Keese hit at (160,110) knocked Link to (160,142); LEFT from there
    crossed the water on the ladder and wedged at x~104 for 6000f."""
    ctrl = _inland(0)
    act = ctrl.step(read_snapshot(_pose(160, 142)))
    assert act.reason == "knocked_off_strip"
    assert ctrl.phase is Maze31WestPhase.EAST_U
    act = ctrl.step(read_snapshot(_pose(160, 142)))
    assert act.reason == "join_maze_west"
    assert ctrl.path_index == 3  # (160,125): the column top, then the clip


def test_inland_walk_past_the_column_is_not_a_knockback() -> None:
    ctrl = _inland(2)
    act = ctrl.step(read_snapshot(_pose(48, 125)))
    assert ctrl.phase is Maze31WestPhase.INLAND
    assert act.reason != "knocked_off_strip"

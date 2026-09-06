"""Unit tests for L6 0x09 left-block stairs. East-of-block from NW leftover."""

from __future__ import annotations

import numpy as np

from retro_harness.nes import nes_action
from zelda_i.level6.hops import ok6, rod_cellar_ok
from zelda_i.level6.path import BLOCK_OBJECT_TYPE
from zelda_i.level6.stairs09 import EAST_CLEAR_X, make_stairs_09_controller
from zelda_i.ram import ADDR_LINK_X, ADDR_LINK_Y, ADDR_OBJ_TYPE, PASSAGE_MODE, PLAY_MODE, read_snapshot
from zelda_i.tests.ram_helpers import make_ram

LEFT_BLOCK = (96, 144)
NW_LEFTOVER = (56, 109)
BOXED_WEST = (56, 157)
HISTORICAL_SOUTH = (112, 173)

_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 6,
    "screen": 0x09,
    "x": NW_LEFTOVER[0],
    "y": NW_LEFTOVER[1],
    "triforce": 0x1F,
    "keys": 3,
    "bombs": 8,
}


def _ram(**fields: int) -> np.ndarray:
    return make_ram(_DEFAULTS, **fields)


def _plant_left_block(ram: np.ndarray) -> None:
    ram[ADDR_OBJ_TYPE + 11] = BLOCK_OBJECT_TYPE
    ram[ADDR_LINK_X + 11] = LEFT_BLOCK[0]
    ram[ADDR_LINK_Y + 11] = LEFT_BLOCK[1]


def test_nw_leftover_rights_on_north_band() -> None:
    ram = _ram()
    _plant_left_block(ram)
    ctl = make_stairs_09_controller()
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "stand_east_x"
    assert list(act.action) == list(nes_action("RIGHT"))
    assert list(act.action) != list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("DOWN"))


def test_north_band_mid_column_still_rights() -> None:
    ram = _ram(x=64, y=NW_LEFTOVER[1])
    _plant_left_block(ram)
    ctl = make_stairs_09_controller()
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "stand_east_x"
    assert list(act.action) == list(nes_action("RIGHT"))


def test_east_of_block_descends() -> None:
    ram = _ram(x=EAST_CLEAR_X, y=NW_LEFTOVER[1])
    _plant_left_block(ram)
    ctl = make_stairs_09_controller()
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "stand_east_y"
    assert list(act.action) == list(nes_action("DOWN"))


def test_historical_south_leftover_x_aligns_left() -> None:
    ram = _ram(x=HISTORICAL_SOUTH[0], y=HISTORICAL_SOUTH[1])
    _plant_left_block(ram)
    ctl = make_stairs_09_controller()
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "stand_x"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("RIGHT"))


def test_boxed_west_face_is_dead_belief_not_west_aisle() -> None:
    """(56,157) tile 118 boxed cardinal DOWN. Not a live start.

    Must not retry west-aisle LEFT or DOWN from this cell.
    """
    ram = _ram(x=BOXED_WEST[0], y=BOXED_WEST[1])
    _plant_left_block(ram)
    ctl = make_stairs_09_controller()
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) != list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("LEFT"))
    assert act.reason != "stand_peel_x"
    assert act.reason != "stand_peel_y"


def test_rod_cellar_ok_accepts_mode_9() -> None:
    ram = _ram(mode=PASSAGE_MODE, screen=0x75, x=136, y=141, rod=1)
    snap = read_snapshot(ram)
    assert rod_cellar_ok(snap)
    assert not ok6(rod=True, tf_eq=0x1F)(snap)


def test_rod_pickup_fails_closed_if_already_owned() -> None:
    from zelda_i.level6.rod import make_rod_75_controller

    ram = _ram(mode=PASSAGE_MODE, screen=0x75, x=48, y=93, rod=1)
    ctl = make_rod_75_controller()
    act = ctl.step(read_snapshot(ram))
    assert ctl.failed and not ctl.success
    assert act.reason == "already_rod"

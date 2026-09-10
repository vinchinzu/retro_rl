"""L8 four-head Gleeok south-stand. No emulator. No RAM writes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.level8.dungeon import GLEEOK_FOUR_HEAD_OBJECT_TYPE
from zelda_i.level8.gleeok import (
    GLEEOK_ROOM,
    HEART_REACH,
    HEART_SLOT,
    HEART_XY,
    Level8FourHeadGleeokController,
    heart_xy,
    make_four_head_gleeok_controller,
)
from zelda_i.level8.path import (
    UnverifiedLevel8PathController,
    make_four_head_gleeok_controller as path_make,
    make_gleeok_passage_controller,
)
from zelda_i.ram import (
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_ROOM_ITEM_ID,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.tests.ram_helpers import make_ram

_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 8,
    "screen": GLEEOK_ROOM,
    "x": 120,
    "y": 189,
    "keys": 8,
    "bombs": 5,
    "magic_key": 1,
    "triforce": 0x7F,
    "health": 0x22,
    "room_item": 0x1A,
}

UP = list(nes_action("UP"))
IDLE = list(nes_idle_action())
LEFT_UP = list(nes_action("LEFT", "UP"))


def _ram(**fields: int) -> np.ndarray:
    return make_ram(_DEFAULTS, **fields)


def _step(ctl, ram: np.ndarray):
    before = ram.copy()
    act = ctl.step(read_snapshot(ram))
    assert np.array_equal(ram, before), "gleeok fight must not write RAM"
    return act


def test_live_type_is_0x45_not_assumed() -> None:
    assert GLEEOK_FOUR_HEAD_OBJECT_TYPE == 0x45
    ctl = make_four_head_gleeok_controller()
    assert ctl.observed_body_type == 0x45
    assert ctl.report()["assumed_0x45"] is False
    assert ctl.report()["route_eligible"] is False


def test_empty_0x3c_waits_for_body() -> None:
    ctl = make_four_head_gleeok_controller()
    act = _step(ctl, _ram())
    assert not ctl.failed and not ctl.success
    assert list(act.action) == IDLE
    assert act.reason == "wait_body"


def test_south_mouth_walks_inland() -> None:
    ctl = make_four_head_gleeok_controller()
    act = _step(ctl, _ram(y=189))
    assert not ctl.failed
    assert act.reason in ("wait_body", "south_inland")


def test_wrong_room_fails_closed() -> None:
    ctl = make_four_head_gleeok_controller()
    act = _step(ctl, _ram(screen=0x4C, x=112, y=125))
    assert ctl.failed and not ctl.success
    assert list(act.action) == IDLE


def test_path_factory_is_fight_not_unverified() -> None:
    ctl = path_make()
    assert isinstance(ctl, Level8FourHeadGleeokController)
    assert not isinstance(ctl, UnverifiedLevel8PathController)
    passage = make_gleeok_passage_controller()
    assert isinstance(passage, UnverifiedLevel8PathController)
    assert HEART_XY == (32, 192)
    assert HEART_REACH == 1


def test_body_gone_walks_sw_heart_not_stand() -> None:
    """F2 leftover (48,153) must LEFT toward (32,192), not heart_stand."""
    ram = _ram(x=48, y=153)
    ram[ADDR_LINK_X + HEART_SLOT] = 32
    ram[ADDR_LINK_Y + HEART_SLOT] = 192
    ram[ADDR_OBJ_TYPE + 1] = 0x45
    ram[ADDR_LINK_X + 1] = 124
    ram[ADDR_LINK_Y + 1] = 111
    ram[ADDR_OBJ_HP + 1] = 160
    ctl = make_four_head_gleeok_controller()
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    _step(ctl, ram)
    ram[ADDR_OBJ_TYPE + 1] = 0
    act = _step(ctl, ram)
    assert ctl.body_gone and not ctl.success and not ctl.failed
    assert act.reason == "heart_x"
    assert list(act.action) == list(nes_action("LEFT"))


def test_f3_stand_walks_south_onto_heart() -> None:
    """F3 (32,164) must DOWN toward (32,192), not heart_stand."""
    ram = _ram(x=32, y=164)
    ram[ADDR_LINK_X + HEART_SLOT] = 32
    ram[ADDR_LINK_Y + HEART_SLOT] = 192
    ram[ADDR_OBJ_TYPE + 1] = 0x45
    ram[ADDR_LINK_X + 1] = 124
    ram[ADDR_LINK_Y + 1] = 111
    ram[ADDR_OBJ_HP + 1] = 160
    ctl = make_four_head_gleeok_controller()
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    _step(ctl, ram)
    ram[ADDR_OBJ_TYPE + 1] = 0
    act = _step(ctl, ram)
    assert ctl.body_gone and not ctl.success
    assert act.reason == "heart_y"
    assert list(act.action) == list(nes_action("DOWN"))


def test_body_gone_without_a_watched_heart_item_does_not_green() -> None:
    """``room_item_id != 0x1A`` alone is not heart-container evidence."""
    ram = _ram(x=48, y=153, room_item=0x00)
    ram[ADDR_LINK_X + HEART_SLOT] = 32
    ram[ADDR_LINK_Y + HEART_SLOT] = 192
    ram[ADDR_OBJ_TYPE + 1] = 0x45
    ram[ADDR_LINK_X + 1] = 124
    ram[ADDR_LINK_Y + 1] = 111
    ram[ADDR_OBJ_HP + 1] = 160
    ctl = make_four_head_gleeok_controller()
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    _step(ctl, ram)
    assert ctl.saw_heart_item is False
    ram[ADDR_OBJ_TYPE + 1] = 0
    act = _step(ctl, ram)
    assert ctl.body_gone and not ctl.success and not ctl.failed
    assert act.reason == "heart_x"
    assert ctl.report()["saw_heart_item"] is False


def test_unbound_env_waits_for_heart_not_walks_frozen() -> None:
    """Unbound env has no RAM access: must heart_wait/idle, not walk to (32, 192)."""
    ram = _ram(x=48, y=153)
    ctl = make_four_head_gleeok_controller()
    act = _body_then_gone(ctl, ram)
    assert ctl.body_gone and not ctl.success and not ctl.failed
    assert act.reason == "heart_wait"
    assert list(act.action) == IDLE
    assert heart_xy(None) is None


def test_empty_slot19_waits_for_heart_not_walks_frozen() -> None:
    """Slot 19 reading (0, 0) must heart_wait/idle, not walk to (32, 192)."""
    ram = _ram(x=48, y=153)
    assert ram[ADDR_LINK_X + HEART_SLOT] == 0
    assert ram[ADDR_LINK_Y + HEART_SLOT] == 0
    ctl = make_four_head_gleeok_controller()
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    act = _body_then_gone(ctl, ram)
    assert ctl.body_gone and not ctl.success and not ctl.failed
    assert act.reason == "heart_wait"
    assert list(act.action) == IDLE
    assert heart_xy(ram) is None


def _body_then_gone(ctl, ram: np.ndarray):
    ram[ADDR_OBJ_TYPE + 1] = 0x45
    ram[ADDR_LINK_X + 1] = 124
    ram[ADDR_LINK_Y + 1] = 111
    ram[ADDR_OBJ_HP + 1] = 160
    _step(ctl, ram)
    ram[ADDR_OBJ_TYPE + 1] = 0
    return _step(ctl, ram)


def test_heart_dest_is_ram_xy_not_frozen_spawn() -> None:
    """Walk toward slot-19 RAM, not the frozen (32,192) leftover."""
    ram = _ram(x=48, y=153)
    ram[ADDR_LINK_X + HEART_SLOT] = 160
    ram[ADDR_LINK_Y + HEART_SLOT] = 141
    ctl = make_four_head_gleeok_controller()
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    act = _body_then_gone(ctl, ram)
    assert ctl.body_gone and not ctl.success and not ctl.failed
    assert act.reason == "heart_x"
    assert list(act.action) == list(nes_action("RIGHT"))
    assert list(act.action) != list(nes_action("LEFT"))
    assert heart_xy(ram) == (160, 141)
    assert heart_xy(ram) != HEART_XY


def test_heart_off_column_leftover_does_not_up_into_wall() -> None:
    """x=208 leftover walks LEFT toward RAM heart x, never UP into a wall."""
    ram = _ram(x=208, y=157)
    ram[ADDR_LINK_X + HEART_SLOT] = 120
    ram[ADDR_LINK_Y + HEART_SLOT] = 141
    ctl = make_four_head_gleeok_controller()
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    act = _body_then_gone(ctl, ram)
    assert ctl.body_gone and not ctl.success
    assert act.reason == "heart_x"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("UP"))


def test_watched_heart_item_falling_edge_still_greens() -> None:
    ram = _ram(x=32, y=190)
    ram[ADDR_OBJ_TYPE + 1] = 0x45
    ram[ADDR_LINK_X + 1] = 124
    ram[ADDR_LINK_Y + 1] = 111
    ram[ADDR_OBJ_HP + 1] = 160
    ctl = make_four_head_gleeok_controller()
    _step(ctl, ram)
    assert ctl.saw_heart_item is True
    ram[ADDR_OBJ_TYPE + 1] = 0
    ram[ADDR_ROOM_ITEM_ID] = 0x00
    _step(ctl, ram)
    assert ctl.success is True and not ctl.failed

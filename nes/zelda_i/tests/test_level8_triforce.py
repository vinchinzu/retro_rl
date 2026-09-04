"""L8 0x3C north shutter dest hop. No emulator. No RAM writes."""

from __future__ import annotations

import numpy as np
from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.level8.path import (
    UnverifiedLevel8PathController,
    make_gleeok_passage_controller,
    make_shard_leave_controller,
)
from zelda_i.level8.triforce import (
    NORTH_3C_DEST,
    NORTH_3C_DEST_POSE,
    NORTH_3C_ORIGIN,
    NORTH_3C_ORIGIN_POSE,
    NORTH_BAND_Y,
    NORTH_DOOR,
    ROOM_ITEM_TF,
    SOUTH_FAIL,
    Level8North3CController,
    Level8Shard2CController,
    make_north_3c_controller,
    make_shard_2c_controller,
    north_3c_step,
    shard_2c_step,
)
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_CUR_OPENED_DOORS,
    ADDR_HEALTH,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MAGIC_KEY,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)

UP = list(nes_action("UP"))
RIGHT = list(nes_action("RIGHT"))
LEFT = list(nes_action("LEFT"))
DOWN = list(nes_action("DOWN"))
IDLE = list(nes_idle_action())
DOORS_UP_DOWN = 12


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 8)
    ram[ADDR_SCREEN] = fields.get("screen", NORTH_3C_ORIGIN)
    ram[ADDR_LINK_X] = fields.get("x", NORTH_3C_ORIGIN_POSE[0])
    ram[ADDR_LINK_Y] = fields.get("y", NORTH_3C_ORIGIN_POSE[1])
    ram[ADDR_KEYS] = fields.get("keys", 8)
    ram[ADDR_BOMBS] = fields.get("bombs", 5)
    ram[ADDR_MAGIC_KEY] = fields.get("magic_key", 1)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x7F)
    ram[ADDR_HEALTH] = fields.get("health", 0x33)
    ram[ADDR_CUR_OPENED_DOORS] = fields.get("doors", DOORS_UP_DOWN)
    return ram


def _step(ctl, ram: np.ndarray):
    before = ram.copy()
    act = ctl.step(read_snapshot(ram))
    assert np.array_equal(ram, before), "north 3C hop must not write RAM"
    return act


def test_factory_fixture_live_not_passage() -> None:
    ctl = make_north_3c_controller()
    assert isinstance(ctl, Level8North3CController)
    assert ctl.report()["route_eligible"] is False
    assert ctl.report()["assumed_0x2c"] is False
    assert NORTH_3C_DEST == 0x2C
    assert NORTH_3C_DEST_POSE == (120, 205)
    assert ROOM_ITEM_TF == 0x1B
    assert NORTH_DOOR == (120, 93)
    passage = make_gleeok_passage_controller()
    shard = make_shard_leave_controller()
    assert isinstance(passage, UnverifiedLevel8PathController)
    assert isinstance(shard, UnverifiedLevel8PathController)


def test_leftover_walks_up_inland_not_down() -> None:
    ctl = make_north_3c_controller()
    act = _step(ctl, _ram())
    assert not ctl.failed and not ctl.success
    assert list(act.action) == UP
    assert act.reason == "north_inland"


def test_north_band_aligns_x_to_door() -> None:
    assert NORTH_BAND_Y == 141
    act = north_3c_step(read_snapshot(_ram(x=32, y=NORTH_BAND_Y)))
    assert list(act.action) == RIGHT
    assert act.reason == "north_align"
    act = north_3c_step(read_snapshot(_ram(x=32, y=133)))
    assert list(act.action) == RIGHT
    act = north_3c_step(read_snapshot(_ram(x=120, y=93)))
    assert list(act.action) == UP
    assert act.reason == "north_push"


def test_south_0x4c_fails_closed() -> None:
    ctl = make_north_3c_controller()
    act = _step(ctl, _ram(screen=SOUTH_FAIL, x=120, y=93))
    assert ctl.failed and not ctl.success
    assert list(act.action) == IDLE
    assert "south_0x4c" in ctl.notes


def test_does_not_lock_rom_0x2c() -> None:
    ctl = make_north_3c_controller(dest=None)
    assert ctl.dest is None
    ram = _ram(screen=0x2C, x=120, y=189)
    act = _step(ctl, ram)
    assert ctl.success and not ctl.failed
    assert "play_0x2c_120_189" in ctl.notes
    assert list(act.action) == IDLE


def test_shard_from_south_mouth_walks_up() -> None:
    ctl = make_shard_2c_controller()
    assert isinstance(ctl, Level8Shard2CController)
    ram = _ram(screen=NORTH_3C_DEST, x=120, y=205)
    act = _step(ctl, ram)
    assert not ctl.failed and not ctl.success
    assert list(act.action) == UP
    assert act.reason == "shard_approach"
    act = shard_2c_step(read_snapshot(_ram(screen=NORTH_3C_DEST, x=120, y=141)))
    assert list(act.action) == UP

"""L8 0x3C north shutter dest hop. No emulator. No RAM writes."""

from __future__ import annotations

import numpy as np
from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.level8.triforce import (
    NORTH_3C_DEST,
    NORTH_3C_ORIGIN,
    NORTH_3C_ORIGIN_POSE,
    NORTH_BAND_Y,
    SOUTH_FAIL,
    Level8Shard2CController,
    make_north_3c_controller,
    make_shard_2c_controller,
    north_3c_step,
    shard_2c_step,
)
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.tests.ram_helpers import make_ram

UP = list(nes_action("UP"))
RIGHT = list(nes_action("RIGHT"))
LEFT = list(nes_action("LEFT"))
DOWN = list(nes_action("DOWN"))
IDLE = list(nes_idle_action())
DOORS_UP_DOWN = 12


_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 8,
    "screen": NORTH_3C_ORIGIN,
    "x": NORTH_3C_ORIGIN_POSE[0],
    "y": NORTH_3C_ORIGIN_POSE[1],
    "keys": 8,
    "bombs": 5,
    "magic_key": 1,
    "triforce": 0x7F,
    "health": 0x33,
    "doors": DOORS_UP_DOWN,
}


def _ram(**fields: int) -> np.ndarray:
    return make_ram(_DEFAULTS, **fields)


def _step(ctl, ram: np.ndarray):
    before = ram.copy()
    act = ctl.step(read_snapshot(ram))
    assert np.array_equal(ram, before), "north 3C hop must not write RAM"
    return act


def test_leftover_walks_up_inland_not_down() -> None:
    """SW leftover (32,181) walks UP inland, never DOWN into open 0x4C bomb hole."""
    ctl = make_north_3c_controller()
    act = _step(ctl, _ram())
    assert not ctl.failed and not ctl.success
    assert list(act.action) == UP
    assert list(act.action) != DOWN
    assert list(act.action) != RIGHT
    assert act.reason == "north_inland"


def test_off_column_knockback_inland_does_not_up_into_wall() -> None:
    """x=208 leftover at inland y=133 must LEFT onto door column, never UP into wall."""
    ctl = make_north_3c_controller()
    act = _step(ctl, _ram(x=208, y=133))
    assert not ctl.failed and not ctl.success
    assert list(act.action) == LEFT
    assert list(act.action) != UP
    assert act.reason == "north_align"


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


def test_shard_refuses_a_state_that_already_holds_tf_0x80() -> None:
    """Post-shard pins (TF 0xFF) must not report a zero-input success."""
    ctl = make_shard_2c_controller()
    ram = _ram(screen=NORTH_3C_DEST, x=120, y=205, triforce=0xFF)
    act = _step(ctl, ram)
    assert ctl.failed is True and ctl.success is False
    assert "l8_shard_already_taken" in ctl.notes
    assert list(act.action) == IDLE
    assert ctl.report()["tf_in"] == 0xFF


def test_shard_greens_on_the_tf_rising_edge() -> None:
    ctl = make_shard_2c_controller()
    _step(ctl, _ram(screen=NORTH_3C_DEST, x=120, y=205))
    assert not ctl.success and not ctl.failed
    act = _step(ctl, _ram(screen=NORTH_3C_DEST, x=120, y=141, triforce=0xFF))
    assert ctl.success is True and not ctl.failed
    assert act.reason == "tf_0x80"

"""Durable L8 Magical Key cellar 0x0F return: east-drop, fail-closed, no writes.

No emulator. Fake snapshots at leftover (136,141) must emit RIGHT toward
the east ladder, never LEFT into pit tile 250. F1 cardinal DOWN did not
move. Spine magic-key factory stays UnverifiedLevel8PathController.
"""

from __future__ import annotations

import numpy as np
from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.level8.cellar import (
    CELLAR_RETURN_DEST,
    CELLAR_RETURN_POSE,
    CELLAR_ROOM,
    EAST_X,
    EXIT_STAIRS,
    FLOOR_Y,
    PAD,
    PIT_TILE,
    WEST_X,
    Level8MagicKeyCellarReturnController,
    magic_key_cellar_return_step,
    make_magic_key_cellar_return_controller,
)
from zelda_i.level8.path import (
    UnverifiedLevel8PathController,
    make_magic_key_stairs_controller,
)
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_COLLIDING_TILE,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MAGIC_KEY,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PASSAGE_MODE,
    PLAY_MODE,
    read_snapshot,
)

LEFT = list(nes_action("LEFT"))
RIGHT = list(nes_action("RIGHT"))
UP = list(nes_action("UP"))
IDLE = list(nes_idle_action())
LEFT_DOWN = list(nes_action("LEFT", "DOWN"))


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PASSAGE_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 8)
    ram[ADDR_SCREEN] = fields.get("screen", CELLAR_ROOM)
    ram[ADDR_LINK_X] = fields.get("x", PAD[0])
    ram[ADDR_LINK_Y] = fields.get("y", PAD[1])
    ram[ADDR_COLLIDING_TILE] = fields.get("tile", 36)
    ram[ADDR_KEYS] = fields.get("keys", 8)
    ram[ADDR_BOMBS] = fields.get("bombs", 6)
    ram[ADDR_MAGIC_KEY] = fields.get("magic_key", 1)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x7F)
    return ram


def _step(ctl, ram: np.ndarray):
    before = ram.copy()
    act = ctl.step(read_snapshot(ram))
    assert np.array_equal(ram, before), "cellar controllers must not write RAM"
    return act


def _no_strafe(act) -> None:
    assert list(act.action) != LEFT, act.reason
    assert list(act.action) != RIGHT, act.reason


def test_leftover_pad_goes_east_never_left_into_pit() -> None:
    snap = read_snapshot(_ram(x=136, y=141, tile=36))
    act = magic_key_cellar_return_step(snap)
    assert act.reason == "cellar_to_east"
    assert list(act.action) == RIGHT
    assert list(act.action) != LEFT

    ctl = make_magic_key_cellar_return_controller()
    act = _step(ctl, _ram(x=136, y=141, tile=36))
    assert not ctl.failed
    assert act.reason == "cellar_to_east"
    assert list(act.action) == RIGHT
    assert list(act.action) != LEFT


def test_east_column_drops_left_down() -> None:
    snap = read_snapshot(_ram(x=EAST_X, y=141))
    act = magic_key_cellar_return_step(snap)
    assert act.reason == "cellar_east_drop"
    assert list(act.action) == LEFT_DOWN

    ctl = make_magic_key_cellar_return_controller()
    act = _step(ctl, _ram(x=EAST_X, y=141))
    assert not ctl.failed
    assert act.reason == "cellar_east_drop"
    assert list(act.action) == LEFT_DOWN


def test_mid_ledge_keeps_right_until_east_column() -> None:
    """F2: x=160 is still the pit. Do not LEFT+DOWN until x>=174."""
    snap = read_snapshot(_ram(x=160, y=141, tile=36))
    act = magic_key_cellar_return_step(snap)
    assert act.reason == "cellar_to_east"
    assert list(act.action) == RIGHT
    assert list(act.action) != LEFT_DOWN

    ctl = make_magic_key_cellar_return_controller()
    act = _step(ctl, _ram(x=160, y=141, tile=36))
    assert not ctl.failed
    assert act.reason == "cellar_to_east"


def test_pit_tile_250_fails_closed_idle() -> None:
    ctl = make_magic_key_cellar_return_controller()
    act = _step(ctl, _ram(x=112, y=141, tile=PIT_TILE))
    assert ctl.failed and not ctl.success
    assert "pit_tile_250" in ctl.notes
    assert act.reason == "pit_tile_250"
    assert list(act.action) == IDLE


def test_floor_walks_west() -> None:
    act = magic_key_cellar_return_step(read_snapshot(_ram(x=100, y=FLOOR_Y)))
    assert act.reason == "cellar_floor_west"
    assert list(act.action) == LEFT

    ctl = make_magic_key_cellar_return_controller()
    act = _step(ctl, _ram(x=100, y=FLOOR_Y))
    assert not ctl.failed
    assert act.reason == "cellar_floor_west"
    assert list(act.action) == LEFT


def test_west_floor_climbs() -> None:
    act = magic_key_cellar_return_step(read_snapshot(_ram(x=WEST_X, y=FLOOR_Y)))
    assert act.reason == "cellar_west_climb"
    assert list(act.action) == UP

    ctl = make_magic_key_cellar_return_controller()
    act = _step(ctl, _ram(x=WEST_X, y=FLOOR_Y))
    assert not ctl.failed
    assert act.reason == "cellar_west_climb"
    assert list(act.action) == UP


def test_west_climb_holds_up() -> None:
    act = magic_key_cellar_return_step(read_snapshot(_ram(x=WEST_X, y=120)))
    assert act.reason == "cellar_west_up"
    assert list(act.action) == UP

    ctl = make_magic_key_cellar_return_controller()
    act = _step(ctl, _ram(x=WEST_X, y=120))
    assert not ctl.failed
    assert act.reason == "cellar_west_up"
    assert list(act.action) == UP


def test_west_lip_tile_0x6f_keeps_up() -> None:
    snap = read_snapshot(_ram(x=EXIT_STAIRS[0], y=EXIT_STAIRS[1], tile=0x6F))
    act = magic_key_cellar_return_step(snap)
    assert act.reason == "cellar_west_lip"
    assert list(act.action) == UP
    assert list(act.action) != IDLE

    ctl = make_magic_key_cellar_return_controller()
    act = _step(ctl, _ram(x=EXIT_STAIRS[0], y=EXIT_STAIRS[1], tile=0x6F))
    assert not ctl.failed
    assert act.reason == "cellar_west_lip"
    assert list(act.action) == UP


def test_stairs_tile_idles_exit_warp() -> None:
    snap = read_snapshot(_ram(x=EXIT_STAIRS[0], y=EXIT_STAIRS[1], tile=0x71))
    act = magic_key_cellar_return_step(snap)
    assert act.reason == "cellar_exit_warp"
    assert list(act.action) == IDLE

    ctl = make_magic_key_cellar_return_controller()
    act = _step(ctl, _ram(x=EXIT_STAIRS[0], y=EXIT_STAIRS[1], tile=0x71))
    assert not ctl.failed
    assert act.reason == "cellar_exit_warp"
    assert list(act.action) == IDLE


def test_play_mode_dest_none_succeeds_on_first_play() -> None:
    ctl = make_magic_key_cellar_return_controller(dest=None)
    act = _step(ctl, _ram(mode=PLAY_MODE, screen=0x1F, x=120, y=205))
    assert ctl.success and not ctl.failed
    assert list(act.action) == IDLE
    assert act.reason == "left_0x0f_stairs"


def test_play_mode_dest_0x1f_rejects_0x3c() -> None:
    ok = make_magic_key_cellar_return_controller(dest=0x1F)
    act = _step(ok, _ram(mode=PLAY_MODE, screen=0x1F, x=120, y=205))
    assert ok.success and not ok.failed
    assert act.reason == "left_0x0f_stairs"

    bad = make_magic_key_cellar_return_controller(dest=0x1F)
    act = _step(bad, _ram(mode=PLAY_MODE, screen=0x3C, x=120, y=205))
    assert not bad.success
    assert bad.failed
    assert "unexpected_play_0x3c" in bad.notes
    assert list(act.action) == IDLE


def test_factory_report_is_fixture_live_not_route_eligible() -> None:
    ctl = make_magic_key_cellar_return_controller()
    assert isinstance(ctl, Level8MagicKeyCellarReturnController)
    report = ctl.report()
    assert report["route_eligible"] is False
    assert report["dest_screen"] == CELLAR_RETURN_DEST == 0x1F
    assert CELLAR_RETURN_POSE == (96, 157)
    assert report["writes"] == 0
    assert report["evidence"] == "fixture-live"
    assert report["natural_entry"] is False
    assert report["door"] == "STAIRS"
    assert report["spec_id"] == "level8_magic_key_cellar_return"


def test_west_column_at_y141_goes_up_never_strafe() -> None:
    snap = read_snapshot(_ram(x=WEST_X, y=141))
    act = magic_key_cellar_return_step(snap)
    assert act.reason == "cellar_west_up"
    assert list(act.action) == UP
    _no_strafe(act)

    ctl = make_magic_key_cellar_return_controller()
    act = _step(ctl, _ram(x=WEST_X, y=141))
    assert not ctl.failed
    assert act.reason == "cellar_west_up"
    assert list(act.action) == UP
    _no_strafe(act)


def test_magic_key_stairs_factory_stays_unverified() -> None:
    ctl = make_magic_key_stairs_controller()
    assert isinstance(ctl, UnverifiedLevel8PathController)
    assert not isinstance(ctl, Level8MagicKeyCellarReturnController)

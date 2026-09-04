"""Durable L8 0x4C bomb-N. No emulator. No Gleeok fight.

Leftover (112,125) must not walk onto centre stairs. Dest is RAM; fail
cellar 0x2F and play 0x3F. No RAM writes. make_gleeok_passage_controller
stays UnverifiedLevel8PathController. Four-head factory stays idle-fail.
"""

from __future__ import annotations

import numpy as np
from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.level8.gleeok_entry import (
    BOMB_NORTH_APPROACH_4C,
    BOMB_NORTH_STAND,
    RAM_CLAIM,
    DEST,
    DEST_POSE,
    ORIGIN,
    ORIGIN_POSE,
    STAIRS_TILES,
    Level8BombNorth4CController,
    make_bomb_north_4c_controller,
)
from zelda_i.level8.path import (
    GLEEOK_HYP,
    UnverifiedLevel8PathController,
    make_four_head_gleeok_controller,
    make_gleeok_passage_controller,
)
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_COLLIDING_TILE,
    ADDR_CUR_OPENED_DOORS,
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

UP = list(nes_action("UP"))
DOWN = list(nes_action("DOWN"))
IDLE = list(nes_idle_action())


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 8)
    ram[ADDR_SCREEN] = fields.get("screen", ORIGIN)
    ram[ADDR_LINK_X] = fields.get("x", ORIGIN_POSE[0])
    ram[ADDR_LINK_Y] = fields.get("y", ORIGIN_POSE[1])
    ram[ADDR_COLLIDING_TILE] = fields.get("tile", 0)
    ram[ADDR_KEYS] = fields.get("keys", 8)
    ram[ADDR_BOMBS] = fields.get("bombs", 6)
    ram[ADDR_MAGIC_KEY] = fields.get("magic_key", 1)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x7F)
    ram[ADDR_CUR_OPENED_DOORS] = fields.get("doors", 0)
    return ram


def _step(ctl, ram: np.ndarray):
    before = ram.copy()
    act = ctl.step(read_snapshot(ram))
    assert np.array_equal(ram, before), "bomb-N 4C controllers must not write RAM"
    return act


def test_leftover_does_not_enter_centre_stairs() -> None:
    ctl = make_bomb_north_4c_controller(dest=None)
    act = _step(ctl, _ram(x=128, y=141, tile=0x71))
    assert ctl.failed
    assert "centre_stairs" in ctl.notes
    assert list(act.action) == IDLE


def test_leftover_goes_up_eight_px_then_left() -> None:
    """N1/N2 boxed N/S. First wp (48,117) is y-first UP then LEFT."""
    ctl = make_bomb_north_4c_controller(dest=None)
    act = _step(ctl, _ram(x=112, y=125, bombs=6))
    assert not ctl.failed
    assert list(act.action) == UP
    assert list(act.action) != DOWN
    assert ORIGIN_POSE == (112, 125)
    assert BOMB_NORTH_STAND == (120, 93)
    assert 0x71 in STAIRS_TILES


def test_dest_none_first_play_succeeds_cellar_and_3f_fail() -> None:
    ok = make_bomb_north_4c_controller(dest=None)
    act = _step(ok, _ram(mode=PLAY_MODE, screen=GLEEOK_HYP, x=120, y=205))
    assert ok.success and not ok.failed
    assert list(act.action) == IDLE

    cellar = make_bomb_north_4c_controller(dest=None)
    act = _step(cellar, _ram(mode=PASSAGE_MODE, screen=0x2F, x=192, y=93))
    assert not cellar.success and cellar.failed
    assert list(act.action) == IDLE

    src = make_bomb_north_4c_controller(dest=None)
    act = _step(src, _ram(mode=PLAY_MODE, screen=0x3F, x=32, y=141))
    assert not src.success and src.failed
    assert list(act.action) == IDLE


def test_dest_live_accepts_3c_rejects_2f_and_3f() -> None:
    ok = make_bomb_north_4c_controller(dest=DEST)
    act = _step(ok, _ram(mode=PLAY_MODE, screen=0x3C, x=120, y=189))
    assert ok.success and not ok.failed
    assert list(act.action) == IDLE

    cellar = make_bomb_north_4c_controller(dest=DEST)
    act = _step(cellar, _ram(mode=PASSAGE_MODE, screen=0x2F, x=192, y=93))
    assert not cellar.success and cellar.failed
    assert list(act.action) == IDLE

    src = make_bomb_north_4c_controller(dest=DEST)
    act = _step(src, _ram(mode=PLAY_MODE, screen=0x3F, x=32, y=141))
    assert not src.success and src.failed
    assert list(act.action) == IDLE
    assert DEST_POSE == (120, 189)
    assert DEST == 0x3C


def test_factory_report_fixture_live_not_route_eligible() -> None:
    ctl = make_bomb_north_4c_controller()
    assert isinstance(ctl, Level8BombNorth4CController)
    report = ctl.report()
    assert report["route_eligible"] is False
    assert report["door"] == "UP"
    assert report["gate"] == "bomb_north"
    assert report["writes"] == 0
    assert report["evidence"] == "fixture-live"
    assert report["natural_entry"] is False
    assert report["dest_screen"] == DEST == 0x3C


def test_gleeok_passage_stays_fail_closed_fight_waits_for_body() -> None:
    passage = make_gleeok_passage_controller()
    assert isinstance(passage, UnverifiedLevel8PathController)
    fight = make_four_head_gleeok_controller()
    ram = _ram(mode=PLAY_MODE, screen=GLEEOK_HYP, x=120, y=189)
    act = fight.step(read_snapshot(ram))
    assert not fight.failed
    assert not fight.success
    assert list(act.action) == IDLE
    assert act.reason == "wait_body"
    assert fight.report()["assumed_0x45"] is False
    assert fight.observed_body_type == 0x45


def test_disclosed_policy_matches_the_executed_stand() -> None:
    """The published ``policy`` string is the 0x4C stand, not 0x3E's (120,105)."""
    assert BOMB_NORTH_STAND == (120, 93)
    assert BOMB_NORTH_APPROACH_4C[-1] == (120, 109)
    assert "(120,93)" in RAM_CLAIM
    assert "(120,105)" not in RAM_CLAIM
    assert "(120,109)" in RAM_CLAIM
    assert make_bomb_north_4c_controller().report()["policy"] == RAM_CLAIM

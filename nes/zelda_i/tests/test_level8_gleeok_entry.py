"""Durable L8 0x4C bomb-N. No emulator. No Gleeok fight.

Leftover (112,125) must not walk onto centre stairs. Dest is RAM; fail
cellar 0x2F and play 0x3F. No RAM writes. make_gleeok_passage_controller
stays UnverifiedLevel8PathController. Four-head factory stays idle-fail.
"""

from __future__ import annotations

import numpy as np
from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.level8.gleeok_entry import (
    BOMB_NORTH_STAND,
    DEST,
    DEST_POSE,
    ORIGIN,
    ORIGIN_POSE,
    STAIRS_TILES,
    make_bomb_north_4c_controller,
)
from zelda_i.level8.path import (
    GLEEOK_HYP,
    UnverifiedLevel8PathController,
    make_four_head_gleeok_controller,
    make_gleeok_passage_controller,
)
from zelda_i.ram import PASSAGE_MODE, PLAY_MODE, read_snapshot
from zelda_i.tests.ram_helpers import make_ram

_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 8,
    "screen": ORIGIN,
    "x": ORIGIN_POSE[0],
    "y": ORIGIN_POSE[1],
    "tile": 0,
    "keys": 8,
    "bombs": 6,
    "magic_key": 1,
    "triforce": 0x7F,
    "doors": 0,
    "selected": 4,
}

UP = list(nes_action("UP"))
DOWN = list(nes_action("DOWN"))
IDLE = list(nes_idle_action())


def _ram(**fields: int) -> np.ndarray:
    return make_ram(_DEFAULTS, **fields)


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


def test_candle_leftover_pause_selects_bombs_before_place() -> None:
    from types import SimpleNamespace

    ram = _ram(x=BOMB_NORTH_STAND[0], y=BOMB_NORTH_STAND[1], bombs=6, selected=4)
    ctl = make_bomb_north_4c_controller(dest=None)
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    reasons: list[str] = []
    for _ in range(30):
        act = _step(ctl, ram)
        reasons.append(act.reason)
        if act.reason in {"place_bomb", "pause_open"}:
            break
    assert "pause_open" in reasons
    assert "place_bomb" not in reasons


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


def test_bomb_north_approach_waypoints_are_standable_on_0x4c() -> None:
    """A waypoint with no lattice node skips the lattice approach for the
    hand x-first press. (120, 109) put Link's feet on 0x4C's block row: run
    21 pressed LEFT into it at (176, 109) for 8000 frames and 71 hearts."""
    from zelda_i.dungeon.tilemap import ow_walkable_nodes
    from zelda_i.level8.gleeok_entry import BOMB_NORTH_APPROACH_4C
    from zelda_i.tests.ram_helpers import room_tile_ram
    from zelda_i.walk.physics import lattice_starts

    nodes = ow_walkable_nodes(room_tile_ram("0x4c", level=8), overworld=False)
    for wp in BOMB_NORTH_APPROACH_4C:
        assert any(n in nodes for n in lattice_starts(*wp)), wp

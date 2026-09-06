"""Durable L8 stairs walk-on from play 0x3F leftover (32,141). No emulator.

RIGHT along y=141 onto tiles 0x70-0x73, then idle. OccupancyWalker banned.
Dest is RAM; fail Magical Key cellar 0x0F and Gleeok 0x3C. No RAM writes.
make_gleeok_passage_controller stays UnverifiedLevel8PathController.
"""

from __future__ import annotations

import numpy as np
from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.level8.cellar import CELLAR_ROOM
from zelda_i.level8.path import (
    GLEEOK_HYP,
    UnverifiedLevel8PathController,
    make_gleeok_passage_controller,
)
from zelda_i.level8.stairs import (
    STAIRS_3F_DEST,
    STAIRS_3F_DEST_HYP,
    STAIRS_3F_DEST_MODE,
    STAIRS_3F_DEST_POSE,
    STAIRS_3F_HYP_XY,
    STAIRS_3F_ORIGIN,
    STAIRS_3F_ORIGIN_POSE,
    Level8Stairs3FController,
    make_stairs_3f_controller,
    stairs_3f_step,
)
from zelda_i.ram import PASSAGE_MODE, PLAY_MODE, read_snapshot
from zelda_i.tests.ram_helpers import make_ram

_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 8,
    "screen": STAIRS_3F_ORIGIN,
    "x": STAIRS_3F_ORIGIN_POSE[0],
    "y": STAIRS_3F_ORIGIN_POSE[1],
    "tile": 0,
    "keys": 8,
    "bombs": 6,
    "magic_key": 1,
    "triforce": 0x7F,
}

LEFT = list(nes_action("LEFT"))
RIGHT = list(nes_action("RIGHT"))
DOWN = list(nes_action("DOWN"))
UP = list(nes_action("UP"))
IDLE = list(nes_idle_action())


def _ram(**fields: int) -> np.ndarray:
    return make_ram(_DEFAULTS, **fields)


def _step(ctl, ram: np.ndarray):
    before = ram.copy()
    act = ctl.step(read_snapshot(ram))
    assert np.array_equal(ram, before), "stairs 3F controllers must not write RAM"
    return act


def test_leftover_emits_right_not_down_or_up() -> None:
    """(32,141) walks RIGHT toward the east stairs. Not DOWN/UP/LEFT."""
    snap = read_snapshot(_ram(x=32, y=141))
    act = stairs_3f_step(snap)
    assert list(act.action) == RIGHT
    assert list(act.action) != DOWN
    assert list(act.action) != UP
    assert list(act.action) != LEFT
    assert act.reason == "stairs_x"

    ctl = make_stairs_3f_controller(dest=None)
    act = _step(ctl, _ram(x=32, y=141))
    assert not ctl.failed
    assert list(act.action) == RIGHT
    assert list(act.action) != DOWN
    assert act.reason == "stairs_x"


def test_stair_tile_idles_for_exact_checkwarp() -> None:
    snap = read_snapshot(_ram(x=176, y=141, tile=0x71))
    act = stairs_3f_step(snap)
    assert list(act.action) == IDLE
    assert act.reason == "stairs_stand"

    ctl = make_stairs_3f_controller(dest=None)
    act = _step(ctl, _ram(x=176, y=141, tile=0x70))
    assert not ctl.failed
    assert list(act.action) == IDLE
    assert act.reason == "stairs_stand"


def test_dest_none_mode9_succeeds_gleeok_and_mk_cellar_fail() -> None:
    ok = make_stairs_3f_controller(dest=None)
    act = _step(
        ok, _ram(mode=PASSAGE_MODE, screen=STAIRS_3F_DEST_HYP, x=136, y=141)
    )
    assert ok.success and not ok.failed
    assert list(act.action) == IDLE

    still = make_stairs_3f_controller(dest=None)
    act = _step(still, _ram(mode=PLAY_MODE, screen=STAIRS_3F_ORIGIN, x=176, y=141))
    assert not still.success and not still.failed
    assert list(act.action) in (RIGHT, UP, IDLE)

    gleeok = make_stairs_3f_controller(dest=None)
    act = _step(gleeok, _ram(mode=PLAY_MODE, screen=GLEEOK_HYP, x=120, y=141))
    assert not gleeok.success and gleeok.failed
    assert "gleeok_0x3c" in gleeok.notes
    assert list(act.action) == IDLE

    cellar = make_stairs_3f_controller(dest=None)
    act = _step(
        cellar, _ram(mode=PASSAGE_MODE, screen=CELLAR_ROOM, x=136, y=141)
    )
    assert not cellar.success and cellar.failed
    assert list(act.action) == IDLE


def test_dest_live_accepts_2f_rejects_3c_and_0f() -> None:
    ok = make_stairs_3f_controller(dest=STAIRS_3F_DEST)
    act = _step(
        ok, _ram(mode=PASSAGE_MODE, screen=0x2F, x=208, y=141, tile=0x71)
    )
    assert ok.success and not ok.failed
    assert list(act.action) == IDLE

    none = make_stairs_3f_controller(dest=None)
    act = _step(none, _ram(mode=PASSAGE_MODE, screen=0x2F, x=208, y=141))
    assert none.success and not none.failed

    bad = make_stairs_3f_controller(dest=STAIRS_3F_DEST)
    act = _step(bad, _ram(mode=PLAY_MODE, screen=GLEEOK_HYP, x=120, y=141))
    assert not bad.success and bad.failed
    assert list(act.action) == IDLE

    mk = make_stairs_3f_controller(dest=STAIRS_3F_DEST)
    act = _step(mk, _ram(mode=PASSAGE_MODE, screen=CELLAR_ROOM, x=136, y=141))
    assert not mk.success and mk.failed
    assert list(act.action) == IDLE
    assert STAIRS_3F_DEST_POSE == (208, 141)
    assert STAIRS_3F_DEST_MODE == 9


def test_factory_report_fixture_live_not_route_eligible() -> None:
    ctl = make_stairs_3f_controller()
    assert isinstance(ctl, Level8Stairs3FController)
    report = ctl.report()
    assert report["route_eligible"] is False
    assert report["door"] == "STAIRS"
    assert report["writes"] == 0
    assert report["evidence"] == "fixture-live"
    assert report["natural_entry"] is False
    assert report["spec_id"] == "level8_stairs_3f"
    assert report["dest_screen"] == STAIRS_3F_DEST == 0x2F
    assert STAIRS_3F_DEST_HYP == STAIRS_3F_DEST == 0x2F
    assert STAIRS_3F_DEST != GLEEOK_HYP
    assert STAIRS_3F_DEST != CELLAR_ROOM
    assert STAIRS_3F_DEST_POSE == (208, 141)
    assert STAIRS_3F_DEST_MODE == PASSAGE_MODE == 9
    assert STAIRS_3F_ORIGIN == 0x3F
    assert STAIRS_3F_ORIGIN_POSE == (32, 141)
    assert STAIRS_3F_HYP_XY == (208, 93)


def test_gleeok_passage_factory_stays_unverified() -> None:
    ctl = make_gleeok_passage_controller()
    assert isinstance(ctl, UnverifiedLevel8PathController)
    assert not isinstance(ctl, Level8Stairs3FController)

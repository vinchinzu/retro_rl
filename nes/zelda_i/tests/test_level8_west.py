"""Durable L8 west gate from play 0x1F leftover (96,157). No emulator.

Cardinal LEFT past 0x68, then y-align, then LEFT push. G1 occupancy
1px-grade boxed at (88,157) tile 118. Dest is RAM; fail cellar 0x0F and
Gleeok 0x3C. No RAM writes.
make_gleeok_passage_controller stays UnverifiedLevel8PathController.
"""

from __future__ import annotations

import numpy as np
from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.level8.path import (
    CELLAR_ROOM,
    GLEEOK_HYP,
    Level8West1FController,
    UnverifiedLevel8PathController,
    STAIRS_WEST_X,
    WEST_DEST,
    WEST_DEST_POSE,
    WEST_DOOR,
    WEST_ORIGIN,
    WEST_ORIGIN_POSE,
    make_gleeok_passage_controller,
    make_west_1f_controller,
    west_1f_step,
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
UP = list(nes_action("UP"))
IDLE = list(nes_idle_action())


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 8)
    ram[ADDR_SCREEN] = fields.get("screen", WEST_ORIGIN)
    ram[ADDR_LINK_X] = fields.get("x", WEST_ORIGIN_POSE[0])
    ram[ADDR_LINK_Y] = fields.get("y", WEST_ORIGIN_POSE[1])
    ram[ADDR_COLLIDING_TILE] = fields.get("tile", 0)
    ram[ADDR_KEYS] = fields.get("keys", 8)
    ram[ADDR_BOMBS] = fields.get("bombs", 6)
    ram[ADDR_MAGIC_KEY] = fields.get("magic_key", 1)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x7F)
    return ram


def _step(ctl, ram: np.ndarray):
    before = ram.copy()
    act = ctl.step(read_snapshot(ram))
    assert np.array_equal(ram, before), "west 1F controllers must not write RAM"
    return act


def test_leftover_emits_left_not_up_into_stairs() -> None:
    snap = read_snapshot(_ram(x=96, y=157))
    act = west_1f_step(snap)
    assert list(act.action) == LEFT
    assert list(act.action) != UP
    assert act.reason == "west_clear_stairs"

    ctl = make_west_1f_controller()
    act = _step(ctl, _ram(x=96, y=157))
    assert not ctl.failed
    assert list(act.action) == LEFT
    assert list(act.action) != UP
    assert act.reason == "west_clear_stairs"


def test_west_door_emits_left_push() -> None:
    ctl = make_west_1f_controller()
    act = _step(ctl, _ram(x=WEST_DOOR[0], y=WEST_DOOR[1]))
    assert not ctl.failed
    assert list(act.action) == LEFT
    assert act.reason == "west_push"


def test_y_align_only_after_west_of_stairs() -> None:
    """x=64 y=157 may UP toward 141; leftover x=96 must LEFT."""
    assert STAIRS_WEST_X == 80
    west = make_west_1f_controller()
    act = _step(west, _ram(x=64, y=157))
    assert not west.failed
    assert list(act.action) == UP
    assert act.reason == "west_align"

    leftover = make_west_1f_controller()
    act = _step(leftover, _ram(x=96, y=157))
    assert not leftover.failed
    assert list(act.action) == LEFT
    assert list(act.action) != UP
    assert act.reason == "west_clear_stairs"


def test_dest_none_first_play_succeeds_gleeok_and_cellar_fail() -> None:
    ok = make_west_1f_controller(dest=None)
    act = _step(ok, _ram(mode=PLAY_MODE, screen=0x1E, x=208, y=141))
    assert ok.success and not ok.failed
    assert list(act.action) == IDLE

    gleeok = make_west_1f_controller(dest=None)
    act = _step(gleeok, _ram(mode=PLAY_MODE, screen=GLEEOK_HYP, x=120, y=141))
    assert not gleeok.success and gleeok.failed
    assert "gleeok_0x3c" in gleeok.notes
    assert list(act.action) == IDLE

    cellar_mode = make_west_1f_controller(dest=None)
    act = _step(
        cellar_mode, _ram(mode=PASSAGE_MODE, screen=CELLAR_ROOM, x=136, y=141)
    )
    assert not cellar_mode.success and cellar_mode.failed
    assert list(act.action) == IDLE

    cellar_screen = make_west_1f_controller(dest=None)
    act = _step(
        cellar_screen, _ram(mode=PLAY_MODE, screen=CELLAR_ROOM, x=136, y=141)
    )
    assert not cellar_screen.success and cellar_screen.failed
    assert list(act.action) == IDLE


def test_dest_0x1e_accepts_1e_rejects_3c() -> None:
    ok = make_west_1f_controller(dest=0x1E)
    act = _step(ok, _ram(mode=PLAY_MODE, screen=0x1E, x=208, y=141))
    assert ok.success and not ok.failed
    assert list(act.action) == IDLE

    bad = make_west_1f_controller(dest=0x1E)
    act = _step(bad, _ram(mode=PLAY_MODE, screen=GLEEOK_HYP, x=120, y=141))
    assert not bad.success and bad.failed
    assert list(act.action) == IDLE


def test_factory_report_fixture_live_not_route_eligible() -> None:
    ctl = make_west_1f_controller()
    assert isinstance(ctl, Level8West1FController)
    report = ctl.report()
    assert report["route_eligible"] is False
    assert report["door"] == "LEFT"
    assert report["writes"] == 0
    assert report["evidence"] == "fixture-live"
    assert report["natural_entry"] is False
    assert report["spec_id"] == "level8_west_1f"
    assert report["dest_screen"] == WEST_DEST == 0x1E
    assert WEST_DEST_POSE == (208, 141)
    assert WEST_DEST != 0x3C
    assert WEST_DOOR == (32, 141)
    assert WEST_ORIGIN_POSE == (96, 157)


def test_gleeok_passage_factory_stays_unverified() -> None:
    ctl = make_gleeok_passage_controller()
    assert isinstance(ctl, UnverifiedLevel8PathController)
    assert not isinstance(ctl, Level8West1FController)

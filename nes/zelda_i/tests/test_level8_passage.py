"""Durable L8 cellar 0x2F B→A cross: DOWN, floor LEFT, never source UP.

No emulator. Fake snapshots at the east-ladder spawn (192,93) must emit
DOWN, never UP (returns to play 0x3F). Dest is RAM; fail source 0x3F and
Gleeok 0x3C. No RAM writes. make_gleeok_passage_controller stays
UnverifiedLevel8PathController.
"""

from __future__ import annotations

import numpy as np
from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.level8.passage import (
    CELLAR_ROOM,
    DEST,
    DEST_HYP,
    DEST_POSE,
    EAST_X,
    FLOOR_Y,
    MOUTH_Y,
    RAM_CLAIM,
    SOURCE_ROOM,
    SPAWN_XY,
    WEST_X,
    Level8Passage2FController,
    make_passage_2f_controller,
    passage_2f_step,
)
from zelda_i.level8.path import (
    GLEEOK_HYP,
    UnverifiedLevel8PathController,
    make_gleeok_passage_controller,
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

DOWN = list(nes_action("DOWN"))
LEFT = list(nes_action("LEFT"))
RIGHT = list(nes_action("RIGHT"))
UP = list(nes_action("UP"))
IDLE = list(nes_idle_action())


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PASSAGE_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 8)
    ram[ADDR_SCREEN] = fields.get("screen", CELLAR_ROOM)
    ram[ADDR_LINK_X] = fields.get("x", SPAWN_XY[0])
    ram[ADDR_LINK_Y] = fields.get("y", SPAWN_XY[1])
    ram[ADDR_COLLIDING_TILE] = fields.get("tile", 0x6F)
    ram[ADDR_KEYS] = fields.get("keys", 8)
    ram[ADDR_BOMBS] = fields.get("bombs", 6)
    ram[ADDR_MAGIC_KEY] = fields.get("magic_key", 1)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x7F)
    return ram


def _step(ctl, ram: np.ndarray):
    before = ram.copy()
    act = ctl.step(read_snapshot(ram))
    assert np.array_equal(ram, before), "passage 2F controllers must not write RAM"
    return act


def test_leftover_emits_down_never_up() -> None:
    snap = read_snapshot(_ram(x=EAST_X, y=MOUTH_Y))
    act = passage_2f_step(snap)
    assert list(act.action) == DOWN
    assert list(act.action) != UP
    assert act.reason == "cellar_east_drop"

    ctl = make_passage_2f_controller(dest=None)
    act = _step(ctl, _ram(x=EAST_X, y=MOUTH_Y))
    assert not ctl.failed
    assert list(act.action) == DOWN
    assert list(act.action) != UP
    assert act.reason == "cellar_east_drop"


def test_floor_goes_left_west_climbs_up() -> None:
    floor = make_passage_2f_controller(dest=None)
    act = _step(floor, _ram(x=EAST_X, y=FLOOR_Y))
    assert not floor.failed
    assert list(act.action) == LEFT
    assert act.reason == "cellar_floor_west"

    west = make_passage_2f_controller(dest=None)
    act = _step(west, _ram(x=WEST_X, y=FLOOR_Y))
    assert not west.failed
    assert list(act.action) == UP
    assert act.reason == "cellar_west_climb"


def test_east_column_mid_still_drops_not_up() -> None:
    ctl = make_passage_2f_controller(dest=None)
    act = _step(ctl, _ram(x=EAST_X, y=125))
    assert not ctl.failed
    assert list(act.action) == DOWN
    assert list(act.action) != UP
    assert act.reason == "cellar_east_drop"


def test_dest_none_first_play_succeeds_source_and_gleeok_fail() -> None:
    ok = make_passage_2f_controller(dest=None)
    act = _step(ok, _ram(mode=PLAY_MODE, screen=DEST_HYP, x=16, y=141))
    assert ok.success and not ok.failed
    assert list(act.action) == IDLE

    src = make_passage_2f_controller(dest=None)
    act = _step(src, _ram(mode=PLAY_MODE, screen=SOURCE_ROOM, x=32, y=141))
    assert not src.success and src.failed
    assert "returned_source_0x3f" in src.notes
    assert list(act.action) == IDLE

    gleeok = make_passage_2f_controller(dest=None)
    act = _step(gleeok, _ram(mode=PLAY_MODE, screen=GLEEOK_HYP, x=120, y=141))
    assert not gleeok.success and gleeok.failed
    assert "gleeok_0x3c" in gleeok.notes
    assert list(act.action) == IDLE


def test_dest_live_accepts_4c_rejects_3c_and_3f() -> None:
    ok = make_passage_2f_controller(dest=DEST)
    act = _step(ok, _ram(mode=PLAY_MODE, screen=0x4C, x=112, y=125))
    assert ok.success and not ok.failed
    assert list(act.action) == IDLE

    bad = make_passage_2f_controller(dest=DEST)
    act = _step(bad, _ram(mode=PLAY_MODE, screen=GLEEOK_HYP, x=120, y=141))
    assert not bad.success and bad.failed
    assert list(act.action) == IDLE

    src = make_passage_2f_controller(dest=DEST)
    act = _step(src, _ram(mode=PLAY_MODE, screen=SOURCE_ROOM, x=32, y=141))
    assert not src.success and src.failed
    assert list(act.action) == IDLE
    assert DEST_POSE == (112, 125)


def test_factory_report_fixture_live_not_route_eligible() -> None:
    ctl = make_passage_2f_controller()
    assert isinstance(ctl, Level8Passage2FController)
    report = ctl.report()
    assert report["route_eligible"] is False
    assert report["door"] == "STAIRS"
    assert report["writes"] == 0
    assert report["evidence"] == "fixture-live"
    assert report["natural_entry"] is False
    assert report["spec_id"] == "level8_passage_2f"
    assert report["dest_screen"] == DEST == 0x4C
    assert DEST_HYP == DEST == 0x4C
    assert DEST != GLEEOK_HYP
    assert DEST != SOURCE_ROOM
    assert DEST_POSE == (112, 125)
    assert SPAWN_XY == (192, 93)
    assert "0x3F" in RAM_CLAIM
    assert "Never UP" in RAM_CLAIM


def test_gleeok_passage_factory_stays_unverified() -> None:
    ctl = make_gleeok_passage_controller()
    assert isinstance(ctl, UnverifiedLevel8PathController)
    assert not isinstance(ctl, Level8Passage2FController)

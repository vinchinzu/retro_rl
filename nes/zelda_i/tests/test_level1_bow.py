"""Unit tests for L1 0x23 KEY-LEFT into ROM bow room 0x22."""

from __future__ import annotations

import numpy as np

from zelda_i.level1_bow import (
    LEVEL1_BOW_ROOM,
    NORTH_BAND_Y,
    NORTH_JOIN_X,
    WEST_AISLE_X,
    WEST_DOOR_X,
    WEST_DOOR_Y,
    level1_bow_stages,
    level1_bow_success,
    level1_to_clear23_stages,
    make_bow22_controller,
)
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOW,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.survival_spine import SPINE_THROUGH, SpineRun


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 1)
    ram[ADDR_SCREEN] = fields.get("screen", 0x23)
    ram[ADDR_LINK_X] = fields.get("x", 114)
    ram[ADDR_LINK_Y] = fields.get("y", 117)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0)
    ram[ADDR_KEYS] = fields.get("keys", 1)
    ram[ADDR_BOW] = fields.get("bow", 0)
    ram[ADDR_ARROWS] = fields.get("arrows", 0)
    return ram


def test_bow_through_is_wired_and_stops_before_tf() -> None:
    assert "level1-bow" in SPINE_THROUGH
    assert SPINE_THROUGH.index("level1-bow") == SPINE_THROUGH.index("level1") + 1
    names = [name for name, _, _ in level1_to_clear23_stages()]
    assert names[0] == "clear52"
    assert names[-1] == "clear23_key"
    assert "backtrack44" not in names
    stages = level1_bow_stages()
    assert [name for name, _, _ in stages][-1] == "level1_bow_0x22"
    run = SpineRun(through="level1-bow", success=True, boot_frames=199)
    assert run.report()["stop"] == "level1_bow_0x22"


def test_bow22_occupancy_north_wall_around() -> None:
    from retro_harness.nes import nes_action

    leftover = _ram(x=136, y=117)
    ctl = make_bow22_controller()
    act = ctl.step(read_snapshot(leftover))
    assert act.reason == "west_path"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("DOWN"))
    # Past the x=80 join: do not keep LEFT into aisle x=64 (westwall v2 solid).
    col = _ram(x=NORTH_JOIN_X, y=117)
    climb = make_bow22_controller()
    act = climb.step(read_snapshot(col))
    assert act.reason == "north_band"
    assert list(act.action) == list(nes_action("UP"))
    moat = _ram(x=WEST_AISLE_X, y=WEST_DOOR_Y)
    peel = make_bow22_controller()
    act = peel.step(read_snapshot(moat))
    assert act.reason == "north_band"
    assert list(act.action) == list(nes_action("UP"))
    assert list(act.action) != list(nes_action("LEFT", "UP"))
    band = _ram(x=NORTH_JOIN_X, y=NORTH_BAND_Y)
    wall = make_bow22_controller()
    act = wall.step(read_snapshot(band))
    assert act.reason == "west_wall"
    assert list(act.action) == list(nes_action("LEFT"))
    nw = _ram(x=WEST_DOOR_X, y=NORTH_BAND_Y)
    drop = make_bow22_controller()
    act = drop.step(read_snapshot(nw))
    assert act.reason == "door_drop"
    assert list(act.action) == list(nes_action("DOWN"))
    door = _ram(x=WEST_DOOR_X, y=WEST_DOOR_Y)
    push = make_bow22_controller()
    act = push.step(read_snapshot(door))
    assert act.reason == "west_push"
    assert list(act.action) == list(nes_action("LEFT"))


def test_bow22_requires_exact_play_0x22() -> None:
    dest = _ram(screen=LEVEL1_BOW_ROOM, x=224, y=141, keys=0)
    assert level1_bow_success(read_snapshot(dest))
    ctl = make_bow22_controller()
    ctl.keys = 1
    act = ctl.step(read_snapshot(dest))
    assert ctl.success
    assert not ctl.failed
    assert act.reason == "arrived_22"
    assert any(n.startswith("key_spent_23_to_22_1->0") for n in ctl.notes)
    still = _ram()
    assert not level1_bow_success(read_snapshot(still))
    back = _ram(screen=0x33, x=120, y=93)
    assert not level1_bow_success(read_snapshot(back))
    backtrack = make_bow22_controller()
    fail_s = backtrack.step(read_snapshot(back))
    assert backtrack.failed
    assert fail_s.reason.startswith("backtrack_33")
    snap = read_snapshot(dest)
    assert snap.bow == 0
    assert snap.arrows == 0

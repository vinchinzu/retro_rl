"""Unit tests for L1 0x23 KEY-LEFT into ROM bow room 0x22."""

from __future__ import annotations

import numpy as np

from zelda_i.level1.bow import (
    LEVEL1_BOW_ROOM,
    NORTH_BAND_Y,
    NORTH_JOIN_X,
    WEST_AISLE_X,
    WEST_DOOR_X,
    WEST_DOOR_Y,
    level1_bow_success,
    make_bow22_controller,
)
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.tests.ram_helpers import make_ram

_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 1,
    "screen": 0x23,
    "x": 114,
    "y": 117,
    "triforce": 0,
    "keys": 1,
    "bow": 0,
    "arrows": 0,
}


def _ram(**fields: int) -> np.ndarray:
    return make_ram(_DEFAULTS, **fields)


def test_bow22_occupancy_plus_stem_x112() -> None:
    from retro_harness.nes import nes_action

    leftover = _ram(x=136, y=117)
    ctl = make_bow22_controller()
    act = ctl.step(read_snapshot(leftover))
    assert act.reason == "west_path"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("DOWN"))
    # Westwall v3 leftover: UP at x=80 is tile 119. RIGHT back to plus stem.
    aisle = _ram(x=80, y=117)
    recover = make_bow22_controller()
    act = recover.step(read_snapshot(aisle))
    assert act.reason == "west_path"
    assert list(act.action) == list(nes_action("RIGHT"))
    assert list(act.action) != list(nes_action("UP"))
    col = _ram(x=NORTH_JOIN_X, y=117)
    climb = make_bow22_controller()
    act = climb.step(read_snapshot(col))
    assert act.reason == "north_band"
    assert list(act.action) == list(nes_action("UP"))
    moat = _ram(x=WEST_AISLE_X, y=WEST_DOOR_Y)
    peel = make_bow22_controller()
    act = peel.step(read_snapshot(moat))
    assert act.reason == "west_path"
    assert list(act.action) != list(nes_action("LEFT"))
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
    # x112 v1 leftover: west landing (32,117) must DOWN, not RIGHT to x=112.
    drop_mid = _ram(x=WEST_DOOR_X, y=117)
    mid = make_bow22_controller()
    act = mid.step(read_snapshot(drop_mid))
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

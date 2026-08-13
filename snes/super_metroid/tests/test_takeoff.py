"""Unit tests for shared in-room takeoff windows (no emulator)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from retro_harness.controls import (
    SNES_DPAD_LEFT,
    SNES_DPAD_RIGHT,
    SNES_L,
    SNES_LEFT,
    SNES_R,
    SNES_RIGHT,
    SNES_SHOULDER_L,
    SNES_SHOULDER_R,
)
from super_metroid.ram import FACING_LEFT, FACING_RIGHT, parse_state
from super_metroid.takeoff import (
    PlatformHop,
    TakeoffWindow,
    approach_window,
    hop_for_y,
    next_hop_above,
    shoulder_pump_button,
    walk_toward_x,
)


def _state(**overrides):
    base = parse_state(np.zeros(0x2000, dtype=np.uint8), frame=0)
    return replace(base, **overrides)


def test_takeoff_uses_door_kinematics_requirement() -> None:
    window = TakeoffWindow((70, 110), "RIGHT", min_momentum=2)
    req = window.requirement()
    assert req.x_range == (70, 110)
    assert req.min_abs_momentum == 2
    assert FACING_RIGHT in req.facings

    cold = _state(samus_x=80, facing=FACING_RIGHT, momentum_x=0)
    assert not window.ready(cold)
    facing_left = _state(samus_x=80, facing=FACING_LEFT, momentum_x=3)
    assert not window.ready(facing_left)
    ready = _state(samus_x=80, facing=FACING_RIGHT, momentum_x=2, samus_x_sub=100)
    assert window.ready(ready)


def test_platform_hop_roundtrip_and_legacy_flat_dict() -> None:
    hop = PlatformHop(571, 40, 130, TakeoffWindow((70, 110), "RIGHT"))
    again = PlatformHop.from_dict(hop.to_dict())
    assert again.y == 571
    assert again.takeoff.x_range == (70, 110)
    assert again.side == "RIGHT"

    legacy = PlatformHop.from_dict(
        {
            "y": 475,
            "x_lo": 90,
            "x_hi": 180,
            "side": "RIGHT",
            "x_jump_lo": 118,
            "x_jump_hi": 158,
            "min_momentum": 1,
        }
    )
    assert legacy.y == 475
    assert legacy.takeoff.x_range == (118, 158)


def test_hop_for_y_and_next_above() -> None:
    hops = (
        PlatformHop(571, 40, 130, TakeoffWindow((70, 110), "RIGHT")),
        PlatformHop(475, 90, 180, TakeoffWindow((118, 158), "RIGHT")),
        PlatformHop(363, 150, 220, TakeoffWindow((165, 205), "LEFT")),
    )
    assert hop_for_y(571, hops) is hops[0]
    assert hop_for_y(200, hops) is None
    assert next_hop_above(571, hops) is hops[1]
    assert next_hop_above(363, hops) is None


def test_ledge_end_depends_on_side() -> None:
    right = PlatformHop(571, 40, 130, TakeoffWindow((70, 110), "RIGHT"))
    left = PlatformHop(363, 150, 220, TakeoffWindow((165, 205), "LEFT"))
    assert right.at_ledge_end(120)
    assert not right.at_ledge_end(80)
    assert left.at_ledge_end(160)
    assert not left.at_ledge_end(190)


def test_takeoff_rejects_bad_side() -> None:
    with pytest.raises(ValueError, match="LEFT or RIGHT"):
        TakeoffWindow((0, 10), "UP")
    with pytest.raises(ValueError, match="D-pad LEFT/RIGHT"):
        TakeoffWindow((0, 10), "L")
    with pytest.raises(ValueError, match="D-pad LEFT/RIGHT"):
        TakeoffWindow((0, 10), "R")


def test_dpad_and_shoulder_names_do_not_collide() -> None:
    # Wire names: D-pad is LEFT/RIGHT (indices 6/7); shoulders are L/R (10/11).
    assert SNES_DPAD_LEFT == "LEFT"
    assert SNES_SHOULDER_L == "L"
    assert SNES_LEFT != SNES_L
    assert SNES_RIGHT != SNES_R
    assert SNES_DPAD_LEFT != SNES_SHOULDER_L
    assert SNES_DPAD_RIGHT != SNES_SHOULDER_R
    assert shoulder_pump_button(0) == SNES_SHOULDER_L
    assert shoulder_pump_button(2) == SNES_SHOULDER_R
    assert walk_toward_x(20, 145) == (SNES_DPAD_RIGHT,)
    assert walk_toward_x(200, 145) == (SNES_DPAD_LEFT,)


def test_approach_uses_dpad_side_and_shoulder_pump() -> None:
    hop = PlatformHop(571, 40, 130, TakeoffWindow((70, 110), "RIGHT"))
    ready = _state(
        samus_x=50, facing=FACING_RIGHT, momentum_x=2, speed_flag=1
    )
    names, nxt = approach_window(ready, hop, pump_i=0)
    assert names[0] == SNES_DPAD_RIGHT
    assert "B" in names
    assert SNES_SHOULDER_L in names
    assert SNES_DPAD_LEFT not in names
    assert nxt == 1

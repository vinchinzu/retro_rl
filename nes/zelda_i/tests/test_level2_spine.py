"""Unit tests for the L2 Survival-spine 0x7d → Magical Boomerang table."""

from __future__ import annotations

import numpy as np

from retro_harness.controls import pressed_nes_buttons
from zelda_i.level2_spine import (
    Level2BacktrackTo7dController,
    Level2Enter6fKeyController,
    Level2NavPhase,
)
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_HEALTH,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)


def _snap(
    *,
    room: int,
    x: int = 120,
    y: int = 141,
    keys: int = 2,
    bombs: int = 0,
    mode: int = PLAY_MODE,
):
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_LEVEL] = 2
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_KEYS] = keys
    ram[ADDR_BOMBS] = bombs
    ram[ADDR_HEALTH] = 0x2F
    return read_snapshot(ram)


def test_backtrack_7d_recenters_live_timeout_pose() -> None:
    """survival_spine_l2_boom timed out in 0x6c at (128, 133) with ALIGN_TOL=6."""
    ctl = Level2BacktrackTo7dController()
    act = ctl.step(_snap(room=0x6C, x=128, y=133))
    assert act.reason == "align_door_y"
    act = ctl.step(_snap(room=0x6C, x=136, y=136))
    assert act.reason == "align_door_y"
    act = ctl.step(_snap(room=0x6C, x=136, y=141))
    assert act.reason == "push_door"


def test_enter_6f_fails_without_keys() -> None:
    ctl = Level2Enter6fKeyController()
    act = ctl.step(_snap(room=0x6E, keys=0))
    assert ctl.phase is Level2NavPhase.FAILED
    assert act.reason == "no_keys"
    assert ctl.success is False
    pushing = Level2Enter6fKeyController()
    pushing.door_phase = "push"
    act = pushing.step(_snap(room=0x6E, x=208, y=141, keys=0))
    assert pushing.phase is Level2NavPhase.WALK
    assert act.reason == "push_r"


def test_enter_6f_south_occupancy_sidesteps_diamonds() -> None:
    """Live timeout sat at (72, 181) then (112, 181); greedy UP hits diamonds."""
    ctl = Level2Enter6fKeyController()
    snap = _snap(room=0x6E, x=112, y=181, keys=2)
    first = ctl.step(snap)
    assert first.reason == "band_occ"
    first_dir = pressed_nes_buttons(list(first.action))
    second = ctl.step(snap)
    assert ctl.walker.misses == 1
    assert second.reason == "band_occ"
    assert pressed_nes_buttons(list(second.action)) != first_dir
    east_south = Level2Enter6fKeyController()
    east_snap = _snap(room=0x6E, x=200, y=181, keys=2)
    act = east_south.step(east_snap)
    assert act.reason == "band_occ"
    assert "RIGHT" not in pressed_nes_buttons(list(act.action))
    east_south.step(east_snap)
    assert east_south.walker.misses == 1

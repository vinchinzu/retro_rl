"""Unit tests for golf action builders."""

from __future__ import annotations

import numpy as np

from hals_golf.core.actions import ACTION_SIZE, idle, named_script, press, press_named
from retro_harness.controls import SNES_A, SNES_START


def test_idle_is_zeros() -> None:
    action = idle()
    assert action.shape == (ACTION_SIZE,)
    assert int(action.sum()) == 0


def test_press_sets_buttons() -> None:
    action = press(SNES_A, SNES_START)
    assert action[SNES_A] == 1
    assert action[SNES_START] == 1
    assert int(action.sum()) == 2


def test_named_script_expands_frames() -> None:
    frames = named_script([("A", 2), ("IDLE", 3), ("START", 1)])
    assert len(frames) == 6
    assert frames[0][SNES_A] == 1
    assert int(frames[2].sum()) == 0
    assert frames[5][SNES_START] == 1


def test_press_named_unknown_is_idle() -> None:
    action = press_named("NOT_A_BUTTON")
    assert np.array_equal(action, idle())

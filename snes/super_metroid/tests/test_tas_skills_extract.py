"""Offline TAS skill-window detector (no emulator)."""

from __future__ import annotations

import pytest

from retro_harness.actions import SNES_ACTION_SIZE
from retro_harness.controls import SNES_BUTTON_NAME_TO_INDEX
from super_metroid.tas.skills_extract import detect_skill_windows, detect_slice_skills


def _frame(*names: str) -> list[int]:
    fr = [0] * SNES_ACTION_SIZE
    for name in names:
        idx = SNES_BUTTON_NAME_TO_INDEX[name]
        if idx is not None:
            fr[idx] = 1
    return fr


def test_arm_pump_window_from_period2_l_taps() -> None:
    """LEFT run + L on/off every frame is shoulder pumping (GameResources)."""
    frames = [_frame()] * 4
    for i in range(10):
        names = ["B", "LEFT"]
        if i % 2 == 0:
            names.append("L")
        frames.append(_frame(*names))
    frames.extend([_frame()] * 4)

    windows = detect_skill_windows(frames, movie_id="synth")
    pumps = [w for w in windows if w.skill == "arm_pump"]
    assert len(pumps) == 1
    pump = pumps[0]
    assert pump.movie_id == "synth"
    assert pump.start == 4
    assert pump.end == 14


def test_hero_bubbleroom_arm_pump_matches_rle_l_taps() -> None:
    """hero_bubbleroom_full L-taps at f3,5,…,21 (period-2) while holding LEFT."""
    try:
        windows = detect_slice_skills("hero_bubbleroom_full")
    except FileNotFoundError:
        pytest.skip("missing hero_bubbleroom_full slice")
    pumps = [w for w in windows if w.skill == "arm_pump"]
    assert len(pumps) == 1
    pump = pumps[0]
    assert pump.movie_id == "hero_bubbleroom_full"
    assert pump.start == 3
    assert pump.end == 23


def test_mockball_window_from_jump_then_down() -> None:
    """Run, jump, then DOWN within a few frames is mockball (GameResources)."""
    frames = [_frame()] * 3
    frames.extend(_frame("B", "RIGHT") for _ in range(5))
    frames.extend(_frame("B", "RIGHT", "A") for _ in range(4))
    frames.extend(_frame("B", "DOWN") for _ in range(8))
    frames.extend([_frame()] * 3)

    windows = detect_skill_windows(frames, movie_id="synth")
    balls = [w for w in windows if w.skill == "mockball"]
    assert len(balls) == 1
    ball = balls[0]
    assert ball.movie_id == "synth"
    assert ball.start == 8
    assert ball.end == 20


def test_hero_kraid_entry_mockball_matches_jump_then_down() -> None:
    """hero_kraid_entry_full jumps at f124, morphs DOWN at f129."""
    try:
        windows = detect_slice_skills("hero_kraid_entry_full")
    except FileNotFoundError:
        pytest.skip("missing hero_kraid_entry_full slice")
    balls = [w for w in windows if w.skill == "mockball"]
    assert [(w.start, w.end) for w in balls] == [(124, 133)]
    assert balls[0].movie_id == "hero_kraid_entry_full"
    assert not any(w.skill == "arm_pump" for w in windows)


def test_hero_bubbleroom_mockball_after_arm_pump() -> None:
    """hero_bubbleroom_full jumps at f28 (LEFT+A), then DOWN at f34 for 26f."""
    try:
        windows = detect_slice_skills("hero_bubbleroom_full")
    except FileNotFoundError:
        pytest.skip("missing hero_bubbleroom_full slice")
    balls = [w for w in windows if w.skill == "mockball"]
    assert [(w.start, w.end) for w in balls] == [(28, 60)]


def test_held_angle_and_crouch_are_not_skills() -> None:
    """A 12-frame L hold is angle, not pump; DOWN without a jump is not mockball."""
    angle = [_frame("B", "L") for _ in range(12)]
    crouch = [_frame("B", "RIGHT") for _ in range(6)]
    crouch.extend(_frame("B", "DOWN") for _ in range(8))
    assert detect_skill_windows(angle) == ()
    assert detect_skill_windows(crouch) == ()

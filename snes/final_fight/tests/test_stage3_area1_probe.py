"""Unit tests for Stage3 Area1 face-Y helpers (no emulator)."""

from __future__ import annotations

from final_fight.scripts.stage3_area1_probe import _face_y_action
from retro_harness.actions import idle_action


def test_face_y_faces_first_then_pulses_y() -> None:
    enemy = {"dx": -52, "dy": 8, "hp": 250, "st": 3}
    act, reason, faced = _face_y_action(1, enemy, faced=False)
    assert reason == "face"
    assert faced is True

    act, reason, faced = _face_y_action(24, enemy, faced=True)
    # f%12 < 2 → Y
    assert reason == "y"

    act, reason, faced = _face_y_action(26, enemy, faced=True)
    assert reason == "gap"
    assert act == idle_action()


def test_face_y_spaces_when_too_close() -> None:
    enemy = {"dx": -10, "dy": 0, "hp": 200, "st": 3}
    _act, reason, _faced = _face_y_action(50, enemy, faced=True)
    assert reason == "space"


def test_face_y_steps_when_far_behind() -> None:
    enemy = {"dx": -120, "dy": 0, "hp": 200, "st": 3}
    _act, reason, _faced = _face_y_action(50, enemy, faced=True)
    assert reason == "step"

"""Unit tests for Stage3 Area1 Andore helpers (no emulator)."""

from __future__ import annotations

from final_fight.edge_combat import area1_andore_action
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
    # Pulse gap used to idle; closing LEFT into grab range is also ok.
    assert reason in {"gap", "close"}


def test_overlap_throws_instead_of_walking_fence() -> None:
    enemy = {"dx": -10, "dy": 0, "hp": 200, "st": 3}
    _act, reason, _faced = _face_y_action(50, enemy, faced=True, sx=110)
    assert reason in {"throw", "close", "gap"}
    assert reason != "space"


def test_far_behind_waits_instead_of_gutter_chase() -> None:
    enemy = {"dx": -120, "dy": 0, "hp": 200, "st": 3}
    _act, reason, _faced = _face_y_action(50, enemy, faced=True, sx=110)
    assert reason in {"wait_far", "wait_desync"}
    assert reason != "step"


def test_area1_clamps_fence_and_gutter() -> None:
    tick, _faced = area1_andore_action(
        frame=10, sx=180, dx=-40, dy=8, faced=True
    )
    assert tick.reason == "clamp_l"
    tick, _faced = area1_andore_action(
        frame=10, sx=40, dx=-40, dy=8, faced=True
    )
    assert tick.reason == "clamp_r"


def test_area1_throw_cycle_on_overlap() -> None:
    tick, _faced = area1_andore_action(
        frame=4, sx=110, dx=-8, dy=0, faced=True
    )
    assert tick.reason == "throw"

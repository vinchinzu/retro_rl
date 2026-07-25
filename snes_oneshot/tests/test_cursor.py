"""Tests for shared cursor helpers."""

from __future__ import annotations

from snes_oneshot.actions import buttons_multi, idle_action_multi
from snes_oneshot.cursor import (
    CursorPose,
    CursorTarget,
    at_target,
    plan_cursor_path,
    step_toward_target,
)


def test_step_toward_prefers_larger_axis() -> None:
    pose = CursorPose(0, 0)
    target = CursorTarget(10, 3)
    action = step_toward_target(pose, target)
    assert action.reason == "cursor_right"


def test_step_toward_confirm_in_deadzone() -> None:
    pose = CursorPose(50, 50)
    target = CursorTarget(51, 49, deadzone=2)
    assert at_target(pose, target)
    action = step_toward_target(pose, target)
    assert action.reason == "confirm_at_target"
    assert action.action[8] == 1  # A


def test_step_toward_holds_fast_button() -> None:
    pose = CursorPose(0, 0)
    target = CursorTarget(10, 0)
    action = step_toward_target(pose, target, fast_button="Y")
    assert action.reason == "cursor_right"
    assert action.action[7] == 1  # RIGHT
    assert action.action[1] == 1  # Y


def test_plan_cursor_path_reaches_target() -> None:
    frames = plan_cursor_path(CursorPose(0, 0), CursorTarget(4, 2, deadzone=0))
    assert len(frames) == 6
    assert frames[0].reason == "cursor_right"


def test_buttons_multi_p2_a() -> None:
    idle = idle_action_multi(players=2)
    assert len(idle) == 24
    assert sum(idle) == 0
    action = buttons_multi(p2=("A",))
    assert action[12 + 8] == 1  # P2 A
    assert sum(action[:12]) == 0

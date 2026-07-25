"""Reusable cursor / magnifier helpers for point-and-click SNES games."""

from __future__ import annotations

from dataclasses import dataclass

from snes_oneshot.actions import buttons, idle_action
from snes_oneshot.primitives import FrameAction


@dataclass(frozen=True)
class CursorTarget:
    """Pixel / RAM-space target for a search cursor."""

    x: int
    y: int
    deadzone: int = 2
    label: str = ""


@dataclass(frozen=True)
class CursorPose:
    """Current cursor position."""

    x: int
    y: int


def at_target(pose: CursorPose, target: CursorTarget) -> bool:
    """True when cursor is within the target deadzone on both axes."""
    return (
        abs(pose.x - target.x) <= target.deadzone
        and abs(pose.y - target.y) <= target.deadzone
    )


def step_toward_target(
    pose: CursorPose,
    target: CursorTarget,
    *,
    confirm_button: str = "A",
    fast_button: str | None = None,
) -> FrameAction:
    """One frame of d-pad toward target, or confirm when close enough.

    Prefers the axis with the larger absolute error so approach is
    axis-aligned. Optional ``fast_button`` (e.g. \"Y\") is held while moving.
    """
    if at_target(pose, target):
        return FrameAction(
            action=buttons(confirm_button),
            reason="confirm_at_target",
        )

    dx = target.x - pose.x
    dy = target.y - pose.y
    hold: list[str] = []
    if abs(dx) >= abs(dy) and abs(dx) > target.deadzone:
        hold.append("RIGHT" if dx > 0 else "LEFT")
        reason = f"cursor_{hold[0].lower()}"
    elif abs(dy) > target.deadzone:
        hold.append("DOWN" if dy > 0 else "UP")
        reason = f"cursor_{hold[0].lower()}"
    else:
        return FrameAction(action=idle_action(), reason="cursor_idle")

    if fast_button is not None:
        hold.append(fast_button)
    return FrameAction(action=buttons(*hold), reason=reason)


def plan_cursor_path(
    start: CursorPose,
    target: CursorTarget,
    *,
    max_steps: int = 512,
    step_x: int = 1,
    step_y: int = 1,
) -> list[FrameAction]:
    """Pure simulation of axis-aligned steps until at target or max_steps.

    Does not confirm; stops once inside the deadzone.
    """
    pose = start
    frames: list[FrameAction] = []
    for _ in range(max_steps):
        if at_target(pose, target):
            break
        action = step_toward_target(pose, target)
        if action.reason == "confirm_at_target":
            break
        frames.append(
            FrameAction(action=list(action.action), reason=action.reason)
        )
        if action.reason == "cursor_right":
            pose = CursorPose(pose.x + step_x, pose.y)
        elif action.reason == "cursor_left":
            pose = CursorPose(pose.x - step_x, pose.y)
        elif action.reason == "cursor_down":
            pose = CursorPose(pose.x, pose.y + step_y)
        elif action.reason == "cursor_up":
            pose = CursorPose(pose.x, pose.y - step_y)
        else:
            break
    return frames

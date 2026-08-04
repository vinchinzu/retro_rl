"""Blue/gray door stage, beam-open, and period exit-push helpers.

Kraid return (and similar reverse hops) share a frozen open-loop pattern:

1. Lip backoff + face + release (do not fire while walking into the shell)
2. Standing beam shot/fuse cycles
3. Period jump/spin/reshot/run until target room (optional transition drain)

Callers own frame budgets and reason labels. This module does not door-warp
or write progression RAM. Super-door pressure stays in :mod:`door`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import hold

if TYPE_CHECKING:
    from super_metroid.routes.runtime import ControllerSession

GuardFn = Callable[[SuperMetroidState], None]
WrongRoomFn = Callable[[SuperMetroidState], None]

# Default jump-enter cadence used by Eye→Baby / Baby→Kihunter / Kraid→Eye.
# Windows on frame % 30: jump / jump-spin / reshot / run.
JUMP_ENTER_PERIOD = 30
JUMP_ENTER_JUMP_END = 4
JUMP_ENTER_SPIN_END = 10
JUMP_ENTER_RESHOT_END = 14


def lip_stage(
    session: ControllerSession,
    *,
    label: str,
    backoff: str = "RIGHT",
    face: str = "LEFT",
    backoff_frames: int = 8,
    face_frames: int = 8,
    release_frames: int = 6,
    settle_frames: int = 0,
) -> None:
    """Backoff off the door lip, face it, then release before beams."""
    if settle_frames > 0:
        hold(session, settle_frames, reason=f"{label}_approach_settle")
    hold(session, backoff_frames, backoff, reason=f"{label}_lip_backoff")
    hold(session, face_frames, face, reason=f"{label}_face")
    hold(session, release_frames, reason=f"{label}_face_release")


def beam_open_door(
    session: ControllerSession,
    *,
    label: str,
    shots: int = 6,
    shot_frames: int = 4,
    fuse_frames: int = 14,
    shot_buttons: Sequence[str] = ("X",),
) -> None:
    """Standing beam/Super shot + fuse cycles (no walk into the shell)."""
    for _ in range(shots):
        hold(session, shot_frames, *shot_buttons, reason=f"{label}_door_shot")
        hold(session, fuse_frames, reason=f"{label}_door_fuse")


def drain_door_transition(
    session: ControllerSession,
    target_room: int,
    *,
    max_frames: int = 80,
    reason: str = "transition",
) -> SuperMetroidState:
    """Idle through a door transition until ordinary in ``target_room``."""
    state = session.state
    for _ in range(max_frames):
        state = hold(session, 1, reason=reason)
        if state.room_id == target_room and state.door_transition == 0:
            break
    return state


def period_exit_push(
    session: ControllerSession,
    target_room: int,
    *,
    label: str,
    max_frames: int,
    period: int,
    windows: Sequence[tuple[int, Sequence[str], str]],
    transition_drain: int = 0,
    transition_reason: str | None = None,
    guard: GuardFn | None = None,
    on_wrong_room: WrongRoomFn | None = None,
    on_state: Callable[[SuperMetroidState], None] | None = None,
) -> SuperMetroidState:
    """Hold period-cadence inputs until ``target_room`` or timeout.

    ``windows`` is ordered ``(end_exclusive, buttons, reason_suffix)`` on
    ``frame % period``. The last window's end should be ``<= period``; any
    remainder uses the last window (callers usually cover the full period).

    Optional ``transition_drain`` idles after ``door_transition`` fires so the
    hop can settle in the target room without walking mid-transition.
    """
    if not windows:
        raise ValueError(f"{label}: period_exit_push requires at least one window")
    drain_reason = transition_reason or f"{label}_transition"
    for index in range(max_frames):
        phase = index % period
        buttons: Sequence[str] = windows[-1][1]
        reason_suffix = windows[-1][2]
        for end, btns, suffix in windows:
            if phase < end:
                buttons = btns
                reason_suffix = suffix
                break
        state = hold(session, 1, *buttons, reason=f"{label}_{reason_suffix}")
        if guard is not None:
            guard(state)
        if on_state is not None:
            on_state(state)
        if state.room_id == target_room:
            return state
        if on_wrong_room is not None and state.room_id != target_room:
            # Caller decides whether the current room is fatal (may raise).
            on_wrong_room(state)
        if transition_drain > 0 and state.door_transition:
            state = drain_door_transition(
                session,
                target_room,
                max_frames=transition_drain,
                reason=drain_reason,
            )
            if state.room_id == target_room:
                return state
    raise TimeoutError(f"{label}: exit timed out: {session.state}")


def jump_enter_exit(
    session: ControllerSession,
    target_room: int,
    *,
    label: str,
    direction: str = "LEFT",
    max_frames: int = 700,
    period: int = JUMP_ENTER_PERIOD,
    jump_end: int = JUMP_ENTER_JUMP_END,
    spin_end: int = JUMP_ENTER_SPIN_END,
    reshot_end: int = JUMP_ENTER_RESHOT_END,
    transition_drain: int = 80,
    guard: GuardFn | None = None,
    on_wrong_room: WrongRoomFn | None = None,
) -> SuperMetroidState:
    """Jump-enter a horizontal blue/gray door (elevated trigger band).

    Default cadence matches Eye→Baby / Baby→Kihunter / Kraid→Eye reverse:
    jump → jump-spin → reshot → run on a 30-frame period.
    """
    return period_exit_push(
        session,
        target_room,
        label=label,
        max_frames=max_frames,
        period=period,
        windows=(
            (jump_end, (direction, "A"), "jump"),
            (spin_end, (direction, "A", "B"), "jump_spin"),
            (reshot_end, ("X",), "reshot"),
            (period, (direction, "B"), "exit"),
        ),
        transition_drain=transition_drain,
        transition_reason=f"{label}_transition",
        guard=guard,
        on_wrong_room=on_wrong_room,
    )


__all__ = [
    "JUMP_ENTER_PERIOD",
    "JUMP_ENTER_JUMP_END",
    "JUMP_ENTER_SPIN_END",
    "JUMP_ENTER_RESHOT_END",
    "lip_stage",
    "beam_open_door",
    "drain_door_transition",
    "period_exit_push",
    "jump_enter_exit",
]

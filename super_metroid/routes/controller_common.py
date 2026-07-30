"""Shared Samus controller primitives for Super Metroid route modules.

Used by KPDR and post-Spore controllers. Keep game-agnostic movement helpers
here; route-specific choreography stays in segment modules.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.runtime import ControllerSession, hold

# Re-export hold under the historical private alias used by controllers.
_hold = hold

# Morph-ball poses observed on Super routes (and facing/air/fall variants).
# 29=0x1D, 30=0x1E falling; 31=0x1F, 32=0x20; 49=0x31, 50=0x32;
# 65=0x41, 66=0x42 ground/move. Expand from live logs only.
MORPH_POSES = frozenset({29, 30, 31, 32, 49, 50, 65, 66})


@dataclass(frozen=True)
class MorphPolicy:
    """Attempt-scaled frame budget for pose-confirmed double-tap morph."""

    max_attempts: int = 5
    base_up: int = 4
    up_step: int = 2
    max_up_extra_steps: int = 3
    base_idle: int = 3
    base_tap1: int = 5
    tap1_step: int = 2
    base_tap2: int = 6
    tap2_step: int = 3
    release: int = 3
    base_poll: int = 28
    poll_step: int = 4

    def up_frames(self, attempt: int) -> int:
        return self.base_up + min(attempt, self.max_up_extra_steps) * self.up_step

    def idle_frames(self, attempt: int) -> int:
        return self.base_idle + attempt

    def tap1_frames(self, attempt: int) -> int:
        return self.base_tap1 + attempt * self.tap1_step

    def tap2_frames(self, attempt: int) -> int:
        return self.base_tap2 + attempt * self.tap2_step

    def poll_timeout(self, attempt: int) -> int:
        return self.base_poll + attempt * self.poll_step


DEFAULT_MORPH_POLICY = MorphPolicy()


def require_room(session: ControllerSession, room_id: int, label: str) -> None:
    """Raise if the session is not in ``room_id``."""
    state = session.state
    if state.room_id != room_id:
        raise RuntimeError(
            f"{label}: expected room 0x{room_id:04X}, got 0x{state.room_id:04X} "
            f"at frame {session.frame}"
        )


# Historical private alias (call sites and re-exports).
_require_room = require_room


def select_weapon(
    session: ControllerSession, target: int, *, max_cycles: int = 8
) -> None:
    """Cycle SELECT until ``selected_item`` matches ``target`` (0–3)."""
    for _ in range(max_cycles):
        if session.state.selected_item == target:
            return
        hold(session, 1, "SELECT", reason="select_weapon")
        hold(session, 25, reason="select_weapon_settle")
    if session.state.selected_item != target:
        raise RuntimeError(
            f"could not select weapon {target}, still {session.state.selected_item}"
        )


_select_weapon = select_weapon


def is_morph(pose: int) -> bool:
    """True when Samus is in morph/spring-ball pose (any facing)."""
    return pose in MORPH_POSES


def unmorph(session: ControllerSession) -> None:
    """Leave morph / crouch poses that block ordinary standing inputs."""
    pose = session.state.pose
    if pose in (39, 40, 137, 138, 9, 10) or is_morph(pose):
        hold(session, 8, "UP", reason="unmorph")
        if not is_morph(session.state.pose):
            hold(session, 8, "A", reason="unmorph")
        hold(session, 10, reason="unmorph_settle")


_unmorph = unmorph


def wait_until(
    session: ControllerSession,
    pred: Callable[[SuperMetroidState], bool],
    *,
    timeout: int = 120,
    reason: str = "wait",
) -> SuperMetroidState:
    """Idle one frame at a time until ``pred(state)`` or raise ``TimeoutError``."""
    for _ in range(timeout):
        if pred(session.state):
            return session.state
        hold(session, 1, reason=reason)
    raise TimeoutError(f"{reason} timed out: {session.state}")


def wait_ordinary_room(
    session: ControllerSession,
    room_id: int,
    *,
    settle_frames: int = 200,
    label: str,
) -> SuperMetroidState:
    """Hold until ordinary gameplay settles in ``room_id``."""
    for frame in range(settle_frames):
        state = hold(session, 1, reason=f"{label}_settle")
        if (
            state.room_id == room_id
            and state.game_state == 8
            and state.door_transition == 0
            and frame > 15
        ):
            return state
    state = session.state
    if state.room_id != room_id:
        raise RuntimeError(
            f"{label}: expected 0x{room_id:04X}, got 0x{state.room_id:04X} @ {state}"
        )
    return state


_wait_ordinary_room = wait_ordinary_room


def ensure_morph(
    session: ControllerSession,
    *,
    max_attempts: int | None = None,
    policy: MorphPolicy | None = None,
) -> SuperMetroidState:
    """Pose-confirmed morph via double-tap DOWN (held DOWN only crouches).

    Each attempt: brief UP to leave crouch, idle, tap–release–tap DOWN, then
    poll for a morph pose. Later attempts hold DOWN longer and re-UP before
    retry. Timing scales via :class:`MorphPolicy`.
    """
    pol = policy or DEFAULT_MORPH_POLICY
    attempts = max_attempts if max_attempts is not None else pol.max_attempts
    for attempt in range(attempts):
        if is_morph(session.state.pose):
            return session.state
        hold(session, pol.up_frames(attempt), "UP", reason="morph_pre")
        hold(session, pol.idle_frames(attempt), reason="morph_idle")
        hold(session, pol.tap1_frames(attempt), "DOWN", reason="morph_tap1")
        hold(session, pol.release, reason="morph_release")
        hold(session, pol.tap2_frames(attempt), "DOWN", reason="morph_tap2")
        try:
            state = wait_until(
                session,
                lambda s: is_morph(s.pose),
                timeout=pol.poll_timeout(attempt),
                reason="morph_poll",
            )
        except TimeoutError:
            continue
        if is_morph(state.pose):
            return state
    raise TimeoutError(f"ensure_morph failed, pose={session.state.pose}")


def play_run_shoot_exit(
    session: ControllerSession,
    *,
    from_room: int,
    to_room: int,
    direction: str,
    label: str,
    run_frames: int = 40,
    shoot_frames: int = 6,
    spin_frames: int = 40,
    hold_frames: int = 160,
    settle_frames: int = 200,
    super_door: bool = False,
) -> SuperMetroidState:
    """Generic horizontal door exit: run + shoot (+ optional Super) + spin through.

    ``direction`` is ``LEFT`` or ``RIGHT``.
    """
    require_room(session, from_room, label)
    if super_door:
        try:
            select_weapon(session, 2)  # supers
        except RuntimeError:
            pass
    else:
        try:
            select_weapon(session, 0)
        except RuntimeError:
            pass
    hold(session, run_frames, direction, "B", reason=f"{label}_run")
    shoot_btns = (direction, "B", "X") if not super_door else (direction, "X")
    hold(session, shoot_frames, *shoot_btns, reason=f"{label}_shoot")
    if super_door:
        hold(session, 20, reason=f"{label}_super_fuse")
        hold(session, 8, direction, "X", reason=f"{label}_super2")
        hold(session, 20, reason=f"{label}_super_fuse2")
    hold(session, spin_frames, direction, "B", "A", reason=f"{label}_spin")
    entered = False
    for _ in range(hold_frames):
        state = hold(session, 1, direction, reason=f"{label}_hold")
        if state.room_id == to_room:
            entered = True
            break
    if not entered:
        raise TimeoutError(f"{label}: did not reach 0x{to_room:04X}: {session.state}")
    return wait_ordinary_room(
        session, to_room, settle_frames=settle_frames, label=label
    )


__all__ = [
    "DEFAULT_MORPH_POLICY",
    "MORPH_POSES",
    "MorphPolicy",
    "_hold",
    "_require_room",
    "_select_weapon",
    "_unmorph",
    "_wait_ordinary_room",
    "ensure_morph",
    "hold",
    "is_morph",
    "play_run_shoot_exit",
    "require_room",
    "select_weapon",
    "unmorph",
    "wait_ordinary_room",
    "wait_until",
]

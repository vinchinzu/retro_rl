"""Shared Samus controller primitives for Super Metroid route modules.

Used by KPDR controllers (including Super collect / Big Pink). Keep
composable movement / exit helpers here; route-specific choreography stays in
segment modules.

Hybrid policy surface (raise abstraction without dropping raw timing):
- :func:`wait_until` / :func:`wait_requirement` — RAM gates
- :func:`require_state` — fail with StateRequirement failure strings
- :func:`hold_until` — hold buttons while polling a predicate
- :func:`play_run_shoot_exit` / :func:`traverse_door` — horizontal door exits
- :func:`collect_item_mask` — wait for PLM item bit
- :func:`wait_ordinary_room` — multi-truth settle (room + phase + door)
- :func:`walljump_once` / :func:`consecutive_walljumps` — room-agnostic WJ
  pulses (Bubble Phase D, Spazer, …)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from super_metroid.policy import StateRequirement
from super_metroid.ram import SuperMetroidState
from super_metroid.routes.runtime import ControllerSession, hold

# Private aliases kept for segment modules that historically imported them.
_hold = hold

# Morph-ball poses observed on Super routes (and facing/air/fall variants).
# 29=0x1D, 30=0x1E falling; 31=0x1F, 32=0x20; 49=0x31, 50=0x32;
# 65=0x41, 66=0x42 ground/move. Expand from live logs only.
MORPH_POSES = frozenset({29, 30, 31, 32, 49, 50, 65, 66})

# Wall-latch pose (ready to wall-jump). Shared across Bubble, Alcatraz, Spazer.
POSE_WALL_LATCH = 132


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


def settle_hold(
    session: ControllerSession, frames: int = 12, *, reason: str = "settle"
) -> SuperMetroidState:
    """Hold idle for a fixed settle window and preserve its reason label."""
    return hold(session, frames, reason=reason)


def short_hop(
    session: ControllerSession,
    direction: str,
    frames: int,
    *,
    buttons_extra: Sequence[str] = (),
    reason: str = "short_hop",
) -> SuperMetroidState:
    """Hold a directional jump with optional additional buttons."""
    return hold(session, frames, direction, *buttons_extra, reason=reason)


def vertical_hop(
    session: ControllerSession,
    frames: int,
    *,
    reason: str = "vertical_hop",
) -> SuperMetroidState:
    """Hold jump in place for a fixed vertical hop window."""
    return hold(session, frames, "A", reason=reason)


# ---------------------------------------------------------------------------
# Wall-jump skills (room-agnostic; Bubble / Parlor / Spazer consumers)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WallJumpTiming:
    """One wall-jump pulse: optional delay into, into+A, amid A, flip+A.

    ``delay_into_frames``: face into the wall **without** A before the A press
    (delayed WJ window experiment). Product Bubble R15 uses 0.
    """

    into: str = "LEFT"
    flip: str = "RIGHT"
    into_frames: int = 20
    amid_frames: int = 4
    flip_frames: int = 8
    delay_into_frames: int = 0


def is_wall_latch(state: SuperMetroidState) -> bool:
    """True when Samus is in wall-latch pose (ready to wall-jump)."""
    return int(state.pose) == POSE_WALL_LATCH


def walljump_once(
    session: ControllerSession,
    timing: WallJumpTiming,
    *,
    reason: str = "wj",
    stop_when: Callable[[SuperMetroidState], bool] | None = None,
) -> SuperMetroidState:
    """Execute **one** wall-jump pulse (delay into → into+A → amid A → flip+A).

    Room-agnostic skill. Callers may pass ``stop_when`` for early exit (top
    band, room leave, track update). Timings are parameters — Bubble R15 and
    Parlor Alcatraz each pass their own.
    """
    tag = reason
    phases: list[tuple[int, tuple[str, ...], str]] = [
        (timing.delay_into_frames, (timing.into,), f"{tag}_delay"),
        (timing.into_frames, (timing.into, "A"), f"{tag}_into"),
        (timing.amid_frames, ("A",), f"{tag}_amid"),
        (timing.flip_frames, (timing.flip, "A"), f"{tag}_flip"),
    ]
    state = session.state
    for n, buttons, phase_reason in phases:
        for _ in range(n):
            state = hold(session, 1, *buttons, reason=phase_reason)
            if stop_when is not None and stop_when(state):
                return state
    return state


def consecutive_walljumps(
    session: ControllerSession,
    jumps: Sequence[WallJumpTiming],
    *,
    reason: str = "wj_chain",
    gap_frames: int = 0,
    stop_when: Callable[[SuperMetroidState], bool] | None = None,
) -> SuperMetroidState:
    """Chain N consecutive wall-jump pulses with optional idle gap between.

    Visible multi-room skill: Bubble Phase D (gap=0, R15 pair), post-Torizo
    Parlor Alcatraz left climb (gap settles between open-loop pulses).
    """
    state = session.state
    for i, timing in enumerate(jumps):
        if stop_when is not None and stop_when(state):
            return state
        state = walljump_once(
            session,
            timing,
            reason=f"{reason}_wj{i + 1}",
            stop_when=stop_when,
        )
        if stop_when is not None and stop_when(state):
            return state
        if gap_frames > 0 and i + 1 < len(jumps):
            for _ in range(gap_frames):
                state = hold(session, 1, reason=f"{reason}_gap")
                if stop_when is not None and stop_when(state):
                    return state
    return state


def wait_requirement(
    session: ControllerSession,
    requirement: StateRequirement,
    *,
    timeout: int = 120,
    reason: str = "wait_requirement",
) -> SuperMetroidState:
    """Idle until :class:`StateRequirement` matches live RAM (or timeout)."""
    try:
        return wait_until(
            session,
            requirement.matches,
            timeout=timeout,
            reason=reason,
        )
    except TimeoutError as exc:
        failures = requirement.failures(session.state)
        detail = "; ".join(failures) if failures else "predicate never true"
        raise TimeoutError(
            f"{reason} timed out ({detail}): frame={session.frame} "
            f"room=0x{session.state.room_id:04X} "
            f"xy=({session.state.samus_x},{session.state.samus_y})"
        ) from exc


def require_state(
    session: ControllerSession,
    requirement: StateRequirement,
    label: str,
) -> SuperMetroidState:
    """Raise immediately if live RAM fails ``requirement`` (no waiting)."""
    failures = requirement.failures(session.state)
    if failures:
        raise RuntimeError(
            f"{label}: {'; '.join(failures)}; "
            f"frame={session.frame} room=0x{session.state.room_id:04X} "
            f"xy=({session.state.samus_x},{session.state.samus_y}) "
            f"items=0x{session.state.collected_items:04X}"
        )
    return session.state


def hold_until(
    session: ControllerSession,
    pred: Callable[[SuperMetroidState], bool],
    *buttons: str,
    timeout: int = 120,
    reason: str = "hold_until",
) -> SuperMetroidState:
    """Hold ``buttons`` each frame until ``pred`` matches or timeout."""
    for _ in range(timeout):
        if pred(session.state):
            return session.state
        hold(session, 1, *buttons, reason=reason)
    raise TimeoutError(
        f"{reason} timed out: frame={session.frame} "
        f"room=0x{session.state.room_id:04X} xy=({session.state.samus_x},{session.state.samus_y})"
    )


def wait_ordinary_room(
    session: ControllerSession,
    room_id: int,
    *,
    settle_frames: int = 200,
    label: str,
    x_range: tuple[int, int] | None = None,
    y_range: tuple[int, int] | None = None,
    min_settle_frame: int = 15,
) -> SuperMetroidState:
    """Hold until multi-truth settle: room + ordinary phase + optional window."""
    for frame in range(settle_frames):
        state = hold(session, 1, reason=f"{label}_settle")
        if (
            state.room_id == room_id
            and state.game_state == 8
            and state.door_transition == 0
            and frame > min_settle_frame
        ):
            if x_range is not None and not (x_range[0] <= state.samus_x <= x_range[1]):
                continue
            if y_range is not None and not (y_range[0] <= state.samus_y <= y_range[1]):
                continue
            return state
    state = session.state
    if state.room_id != room_id:
        raise RuntimeError(
            f"{label}: expected 0x{room_id:04X}, got 0x{state.room_id:04X} @ {state}"
        )
    if x_range is not None or y_range is not None:
        raise TimeoutError(
            f"{label}: settled in room but position window missed "
            f"xy=({state.samus_x},{state.samus_y}) x_range={x_range} y_range={y_range}"
        )
    return state


_wait_ordinary_room = wait_ordinary_room


def collect_item_mask(
    session: ControllerSession,
    item_mask: int,
    *,
    timeout: int = 600,
    reason: str = "collect_item",
    buttons: Sequence[str] = (),
) -> SuperMetroidState:
    """Hold optional buttons until ``collected_items`` gains ``item_mask``."""
    if session.state.collected_items & item_mask == item_mask:
        return session.state
    target = session.state.collected_items | item_mask

    def _has_items(state: SuperMetroidState) -> bool:
        return state.collected_items & item_mask == item_mask

    if buttons:
        return hold_until(
            session, _has_items, *buttons, timeout=timeout, reason=reason
        )
    try:
        return wait_until(session, _has_items, timeout=timeout, reason=reason)
    except TimeoutError as exc:
        raise TimeoutError(
            f"{reason}: items still 0x{session.state.collected_items:04X}, "
            f"want mask 0x{item_mask:04X} (target 0x{target:04X})"
        ) from exc


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


def traverse_door(
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
    entry_x_range: tuple[int, int] | None = None,
    entry_y_range: tuple[int, int] | None = None,
) -> SuperMetroidState:
    """Door exit primitive; optional local position window after settle.

    Same core as :func:`play_run_shoot_exit`, with multi-truth entry window for
    continuous variance (door-spawn x/y bands).
    """
    play_run_shoot_exit(
        session,
        from_room=from_room,
        to_room=to_room,
        direction=direction,
        label=label,
        run_frames=run_frames,
        shoot_frames=shoot_frames,
        spin_frames=spin_frames,
        hold_frames=hold_frames,
        settle_frames=settle_frames,
        super_door=super_door,
    )
    if entry_x_range is None and entry_y_range is None:
        return session.state
    # Re-check window; if already good, return; else short poll without moving.
    req = StateRequirement(
        room_id=to_room,
        game_states=frozenset({8}),
        x_range=entry_x_range,
        y_range=entry_y_range,
    )
    if req.matches(session.state):
        return session.state
    return wait_requirement(
        session,
        req,
        timeout=min(90, settle_frames),
        reason=f"{label}_entry_window",
    )


__all__ = [
    "DEFAULT_MORPH_POLICY",
    "MORPH_POSES",
    "MorphPolicy",
    "POSE_WALL_LATCH",
    "WallJumpTiming",
    "_hold",
    "_require_room",
    "_select_weapon",
    "_unmorph",
    "_wait_ordinary_room",
    "collect_item_mask",
    "consecutive_walljumps",
    "ensure_morph",
    "hold",
    "hold_until",
    "is_morph",
    "is_wall_latch",
    "play_run_shoot_exit",
    "require_room",
    "require_state",
    "settle_hold",
    "select_weapon",
    "short_hop",
    "traverse_door",
    "unmorph",
    "vertical_hop",
    "wait_ordinary_room",
    "wait_requirement",
    "wait_until",
    "walljump_once",
]

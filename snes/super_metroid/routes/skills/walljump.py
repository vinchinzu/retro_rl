"""Room-agnostic wall-jump climb skills (policy-driven timings/bands).

Physics (pre-Speed, Hi-Jump equipped) — fold into skills, not superstition:

* Pure horizontal runway speed does **not** strongly raise initial vertical
  velocity of a normal jump or wall-jump.
* What runway speed *does* buy: better wall-contact positioning/timing,
  longer air time before the next jump, and spin-held horizontal momentum.
* Real “horizontal → vertical” transfer comes from: consecutive wall-jumps
  (chain before vy decays), delayed WJ window, crouch-jump, or enemy damage
  boost (experiment only).

Pulse primitives live in :mod:`controller_common`
(:func:`walljump_once`, :func:`consecutive_walljumps`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    POSE_WALL_LATCH,
    WallJumpTiming,
    hold,
    is_wall_latch,
    walljump_once as _walljump_once_common,
)
from super_metroid.routes.skills.geometry import (
    ClimbTrack,
    phase_d_top_band,
    track_state,
)
from super_metroid.routes.skills.knockback import is_knockback

if TYPE_CHECKING:
    from super_metroid.routes.runtime import ControllerSession


class WallJumpPolicy(Protocol):
    """Policy fields used by wall-jump climb skills."""

    ROOM_ID: int
    HEIGHT_CLASS_Y: int
    WJ_INTO_X: int
    WJ_LATCH_TIMEOUT: int
    WJ_APPROACH_X: tuple[int, int]
    WJ_APPROACH_Y: tuple[int, int]
    SAVE_APPROACH_BA: int
    SAVE_APPROACH_IDLE: int
    SAVE_APPROACH_TURN: int
    SAVE_WJ_FOLLOW: int
    RIGHT_SHELF_X: int
    MIDHIGH_Y: int
    RIGHT_WJ_PERIOD: int
    RIGHT_WJ_INTO: int
    RIGHT_WJ_BOUNCE: int
    WJ2_LEFT_X: int
    WJ2_LEFT_SEEK: int
    DMG_BOOST_HOLD_FRAMES: int
    R15_WJ1: WallJumpTiming
    R15_WJ2: WallJumpTiming
    R15_DOUBLE: tuple[WallJumpTiming, WallJumpTiming]


def _default_policy() -> WallJumpPolicy:
    from super_metroid.routes.skills.policies import bubble_to_bat as pol

    return pol  # type: ignore[return-value]


def wall_approach_band(
    state: SuperMetroidState,
    *,
    room_id: int | None = None,
    x_min: int | None = None,
    x_max: int | None = None,
    y_min: int | None = None,
    y_max: int | None = None,
    policy: WallJumpPolicy | None = None,
) -> bool:
    """Geometry band approaching a wall structure (not a latch)."""
    pol = policy if policy is not None else _default_policy()
    rid = pol.ROOM_ID if room_id is None else room_id
    x0 = pol.WJ_APPROACH_X[0] if x_min is None else x_min
    x1 = pol.WJ_APPROACH_X[1] if x_max is None else x_max
    y0 = pol.WJ_APPROACH_Y[0] if y_min is None else y_min
    y1 = pol.WJ_APPROACH_Y[1] if y_max is None else y_max
    return (
        int(state.room_id) == rid
        and x0 <= int(state.samus_x) <= x1
        and y0 <= int(state.samus_y) <= y1
    )


def _track_upd(
    session: ControllerSession,
    track: ClimbTrack,
    state: SuperMetroidState,
    policy: WallJumpPolicy,
    height_box: list[bool] | None,
) -> bool:
    """Update track; return True if caller should stop (top / left room)."""
    track_state(session, track, state, policy)  # type: ignore[arg-type]
    if state.room_id != policy.ROOM_ID:
        return True
    if height_box is not None and state.samus_y <= policy.HEIGHT_CLASS_Y:
        height_box[0] = True
    if phase_d_top_band(state, policy):  # type: ignore[arg-type]
        track.top_reached = True
        return True
    return False


def wait_wall_ready(
    session: ControllerSession,
    track: ClimbTrack,
    *,
    policy: WallJumpPolicy | None = None,
    max_frames: int | None = None,
    into_x: int | None = None,
    height_box: list[bool] | None = None,
) -> bool:
    """Extend **right spin only** until wall-ready (band / latch / Phase D).

    Never presses LEFT+A here — that **is** the wall-jump input and burns the
    first WJ window (turns double into single-class mx200~251 stalls).
    """
    pol = policy if policy is not None else _default_policy()
    label = track.label
    budget = pol.WJ_LATCH_TIMEOUT if max_frames is None else max_frames
    edge = pol.WJ_INTO_X if into_x is None else into_x
    for _ in range(budget):
        state = session.state
        if _track_upd(session, track, state, pol, height_box):
            return True
        if is_wall_latch(state):
            return True
        if int(state.samus_x) >= edge and wall_approach_band(
            state, policy=pol
        ):
            return True
        state = hold(
            session, 1, "RIGHT", "B", "A", reason=f"{label}_wj_ready"
        )
        if _track_upd(session, track, state, pol, height_box):
            return True
        if is_wall_latch(state):
            return True
    st = session.state
    return (
        is_wall_latch(st)
        or int(st.samus_x) >= edge
        or bool(track.top_reached)
    )


def wait_wall_latch(
    session: ControllerSession,
    track: ClimbTrack,
    *,
    policy: WallJumpPolicy | None = None,
    max_frames: int | None = None,
    into_x: int | None = None,
    height_box: list[bool] | None = None,
) -> bool:
    """Alias: wait for readiness without burning WJ."""
    return wait_wall_ready(
        session,
        track,
        policy=policy,
        max_frames=max_frames,
        into_x=into_x,
        height_box=height_box,
    )


def walljump_approach_coast(
    session: ControllerSession,
    track: ClimbTrack,
    *,
    policy: WallJumpPolicy | None = None,
    height_box: list[bool] | None = None,
) -> bool:
    """Pre-WJ coast: B+A → release → idle → face into wall (LEFT)."""
    pol = policy if policy is not None else _default_policy()
    label = track.label
    for _ in range(pol.SAVE_APPROACH_BA):
        state = hold(session, 1, "B", "A", reason=f"{label}_dwj_coast")
        if _track_upd(session, track, state, pol, height_box):
            return bool(track.top_reached)
    hold(session, 1, "B", reason=f"{label}_dwj_rel")
    for _ in range(pol.SAVE_APPROACH_IDLE):
        hold(session, 1, reason=f"{label}_dwj_idle")
    for _ in range(pol.SAVE_APPROACH_TURN):
        hold(session, 1, "LEFT", reason=f"{label}_dwj_turn")
    return bool(track.top_reached)


def walljump_once(
    session: ControllerSession,
    track: ClimbTrack,
    timing: WallJumpTiming | None = None,
    *,
    policy: WallJumpPolicy | None = None,
    height_box: list[bool] | None = None,
    reason: str = "wj",
) -> bool:
    """Execute **one** wall-jump pulse via shared :func:`walljump_once`.

    Track / height-box side effects ride on ``stop_when``. Caller must chain
    ≥2 for Phase D on the save-runway arc.
    """
    pol = policy if policy is not None else _default_policy()
    label = track.label
    t = timing if timing is not None else pol.R15_WJ1

    def _stop(state: SuperMetroidState) -> bool:
        return _track_upd(session, track, state, pol, height_box)

    _walljump_once_common(
        session,
        t,
        reason=f"{label}_{reason}",
        stop_when=_stop,
    )
    return bool(track.top_reached)


def consecutive_walljumps(
    session: ControllerSession,
    track: ClimbTrack,
    jumps: Sequence[WallJumpTiming] | None = None,
    *,
    policy: WallJumpPolicy | None = None,
    count: int | None = None,
    pre_approach: bool = True,
    extend_spin_ready: bool = False,
    height_box: list[bool] | None = None,
    follow_spin: bool = True,
    follow_frames: int | None = None,
) -> bool:
    """Chain N consecutive wall-jumps (default: policy **double** = 2).

    Single WJ is **insufficient** for Phase D (mx200 stalls ~251) unless a
    lucky enemy damage clip converts horizontal KB — never product.
    """
    pol = policy if policy is not None else _default_policy()
    label = track.label
    if jumps is None:
        pair = list(pol.R15_DOUBLE)
        n = 2 if count is None else max(2, count)  # never default to single
        while len(pair) < n:
            pair.append(pol.R15_WJ2)
        chain = pair[:n]
    else:
        chain = list(jumps)
        if count is not None:
            chain = chain[: max(1, count)]
        if len(chain) < 2:
            # Explicit single is allowed for experiments only — caller opts in.
            pass

    if pre_approach:
        if walljump_approach_coast(
            session, track, policy=pol, height_box=height_box
        ):
            return True
    if extend_spin_ready:
        wait_wall_ready(
            session, track, policy=pol, height_box=height_box
        )
        if track.top_reached or session.state.room_id != pol.ROOM_ID:
            return bool(track.top_reached)

    for i, timing in enumerate(chain):
        if track.top_reached or session.state.room_id != pol.ROOM_ID:
            break
        walljump_once(
            session,
            track,
            timing,
            policy=pol,
            height_box=height_box,
            reason=f"wj{i + 1}",
        )

    if (
        follow_spin
        and not track.top_reached
        and session.state.room_id == pol.ROOM_ID
    ):
        n_follow = pol.SAVE_WJ_FOLLOW if follow_frames is None else follow_frames
        for _ in range(n_follow):
            state = hold(
                session, 1, "RIGHT", "B", "A", reason=f"{label}_dwj_pd_spin"
            )
            if _track_upd(session, track, state, pol, height_box):
                break
            if (
                state.samus_x >= pol.RIGHT_SHELF_X
                and state.samus_y <= pol.MIDHIGH_Y
            ):
                break

    return bool(track.top_reached)


def double_walljump(
    session: ControllerSession,
    track: ClimbTrack,
    *,
    policy: WallJumpPolicy | None = None,
    height_class_out: list[bool] | None = None,
    extend_spin_ready: bool = False,
) -> bool:
    """Consecutive **double** wall-jump + right-spin Phase D push.

    Open-loop policy timings. Optional ``extend_spin_ready`` only stretches
    RIGHT+B+A if short of the wall — does **not** LEFT+A-seek.
    """
    pol = policy if policy is not None else _default_policy()
    height_box = height_class_out if height_class_out is not None else [False]
    ok = consecutive_walljumps(
        session,
        track,
        pol.R15_DOUBLE,
        policy=pol,
        pre_approach=True,
        extend_spin_ready=extend_spin_ready,
        height_box=height_box,
        follow_spin=True,
    )
    if height_class_out is not None:
        height_class_out[0] = bool(height_box[0])
    return ok


def walljump_second_left_wall(
    session: ControllerSession,
    track: ClimbTrack,
    *,
    policy: WallJumpPolicy | None = None,
    seek_frames: int | None = None,
    into_frames: int = 8,
    flip_frames: int = 16,
    height_box: list[bool] | None = None,
) -> bool:
    """Experiment: after right-wall WJ1, seek **left-wall** contact (pose 84).

    Does **not** LEFT+A-seek the right wall (that burns WJ1). Seeks with
    LEFT+B+A / LEFT spin toward the left column, then RIGHT+A flip.
    """
    pol = policy if policy is not None else _default_policy()
    label = track.label
    budget = pol.WJ2_LEFT_SEEK if seek_frames is None else seek_frames
    for _ in range(budget):
        state = session.state
        if _track_upd(session, track, state, pol, height_box):
            return bool(track.top_reached)
        pose = int(state.pose)
        if pose in (83, 84) or (
            int(state.samus_x) <= pol.WJ2_LEFT_X and int(state.samus_y) <= 200
        ):
            break
        state = hold(
            session, 1, "LEFT", "B", "A", reason=f"{label}_wj2_left_seek"
        )
        if _track_upd(session, track, state, pol, height_box):
            return bool(track.top_reached)
    for _ in range(into_frames):
        state = hold(session, 1, "RIGHT", "A", reason=f"{label}_wj2_left_into")
        if _track_upd(session, track, state, pol, height_box):
            return bool(track.top_reached)
    for _ in range(flip_frames):
        state = hold(
            session, 1, "RIGHT", "B", "A", reason=f"{label}_wj2_left_flip"
        )
        if _track_upd(session, track, state, pol, height_box):
            return bool(track.top_reached)
    return bool(track.top_reached)


def period_walljump_climb(
    session: ControllerSession,
    track: ClimbTrack,
    *,
    policy: WallJumpPolicy | None = None,
    frames: int = 48,
    period: int | None = None,
    into: int | None = None,
    bounce: int | None = None,
    height_box: list[bool] | None = None,
) -> bool:
    """Period-N right-structure WJ climb (bug-layout tolerant height farm).

    On latch, upgrades to a full consecutive double + follow (no single ship).
    """
    pol = policy if policy is not None else _default_policy()
    label = track.label
    per = pol.RIGHT_WJ_PERIOD if period is None else period
    n_into = pol.RIGHT_WJ_INTO if into is None else into
    n_bounce = pol.RIGHT_WJ_BOUNCE if bounce is None else bounce
    for i in range(frames):
        state = session.state
        if _track_upd(session, track, state, pol, height_box):
            return bool(track.top_reached)
        if is_wall_latch(state):
            return consecutive_walljumps(
                session,
                track,
                pol.R15_DOUBLE,
                policy=pol,
                pre_approach=False,
                extend_spin_ready=False,
                height_box=height_box,
                follow_spin=True,
            )
        ph = i % per
        if ph < n_into:
            state = hold(session, 1, "LEFT", "A", reason=f"{label}_pwj_into")
        elif ph < n_into + n_bounce:
            state = hold(session, 1, "RIGHT", "A", reason=f"{label}_pwj_bounce")
        else:
            state = hold(
                session, 1, "RIGHT", "B", "A", reason=f"{label}_pwj_spin"
            )
        if _track_upd(session, track, state, pol, height_box):
            return bool(track.top_reached)
    return bool(track.top_reached)


def damage_boost_hold(
    session: ControllerSession,
    track: ClimbTrack,
    *,
    policy: WallJumpPolicy | None = None,
    direction: str = "RIGHT",
    frames: int | None = None,
    height_box: list[bool] | None = None,
) -> bool:
    """**Experiment only** — hold A+dir during knockback to convert KB to height.

    Never gate product Phase D on this.
    """
    pol = policy if policy is not None else _default_policy()
    label = track.label
    n = pol.DMG_BOOST_HOLD_FRAMES if frames is None else frames
    for _ in range(n):
        state = hold(
            session, 1, direction, "A", reason=f"{label}_dmg_boost"
        )
        if _track_upd(session, track, state, pol, height_box):
            return bool(track.top_reached)
        if not is_knockback(session.state) and int(session.state.pose) not in (
            25,
            26,
            129,
            130,
        ):
            break
    return bool(track.top_reached)


# Historical bubble_* aliases (tests / probes that still use the old surface).
bubble_is_wall_latch = is_wall_latch
bubble_is_knockback = is_knockback
bubble_wall_approach_band = wall_approach_band
bubble_wait_wall_ready = wait_wall_ready
bubble_wait_wall_latch = wait_wall_latch
bubble_walljump_approach_coast = walljump_approach_coast
bubble_walljump_once = walljump_once
bubble_consecutive_walljumps = consecutive_walljumps
bubble_double_walljump_r15 = double_walljump
bubble_walljump_second_left_wall = walljump_second_left_wall
bubble_period_walljump_climb = period_walljump_climb
bubble_damage_boost_hold = damage_boost_hold

__all__ = [
    "POSE_WALL_LATCH",
    "WallJumpTiming",
    "is_wall_latch",
    "is_knockback",
    "wall_approach_band",
    "wait_wall_ready",
    "wait_wall_latch",
    "walljump_approach_coast",
    "walljump_once",
    "consecutive_walljumps",
    "double_walljump",
    "walljump_second_left_wall",
    "period_walljump_climb",
    "damage_boost_hold",
    "bubble_is_wall_latch",
    "bubble_is_knockback",
    "bubble_wall_approach_band",
    "bubble_wait_wall_ready",
    "bubble_wait_wall_latch",
    "bubble_walljump_approach_coast",
    "bubble_walljump_once",
    "bubble_consecutive_walljumps",
    "bubble_double_walljump_r15",
    "bubble_walljump_second_left_wall",
    "bubble_period_walljump_climb",
    "bubble_damage_boost_hold",
]

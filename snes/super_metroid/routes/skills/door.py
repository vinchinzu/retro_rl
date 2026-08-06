"""Parameterized Super-door pressure helpers.

Two surfaces:

* :func:`top_super_door` — Bubble → Bat Phase E (policy-driven WJ + Super band).
* :func:`super_door_pressure_frame` — simple plant/shoot cadence for K4-style
  colored Super doors (Cathedral Entrance / Cathedral green). Callers own xy
  bands and frame budgets; this does **not** require Bubble ``DoorPolicy``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    WallJumpTiming,
    hold,
    is_wall_latch,
    select_weapon,
    wait_ordinary_room,
    walljump_once,
)
from super_metroid.routes.skills.geometry import (
    ClimbTrack,
    avoid_wrong_door,
    track_state,
)
from super_metroid.routes.skills.knockback import is_knockback

if TYPE_CHECKING:
    from super_metroid.routes.runtime import ControllerSession


class DoorPolicy(Protocol):
    """Policy fields for Super-door pressure."""

    ROOM_ID: int
    EXIT_ROOM_ID: int
    TRUE_GROUND: frozenset[int]
    DOOR_FRAMES: int
    DOOR_SUPER_X: int
    DOOR_SUPER_Y: int
    DOOR_WJ_PERIOD: int
    DOOR_WJ_INTO: int
    DOOR_WJ_BOUNCE: int
    DOOR_X_CAP: int
    DOOR_OUTER_X: int
    DOOR_FALL_Y: int
    DOOR_CROUCH_FRAMES: int
    DOOR_WJ_POSES: frozenset[int]
    DOOR_WJ: WallJumpTiming
    TO_BAT_SETTLE_FRAMES: int
    # GeometryPolicy surface for track_state / avoid_wrong_door
    STAND_PIN: frozenset[int]
    MID_Y: int
    MID_STAND_X: tuple[int, int]
    LIP_X: tuple[int, int]
    LIP_Y: tuple[int, int]
    RIGHT_SHELF_X: int
    RIGHT_SHELF_Y: int
    CAVITY_X_MAX: int
    SAVE_RUNWAY_X: tuple[int, int]
    SAVE_RUNWAY_Y: tuple[int, int]
    SAVE_RUNWAY_FIRE_X: tuple[int, int]
    PHASE_C_X_MIN: int
    PHASE_C_Y_MIN: int
    PHASE_C_Y_MAX: int
    PHASE_D_X: int
    PHASE_D_Y: int


def _default_policy() -> DoorPolicy:
    from super_metroid.routes.skills.policies import bubble_to_bat as pol

    return pol  # type: ignore[return-value]


def super_door_pressure_frame(
    session: ControllerSession,
    frame: int,
    *,
    label: str,
    face: str = "RIGHT",
    weapon: int = 2,
    period: int,
    shoot_end: int,
    idle_end: int | None = None,
    face_end: int | None = None,
    run_end: int | None = None,
    ensure_weapon: bool = True,
    reason: str = "door",
) -> SuperMetroidState:
    """One open-loop Super-door plant/shoot pressure frame (K4-style).

    Phase windows on ``frame % period`` (exclusive ends), evaluated in order:

    * ``[0, shoot_end)`` → ``face`` + X
    * ``[shoot_end, idle_end)`` → idle (only if ``idle_end`` set)
    * up to ``face_end`` → ``face`` only
    * up to ``run_end`` → ``face`` + B
    * remainder → ``face`` + B + A

    Cathedral Entrance door band: ``period=28, shoot_end=5, face_end=12,
    run_end=20``. Cathedral green Super: ``period=40, shoot_end=4,
    idle_end=20, face_end=28, run_end=34``. Does not touch Bubble Phase E.
    """
    if ensure_weapon and int(session.state.selected_item) != weapon:
        select_weapon(session, weapon)
    phase = frame % period
    if phase < shoot_end:
        inputs: tuple[str, ...] = (face, "X")
    elif idle_end is not None and phase < idle_end:
        inputs = ()
    elif face_end is not None and phase < face_end:
        inputs = (face,)
    elif run_end is not None and phase < run_end:
        inputs = (face, "B")
    else:
        inputs = (face, "B", "A")
    return hold(session, 1, *inputs, reason=f"{label}_{reason}")


def top_super_door(
    session: ControllerSession,
    track: ClimbTrack,
    *,
    policy: DoorPolicy | None = None,
) -> SuperMetroidState:
    """Phase E: top-right Super door → ordinary exit room.

    From Phase D pin: keep Supers selected; latch-gated right-structure WJ
    until super-pressure band, then RIGHT+X/B Super pressure into exit room.

    Open-loop period WJ alone is pure-source-fragile; latch-gated pulses +
    fall re-climb recover.
    """
    pol = policy if policy is not None else _default_policy()
    label = track.label
    room_id = pol.ROOM_ID
    exit_id = pol.EXIT_ROOM_ID

    if session.state.selected_item != 2:
        select_weapon(session, 2)

    # Land + crouch settle on top structure before sticky WJ.
    # DOOR_CROUCH_FRAMES (policy): continuous Spazer Phase E needs longer
    # crouch than pure baseline 8f so SEEK phase matches Geruta-clear window.
    for _ in range(24):
        st = session.state
        if st.room_id != room_id:
            break
        track_state(session, track, st, pol)  # type: ignore[arg-type]
        if int(st.pose) in pol.TRUE_GROUND or int(st.pose) in (25, 26, 27, 28):
            break
        hold(session, 1, "RIGHT", reason=f"{label}_door_land")
    crouch_n = int(getattr(pol, "DOOR_CROUCH_FRAMES", 8))
    for _ in range(max(0, crouch_n)):
        if session.state.room_id != room_id:
            break
        hold(session, 1, "DOWN", reason=f"{label}_door_crouch")
    for _ in range(4):
        if session.state.room_id != room_id:
            break
        hold(session, 1, reason=f"{label}_door_uncrouch")

    for frame in range(pol.DOOR_FRAMES):
        state = session.state
        if state.room_id == exit_id:
            break
        if state.room_id != room_id:
            break
        track_state(session, track, state, pol)  # type: ignore[arg-type]
        if avoid_wrong_door(session, track, state, pol):  # type: ignore[arg-type]
            continue
        if is_knockback(state):
            for _ in range(8):
                hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_door_kb")
            continue

        # Stay on Supers for the whole Phase E (beam swap was a prior regress).
        if state.selected_item != 2:
            select_weapon(session, 2)

        x = int(state.samus_x)
        y = int(state.samus_y)
        if y <= pol.DOOR_SUPER_Y and x >= pol.DOOR_SUPER_X:
            track.door_reached = True
            if frame % 5 < 2:
                hold(session, 1, "RIGHT", "X", reason=f"{label}_door_super")
            else:
                hold(session, 1, "RIGHT", "B", reason=f"{label}_door_press")
            if session.state.room_id == exit_id:
                break
            continue

        # Fell below usable top band: re-approach right structure.
        if y > pol.DOOR_FALL_Y:
            if x >= pol.DOOR_OUTER_X:
                hold(session, 1, "LEFT", "B", reason=f"{label}_door_outer_pull")
            elif x < 360:
                hold(
                    session, 1, "RIGHT", "B", "A", reason=f"{label}_door_reclimb"
                )
            elif int(state.pose) in pol.DOOR_WJ_POSES or is_wall_latch(state):
                walljump_once(
                    session, pol.DOOR_WJ, reason=f"{label}_door_reclimb_wj"
                )
            else:
                ph = frame % pol.DOOR_WJ_PERIOD
                if ph < pol.DOOR_WJ_INTO:
                    hold(
                        session, 1, "LEFT", "A", reason=f"{label}_door_reseek"
                    )
                elif ph < pol.DOOR_WJ_INTO + pol.DOOR_WJ_BOUNCE:
                    hold(
                        session,
                        1,
                        "RIGHT",
                        "A",
                        reason=f"{label}_door_rebounce",
                    )
                else:
                    hold(
                        session,
                        1,
                        "RIGHT",
                        "B",
                        "A",
                        reason=f"{label}_door_respin",
                    )
            if session.state.room_id == exit_id:
                break
            continue

        # High band: prefer latch-gated WJ over open-loop period.
        if int(state.pose) in pol.DOOR_WJ_POSES or is_wall_latch(state):
            walljump_once(session, pol.DOOR_WJ, reason=f"{label}_door_wj")
            if session.state.room_id == exit_id:
                break
            continue

        if x > pol.DOOR_X_CAP:
            hold(session, 1, "LEFT", "B", reason=f"{label}_door_cap")
            if session.state.room_id == exit_id:
                break
            continue

        # Not latched yet: period seek toward right structure (fallback).
        ph = frame % pol.DOOR_WJ_PERIOD
        if ph < pol.DOOR_WJ_INTO:
            hold(session, 1, "LEFT", "A", reason=f"{label}_door_wj_into")
        elif ph < pol.DOOR_WJ_INTO + pol.DOOR_WJ_BOUNCE:
            hold(session, 1, "RIGHT", "A", reason=f"{label}_door_wj_bounce")
        else:
            hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_door_wj_spin")
        if session.state.room_id == exit_id:
            break
    else:
        state = session.state
        raise TimeoutError(
            f"{label}: exit Super door missed before room "
            f"0x{exit_id:04X}; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"door_transition={state.door_transition} max_x={track.max_x} "
            f"min_y={track.min_y} mid_reached={track.mid_reached} "
            f"top_reached={track.top_reached} door_reached={track.door_reached} "
            f"standing_mid_pinned={track.standing_mid_pinned} "
            f"launched={track.launched} phase_c_hit={track.phase_c_hit} "
            f"supers={state.super_missiles} selected={state.selected_item}"
        )

    if session.state.room_id != exit_id:
        state = session.state
        raise TimeoutError(
            f"{label}: left climb room without ordinary exit; "
            f"room=0x{state.room_id:04X} pose={state.pose} "
            f"xy=({state.samus_x},{state.samus_y}) max_x={track.max_x} "
            f"min_y={track.min_y} mid_reached={track.mid_reached} "
            f"top_reached={track.top_reached} door_reached={track.door_reached} "
            f"standing_mid_pinned={track.standing_mid_pinned} "
            f"phase_c_hit={track.phase_c_hit}"
        )

    return wait_ordinary_room(
        session,
        exit_id,
        settle_frames=pol.TO_BAT_SETTLE_FRAMES,
        label=label,
    )


# Historical name used by product hop / tests.
bubble_top_super_door = top_super_door

__all__ = [
    "super_door_pressure_frame",
    "top_super_door",
    "bubble_top_super_door",
]

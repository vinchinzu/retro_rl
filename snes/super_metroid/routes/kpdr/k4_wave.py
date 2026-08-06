"""K4 Wave branch pure controllers — Bubble → Single Chamber (K4.8).

Post-Speed return handoff sits Bubble Mountain top-right ``0xACB3``
~(472,115). Wave path exits via middle-right blue door into Single Chamber
``0xAD5E`` left shaft top door.

Human reference: ``tasks/speed_to_ice_moat_human.json`` frames 2131–2637
(Bubble segment after Bat return). Path: top platform walk-left → drop shaft
at x≈381 → mid hop → floor sill y≈395 → RIGHT blue door.

Caps: Morph, Bombs, Missiles, Supers, Hi-Jump, Varia, Speed.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    hold,
    require_room,
    select_weapon,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.k4_common import _STANDING_POSES
from super_metroid.routes.kpdr.rooms import (
    ROOM_BUBBLE,
    ROOM_SINGLE_CHAMBER,
)
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import (
    escape_knockback_spin,
    is_knockback,
)

# Live pins from post_speed_return_to_bubble_pure + human Bubble→SC (2026-08-06).
# Top settle ~(472,115); drop band x≈381; door sill ~(492,395).
_TOP_Y_MAX = 200
_DROP_X = (370, 400)
_DROP_TARGET_X = 385
_MID_Y = (220, 340)
_FLOOR_Y = 360
_DOOR_Y = (380, 420)
_DOOR_X = 470
_SINGLE_SETTLE = 320
_TOTAL_BUDGET = 5000
_TOP_WALK_FRAMES = 400
_DROP_FRAMES = 500
_NAV_TO_DOOR_FRAMES = 1200
_DOOR_PUSH_FRAMES = 400


def _escape_kb(session: ControllerSession, label: str, prefer: str) -> None:
    escape_knockback_spin(
        session,
        prefer_dir=prefer,
        run_frames=6,
        spin_frames=18,
        label=label,
        stop_room_id=ROOM_SINGLE_CHAMBER,
    )


def _top_walk_to_drop(session: ControllerSession, label: str) -> None:
    """Top-right settle → walk LEFT to drop shaft band x∈[370,400]."""
    unmorph(session)
    select_weapon(session, 0)
    for _ in range(30):
        state = hold(session, 1, reason=f"{label}_top_stand")
        if state.velocity_y == 0 and state.pose in _STANDING_POSES:
            break

    for frame in range(_TOP_WALK_FRAMES):
        state = session.state
        if state.room_id != ROOM_BUBBLE:
            return
        if state.samus_y > _TOP_Y_MAX:
            return  # already dropping / below top
        if (
            _DROP_X[0] <= state.samus_x <= _DROP_X[1]
            and state.velocity_y == 0
        ):
            return
        if is_knockback(state):
            _escape_kb(session, label, "LEFT")
            continue
        # Near drop band: short walk to center then stop.
        if state.samus_x < _DROP_X[0]:
            hold(session, 1, "RIGHT", reason=f"{label}_top_r")
        elif state.samus_x > _DROP_X[1]:
            phase = frame % 14
            if phase < 10:
                hold(session, 1, "LEFT", "B", reason=f"{label}_top_run")
            else:
                hold(session, 1, "LEFT", reason=f"{label}_top_walk")
        else:
            hold(session, 1, reason=f"{label}_top_seat")


def _drop_shaft(session: ControllerSession, label: str) -> None:
    """Drop from top band through right shaft to floor/mid y≥360."""
    for frame in range(_DROP_FRAMES):
        state = session.state
        if state.room_id != ROOM_BUBBLE:
            return
        if state.samus_y >= _FLOOR_Y and state.velocity_y == 0:
            return
        if is_knockback(state):
            _escape_kb(session, label, "LEFT")
            continue

        # Keep roughly over the shaft while falling / hopping off lip.
        if state.velocity_y == 0 and state.samus_y <= _TOP_Y_MAX:
            # Step off lip: brief LEFT then free fall.
            if state.samus_x > _DROP_TARGET_X + 8:
                hold(session, 1, "LEFT", reason=f"{label}_lip_left")
            elif state.samus_x < _DROP_X[0]:
                hold(session, 1, "RIGHT", reason=f"{label}_lip_right")
            else:
                # Nudge off edge / open air.
                phase = frame % 12
                if phase < 4:
                    hold(session, 1, "LEFT", reason=f"{label}_step_off")
                elif phase < 7:
                    hold(session, 1, "A", reason=f"{label}_lip_hop")
                else:
                    hold(session, 1, reason=f"{label}_lip_wait")
            continue

        # Air: slight left bias to land mid-right platforms (human ~x381).
        if state.samus_x > 400:
            hold(session, 1, "LEFT", reason=f"{label}_fall_l")
        elif state.samus_x < 350:
            hold(session, 1, "RIGHT", reason=f"{label}_fall_r")
        else:
            hold(session, 1, reason=f"{label}_fall")


def _nav_floor_to_door(session: ControllerSession, label: str) -> None:
    """Mid/floor platforms → right blue door sill ~(492,395)."""
    unmorph(session)
    select_weapon(session, 0)

    for frame in range(_NAV_TO_DOOR_FRAMES):
        state = session.state
        if state.room_id == ROOM_SINGLE_CHAMBER:
            return
        if state.room_id != ROOM_BUBBLE:
            return
        if is_knockback(state):
            _escape_kb(session, label, "RIGHT")
            continue

        on_door_sill = (
            state.samus_x >= _DOOR_X
            and _DOOR_Y[0] <= state.samus_y <= _DOOR_Y[1]
            and state.velocity_y == 0
        )
        if on_door_sill:
            return

        # Too high mid: drop further or hop toward right.
        if state.samus_y < _FLOOR_Y:
            if state.velocity_y == 0 and state.pose in _STANDING_POSES:
                # Mid platforms: human hops left once then runs right down.
                # Prefer right progress once below top.
                if state.samus_x < 360 and state.samus_y < 300:
                    # Short left hop onto solid mid ledge (human ~341,228).
                    hold(session, 1, "LEFT", "A", reason=f"{label}_mid_hop")
                elif state.samus_x < 420:
                    phase = frame % 18
                    if phase < 6:
                        hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_mid_spin")
                    elif phase < 12:
                        hold(session, 1, "RIGHT", "B", reason=f"{label}_mid_run")
                    else:
                        hold(session, 1, "RIGHT", reason=f"{label}_mid_walk")
                else:
                    # Over right, drop down.
                    hold(session, 1, "RIGHT", reason=f"{label}_mid_drop")
            else:
                # Air: drift right toward door column.
                if state.samus_x < 450:
                    hold(session, 1, "RIGHT", reason=f"{label}_air_r")
                else:
                    hold(session, 1, reason=f"{label}_air")
            continue

        # Floor band y≥360: run right toward door; hop small gaps.
        if state.samus_x < _DOOR_X:
            phase = frame % 22
            if phase < 6:
                hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_floor_hop")
            elif phase < 14:
                hold(session, 1, "RIGHT", "B", reason=f"{label}_floor_run")
            else:
                hold(session, 1, "RIGHT", reason=f"{label}_floor_walk")
        else:
            hold(session, 1, "RIGHT", reason=f"{label}_sill_nudge")


def _push_right_blue_door(session: ControllerSession, label: str) -> None:
    """Sill pressure: RIGHT+X + dash through middle-right blue door."""
    select_weapon(session, 0)
    for frame in range(_DOOR_PUSH_FRAMES):
        state = session.state
        if state.room_id == ROOM_SINGLE_CHAMBER:
            return
        if state.room_id != ROOM_BUBBLE:
            return
        if is_knockback(state):
            _escape_kb(session, label, "RIGHT")
            continue

        # Fell off sill: climb back.
        if state.samus_y > 430:
            hold(session, 1, "LEFT", "B", "A", reason=f"{label}_under_recover")
            continue
        if state.samus_x < _DOOR_X - 20:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_reapproach")
            continue

        phase = frame % 16
        if phase < 5:
            hold(session, 1, "RIGHT", "X", reason=f"{label}_door_shot")
        elif phase < 11:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_door_push")
        else:
            hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_door_spin")


def play_bubble_to_single_chamber(session: ControllerSession) -> SuperMetroidState:
    """Bubble Mountain (post-Speed return) → ordinary Single Chamber.

    Path: top-right → drop shaft ~x385 → floor sill → middle-right blue door
    into ``0xAD5E``. Caps: Morph, Bombs, Missiles, Supers, Hi-Jump, Varia, Speed.
    """
    label = "bubble_to_single_chamber"
    require_room(session, ROOM_BUBBLE, label)
    start = session.frame

    if session.state.samus_y <= _TOP_Y_MAX:
        _top_walk_to_drop(session, label)
        if session.state.room_id == ROOM_BUBBLE and session.state.samus_y < _FLOOR_Y:
            _drop_shaft(session, label)

    if session.state.room_id == ROOM_BUBBLE:
        _nav_floor_to_door(session, label)

    if session.state.room_id == ROOM_BUBBLE:
        _push_right_blue_door(session, label)

    if session.state.room_id != ROOM_SINGLE_CHAMBER:
        state = session.state
        raise TimeoutError(
            f"{label}: Single Chamber door missed; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"door_transition={state.door_transition} "
            f"frames={session.frame - start}"
        )

    state = wait_ordinary_room(
        session, ROOM_SINGLE_CHAMBER, settle_frames=_SINGLE_SETTLE, label=label
    )
    if session.frame - start > _TOTAL_BUDGET:
        pass  # soft budget; room success is the gate
    return state


__all__ = [
    "play_bubble_to_single_chamber",
    "ROOM_BUBBLE",
    "ROOM_SINGLE_CHAMBER",
]

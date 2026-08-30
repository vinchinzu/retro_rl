"""Bat Room → Red Tower LEFT across dry pipe platforms (K5 hop 11).

Room is Skree Boost Room ``0xA3DD`` (also called Bat Room) — not Norfair
Bat Cave. Left blue door is Red Tower bottom ``0xA253``. Wave is already
held on this stack (beams ``0x1007``).

Public policy (Hi-Jump already held): spin-jump left across the three dry
pipe platforms and shoot crawlers. Reverse of ``play_bat_to_below_spazer``.
The wiki Skree boost is the no-HJ early-game strat, not this hop.
https://wiki.supermetroid.run/Skree_Boost_Room

Water under the pipes is a different climb. HJ without Gravity does not
spin-jump out — mash-A bonks the pipe underside. Map Rando helper
``h_underwaterCrouchJumpDownGrab``: crouch-jump then down-grab the lip.

``below_to_bat`` pins the right sill crouched (pose 12 at ~(472,139)).
A frame-0 spin-jump from that crouch drops into the water. Stand first,
run up, then hop. Door exit only when planted on the high left sill.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    hold,
    play_run_shoot_exit,
    require_room,
    select_weapon,
)
from super_metroid.routes.kpdr.red_tower.geometry import (
    BAT_TO_RED_DOOR_SEAT_X,
    BAT_TO_RED_EXIT_HOLD,
    BAT_TO_RED_EXIT_RUN,
    BAT_TO_RED_EXIT_SHOOT,
    BAT_TO_RED_EXIT_SPIN,
    BAT_TO_RED_HIGH_Y,
    BAT_TO_RED_JUMP_HOLD,
    BAT_TO_RED_JUMP_PERIOD,
    BAT_TO_RED_PROGRESS_WINDOW,
    BAT_TO_RED_RUNUP,
    BAT_TO_RED_TRAVERSE_BUDGET,
    BAT_TO_RED_WATER_CJ_CROUCH,
    BAT_TO_RED_WATER_CJ_JUMP,
    BAT_TO_RED_WATER_GRAB,
)
from super_metroid.routes.kpdr.rooms import ROOM_BAT, ROOM_RED_TOWER
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.basic_moves import crouch_jump, down_grab
from super_metroid.routes.skills.geometry import POSE_KNOCKBACK
from super_metroid.routes.skills.knockback import escape_knockback_spin

# Morph / spring. Do not include spin-jump 27/28 — UP would cancel the
# pipe hops. 41/42/43 are morph-falling on the sill edge. 11/12 are crouch.
_MORPH = frozenset(
    {
        29,
        30,
        31,
        32,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        49,
        50,
        65,
        66,
    }
)
_CROUCH = frozenset({11, 12})


def _is_morph_like(pose: int) -> bool:
    return int(pose) in _MORPH


def _is_crouch(pose: int) -> bool:
    return int(pose) in _CROUCH


def _is_knockback(state: SuperMetroidState) -> bool:
    return int(state.pose) in POSE_KNOCKBACK


def _in_water(state: SuperMetroidState) -> bool:
    """Below the dry pipe tops — the pool, not the high path."""
    return int(state.samus_y) > BAT_TO_RED_HIGH_Y


def _on_left_door_seat(state: SuperMetroidState) -> bool:
    """High left sill in front of the Red Tower blue door (block [0,7])."""
    return (
        int(state.room_id) == ROOM_BAT
        and int(state.samus_x) <= BAT_TO_RED_DOOR_SEAT_X
        and int(state.samus_y) <= BAT_TO_RED_HIGH_Y
    )


def _traverse_buttons(frame: int, state: SuperMetroidState) -> list[str]:
    """LEFT run/jump/beam on the dry pipes. No jump during the first run-up.

    A must be released between jumps — Super Metroid only jumps on press.
    """
    del state
    jumping = (
        frame >= BAT_TO_RED_RUNUP
        and frame % BAT_TO_RED_JUMP_PERIOD < BAT_TO_RED_JUMP_HOLD
    )
    buttons = ["LEFT", "B", "X"]
    if jumping:
        buttons.append("A")
    return buttons


def _stand_up(session: ControllerSession) -> None:
    """Leave crouch (pose 12) and morph. Do not idle — this sill subpixel-falls."""
    pose = int(session.state.pose)
    if pose not in _CROUCH and not _is_morph_like(pose):
        return
    hold(session, 8, "UP", reason="bat_to_red_stand")
    pose = int(session.state.pose)
    if pose in _CROUCH or _is_morph_like(pose):
        hold(session, 8, "UP", reason="bat_to_red_stand")


def _water_direction(state: SuperMetroidState) -> str:
    return "RIGHT" if int(state.samus_x) < 40 else "LEFT"


def _climb_out_of_water(session: ControllerSession) -> None:
    """Crouch-jump + down-grab the pipe lip. Mash-A does not clear HJ water."""
    x0 = int(session.state.samus_x)
    direction = _water_direction(session.state)
    if _is_morph_like(int(session.state.pose)):
        for _ in range(40):
            hold(session, 1, direction, reason="bat_to_red_water_roll")
            if int(session.state.room_id) != ROOM_BAT:
                return
            if not _is_morph_like(int(session.state.pose)):
                break
            if int(session.state.samus_x) < x0 - 24:
                break
        _stand_up(session)
        if _is_morph_like(int(session.state.pose)) or not _in_water(session.state):
            return
    else:
        _stand_up(session)
    direction = _water_direction(session.state)
    crouch_jump(
        session,
        crouch_frames=BAT_TO_RED_WATER_CJ_CROUCH,
        jump_frames=BAT_TO_RED_WATER_CJ_JUMP,
        direction=direction,
        reason="bat_to_red_water_cj",
    )
    down_grab(
        session,
        frames=BAT_TO_RED_WATER_GRAB,
        direction=direction,
        reason="bat_to_red_water_grab",
    )


def _clear_obstacle(session: ControllerSession, *, label: str) -> None:
    """High-path geemer: jump and shoot. Water: climb the lip, do not mash A."""
    if _in_water(session.state):
        _climb_out_of_water(session)
        return
    x = int(session.state.samus_x)
    direction = "RIGHT" if x <= 40 else "LEFT"
    for _ in range(18):
        hold(session, 1, direction, "A", reason=f"{label}_jump")
    for frame in range(34):
        buttons = [direction]
        if frame % 3 == 0:
            buttons.append("X")
        hold(session, 1, *buttons, reason=f"{label}_aim_shoot")
    hold(session, 10, reason=f"{label}_land")


def _break_knockback(session: ControllerSession) -> None:
    """A-burst then land; idle A every frame re-hits and never clears pose 138."""
    hold(session, 12, "A", reason="bat_to_red_kb_a")
    for _ in range(24):
        state = hold(session, 1, reason="bat_to_red_kb_land")
        if not _is_knockback(state):
            return
    if _is_knockback(session.state):
        escape_knockback_spin(
            session,
            prefer_dir="LEFT",
            run_frames=3,
            spin_frames=12,
            label="bat_to_red",
            stop_room_id=ROOM_RED_TOWER,
        )


def _exit_to_red(session: ControllerSession) -> SuperMetroidState:
    play_run_shoot_exit(
        session,
        from_room=ROOM_BAT,
        to_room=ROOM_RED_TOWER,
        direction="LEFT",
        label="bat_to_red",
        run_frames=BAT_TO_RED_EXIT_RUN,
        shoot_frames=BAT_TO_RED_EXIT_SHOOT,
        spin_frames=BAT_TO_RED_EXIT_SPIN,
        hold_frames=BAT_TO_RED_EXIT_HOLD,
        settle_frames=40,
    )
    return _seat_red_bottom(session)


def _seat_red_bottom(session: ControllerSession) -> SuperMetroidState:
    """Land on Red bottom at the 718f successor ~(216,2443) p10.

    ``red_to_hellway``'s open-loop body is calibrated to that pin. Airborne
    p82 or land-pose 165 on the Bat door lip walks back into Skree Boost.
    Return while still running LEFT — idle turns p10 into p165.
    """
    for _ in range(120):
        st = session.state
        if int(st.room_id) != ROOM_RED_TOWER:
            return st
        x = int(st.samus_x)
        y = int(st.samus_y)
        grounded = abs(int(st.velocity_y)) == 0 and y >= 2430
        if x >= 222:
            hold(session, 1, "LEFT", reason="bat_to_red_bat_door")
            continue
        if not grounded:
            hold(session, 1, reason="bat_to_red_land")
            continue
        if x > 216:
            hold(session, 1, "LEFT", reason="bat_to_red_seat_l")
            continue
        if x < 216:
            hold(session, 1, "RIGHT", reason="bat_to_red_seat_r")
            continue
        return st
    hold(session, 4, "LEFT", reason="bat_to_red_pose10")
    return session.state


def play_bat_to_red(session: ControllerSession) -> SuperMetroidState:
    """Bat left blue door → ordinary Red Tower bottom.

    Expects the right high sill after below_to_bat. Stand out of the pose-12
    crouch before hopping; if a live geemer knocks Samus into the pool, climb
    the lip with crouch-jump + down-grab.
    """
    require_room(session, ROOM_BAT, "bat_to_red")
    _stand_up(session)
    select_weapon(session, 0)

    best_x = int(session.state.samus_x)
    best_y = int(session.state.samus_y)
    stale = 0
    deadline = int(session.frame) + BAT_TO_RED_TRAVERSE_BUDGET
    frame = 0
    while int(session.frame) < deadline:
        state = session.state
        if int(state.room_id) == ROOM_RED_TOWER:
            return _seat_red_bottom(session)
        if int(state.room_id) != ROOM_BAT:
            raise RuntimeError(
                "bat_to_red: unexpected room "
                f"0x{int(state.room_id):04X}: {state}"
            )

        if _is_knockback(state):
            _break_knockback(session)
            continue

        if _on_left_door_seat(state):
            if abs(int(state.velocity_y)) > 0:
                hold(session, 1, "LEFT", "B", "X", reason="bat_to_red_door_air")
                continue
            try:
                return _exit_to_red(session)
            except TimeoutError:
                _break_knockback(session)
                continue

        if _in_water(state):
            _climb_out_of_water(session)
            stale = 0
            best_x = int(session.state.samus_x)
            best_y = int(session.state.samus_y)
            continue

        if _is_morph_like(int(state.pose)) or _is_crouch(int(state.pose)):
            _stand_up(session)
            continue

        x = int(state.samus_x)
        y = int(state.samus_y)
        progressed = x < best_x - 2
        if progressed:
            best_x = min(best_x, x)
            best_y = min(best_y, y)
            stale = 0
        else:
            stale += 1

        if stale >= BAT_TO_RED_PROGRESS_WINDOW:
            _clear_obstacle(session, label="bat_to_red_stall")
            stale = 0
            best_x = int(session.state.samus_x)
            best_y = int(session.state.samus_y)
            continue

        hold(session, 1, *_traverse_buttons(frame, state), reason="bat_to_red_traverse")
        frame += 1

    raise TimeoutError(f"bat_to_red: traverse timeout: {session.state}")


__all__ = ["play_bat_to_red"]

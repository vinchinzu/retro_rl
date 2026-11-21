"""Tourian First Room (0xDAAE) direct crossing to Metroid Room 1 (0xDAE1).

Replaces the Upper Tourian Save Room detour (rows 313-315) with a restore-free
RAM-driven controller: settles the arriving Statues elevator, cancels Grapple
so Beam is active, drops to the lower left door, and settles in Metroid Room 1.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    hold,
    require_room,
    select_weapon,
    wait_ordinary_room,
)
from super_metroid.routes.runtime import ControllerSession

ROOM_TOURIAN_FIRST = 0xDAAE
ROOM_METROID_1 = 0xDAE1
_POSE_ELEVATOR = 155


def play_tourian_first_room_direct(session: ControllerSession) -> SuperMetroidState:
    """Tourian First Room elevator → Metroid Room 1 (0xDAE1) without Save Room."""
    label = "tourian_first_room_direct"
    require_room(session, ROOM_TOURIAN_FIRST, label)

    # Ride the arriving elevator down until it settles at y=680.
    stable_frames = 0
    last_y = session.state.samus_y
    for _ in range(700):
        state = hold(session, 1, reason=f"{label}_elevator_settle")
        if state.samus_y == last_y and state.samus_y >= 670 and state.velocity_y == 0:
            stable_frames += 1
            if stable_frames >= 12:
                break
        else:
            stable_frames = 0
            last_y = state.samus_y
    else:
        raise TimeoutError(f"{label}: elevator did not settle: {session.state}")

    # Step left off the elevator pad (x=128) and cancel any selected sub-weapon.
    hold(session, 10, "LEFT", "B", reason=f"{label}_step_off")
    if session.state.selected_item != 0:
        hold(session, 2, "LEFT", "B", "Y", reason=f"{label}_cancel_weapon")
        hold(session, 4, "LEFT", "B", reason=f"{label}_cancel_settle")
    if session.state.selected_item != 0:
        select_weapon(session, 0)

    # Drop off the left side of the elevator pad (y < 760 -> LEFT) onto the
    # y=827 shelf, run right off its edge (x=130, y=865), then Space-Jump left
    # over the acid pit straight onto the (20, 907) left door into Metroid 1.
    for frame in range(500):
        state = session.state
        if state.room_id == ROOM_METROID_1:
            break
        x, y = state.samus_x, state.samus_y
        if y < 780:
            inputs = ("LEFT", "B")
        elif y < 860:
            inputs = ("RIGHT", "B")
        elif y > 908:
            # Hold A while jumping/rising; step straight up when next to x=69 wall.
            rising = state.vertical_direction == 1
            can_start = state.vertical_direction == 0 and (frame % 24) < 16
            sj = state.vertical_direction == 2 and (frame % 10) < 3
            if rising or can_start or sj:
                inputs = ("A",) if x < 82 else ("LEFT", "B", "A")
            else:
                inputs = ("RIGHT", "B") if x < 78 else ("LEFT", "B")
        else:
            inputs = ("LEFT", "B", "X") if (frame % 10) < 3 else ("LEFT", "B")
        hold(session, 1, *inputs, reason=f"{label}_to_metroid1")
    else:
        raise TimeoutError(f"{label}: Metroid Room 1 door missed: {session.state}")

    return wait_ordinary_room(
        session,
        ROOM_METROID_1,
        settle_frames=320,
        label=label,
    )

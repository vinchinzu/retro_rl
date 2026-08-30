"""Blue ceiling-door tap cycle shared by Basement hatch and Attic door.

Seat / lip / shaft geometry is hop-owned. Both hops keep the same buttons
for the same x / y / pose / frame. The session loop is shared too.
"""

from __future__ import annotations

from collections.abc import Callable

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    hold,
    is_morph,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import is_knockback

_HURT = frozenset({137, 138})
_LAND_POSES = frozenset({1, 2, 9, 10})
_AIR_DRIFT_POSES = frozenset({21, 22, 25, 26, 47, 48, 81, 82, 105, 106})
_SHOOT_FRAMES = 240
_CYCLE = 80
_FIRE_PHASE = 60
_WAIT_PHASE = 68
_SHAFT_SLACK = 4


def tap_up_action(
    frame: int,
    *,
    hold_charge: bool,
    fire_phase: int = _FIRE_PHASE,
    wait_phase: int = _WAIT_PHASE,
) -> tuple[str, ...]:
    """Charge-release UP, then A. ``hold_charge`` waits ``_SHOOT_FRAMES``."""
    phase = int(frame) % _CYCLE
    if phase < int(fire_phase):
        return ("UP", "X")
    waiting = phase < int(wait_phase)
    if hold_charge:
        waiting = int(frame) < _SHOOT_FRAMES or waiting
    if waiting:
        return ("UP",)
    return ("UP", "A")


def ceiling_door_action(
    samus_x: int,
    samus_y: int,
    pose: int,
    frame: int,
    *,
    seat_x: int,
    lip_y: int,
    shaft_y: int,
    slack: int,
    hold_charge: bool,
    fire_phase: int = _FIRE_PHASE,
    wait_phase: int = _WAIT_PHASE,
) -> tuple[str, ...] | None:
    """Shaft and lip of a blue ceiling door. None = still below ``lip_y``."""
    if int(pose) in _HURT:
        return ()
    x = int(samus_x)
    y = int(samus_y)
    if y < int(shaft_y):
        if x > int(seat_x) + _SHAFT_SLACK:
            return ("LEFT", "A")
        if x < int(seat_x) - _SHAFT_SLACK:
            return ("RIGHT", "A")
        return ("A",)
    if y < int(lip_y):
        # Attic take02 shoots from the left edge of the seat, then uses the
        # first (blocked) jump to center the second jump under the open door.
        if int(pose) in _AIR_DRIFT_POSES:
            if x > int(seat_x) + _SHAFT_SLACK:
                return ("LEFT", "UP", "A")
            if x < int(seat_x) - _SHAFT_SLACK:
                return ("RIGHT", "UP", "A")
        if x > int(seat_x) + int(slack):
            return ("LEFT", "UP", "A")
        if x < int(seat_x) - int(slack):
            return ("RIGHT", "UP", "A")
        return tap_up_action(
            frame,
            hold_charge=hold_charge,
            fire_phase=fire_phase,
            wait_phase=wait_phase,
        )
    return None


def play_ceiling_door(
    session: ControllerSession,
    *,
    label: str,
    dest_room: int,
    lip_y: int,
    remount: Callable[[SuperMetroidState], tuple[str, ...]],
    door_action: Callable[[SuperMetroidState, int], tuple[str, ...]],
    guard: Callable[[ControllerSession, str], None],
    on_knockback: Callable[[ControllerSession, str], None],
    side_rooms: tuple[int, ...] = (),
    on_side_room: Callable[[ControllerSession, str], None] | None = None,
    budget: int = 800,
) -> None:
    """Shoot then jump UP through a blue ceiling door until ``dest_room``."""
    if int(session.state.room_id) == int(dest_room):
        return
    shoot_i = 0
    for _ in range(int(budget)):
        st = session.state
        guard(session, label)
        if int(st.room_id) == int(dest_room):
            return
        if side_rooms and int(st.room_id) in side_rooms:
            if on_side_room is None:
                raise TimeoutError(
                    f"{label}: side room 0x{int(st.room_id):04X}: {st}"
                )
            on_side_room(session, label)
            continue
        if is_knockback(st):
            on_knockback(session, f"{label}_door_kb")
            continue
        if is_morph(int(st.pose)):
            unmorph(session)
            continue
        if int(st.samus_y) > int(lip_y):
            shoot_i = 0
            names = remount(st)
            reason = f"{label}_remount"
        else:
            names = door_action(st, shoot_i)
            shoot_i += 1
            reason = f"{label}_door"
        if names:
            hold(session, 1, *names, reason=reason)
        else:
            hold(session, 1, reason=f"{label}_hurt")
    if int(session.state.room_id) != int(dest_room):
        raise TimeoutError(f"{label}: ceiling door missed: {session.state}")


def settle_ceiling_dest(
    session: ControllerSession,
    dest_room: int,
    *,
    label: str,
    settle_frames: int = 200,
    land_frames: int = 90,
) -> SuperMetroidState:
    """Ordinary dest-room settle, then wait until a standing land."""
    wait_ordinary_room(
        session, dest_room, settle_frames=settle_frames, label=label
    )
    for _ in range(int(land_frames)):
        st = session.state
        if int(st.pose) in _LAND_POSES and abs(int(st.velocity_y)) <= 1:
            break
        hold(session, 1, reason=f"{label}_land")
    return session.state


__all__ = [
    "ceiling_door_action",
    "play_ceiling_door",
    "settle_ceiling_dest",
    "tap_up_action",
]

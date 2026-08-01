"""Pure-first controllers for the K4 Business-to-Bubble Norfair path.

Business Center → Frog Save is the accepted K4.0 continuous extension. The
remaining Frog Save → Bubble controllers are intentionally bounded scaffolds.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    hold,
    require_room,
    select_weapon,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.rooms import (
    ROOM_BUBBLE,
    ROOM_BUSINESS,
    ROOM_FROG_SAVE,
    ROOM_FROG_SPEEDWAY,
    ROOM_SPEED,
    ROOM_UPPER_NORFAIR_FARM,
)
from super_metroid.routes.runtime import ControllerSession


_MAX_SCAFFOLD_FRAMES = 240
_ELEVATOR_Y = 680
_FLOOR_Y_MIN = 1405
_STANDING_POSES = frozenset({1, 2, 9, 10, 25, 26, 27, 28, 37, 38, 137, 138})
_FROG_SPEEDWAY_DOOR_FRAMES = 400
_FROG_SPEEDWAY_SETTLE_FRAMES = 320


def _scaffold_exit(
    session: ControllerSession,
    *,
    entry_room: int,
    target_room: int,
    label: str,
) -> SuperMetroidState:
    """Run a bounded placeholder toward a door and report a useful failure."""
    require_room(session, entry_room, label)

    # TODO(SM-K4-BUBBLE-PURE): replace this placeholder with room geometry.
    for _ in range(_MAX_SCAFFOLD_FRAMES):
        state = hold(session, 1, "RIGHT", "B", reason=f"{label}_scaffold")
        if state.room_id == target_room:
            return state

    state = session.state
    raise TimeoutError(
        f"{label}: scaffold timeout before room 0x{target_room:04X}; "
        f"room=0x{state.room_id:04X} pose={state.pose} "
        f"xy=({state.samus_x},{state.samus_y})"
    )


def play_business_to_frog_save(session: ControllerSession) -> SuperMetroidState:
    """Business Center elevator → Frog Savestation through the blue door.

    The accepted Business checkpoint is still riding the arriving elevator.
    Let it settle at ``y=680``, snake down the central shaft to the floor, and
    beam-shot the right-hand Frog door at the floor lip.  Two integrity-green
    power-on runs compose this controller into the accepted Frog Save tip.
    """
    label = "business_to_frog_save"
    require_room(session, ROOM_BUSINESS, label)

    # The previous Warehouse elevator exit returns ordinary gameplay while
    # Samus is still descending (pose 155).  Require a stable landing rather
    # than treating the room-id change as an immediately playable source.
    stable_elevator_frames = 0
    for _ in range(600):
        state = hold(session, 1, reason=f"{label}_elevator_settle")
        if state.samus_y == _ELEVATOR_Y:
            stable_elevator_frames += 1
            if stable_elevator_frames >= 24:
                break
        else:
            stable_elevator_frames = 0
    else:
        raise TimeoutError(f"{label}: elevator did not settle: {session.state}")

    # Descend through the staggered Business platforms without trying to
    # recreate the upward Warehouse climb.  The 70f direction swaps avoid
    # pinning either wall and naturally land on the Frog-door floor band.
    for frame in range(650):
        state = session.state
        if (
            state.samus_y >= _FLOOR_Y_MIN
            and state.velocity_y == 0
            and state.pose in _STANDING_POSES
        ):
            break
        buttons = ("LEFT", "B") if (frame // 70) % 2 == 0 else ("RIGHT", "B")
        hold(session, 1, *buttons, reason=f"{label}_descend")
    else:
        raise TimeoutError(f"{label}: floor band missed: {session.state}")

    # It is a closed blue door in this source; select beam and shoot while
    # running right so the transition starts as soon as Samus reaches its lip.
    select_weapon(session, 0)
    for _ in range(400):
        state = hold(session, 1, "RIGHT", "B", "X", reason=f"{label}_door")
        if state.room_id == ROOM_FROG_SAVE:
            break
    else:
        raise TimeoutError(f"{label}: Frog door missed: {session.state}")

    return wait_ordinary_room(
        session,
        ROOM_FROG_SAVE,
        settle_frames=320,
        label=label,
    )


def play_frog_save_to_speedway(session: ControllerSession) -> SuperMetroidState:
    """Frog Savestation right door → ordinary Frog Speedway.

    The accepted Frog checkpoint settles on the left side of the short save
    room.  Its right door is blue, so run toward it while beam-shooting rather
    than relying on the generic scaffold's unarmed right hold.
    """
    label = "frog_save_to_speedway"
    require_room(session, ROOM_FROG_SAVE, label)

    select_weapon(session, 0)
    for _ in range(_FROG_SPEEDWAY_DOOR_FRAMES):
        state = hold(session, 1, "RIGHT", "B", "X", reason=f"{label}_door")
        if state.room_id == ROOM_FROG_SPEEDWAY:
            break
    else:
        state = session.state
        raise TimeoutError(
            f"{label}: right door missed before room "
            f"0x{ROOM_FROG_SPEEDWAY:04X}; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"door_transition={state.door_transition}"
        )

    return wait_ordinary_room(
        session,
        ROOM_FROG_SPEEDWAY,
        settle_frames=_FROG_SPEEDWAY_SETTLE_FRAMES,
        label=label,
    )


def play_speedway_to_farm(session: ControllerSession) -> SuperMetroidState:
    """Scaffold Frog Speedway ``0xB106`` → Upper Norfair Farm ``0xAF72``."""
    return _scaffold_exit(
        session,
        entry_room=ROOM_FROG_SPEEDWAY,
        target_room=ROOM_UPPER_NORFAIR_FARM,
        label="speedway_to_farm",
    )


def play_farm_to_bubble(session: ControllerSession) -> SuperMetroidState:
    """Scaffold Upper Norfair Farm ``0xAF72`` → Bubble Mountain ``0xACB3``."""
    return _scaffold_exit(
        session,
        entry_room=ROOM_UPPER_NORFAIR_FARM,
        target_room=ROOM_BUBBLE,
        label="farm_to_bubble",
    )


__all__ = [
    "ROOM_BUBBLE",
    "ROOM_BUSINESS",
    "ROOM_FROG_SAVE",
    "ROOM_FROG_SPEEDWAY",
    "ROOM_SPEED",
    "ROOM_UPPER_NORFAIR_FARM",
    "play_business_to_frog_save",
    "play_frog_save_to_speedway",
    "play_speedway_to_farm",
    "play_farm_to_bubble",
]

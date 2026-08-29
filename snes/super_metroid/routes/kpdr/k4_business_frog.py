"""K4 Business ↔ Frog Save, Frog Speedway, and post-Speed farm shortcuts.

Business Center → Frog Save is the accepted K4.0 continuous extension (save
milestone). Frog Speedway is a post-Speed shortcut only (Boost Blocks need
Speed Booster). Farm → Bubble is scaffold-only until a pure card exists.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    hold,
    require_room,
    select_weapon,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.k4_common import (
    _ELEVATOR_Y,
    _STANDING_POSES,
)
from super_metroid.routes.kpdr.rooms import (
    ROOM_BUBBLE,
    ROOM_BUSINESS,
    ROOM_FROG_SAVE,
    ROOM_FROG_SPEEDWAY,
    ROOM_UPPER_NORFAIR_FARM,
)
from super_metroid.routes.runtime import ControllerSession

_MAX_SCAFFOLD_FRAMES = 240
_FLOOR_Y_MIN = 1405
_FROG_SPEEDWAY_DOOR_FRAMES = 400
_FROG_SPEEDWAY_SETTLE_FRAMES = 320
# Frog Speedway is an 8-screen horizontal tunnel (left entry → right farm door).
# Continuous loadout has no Speed; mid-room Boost Blocks may stop progress.
_SPEEDWAY_TO_FARM_DOOR_FRAMES = 1100
_SPEEDWAY_TO_FARM_SETTLE_FRAMES = 320


def _scaffold_exit(
    session: ControllerSession,
    *,
    entry_room: int,
    target_room: int,
    label: str,
    face: str = "RIGHT",
) -> SuperMetroidState:
    """Run a bounded placeholder toward a door and report a useful failure."""
    require_room(session, entry_room, label)

    # TODO: replace placeholder with room geometry (one pure card per hop).
    for _ in range(_MAX_SCAFFOLD_FRAMES):
        state = hold(session, 1, face, "B", reason=f"{label}_scaffold")
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
    room.  The central save-tube blocks a flat run, so clear its two sides with
    separated Hi-Jump pulses before continuing to the blue right door.
    """
    label = "frog_save_to_speedway"
    require_room(session, ROOM_FROG_SAVE, label)

    select_weapon(session, 0)
    for frame in range(_FROG_SPEEDWAY_DOOR_FRAMES):
        inputs = ("RIGHT", "B", "X")
        if 30 <= frame < 40 or 90 <= frame < 100:
            inputs += ("A",)
        state = hold(session, 1, *inputs, reason=f"{label}_door")
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
    """Frog Speedway left entry → ordinary Upper Norfair Farm via right door.

    Continuous-like source spawns on the left of the long horizontal tunnel
    (x≈39).  Run and beam-shot the blue right door into ``0xAF72``.  Mid-room
    Boost Blocks normally need Speed Booster; this controller is pure-first
    **without** a Speed grant — if blocked, timeout reports pose/xy for the
    residual.
    """
    label = "speedway_to_farm"
    require_room(session, ROOM_FROG_SPEEDWAY, label)

    select_weapon(session, 0)
    max_x = session.state.samus_x
    for _ in range(_SPEEDWAY_TO_FARM_DOOR_FRAMES):
        state = hold(session, 1, "RIGHT", "B", "X", reason=f"{label}_door")
        if state.samus_x > max_x:
            max_x = state.samus_x
        if state.room_id == ROOM_UPPER_NORFAIR_FARM:
            break
    else:
        state = session.state
        # Mid-room Boost Blocks (~x=795 from left entry) stop progress without
        # Speed Booster; report max_x so residuals can name the lock.
        raise TimeoutError(
            f"{label}: right door missed before room "
            f"0x{ROOM_UPPER_NORFAIR_FARM:04X}; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"door_transition={state.door_transition} max_x={max_x}"
            + (
                " (boost-block stall; no Speed)"
                if max_x <= 820 and state.samus_x <= 820
                else ""
            )
        )

    return wait_ordinary_room(
        session,
        ROOM_UPPER_NORFAIR_FARM,
        settle_frames=_SPEEDWAY_TO_FARM_SETTLE_FRAMES,
        label=label,
    )


def play_farm_to_bubble(session: ControllerSession) -> SuperMetroidState:
    """Scaffold Upper Norfair Farm ``0xAF72`` → Bubble Mountain ``0xACB3``.

    Post-Speed farm entry only (see ``speedway_to_farm`` requires Speed).
    """
    return _scaffold_exit(
        session,
        entry_room=ROOM_UPPER_NORFAIR_FARM,
        target_room=ROOM_BUBBLE,
        label="farm_to_bubble",
    )


__all__ = [
    "play_business_to_frog_save",
    "play_farm_to_bubble",
    "play_frog_save_to_speedway",
    "play_speedway_to_farm",
]

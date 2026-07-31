"""K4 reverse hops from Kraid's Eye Room back to Warehouse.

These bounded controllers are ``controller_dev`` scaffolds only. They use
ordinary inputs and do not own the emulator, door-warp setup, or progression
state; natural-entry evidence is still required before continuous use.
"""

from __future__ import annotations

from super_metroid.policy import StateRequirement
from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    hold,
    require_room,
    require_state,
    select_weapon,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.rooms import (
    ROOM_BABY_KRAID,
    ROOM_KRAID_EYE,
    ROOM_WAREHOUSE,
    ROOM_WAREHOUSE_KIHUNTER,
    ROOM_ZEELA,
)
from super_metroid.routes.runtime import ControllerSession


def _play_return(
    session: ControllerSession,
    *,
    source_room: int,
    target_room: int,
    direction: str,
    label: str,
    spin: bool = False,
) -> SuperMetroidState:
    require_state(
        session,
        StateRequirement(room_id=source_room, game_states=frozenset({8})),
        label,
    )
    require_room(session, source_room, label)
    select_weapon(session, 0)

    buttons = (direction, "B", "A") if spin else (direction, "A")
    for _ in range(900):
        state = hold(session, 1, *buttons, reason=f"{label}_exit")
        if state.room_id == target_room:
            break
    else:
        raise TimeoutError(f"{label}: exit timed out: {session.state}")

    return wait_ordinary_room(
        session,
        target_room,
        settle_frames=320,
        label=label,
    )


def play_eye_to_baby_return(session: ControllerSession) -> SuperMetroidState:
    """Open the Baby Kraid door, then push left from Kraid Eye toward Baby Kraid.

    This is not continuous evidence; it requires a natural source state.
    """
    # Unlike the naive spin-push scaffold, explicitly open the return door
    # before crossing it, mirroring the forward Eye-room beam-shot pattern.
    require_state(
        session,
        StateRequirement(room_id=ROOM_KRAID_EYE, game_states=frozenset({8})),
        "eye_to_baby_return",
    )
    require_room(session, ROOM_KRAID_EYE, "eye_to_baby_return")
    select_weapon(session, 0)
    hold(session, 8, "LEFT", reason="eye_to_baby_face_left")
    hold(session, 6, reason="eye_to_baby_face_release")
    for _ in range(4):
        hold(session, 4, "LEFT", "X", reason="eye_to_baby_door_shot")
        hold(session, 18, reason="eye_to_baby_door_fuse")
    return _play_return(
        session,
        source_room=ROOM_KRAID_EYE,
        target_room=ROOM_BABY_KRAID,
        direction="LEFT",
        label="eye_to_baby_return",
        spin=True,
    )


def play_baby_to_kihunter_return(session: ControllerSession) -> SuperMetroidState:
    """Controller-dev scaffold: push left from Baby Kraid toward Kihunter.

    This is not continuous evidence; it requires a natural source state.
    """
    return _play_return(
        session,
        source_room=ROOM_BABY_KRAID,
        target_room=ROOM_WAREHOUSE_KIHUNTER,
        direction="LEFT",
        label="baby_to_kihunter_return",
        spin=True,
    )


def play_kihunter_to_zeela_return(session: ControllerSession) -> SuperMetroidState:
    """Controller-dev scaffold: push down from Kihunter toward Zeela.

    This is not continuous evidence; it requires a natural source state.
    """
    return _play_return(
        session,
        source_room=ROOM_WAREHOUSE_KIHUNTER,
        target_room=ROOM_ZEELA,
        direction="DOWN",
        label="kihunter_to_zeela_return",
    )


def play_zeela_to_warehouse_return(session: ControllerSession) -> SuperMetroidState:
    """Controller-dev scaffold: push left from Zeela toward Warehouse.

    This is not continuous evidence; it requires a natural source state.
    """
    return _play_return(
        session,
        source_room=ROOM_ZEELA,
        target_room=ROOM_WAREHOUSE,
        direction="LEFT",
        label="zeela_to_warehouse_return",
        spin=True,
    )

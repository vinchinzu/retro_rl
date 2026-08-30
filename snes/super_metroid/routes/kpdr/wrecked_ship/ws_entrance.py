"""Wrecked Ship Entrance → Main Shaft (rr-ahjo).

Unpowered 4-screen hallway. Walk/run right. Coverns only. Blue door into
Main Shaft. Do not invent a fight. Energy assist is on — tank Coverns if
they touch. Shoot the blue door (beam, not Super).
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    hold_until,
    play_run_shoot_exit,
    require_room,
    select_weapon,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.room_ids import ROOM_WS_ENTRANCE, ROOM_WS_MAIN
from super_metroid.routes.runtime import ControllerSession

# Beam, not Super — pin after the green Super still has selected_item=2.
WEAPON_BEAM = 0
# Dump from post_ws_poweron: first x>=960 at ~(968,139) p9 speed=4; closed
# blue-door crash wall is x=987 p137. Start beam pressure before that wall.
WS_ENTRANCE_DOOR_X_MIN = 900
WS_ENTRANCE_DOOR_X_MAX = 1024
_WS_ENTRANCE_RUN_TIMEOUT = 400
_WS_ENTRANCE_SETTLE = 200


def at_ws_entrance_door_seat(state: SuperMetroidState) -> bool:
    """True on the right-door approach band of unpowered Entrance ``0xCA08``."""
    x = int(state.samus_x)
    return (
        int(state.room_id) == ROOM_WS_ENTRANCE
        and WS_ENTRANCE_DOOR_X_MIN <= x <= WS_ENTRANCE_DOOR_X_MAX
    )


def ws_entrance_to_main_action(state: SuperMetroidState) -> tuple[str, ...]:
    """One-frame buttons. Cycle to beam before any X; never Super the blue door."""
    room = int(state.room_id)
    if room == ROOM_WS_MAIN:
        return ()
    if int(state.selected_item) != WEAPON_BEAM:
        return ("SELECT",)
    if room != ROOM_WS_ENTRANCE:
        return ()
    if int(state.samus_x) < WS_ENTRANCE_DOOR_X_MIN:
        return ("RIGHT", "B")
    return ("RIGHT", "B", "X")


def ws_entrance_main_settled(state: SuperMetroidState) -> bool:
    """Ordinary Main Shaft handoff: room ``0xCAF6`` gs=8 door_transition=0."""
    return (
        int(state.room_id) == ROOM_WS_MAIN
        and int(state.game_state) == 8
        and int(state.door_transition) == 0
    )


def play_ws_entrance_to_main(session: ControllerSession) -> SuperMetroidState:
    """Unpowered 4-screen hallway. Walk/run right. Coverns only. Blue door into Main Shaft.

    https://wiki.supermetroid.run/Wrecked_Ship_Entrance

    Do not invent a fight. Energy assist is on — tank Coverns if they touch.
    Shoot the blue door (beam, not Super). Lands ordinary ``gs=8`` in Main
    Shaft ``0xCAF6`` (game state 11 can last 50–100+f).

    Source: ``scratch/post_ws_poweron.state`` ``0xCA08`` ~(57,139) p1 gs=8.
    """
    label = "ws_entrance_to_main"
    require_room(session, ROOM_WS_ENTRANCE, label)
    select_weapon(session, WEAPON_BEAM)

    def _at_seat_or_main(state: SuperMetroidState) -> bool:
        return int(state.room_id) == ROOM_WS_MAIN or at_ws_entrance_door_seat(state)

    hold_until(
        session,
        _at_seat_or_main,
        "RIGHT",
        "B",
        timeout=_WS_ENTRANCE_RUN_TIMEOUT,
        reason=f"{label}_run",
    )
    if int(session.state.room_id) == ROOM_WS_MAIN:
        return wait_ordinary_room(
            session, ROOM_WS_MAIN, settle_frames=_WS_ENTRANCE_SETTLE, label=label
        )
    return play_run_shoot_exit(
        session,
        from_room=ROOM_WS_ENTRANCE,
        to_room=ROOM_WS_MAIN,
        direction="RIGHT",
        label=label,
        run_frames=0,
        shoot_frames=10,
        spin_frames=0,
        hold_frames=200,
        settle_frames=_WS_ENTRANCE_SETTLE,
        super_door=False,
    )


__all__ = [
    "ROOM_WS_ENTRANCE",
    "ROOM_WS_MAIN",
    "WEAPON_BEAM",
    "WS_ENTRANCE_DOOR_X_MIN",
    "WS_ENTRANCE_DOOR_X_MAX",
    "at_ws_entrance_door_seat",
    "play_ws_entrance_to_main",
    "ws_entrance_main_settled",
    "ws_entrance_to_main_action",
]

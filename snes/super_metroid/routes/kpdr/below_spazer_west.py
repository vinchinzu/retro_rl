"""Below Spazer floor/water → West Tunnel (RIGHT runner only).

Shared by red-stack spine hops and Spazer detour exit so neither package
imports the other for this hop.
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
from super_metroid.routes.kpdr.rooms import ROOM_BELOW_SPAZER, ROOM_WEST_TUNNEL
from super_metroid.routes.runtime import ControllerSession


def play_below_spazer_floor_to_west(
    session: ControllerSession,
) -> SuperMetroidState:
    """Floor/water band Below Spazer → West Tunnel (RIGHT runner only).

    Used after Spazer is held and Samus is on mid/floor (not from natural
    floor entry without Spazer — that path is the mainline detour).
    """
    require_room(session, ROOM_BELOW_SPAZER, "below_spazer_floor_to_west")
    # Let the door-exit running pose settle before `unmorph`; pose 9/10 is
    # intentionally handled by that shared helper and would otherwise turn
    # this ordinary entry glide into an unwanted jump.
    hold(session, 6, reason="below_spazer_entry_glide")
    unmorph(session)
    select_weapon(session, 0)
    for frame in range(2000):
        buttons = ("RIGHT", "B", "X") if frame % 35 < 10 else ("RIGHT", "B", "A")
        state = hold(session, 1, *buttons, reason="below_spazer_right")
        if state.room_id == ROOM_WEST_TUNNEL:
            break
    else:
        raise TimeoutError(
            f"below_spazer_floor_to_west: West Tunnel not reached: {state}"
        )
    return wait_ordinary_room(
        session,
        ROOM_WEST_TUNNEL,
        settle_frames=260,
        label="below_spazer_floor_to_west",
    )


__all__ = ["play_below_spazer_floor_to_west"]

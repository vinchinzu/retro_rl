"""Development-only Wrecked Ship approach scaffold.

K6 currently has only dev-warp entry evidence.  These bounded placeholders
define the room-by-room controller surface for the future pure geometry pass;
they do not own emulator state, warp Samus, or write progression/boss RAM.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import hold, require_room
from super_metroid.routes.runtime import ControllerSession


ROOM_MOAT = 0x95FF
ROOM_WEST_OCEAN = 0x93FE
ROOM_WS_ENTRANCE = 0xCA08
ROOM_WS_MAIN = 0xCAF6
ROOM_WS_BASEMENT = 0xCC6F
ROOM_PHANTOON = 0xCD13

_MAX_SCAFFOLD_FRAMES = 240


def _scaffold_exit(
    session: ControllerSession,
    *,
    entry_room: int,
    target_room: int,
    label: str,
) -> SuperMetroidState:
    """Run a bounded placeholder toward the next ship-room door."""
    require_room(session, entry_room, label)

    # TODO(SM-WS-PURE): replace with source-state-driven room geometry.
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


def play_moat_to_west_ocean(session: ControllerSession) -> SuperMetroidState:
    """Scaffold Moat ``0x95FF`` -> West Ocean ``0x93FE``."""
    return _scaffold_exit(
        session,
        entry_room=ROOM_MOAT,
        target_room=ROOM_WEST_OCEAN,
        label="moat_to_west_ocean",
    )


def play_west_ocean_to_ws(session: ControllerSession) -> SuperMetroidState:
    """Scaffold West Ocean ``0x93FE`` -> WS entrance ``0xCA08``."""
    return _scaffold_exit(
        session,
        entry_room=ROOM_WEST_OCEAN,
        target_room=ROOM_WS_ENTRANCE,
        label="west_ocean_to_ws",
    )


def play_ws_entrance_to_main(session: ControllerSession) -> SuperMetroidState:
    """Scaffold WS entrance ``0xCA08`` -> WS main/attic ``0xCAF6``."""
    return _scaffold_exit(
        session,
        entry_room=ROOM_WS_ENTRANCE,
        target_room=ROOM_WS_MAIN,
        label="ws_entrance_to_main",
    )


def play_ws_main_to_basement(session: ControllerSession) -> SuperMetroidState:
    """Scaffold WS main/attic ``0xCAF6`` -> basement ``0xCC6F``."""
    return _scaffold_exit(
        session,
        entry_room=ROOM_WS_MAIN,
        target_room=ROOM_WS_BASEMENT,
        label="ws_main_to_basement",
    )


def play_ws_basement_to_phantoon(session: ControllerSession) -> SuperMetroidState:
    """Scaffold WS basement ``0xCC6F`` -> Phantoon ``0xCD13``."""
    return _scaffold_exit(
        session,
        entry_room=ROOM_WS_BASEMENT,
        target_room=ROOM_PHANTOON,
        label="ws_basement_to_phantoon",
    )


__all__ = [
    "ROOM_MOAT",
    "ROOM_WEST_OCEAN",
    "ROOM_WS_ENTRANCE",
    "ROOM_WS_MAIN",
    "ROOM_WS_BASEMENT",
    "ROOM_PHANTOON",
    "play_moat_to_west_ocean",
    "play_west_ocean_to_ws",
    "play_ws_entrance_to_main",
    "play_ws_main_to_basement",
    "play_ws_basement_to_phantoon",
]

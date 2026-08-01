"""Scaffold for the K4 Business-to-Bubble Norfair path.

This module is a controller home for the reverse K4 spine.  It is not
continuous evidence and its movement geometry is intentionally incomplete;
pure-green verification is deferred until a continuous-like Business source
state is available.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import hold, require_room
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
    """Scaffold Business Center ``0xA7DE`` → Frog Save ``0xB167``."""
    return _scaffold_exit(
        session,
        entry_room=ROOM_BUSINESS,
        target_room=ROOM_FROG_SAVE,
        label="business_to_frog_save",
    )


def play_frog_save_to_speedway(session: ControllerSession) -> SuperMetroidState:
    """Scaffold Frog Save ``0xB167`` → Frog Speedway ``0xB106``."""
    return _scaffold_exit(
        session,
        entry_room=ROOM_FROG_SAVE,
        target_room=ROOM_FROG_SPEEDWAY,
        label="frog_save_to_speedway",
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

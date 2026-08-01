"""Development-only Tourian escape chain scaffold.

This module deliberately contains no emulator ownership, movement policy, or
progression writes.  It records the room order and timeout budget that a later
natural-entry escape controller must implement.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


ROOM_ESCAPE_1 = 0xDE4D
ROOM_ESCAPE_2 = 0xDE7A
ROOM_ESCAPE_3 = 0xDEA7
ROOM_ESCAPE_4 = 0xDEDE
ROOM_LANDING_SITE = 0x91F8

ESCAPE_ROOM_IDS = (
    ROOM_ESCAPE_1,
    ROOM_ESCAPE_2,
    ROOM_ESCAPE_3,
    ROOM_ESCAPE_4,
)
ESCAPE_CHAIN_ROOM_IDS = ESCAPE_ROOM_IDS + (ROOM_LANDING_SITE,)

# These are planning bounds only. They do not drive emulator frames yet.
DEFAULT_ESCAPE_TIMEOUTS: dict[int, int] = {
    ROOM_ESCAPE_1: 1_800,
    ROOM_ESCAPE_2: 1_800,
    ROOM_ESCAPE_3: 1_800,
    ROOM_ESCAPE_4: 1_800,
    ROOM_LANDING_SITE: 2_400,
}


def _room_evidence(room_id: int, timeout_frames: int) -> dict[str, object]:
    """Describe one deferred room controller without touching a session."""
    return {
        "roomId": room_id,
        "roomIdHex": f"0x{room_id:04X}",
        "timeoutFrames": timeout_frames,
        "status": "stub",
        "success": False,
        "implemented": False,
    }


def play_escape_chain_scaffold(
    session: Any,
    *,
    room_timeouts: Mapping[int, int] | None = None,
) -> dict[str, object]:
    """Return bounded escape-chain evidence for future controller work.

    ``session`` is accepted to match the eventual segment surface, but is only
    inspected for an optional current state. This stub never steps, warps,
    places Samus, or writes RAM. The returned evidence is not a clear.
    """
    timeouts = dict(DEFAULT_ESCAPE_TIMEOUTS)
    if room_timeouts is not None:
        timeouts.update(room_timeouts)
    invalid = {
        room_id: timeout
        for room_id, timeout in timeouts.items()
        if room_id in ESCAPE_CHAIN_ROOM_IDS and timeout <= 0
    }
    if invalid:
        raise ValueError(f"escape room timeouts must be positive: {invalid}")

    state = getattr(session, "state", None)
    current_room = getattr(state, "room_id", None)
    return {
        "developmentOnly": True,
        "outcome": "scaffold_only",
        "success": False,
        "currentRoomId": current_room,
        "rooms": [
            _room_evidence(room_id, timeouts[room_id])
            for room_id in ESCAPE_CHAIN_ROOM_IDS
        ],
    }


def play_escape_room_1(session: Any, *, timeout_frames: int = 1_800) -> dict[str, object]:
    """Deferred bounded stub for Tourian escape room 1."""
    return _play_escape_room_stub(session, ROOM_ESCAPE_1, timeout_frames)


def play_escape_room_2(session: Any, *, timeout_frames: int = 1_800) -> dict[str, object]:
    """Deferred bounded stub for Tourian escape room 2."""
    return _play_escape_room_stub(session, ROOM_ESCAPE_2, timeout_frames)


def play_escape_room_3(session: Any, *, timeout_frames: int = 1_800) -> dict[str, object]:
    """Deferred bounded stub for Tourian escape room 3."""
    return _play_escape_room_stub(session, ROOM_ESCAPE_3, timeout_frames)


def play_escape_room_4(session: Any, *, timeout_frames: int = 1_800) -> dict[str, object]:
    """Deferred bounded stub for Tourian escape room 4."""
    return _play_escape_room_stub(session, ROOM_ESCAPE_4, timeout_frames)


def _play_escape_room_stub(
    session: Any, room_id: int, timeout_frames: int
) -> dict[str, object]:
    if timeout_frames <= 0:
        raise ValueError("escape room timeout must be positive")
    state = getattr(session, "state", None)
    return {
        **_room_evidence(room_id, timeout_frames),
        "currentRoomId": getattr(state, "room_id", None),
    }

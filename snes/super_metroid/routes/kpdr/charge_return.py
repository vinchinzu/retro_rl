"""Scaffold for the optional Big Pink Charge Beam side trip.

Charge Beam is collected in Big Pink itself, at the Chozo below the main
mass.  The active K1 route takes the direct GHZ exit instead, so the
conventional return remains an optional pure-geometry task.  These helpers
define the room contract without attempting unverified movement.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import require_room
from super_metroid.routes.runtime import ControllerSession


# The Charge Chozo is a sub-area of Big Pink, not a separate room.
ROOM_CHARGE = 0x9D19
ROOM_BIG_PINK = ROOM_CHARGE


def _charge_geometry_pending(session: ControllerSession, label: str) -> SuperMetroidState:
    """Validate the expected room and fail until geometry is sourced."""
    require_room(session, ROOM_CHARGE, label)
    raise NotImplementedError(
        f"{label}: Charge Beam geometry is pending a natural source-state capture"
    )


def play_charge_beam_collect(session: ControllerSession) -> SuperMetroidState:
    """Scaffold the Big Pink -> Charge Chozo collect hop.

    The helper intentionally performs no movement or item/progression writes.
    """
    return _charge_geometry_pending(session, "charge_beam_collect")


def play_charge_beam_return(session: ControllerSession) -> SuperMetroidState:
    """Scaffold the Charge Chozo -> Big Pink main-shaft return hop."""
    return _charge_geometry_pending(session, "charge_beam_return")


__all__ = [
    "ROOM_BIG_PINK",
    "ROOM_CHARGE",
    "play_charge_beam_collect",
    "play_charge_beam_return",
]

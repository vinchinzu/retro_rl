"""Scaffold for the early Spazer Beam detour (Below Spazer -> collect -> return).

Spazer Room ``0xA447`` is KPDR K2.2 optional.  Continuous already reaches
Below Spazer ``0xA408`` (``--to below_spazer``); the current path skips the
item room altogether.  These helpers define the room contract without
attempting unverified movement.

Walljump residual risk: the red-room approach path (tall shafts and ledges
before the Spazer pedestal) may require walljump-capable movement similar to
the Bubble Mountain mid patterns (pose-26 / fresh-A).  Do not force-green
until a natural continuous-like source is captured and pure geometry is
proven.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import require_room
from super_metroid.routes.runtime import ControllerSession

from super_metroid.routes.kpdr.rooms import ROOM_BELOW_SPAZER  # noqa: F811
from super_metroid.routes.kpdr.rooms import ROOM_SPAZER

def _spazer_geometry_pending(
    session: ControllerSession, label: str
) -> SuperMetroidState:
    """Validate the expected room and fail until geometry is sourced."""
    require_room(session, ROOM_SPAZER, label)
    raise NotImplementedError(
        f"{label}: Spazer geometry is pending a natural source-state capture"
    )

def play_below_spazer_to_spazer(
    session: ControllerSession,
) -> SuperMetroidState:
    """Scaffold the Below Spazer ``0xA408`` -> Spazer Room ``0xA447`` entry hop.

    Requires: Below Spazer ordinary room state on entry.
    Expected exit: Spazer Room with Samus just inside the left door.

    Residual risk: the red-room approach may require walljump movement.
    No movement or item/progression writes are attempted here.
    """
    require_room(session, ROOM_BELOW_SPAZER, "below_spazer_to_spazer")
    raise NotImplementedError(
        "below_spazer_to_spazer: geometry pending source-state capture (SM-SPAZER-SRC)"
    )

def play_spazer_collect(session: ControllerSession) -> SuperMetroidState:
    """Scaffold the Spazer Room pedestal collect hop.

    Requires: Spazer Room ``0xA447`` ordinary room state on entry.
    Goal: approach the Chozo pedestal and collect the Spazer Beam item.

    Residual risk: bounded pedestal approach; walljump geometry may be
    needed for the red-room climb.  No movement or item writes here.
    """
    return _spazer_geometry_pending(session, "spazer_collect")

def play_spazer_return_to_below(
    session: ControllerSession,
) -> SuperMetroidState:
    """Scaffold the Spazer Room ``0xA447`` -> Below Spazer ``0xA408`` return hop.

    Requires: Spazer Room with Spazer beam item bit set (player holds it).
    Expected exit: Below Spazer ordinary room, ready to continue West.

    Residual risk: the return descent through the red room is the walljump-
    critical segment.  No movement or item writes here.
    """
    return _spazer_geometry_pending(session, "spazer_return_to_below")

__all__ = [
    "ROOM_SPAZER",
    "play_below_spazer_to_spazer",
    "play_spazer_collect",
    "play_spazer_return_to_below",
]

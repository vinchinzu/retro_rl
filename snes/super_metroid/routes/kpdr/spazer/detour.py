"""Mainline K2.2 fuse: climb → Super door → collect → return → West."""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import require_room
from super_metroid.routes.kpdr.rooms import ROOM_BELOW_SPAZER
from super_metroid.routes.kpdr.spazer.approach import play_below_spazer_to_spazer
from super_metroid.routes.kpdr.spazer.collect import (
    play_spazer_collect,
    play_spazer_return_to_below,
)
from super_metroid.routes.kpdr.spazer.drop import play_spazer_top_to_west
from super_metroid.routes.kpdr.spazer.geometry import has_spazer
from super_metroid.routes.runtime import ControllerSession


def play_spazer_detour(session: ControllerSession) -> SuperMetroidState:
    """Mainline K2.2: climb → Super door → collect → return → West.

    Called by :func:`~super_metroid.routes.kpdr.red_stack.play_below_spazer_to_west`
    on every continuous Below→West hop when Spazer is missing (always — no
    floor skip). Also pure-probeable as ``spazer-detour``.

    Ends West Tunnel ordinary with beams ``0x1004`` when all sub-hops clear.
    If Spazer already held: :func:`play_spazer_top_to_west` only.
    """
    require_room(session, ROOM_BELOW_SPAZER, "spazer_detour")
    if has_spazer(session.state):
        return play_spazer_top_to_west(session)

    play_below_spazer_to_spazer(session)
    play_spazer_collect(session)
    play_spazer_return_to_below(session)
    return play_spazer_top_to_west(session)


__all__ = ["play_spazer_detour"]

"""Glass Tunnel → West Tunnel pure return (K5 hop 7).

Source: ``post_ice_east_to_glass_pure`` ~(216, 395) pose 12 mid-floor residual
after East→Glass dual **253f**. Reverse of ``play_west_to_glass``
(West RIGHT-run into Glass).

Hybrid pure::

  1. Accept Glass mid-bottom standing residual (x∈[150,280], y∈[350,420] p12)
  2. Beam select + LEFT-run/shoot into West Tunnel blue door
  3. Ordinary West settle (room-id primary)

Tape: ``tasks/speed_to_wave_ice_moat_human.json`` Phase C hop 25→26
(f21438 Glass ~(17,395) → f21648 West ~(16,395)). Pure pin is mid-floor
after reverse entry from East (~x=216); same LEFT door.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import play_run_shoot_exit
from super_metroid.routes.kpdr.red_tower.geometry import (
    GLASS_TO_WEST_HOLD,
    GLASS_TO_WEST_RUN,
    GLASS_TO_WEST_SETTLE,
    GLASS_TO_WEST_SHOOT,
    GLASS_TO_WEST_SPIN,
)
from super_metroid.routes.kpdr.rooms import ROOM_GLASS, ROOM_WEST_TUNNEL
from super_metroid.routes.runtime import ControllerSession


def play_glass_to_west(session: ControllerSession) -> SuperMetroidState:
    """Glass left blue door → ordinary West Tunnel (reverse of west→glass).

    Expects mid-bottom Glass handoff after east_to_glass. LEFT-run/shoot into
    ``0xCF54`` — mirror of outbound ``play_west_to_glass`` (RIGHT).
    """
    return play_run_shoot_exit(
        session,
        from_room=ROOM_GLASS,
        to_room=ROOM_WEST_TUNNEL,
        direction="LEFT",
        label="glass_to_west",
        run_frames=GLASS_TO_WEST_RUN,
        shoot_frames=GLASS_TO_WEST_SHOOT,
        spin_frames=GLASS_TO_WEST_SPIN,
        hold_frames=GLASS_TO_WEST_HOLD,
        settle_frames=GLASS_TO_WEST_SETTLE,
    )


__all__ = ["play_glass_to_west"]

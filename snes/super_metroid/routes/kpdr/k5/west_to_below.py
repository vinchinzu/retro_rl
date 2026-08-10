"""West Tunnel → Below Spazer pure return (K5 hop 8).

Source: ``post_ice_glass_to_west_pure`` ~(216, 139) pose 10 mid-floor residual
after Glass→West dual **211f**. Reverse of ``play_below_spazer_floor_to_west``
(Below RIGHT-run into West) — floor path only; Spazer already held on K5 stack
(beams ``0x1007``).

Hybrid pure::

  1. Accept West mid-floor standing residual (x∈[150,280], y∈[100,180] p10)
  2. Beam select + LEFT-run/shoot into Below Spazer blue door
  3. Ordinary Below Spazer settle (room-id primary)

Tape: ``tasks/speed_to_wave_ice_moat_human.json`` Phase C hop 26→27
(f21648 West ~(16,395) → f21858 Below ~(18,139)). Pure pin is mid-right
after reverse entry from Glass (~x=216 y=139); same LEFT door band.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import play_run_shoot_exit
from super_metroid.routes.kpdr.k5.geometry import (
    WEST_TO_BELOW_HOLD,
    WEST_TO_BELOW_RUN,
    WEST_TO_BELOW_SETTLE,
    WEST_TO_BELOW_SHOOT,
    WEST_TO_BELOW_SPIN,
)
from super_metroid.routes.kpdr.rooms import ROOM_BELOW_SPAZER, ROOM_WEST_TUNNEL
from super_metroid.routes.runtime import ControllerSession


def play_west_to_below(session: ControllerSession) -> SuperMetroidState:
    """West left blue door → ordinary Below Spazer (reverse of below→west floor).

    Expects mid-right West handoff after glass_to_west. LEFT-run/shoot into
    ``0xA408`` — mirror of outbound ``play_below_spazer_floor_to_west`` (RIGHT).
    """
    return play_run_shoot_exit(
        session,
        from_room=ROOM_WEST_TUNNEL,
        to_room=ROOM_BELOW_SPAZER,
        direction="LEFT",
        label="west_to_below",
        run_frames=WEST_TO_BELOW_RUN,
        shoot_frames=WEST_TO_BELOW_SHOOT,
        spin_frames=WEST_TO_BELOW_SPIN,
        hold_frames=WEST_TO_BELOW_HOLD,
        settle_frames=WEST_TO_BELOW_SETTLE,
    )


__all__ = ["play_west_to_below"]

"""Red Tower Ice checkpoint chain: bottom floor → ordinary Hellway left-door.

Ice-pin spine hop for ``play_red_to_hellway``. Do not extra-settle: 5f idle
drops the airborne ``(39,139)`` p11 seat into a Samus Eater. Mid→thin is
still the period-WJ body.

Public policy: Hi-Jump + Ice. Freeze Rippers, hop the platforms, keep RIGHT
until ordinary Hellway left-door (gs=8, x≤80). The first Hellway ``room_id``
is still the Red Tower door slot.
https://wiki.supermetroid.run/Red_Tower
https://wiki.supermetroid.run/Hellway
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.kpdr.red_tower.red_ice_climb import (
    HELLWAY_SILL,
    can_attach_bottom_edge,
    play_bottom_to_ripper1,
)
from super_metroid.routes.kpdr.red_tower.red_ice_mid_to_thin import play_mid_floor_to_thin_seat
from super_metroid.routes.kpdr.red_tower.red_ice_r1_to_r2 import play_ripper1_to_ripper2
from super_metroid.routes.kpdr.red_tower.red_ice_r2_to_r3 import play_ripper2_to_ripper3
from super_metroid.routes.kpdr.red_tower.red_ice_r3_to_r4 import play_ripper3_to_ripper4
from super_metroid.routes.kpdr.red_tower.red_ice_r4_to_tunnel import play_ripper4_to_tunnel
from super_metroid.routes.kpdr.red_tower.red_ice_thin_to_ur1 import (
    play_thin_seat_to_upper_ripper1,
)
from super_metroid.routes.kpdr.red_tower.red_ice_tunnel_to_mid import play_tunnel_to_mid_floor
from super_metroid.routes.kpdr.red_tower.red_ice_upper_hops import (
    play_upper_ripper1_to_2,
    play_upper_ripper2_to_3,
)
from super_metroid.routes.kpdr.red_tower.red_ice_ur3_to_hellway import (
    play_upper_ripper3_to_hellway,
)
from super_metroid.routes.runtime import ControllerSession

POLICY_ID = "red_tower_ice_bottom_to_hellway"


def play_ice_climb_to_hellway(session: ControllerSession) -> SuperMetroidState:
    """Bottom floor → ordinary Hellway left-door. No door-slot fire, no settle."""
    if not can_attach_bottom_edge(session.state):
        raise TimeoutError(
            f"{POLICY_ID}: not on Ice+HJ bottom floor "
            f"xy=({session.state.samus_x},{session.state.samus_y}) "
            f"p={session.state.pose}"
        )
    play_bottom_to_ripper1(session)
    play_ripper1_to_ripper2(session)
    play_ripper2_to_ripper3(session)
    play_ripper3_to_ripper4(session)
    play_ripper4_to_tunnel(session)
    play_tunnel_to_mid_floor(session)
    play_mid_floor_to_thin_seat(session)
    play_thin_seat_to_upper_ripper1(session)
    play_upper_ripper1_to_2(session)
    play_upper_ripper2_to_3(session)
    play_upper_ripper3_to_hellway(session)
    state = session.state
    if not HELLWAY_SILL.matches(state):
        raise TimeoutError(
            f"{POLICY_ID}: not ordinary Hellway left-door "
            f"room=0x{int(state.room_id):04X} "
            f"xy=({state.samus_x},{state.samus_y}) p={state.pose}"
        )
    return state


__all__ = ["POLICY_ID", "play_ice_climb_to_hellway"]

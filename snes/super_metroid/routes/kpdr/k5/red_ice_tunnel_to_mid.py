"""Red Tower Ice edge: solid tunnel alcove → temporary mid floor.

The first arc lands on the bomb ledge at y=1755.  The old monolithic climb
mistook its y=1719 apex for a landing; both vertical direction and the known
ledge band are required here.  A centered double-bomb climb then drifts left
at the y=1591 apex and catches the temporary floor at y=1625.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import ensure_morph, hold, is_morph, settle_hold
from super_metroid.routes.kpdr.k5.red_ice_climb import (
    MID_FLOOR,
    TUNNEL_FLOOR,
    VARIANT_ID,
    can_attach_tunnel_edge,
)
from super_metroid.routes.kpdr.k5.red_to_hellway import (
    _MORPH,
    _ibj_double,
    _tunnel_to_midplat,
)
from super_metroid.routes.kpdr.rooms import ROOM_RED_TOWER
from super_metroid.routes.runtime import ControllerSession

POLICY_ID = "red_tower_ice_tunnel_to_mid_floor"


def _grounded(state: SuperMetroidState) -> bool:
    return int(state.velocity_y) == 0 and int(state.vertical_direction) == 0


def play_tunnel_to_mid_floor(
    session: ControllerSession,
    *,
    max_cycles: int = 50,
) -> SuperMetroidState:
    """Climb from the natural tunnel checkpoint to the grounded mid floor."""
    if not can_attach_tunnel_edge(session.state):
        raise TimeoutError(
            f"{POLICY_ID}: not on tunnel_floor "
            f"xy=({session.state.samus_x},{session.state.samus_y}) "
            f"p={session.state.pose}"
        )

    _tunnel_to_midplat(session, f"{POLICY_ID}_ledge")
    for _ in range(50):
        state = session.state
        if (
            _grounded(state)
            and 1740 <= int(state.samus_y) <= 1770
            and 115 <= int(state.samus_x) <= 180
        ):
            break
        hold(session, 1, reason=f"{POLICY_ID}_ledge_land")
    else:
        raise TimeoutError(
            f"{POLICY_ID}: missed bomb ledge "
            f"xy=({session.state.samus_x},{session.state.samus_y})"
        )

    # Drop aim-up, walk to the right edge of the ledge, then morph for IBJ.
    hold(session, 1, "UP", reason=f"{POLICY_ID}_stand")
    for _ in range(12):
        hold(session, 1, reason=f"{POLICY_ID}_drop_aim")
    for _ in range(50):
        if int(session.state.samus_x) >= 168:
            break
        hold(session, 1, "RIGHT", reason=f"{POLICY_ID}_center")
    settle_hold(session, 4, reason=f"{POLICY_ID}_center_settle")
    if not is_morph(session.state.pose) and int(session.state.pose) not in _MORPH:
        ensure_morph(session)

    for cycle in range(max(1, int(max_cycles))):
        _ibj_double(
            session,
            f"{POLICY_ID}_ibj_{cycle}",
            center_x=171,
            stop_y=1595,
        )
        if int(session.state.room_id) != ROOM_RED_TOWER:
            break
        if int(session.state.samus_y) <= 1610:
            # Staying centered falls through.  LEFT catches the temporary
            # floor deterministically at x=141, y=1625.
            for _ in range(80):
                state = hold(session, 1, "LEFT", reason=f"{POLICY_ID}_catch")
                if MID_FLOOR.matches(state):
                    settle_hold(session, 8, reason=f"{POLICY_ID}_settle")
                    if MID_FLOOR.matches(session.state):
                        return session.state
                    break

    raise TimeoutError(
        f"{POLICY_ID}: mid floor not reached "
        f"xy=({session.state.samus_x},{session.state.samus_y}) "
        f"p={session.state.pose}"
    )


__all__ = ["POLICY_ID", "play_tunnel_to_mid_floor"]

"""Warehouse Entrance → East Tunnel pure return (K5 hop 5).

Source: ``post_ice_business_to_warehouse_pure`` ~(37, 139) pose 138 upper-left
elev band after Business→Warehouse dual **10255f**. Reverse of
``play_east_to_warehouse`` (East RIGHT-run into Warehouse).

Hybrid pure::

  1. Accept upper-left elev band (x≤60, y≤160); unmorph elev residual
  2. Beam select + LEFT door pressure (shot pulses near lip)
  3. Ordinary East Tunnel settle (multi-screen room; room-id primary)

Tape: ``tasks/speed_to_wave_ice_moat_human.json`` Phase C hop 23→24
(f20838 Warehouse elev → f21201 East ~(19,139)).
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    hold,
    require_room,
    select_weapon,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.k4_common import _STANDING_POSES
from super_metroid.routes.kpdr.k5.geometry import (
    WH_EAST_DOOR_X,
    WH_TO_EAST_FRAMES,
    WH_TO_EAST_SETTLE,
)
from super_metroid.routes.kpdr.rooms import ROOM_EAST_TUNNEL, ROOM_WAREHOUSE
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import escape_kb, is_knockback

_MORPH = frozenset({27, 28, 29, 30, 31, 37, 38, 39, 40, 41, 42, 43, 45, 49, 50, 55, 65, 137, 138})


def play_warehouse_to_east(session: ControllerSession) -> SuperMetroidState:
    """Warehouse left blue door → ordinary East Tunnel (reverse of east→wh).

    Expects upper-left elev band handoff (post Business elev up + LEFT step).
    Unmorphs elev residual pose 138, then LEFT-run/shoot into ``0xCF80``.
    """
    label = "warehouse_to_east"
    require_room(session, ROOM_WAREHOUSE, label)

    unmorph(session)
    select_weapon(session, 0)
    for _ in range(48):
        state = hold(session, 1, reason=f"{label}_stand")
        if state.pose in _MORPH:
            hold(session, 1, "UP", reason=f"{label}_unmorph")
            continue
        if int(state.velocity_y) == 0 and state.pose in _STANDING_POSES | frozenset(
            {25, 75, 77, 81}
        ):
            break

    for frame in range(WH_TO_EAST_FRAMES):
        state = session.state
        if state.room_id == ROOM_EAST_TUNNEL:
            break
        if state.room_id != ROOM_WAREHOUSE:
            break
        if is_knockback(state):
            escape_kb(
                session,
                label,
                "LEFT",
                stop_room_id=ROOM_EAST_TUNNEL,
            )
            continue
        if state.pose in _MORPH:
            hold(session, 6, "UP", reason=f"{label}_unmorph")
            continue

        x = int(state.samus_x)
        # Near left blue door: shot pulses + push through.
        if x <= WH_EAST_DOOR_X and int(state.velocity_y) == 0:
            phase = frame % 16
            if phase < 4:
                hold(session, 1, "LEFT", "X", reason=f"{label}_door_shot")
            elif phase < 12:
                hold(session, 1, "LEFT", "B", reason=f"{label}_door_push")
            else:
                hold(session, 1, "LEFT", "B", "A", reason=f"{label}_door_spin")
            continue

        # Elev band → door: run LEFT with light hop / shot cadence.
        phase = frame % 20
        if phase < 10:
            hold(session, 1, "LEFT", "B", reason=f"{label}_run")
        elif phase < 14:
            hold(session, 1, "LEFT", "B", "A", reason=f"{label}_hop")
        elif phase < 17:
            hold(session, 1, "LEFT", "X", reason=f"{label}_shot")
        else:
            hold(session, 1, "LEFT", reason=f"{label}_walk")
    else:
        state = session.state
        raise TimeoutError(
            f"{label}: left Warehouse door missed; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"door_transition={state.door_transition}"
        )

    return wait_ordinary_room(
        session,
        ROOM_EAST_TUNNEL,
        settle_frames=WH_TO_EAST_SETTLE,
        label=label,
    )


__all__ = ["play_warehouse_to_east"]

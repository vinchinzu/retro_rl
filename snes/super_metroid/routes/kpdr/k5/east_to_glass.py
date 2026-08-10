"""East Tunnel → Glass Tunnel pure return (K5 hop 6).

Source: ``post_ice_warehouse_to_east_pure`` ~(216, 364) pose 26 crouch residual
after Warehouse→East dual **285f**. Reverse of ``play_glass_to_east``
(Glass RIGHT-run into East / Boyon Gate Hall).

Hybrid pure::

  1. Accept East mid-bottom crouch residual (x∈[150,280], y∈[300,420] p26)
  2. Uncrouch / unmorph residual
  3. LEFT-run + shot pressure into Glass bottom-left blue door
  4. Ordinary Glass settle (room-id primary)

Tape: ``tasks/speed_to_wave_ice_moat_human.json`` Phase C hop 24→25
(f21201 East ~(19,139)→bottom floor LEFT → f21438 Glass ~(17,395)).
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
    EAST_GLASS_DOOR_X,
    EAST_TO_GLASS_FRAMES,
    EAST_TO_GLASS_SETTLE,
)
from super_metroid.routes.kpdr.rooms import ROOM_EAST_TUNNEL, ROOM_GLASS
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import escape_kb, is_knockback

_MORPH = frozenset({27, 28, 29, 30, 31, 37, 38, 39, 40, 41, 42, 43, 45, 49, 50, 55, 65, 137, 138})
# Crouch / kneel residuals from multi-screen East settle (pose 26 accepted).
_CROUCH = frozenset({26, 39, 40, 43})


def play_east_to_glass(session: ControllerSession) -> SuperMetroidState:
    """East Tunnel left blue door → ordinary Glass (reverse of glass→east).

    Expects mid-bottom East handoff after warehouse_to_east. Uncrouches pose 26
    residual, then LEFT-run/shoot into ``0xCEFB``.
    """
    label = "east_to_glass"
    require_room(session, ROOM_EAST_TUNNEL, label)

    unmorph(session)
    select_weapon(session, 0)
    for _ in range(48):
        state = hold(session, 1, reason=f"{label}_stand")
        if state.pose in _MORPH | _CROUCH:
            hold(session, 1, "UP", reason=f"{label}_uncrouch")
            continue
        if int(state.velocity_y) == 0 and state.pose in _STANDING_POSES | frozenset(
            {1, 2, 9, 10, 12, 25, 75, 77, 81}
        ):
            break

    for frame in range(EAST_TO_GLASS_FRAMES):
        state = session.state
        if state.room_id == ROOM_GLASS:
            break
        if state.room_id != ROOM_EAST_TUNNEL:
            break
        if is_knockback(state):
            escape_kb(
                session,
                label,
                "LEFT",
                stop_room_id=ROOM_GLASS,
            )
            continue
        if state.pose in _MORPH | _CROUCH:
            hold(session, 6, "UP", reason=f"{label}_uncrouch")
            continue

        x = int(state.samus_x)
        # Near left blue door (Glass): shot pulses + push through.
        if x <= EAST_GLASS_DOOR_X and int(state.velocity_y) == 0:
            phase = frame % 16
            if phase < 4:
                hold(session, 1, "LEFT", "X", reason=f"{label}_door_shot")
            elif phase < 12:
                hold(session, 1, "LEFT", "B", reason=f"{label}_door_push")
            else:
                hold(session, 1, "LEFT", "B", "A", reason=f"{label}_door_spin")
            continue

        # Mid-bottom floor → door: run LEFT with light hop / shot cadence
        # (tape: LEFT+B then LEFT+B+X near lip).
        phase = frame % 20
        if phase < 12:
            hold(session, 1, "LEFT", "B", reason=f"{label}_run")
        elif phase < 16:
            hold(session, 1, "LEFT", "B", "X", reason=f"{label}_shot")
        elif phase < 18:
            hold(session, 1, "LEFT", "B", "A", reason=f"{label}_hop")
        else:
            hold(session, 1, "LEFT", reason=f"{label}_walk")
    else:
        state = session.state
        raise TimeoutError(
            f"{label}: left East door missed; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"door_transition={state.door_transition}"
        )

    return wait_ordinary_room(
        session,
        ROOM_GLASS,
        settle_frames=EAST_TO_GLASS_SETTLE,
        label=label,
    )


__all__ = ["play_east_to_glass"]

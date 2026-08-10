"""Ice Beam Room → Ice Snake pure return (K5 stack first hop / ice return).

Post-Ice pure handoff ``post_ice_snake_to_ice_pure`` ends ``0xA890``
~(187, 120) pose 81 with Ice collected (beams ``0x1007`` Spazer mainline).
Human tape Phase B return hop 19 (f16491 leave): unmorph chozo, LEFT run
along y≈139 shelf into left blue door → Snake top-left ~(18, 139).

Tape recon: ``docs/tasks/SM-SPEED-ICE-MOAT-HUMAN.md`` Phase B return.
Do not clone thrash RLE f16277–16365 — re-solve clean leave.
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
from super_metroid.routes.kpdr.rooms import ROOM_ICE, ROOM_ICE_SNAKE
from super_metroid.routes.kpdr.ice.geometry import (
    ICE_BEAM_MASK,
    ICE_LEAVE_DOOR_X,
    ICE_LEAVE_FRAMES,
    ICE_SNAKE_RETURN_SETTLE,
    has_ice,
)
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import escape_kb, is_knockback


def play_ice_to_snake(session: ControllerSession) -> SuperMetroidState:
    """Ice Beam Room left blue door → ordinary Ice Snake top-left.

    Expects Ice collected (pure Snake→Ice PLM / continuous ice tip end).
    Unmorphs chozo seat, selects beam, LEFT-run into Snake ``0xA8B9``.
    """
    label = "ice_to_snake"
    require_room(session, ROOM_ICE, label)
    if not has_ice(session.state):
        raise RuntimeError(
            f"{label}: Ice not collected "
            f"(beams=0x{int(session.state.collected_beams):04X}; "
            f"need bit 0x{ICE_BEAM_MASK:04X})"
        )

    unmorph(session)
    select_weapon(session, 0)
    for _ in range(40):
        state = hold(session, 1, reason=f"{label}_stand")
        if state.pose in (137, 138, 39, 40):
            hold(session, 1, "UP", reason=f"{label}_unmorph")
            continue
        if state.velocity_y == 0 and state.pose in _STANDING_POSES:
            break

    for frame in range(ICE_LEAVE_FRAMES):
        state = session.state
        if state.room_id == ROOM_ICE_SNAKE:
            break
        if state.room_id != ROOM_ICE:
            break
        if is_knockback(state):
            escape_kb(
                session,
                label,
                "LEFT",
                stop_room_id=ROOM_ICE_SNAKE,
            )
            continue
        if state.pose in (137, 138, 39, 40):
            hold(session, 6, "UP", reason=f"{label}_unmorph")
            continue

        x = int(state.samus_x)
        # Near left blue door: shot pulses + push through.
        if x <= ICE_LEAVE_DOOR_X and int(state.velocity_y) == 0:
            phase = frame % 16
            if phase < 4:
                hold(session, 1, "LEFT", "X", reason=f"{label}_door_shot")
            elif phase < 12:
                hold(session, 1, "LEFT", "B", reason=f"{label}_door_push")
            else:
                hold(session, 1, "LEFT", "B", "A", reason=f"{label}_door_spin")
            continue

        # Chozo shelf → door: run LEFT with light hop / shot cadence.
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
            f"{label}: left Ice door missed; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"door_transition={state.door_transition} "
            f"beams=0x{int(state.collected_beams):04X}"
        )

    return wait_ordinary_room(
        session,
        ROOM_ICE_SNAKE,
        settle_frames=ICE_SNAKE_RETURN_SETTLE,
        label=label,
    )


__all__ = ["play_ice_to_snake"]

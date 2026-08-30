"""K4 Wave Beam Room → Double Chamber pure return (rr-pd0i / rr-vqv3 stack).

Post-Wave continuous tip ends ``0xADDE`` ~(171,123) pose 137 (morph on chozo).
Human tape Phase B leave (f5720–5908 after thrash) walks LEFT to the blue
door into Double top-right ~(18,139). Re-solve geometry — do not clone thrash
RLE from f5468–5700.

Tape recon: ``docs/tasks/SM-SPEED-ICE-MOAT-HUMAN.md`` Phase B hop 8.
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
from super_metroid.routes.kpdr.norfair.common import _STANDING_POSES
from super_metroid.routes.kpdr.rooms import ROOM_DOUBLE_CHAMBER, ROOM_WAVE
from super_metroid.routes.kpdr.wave.geometry import (
    WAVE_BEAM_MASK,
    WAVE_DOOR_X,
    WAVE_DOUBLE_SETTLE,
    WAVE_LEAVE_FRAMES,
    has_wave,
)
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import escape_kb, is_knockback


def play_wave_to_double_chamber(session: ControllerSession) -> SuperMetroidState:
    """Wave Beam Room left blue door → ordinary Double Chamber top-right.

    Expects Wave collected (continuous tip / pure Wave successor). Unmorphs
    chozo seat, selects beam, and LEFT-run/shot into Double ``0xADAD``.
    """
    label = "wave_to_double_chamber"
    require_room(session, ROOM_WAVE, label)
    if not has_wave(session.state):
        raise RuntimeError(
            f"{label}: Wave not collected "
            f"(beams=0x{int(session.state.collected_beams):04X}; "
            f"need bit 0x{WAVE_BEAM_MASK:04X})"
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

    for frame in range(WAVE_LEAVE_FRAMES):
        state = session.state
        if state.room_id == ROOM_DOUBLE_CHAMBER:
            break
        if state.room_id != ROOM_WAVE:
            break
        if is_knockback(state):
            escape_kb(
                session,
                label,
                "LEFT",
                stop_room_id=ROOM_DOUBLE_CHAMBER,
            )
            continue
        if state.pose in (137, 138, 39, 40):
            hold(session, 6, "UP", reason=f"{label}_unmorph")
            continue

        x = int(state.samus_x)
        # Near door: shot pulses + push through blue door.
        if x <= WAVE_DOOR_X and int(state.velocity_y) == 0:
            phase = frame % 16
            if phase < 4:
                hold(session, 1, "LEFT", "X", reason=f"{label}_door_shot")
            elif phase < 12:
                hold(session, 1, "LEFT", "B", reason=f"{label}_door_push")
            else:
                hold(session, 1, "LEFT", "B", "A", reason=f"{label}_door_spin")
            continue

        # Mid-room: run LEFT with light hop / shot cadence (chozo shelf → door).
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
            f"{label}: left Wave door missed; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"door_transition={state.door_transition}"
        )

    return wait_ordinary_room(
        session,
        ROOM_DOUBLE_CHAMBER,
        settle_frames=WAVE_DOUBLE_SETTLE,
        label=label,
    )


__all__ = ["play_wave_to_double_chamber"]

"""Pure Business Center → Ice Beam Gate Room (Super green LEFT).

Tape recon: ``tasks/speed_to_wave_ice_moat_human.json`` f9988→10817 entry
Ice Gate ~(18,907). Human path had thrash (missile SELECT, floor climbs);
product re-solves from continuous Business elevator pin via door-height drop
+ Super pressure — do not clone thrash RLE.

Reusable Super-door cadence: :func:`~super_metroid.routes.skills.door.super_door_pressure_frame`
(face LEFT). Geometry: :mod:`super_metroid.routes.kpdr.ice.geometry`.
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
from super_metroid.routes.kpdr.ice.geometry import (
    BUSINESS_ELEVATOR_Y,
    DOOR_BAND_FRAMES,
    ELEVATOR_SETTLE_FRAMES,
    ICE_GATE_SETTLE_FRAMES,
    ICE_SUPER_DOOR_X,
    ICE_SUPER_LIP_X_MAX,
    ICE_SUPER_Y_MAX,
    ICE_SUPER_Y_MIN,
    LEDGE_POSES,
    SUPER_PRESSURE_FRAMES,
    on_ice_super_lip,
)
from super_metroid.routes.kpdr.rooms import ROOM_BUSINESS, ROOM_ICE_GATE
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.door import super_door_pressure_frame
from super_metroid.routes.skills.knockback import (
    escape_knockback_spin,
    is_knockback,
)


def _settle_business_elevator(session: ControllerSession, label: str) -> None:
    """Land continuous Business elev tip (pose 155 / y drift) at platform y."""
    stable = 0
    for _ in range(ELEVATOR_SETTLE_FRAMES):
        state = hold(session, 1, reason=f"{label}_elevator_settle")
        if int(state.samus_y) == BUSINESS_ELEVATOR_Y:
            stable += 1
            if stable >= 24:
                return
        else:
            stable = 0
    # Already past elev (e.g. mid-platform pure source) — continue.
    if int(session.state.samus_y) > BUSINESS_ELEVATOR_Y + 40:
        return
    raise TimeoutError(f"{label}: elevator did not settle: {session.state}")


def _drop_to_ice_super_band(session: ControllerSession, label: str) -> None:
    """Elevator / upper Business → Super-door height band (prefer left).

    Cathedral hop uses RIGHT-first shallow drop for the top-right blue door.
    Ice Super is **left** mid-shaft slightly lower (y≈900–940). Prefer LEFT
    once near door height so the first solid land is the Super lip, not the
    right Cathedral shelf.
    """
    for frame in range(DOOR_BAND_FRAMES):
        state = session.state
        if state.room_id != ROOM_BUSINESS:
            return
        if on_ice_super_lip(state):
            return
        if (
            ICE_SUPER_Y_MIN <= int(state.samus_y) <= ICE_SUPER_Y_MAX
            and int(state.velocity_y) == 0
            and int(state.pose) in LEDGE_POSES
            and int(state.samus_x) <= ICE_SUPER_LIP_X_MAX + 40
        ):
            return

        if is_knockback(state):
            escape_knockback_spin(
                session,
                prefer_dir="LEFT",
                run_frames=4,
                spin_frames=12,
                label=f"{label}_kb",
            )
            continue

        # Above door band: drop with short Hi-Jump pulses; bias LEFT near target.
        y = int(state.samus_y)
        if y < ICE_SUPER_Y_MIN - 40:
            direction = "LEFT" if (frame // 40) % 2 == 0 else "RIGHT"
            if frame % 40 < 10:
                buttons = (direction, "B", "A")
            else:
                buttons = (direction, "B")
        elif y < ICE_SUPER_Y_MIN:
            # Approaching band — LEFT walk / short hop onto left shelves.
            if frame % 28 < 8:
                buttons = ("LEFT", "B", "A")
            else:
                buttons = ("LEFT", "B")
        else:
            # In/near band but not lip: walk left toward door, light hop if stuck.
            if int(state.samus_x) > ICE_SUPER_LIP_X_MAX:
                if frame % 24 < 6 and int(state.velocity_y) == 0:
                    buttons = ("LEFT", "B", "A")
                else:
                    buttons = ("LEFT", "B")
            else:
                buttons = ("LEFT",) if int(state.velocity_y) == 0 else ("LEFT", "B")
        hold(session, 1, *buttons, reason=f"{label}_door_band")
    else:
        raise TimeoutError(
            f"{label}: Ice Super door band missed: {session.state} "
            f"(want y∈[{ICE_SUPER_Y_MIN},{ICE_SUPER_Y_MAX}] x≤{ICE_SUPER_LIP_X_MAX})"
        )


def _open_ice_super_and_enter(session: ControllerSession, label: str) -> None:
    """Super pressure LEFT on green door + walk into Ice Gate."""
    unmorph(session)
    if int(session.state.selected_item) != 2:
        select_weapon(session, 2)
    hold(session, 4, reason=f"{label}_super_ready")

    for frame in range(SUPER_PRESSURE_FRAMES):
        state = session.state
        if state.room_id == ROOM_ICE_GATE:
            return
        if is_knockback(state):
            escape_knockback_spin(
                session,
                prefer_dir="LEFT",
                run_frames=3,
                spin_frames=10,
                label=f"{label}_door_kb",
            )
            if int(session.state.selected_item) != 2:
                select_weapon(session, 2)
            continue

        # Too far right of lip: reseat left before more Supers.
        if (
            int(state.samus_x) > ICE_SUPER_LIP_X_MAX + 20
            and ICE_SUPER_Y_MIN <= int(state.samus_y) <= ICE_SUPER_Y_MAX
        ):
            hold(session, 1, "LEFT", "B", reason=f"{label}_reseat")
            continue

        # Close enough: Super cadence + enter push.
        if int(state.samus_x) <= ICE_SUPER_DOOR_X + 30:
            state = super_door_pressure_frame(
                session,
                frame,
                label=label,
                face="LEFT",
                period=28,
                shoot_end=6,
                face_end=14,
                run_end=22,
            )
            if state.room_id == ROOM_ICE_GATE:
                return
            continue

        # Approach lip from mid-band.
        phase = frame % 24
        if phase < 6:
            hold(session, 1, "LEFT", "X", reason=f"{label}_super_plant")
        elif phase < 12:
            hold(session, 1, "LEFT", reason=f"{label}_face")
        else:
            hold(session, 1, "LEFT", "B", reason=f"{label}_approach")

    raise TimeoutError(f"{label}: Ice Super door did not open: {session.state}")


def play_business_to_ice_gate(session: ControllerSession) -> SuperMetroidState:
    """Business Center → ordinary Ice Beam Gate Room via mid-left Super green.

    Source: continuous Business elev pin (``post_business_continuous`` /
    Spazer twin) or mid-platform pure sources already in Business.
    Exit: Ice Gate ``0xA815`` ordinary gameplay (tape entry ~(18,907)).
    """
    label = "business_to_ice_gate"
    require_room(session, ROOM_BUSINESS, label)

    # Elev tip (y near 0–680, pose 155): settle. Mid-room sources skip.
    y0 = int(session.state.samus_y)
    if y0 < BUSINESS_ELEVATOR_Y + 80 or int(session.state.pose) == 155:
        _settle_business_elevator(session, label)

    if not on_ice_super_lip(session.state):
        _drop_to_ice_super_band(session, label)

    _open_ice_super_and_enter(session, label)

    state = wait_ordinary_room(
        session,
        ROOM_ICE_GATE,
        settle_frames=ICE_GATE_SETTLE_FRAMES,
        label=label,
    )
    # Human tape ordinary pin after transition: ~(1752, 651) pose 10 facing left.
    # wait_ordinary may still leave spin-air (pose 82); brief stand for next hop.
    unmorph(session)
    for _ in range(40):
        st = hold(session, 1, reason=f"{label}_stand")
        if (
            int(st.velocity_y) == 0
            and int(st.pose) in (1, 2, 9, 10)
            and int(st.door_transition) == 0
        ):
            return st
    return state


__all__ = [
    "play_business_to_ice_gate",
]

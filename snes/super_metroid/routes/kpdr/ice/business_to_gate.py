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
from super_metroid.routes.kpdr.business_climb import (
    _business_high_jump_platforms,
    _maybe_dump_climb_state,
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
from super_metroid.routes.kpdr.rooms import ROOM_BUSINESS, ROOM_HJ_SHAFT, ROOM_ICE_GATE
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.door import super_door_pressure_frame
from super_metroid.routes.skills.knockback import (
    escape_knockback_spin,
    is_knockback,
)

# Frog Save → Business pure settle is floor ~(216,1419). Ice Super lip is
# mid-shaft y∈[880,960]. Floor / below-Super pins climb to elevator then drop.
_BUSINESS_FLOOR_Y_MIN = 1350


def _settle_business_elevator(session: ControllerSession, label: str) -> None:
    """Land continuous Business elev tip (pose 155 / y drift) at platform y.

    Platform land is y≈683 (business_climb); continuous elev tip targets 680.
    Accept a small band so floor-climb and elev-ride pins both settle.
    """
    elev_lo = BUSINESS_ELEVATOR_Y - 5
    elev_hi = BUSINESS_ELEVATOR_Y + 10
    stable = 0
    for _ in range(ELEVATOR_SETTLE_FRAMES):
        state = hold(session, 1, reason=f"{label}_elevator_settle")
        y = int(state.samus_y)
        if elev_lo <= y <= elev_hi and int(state.velocity_y) == 0:
            stable += 1
            if stable >= 24:
                return
        else:
            stable = 0
    # Already past elev (e.g. mid-platform pure source) — continue.
    if int(session.state.samus_y) > BUSINESS_ELEVATOR_Y + 40:
        return
    # Top / mid pin that never rode elev: drop path handles Super band.
    if int(session.state.samus_y) < elev_lo:
        return
    raise TimeoutError(f"{label}: elevator did not settle: {session.state}")


def _anchor_business_floor_for_climb(
    session: ControllerSession, label: str
) -> None:
    """Re-pin to pure dual floor band ~(200–240,1419) before elev climb.

    Continuous frog→business often leaves floor-left (x≈20–30). Stock
    ``_business_high_jump_platforms`` setup uses long LEFT+B+A runs that exit
    into HJ shaft ``0xAA41`` from that pin (ice_r1). Pure dual frog settle
    ~(216,1419) is the proven climb start — walk there first, never LEFT
    along the floor near the HJ door.
    """
    unmorph(session)
    # Soft left bound: never hold LEFT on floor when x is already mid-low.
    for _ in range(200):
        st = session.state
        if st.room_id != ROOM_BUSINESS:
            raise TimeoutError(
                f"{label}: left Business during floor anchor: {st}"
            )
        x = int(st.samus_x)
        y = int(st.samus_y)
        if y < _BUSINESS_FLOOR_Y_MIN - 20:
            return  # no longer on floor
        grounded = int(st.velocity_y) == 0 and int(st.pose) in LEDGE_POSES
        if 200 <= x <= 240 and grounded:
            hold(session, 12, reason=f"{label}_floor_anchor_settle")
            return
        if x < 200:
            # Only RIGHT along floor — LEFT risks HJ door at x≲40.
            hold(session, 1, "RIGHT", "B", reason=f"{label}_floor_anchor_r")
        elif x > 240 and grounded:
            # Nudge left carefully; stop if door-close.
            if x > 80:
                hold(session, 1, "LEFT", reason=f"{label}_floor_anchor_l")
            else:
                hold(session, 1, "RIGHT", reason=f"{label}_floor_anchor_bounce")
        else:
            hold(session, 1, reason=f"{label}_floor_anchor_idle")
    raise TimeoutError(f"{label}: floor climb anchor missed: {session.state}")


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


def _recover_hj_door_to_business(
    session: ControllerSession, label: str
) -> bool:
    """If setup kissed HJ shaft, pressure RIGHT back into Business floor."""
    if session.state.room_id == ROOM_BUSINESS:
        return True
    if session.state.room_id != ROOM_HJ_SHAFT:
        return False
    unmorph(session)
    for frame in range(320):
        st = session.state
        if st.room_id == ROOM_BUSINESS:
            wait_ordinary_room(
                session,
                ROOM_BUSINESS,
                settle_frames=40,
                label=f"{label}_hj_return",
            )
            return True
        if is_knockback(st):
            escape_knockback_spin(
                session,
                prefer_dir="RIGHT",
                run_frames=4,
                spin_frames=10,
                label=f"{label}_hj_kb",
            )
            continue
        y = int(st.samus_y)
        if y < 1200:
            if frame % 24 < 8:
                hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_hj_drop")
            else:
                hold(session, 1, "RIGHT", "B", reason=f"{label}_hj_drop_run")
        else:
            phase = frame % 16
            if phase < 5:
                hold(session, 1, "RIGHT", "X", reason=f"{label}_hj_door_shot")
            elif phase < 12:
                hold(session, 1, "RIGHT", "B", reason=f"{label}_hj_door_push")
            else:
                hold(session, 1, "RIGHT", reason=f"{label}_hj_door_walk")
    return session.state.room_id == ROOM_BUSINESS


def _right_biased_floor_recover(session: ControllerSession, label: str) -> None:
    """Drop to Business floor without kissing the left HJ door.

    Stock :func:`_fall_to_business_floor` can exit into ``0xAA41`` on continuous
    Ice natural entry (rr-kxge). Prefer RIGHT on the left half of the room.
    """
    if session.state.room_id == ROOM_HJ_SHAFT:
        if not _recover_hj_door_to_business(session, label):
            raise TimeoutError(
                f"{label}: HJ recover failed before floor recover: {session.state}"
            )
    unmorph(session)
    for frame in range(500):
        state = session.state
        if state.room_id != ROOM_BUSINESS:
            if state.room_id == ROOM_HJ_SHAFT and _recover_hj_door_to_business(
                session, label
            ):
                continue
            raise TimeoutError(
                f"{label}: left Business during floor recover: {session.state}"
            )
        if (
            int(state.pose) in LEDGE_POSES
            and int(state.velocity_y) == 0
            and int(state.samus_y) >= 1405
        ):
            break
        x = int(state.samus_x)
        if x <= 90:
            direction = "RIGHT"
        elif x >= 230:
            direction = "LEFT"
        else:
            direction = "RIGHT" if (frame % 40) < 28 else "LEFT"
        phase = frame % 70
        buttons = (direction, "B") if phase < 45 else (direction, "B", "A")
        hold(session, 1, *buttons, reason=f"{label}_rbiased_recover")
    else:
        raise TimeoutError(f"{label}: right-biased floor recover: {session.state}")
    _anchor_business_floor_for_climb(session, label)
    hold(session, 16, "RIGHT", "B", reason=f"{label}_rbiased_buf")
    hold(session, 12, reason=f"{label}_rbiased_buf_settle")


def _climb_business_floor_to_elevator(session: ControllerSession, label: str) -> None:
    """Business floor / below-Super → elevator platform (then drop to Super).

    Attempt order (rr-kxge continuous Ice stabilize):
    1. Pure dual first try — runup 8 / pos_1339=84
    2. Pure 907 retry — runup 14 / pos 84 after RIGHT-biased recover
    3–4. Continuous 1227/907 — pos_1339=90, runup 8 then 14
    5–6. Extra continuous attempts after HJ door recover
    """
    unmorph(session)
    if session.state.room_id != ROOM_BUSINESS:
        raise TimeoutError(f"{label}: not in Business for floor climb: {session.state}")
    _anchor_business_floor_for_climb(session, label)
    beams = int(session.state.collected_beams)
    dump = (
        "business_floor_pre_ice_climb_wave"
        if beams & 0x1000
        else "business_floor_pre_ice_climb"
    )
    _maybe_dump_climb_state(session, dump)

    attempts: list[tuple[int, int]] = [
        (8, 84),
        (14, 84),
        (8, 90),
        (14, 90),
        (8, 90),
        (14, 90),
    ]
    last_err: TimeoutError | None = None
    for i, (runup, pos_1339) in enumerate(attempts):
        try:
            if i > 0:
                if session.state.room_id == ROOM_HJ_SHAFT:
                    if not _recover_hj_door_to_business(session, label):
                        raise TimeoutError(
                            f"{label}: left Business during floor climb: "
                            f"{session.state}"
                        )
                if session.state.room_id != ROOM_BUSINESS:
                    raise TimeoutError(
                        f"{label}: left Business during floor climb: {session.state}"
                    )
                _right_biased_floor_recover(session, label)
            _business_high_jump_platforms(
                session,
                runup_907=runup,
                pos_1339=pos_1339,
                # Bound LEFT setup only on continuous-tuned pos≈90 retries.
                bound_floor_left=(pos_1339 >= 90),
            )
            last_err = None
            break
        except TimeoutError as exc:
            last_err = exc
            _maybe_dump_climb_state(session, f"business_ice_climb_fail_{i}")
            if session.state.room_id == ROOM_HJ_SHAFT:
                if not _recover_hj_door_to_business(session, label):
                    # Keep trying remaining attempts only if we get back.
                    continue
                continue
            if session.state.room_id != ROOM_BUSINESS:
                raise TimeoutError(
                    f"{label}: left Business during floor climb: {session.state}"
                ) from exc
            continue
    if last_err is not None:
        raise last_err
    if session.state.room_id != ROOM_BUSINESS:
        raise TimeoutError(
            f"{label}: left Business after floor climb: {session.state}"
        )
    y = int(session.state.samus_y)
    if y > BUSINESS_ELEVATOR_Y + 40:
        raise TimeoutError(
            f"{label}: floor climb missed elevator (y={y}): {session.state}"
        )


def play_business_to_ice_gate(session: ControllerSession) -> SuperMetroidState:
    """Business Center → ordinary Ice Beam Gate Room via mid-left Super green.

    Source: continuous Business elev pin (``post_business_continuous`` /
    Spazer twin), mid-platform pure sources, or Wave→Business **floor** settle
    ~(216,1419) after Frog Save (rr-vsjy / continuous ice prefix). Floor pins
    climb to elevator then drop to Super lip — drop-only cannot climb.
    Exit: Ice Gate ``0xA815`` ordinary gameplay (tape entry ~(18,907)).
    """
    label = "business_to_ice_gate"
    require_room(session, ROOM_BUSINESS, label)

    y0 = int(session.state.samus_y)
    # Floor / below Super band: climb up first (Frog door settle, mid shelves).
    climbed_from_floor = False
    if y0 >= _BUSINESS_FLOOR_Y_MIN or y0 > ICE_SUPER_Y_MAX + 40:
        if not on_ice_super_lip(session.state):
            _climb_business_floor_to_elevator(session, label)
            climbed_from_floor = True
    # Elev tip (pose 155 / y drift): settle. Floor climb lands standing y≈683
    # (business_climb platform); BUSINESS_ELEVATOR_Y is 680 — skip exact-y settle.
    y1 = int(session.state.samus_y)
    on_elev_platform = (
        BUSINESS_ELEVATOR_Y - 5 <= y1 <= BUSINESS_ELEVATOR_Y + 10
        and int(session.state.velocity_y) == 0
        and int(session.state.pose) in LEDGE_POSES
    )
    if climbed_from_floor and on_elev_platform:
        pass  # ready to drop to Super lip
    elif y1 < BUSINESS_ELEVATOR_Y + 80 or int(session.state.pose) == 155:
        if not on_elev_platform:
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

"""K4 Business → Cathedral Entrance → Cathedral → Rising Tide controllers.

First Bubble visit is **Cathedral climb** (no Speed). These pure controllers
walk Business elevator → top-right Cathedral door → Super door into Cathedral
→ lower-right green Super door into Rising Tide.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    ensure_morph,
    hold,
    require_room,
    select_weapon,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.k4_common import (
    _ELEVATOR_Y,
    _LEDGE_POSES,
    _STANDING_POSES,
)
from super_metroid.routes.kpdr.rooms import (
    ROOM_BUSINESS,
    ROOM_CATHEDRAL,
    ROOM_CATHEDRAL_ENTRANCE,
    ROOM_RISING_TIDE,
)
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.door import super_door_pressure_frame
from super_metroid.routes.skills.knockback import (
    escape_knockback_spin,
    is_knockback,
)

# Top-right Cathedral door is block [15, 55] → pixel y ≈ 880 (screen 3 of 7).
# Keep the band tight: y>900 is a lower shelf that falls past the door lip.
_CATHEDRAL_DOOR_Y_MIN = 840
_CATHEDRAL_DOOR_Y_MAX = 900
_CATHEDRAL_DOOR_BAND_FRAMES = 500
_CATHEDRAL_DOOR_FRAMES = 400
_CATHEDRAL_SETTLE_FRAMES = 320
# Grounded poses only for the Cathedral door ledge (exclude knockback 137/138).
# Same set as k4_common._LEDGE_POSES; local alias preserves historical names.
_CATHEDRAL_LEDGE_POSES = _LEDGE_POSES
_CATH_LEDGE_POSES = _LEDGE_POSES
# Cathedral Entrance is 3×2 screens; right red Super door is block [47, 7].
# Upper left lip is a dead-end (solid at x≈91); KPDR path bombs through the
# left morph-tunnel floor, crosses the bottom, climbs mid platforms (x≈560–680
# — extreme-right wall at x≈730 is a dead climb), Supers the right red door.
# Door ledge standing height ≈ y 112–150; mid shelf ≈ y 280–320.
_CATH_ENTRANCE_BOMB_FRAMES = 12
_CATH_ENTRANCE_FLOOR_FRAMES = 800
_CATH_ENTRANCE_CLIMB_FRAMES = 1800
_CATH_ENTRANCE_DOOR_X = 680
_CATH_ENTRANCE_DOOR_Y = 170
_CATH_ENTRANCE_CLIMB_X_MIN = 560
_CATH_ENTRANCE_CLIMB_X_MAX = 680
_CATH_ENTRANCE_MID_Y = (270, 330)
_CATH_ENTRANCE_TO_CATH_SETTLE_FRAMES = 320
# Cathedral (0xA788) is 3×2 screens; left blue lip spawn ≈ (39, 124).
# Right green Super door is on the **lower** right near lava (empirical pure pin
# band x≈700–730 y≈350–380) — not the upper y≈120 lip. Graph block [47, 7]
# matches lower-screen local Y when the door sits on the bottom map row.
# Cross ridge then drop/approach lower right; select Supers; open green door.
# NOTE: unlimited-energy assist can stunlock pose 137/138 in contact damage —
# never idle-plant through knockback; always jump-escape.
_CATH_CROSS_FRAMES = 5000
_CATH_DOOR_X = 640
_CATH_DOOR_Y_MIN = 300
_CATH_DOOR_Y_MAX = 420
_CATH_FALL_Y = 230
_CATH_TO_RISING_SETTLE_FRAMES = 320


def play_business_to_cathedral_entrance(
    session: ControllerSession,
) -> SuperMetroidState:
    """Business Center elevator → Cathedral Entrance via top-right blue door.

    Continuous Business tip is still on the arriving elevator.  Settle at
    ``y=680``, drop only to the top-right door band (block ``[15, 55]`` /
    pixel y≈880 — **not** the floor Frog door), beam-shot the right blue
    door, and settle ordinary gameplay in ``0xA7B3``.
    """
    label = "business_to_cathedral_entrance"
    require_room(session, ROOM_BUSINESS, label)

    stable_elevator_frames = 0
    for _ in range(600):
        state = hold(session, 1, reason=f"{label}_elevator_settle")
        if state.samus_y == _ELEVATOR_Y:
            stable_elevator_frames += 1
            if stable_elevator_frames >= 24:
                break
        else:
            stable_elevator_frames = 0
    else:
        raise TimeoutError(f"{label}: elevator did not settle: {session.state}")

    # Shallow drop: leave the elevator platform and land on the top-right
    # door ledge.  Prefer RIGHT-first so the first landing is the upper
    # door shelf (LEFT-first often catches a lower y≈923 shelf that falls
    # past the lip).  Short alternating Hi-Jump pulses clear center shelves
    # without committing to the full Frog-floor descent.
    for frame in range(_CATHEDRAL_DOOR_BAND_FRAMES):
        state = session.state
        if (
            _CATHEDRAL_DOOR_Y_MIN <= state.samus_y <= _CATHEDRAL_DOOR_Y_MAX
            and state.velocity_y == 0
            and state.pose in _CATHEDRAL_LEDGE_POSES
        ):
            break
        direction = "RIGHT" if (frame // 50) % 2 == 0 else "LEFT"
        if frame % 50 < 12:
            buttons = (direction, "B", "A")
        else:
            buttons = (direction, "B")
        hold(session, 1, *buttons, reason=f"{label}_door_band")
    else:
        raise TimeoutError(f"{label}: cathedral door band missed: {session.state}")

    select_weapon(session, 0)
    # Spazer continuous Business: Sovas on the door lip can pin pose 137 at
    # y≈907 mid open-loop. Keep shooting, escape knockback briefly, resume.
    for _ in range(_CATHEDRAL_DOOR_FRAMES + 280):
        state = session.state
        if state.room_id == ROOM_CATHEDRAL_ENTRANCE:
            break
        if is_knockback(state):
            escape_knockback_spin(
                session,
                prefer_dir="RIGHT",
                run_frames=4,
                spin_frames=12,
                label=f"{label}_kb",
            )
            continue
        hold(session, 1, "RIGHT", "B", "X", reason=f"{label}_door")
    else:
        raise TimeoutError(f"{label}: Cathedral door missed: {session.state}")

    state = wait_ordinary_room(
        session,
        ROOM_CATHEDRAL_ENTRANCE,
        settle_frames=_CATHEDRAL_SETTLE_FRAMES,
        label=label,
    )
    # Brief standing settle only — do not walk into the room (Sovas on the
    # left lip). CATH-02 starts from the left blue spawn band.
    unmorph(session)
    for _ in range(30):
        st = hold(session, 1, reason=f"{label}_lip_settle")
        if (
            st.velocity_y == 0
            and st.pose in _CATH_LEDGE_POSES
            and st.door_transition == 0
        ):
            return st
    return state


def _cath_entrance_land(session: ControllerSession, label: str) -> None:
    """Standing land on left blue lip (entry settle only)."""
    for _ in range(40):
        state = hold(session, 1, reason=f"{label}_land")
        if state.velocity_y == 0 and state.pose in _STANDING_POSES:
            break


def _cath_entrance_bomb_drop(session: ControllerSession, label: str) -> None:
    """Phase 1: bomb-drop through left morph-tunnel floor."""
    unmorph(session)
    ensure_morph(session)
    for _ in range(40):
        state = hold(session, 1, "RIGHT", reason=f"{label}_morph_edge")
        if state.samus_x >= 82:
            break
    for _ in range(_CATH_ENTRANCE_BOMB_FRAMES):
        hold(session, 2, "X", reason=f"{label}_bomb")
        state = hold(session, 48, reason=f"{label}_bomb_fuse")
        if state.samus_y > 300:
            break
    for _ in range(100):
        state = hold(session, 1, reason=f"{label}_bomb_fall")
        if state.velocity_y == 0 and state.samus_y > 300:
            break
    unmorph(session)


def _cath_entrance_floor_cross(
    session: ControllerSession, label: str
) -> tuple[int, int]:
    """Phase 2: floor cross to climb start (~x 620–680 floor band).

    Spazer continuous live chains leave Sovas on a hotter phase than the saved
    pure tip: idle-plant knockback on the floor overshoots into the x≈730 dead
    wall (pose 137) and burns the climb budget. Spin-escape + beam, stop once
    grounded in the climb-start band, and pull left if past x 700.
    """
    select_weapon(session, 0)
    # Grounded floor poses that can start the climb (stand / crouch / turn).
    floor_ok = _CATHEDRAL_LEDGE_POSES | frozenset({11, 12, 37, 38})
    max_x = session.state.samus_x
    min_y = session.state.samus_y
    for frame in range(_CATH_ENTRANCE_FLOOR_FRAMES):
        state = session.state
        if state.room_id == ROOM_CATHEDRAL:
            break
        if is_knockback(state):
            # Prefer LEFT once past climb corridor so KB does not push deeper
            # into the dead wall at x≈730.
            prefer = "LEFT" if state.samus_x >= 640 else "RIGHT"
            escape_knockback_spin(
                session,
                prefer_dir=prefer,
                run_frames=4,
                spin_frames=12,
                label=f"{label}_floor",
                run_with=("B", "X"),
                spin_with=("B", "A"),
                ensure_beam=True,
                break_on_motion_clear=True,
            )
            state = session.state
            max_x = max(max_x, state.samus_x)
            min_y = min(min_y, state.samus_y)
            continue
        # Dead-wall recovery during floor: walk back to climb corridor.
        if (
            state.samus_x >= 700
            and state.samus_y >= 380
            and state.velocity_y == 0
            and not is_knockback(state)
        ):
            hold(session, 1, "LEFT", "B", reason=f"{label}_floor_deadwall")
            state = session.state
            max_x = max(max_x, state.samus_x)
            min_y = min(min_y, state.samus_y)
            if (
                620 <= state.samus_x <= 680
                and state.velocity_y == 0
                and state.pose in floor_ok
            ):
                break
            continue
        if (
            state.samus_x >= 620
            and state.samus_x <= 690
            and state.velocity_y == 0
            and state.samus_y >= 380
            and state.pose in floor_ok
        ):
            break
        phase = frame % 45
        if phase < 20:
            inputs = ("RIGHT", "B", "A")
        elif phase < 28:
            inputs = ("RIGHT", "B", "X")
        else:
            inputs = ("RIGHT", "B")
        state = hold(session, 1, *inputs, reason=f"{label}_floor")
        max_x = max(max_x, state.samus_x)
        min_y = min(min_y, state.samus_y)
    return max_x, min_y


def _cath_entrance_climb_and_super_door(
    session: ControllerSession,
    label: str,
    *,
    max_x: int,
    min_y: int,
) -> None:
    """Phase 3: climb mid platforms (x 560–680) + Super door into Cathedral.

    Under Charge+Spazer, Sova contact on the right climb stunlocks pose 137 if
    we idle-plant through knockback — spin-escape instead (LEFT off dead wall).
    Mid Hi-Jump kept close to non-Spazer pure cadence so historical CATH-02
    source stays green; dead-wall floor recovery retries the climb corridor.

    Continuous Spazer spin-bounces the mid shelf (poses 25/26, vy≈0 apex) without
    standing. After enough near-mid spin without a stand plant, force a short
    no-A settle window so spin can land into stand_mid, then reuse pure crest.
    """
    stand_mid = frozenset({1, 2, 9, 10})
    spin_air = frozenset({25, 26, 27, 28, 81, 82})
    mid_landed = False
    door_reached = False
    settle_budget = 0  # frames of forced no-A settle remaining
    near_mid_spin = 0
    y_lo, y_hi = _CATH_ENTRANCE_MID_Y
    for frame in range(_CATH_ENTRANCE_CLIMB_FRAMES):
        state = session.state
        if state.room_id == ROOM_CATHEDRAL:
            break
        max_x = max(max_x, state.samus_x)
        min_y = min(min_y, state.samus_y)

        if is_knockback(state):
            prefer = "LEFT" if state.samus_x >= 640 else "RIGHT"
            escape_knockback_spin(
                session,
                prefer_dir=prefer,
                run_frames=4,
                spin_frames=14,
                label=label,
                run_with=("B", "X"),
                spin_with=("B", "A"),
                ensure_beam=True,
                break_on_motion_clear=True,
            )
            continue

        # Door band: Super pulses + enter.
        if (
            state.samus_y <= _CATH_ENTRANCE_DOOR_Y
            and state.samus_x >= _CATH_ENTRANCE_DOOR_X
        ):
            door_reached = True
            state = super_door_pressure_frame(
                session,
                frame,
                label=label,
                period=28,
                shoot_end=5,
                face_end=12,
                run_end=20,
            )
            if state.room_id == ROOM_CATHEDRAL:
                break
            continue

        # High near-door: push right onto the door ledge.
        if state.samus_y <= 220 and state.samus_x >= 600:
            if state.selected_item != 2:
                select_weapon(session, 2)
            phase = frame % 24
            if phase < 8:
                inputs = ("RIGHT", "B", "A")
            elif phase < 12:
                inputs = ("RIGHT", "X")
            else:
                inputs = ("RIGHT", "B")
            state = hold(session, 1, *inputs, reason=f"{label}_high")
            if state.room_id == ROOM_CATHEDRAL:
                break
            continue

        # Mid shelf: land standing, charge Hi-Jump UP-RIGHT to door.
        if (
            y_lo <= state.samus_y <= y_hi
            and state.samus_x >= 580
            and state.velocity_y == 0
            and state.pose in stand_mid
        ):
            mid_landed = True
            settle_budget = 0
            near_mid_spin = 0
            if state.selected_item != 0:
                select_weapon(session, 0)
            if state.samus_x > 650:
                for _ in range(8):
                    hold(session, 1, "LEFT", "B", reason=f"{label}_mid_runway")
            hold(session, 6, reason=f"{label}_mid_plant")
            for _ in range(30):
                hold(session, 1, "A", reason=f"{label}_mid_charge")
            for _ in range(110):
                state = hold(
                    session, 1, "RIGHT", "B", "A", reason=f"{label}_mid_jump"
                )
                min_y = min(min_y, state.samus_y)
                max_x = max(max_x, state.samus_x)
                if state.room_id == ROOM_CATHEDRAL:
                    break
                if (
                    state.samus_y <= _CATH_ENTRANCE_DOOR_Y
                    and state.samus_x >= _CATH_ENTRANCE_DOOR_X
                ):
                    door_reached = True
                    break
            if state.room_id == ROOM_CATHEDRAL:
                break
            continue

        # Continuous-only assist: cumulative spin frames near mid (no streak
        # decay — intermittent spin still counts). Every ~90 spin frames without
        # a stand plant, open a short no-A settle so spin can land. Gated on
        # not mid_landed so pure's first crest cadence is untouched.
        if (
            not mid_landed
            and y_lo - 20 <= state.samus_y <= y_hi + 30
            and state.samus_x >= 560
            and state.pose in spin_air
        ):
            near_mid_spin += 1
            if near_mid_spin >= 90 and settle_budget <= 0:
                settle_budget = 28
                near_mid_spin = 0

        if (
            settle_budget > 0
            and not mid_landed
            and state.samus_y < 380
            and state.samus_x >= 560
        ):
            settle_budget -= 1
            if state.samus_x > 640:
                dir_h = "LEFT"
            elif state.samus_x < 600:
                dir_h = "RIGHT"
            else:
                dir_h = "RIGHT"
            hold(session, 1, dir_h, "B", reason=f"{label}_mid_settle")
            continue

        # Extreme-right floor dead wall (Spazer short-crest fallout): walk left.
        if (
            state.samus_y >= 390
            and state.samus_x >= 700
            and state.velocity_y == 0
            and state.pose in stand_mid
        ):
            hold(session, 1, "LEFT", "B", reason=f"{label}_deadwall_left")
            continue

        # Climb: keep x in [560, 680] where mid platforms live.
        if state.selected_item != 0 and not door_reached:
            select_weapon(session, 0)
        x = state.samus_x
        if x > _CATH_ENTRANCE_CLIMB_X_MAX:
            dir_h = "LEFT"
        elif x < _CATH_ENTRANCE_CLIMB_X_MIN:
            dir_h = "RIGHT"
        else:
            dir_h = "RIGHT" if (frame // 40) % 2 == 0 else "LEFT"
        phase = frame % 60
        if phase < 30:
            inputs = (dir_h, "B", "A")
        elif phase < 40:
            inputs = ("A",)
        elif phase < 50:
            inputs = (dir_h, "B")
        else:
            inputs = (dir_h, "B", "X")
        state = hold(session, 1, *inputs, reason=f"{label}_climb")
        if state.room_id == ROOM_CATHEDRAL:
            break
    else:
        state = session.state
        raise TimeoutError(
            f"{label}: right Super door missed before room "
            f"0x{ROOM_CATHEDRAL:04X}; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"door_transition={state.door_transition} max_x={max_x} "
            f"min_y={min_y} mid_landed={mid_landed} door_reached={door_reached} "
            f"supers={state.super_missiles} selected={state.selected_item}"
        )


def play_cathedral_entrance_to_cathedral(
    session: ControllerSession,
) -> SuperMetroidState:
    """Cathedral Entrance left spawn → ordinary Cathedral via right red Super door.

    CATH-01 pure successor lands near the left blue lip (x≈39 / y≈139).  The
    upper ledge is a dead-end solid at x≈91 — bomb through the left morph-tunnel
    floor, cross the bottom, Hi-Jump climb mid platforms toward the right red
    Super door (block ``[47, 7]``), open it, and settle ordinary ``0xA788``.

    Open-loop phases are named helpers only — frame budgets unchanged.
    """
    label = "cathedral_entrance_to_cathedral"
    require_room(session, ROOM_CATHEDRAL_ENTRANCE, label)

    _cath_entrance_land(session, label)
    _cath_entrance_bomb_drop(session, label)
    max_x, min_y = _cath_entrance_floor_cross(session, label)
    if session.state.room_id != ROOM_CATHEDRAL:
        _cath_entrance_climb_and_super_door(
            session, label, max_x=max_x, min_y=min_y
        )

    return wait_ordinary_room(
        session,
        ROOM_CATHEDRAL,
        settle_frames=_CATH_ENTRANCE_TO_CATH_SETTLE_FRAMES,
        label=label,
    )


def play_cathedral_to_rising_tide(
    session: ControllerSession,
) -> SuperMetroidState:
    """Cathedral left lip → ordinary Rising Tide via right green Super door.

    CATH-02 pure successor lands near the left blue lip (x≈39 / y≈124).  Crest
    the **upper ridge** (Sovas + Gerutas; never plant through knockback), keep
    altitude until x≈620, then drop to the **lower-right** green Super door near
    lava (≈x700–730 / y350–380), open it, and settle ordinary ``0xAFA3``.

    Natural continuous chain after Business can leave extra mid-room contact vs
    pure isolation; re-crest with charged Hi-Jump when knocked below the ridge
    (stuck max_x≈360 class).  Caps: Morph, Bombs, Missiles, Supers, Hi-Jump,
    Varia.
    """
    label = "cathedral_to_rising_tide"
    require_room(session, ROOM_CATHEDRAL, label)

    # Natural CATH-02 exit is often mid-air (pose 81, vy≠0). Settle standing on
    # the left lip and nudge toward the pure-source pin (~x50–80) before the hop.
    for _ in range(90):
        state = hold(session, 1, reason=f"{label}_land")
        if (
            state.velocity_y == 0
            and state.pose in _CATH_LEDGE_POSES
            and state.door_transition == 0
        ):
            break
    unmorph(session)
    select_weapon(session, 0)
    for _ in range(40):
        state = session.state
        if state.samus_x >= 55 and state.velocity_y == 0:
            break
        hold(session, 1, "RIGHT", reason=f"{label}_lip_walk")

    # Opening hop off the left lip toward the first ridge structure.
    for _ in range(12):
        hold(session, 1, "RIGHT", "B", reason=f"{label}_open_run")
    for _ in range(18):
        hold(session, 1, "A", reason=f"{label}_open_charge")
    for _ in range(36):
        hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_open_jump")
    for _ in range(50):
        state = hold(session, 1, "RIGHT", "B", "X", reason=f"{label}_open_fall")
        if state.velocity_y == 0 or is_knockback(state):
            break

    max_x = session.state.samus_x
    min_y = session.state.samus_y
    door_reached = False
    high_reached = False
    stuck_frames = 0
    last_x = session.state.samus_x
    last_progress_x = session.state.samus_x
    no_progress = 0

    for frame in range(_CATH_CROSS_FRAMES):
        state = session.state
        if state.room_id == ROOM_RISING_TIDE:
            break

        max_x = max(max_x, state.samus_x)
        min_y = min(min_y, state.samus_y)
        if state.samus_y <= 160 and state.samus_x >= 180:
            high_reached = True
        if abs(state.samus_x - last_x) <= 2:
            stuck_frames += 1
        else:
            stuck_frames = 0
            last_x = state.samus_x
        if state.samus_x > last_progress_x + 4:
            last_progress_x = state.samus_x
            no_progress = 0
        else:
            no_progress += 1

        # Knockback / contact-damage: spin-escape + beam (never idle-plant).
        # Unlimited-energy assist can stunlock pose 137/138 on continuous contact
        # (Sovas / Gerutas). Prefer RIGHT until past mid pillars.
        if is_knockback(state):
            prefer = "LEFT" if state.samus_x >= 700 else "RIGHT"
            escape_knockback_spin(
                session,
                prefer_dir=prefer,
                run_frames=6,
                spin_frames=24,
                label=label,
                run_with=("B", "X"),
                spin_with=("B", "A"),
                run_reason="kb_clear",
                spin_reason="kb_spin",
                stop_room_id=ROOM_RISING_TIDE,
                break_on_motion_clear=True,
                ensure_beam=True,
            )
            stuck_frames = 0
            last_x = session.state.samus_x
            continue

        # Lower-right green Super door band (near lava, ~y350–380).
        # Shots from y≈300–330 (upper ledge) miss the shell — drop first.
        if state.samus_x >= _CATH_DOOR_X and state.samus_y >= 280:
            door_reached = True
            if state.selected_item != 2:
                select_weapon(session, 2)
            # Too high on right wall: step left and drop onto door platform.
            if state.samus_y < 340:
                phase = frame % 24
                if phase < 10:
                    inputs = ("LEFT",)
                elif phase < 16:
                    inputs = ()
                else:
                    inputs = ("RIGHT",)
                hold(session, 1, *inputs, reason=f"{label}_door_drop")
                continue
            # Too deep in lava: hop back onto platform.
            if state.samus_y > 400:
                dir_h = "LEFT" if state.samus_x > 740 else "RIGHT"
                hold(session, 1, dir_h, "B", "A", reason=f"{label}_lava_hop")
                continue
            # Aligned ~y340–400: deliberate Super pulses then enter.
            state = super_door_pressure_frame(
                session,
                frame,
                label=label,
                period=40,
                shoot_end=4,
                idle_end=20,
                face_end=28,
                run_end=34,
                ensure_weapon=False,
            )
            if state.room_id == ROOM_RISING_TIDE:
                break
            continue

        # High on the right (above door band): drop toward lower platform.
        if state.samus_x >= 620 and state.samus_y < 280:
            door_reached = True
            phase = frame % 30
            if phase < 12:
                inputs = ("RIGHT", "B")
            elif phase < 20:
                inputs = ("RIGHT",)
            else:
                inputs = ("LEFT",) if state.samus_x > 720 else ("RIGHT", "B")
            hold(session, 1, *inputs, reason=f"{label}_drop_to_door")
            continue

        # Knocked off ridge / low mid-room: re-crest with charged Hi-Jump.
        # Weak hop cadence stranded natural chains at max_x≈360 y≈360.
        if state.samus_y > _CATH_FALL_Y and state.samus_x < _CATH_DOOR_X:
            if state.selected_item != 0:
                select_weapon(session, 0)
            # Stuck on mid pillar: short LEFT runway then full charged HJ.
            if no_progress > 50 and 250 <= state.samus_x <= 480:
                for _ in range(14):
                    hold(session, 1, "LEFT", "B", reason=f"{label}_runway")
                for _ in range(16):
                    hold(session, 1, "A", reason=f"{label}_recrest_charge")
                for _ in range(40):
                    st = hold(
                        session,
                        1,
                        "RIGHT",
                        "B",
                        "A",
                        reason=f"{label}_recrest_jump",
                    )
                    min_y = min(min_y, st.samus_y)
                    max_x = max(max_x, st.samus_x)
                    if st.room_id == ROOM_RISING_TIDE:
                        break
                    if st.samus_y <= 160:
                        break
                for _ in range(10):
                    hold(session, 1, "RIGHT", "B", "X", reason=f"{label}_recrest_shot")
                no_progress = 0
                stuck_frames = 0
                last_x = session.state.samus_x
                continue
            # General low recovery: charge then UP-RIGHT Hi-Jump, keep shooting.
            grounded = (
                state.velocity_y == 0 and state.pose in _CATH_LEDGE_POSES
            )
            if grounded or stuck_frames > 20:
                for _ in range(14):
                    hold(session, 1, "A", reason=f"{label}_recover_charge")
                for _ in range(32):
                    st = hold(
                        session,
                        1,
                        "RIGHT",
                        "B",
                        "A",
                        reason=f"{label}_recover_jump",
                    )
                    min_y = min(min_y, st.samus_y)
                    max_x = max(max_x, st.samus_x)
                    if st.room_id == ROOM_RISING_TIDE or st.samus_y <= 150:
                        break
                hold(session, 1, "RIGHT", "B", "X", reason=f"{label}_recover_shot")
                stuck_frames = 0
                last_x = session.state.samus_x
                continue
            # Still airborne low: hold jump + beam right.
            phase = frame % 20
            if phase < 12:
                inputs = ("RIGHT", "B", "A")
            else:
                inputs = ("RIGHT", "B", "X")
            hold(session, 1, *inputs, reason=f"{label}_recover_air")
            continue

        # Upper ridge cross: longer Hi-Jumps + frequent beam (clear Gerutas).
        if state.selected_item != 0:
            select_weapon(session, 0)
        x = state.samus_x
        # Cadence by x-band. Mid gaps (~x300–450) need charged length; late
        # gaps (~x450–560) need long hold-A. Short crests stranded max_x≈539.
        if x < 220:
            period, jump_end, shoot_end = 38, 20, 6
        elif x < 360:
            period, jump_end, shoot_end = 44, 28, 6
        elif x < 480:
            period, jump_end, shoot_end = 46, 30, 5
        else:
            period, jump_end, shoot_end = 44, 28, 4
        phase = frame % period
        if phase < shoot_end:
            inputs = ("RIGHT", "B", "X")
        elif phase < shoot_end + 3:
            inputs = ("RIGHT", "B")
        elif phase < shoot_end + 3 + jump_end:
            inputs = ("RIGHT", "B", "A")
        else:
            inputs = ("RIGHT", "B")
        state = hold(session, 1, *inputs, reason=f"{label}_ridge")
        if state.room_id == ROOM_RISING_TIDE:
            break
    else:
        state = session.state
        raise TimeoutError(
            f"{label}: right green Super door missed before room "
            f"0x{ROOM_RISING_TIDE:04X}; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"door_transition={state.door_transition} max_x={max_x} "
            f"min_y={min_y} high_reached={high_reached} "
            f"door_reached={door_reached} "
            f"supers={state.super_missiles} selected={state.selected_item}"
        )

    return wait_ordinary_room(
        session,
        ROOM_RISING_TIDE,
        settle_frames=_CATH_TO_RISING_SETTLE_FRAMES,
        label=label,
    )


__all__ = [
    "play_business_to_cathedral_entrance",
    "play_cathedral_entrance_to_cathedral",
    "play_cathedral_to_rising_tide",
]

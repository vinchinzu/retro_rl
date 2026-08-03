"""Pure-first controllers for the K4 Business-to-Bubble Norfair path.

Business Center → Frog Save is the accepted K4.0 continuous extension (save
milestone). First Bubble visit is **Cathedral climb** (no Speed). Frog
Speedway is a post-Speed shortcut only (Boost Blocks need Speed Booster).
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
from super_metroid.routes.kpdr.rooms import (
    ROOM_BAT_CAVE,
    ROOM_BUBBLE,
    ROOM_BUSINESS,
    ROOM_CATHEDRAL,
    ROOM_CATHEDRAL_ENTRANCE,
    ROOM_FROG_SAVE,
    ROOM_FROG_SPEEDWAY,
    ROOM_RISING_TIDE,
    ROOM_SPEED,
    ROOM_UPPER_NORFAIR_FARM,
)
from super_metroid.routes.runtime import ControllerSession


# Bubble Mountain climb lives in bubble_mountain.py (re-exported for compat).
from super_metroid.routes.kpdr.bubble_mountain import (
    BUBBLE_PHASE_C_X_MIN,
    BUBBLE_PHASE_C_Y_MAX,
    BUBBLE_PHASE_C_Y_MIN,
    BUBBLE_PHASE_D_X,
    BUBBLE_PHASE_D_Y,
    BubblePhaseStop,
    bubble_phase_c_usable_right_contact,
    bubble_phase_d_top_band,
    play_bubble_to_bat_cave,
)


_MAX_SCAFFOLD_FRAMES = 240
_ELEVATOR_Y = 680
_FLOOR_Y_MIN = 1405
# Top-right Cathedral door is block [15, 55] → pixel y ≈ 880 (screen 3 of 7).
# Keep the band tight: y>900 is a lower shelf that falls past the door lip.
_CATHEDRAL_DOOR_Y_MIN = 840
_CATHEDRAL_DOOR_Y_MAX = 900
_CATHEDRAL_DOOR_BAND_FRAMES = 500
_CATHEDRAL_DOOR_FRAMES = 400
_CATHEDRAL_SETTLE_FRAMES = 320
_STANDING_POSES = frozenset({1, 2, 9, 10, 25, 26, 27, 28, 37, 38, 137, 138})
# Grounded poses only for the Cathedral door ledge (exclude knockback 137/138).
_CATHEDRAL_LEDGE_POSES = frozenset({1, 2, 9, 10, 25, 26, 27, 28, 37, 38})
_FROG_SPEEDWAY_DOOR_FRAMES = 400
_FROG_SPEEDWAY_SETTLE_FRAMES = 320
# Frog Speedway is an 8-screen horizontal tunnel (left entry → right farm door).
# Continuous loadout has no Speed; mid-room Boost Blocks may stop progress.
_SPEEDWAY_TO_FARM_DOOR_FRAMES = 1100
_SPEEDWAY_TO_FARM_SETTLE_FRAMES = 320
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
_CATH_LEDGE_POSES = frozenset({1, 2, 9, 10, 25, 26, 27, 28, 37, 38})
# Rising Tide (0xAFA3) is 5×1 screens (80×16 blocks). Left lip spawn ≈ (39, 139).
# Right blue door is block [63, 7] → pixel ≈ (1008, 112). Platforming cross with
# charged Hi-Jumps when low; continuous RIGHT+B+X door pressure from x≥930
# while staying on the door ledge (y≤170). No Super required. Caps: Morph,
# Bombs, Missiles, Supers, Hi-Jump, Varia — no Speed.
_RISING_CROSS_FRAMES = 5000
_RISING_TO_BUBBLE_SETTLE_FRAMES = 320
def _scaffold_exit(
    session: ControllerSession,
    *,
    entry_room: int,
    target_room: int,
    label: str,
    face: str = "RIGHT",
) -> SuperMetroidState:
    """Run a bounded placeholder toward a door and report a useful failure."""
    require_room(session, entry_room, label)

    # TODO: replace placeholder with room geometry (one pure card per hop).
    for _ in range(_MAX_SCAFFOLD_FRAMES):
        state = hold(session, 1, face, "B", reason=f"{label}_scaffold")
        if state.room_id == target_room:
            return state

    state = session.state
    raise TimeoutError(
        f"{label}: scaffold timeout before room 0x{target_room:04X}; "
        f"room=0x{state.room_id:04X} pose={state.pose} "
        f"xy=({state.samus_x},{state.samus_y})"
    )


def play_business_to_frog_save(session: ControllerSession) -> SuperMetroidState:
    """Business Center elevator → Frog Savestation through the blue door.

    The accepted Business checkpoint is still riding the arriving elevator.
    Let it settle at ``y=680``, snake down the central shaft to the floor, and
    beam-shot the right-hand Frog door at the floor lip.  Two integrity-green
    power-on runs compose this controller into the accepted Frog Save tip.
    """
    label = "business_to_frog_save"
    require_room(session, ROOM_BUSINESS, label)

    # The previous Warehouse elevator exit returns ordinary gameplay while
    # Samus is still descending (pose 155).  Require a stable landing rather
    # than treating the room-id change as an immediately playable source.
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

    # Descend through the staggered Business platforms without trying to
    # recreate the upward Warehouse climb.  The 70f direction swaps avoid
    # pinning either wall and naturally land on the Frog-door floor band.
    for frame in range(650):
        state = session.state
        if (
            state.samus_y >= _FLOOR_Y_MIN
            and state.velocity_y == 0
            and state.pose in _STANDING_POSES
        ):
            break
        buttons = ("LEFT", "B") if (frame // 70) % 2 == 0 else ("RIGHT", "B")
        hold(session, 1, *buttons, reason=f"{label}_descend")
    else:
        raise TimeoutError(f"{label}: floor band missed: {session.state}")

    # It is a closed blue door in this source; select beam and shoot while
    # running right so the transition starts as soon as Samus reaches its lip.
    select_weapon(session, 0)
    for _ in range(400):
        state = hold(session, 1, "RIGHT", "B", "X", reason=f"{label}_door")
        if state.room_id == ROOM_FROG_SAVE:
            break
    else:
        raise TimeoutError(f"{label}: Frog door missed: {session.state}")

    return wait_ordinary_room(
        session,
        ROOM_FROG_SAVE,
        settle_frames=320,
        label=label,
    )


def play_frog_save_to_speedway(session: ControllerSession) -> SuperMetroidState:
    """Frog Savestation right door → ordinary Frog Speedway.

    The accepted Frog checkpoint settles on the left side of the short save
    room.  The central save-tube blocks a flat run, so clear its two sides with
    separated Hi-Jump pulses before continuing to the blue right door.
    """
    label = "frog_save_to_speedway"
    require_room(session, ROOM_FROG_SAVE, label)

    select_weapon(session, 0)
    for frame in range(_FROG_SPEEDWAY_DOOR_FRAMES):
        inputs = ("RIGHT", "B", "X")
        if 30 <= frame < 40 or 90 <= frame < 100:
            inputs += ("A",)
        state = hold(session, 1, *inputs, reason=f"{label}_door")
        if state.room_id == ROOM_FROG_SPEEDWAY:
            break
    else:
        state = session.state
        raise TimeoutError(
            f"{label}: right door missed before room "
            f"0x{ROOM_FROG_SPEEDWAY:04X}; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"door_transition={state.door_transition}"
        )

    return wait_ordinary_room(
        session,
        ROOM_FROG_SPEEDWAY,
        settle_frames=_FROG_SPEEDWAY_SETTLE_FRAMES,
        label=label,
    )


def play_speedway_to_farm(session: ControllerSession) -> SuperMetroidState:
    """Frog Speedway left entry → ordinary Upper Norfair Farm via right door.

    Continuous-like source spawns on the left of the long horizontal tunnel
    (x≈39).  Run and beam-shot the blue right door into ``0xAF72``.  Mid-room
    Boost Blocks normally need Speed Booster; this controller is pure-first
    **without** a Speed grant — if blocked, timeout reports pose/xy for the
    residual.
    """
    label = "speedway_to_farm"
    require_room(session, ROOM_FROG_SPEEDWAY, label)

    select_weapon(session, 0)
    max_x = session.state.samus_x
    for _ in range(_SPEEDWAY_TO_FARM_DOOR_FRAMES):
        state = hold(session, 1, "RIGHT", "B", "X", reason=f"{label}_door")
        if state.samus_x > max_x:
            max_x = state.samus_x
        if state.room_id == ROOM_UPPER_NORFAIR_FARM:
            break
    else:
        state = session.state
        # Mid-room Boost Blocks (~x=795 from left entry) stop progress without
        # Speed Booster; report max_x so residuals can name the lock.
        raise TimeoutError(
            f"{label}: right door missed before room "
            f"0x{ROOM_UPPER_NORFAIR_FARM:04X}; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"door_transition={state.door_transition} max_x={max_x}"
            + (
                " (boost-block stall; no Speed)"
                if max_x <= 820 and state.samus_x <= 820
                else ""
            )
        )

    return wait_ordinary_room(
        session,
        ROOM_UPPER_NORFAIR_FARM,
        settle_frames=_SPEEDWAY_TO_FARM_SETTLE_FRAMES,
        label=label,
    )


def play_farm_to_bubble(session: ControllerSession) -> SuperMetroidState:
    """Scaffold Upper Norfair Farm ``0xAF72`` → Bubble Mountain ``0xACB3``.

    Post-Speed farm entry only (see ``speedway_to_farm`` requires Speed).
    """
    return _scaffold_exit(
        session,
        entry_room=ROOM_UPPER_NORFAIR_FARM,
        target_room=ROOM_BUBBLE,
        label="farm_to_bubble",
    )


def play_frog_save_to_business(session: ControllerSession) -> SuperMetroidState:
    """Scaffold Frog Save left door reverse → Business Center."""
    return _scaffold_exit(
        session,
        entry_room=ROOM_FROG_SAVE,
        target_room=ROOM_BUSINESS,
        label="frog_save_to_business",
        face="LEFT",
    )


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
    for _ in range(_CATHEDRAL_DOOR_FRAMES):
        state = hold(session, 1, "RIGHT", "B", "X", reason=f"{label}_door")
        if state.room_id == ROOM_CATHEDRAL_ENTRANCE:
            break
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


def play_cathedral_entrance_to_cathedral(
    session: ControllerSession,
) -> SuperMetroidState:
    """Cathedral Entrance left spawn → ordinary Cathedral via right red Super door.

    CATH-01 pure successor lands near the left blue lip (x≈39 / y≈139).  The
    upper ledge is a dead-end solid at x≈91 — bomb through the left morph-tunnel
    floor, cross the bottom, Hi-Jump climb mid platforms toward the right red
    Super door (block ``[47, 7]``), open it, and settle ordinary ``0xA788``.
    """
    label = "cathedral_entrance_to_cathedral"
    require_room(session, ROOM_CATHEDRAL_ENTRANCE, label)

    for _ in range(40):
        state = hold(session, 1, reason=f"{label}_land")
        if state.velocity_y == 0 and state.pose in _STANDING_POSES:
            break

    # --- Phase 1: bomb-drop through left morph-tunnel floor ---
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

    # --- Phase 2: floor cross to climb start (~x 620 floor band) ---
    select_weapon(session, 0)
    max_x = session.state.samus_x
    min_y = session.state.samus_y
    for frame in range(_CATH_ENTRANCE_FLOOR_FRAMES):
        state = session.state
        if state.room_id == ROOM_CATHEDRAL:
            break
        if state.pose in (137, 138):
            hold(session, 8, reason=f"{label}_kb")
            continue
        if (
            state.samus_x >= 620
            and state.velocity_y == 0
            and state.samus_y >= 380
            and state.pose in _CATHEDRAL_LEDGE_POSES
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

    # --- Phase 3: climb mid platforms (x 560–680) + Super door ---
    mid_ground = frozenset({1, 2, 9, 10})
    mid_landed = False
    door_reached = False
    y_lo, y_hi = _CATH_ENTRANCE_MID_Y
    for frame in range(_CATH_ENTRANCE_CLIMB_FRAMES):
        state = session.state
        if state.room_id == ROOM_CATHEDRAL:
            break
        max_x = max(max_x, state.samus_x)
        min_y = min(min_y, state.samus_y)

        if state.pose in (137, 138):
            hold(session, 12, reason=f"{label}_kb")
            continue

        # Door band: Super pulses + enter.
        if (
            state.samus_y <= _CATH_ENTRANCE_DOOR_Y
            and state.samus_x >= _CATH_ENTRANCE_DOOR_X
        ):
            door_reached = True
            if state.selected_item != 2:
                select_weapon(session, 2)
            phase = frame % 28
            if phase < 5:
                inputs = ("RIGHT", "X")
            elif phase < 12:
                inputs = ("RIGHT",)
            elif phase < 20:
                inputs = ("RIGHT", "B")
            else:
                inputs = ("RIGHT", "B", "A")
            state = hold(session, 1, *inputs, reason=f"{label}_door")
            if state.room_id == ROOM_CATHEDRAL:
                break
            continue

        # High near-door: push right onto the door ledge.
        if state.samus_y <= 220 and state.samus_x >= 600:
            if state.selected_item != 2:
                select_weapon(session, 2)
            phase = frame % 24
            if phase < 6:
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
            and state.pose in mid_ground
        ):
            mid_landed = True
            hold(session, 8, reason=f"{label}_mid_plant")
            for _ in range(22):
                hold(session, 1, "A", reason=f"{label}_mid_charge")
            for _ in range(90):
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
        if state.velocity_y == 0 or state.pose in (137, 138):
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
        if state.pose in (137, 138):
            prefer = "LEFT" if state.samus_x >= 700 else "RIGHT"
            if state.selected_item != 0:
                select_weapon(session, 0)
            for _ in range(6):
                hold(session, 1, prefer, "B", "X", reason=f"{label}_kb_clear")
            for _ in range(24):
                st = hold(
                    session, 1, prefer, "B", "A", reason=f"{label}_kb_spin"
                )
                if st.room_id == ROOM_RISING_TIDE:
                    break
                if st.pose not in (137, 138) and abs(st.samus_x - state.samus_x) > 2:
                    break
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
            phase = frame % 40
            if phase < 4:
                inputs = ("RIGHT", "X")
            elif phase < 20:
                inputs = ()
            elif phase < 28:
                inputs = ("RIGHT",)
            elif phase < 34:
                inputs = ("RIGHT", "B")
            else:
                inputs = ("RIGHT", "B", "A")
            state = hold(session, 1, *inputs, reason=f"{label}_door")
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


def play_rising_tide_to_bubble(session: ControllerSession) -> SuperMetroidState:
    """Rising Tide left lip → ordinary Bubble Mountain via right blue door.

    CATH-03 pure successor lands near the left blue lip (x≈39 / y≈139).  Cross
    the 5-screen heated lava corridor with charged Hi-Jumps when low, knockback
    spin-escapes, then continuous beam-shot pressure on the right blue door
    (block ``[63, 7]`` / ≈x1008 y112) — plant-and-shoot is unreliable here;
    frog-style RIGHT+B+X while keeping door-ledge altitude works.  Settle
    ordinary ``0xACB3`` (Bubble node 3, mid-left).  Caps: Morph, Bombs,
    Missiles, Supers, Hi-Jump, Varia — **no Speed**.
    """
    label = "rising_tide_to_bubble"
    require_room(session, ROOM_RISING_TIDE, label)

    for _ in range(40):
        state = hold(session, 1, reason=f"{label}_land")
        if state.velocity_y == 0 and state.pose in _STANDING_POSES:
            break
    unmorph(session)
    select_weapon(session, 0)

    max_x = session.state.samus_x
    min_y = session.state.samus_y
    door_reached = False
    stuck_frames = 0
    last_x = session.state.samus_x

    for frame in range(_RISING_CROSS_FRAMES):
        state = session.state
        if state.room_id == ROOM_BUBBLE:
            break

        max_x = max(max_x, state.samus_x)
        min_y = min(min_y, state.samus_y)
        if abs(state.samus_x - last_x) <= 1:
            stuck_frames += 1
        else:
            stuck_frames = 0
            last_x = state.samus_x

        # Knockback / contact: spin-escape right (assist energy can stunlock).
        if state.pose in (137, 138):
            for _ in range(6):
                hold(session, 1, "RIGHT", "B", reason=f"{label}_kb_run")
            for _ in range(20):
                st = hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_kb_spin")
                if st.room_id == ROOM_BUBBLE:
                    break
            stuck_frames = 0
            last_x = session.state.samus_x
            continue

        # Door approach band (x≥930): continuous shoot-run; keep door altitude.
        # Falling under the door platform (y>170) and walking past the shell
        # without transition is the common miss — climb back, then re-pressure.
        if state.samus_x >= 930:
            door_reached = True
            if state.selected_item != 0:
                select_weapon(session, 0)
            if state.samus_y > 170:
                if state.samus_x > 1040:
                    hold(session, 1, "LEFT", "B", reason=f"{label}_under_back")
                elif (
                    state.velocity_y == 0
                    and state.pose in _STANDING_POSES
                ):
                    for _ in range(14):
                        hold(session, 1, "A", reason=f"{label}_door_charge")
                    for _ in range(40):
                        st = hold(
                            session, 1, "RIGHT", "B", "A", reason=f"{label}_door_up"
                        )
                        if st.room_id == ROOM_BUBBLE:
                            break
                else:
                    hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_under_hop")
                continue
            phase = frame % 16
            if phase < 8:
                inputs = ("RIGHT", "B", "X")
            elif phase < 12:
                inputs = ("RIGHT", "B", "A")
            else:
                inputs = ("RIGHT", "B")
            state = hold(session, 1, *inputs, reason=f"{label}_door")
            if state.room_id == ROOM_BUBBLE:
                break
            continue

        # Mid-room low: charged Hi-Jump to stay on platforms above lava.
        if (
            state.velocity_y == 0
            and state.samus_y > 150
            and state.pose in _STANDING_POSES
        ):
            for _ in range(12):
                hold(session, 1, "A", reason=f"{label}_charge")
            for _ in range(32):
                st = hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_hj")
                if st.room_id == ROOM_BUBBLE:
                    break
            continue

        # Stuck on a ledge / enemy body: short reverse then re-commit right.
        if stuck_frames > 40:
            for _ in range(10):
                hold(session, 1, "LEFT", "B", reason=f"{label}_unstick_back")
            for _ in range(10):
                hold(session, 1, "A", reason=f"{label}_unstick_charge")
            for _ in range(35):
                hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_unstick_jump")
            stuck_frames = 0
            last_x = session.state.samus_x
            continue

        # Default cross: run-jump cadence; occasional beam for Sovas.
        if state.selected_item != 0:
            select_weapon(session, 0)
        phase = frame % 32
        if phase < 3:
            inputs = ("RIGHT", "B", "X")
        elif phase < 22:
            inputs = ("RIGHT", "B", "A")
        else:
            inputs = ("RIGHT", "B")
        state = hold(session, 1, *inputs, reason=f"{label}_cross")
        if state.room_id == ROOM_BUBBLE:
            break
    else:
        state = session.state
        raise TimeoutError(
            f"{label}: right blue door missed before room "
            f"0x{ROOM_BUBBLE:04X}; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"door_transition={state.door_transition} max_x={max_x} "
            f"min_y={min_y} door_reached={door_reached} "
            f"selected={state.selected_item}"
        )

    return wait_ordinary_room(
        session,
        ROOM_BUBBLE,
        settle_frames=_RISING_TO_BUBBLE_SETTLE_FRAMES,
        label=label,
    )



__all__ = [
    "BUBBLE_PHASE_C_X_MIN",
    "BUBBLE_PHASE_C_Y_MAX",
    "BUBBLE_PHASE_C_Y_MIN",
    "BUBBLE_PHASE_D_X",
    "BUBBLE_PHASE_D_Y",
    "BubblePhaseStop",
    "ROOM_BAT_CAVE",
    "ROOM_BUBBLE",
    "ROOM_BUSINESS",
    "ROOM_CATHEDRAL",
    "ROOM_CATHEDRAL_ENTRANCE",
    "ROOM_FROG_SAVE",
    "ROOM_FROG_SPEEDWAY",
    "ROOM_RISING_TIDE",
    "ROOM_SPEED",
    "ROOM_UPPER_NORFAIR_FARM",
    "bubble_phase_c_usable_right_contact",
    "bubble_phase_d_top_band",
    "play_bubble_to_bat_cave",
    "play_business_to_cathedral_entrance",
    "play_business_to_frog_save",
    "play_cathedral_entrance_to_cathedral",
    "play_cathedral_to_rising_tide",
    "play_farm_to_bubble",
    "play_frog_save_to_business",
    "play_frog_save_to_speedway",
    "play_rising_tide_to_bubble",
    "play_speedway_to_farm",
]

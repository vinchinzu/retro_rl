"""Pure-first controllers for the K4 Business-to-Bubble Norfair path.

Business Center → Frog Save is the accepted K4.0 continuous extension (save
milestone). First Bubble visit is **Cathedral climb** (no Speed). Frog
Speedway is a post-Speed shortcut only (Boost Blocks need Speed Booster).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
# Bubble Mountain (0xACB3) is 2×4 screens. Entry node 3 mid-left ≈ (39–60, 634).
# Climb to top-right green Super door node 7 (block [31, 7] ≈ x496 y112).
# Wrong-door traps: left y≈624 Rising Tide, y≈368 Save, y≈112 Missiles Super;
# right y≈368 Single Chamber. Maprando junction→top-right: Walljump with HiJump
# from save-door platform (left mid) — run jump to cavity right wall, two WJ.
_BUBBLE_LOWER_FRAMES = 3500
_BUBBLE_MID_REPIN_FRAMES = 900
_BUBBLE_MID_FRAMES = 5500
_BUBBLE_DOOR_FRAMES = 900
_BUBBLE_TO_BAT_SETTLE_FRAMES = 320
_BUBBLE_MID_Y = 400
_BUBBLE_TOP_Y = 200
_BUBBLE_TOP_X = 300
_BUBBLE_DOOR_X = 420
_BUBBLE_GROUND = frozenset({1, 2, 9, 10})
# Mid-iso handoff (post_bubble_mid_climb_pure: pose=26 x≈98 y≈374 |vy|≈1).
_BUBBLE_STAND_PIN = frozenset({1, 2, 9, 10, 25, 26, 27, 28})
# R2 mid open-loop + R3 standing mid re-pin before launch. Hard-cap x to avoid
# Single Chamber outer-wall height trap (~y360 at x≥400).
_BUBBLE_CAVITY_X_MAX = 395
_BUBBLE_MID_STAND_X = (77, 160)
_BUBBLE_PEAK_CROSS_Y = 250
_BUBBLE_ALT_WJ_PERIOD = 12
# R5 lower-left ledge path: solid shelves from place-grid recon (entry floor
# ~y651 → save-door pin ~y370). Multi-hop charged HJ along left column; dir
# bias alone favored cavity mid-right and never reconstructed mid-iso pin.
_BUBBLE_FLOOR_SHELF_X = 108
_BUBBLE_LOWER_SHELVES: tuple[tuple[int, int], ...] = (
    (120, 560),
    (110, 515),
    (100, 475),
    (90, 450),
    (95, 430),
    (105, 370),
)
# R6 solid save-door lip (place-grid jumpable; mid float at y~370 is NOT).
# Idle from mid-iso drifts to ~(69,427) pose 1/2; charged HJ from there
# reaches min_y≈228–260. Launch only from this lip, not unstable mid pin.
_BUBBLE_LIP_X = (65, 100)
_BUBBLE_LIP_Y = (410, 450)
# R7 second-hop / peak-cross: after lip HJ hits height class (y≤280), re-seat
# and hop toward right-structure shelves that one-hop to top band. Place-grid
# solids: (384,363) (368,331) (352,283) (336,219) — each charged-HJ → top.
# Uncharged run-jump from lip can reseat mid-nub ~(140–175, 270–295).
_BUBBLE_HEIGHT_CLASS_Y = 280
_BUBBLE_MID_RESEAT_Y = 320
_BUBBLE_RIGHT_SHELF_X = 300
_BUBBLE_RIGHT_SHELF_Y = 390  # R9: include lower shelf landings (~y379)
# R9 open-loop right-structure land then top: once height class, period-8
# WJ from x≥250 (into 2 / bounce 2) or no-A drop onto shelf class.
# Grounded shelf → LEFT charged HJ (RIGHT hits SC outer-wall trap).
# Recon: air (360,320) period-8 WJ → top; place shelf LEFT HJ → top.
_BUBBLE_RIGHT_WJ_PERIOD = 8
_BUBBLE_RIGHT_WJ_INTO = 2
_BUBBLE_RIGHT_WJ_BOUNCE = 2
# R10 mid-high approach: engage open-loop WJ/drop while y≤450 (R9 used 400
# — natural path first hits x≥250 only after y≈400+, so R9 skipped WJ on
# the peak arc). Place: period-8 WJ from air (360,y≤370) → top.
_BUBBLE_MIDHIGH_Y = 450
# Lip launch timings (R6/R9/R10 proven height class min_y=260). R11 full
# maprando walk-left+run regressed pure min_y~365 — keep charge/spin band.
_BUBBLE_LIP_CHARGE = 12
_BUBBLE_LIP_SPIN = 44
_BUBBLE_LIP_EXTEND = 70
# Phase ladder (docs/tasks/SM-K4.4-PHASE-LADDER.md): usable right contact is
# the R11 bottleneck — not thrash max_x on the cavity floor.
BUBBLE_PHASE_C_X_MIN = 300
BUBBLE_PHASE_C_Y_MAX = 430
BUBBLE_PHASE_C_Y_MIN = 200
BUBBLE_PHASE_D_X = _BUBBLE_TOP_X
BUBBLE_PHASE_D_Y = _BUBBLE_TOP_Y
_BUBBLE_START_PHASES = frozenset({"auto", "full", "climb", "door"})


class BubblePhaseStop(Exception):
    """Diagnostic early exit when a pure probe stops at a phase pin.

    Probe CLI may treat this as success for capture/recon only — never as
    hop GREEN to Bat Cave / continuous evidence.
    """

    def __init__(
        self,
        phase: str,
        state: SuperMetroidState,
        *,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self.phase = phase
        self.state = state
        self.metrics = dict(metrics or {})
        super().__init__(
            f"bubble_phase_stop:{phase} room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"vx={state.velocity_x} vy={state.velocity_y}"
        )


def bubble_phase_c_usable_right_contact(st: SuperMetroidState) -> bool:
    """Phase C: usable right-structure contact at height (not floor thrash)."""
    return (
        int(st.room_id) == ROOM_BUBBLE
        and BUBBLE_PHASE_C_X_MIN <= int(st.samus_x) <= _BUBBLE_CAVITY_X_MAX
        and BUBBLE_PHASE_C_Y_MIN <= int(st.samus_y) <= BUBBLE_PHASE_C_Y_MAX
    )


def bubble_phase_d_top_band(st: SuperMetroidState) -> bool:
    """Phase D: top band before Super door."""
    return (
        int(st.room_id) == ROOM_BUBBLE
        and int(st.samus_y) <= BUBBLE_PHASE_D_Y
        and int(st.samus_x) >= BUBBLE_PHASE_D_X
    )


def _maybe_dump_bubble_phase_c(
    session: ControllerSession,
    state: SuperMetroidState,
    dump_path: Path | None,
    *,
    dumped: list[bool],
) -> None:
    """Save first Phase-C pin when probe session exposes ``env``."""
    if dump_path is None or dumped[0]:
        return
    if not bubble_phase_c_usable_right_contact(state):
        return
    env = getattr(session, "env", None)
    if env is None:
        return
    from super_metroid.dev.common import save_dev_state

    save_dev_state(env, dump_path)
    dumped[0] = True


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

    return wait_ordinary_room(
        session,
        ROOM_CATHEDRAL_ENTRANCE,
        settle_frames=_CATHEDRAL_SETTLE_FRAMES,
        label=label,
    )


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
    the upper ridge with short run-jumps (pillars / gaps; no Speed), then drop
    to the **lower-right** green Super door near lava (≈x700–730 / y350–380),
    open it, and settle ordinary ``0xAFA3``.  Caps: Morph, Bombs, Missiles,
    Supers, Hi-Jump, Varia.
    """
    label = "cathedral_to_rising_tide"
    require_room(session, ROOM_CATHEDRAL, label)

    for _ in range(40):
        state = hold(session, 1, reason=f"{label}_land")
        if state.velocity_y == 0 and state.pose in _STANDING_POSES:
            break
    unmorph(session)
    select_weapon(session, 0)

    # Opening hop off the left lip toward the first ridge structure.
    for _ in range(15):
        hold(session, 1, "RIGHT", "B", reason=f"{label}_open_run")
    for _ in range(22):
        hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_open_jump")
    for _ in range(40):
        state = hold(session, 1, reason=f"{label}_open_fall")
        if state.velocity_y == 0 or state.pose in (137, 138):
            break

    max_x = session.state.samus_x
    min_y = session.state.samus_y
    door_reached = False
    high_reached = False
    stuck_frames = 0
    last_x = session.state.samus_x

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

        # Knockback / contact-damage: multi-frame spin-escape (never idle-plant).
        # Assist energy refill can stunlock pose 137/138 on continuous contact.
        if state.pose in (137, 138) or (
            stuck_frames > 45 and 560 <= state.samus_x <= 620
        ):
            prefer = "LEFT" if state.samus_x >= 700 else "RIGHT"
            for _ in range(8):
                hold(session, 1, prefer, "B", reason=f"{label}_kb_run")
            for _ in range(28):
                st = hold(session, 1, prefer, "B", "A", reason=f"{label}_kb_spin")
                if st.room_id == ROOM_RISING_TIDE:
                    break
                if st.pose not in (137, 138) and st.samus_x != state.samus_x:
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
            if phase < 3:
                inputs = ("RIGHT", "X")
            elif phase < 22:
                inputs = ()
            elif phase < 30:
                inputs = ("RIGHT",)
            elif phase < 36:
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

        # Mid-room fall: prefer RIGHT toward door; recover height only if left.
        if state.samus_y > _CATH_FALL_Y and state.samus_x < _CATH_DOOR_X:
            if state.selected_item != 0:
                select_weapon(session, 0)
            if state.samus_x < 80:
                dir_h = "RIGHT"
            else:
                dir_h = "RIGHT" if (frame // 40) % 3 != 2 else "LEFT"
            phase = frame % 48
            if phase < 12:
                inputs = ("A",)
            elif phase < 36:
                inputs = (dir_h, "B", "A")
            else:
                inputs = (dir_h, "B", "X")
            hold(session, 1, *inputs, reason=f"{label}_recover")
            continue

        # Upper ridge cross: short run-jumps, periodic beam, stay high.
        if state.selected_item != 0:
            select_weapon(session, 0)
        x = state.samus_x
        # Cadence by x-band. Late gaps (~x450–560) need longer Hi-Jumps;
        # shortening on high crests was stranding max_x≈539.
        if x < 250:
            period, jump_end, shoot_end = 40, 22, 4
        elif x < 350:
            period, jump_end, shoot_end = 36, 18, 3
        elif x < 450:
            period, jump_end, shoot_end = 40, 24, 3
        else:
            period, jump_end, shoot_end = 42, 28, 3
        phase = frame % period
        if phase < shoot_end:
            inputs = ("RIGHT", "B", "X")
        elif phase < shoot_end + 4:
            inputs = ("RIGHT", "B")
        elif phase < shoot_end + 4 + jump_end:
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


def play_bubble_to_bat_cave(
    session: ControllerSession,
    *,
    start_phase: str = "auto",
    dump_phase_c: Path | str | None = None,
    stop_at_phase_c: bool = False,
) -> SuperMetroidState:
    """Bubble Mountain mid-left entry → ordinary Bat Cave via top-right Super door.

    Pure source is the CATH-04 successor at node 3 (≈x39–60 / y634).  Phases:
    (1 R5) lower-left ledge multi-hop (floor shelf → save-door pin band
    x∈[77,160] y≤400), (1.5 R3) standing mid re-pin settle, (2 R6) solid
    lip launch, (2b R7–R10) after height class (y≤280): mid-high open-loop
    into right air/shelf band (x≥340, y∈[280,370]), period-8 WJ or no-A
    shelf land, then LEFT charged HJ to top band (y≤200, x≥300) without
    leaving ``0xACB3``, (3) Super-open the top-right green door into
    ordinary ``0xB07A``.

    Phase ladder (docs/tasks/SM-K4.4-PHASE-LADDER.md):
    A mid pin · B height class · C usable right contact · D top · E Bat door.
    Dev-only kwargs (never continuous evidence):

    - ``start_phase="climb"``: skip lower / repin / lip launch; assume prior
      height class and enter right-structure climb (Phase-C handoff iteration).
    - ``dump_phase_c``: write first Phase-C save-state when ``session.env``
      exists (probe sessions).
    - ``stop_at_phase_c``: raise :class:`BubblePhaseStop` at first Phase C
      (capture/recon only).

    Caps: Morph, Bombs, Missiles, Supers (≥1), Hi-Jump, Varia — **no Speed**.
    Hard-avoid wrong doors (Rising Tide / Save / Missiles Super left; Single
    Chamber right mid).
    """
    label = "bubble_to_bat_cave"
    phase_key = (start_phase or "auto").strip().lower()
    if phase_key not in _BUBBLE_START_PHASES:
        raise ValueError(
            f"{label}: start_phase must be one of "
            f"{sorted(_BUBBLE_START_PHASES)}, got {start_phase!r}"
        )
    dump_path = Path(dump_phase_c) if dump_phase_c is not None else None
    phase_c_dumped = [False]
    phase_c_hit = False

    require_room(session, ROOM_BUBBLE, label)

    # Land: if already mid-iso pin class (isolation source), break immediately
    # with |vy|≤2 — a full 40f idle settle from vy≈1 drifts left/down off the
    # save-door platform (x~69 y~427) and forces a failed lower re-climb.
    # Climb-only / door-only starts: short settle only (handoff may be air).
    stand_lo, stand_hi = _BUBBLE_MID_STAND_X
    land_frames = 8 if phase_key in ("climb", "door") else 40
    for _ in range(land_frames):
        state = hold(session, 1, reason=f"{label}_land")
        if (
            abs(state.velocity_y) <= 2
            and state.pose in _BUBBLE_STAND_PIN
            and stand_lo <= state.samus_x <= stand_hi
            and state.samus_y <= _BUBBLE_MID_Y + 10
        ):
            break
        if state.velocity_y == 0 and state.pose in _STANDING_POSES:
            break
    unmorph(session)
    select_weapon(session, 0)

    max_x = session.state.samus_x
    min_y = session.state.samus_y
    mid_reached = False
    top_reached = False
    door_reached = False
    standing_mid_pinned = False
    launched = False

    def _track(state: SuperMetroidState) -> None:
        nonlocal max_x, min_y, mid_reached, top_reached, phase_c_hit
        max_x = max(max_x, state.samus_x)
        min_y = min(min_y, state.samus_y)
        if state.samus_y <= _BUBBLE_MID_Y and state.samus_x >= 90:
            mid_reached = True
        if bubble_phase_d_top_band(state):
            top_reached = True
        if bubble_phase_c_usable_right_contact(state):
            if not phase_c_hit:
                phase_c_hit = True
                _maybe_dump_bubble_phase_c(
                    session, state, dump_path, dumped=phase_c_dumped
                )
                if stop_at_phase_c:
                    raise BubblePhaseStop(
                        "C",
                        state,
                        metrics={
                            "max_x": max_x,
                            "min_y": min_y,
                            "mid_reached": mid_reached,
                            "top_reached": top_reached,
                            "phase_c_hit": True,
                            "dump_phase_c": str(dump_path) if dump_path else None,
                            "dumped": phase_c_dumped[0],
                        },
                    )

    def _avoid_wrong_door(state: SuperMetroidState) -> bool:
        """Hard-steer away from side doors; return True if a steer was applied."""
        x, y = state.samus_x, state.samus_y
        # Left doors: Rising Tide ~y624, Save ~y368, Missiles Super ~y112.
        if x < 55:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_avoid_left")
            return True
        # Right mid Single Chamber ~y368 / x≈496.
        if x > 470 and 300 <= y <= 430:
            hold(session, 1, "LEFT", "B", reason=f"{label}_avoid_sc")
            return True
        return False

    # --- Phase 1 (R5): lower-left ledge path → save-door pin ---
    # One change vs R4: scripted multi-hop along place-grid solid shelves in
    # the left column (not HJ dir bias). Recon: natural entry walk-to-floor
    # (~x108,y651) then hops (120,560)→…→(105,370) pins 5/5; dir bias alone
    # stalled cavity mid-right (full pure min_y≈364 unpinned).

    def _on_mid_iso_pin(st: SuperMetroidState) -> bool:
        return (
            abs(st.velocity_y) <= 2
            and st.pose in _BUBBLE_STAND_PIN
            and stand_lo <= st.samus_x <= stand_hi
            and st.samus_y <= _BUBBLE_MID_Y + 10
        )

    # climb/door: skip lower + repin. auto: skip lower when already mid pin.
    skip_lower = phase_key in ("climb", "door") or _on_mid_iso_pin(session.state)
    skip_repin = phase_key in ("climb", "door")
    skip_launch = phase_key == "climb"
    skip_to_door = phase_key == "door"

    if skip_to_door:
        mid_reached = True
        standing_mid_pinned = True
        launched = True
        top_reached = bubble_phase_d_top_band(session.state) or (
            session.state.samus_y <= _BUBBLE_TOP_Y + 40
            and session.state.samus_x >= _BUBBLE_TOP_X - 40
        )
        _track(session.state)
    elif skip_lower:
        mid_reached = True
        _track(session.state)
    else:
        # 1a: walk onto lower-left floor shelf (solid place-grid band ~y651).
        for frame in range(140):
            state = session.state
            if state.room_id != ROOM_BUBBLE:
                break
            _track(state)
            if _on_mid_iso_pin(state):
                mid_reached = True
                break
            if _avoid_wrong_door(state):
                continue
            if state.pose in (137, 138):
                hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_floor_kb")
                continue
            if (
                state.samus_x >= _BUBBLE_FLOOR_SHELF_X
                and abs(state.velocity_y) <= 1
                and state.pose in _BUBBLE_GROUND
            ):
                break
            if frame % 12 < 3:
                hold(session, 1, "RIGHT", "B", "X", reason=f"{label}_floor_shot")
            else:
                hold(session, 1, "RIGHT", "B", reason=f"{label}_floor_walk")
        for _ in range(20):
            state = hold(session, 1, reason=f"{label}_floor_settle")
            _track(state)
            if state.pose in _BUBBLE_GROUND and abs(state.velocity_y) <= 1:
                break
            if state.pose in (137, 138):
                hold(session, 1, "A", reason=f"{label}_floor_kb_clear")

        # 1b: multi-hop along left-column shelves to save-door pin.
        shelf_i = 0
        for frame in range(_BUBBLE_LOWER_FRAMES):
            state = session.state
            if state.room_id != ROOM_BUBBLE:
                break
            _track(state)
            if _on_mid_iso_pin(state):
                mid_reached = True
                break
            if _avoid_wrong_door(state):
                continue
            if state.pose in (137, 138):
                for _ in range(8):
                    hold(
                        session, 1, "RIGHT", "B", "A", reason=f"{label}_lower_kb"
                    )
                continue
            if state.pose in (27, 28):
                hold(session, 1, "UP", reason=f"{label}_lower_unmorph")
                continue

            x = state.samus_x
            y = state.samus_y
            # Pull back from cavity mid-right shelves (R3/R4 failure class).
            if x > 250 and y > _BUBBLE_MID_Y:
                hold(session, 1, "LEFT", "B", reason=f"{label}_lower_cavity")
                continue

            shelves = _BUBBLE_LOWER_SHELVES
            while (
                shelf_i < len(shelves) - 1
                and y <= shelves[shelf_i][1] + 12
                and abs(x - shelves[shelf_i][0]) < 40
            ):
                shelf_i += 1
            tx, ty = shelves[shelf_i]

            grounded = (
                abs(state.velocity_y) <= 1 and state.pose in _BUBBLE_STAND_PIN
            )
            if grounded and y > ty + 20:
                for _ in range(8):
                    hold(session, 1, "A", reason=f"{label}_lower_charge")
                if x < tx - 10:
                    dir_h = "RIGHT"
                elif x > tx + 10:
                    dir_h = "LEFT"
                else:
                    dir_h = "RIGHT" if x < 115 else "LEFT"
                hop = 28 if (y - ty) > 80 else 36
                for _ in range(hop):
                    state = hold(
                        session, 1, dir_h, "B", "A", reason=f"{label}_lower_hop"
                    )
                    _track(state)
                    if state.room_id != ROOM_BUBBLE:
                        break
                    if _on_mid_iso_pin(state):
                        mid_reached = True
                        break
                if mid_reached or state.room_id != ROOM_BUBBLE:
                    break
                continue

            if grounded and y <= ty + 20:
                if abs(x - tx) > 8:
                    dir_h = "RIGHT" if x < tx else "LEFT"
                    hold(session, 1, dir_h, "B", reason=f"{label}_lower_align")
                else:
                    hold(session, 1, reason=f"{label}_lower_idle")
                continue

            # Air: steer toward current shelf target, keep left column.
            if x < tx - 5:
                dir_h = "RIGHT"
            elif x > tx + 5:
                dir_h = "LEFT"
            else:
                dir_h = "LEFT" if x > 120 else "RIGHT"
            hold(session, 1, dir_h, "B", "A", reason=f"{label}_lower_air")

    # --- Phase 1.5 (R3): standing mid re-pin before open-loop launch ---
    # After lower (R5 ledge path), settle / recover into mid-iso handoff
    # before open-loop. Budget is short; phase 2 still launches from grounded
    # mid if pin misses (R2 fallback). Skip on climb/door handoff starts.
    if not skip_repin:
        for frame in range(_BUBBLE_MID_REPIN_FRAMES):
            state = session.state
            if state.room_id != ROOM_BUBBLE:
                break
            _track(state)
            if (
                state.samus_y <= _BUBBLE_TOP_Y
                and state.samus_x >= _BUBBLE_TOP_X
            ):
                top_reached = True
                standing_mid_pinned = True
                break
            if _avoid_wrong_door(state):
                continue
            if state.pose in (137, 138):
                for _ in range(10):
                    hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_repin_kb")
                continue

            x = state.samus_x
            y = state.samus_y
            if x > _BUBBLE_CAVITY_X_MAX and y > _BUBBLE_TOP_Y:
                hold(session, 1, "LEFT", "B", reason=f"{label}_repin_cap")
                continue

            if _on_mid_iso_pin(state):
                for _ in range(4):
                    state = hold(session, 1, reason=f"{label}_repin_settle")
                    _track(state)
                if state.room_id == ROOM_BUBBLE and _on_mid_iso_pin(state):
                    standing_mid_pinned = True
                    mid_reached = True
                    break
                continue

            # Below mid: charged HJ, prefer left column for save-door platform.
            if y > _BUBBLE_MID_Y + 10:
                if state.velocity_y == 0 and state.pose in _BUBBLE_GROUND:
                    dir_h = "RIGHT" if x < 140 else "LEFT"
                    for _ in range(10):
                        hold(session, 1, "A", reason=f"{label}_repin_charge")
                    for _ in range(36):
                        state = hold(
                            session, 1, dir_h, "B", "A", reason=f"{label}_repin_hj"
                        )
                        _track(state)
                        if state.room_id != ROOM_BUBBLE:
                            break
                        if _on_mid_iso_pin(state):
                            break
                    continue
                dir_h = "RIGHT" if x < 160 else "LEFT"
                hold(session, 1, dir_h, "B", "A", reason=f"{label}_repin_low_spin")
                continue

            # Mid height, wrong x: walk into save-door band.
            if state.velocity_y == 0 and state.pose in _BUBBLE_STAND_PIN:
                if x < stand_lo:
                    hold(session, 1, "RIGHT", "B", reason=f"{label}_repin_walk_r")
                elif x > stand_hi:
                    hold(session, 1, "LEFT", "B", reason=f"{label}_repin_walk_l")
                else:
                    hold(session, 1, reason=f"{label}_repin_idle")
                continue

            dir_h = "RIGHT" if x < stand_lo else ("LEFT" if x > stand_hi else "RIGHT")
            hold(session, 1, dir_h, "B", reason=f"{label}_repin_air")

    # --- Phase 2 (R6–R10): solid lip launch + mid-high right shelf → top ---
    # R6: do NOT launch from unstable mid-iso float (y~370). Drop/walk to
    # solid save-door lip, charged HJ. Recon: lip launch min_y≈228–260.
    # R7: once height class (y≤280), stop left-column thrash; peak-cross right.
    # R8: reactive shelf_drop (no metric advance).
    # R9: open-loop approach right structure, period-8 WJ / no-A drop, LEFT
    # shelf HJ. Hard-cap x<_BUBBLE_CAVITY_X_MAX (SC height trap).
    # R10 one-change: mid-high band (y≤450) so WJ/drop engage on the
    # natural fall arc. No lip run-up (pure height regress).
    # start_phase=climb: enter climb with sticky height_class (handoff assumes
    # prior peak); start_phase=door skips this block entirely.
    lip_lo, lip_hi = _BUBBLE_LIP_X
    lip_y_lo, lip_y_hi = _BUBBLE_LIP_Y

    def _on_launch_lip(st: SuperMetroidState) -> bool:
        return (
            abs(st.velocity_y) <= 1
            and st.pose in _BUBBLE_STAND_PIN
            and lip_lo <= st.samus_x <= lip_hi
            and lip_y_lo <= st.samus_y <= lip_y_hi
        )

    def _on_right_shelf(st: SuperMetroidState) -> bool:
        """Grounded on place-proven right-structure shelf class."""
        return (
            abs(st.velocity_y) <= 1
            and st.pose in _BUBBLE_STAND_PIN
            and st.samus_x >= _BUBBLE_RIGHT_SHELF_X
            and st.samus_x <= _BUBBLE_CAVITY_X_MAX
            and st.samus_y <= _BUBBLE_RIGHT_SHELF_Y
            and st.samus_y >= 200
        )

    mid_phase = "climb" if skip_launch else "launch"  # launch | climb
    mid_i = 0
    # Climb handoff: sticky height class (natural path already peaked).
    height_class = bool(skip_launch) or session.state.samus_y <= _BUBBLE_HEIGHT_CLASS_Y
    if skip_launch:
        launched = True
        mid_reached = True
    if not skip_to_door:
        for frame in range(_BUBBLE_MID_FRAMES):
            state = session.state
            if state.room_id != ROOM_BUBBLE:
                break
            _track(state)
            if (
                state.samus_y <= _BUBBLE_TOP_Y
                and state.samus_x >= _BUBBLE_TOP_X
            ):
                top_reached = True
                break
            if _avoid_wrong_door(state):
                continue
            if state.pose in (137, 138):
                for _ in range(10):
                    hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_mid_kb")
                continue

            x = state.samus_x
            y = state.samus_y
            if y <= _BUBBLE_HEIGHT_CLASS_Y:
                height_class = True
            if x > _BUBBLE_CAVITY_X_MAX and y > _BUBBLE_TOP_Y:
                hold(session, 1, "LEFT", "B", reason=f"{label}_mid_cap")
                continue

            mid_i += 1

            # --- launch: solid lip only (not mid float / cavity mid) ---
            if mid_phase == "launch" and not launched:
                if _on_launch_lip(state):
                    # Align on lip, charged HJ up-right into left column.
                    # Do not pre-run (any dash before charge regressed pure
                    # min_y to ~365). R9/R10 spin + extend only.
                    if x < 70:
                        hold(session, 1, "RIGHT", "B", reason=f"{label}_lip_align")
                        continue
                    if x > 90:
                        hold(session, 1, "LEFT", "B", reason=f"{label}_lip_align_l")
                        continue
                    for _ in range(_BUBBLE_LIP_CHARGE):
                        hold(session, 1, "A", reason=f"{label}_lip_charge")
                    for _ in range(_BUBBLE_LIP_SPIN):
                        state = hold(
                            session, 1, "RIGHT", "B", "A", reason=f"{label}_lip_hj"
                        )
                        _track(state)
                        if state.room_id != ROOM_BUBBLE:
                            break
                        if state.samus_y <= _BUBBLE_HEIGHT_CLASS_Y:
                            height_class = True
                        if bubble_phase_d_top_band(state):
                            top_reached = True
                            break
                    if (
                        not top_reached
                        and height_class
                        and state.room_id == ROOM_BUBBLE
                    ):
                        for _ in range(_BUBBLE_LIP_EXTEND):
                            state = hold(
                                session,
                                1,
                                "RIGHT",
                                "B",
                                "A",
                                reason=f"{label}_ol_extend",
                            )
                            _track(state)
                            if state.room_id != ROOM_BUBBLE:
                                break
                            if bubble_phase_d_top_band(state):
                                top_reached = True
                                break
                            if (
                                state.samus_x >= _BUBBLE_RIGHT_SHELF_X
                                and state.samus_y <= _BUBBLE_MIDHIGH_Y
                            ):
                                break
                            # R11: mid-nub land only on *true* ground poses
                            # (1/2/9/10). Spin apex pose 25 + vy≈0 is NOT a
                            # land — breaking extend there killed max_x.
                            if (
                                abs(state.velocity_y) <= 1
                                and state.pose in _BUBBLE_GROUND
                                and state.samus_y <= _BUBBLE_MID_RESEAT_Y
                                and 140 <= state.samus_x < _BUBBLE_RIGHT_SHELF_X
                            ):
                                break
                    launched = True
                    mid_phase = "climb"
                    mid_i = 0
                    if top_reached or state.room_id != ROOM_BUBBLE:
                        break
                    continue

                # Unstable mid float (y≤410, pin band): drop left onto solid lip.
                # (Mid-iso dash launch regressed pure: enemy KB thrash + height.)
                if (
                    y <= lip_y_lo
                    and stand_lo - 10 <= x <= stand_hi + 20
                ):
                    if x > lip_hi:
                        hold(session, 1, "LEFT", "B", reason=f"{label}_drop_left")
                    else:
                        hold(session, 1, reason=f"{label}_drop_idle")
                    continue

                # Too far right / low cavity: pull left then HJ toward lip.
                if x > 160 and y > lip_y_lo:
                    hold(session, 1, "LEFT", "B", reason=f"{label}_to_lip_left")
                    continue
                if abs(state.velocity_y) <= 1 and state.pose in _BUBBLE_STAND_PIN:
                    dir_h = "LEFT" if x > lip_hi else "RIGHT"
                    if y > lip_y_hi + 20:
                        # Below lip: charged hop up.
                        for _ in range(10):
                            hold(session, 1, "A", reason=f"{label}_below_charge")
                        for _ in range(36):
                            state = hold(
                                session,
                                1,
                                dir_h,
                                "B",
                                "A",
                                reason=f"{label}_below_hj",
                            )
                            _track(state)
                            if state.room_id != ROOM_BUBBLE:
                                break
                            if _on_launch_lip(state):
                                break
                        continue
                    hold(session, 1, dir_h, "B", reason=f"{label}_to_lip_walk")
                    continue
                dir_h = "LEFT" if x > 100 else "RIGHT"
                hold(session, 1, dir_h, "B", "A", reason=f"{label}_to_lip_air")
                if mid_i > 600:
                    # Budget escape: attempt climb from wherever we are.
                    mid_phase = "climb"
                    mid_i = 0
                continue

            # --- climb (R7/R10): mid-high open-loop right shelf then top hop ---
            if mid_phase == "climb":
                # True ground only (1/2/9/10). Spin apex pose 25 + vy≈0 is air.
                grounded = (
                    abs(state.velocity_y) <= 1 and state.pose in _BUBBLE_GROUND
                )
                if grounded:
                    # R9: right-structure shelf → LEFT charged HJ to top band.
                    # Place recon: LEFT from (360–380,331–363) hits top; RIGHT
                    # drifts into Single Chamber outer-wall height trap.
                    if height_class and _on_right_shelf(state):
                        for _ in range(12):
                            hold(session, 1, "A", reason=f"{label}_shelf_charge")
                        for _ in range(56):
                            state = hold(
                                session,
                                1,
                                "LEFT",
                                "B",
                                "A",
                                reason=f"{label}_shelf_hj",
                            )
                            _track(state)
                            if state.room_id != ROOM_BUBBLE:
                                break
                            if (
                                state.samus_y <= _BUBBLE_TOP_Y
                                and state.samus_x >= _BUBBLE_TOP_X
                            ):
                                top_reached = True
                                break
                        if top_reached or state.room_id != ROOM_BUBBLE:
                            break
                        continue

                    # R11 mid reseat: only true ground nubs (not spin apex).
                    # Short settle + charge + hop right into open-loop.
                    if height_class and y <= _BUBBLE_MID_RESEAT_Y and x < _BUBBLE_RIGHT_SHELF_X:
                        for _ in range(2):
                            hold(session, 1, reason=f"{label}_reseat_settle")
                        for _ in range(6):
                            hold(session, 1, "A", reason=f"{label}_reseat_charge")
                        for _ in range(56):
                            state = hold(
                                session,
                                1,
                                "RIGHT",
                                "B",
                                "A",
                                reason=f"{label}_reseat_hop",
                            )
                            _track(state)
                            if state.room_id != ROOM_BUBBLE:
                                break
                            if bubble_phase_d_top_band(state):
                                top_reached = True
                                break
                            if state.samus_y <= _BUBBLE_HEIGHT_CLASS_Y:
                                height_class = True
                        if top_reached or state.room_id != ROOM_BUBBLE:
                            break
                        continue

                    # Pre-height or fell low: R6-style lip re-seat / left column.
                    if y >= lip_y_lo and not (
                        lip_lo <= x <= lip_hi and y <= lip_y_hi
                    ):
                        if y <= lip_y_hi + 10 and x > lip_hi:
                            hold(
                                session, 1, "LEFT", "B", reason=f"{label}_climb_relip"
                            )
                            continue
                    for _ in range(10):
                        hold(session, 1, "A", reason=f"{label}_climb_charge")
                    if height_class:
                        # Height already earned: bias right peak-cross, not left thrash.
                        if x < 340:
                            dir_h = "RIGHT"
                        else:
                            dir_h = "LEFT"
                    elif y <= 220:
                        dir_h = "RIGHT"
                    elif y <= 280:
                        dir_h = "RIGHT" if x < 280 else "LEFT"
                    else:
                        # Pre-height low/mid: keep left column (x≈70–130).
                        if x < 70:
                            dir_h = "RIGHT"
                        elif x > 130:
                            dir_h = "LEFT"
                        else:
                            dir_h = "RIGHT" if (mid_i // 40) % 2 == 0 else "LEFT"
                    if x > _BUBBLE_CAVITY_X_MAX - 15:
                        dir_h = "LEFT"
                    for _ in range(44):
                        state = hold(
                            session,
                            1,
                            dir_h,
                            "B",
                            "A",
                            reason=f"{label}_climb_hj",
                        )
                        _track(state)
                        if state.room_id != ROOM_BUBBLE:
                            break
                        if state.samus_y <= _BUBBLE_HEIGHT_CLASS_Y:
                            height_class = True
                        if (
                            state.samus_y <= _BUBBLE_TOP_Y
                            and state.samus_x >= _BUBBLE_TOP_X
                        ):
                            top_reached = True
                            break
                    if top_reached or state.room_id != ROOM_BUBBLE:
                        break
                    continue

                # Air (R10): after height class *while still mid-high*
                # (y≤_BUBBLE_MIDHIGH_Y=450; R9 used 400). Natural path first
                # reaches x≥250 only near y≈400+, so R9 skipped open-loop on
                # the peak fall. Open-loop cross + period-8 WJ, or no-A shelf
                # land. Place: period-8 from air (360,y≤370) → top.
                if height_class and y <= _BUBBLE_MIDHIGH_Y:
                    if x > _BUBBLE_CAVITY_X_MAX - 15:
                        hold(session, 1, "LEFT", "B", reason=f"{label}_peak_sc")
                        continue
                    # Falling through shelf band on right → no-A re-seat.
                    if (
                        x >= _BUBBLE_RIGHT_SHELF_X
                        and 280 <= y <= _BUBBLE_MIDHIGH_Y
                        and state.velocity_y >= 0
                    ):
                        dir_h = "RIGHT" if x < 365 else "LEFT"
                        hold(
                            session,
                            1,
                            dir_h,
                            "B",
                            reason=f"{label}_shelf_drop",
                        )
                        continue
                    # Engage right wall (x≥250) with period-8 WJ while mid-high.
                    if x >= 250 and y > _BUBBLE_TOP_Y:
                        phase = mid_i % _BUBBLE_RIGHT_WJ_PERIOD
                        if phase < _BUBBLE_RIGHT_WJ_INTO:
                            hold(
                                session,
                                1,
                                "RIGHT",
                                "B",
                                reason=f"{label}_ol_into",
                            )
                        elif phase < (
                            _BUBBLE_RIGHT_WJ_INTO + _BUBBLE_RIGHT_WJ_BOUNCE
                        ):
                            hold(
                                session, 1, "LEFT", "A", reason=f"{label}_ol_wj"
                            )
                        else:
                            hold(
                                session,
                                1,
                                "RIGHT",
                                "B",
                                "A",
                                reason=f"{label}_ol_spin",
                            )
                        continue
                    # Left of wall: hard right spin to approach mid-high band.
                    hold(
                        session, 1, "RIGHT", "B", "A", reason=f"{label}_ol_cross"
                    )
                    continue

                # Pre-height air (or fell below mid-high after height class): climb.
                if y <= 240:
                    dir_h = "RIGHT" if x < 340 else "LEFT"
                elif height_class and x < 340:
                    # Recovered height once: bias right re-climb, not left thrash.
                    dir_h = "RIGHT"
                elif x < 70:
                    dir_h = "RIGHT"
                elif x > 150 and y > 300:
                    dir_h = "LEFT"
                else:
                    dir_h = "RIGHT" if x < 120 else "LEFT"
                if x > _BUBBLE_CAVITY_X_MAX - 15:
                    dir_h = "LEFT"
                phase = mid_i % 12
                if phase < 2:
                    hold(session, 1, dir_h, "B", reason=f"{label}_climb_rel")
                elif phase < 4:
                    opp = "LEFT" if dir_h == "RIGHT" else "RIGHT"
                    hold(session, 1, opp, "A", reason=f"{label}_climb_wj")
                else:
                    hold(
                        session, 1, dir_h, "B", "A", reason=f"{label}_climb_spin"
                    )
                continue

            # Fallback
            hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_mid_fallback")

    # --- Phase 3: top-right Super door ---
    if session.state.selected_item != 2:
        select_weapon(session, 2)

    for frame in range(_BUBBLE_DOOR_FRAMES):
        state = session.state
        if state.room_id == ROOM_BAT_CAVE:
            break
        if state.room_id != ROOM_BUBBLE:
            break
        _track(state)
        if _avoid_wrong_door(state):
            continue
        if state.pose in (137, 138):
            for _ in range(8):
                hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_door_kb")
            continue

        # Need top band before door pressure.
        if state.samus_y > 220 or state.samus_x < 280:
            if state.selected_item != 0:
                select_weapon(session, 0)
            dir_h = "RIGHT" if state.samus_x < 320 else "LEFT"
            phase = frame % 16
            if phase < 10:
                hold(session, 1, dir_h, "B", "A", reason=f"{label}_door_climb")
            elif phase < 12:
                hold(session, 1, dir_h, "B", reason=f"{label}_door_rel")
            else:
                opp = "LEFT" if dir_h == "RIGHT" else "RIGHT"
                hold(session, 1, opp, "A", reason=f"{label}_door_wj")
            continue

        door_reached = True
        if state.selected_item != 2:
            select_weapon(session, 2)
        # Continuous Super pulses + RIGHT pressure (place proof ~153f).
        phase = frame % 28
        if phase < 4:
            inputs = ("RIGHT", "X")
        elif phase < 14:
            inputs = ("RIGHT",)
        elif phase < 20:
            inputs = ("RIGHT", "B")
        else:
            inputs = ("RIGHT", "B", "A")
        state = hold(session, 1, *inputs, reason=f"{label}_door")
        if state.room_id == ROOM_BAT_CAVE:
            break
    else:
        state = session.state
        raise TimeoutError(
            f"{label}: Bat Cave Super door missed before room "
            f"0x{ROOM_BAT_CAVE:04X}; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"door_transition={state.door_transition} max_x={max_x} "
            f"min_y={min_y} mid_reached={mid_reached} top_reached={top_reached} "
            f"door_reached={door_reached} standing_mid_pinned={standing_mid_pinned} "
            f"launched={launched} phase_c_hit={phase_c_hit} "
            f"supers={state.super_missiles} selected={state.selected_item}"
        )

    if session.state.room_id != ROOM_BAT_CAVE:
        state = session.state
        raise TimeoutError(
            f"{label}: left Bubble without ordinary Bat Cave; "
            f"room=0x{state.room_id:04X} pose={state.pose} "
            f"xy=({state.samus_x},{state.samus_y}) max_x={max_x} min_y={min_y} "
            f"mid_reached={mid_reached} top_reached={top_reached} "
            f"door_reached={door_reached} standing_mid_pinned={standing_mid_pinned} "
            f"phase_c_hit={phase_c_hit}"
        )

    return wait_ordinary_room(
        session,
        ROOM_BAT_CAVE,
        settle_frames=_BUBBLE_TO_BAT_SETTLE_FRAMES,
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

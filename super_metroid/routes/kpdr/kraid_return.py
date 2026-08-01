"""Kraid return reverse hops (Eye → Baby → Kihunter → Zeela → Warehouse).

Several hops on this spine are locked ``continuous`` on the Business /
Frog Save tip (see ``START_TO_SPEED_GRAPH``). Controllers stay pure — no
env ownership, door-warps, or progression writes. Prefer named phases over
mid-loop magic when extending dense reverse climbs (Zeela).
"""

from __future__ import annotations

from super_metroid.policy import StateRequirement
from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    ensure_morph,
    hold,
    require_room,
    require_state,
    select_weapon,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.rooms import (
    ROOM_BABY_KRAID,
    ITEM_HI_JUMP,
    ROOM_KRAID_EYE,
    ROOM_WAREHOUSE,
    ROOM_WAREHOUSE_KIHUNTER,
    ROOM_ZEELA,
)
from super_metroid.routes.runtime import ControllerSession


def _play_return(
    session: ControllerSession,
    *,
    source_room: int,
    target_room: int,
    direction: str,
    label: str,
    spin: bool = False,
) -> SuperMetroidState:
    require_state(
        session,
        StateRequirement(room_id=source_room, game_states=frozenset({8})),
        label,
    )
    require_room(session, source_room, label)
    select_weapon(session, 0)

    buttons = (direction, "B", "A") if spin else (direction, "A")
    for _ in range(900):
        state = hold(session, 1, *buttons, reason=f"{label}_exit")
        if state.room_id == target_room:
            break
    else:
        raise TimeoutError(f"{label}: exit timed out: {session.state}")

    return wait_ordinary_room(
        session,
        target_room,
        settle_frames=320,
        label=label,
    )


def play_eye_to_baby_return(session: ControllerSession) -> SuperMetroidState:
    """Left return from Kraid Eye Door Room to Baby Kraid (post pure K3.3).

    Controller-dev reverse hop. Not continuous evidence.

    Geometry (SM-K4-R-01B): walk-only / floor spin pins mid-room (~x=373,
    pose 138) on a ledge. Clear with sustained jump-left, open the left blue
    hatch with standing beams near the lip, then jump-enter. Source:
    ``scratch/post_kraid_to_eye_return.state`` (0xA56B).
    """
    require_state(
        session,
        StateRequirement(room_id=ROOM_KRAID_EYE, game_states=frozenset({8})),
        "eye_to_baby_return",
    )
    require_room(session, ROOM_KRAID_EYE, "eye_to_baby_return")
    select_weapon(session, 0)

    # Jump-left across the mid-room ledge that floor-walk pins at x≈373.
    for index in range(500):
        state = session.state
        if state.samus_x <= 140:
            break
        phase = index % 30
        if phase < 10:
            hold(session, 1, "LEFT", "A", reason="eye_to_baby_hop")
        elif phase < 18:
            hold(session, 1, "LEFT", "A", "B", reason="eye_to_baby_spin")
        elif phase < 22:
            hold(session, 1, "X", reason="eye_to_baby_clear_shot")
        else:
            hold(session, 1, "LEFT", "B", reason="eye_to_baby_run")
    else:
        raise TimeoutError(
            f"eye_to_baby_return: mid-room approach timed out: {session.state}"
        )

    # Stage just right of the left lip; open blue hatch with X-only beams.
    hold(session, 8, reason="eye_to_baby_approach_settle")
    hold(session, 8, "RIGHT", reason="eye_to_baby_lip_backoff")
    hold(session, 8, "LEFT", reason="eye_to_baby_face_left")
    hold(session, 6, reason="eye_to_baby_face_release")
    for _ in range(6):
        hold(session, 4, "X", reason="eye_to_baby_door_shot")
        hold(session, 14, reason="eye_to_baby_door_fuse")

    for index in range(700):
        phase = index % 30
        if phase < 4:
            state = hold(session, 1, "LEFT", "A", reason="eye_to_baby_jump")
        elif phase < 10:
            state = hold(session, 1, "LEFT", "A", "B", reason="eye_to_baby_jump_spin")
        elif phase < 14:
            state = hold(session, 1, "X", reason="eye_to_baby_reshot")
        else:
            state = hold(session, 1, "LEFT", "B", reason="eye_to_baby_exit")
        if state.room_id == ROOM_BABY_KRAID:
            break
        if state.door_transition:
            for _ in range(80):
                state = hold(session, 1, reason="eye_to_baby_transition")
                if state.room_id == ROOM_BABY_KRAID and state.door_transition == 0:
                    break
            if state.room_id == ROOM_BABY_KRAID:
                break
    else:
        raise TimeoutError(f"eye_to_baby_return: left exit timed out: {session.state}")

    return wait_ordinary_room(
        session,
        ROOM_BABY_KRAID,
        settle_frames=320,
        label="eye_to_baby_return",
    )


def play_baby_to_kihunter_return(session: ControllerSession) -> SuperMetroidState:
    """Left return Baby Kraid → Warehouse Kihunter (post pure eye→baby).

    Controller-dev reverse hop. Not continuous evidence.

    The left hatch is **gray** with ``clear_room_enemies`` (graph node Baby
    Kraid Left Door). Floor spin alone pins at the shell (x≈37–69, pose 138,
    door_transition=0). Clear pirates/Mini-Kraid first, then beam-open and
    exit left. Source: ``scratch/post_eye_to_baby_return.state``.
    """
    from super_metroid.routes.kpdr.kraid_approach import _baby_kraid_sweep

    require_state(
        session,
        StateRequirement(room_id=ROOM_BABY_KRAID, game_states=frozenset({8})),
        "baby_to_kihunter_return",
    )
    require_room(session, ROOM_BABY_KRAID, "baby_to_kihunter_return")
    # Supers match play_baby_kraid_to_eye — beams alone leave Mini-Kraid alive
    # and the gray left hatch stays locked (clear_room_enemies).
    select_weapon(session, 2)

    # Land from elevated eye-door entry if needed, then clear the room lock.
    for _ in range(30):
        state = hold(session, 1, reason="baby_return_land")
        if state.velocity_y == 0 and state.pose in (1, 2, 5, 6, 9, 10, 137, 138):
            break
    _baby_kraid_sweep(session, "LEFT", 80, limit=1700, label="baby_return_clear_left")
    if session.state.enemies_killed < session.state.num_enemies:
        _baby_kraid_sweep(
            session, "RIGHT", 1490, limit=1900, label="baby_return_clear_right"
        )
    if session.state.enemies_killed < session.state.num_enemies:
        _baby_kraid_sweep(
            session, "LEFT", 80, limit=1900, label="baby_return_clear_left2"
        )

    # Stage near left gray door; beams open the unlocked shell.
    select_weapon(session, 0)
    for _ in range(200):
        state = hold(session, 1, "LEFT", "B", reason="baby_return_door_approach")
        if state.samus_x <= 120:
            break
    hold(session, 10, "RIGHT", reason="baby_return_lip_backoff")
    hold(session, 8, "LEFT", reason="baby_return_face_left")
    hold(session, 6, reason="baby_return_face_release")
    for _ in range(6):
        hold(session, 4, "X", reason="baby_return_door_shot")
        hold(session, 14, reason="baby_return_door_fuse")

    for index in range(700):
        phase = index % 30
        if phase < 4:
            state = hold(session, 1, "LEFT", "A", reason="baby_return_jump")
        elif phase < 10:
            state = hold(session, 1, "LEFT", "A", "B", reason="baby_return_jump_spin")
        elif phase < 14:
            state = hold(session, 1, "X", reason="baby_return_reshot")
        else:
            state = hold(session, 1, "LEFT", "B", reason="baby_return_exit")
        if state.room_id == ROOM_WAREHOUSE_KIHUNTER:
            break
        if state.door_transition:
            for _ in range(80):
                state = hold(session, 1, reason="baby_return_transition")
                if (
                    state.room_id == ROOM_WAREHOUSE_KIHUNTER
                    and state.door_transition == 0
                ):
                    break
            if state.room_id == ROOM_WAREHOUSE_KIHUNTER:
                break
    else:
        raise TimeoutError(
            f"baby_to_kihunter_return: left gray exit timed out: {session.state}"
        )

    return wait_ordinary_room(
        session,
        ROOM_WAREHOUSE_KIHUNTER,
        settle_frames=320,
        label="baby_to_kihunter_return",
    )


def _kihunter_guard_rooms(
    session: ControllerSession, label: str, *, allow_zeela: bool = False
) -> SuperMetroidState:
    """Fail loud on Baby / wrong-room transitions during the reverse climb."""
    state = session.state
    if state.room_id == ROOM_BABY_KRAID:
        raise TimeoutError(f"{label}: crossed wrong door into Baby Kraid: {state}")
    if state.room_id == ROOM_ZEELA:
        if allow_zeela:
            return state
        raise TimeoutError(
            f"{label}: reached Zeela before true upper Kihunter land: {state}"
        )
    if state.room_id != ROOM_WAREHOUSE_KIHUNTER:
        raise TimeoutError(f"{label}: left source room during climb: {state}")
    return state


def _kihunter_upper_band(state: SuperMetroidState) -> bool:
    """True when Samus is in the upper Kihunter band (not mid ledge y≈299)."""
    return (
        state.samus_y < 220
        and state.room_id == ROOM_WAREHOUSE_KIHUNTER
        and state.door_transition == 0
    )


def play_kihunter_to_zeela_return(session: ControllerSession) -> SuperMetroidState:
    """Climb out of the lower Kihunter alcove and drop through Zeela's door.

    This is not continuous evidence; it requires a natural source state.

    Redesign (Wave 9 / SM-K4-R-CLIMB-REDESIGN) — room geometry:

    * Floor hard wall at **x≈357** blocks walking under the hole.
    * Mid ledge at **y≈299, x≈367–379** is reached by wall-plant Hi-Jump then
      RIGHT drift at apex.
    * Forward bomb hole is exactly **x≈376** (upper floor y≈171). From the mid
      ledge, **morph bomb-jump** through that hole, then left to the Zeela
      down-door window **x∈[96,160]**.
    """
    label = "kihunter_to_zeela_return"
    require_state(
        session,
        StateRequirement(
            room_id=ROOM_WAREHOUSE_KIHUNTER,
            game_states=frozenset({8}),
        ),
        label,
    )
    require_room(session, ROOM_WAREHOUSE_KIHUNTER, label)
    select_weapon(session, 0)
    unmorph(session)

    best_min_y = session.state.samus_y

    # --- 1) Plant on the hard wall at x≈357 ---
    hold(session, 10, reason="kihunter_zeela_entry_release")
    for _ in range(220):
        state = _kihunter_guard_rooms(session, label)
        if state.samus_x <= 358:
            break
        hold(session, 1, "LEFT", "B", reason="kihunter_zeela_wall_run")
    for _ in range(40):
        state = _kihunter_guard_rooms(session, label)
        if state.samus_x <= 120:
            break
        hold(session, 1, "LEFT", reason="kihunter_zeela_wall_plant")
    hold(session, 4, reason="kihunter_zeela_wall_settle")

    # --- 2) Hi-Jump, drift RIGHT onto mid ledge (y≈299, x≥365) ---
    hold(session, 3, "RIGHT", reason="kihunter_zeela_face_ledge")
    hold(session, 2, reason="kihunter_zeela_face_settle")
    hold(session, 8, "DOWN", reason="kihunter_zeela_crouch_load")
    mid = False
    has_hi_jump = bool(session.state.collected_items & ITEM_HI_JUMP)
    for frame in range(110):
        state = session.state
        # Hi-Jump reaches the y=291 ledge several frames earlier than the
        # historical no-Hi-Jump fixture.  Start the rightward landing drift
        # from the live height rather than an absolute loop frame, then cap it
        # at the same x≈367 hand-off used by the fixture route.
        if has_hi_jump and state.samus_y <= 300:
            buttons = ("RIGHT", "A", "B") if state.samus_x < 367 else ()
        elif frame < 30:
            buttons = ("A",)
        elif frame < 45:
            buttons = ("A", "UP", "X")
        elif state.samus_y <= 300:
            buttons = ("RIGHT", "A", "B")
        else:
            buttons = ("RIGHT", "B")
        state = hold(session, 1, *buttons, reason="kihunter_zeela_mid_ledge")
        best_min_y = min(best_min_y, state.samus_y)
        _kihunter_guard_rooms(session, label)
        if (
            state.samus_y <= 299
            and state.velocity_y == 0
            and state.samus_x >= 365
            and frame > 40
        ):
            mid = True
            break
    hold(session, 16, reason="kihunter_zeela_mid_settle")
    if not mid and not (session.state.samus_y <= 305 and session.state.samus_x >= 360):
        raise TimeoutError(
            f"{label}: mid ledge missed: {session.state}; best_min_y={best_min_y}"
        )

    # --- 3) Align under bomb hole x≈376 ---
    for _ in range(50):
        state = _kihunter_guard_rooms(session, label)
        if 374 <= state.samus_x <= 380:
            break
        if state.samus_x < 374:
            hold(session, 1, "RIGHT", reason="kihunter_zeela_hole_align")
        else:
            hold(session, 1, "LEFT", reason="kihunter_zeela_hole_align")
    hold(session, 6, reason="kihunter_zeela_hole_settle")

    # --- 4) Morph bomb-jump through the x≈376 floor hole to upper band ---
    ensure_morph(session)
    climbed = False
    for cycle in range(90):
        state = _kihunter_guard_rooms(session, label)
        if state.samus_x < 372:
            hold(session, 2, "RIGHT", reason="kihunter_zeela_hole_recenter")
        elif state.samus_x > 382:
            hold(session, 2, "LEFT", reason="kihunter_zeela_hole_recenter")
        hold(session, 2, "X", reason="kihunter_zeela_hole_bomb")
        if state.samus_y < 260:
            wait = 22
        elif state.samus_y < 280:
            wait = 30
        else:
            wait = 50
        for _ in range(wait):
            state = hold(session, 1, reason="kihunter_zeela_hole_bomb_wait")
            best_min_y = min(best_min_y, state.samus_y)
            _kihunter_guard_rooms(session, label)
        # Firm upper: current settle height, not only a peak boost.
        if session.state.samus_y < 200:
            climbed = True
            break
        # Once peaking through the hole, keep rapid bombs to settle on top.
        if best_min_y < 210 and session.state.samus_y < 240:
            hold(session, 2, "X", reason="kihunter_zeela_hole_top_bomb")
            for _ in range(20):
                state = hold(session, 1, reason="kihunter_zeela_hole_top_wait")
                best_min_y = min(best_min_y, state.samus_y)
                _kihunter_guard_rooms(session, label)
                if state.samus_y < 195:
                    climbed = True
                    break
            if climbed:
                break
    if not climbed:
        raise TimeoutError(
            f"{label}: bomb-hole climb timed out: {session.state}; "
            f"best_min_y={best_min_y}"
        )

    # Extra bombs to plant on the upper floor (y≈171–190) before rolling.
    ensure_morph(session)
    for _ in range(8):
        if session.state.samus_y < 190:
            break
        hold(session, 2, "X", reason="kihunter_zeela_upper_plant_bomb")
        for _wait in range(22):
            state = hold(session, 1, reason="kihunter_zeela_upper_plant_wait")
            best_min_y = min(best_min_y, state.samus_y)
            _kihunter_guard_rooms(session, label)
    hold(session, 10, reason="kihunter_zeela_upper_morph_settle")
    if session.state.samus_y >= 230:
        raise TimeoutError(
            f"{label}: fell off upper after hole climb: {session.state}; "
            f"best_min_y={best_min_y}"
        )

    # --- 5) Morph-roll left to Zeela down-door window x=96..160 ---
    for _ in range(500):
        state = session.state
        if state.room_id == ROOM_BABY_KRAID:
            raise TimeoutError(
                f"{label}: upper traverse crossed wrong door: {session.state}"
            )
        if state.room_id != ROOM_WAREHOUSE_KIHUNTER:
            raise TimeoutError(
                f"{label}: upper traverse left source room: {session.state}"
            )
        if state.samus_y > 300:
            raise TimeoutError(f"{label}: fell during upper traverse: {session.state}")
        # Bomb-boost if starting to sink through residual floor tiles.
        if state.samus_y > 210:
            hold(session, 2, "X", reason="kihunter_zeela_traverse_boost")
            for _wait in range(18):
                state = hold(session, 1, reason="kihunter_zeela_traverse_boost_wait")
                _kihunter_guard_rooms(session, label)
        if state.samus_x < 96:
            hold(session, 1, "RIGHT", reason="kihunter_zeela_window_recover")
            continue
        if state.samus_x <= 160 and state.samus_y < 230:
            break
        hold(session, 1, "LEFT", reason="kihunter_zeela_upper_roll")
    else:
        raise TimeoutError(
            f"{label}: Zeela x-window approach timed out: {session.state}"
        )

    unmorph(session)
    select_weapon(session, 0)
    hold(session, 10, reason="kihunter_zeela_window_stand")
    state = session.state
    if not 90 <= state.samus_x <= 170 or state.samus_y >= 250:
        raise TimeoutError(f"{label}: invalid Zeela door window: {state}")
    hold(session, 8, "LEFT", reason="kihunter_zeela_door_face")
    hold(session, 6, reason="kihunter_zeela_door_release")
    for _ in range(6):
        hold(session, 4, "DOWN", "X", reason="kihunter_zeela_door_shot")
        hold(session, 14, reason="kihunter_zeela_door_fuse")

    for index in range(520):
        phase = index % 30
        if phase < 8:
            state = hold(session, 1, "DOWN", "A", reason="kihunter_zeela_drop")
        elif phase < 14:
            state = hold(
                session, 1, "DOWN", "A", "B", reason="kihunter_zeela_drop_spin"
            )
        elif phase < 18:
            state = hold(session, 1, "DOWN", "X", reason="kihunter_zeela_reshot")
        else:
            state = hold(session, 1, "DOWN", reason="kihunter_zeela_drop")
        if state.room_id == ROOM_ZEELA:
            break
        if state.room_id == ROOM_BABY_KRAID:
            raise TimeoutError(
                f"{label}: blue down-door entered Baby Kraid: {session.state}"
            )
    else:
        raise TimeoutError(f"{label}: blue down-door exit timed out: {session.state}")

    return wait_ordinary_room(
        session,
        ROOM_ZEELA,
        settle_frames=320,
        label=label,
        # The down-door transition becomes ordinary while Samus is still
        # falling.  Do not hand its successor the airborne x≈357/y≈362 frame:
        # Zeela's reverse controller needs the real floor handoff near y=395.
        y_range=(385, 410),
    )


# Named phases for the continuous Zeela→Warehouse reverse climb. Keep geometry
# edits inside one phase; do not invent mid-loop lineage branches.
_ZEELA_PHASE_BOTTOM_ROLL = "bottom_roll"
_ZEELA_PHASE_MID_PLATFORM = "mid_platform"
_ZEELA_PHASE_BELOW_LIP = "below_platform_lip"
_ZEELA_PHASE_WALL_PLANT = "wall_plant"
_ZEELA_PHASE_SHOTBLOCK_CLIMB = "shotblock_wall_climb"
_ZEELA_PHASE_WAREHOUSE_DOOR = "warehouse_door_exit"


def play_zeela_to_warehouse_return(session: ControllerSession) -> SuperMetroidState:
    """Climb Zeela reverse to the upper-left Warehouse door.

    Continuous on the Business / Frog Save tip (graph verification
    ``continuous``). Pure probes still need a continuous-like source state.

    Phases (named; one-knob geometry edits stay inside a single phase):

    1. ``bottom_roll`` — morph reverse-roll + align second-drop lane
    2. ``mid_platform`` — reverse-shot climb onto middle platform
    3. ``below_platform_lip`` — crouch-load from mid right-edge to lip
    4. ``wall_plant`` — hop-left to wall plant band (x≈37, y≈219)
    5. ``shotblock_wall_climb`` — clear shot blocks + wall-spin to top band
    6. ``warehouse_door_exit`` — standing LEFT beams into Warehouse 0xA6A1

    Geometry facts (SM-K4-R-ZEELA-REDESIGN):

    * Historical fixtures begin bottom-right after the Kihunter drop (~x=403)
      without Hi-Jump (``items=0x1005``); the natural Varia lineage has
      Hi-Jump and needs a farther-right mid-platform landing gate.
    * Floor-left is the Energy Tank door ``0xA4B1`` — fail-loud if ``y>250``.
    * Middle platform is narrow (~x=90–107, y≈331). Reach it with reverse-shot
      **RIGHT bias in the hole** (left-only peaks the shaft at x≈52 and falls).
    * Below-platform lip (~x=40–70, y≈219–235) is reached by crouch-load from
      mid right-edge (x≈107), then LEFT drift — not by floor-left spam.
    * Lip crouch-load lands ~x=69 (LEFT walk blocked); hop-left to plant
      x≈37 y≈219.
    * Reverse re-entry leaves shot blocks solid above the left wall — clear
      with ~40 UP+X before wall-spin climb. Hard cap without clear: y≈188.
    * Top door band is **y≈112–139**. Stop only on grounded standing/crouch
      at y≤150 (pose 1/2/137/138), not walljump pose 26.
    * Warehouse door opens with standing LEFT beams from that top band.
    """
    label = "zeela_to_warehouse_return"
    require_state(
        session,
        StateRequirement(room_id=ROOM_ZEELA, game_states=frozenset({8})),
        label,
    )
    require_room(session, ROOM_ZEELA, label)
    select_weapon(session, 0)
    has_hi_jump = bool(session.state.collected_items & ITEM_HI_JUMP)

    def guard_climb(state: SuperMetroidState, phase: str) -> None:
        if state.door_transition and state.samus_y > 250:
            raise TimeoutError(
                f"{label}: floor door transition during {phase}: {state}"
            )
        if state.room_id != ROOM_ZEELA:
            raise TimeoutError(f"{label}: left Zeela during {phase}: {state}")

    # --- phase: bottom_roll ---
    ensure_morph(session)
    for _ in range(900):
        state = hold(session, 1, "LEFT", reason="zeela_warehouse_bottom_roll")
        guard_climb(state, _ZEELA_PHASE_BOTTOM_ROLL)
        if state.samus_x <= 160 and state.samus_y >= 350:
            break
    else:
        raise TimeoutError(f"{label}: {_ZEELA_PHASE_BOTTOM_ROLL} stalled: {session.state}")

    unmorph(session)
    for _ in range(160):
        state = session.state
        guard_climb(state, _ZEELA_PHASE_BOTTOM_ROLL)
        if 110 <= state.samus_x <= 140:
            break
        if state.samus_x < 110:
            hold(session, 1, "RIGHT", reason="zeela_warehouse_second_align")
        else:
            hold(session, 1, "LEFT", reason="zeela_warehouse_second_align")
    hold(session, 8, reason="zeela_warehouse_second_align_settle")

    # --- phase: mid_platform ---
    # Landing target: y∈[300,350], grounded, x≥90 (not shaft peak x≈52).
    select_weapon(session, 0)
    mid = False
    for frame in range(700):
        state = session.state
        guard_climb(state, _ZEELA_PHASE_MID_PLATFORM)
        mid_x_min = 96 if has_hi_jump else 90
        if (
            300 <= state.samus_y <= 350
            and state.velocity_y == 0
            and state.samus_x >= mid_x_min
            and frame > 50
        ):
            mid = True
            break
        cadence = frame % 28
        if cadence < 5:
            buttons: tuple[str, ...] = ("UP", "X")
        elif state.samus_y <= 360:
            # In/near the hole — drift RIGHT onto the mid platform.
            buttons = ("RIGHT", "A", "B") if cadence >= 16 else ("RIGHT", "A")
        elif cadence < 14:
            buttons = ("A",)
        elif state.samus_x <= 70:
            buttons = ("RIGHT", "A")
        else:
            buttons = ("LEFT", "A")
        hold(session, 1, *buttons, reason="zeela_warehouse_second_reverse_shot")
    hold(session, 12, reason="zeela_warehouse_mid_settle")
    if not mid and not (
        300 <= session.state.samus_y <= 355 and session.state.samus_x >= 85
    ):
        raise TimeoutError(f"{label}: {_ZEELA_PHASE_MID_PLATFORM} missed: {session.state}")

    # --- phase: below_platform_lip ---
    for _ in range(80):
        state = session.state
        guard_climb(state, _ZEELA_PHASE_BELOW_LIP)
        if state.samus_y > 360:
            raise TimeoutError(f"{label}: fell off mid platform: {state}")
        if state.samus_x >= 104:
            break
        hold(session, 1, "RIGHT", reason="zeela_warehouse_mid_right_edge")
    hold(session, 8, reason="zeela_warehouse_mid_edge_settle")

    unmorph(session)
    select_weapon(session, 0)
    hold(session, 10, "DOWN", reason="zeela_warehouse_first_crouch_load")
    hold(session, 2, reason="zeela_warehouse_first_crouch_release")
    lip = False
    for frame in range(600):
        state = session.state
        guard_climb(state, _ZEELA_PHASE_BELOW_LIP)
        if (
            state.samus_y <= 240
            and state.velocity_y == 0
            and state.samus_x <= 80
            and frame > 40
        ):
            lip = True
            break
        if frame < 18:
            buttons = ("A",)
        elif frame < 30:
            buttons = ("A", "UP", "X")
        elif state.samus_y < 280:
            buttons = ("LEFT", "A", "B")
        else:
            cadence = frame % 24
            if cadence < 6:
                buttons = ("UP", "X")
            elif cadence < 16:
                buttons = ("A",)
            else:
                buttons = ()
        hold(session, 1, *buttons, reason="zeela_warehouse_first_crouch_climb")
    hold(session, 15, reason="zeela_warehouse_lip_settle")
    if not lip and not (session.state.samus_y <= 250 and session.state.samus_x <= 90):
        raise TimeoutError(f"{label}: {_ZEELA_PHASE_BELOW_LIP} missed: {session.state}")

    # --- phase: wall_plant ---
    # The crouch-load lands ~x=69 where LEFT walk is blocked; a short hop-left
    # reaches the below-platform plant used by the wall climb.
    unmorph(session)
    for frame in range(50):
        state = session.state
        guard_climb(state, _ZEELA_PHASE_WALL_PLANT)
        if (
            state.samus_x <= 45
            and state.velocity_y == 0
            and 210 <= state.samus_y <= 230
            and frame > 10
        ):
            break
        if frame < 8:
            buttons = ("A",)
        elif frame < 28:
            buttons = ("LEFT", "A")
        else:
            buttons = ("LEFT",)
        hold(session, 1, *buttons, reason="zeela_warehouse_lip_hop_left")
    hold(session, 20, reason="zeela_warehouse_bp_settle")
    if not (session.state.samus_y <= 230 and session.state.samus_x <= 55):
        raise TimeoutError(f"{label}: {_ZEELA_PHASE_WALL_PLANT} missed: {session.state}")

    # --- phase: shotblock_wall_climb ---
    # Reverse source re-enters Zeela from Kihunter without the forward shot-block
    # clear; pure UP+X (≈40) opens the column so the wall climb can reach y≈139.
    # Without that clear the climb hard-caps near y=188.
    top = False
    for attempt in range(3):
        unmorph(session)
        select_weapon(session, 0)
        for _ in range(20):
            if session.state.pose in (1, 2):
                break
            hold(session, 1, "UP", reason="zeela_warehouse_clear_stand")
        hold(session, 8, reason="zeela_warehouse_clear_stand_settle")
        if session.state.samus_y > 250:
            raise TimeoutError(f"{label}: fell off lip before clear: {session.state}")

        for _ in range(40):
            hold(session, 2, "UP", "X", reason="zeela_warehouse_shotblock_clear")
            hold(session, 4, reason="zeela_warehouse_shotblock_fuse")
        for frame in range(60):
            state = session.state
            guard_climb(state, _ZEELA_PHASE_SHOTBLOCK_CLIMB)
            cadence = frame % 8
            if cadence < 3:
                buttons = ("UP", "X")
            elif cadence < 6:
                buttons = ("A",)
            else:
                buttons = ()
            hold(session, 1, *buttons, reason="zeela_warehouse_clear_jump")

        for _ in range(30):
            state = session.state
            guard_climb(state, _ZEELA_PHASE_SHOTBLOCK_CLIMB)
            if state.samus_x <= 35:
                break
            if state.samus_y > 250:
                break
            hold(session, 1, "LEFT", "B", reason="zeela_warehouse_wall_plant")
        hold(session, 4, reason="zeela_warehouse_wall_plant_settle")

        for frame in range(900):
            state = session.state
            guard_climb(state, _ZEELA_PHASE_SHOTBLOCK_CLIMB)
            if (
                state.samus_y <= 150
                and state.velocity_y == 0
                and state.pose in (1, 2, 39, 40, 137, 138)
                and frame > 20
            ):
                hold(session, 8, reason="zeela_warehouse_top_confirm")
                if (
                    session.state.samus_y <= 155
                    and session.state.velocity_y == 0
                    and session.state.room_id == ROOM_ZEELA
                ):
                    top = True
                    break
            cadence = frame % 14
            if cadence < 7:
                buttons = ("LEFT", "A", "B")
            elif cadence < 10:
                buttons = ("RIGHT", "A")
            elif cadence < 12:
                buttons = ("LEFT", "A")
            else:
                buttons = ("LEFT",)
            state = hold(session, 1, *buttons, reason="zeela_warehouse_wall_climb")
            # Hi-Jump continuous lineage may naturally enter Warehouse during
            # the final wall-climb cadence (upper-left door, not floor E-Tank).
            if state.room_id == ROOM_WAREHOUSE:
                if state.samus_y > 250:
                    raise TimeoutError(
                        f"{label}: floor door transition during wall climb: {state}"
                    )
                return wait_ordinary_room(
                    session,
                    ROOM_WAREHOUSE,
                    settle_frames=320,
                    label=label,
                )
            if session.state.samus_y > 280:
                break
        if top:
            break
        # Still on the lip — retry clear/climb once more.
        if not (session.state.samus_y <= 230 and session.state.samus_x <= 55):
            break
    if not top:
        raise TimeoutError(
            f"{label}: {_ZEELA_PHASE_SHOTBLOCK_CLIMB} top band missed: {session.state}"
        )

    # --- phase: warehouse_door_exit ---
    unmorph(session)
    select_weapon(session, 0)
    for _ in range(15):
        if session.state.pose in (1, 2):
            break
        hold(session, 1, "UP", reason="zeela_warehouse_door_stand")
    hold(session, 10, reason="zeela_warehouse_door_stand_settle")
    if session.state.samus_y > 170:
        raise TimeoutError(f"{label}: fell before Warehouse door: {session.state}")

    hold(session, 6, "LEFT", reason="zeela_warehouse_door_face")
    hold(session, 4, reason="zeela_warehouse_door_face_release")
    for _ in range(10):
        hold(session, 3, "LEFT", "X", reason="zeela_warehouse_door_shot")
        hold(session, 12, reason="zeela_warehouse_door_fuse")

    for index in range(400):
        cadence = index % 20
        if cadence < 10:
            state = hold(session, 1, "LEFT", "A", reason="zeela_warehouse_exit")
        elif cadence < 14:
            state = hold(
                session, 1, "LEFT", "A", "B", reason="zeela_warehouse_exit_spin"
            )
        else:
            state = hold(session, 1, "LEFT", reason="zeela_warehouse_exit")
        if state.room_id == ROOM_WAREHOUSE:
            break
        if state.door_transition and state.samus_y > 250:
            raise TimeoutError(
                f"{label}: floor door transition at Warehouse exit: {state}"
            )
        if state.room_id != ROOM_ZEELA:
            raise TimeoutError(f"{label}: unexpected exit room: {state}")
        if state.samus_y > 200:
            raise TimeoutError(f"{label}: fell during Warehouse exit: {state}")
    else:
        raise TimeoutError(
            f"{label}: {_ZEELA_PHASE_WAREHOUSE_DOOR} timed out: {session.state}"
        )

    return wait_ordinary_room(
        session,
        ROOM_WAREHOUSE,
        settle_frames=320,
        label=label,
    )

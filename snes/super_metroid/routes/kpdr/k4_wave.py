"""K4 Wave branch pure controllers — Bubble → Single → Double → Wave.

K4.8 Bubble → Single Chamber (``0xAD5E``): post-Speed return Bubble top-right
→ drop shaft → middle-right blue door into Single left shaft.

K4.9 Single → Double Chamber (``0xADAD``): left-shaft top entry → mid ledge →
floor platform y≈395 → missile red door (Second Top Right) into Double top.

K4.10 Double Chamber → Wave Beam PLM (``0xADDE``): top-left pin → upper path
→ blue gate → right Super/missile door → Wave chozo collect (beam bit 0x0001).

Human reference: ``tasks/speed_to_ice_moat_human.json`` frames 2637–3303
(Single segment). Caps: Morph, Bombs, Missiles, Supers, Hi-Jump, Varia, Speed.
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
from super_metroid.routes.kpdr.rooms import (
    ROOM_BUBBLE,
    ROOM_DOUBLE_CHAMBER,
    ROOM_SINGLE_CHAMBER,
    ROOM_WAVE,
)
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import (
    escape_knockback_spin,
    is_knockback,
)

# Wave Beam collected bit on ``collected_beams`` / ``equipped_beams``.
WAVE_BEAM_MASK = 0x0001

# ---------------------------------------------------------------------------
# K4.8 Bubble → Single Chamber
# Live pins from post_speed_return_to_bubble_pure + human Bubble→SC (2026-08-06).
# Top settle ~(472,115); drop band x≈381; door sill ~(492,395).
# ---------------------------------------------------------------------------
_TOP_Y_MAX = 200
_DROP_X = (370, 400)
_DROP_TARGET_X = 385
_MID_Y = (220, 340)
_FLOOR_Y = 360
_DOOR_Y = (380, 420)
_DOOR_X = 470
_SINGLE_SETTLE = 320
_TOTAL_BUDGET = 5000
_TOP_WALK_FRAMES = 400
_DROP_FRAMES = 500
_NAV_TO_DOOR_FRAMES = 1200
_DOOR_PUSH_FRAMES = 400

# ---------------------------------------------------------------------------
# K4.9 Single → Double Chamber
# Live pure (2026-08-06): top→mid y267→floor y395 → stationary missiles →
# spin-hop gap → RIGHT into Double ``0xADAD``. Upper door is red (missiles).
# ---------------------------------------------------------------------------
_SC_TOP_Y = 200
_SC_MID_Y = (250, 290)
_SC_FLOOR_Y = (380, 420)
_SC_SHOT_X = (115, 135)
_SC_DOOR_X = 220
_DOUBLE_SETTLE = 320
_SC_TOTAL_BUDGET = 5000


def _escape_kb(session: ControllerSession, label: str, prefer: str) -> None:
    escape_knockback_spin(
        session,
        prefer_dir=prefer,
        run_frames=6,
        spin_frames=18,
        label=label,
        stop_room_id=ROOM_SINGLE_CHAMBER,
    )


def _escape_kb_sc(session: ControllerSession, label: str, prefer: str) -> None:
    escape_knockback_spin(
        session,
        prefer_dir=prefer,
        run_frames=6,
        spin_frames=18,
        label=label,
        stop_room_id=ROOM_DOUBLE_CHAMBER,
    )


# ---- Bubble → Single -------------------------------------------------------


def _top_walk_to_drop(session: ControllerSession, label: str) -> None:
    """Top-right settle → walk LEFT to drop shaft band x∈[370,400]."""
    unmorph(session)
    select_weapon(session, 0)
    for _ in range(30):
        state = hold(session, 1, reason=f"{label}_top_stand")
        if state.velocity_y == 0 and state.pose in _STANDING_POSES:
            break

    for frame in range(_TOP_WALK_FRAMES):
        state = session.state
        if state.room_id != ROOM_BUBBLE:
            return
        if state.samus_y > _TOP_Y_MAX:
            return  # already dropping / below top
        if (
            _DROP_X[0] <= state.samus_x <= _DROP_X[1]
            and state.velocity_y == 0
        ):
            return
        if is_knockback(state):
            _escape_kb(session, label, "LEFT")
            continue
        # Near drop band: short walk to center then stop.
        if state.samus_x < _DROP_X[0]:
            hold(session, 1, "RIGHT", reason=f"{label}_top_r")
        elif state.samus_x > _DROP_X[1]:
            phase = frame % 14
            if phase < 10:
                hold(session, 1, "LEFT", "B", reason=f"{label}_top_run")
            else:
                hold(session, 1, "LEFT", reason=f"{label}_top_walk")
        else:
            hold(session, 1, reason=f"{label}_top_seat")


def _drop_shaft(session: ControllerSession, label: str) -> None:
    """Drop from top band through right shaft to floor/mid y≥360."""
    for frame in range(_DROP_FRAMES):
        state = session.state
        if state.room_id != ROOM_BUBBLE:
            return
        if state.samus_y >= _FLOOR_Y and state.velocity_y == 0:
            return
        if is_knockback(state):
            _escape_kb(session, label, "LEFT")
            continue

        # Keep roughly over the shaft while falling / hopping off lip.
        if state.velocity_y == 0 and state.samus_y <= _TOP_Y_MAX:
            # Step off lip: brief LEFT then free fall.
            if state.samus_x > _DROP_TARGET_X + 8:
                hold(session, 1, "LEFT", reason=f"{label}_lip_left")
            elif state.samus_x < _DROP_X[0]:
                hold(session, 1, "RIGHT", reason=f"{label}_lip_right")
            else:
                # Nudge off edge / open air.
                phase = frame % 12
                if phase < 4:
                    hold(session, 1, "LEFT", reason=f"{label}_step_off")
                elif phase < 7:
                    hold(session, 1, "A", reason=f"{label}_lip_hop")
                else:
                    hold(session, 1, reason=f"{label}_lip_wait")
            continue

        # Air: slight left bias to land mid-right platforms (human ~x381).
        if state.samus_x > 400:
            hold(session, 1, "LEFT", reason=f"{label}_fall_l")
        elif state.samus_x < 350:
            hold(session, 1, "RIGHT", reason=f"{label}_fall_r")
        else:
            hold(session, 1, reason=f"{label}_fall")


def _nav_floor_to_door(session: ControllerSession, label: str) -> None:
    """Mid/floor platforms → right blue door sill ~(492,395)."""
    unmorph(session)
    select_weapon(session, 0)

    for frame in range(_NAV_TO_DOOR_FRAMES):
        state = session.state
        if state.room_id == ROOM_SINGLE_CHAMBER:
            return
        if state.room_id != ROOM_BUBBLE:
            return
        if is_knockback(state):
            _escape_kb(session, label, "RIGHT")
            continue

        on_door_sill = (
            state.samus_x >= _DOOR_X
            and _DOOR_Y[0] <= state.samus_y <= _DOOR_Y[1]
            and state.velocity_y == 0
        )
        if on_door_sill:
            return

        # Too high mid: drop further or hop toward right.
        if state.samus_y < _FLOOR_Y:
            if state.velocity_y == 0 and state.pose in _STANDING_POSES:
                # Mid platforms: human hops left once then runs right down.
                # Prefer right progress once below top.
                if state.samus_x < 360 and state.samus_y < 300:
                    # Short left hop onto solid mid ledge (human ~341,228).
                    hold(session, 1, "LEFT", "A", reason=f"{label}_mid_hop")
                elif state.samus_x < 420:
                    phase = frame % 18
                    if phase < 6:
                        hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_mid_spin")
                    elif phase < 12:
                        hold(session, 1, "RIGHT", "B", reason=f"{label}_mid_run")
                    else:
                        hold(session, 1, "RIGHT", reason=f"{label}_mid_walk")
                else:
                    # Over right, drop down.
                    hold(session, 1, "RIGHT", reason=f"{label}_mid_drop")
            else:
                # Air: drift right toward door column.
                if state.samus_x < 450:
                    hold(session, 1, "RIGHT", reason=f"{label}_air_r")
                else:
                    hold(session, 1, reason=f"{label}_air")
            continue

        # Floor band y≥360: run right toward door; hop small gaps.
        if state.samus_x < _DOOR_X:
            phase = frame % 22
            if phase < 6:
                hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_floor_hop")
            elif phase < 14:
                hold(session, 1, "RIGHT", "B", reason=f"{label}_floor_run")
            else:
                hold(session, 1, "RIGHT", reason=f"{label}_floor_walk")
        else:
            hold(session, 1, "RIGHT", reason=f"{label}_sill_nudge")


def _push_right_blue_door(session: ControllerSession, label: str) -> None:
    """Sill pressure: RIGHT+X + dash through middle-right blue door."""
    select_weapon(session, 0)
    for frame in range(_DOOR_PUSH_FRAMES):
        state = session.state
        if state.room_id == ROOM_SINGLE_CHAMBER:
            return
        if state.room_id != ROOM_BUBBLE:
            return
        if is_knockback(state):
            _escape_kb(session, label, "RIGHT")
            continue

        # Fell off sill: climb back.
        if state.samus_y > 430:
            hold(session, 1, "LEFT", "B", "A", reason=f"{label}_under_recover")
            continue
        if state.samus_x < _DOOR_X - 20:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_reapproach")
            continue

        phase = frame % 16
        if phase < 5:
            hold(session, 1, "RIGHT", "X", reason=f"{label}_door_shot")
        elif phase < 11:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_door_push")
        else:
            hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_door_spin")


def play_bubble_to_single_chamber(session: ControllerSession) -> SuperMetroidState:
    """Bubble Mountain (post-Speed return) → ordinary Single Chamber.

    Path: top-right → drop shaft ~x385 → floor sill → middle-right blue door
    into ``0xAD5E``. Caps: Morph, Bombs, Missiles, Supers, Hi-Jump, Varia, Speed.
    """
    label = "bubble_to_single_chamber"
    require_room(session, ROOM_BUBBLE, label)
    start = session.frame

    if session.state.samus_y <= _TOP_Y_MAX:
        _top_walk_to_drop(session, label)
        if session.state.room_id == ROOM_BUBBLE and session.state.samus_y < _FLOOR_Y:
            _drop_shaft(session, label)

    if session.state.room_id == ROOM_BUBBLE:
        _nav_floor_to_door(session, label)

    if session.state.room_id == ROOM_BUBBLE:
        _push_right_blue_door(session, label)

    if session.state.room_id != ROOM_SINGLE_CHAMBER:
        state = session.state
        raise TimeoutError(
            f"{label}: Single Chamber door missed; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"door_transition={state.door_transition} "
            f"frames={session.frame - start}"
        )

    state = wait_ordinary_room(
        session, ROOM_SINGLE_CHAMBER, settle_frames=_SINGLE_SETTLE, label=label
    )
    if session.frame - start > _TOTAL_BUDGET:
        pass  # soft budget; room success is the gate
    return state


# ---- Single → Double -------------------------------------------------------


def _sc_descend_to_floor(session: ControllerSession, label: str) -> None:
    """Top ~(39,139) → mid y≈267 → floor platform y≈395 at missile seat.

    Live pure (2026-08-06): walk RIGHT to ~130, fall to mid, LEFT to x≈60,
    drop with RIGHT drift to land ~(75–100,395), walk to shot seat ~x124.
    """
    unmorph(session)
    select_weapon(session, 0)
    for _ in range(20):
        state = hold(session, 1, reason=f"{label}_top_stand")
        if state.velocity_y == 0 and state.pose in _STANDING_POSES:
            break

    # --- Top walk RIGHT with beam ---
    for frame in range(120):
        state = session.state
        if state.room_id != ROOM_SINGLE_CHAMBER:
            return
        if state.samus_y > _SC_TOP_Y:
            break
        if is_knockback(state):
            _escape_kb_sc(session, label, "LEFT")
            continue
        if state.samus_x < 130:
            if frame % 12 < 4:
                hold(session, 1, "RIGHT", "X", reason=f"{label}_top_shot")
            else:
                hold(session, 1, "RIGHT", reason=f"{label}_top_walk")
        else:
            hold(session, 1, "RIGHT", reason=f"{label}_top_edge")

    # --- Fall to mid ledge (LEFT bias if past x150) ---
    for _ in range(140):
        state = session.state
        if state.room_id != ROOM_SINGLE_CHAMBER:
            return
        if (
            _SC_MID_Y[0] <= state.samus_y <= _SC_MID_Y[1]
            and state.velocity_y == 0
        ):
            break
        if (
            _SC_FLOOR_Y[0] <= state.samus_y <= _SC_FLOOR_Y[1]
            and state.velocity_y == 0
        ):
            return  # skipped mid — already on door floor
        if is_knockback(state):
            _escape_kb_sc(session, label, "LEFT")
            continue
        if state.samus_x > 150:
            hold(session, 1, "LEFT", reason=f"{label}_air_l")
        else:
            hold(session, 1, reason=f"{label}_air")

    # --- Mid walk LEFT to drop column ~x60, then step off ---
    for frame in range(100):
        state = session.state
        if state.room_id != ROOM_SINGLE_CHAMBER:
            return
        if state.samus_y > _SC_MID_Y[1] + 10:
            break  # already dropping
        if (
            _SC_FLOOR_Y[0] <= state.samus_y <= _SC_FLOOR_Y[1]
            and state.velocity_y == 0
        ):
            break
        if is_knockback(state):
            _escape_kb_sc(session, label, "LEFT")
            continue
        if state.samus_x > 62:
            if frame % 10 < 3:
                hold(session, 1, "LEFT", "X", reason=f"{label}_mid_shot")
            else:
                hold(session, 1, "LEFT", reason=f"{label}_mid_walk")
        else:
            # At drop column: release walk and fall (do not keep LEFT into wall).
            break

    # --- Drop to floor with RIGHT drift → land ~x75–100 ---
    for frame in range(140):
        state = session.state
        if state.room_id != ROOM_SINGLE_CHAMBER:
            return
        if (
            _SC_FLOOR_Y[0] <= state.samus_y <= _SC_FLOOR_Y[1]
            and state.velocity_y == 0
        ):
            break
        if is_knockback(state):
            _escape_kb_sc(session, label, "RIGHT")
            continue
        # Still seated on mid: nudge off lip once.
        if (
            _SC_MID_Y[0] <= state.samus_y <= _SC_MID_Y[1]
            and state.velocity_y == 0
        ):
            if frame < 8:
                hold(session, 1, "LEFT", reason=f"{label}_step_off")
            else:
                hold(session, 1, reason=f"{label}_lip_wait")
            continue
        if state.samus_x < 75:
            hold(session, 1, "RIGHT", reason=f"{label}_floor_drift_r")
        elif state.samus_x > 100:
            hold(session, 1, "LEFT", reason=f"{label}_floor_drift_l")
        else:
            hold(session, 1, reason=f"{label}_floor_fall")

    # --- Floor walk to missile seat ---
    unmorph(session)
    for _ in range(60):
        state = session.state
        if state.room_id != ROOM_SINGLE_CHAMBER:
            return
        if state.room_id == ROOM_DOUBLE_CHAMBER:
            return
        if is_knockback(state):
            _escape_kb_sc(session, label, "RIGHT")
            continue
        if not (
            _SC_FLOOR_Y[0] <= state.samus_y <= _SC_FLOOR_Y[1]
            and state.velocity_y == 0
        ):
            # Overshot deep or still air — stop; door open may recover.
            if state.samus_y > _SC_FLOOR_Y[1] + 40:
                return
            hold(session, 1, reason=f"{label}_floor_wait")
            continue
        if _SC_SHOT_X[0] <= state.samus_x <= _SC_SHOT_X[1]:
            return
        if state.samus_x < _SC_SHOT_X[0]:
            hold(session, 1, "RIGHT", reason=f"{label}_seat_r")
        else:
            hold(session, 1, "LEFT", reason=f"{label}_seat_l")


def _sc_missile_door_and_enter(session: ControllerSession, label: str) -> None:
    """Stationary missiles open upper red door; spin-hop gap; RIGHT into Double.

    Live pure (2026-08-06): seat ~x124, ~100f missile volley, short walk to
    ~x145, 12f RIGHT+B+A gap hop, then hold RIGHT into ``0xADAD``.
    """
    unmorph(session)
    select_weapon(session, 1)

    # Face right without walking far off the seat.
    hold(session, 3, "RIGHT", reason=f"{label}_face")
    hold(session, 8, reason=f"{label}_face_release")

    # Stationary missile volley (human ~90–120f at x≈124).
    for frame in range(110):
        state = session.state
        if state.room_id == ROOM_DOUBLE_CHAMBER:
            return
        if state.room_id != ROOM_SINGLE_CHAMBER:
            return
        if is_knockback(state):
            _escape_kb_sc(session, label, "RIGHT")
            select_weapon(session, 1)
            continue
        # Keep seat; do not walk during volley.
        if state.samus_x > _SC_SHOT_X[1] + 20 and state.velocity_y == 0:
            hold(session, 1, "LEFT", reason=f"{label}_reseat")
            continue
        if state.samus_x < _SC_SHOT_X[0] - 15 and state.velocity_y == 0:
            hold(session, 1, "RIGHT", reason=f"{label}_reseat_r")
            continue
        if frame % 10 < 2:
            hold(session, 1, "X", reason=f"{label}_missile")
        else:
            hold(session, 1, reason=f"{label}_missile_wait")

    # Fuse / door open settle.
    hold(session, 12, reason=f"{label}_fuse")

    # Short walk-up on solid floor before the gap (live GREEN ~x145).
    for _ in range(30):
        state = session.state
        if state.room_id != ROOM_SINGLE_CHAMBER:
            return
        if state.samus_x >= 145 and state.velocity_y == 0:
            break
        if state.samus_y > _SC_FLOOR_Y[1] + 20:
            break
        hold(session, 1, "RIGHT", reason=f"{label}_walkup")

    # One spin-hop across the gap, then commit RIGHT (no mid-air rehop spam).
    for frame in range(12):
        state = session.state
        if state.room_id == ROOM_DOUBLE_CHAMBER:
            return
        if state.room_id != ROOM_SINGLE_CHAMBER:
            return
        hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_gap_hop")

    for frame in range(260):
        state = session.state
        if state.room_id == ROOM_DOUBLE_CHAMBER:
            return
        if state.room_id != ROOM_SINGLE_CHAMBER:
            return
        if is_knockback(state):
            _escape_kb_sc(session, label, "RIGHT")
            continue

        # Deep shaft: abort hop spam; try to get back left onto something solid.
        if state.samus_y > _SC_FLOOR_Y[1] + 50:
            if state.samus_x > 100:
                hold(session, 1, "LEFT", reason=f"{label}_under_left")
            else:
                hold(session, 1, "LEFT", "A", reason=f"{label}_under_up")
            continue

        # Airborne: hold RIGHT only (let spin carry).
        if state.velocity_y != 0 or state.samus_y < _SC_FLOOR_Y[0] - 5:
            hold(session, 1, "RIGHT", reason=f"{label}_air_r")
            continue

        # Grounded short of door: run; occasional re-missile if blocked at wall.
        if state.samus_x < _SC_DOOR_X:
            if frame > 0 and frame % 90 == 0:
                select_weapon(session, 1)
                hold(session, 2, "RIGHT", "X", reason=f"{label}_reopen")
                hold(session, 20, reason=f"{label}_reopen_fuse")
                continue
            hold(session, 1, "RIGHT", "B", reason=f"{label}_run")
        else:
            hold(session, 1, "RIGHT", reason=f"{label}_door_push")


def play_single_to_double_chamber(session: ControllerSession) -> SuperMetroidState:
    """Single Chamber (post Bubble→Single pure) → ordinary Double Chamber.

    Path: left-shaft top → mid ledge → floor y≈395 → missile red door (upper)
    into Double Chamber ``0xADAD``. Caps include missiles.
    """
    label = "single_to_double_chamber"
    require_room(session, ROOM_SINGLE_CHAMBER, label)
    start = session.frame

    if session.state.room_id == ROOM_SINGLE_CHAMBER:
        _sc_descend_to_floor(session, label)

    if session.state.room_id == ROOM_SINGLE_CHAMBER:
        _sc_missile_door_and_enter(session, label)

    if session.state.room_id != ROOM_DOUBLE_CHAMBER:
        state = session.state
        raise TimeoutError(
            f"{label}: Double Chamber door missed; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"door_transition={state.door_transition} "
            f"missiles={state.missiles} selected={state.selected_item} "
            f"frames={session.frame - start}"
        )

    state = wait_ordinary_room(
        session, ROOM_DOUBLE_CHAMBER, settle_frames=_DOUBLE_SETTLE, label=label
    )
    if session.frame - start > _SC_TOTAL_BUDGET:
        pass
    return state


# ---------------------------------------------------------------------------
# K4.10 Double Chamber → Wave Beam PLM
# Live recon (2026-08-06): entry ~(61,139); upper hop path; blue gate ~x410
# is the open RED pin (shots register visually but PLM may not trip from pure
# pin geometry). Past-gate solid ~(520,140); right Super door → Wave chozo
# collect beams|=0x0001 at ~(171,120). Caps: missiles/supers for red door.
# ---------------------------------------------------------------------------
_DC_TOTAL_BUDGET = 9000
_WAVE_SETTLE = 280
_GATE_X = (360, 430)
_PAST_GATE_X = 480
_DOOR_X = 920
_DOOR_Y_MAX = 180


def _escape_kb_dc(session: ControllerSession, label: str, prefer: str) -> None:
    escape_knockback_spin(
        session,
        prefer_dir=prefer,
        run_frames=6,
        spin_frames=18,
        label=label,
        stop_room_id=ROOM_WAVE,
    )


def _has_wave(state: SuperMetroidState) -> bool:
    return bool(int(state.collected_beams) & WAVE_BEAM_MASK)


def _dc_hop_to_gate_zone(session: ControllerSession, label: str) -> None:
    """Top-left ~(61,139) → upper platforms → gate band x∈[360,430] y≲200.

    Live recon cadence: hop_run to x≈210, then spin16_run12 toward gate.
    """
    unmorph(session)
    select_weapon(session, 0)
    for _ in range(20):
        state = hold(session, 1, reason=f"{label}_top_stand")
        if state.velocity_y == 0 and state.pose in _STANDING_POSES:
            break

    # hop_run toward mid platforms
    for frame in range(160):
        state = session.state
        if state.room_id != ROOM_DOUBLE_CHAMBER:
            return
        if state.samus_x >= 210 and state.velocity_y == 0 and state.samus_y < 200:
            break
        if is_knockback(state):
            _escape_kb_dc(session, label, "RIGHT")
            continue
        if state.pose in (137, 138):
            unmorph(session)
            continue
        phase = frame % 30
        if phase < 4:
            hold(session, 1, "RIGHT", "X", reason=f"{label}_hop_shot")
        elif phase < 12:
            hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_hop_spin")
        elif phase < 22:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_hop_run")
        else:
            hold(session, 1, "RIGHT", reason=f"{label}_hop_walk")

    # spin16_run12 toward gate / high platforms
    for frame in range(280):
        state = session.state
        if state.room_id != ROOM_DOUBLE_CHAMBER:
            return
        if (
            _GATE_X[0] <= state.samus_x <= _GATE_X[1]
            and state.samus_y < 200
            and state.velocity_y == 0
        ):
            return
        if state.samus_x >= _PAST_GATE_X and state.samus_y < 220:
            return
        if state.samus_y > 360 and state.velocity_y == 0:
            return  # fell; door phase may still recover poorly
        if is_knockback(state):
            _escape_kb_dc(session, label, "RIGHT")
            continue
        if state.pose in (137, 138):
            unmorph(session)
            continue
        phase = frame % 28
        if phase < 16:
            hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_gate_spin")
        else:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_gate_run")


def _dc_open_blue_gate(session: ControllerSession, label: str) -> None:
    """Open mid blue gate (obstacle A) and push onto past-gate platform.

    Switch is the top mechanism; prefer UP+RIGHT beam while high (y≲140).
    Live pure (2026-08-06): power beam selected; residual if PLM does not trip.
    """
    select_weapon(session, 0)
    for frame in range(400):
        state = session.state
        if state.room_id != ROOM_DOUBLE_CHAMBER:
            return
        if state.samus_x >= _PAST_GATE_X and state.samus_y < 220:
            return
        if state.samus_y > 360 and state.velocity_y == 0:
            return
        if is_knockback(state):
            _escape_kb_dc(session, label, "RIGHT")
            select_weapon(session, 0)
            continue
        if state.pose in (137, 138):
            unmorph(session)
            continue

        high = state.samus_y < 145
        in_gate = _GATE_X[0] <= state.samus_x <= _GATE_X[1]

        if high and in_gate:
            phase = frame % 14
            if phase < 4:
                hold(session, 1, "UP", "RIGHT", "X", reason=f"{label}_switch")
            elif phase < 7:
                hold(session, 1, "RIGHT", "X", reason=f"{label}_beam")
            elif phase < 11:
                hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_gate_hop")
            else:
                hold(session, 1, "RIGHT", reason=f"{label}_gate_nudge")
        elif high:
            phase = frame % 12
            if phase < 3:
                hold(session, 1, "UP", "RIGHT", "X", reason=f"{label}_hi_shot")
            else:
                hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_hi_spin")
        elif state.velocity_y != 0:
            hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_gate_air")
        else:
            # Low: try re-hop onto upper path
            phase = frame % 20
            if phase < 10:
                hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_recover")
            elif phase < 14:
                hold(session, 1, "X", reason=f"{label}_low_shot")
            else:
                hold(session, 1, "RIGHT", "B", reason=f"{label}_low_run")


def _dc_to_wave_door(session: ControllerSession, label: str) -> None:
    """Past-gate platforms → right red door (Supers open red) → Wave room."""
    for frame in range(1200):
        state = session.state
        if state.room_id == ROOM_WAVE:
            return
        if state.room_id != ROOM_DOUBLE_CHAMBER:
            return
        if is_knockback(state):
            _escape_kb_dc(session, label, "RIGHT")
            continue
        if state.pose in (137, 138):
            unmorph(session)
            continue

        x, y = state.samus_x, state.samus_y
        near_door = x >= _DOOR_X and y < _DOOR_Y_MAX

        if near_door:
            # Red door: Supers (also open missile doors).
            select_weapon(session, 2)
            if state.velocity_y == 0:
                if frame % 36 < 3:
                    hold(session, 1, "RIGHT", "X", reason=f"{label}_super")
                elif frame % 36 < 12:
                    hold(session, 1, reason=f"{label}_super_fuse")
                else:
                    hold(session, 1, "RIGHT", "B", reason=f"{label}_door_push")
            else:
                hold(session, 1, "RIGHT", reason=f"{label}_door_air")
            continue

        select_weapon(session, 0)
        if x >= 750:
            # Right structure walljump climb to door sill y≈140.
            if y > _DOOR_Y_MAX:
                phase = frame % 14
                if phase < 5:
                    hold(session, 1, "LEFT", "A", reason=f"{label}_wj_l")
                elif phase < 10:
                    hold(session, 1, "RIGHT", "A", reason=f"{label}_wj_r")
                else:
                    hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_wj_up")
            else:
                hold(session, 1, "RIGHT", "B", reason=f"{label}_sill_run")
            continue

        if state.velocity_y != 0:
            hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_mid_air")
        else:
            phase = frame % 26
            if phase < 14:
                hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_mid_spin")
            elif phase < 20:
                hold(session, 1, "RIGHT", "B", reason=f"{label}_mid_run")
            else:
                hold(session, 1, "RIGHT", "X", reason=f"{label}_mid_shot")


def _wave_collect_plm(session: ControllerSession, label: str) -> SuperMetroidState:
    """Wave Room left entry → chozo PLM → beam bit 0x0001."""
    require_room(session, ROOM_WAVE, label)
    if _has_wave(session.state):
        return session.state

    unmorph(session)
    select_weapon(session, 0)
    for _ in range(30):
        state = hold(session, 1, reason=f"{label}_stand")
        if state.velocity_y == 0 and state.pose in _STANDING_POSES:
            break
        if state.pose in (137, 138, 39, 40):
            hold(session, 1, "UP", reason=f"{label}_unmorph")

    for frame in range(500):
        state = session.state
        if _has_wave(state):
            break
        if state.room_id != ROOM_WAVE:
            raise TimeoutError(
                f"{label}: left Wave Room during collect; "
                f"room=0x{state.room_id:04X} xy=({state.samus_x},{state.samus_y})"
            )
        if state.pose in (137, 138):
            hold(session, 8, "UP", reason=f"{label}_unmorph")
            continue
        if state.samus_x < 160:
            phase = frame % 20
            if phase < 8:
                hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_chozo_hop")
            elif phase < 14:
                hold(session, 1, "RIGHT", "B", reason=f"{label}_chozo_run")
            else:
                hold(session, 1, "RIGHT", "X", reason=f"{label}_chozo_shot")
        else:
            if frame % 10 == 0:
                hold(session, 1, "X", reason=f"{label}_plm_shot")
            else:
                hold(session, 1, "RIGHT", reason=f"{label}_plm_walk")
    else:
        state = session.state
        raise TimeoutError(
            f"{label}: Wave PLM not collected; beams=0x{state.collected_beams:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y})"
        )

    hold(session, 80, reason=f"{label}_fanfare")
    unmorph(session)
    for _ in range(40):
        state = hold(session, 1, reason=f"{label}_post_stand")
        if state.velocity_y == 0 and state.pose in _STANDING_POSES:
            break
    return session.state


def play_double_chamber_to_wave(session: ControllerSession) -> SuperMetroidState:
    """Double Chamber (post Single→Double pure) → Wave Beam PLM collect.

    Path: top-left ~(61,139) → upper hop path → blue gate → right Super door
    into Wave ``0xADDE`` → chozo collect (``WAVE_BEAM_MASK`` 0x0001).

    Caps: Morph, Bombs, Missiles, Supers, Hi-Jump, Varia, Speed.
    """
    label = "double_chamber_to_wave"
    require_room(session, ROOM_DOUBLE_CHAMBER, label)
    start = session.frame

    if _has_wave(session.state) and session.state.room_id == ROOM_WAVE:
        return session.state

    if session.state.room_id == ROOM_DOUBLE_CHAMBER:
        _dc_hop_to_gate_zone(session, label)

    if (
        session.state.room_id == ROOM_DOUBLE_CHAMBER
        and session.state.samus_x < _PAST_GATE_X
    ):
        _dc_open_blue_gate(session, label)

    if session.state.room_id == ROOM_DOUBLE_CHAMBER:
        _dc_to_wave_door(session, label)

    if session.state.room_id != ROOM_WAVE:
        state = session.state
        raise TimeoutError(
            f"{label}: Wave door missed; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"door_transition={state.door_transition} "
            f"missiles={state.missiles} supers={state.super_missiles} "
            f"selected={state.selected_item} "
            f"beams=0x{state.collected_beams:04X} "
            f"frames={session.frame - start}"
        )

    wait_ordinary_room(
        session, ROOM_WAVE, settle_frames=_WAVE_SETTLE, label=label
    )
    state = _wave_collect_plm(session, label)

    if not _has_wave(state):
        raise TimeoutError(
            f"{label}: finished without Wave bit; "
            f"beams=0x{state.collected_beams:04X} room=0x{state.room_id:04X} "
            f"xy=({state.samus_x},{state.samus_y}) "
            f"frames={session.frame - start}"
        )
    if session.frame - start > _DC_TOTAL_BUDGET:
        pass
    return state


__all__ = [
    "WAVE_BEAM_MASK",
    "play_bubble_to_single_chamber",
    "play_single_to_double_chamber",
    "play_double_chamber_to_wave",
    "ROOM_BUBBLE",
    "ROOM_SINGLE_CHAMBER",
    "ROOM_DOUBLE_CHAMBER",
    "ROOM_WAVE",
]

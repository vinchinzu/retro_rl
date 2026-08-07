"""K4.9 Single Chamber → Double Chamber pure controller."""

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
from super_metroid.routes.kpdr.rooms import ROOM_DOUBLE_CHAMBER, ROOM_SINGLE_CHAMBER
from super_metroid.routes.kpdr.wave.geometry import (
    SC_DOOR_X,
    SC_DOUBLE_SETTLE,
    SC_FLOOR_Y,
    SC_MID_Y,
    SC_SHOT_X,
    SC_TOP_Y,
)
from super_metroid.routes.skills.knockback import escape_kb, is_knockback
from super_metroid.routes.runtime import ControllerSession


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
        if state.samus_y > SC_TOP_Y:
            break
        if is_knockback(state):
            escape_kb(session, label, "LEFT", stop_room_id=ROOM_DOUBLE_CHAMBER)
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
            SC_MID_Y[0] <= state.samus_y <= SC_MID_Y[1]
            and state.velocity_y == 0
        ):
            break
        if (
            SC_FLOOR_Y[0] <= state.samus_y <= SC_FLOOR_Y[1]
            and state.velocity_y == 0
        ):
            return  # skipped mid — already on door floor
        if is_knockback(state):
            escape_kb(session, label, "LEFT", stop_room_id=ROOM_DOUBLE_CHAMBER)
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
        if state.samus_y > SC_MID_Y[1] + 10:
            break  # already dropping
        if (
            SC_FLOOR_Y[0] <= state.samus_y <= SC_FLOOR_Y[1]
            and state.velocity_y == 0
        ):
            break
        if is_knockback(state):
            escape_kb(session, label, "LEFT", stop_room_id=ROOM_DOUBLE_CHAMBER)
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
            SC_FLOOR_Y[0] <= state.samus_y <= SC_FLOOR_Y[1]
            and state.velocity_y == 0
        ):
            break
        if is_knockback(state):
            escape_kb(session, label, "RIGHT", stop_room_id=ROOM_DOUBLE_CHAMBER)
            continue
        # Still seated on mid: nudge off lip once.
        if (
            SC_MID_Y[0] <= state.samus_y <= SC_MID_Y[1]
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
            escape_kb(session, label, "RIGHT", stop_room_id=ROOM_DOUBLE_CHAMBER)
            continue
        if not (
            SC_FLOOR_Y[0] <= state.samus_y <= SC_FLOOR_Y[1]
            and state.velocity_y == 0
        ):
            # Overshot deep or still air — stop; door open may recover.
            if state.samus_y > SC_FLOOR_Y[1] + 40:
                return
            hold(session, 1, reason=f"{label}_floor_wait")
            continue
        if SC_SHOT_X[0] <= state.samus_x <= SC_SHOT_X[1]:
            return
        if state.samus_x < SC_SHOT_X[0]:
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
            escape_kb(session, label, "RIGHT", stop_room_id=ROOM_DOUBLE_CHAMBER)
            select_weapon(session, 1)
            continue
        # Keep seat; do not walk during volley.
        if state.samus_x > SC_SHOT_X[1] + 20 and state.velocity_y == 0:
            hold(session, 1, "LEFT", reason=f"{label}_reseat")
            continue
        if state.samus_x < SC_SHOT_X[0] - 15 and state.velocity_y == 0:
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
        if state.samus_y > SC_FLOOR_Y[1] + 20:
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
            escape_kb(session, label, "RIGHT", stop_room_id=ROOM_DOUBLE_CHAMBER)
            continue

        # Deep shaft: abort hop spam; try to get back left onto something solid.
        if state.samus_y > SC_FLOOR_Y[1] + 50:
            if state.samus_x > 100:
                hold(session, 1, "LEFT", reason=f"{label}_under_left")
            else:
                hold(session, 1, "LEFT", "A", reason=f"{label}_under_up")
            continue

        # Airborne: hold RIGHT only (let spin carry).
        if state.velocity_y != 0 or state.samus_y < SC_FLOOR_Y[0] - 5:
            hold(session, 1, "RIGHT", reason=f"{label}_air_r")
            continue

        # Grounded short of door: run; occasional re-missile if blocked at wall.
        if state.samus_x < SC_DOOR_X:
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

    return wait_ordinary_room(
        session, ROOM_DOUBLE_CHAMBER, settle_frames=SC_DOUBLE_SETTLE, label=label
    )


__all__ = ["play_single_to_double_chamber"]

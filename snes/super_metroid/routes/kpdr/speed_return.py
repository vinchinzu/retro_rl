"""Speed Booster Room → Bubble Mountain pure return (K4.7).

Post-collect handoff ``0xAD1B`` ~(169,123) with Speed bit → reverse of the
outbound Bat→Hall→Speed spine:

* Speed Room left blue door → Speed Hall right lip
* LEFT+B dash across Speed Hall incline → left blue door
* Bat Cave top-right shelf → drop to floor → bottom-left blue door
* Ordinary Bubble Mountain ``0xACB3``

Human reference: ``tasks/speed_to_ice_moat_human.json`` frames 0–2131
(source ``scratch/post_speed_collected``). Caps: Morph, Bombs, Missiles,
Supers, Hi-Jump, Varia, Speed.
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
    ITEM_SPEED,
    ROOM_BAT_CAVE,
    ROOM_BUBBLE,
    ROOM_SPEED,
    ROOM_SPEED_HALL,
)
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import (
    escape_knockback_spin,
    is_knockback,
)

# Live pins from post_speed_collected + pure recon (2026-08-06).
_SPEED_DOOR_X = 40
_HALL_DASH_FRAMES = 900
_HALL_LEFT_DOOR_X = 80
_BAT_SETTLE = 280
# Reverse of outbound Bat climb (to_speed.py):
# * Shelf → cavity: morph bombs at x≈165–175
# * Cavity → floor: power DOWN+X at tight hole x∈[148,154] (same band as climb)
# * Floor → door: LEFT hop rhythm over lava gaps (not a flat walk)
_SHELF_BOMB_X = (160, 180)
_SHELF_BOMB_TARGET = 168
_HOLE_X = (148, 154)  # tight — outbound climb band
_HOLE_TARGET_X = 151
_SHELF_Y = 170
_CAVITY_Y = (220, 300)
_BAT_FLOOR_Y = 360
_LAVA_Y = 430
_BAT_DOOR_X = 45
_BUBBLE_SETTLE = 320
_TOTAL_BUDGET = 6000
_MORPH_BOMB_FRAMES = 320
_HOLE_OPEN_FRAMES = 400
_FLOOR_TO_DOOR_FRAMES = 900


def _leave_speed_room(session: ControllerSession, label: str) -> SuperMetroidState:
    """Chozo shelf → left blue door → ordinary Speed Hall right lip."""
    require_room(session, ROOM_SPEED, label)
    if not (session.state.collected_items & ITEM_SPEED):
        raise RuntimeError(
            f"{label}: Speed not collected "
            f"(items=0x{session.state.collected_items:04X})"
        )

    unmorph(session)
    select_weapon(session, 0)
    hold(session, 8, reason=f"{label}_stand")

    # Walk/dash left off the chozo shelf toward the blue door.
    for frame in range(280):
        state = session.state
        if state.room_id == ROOM_SPEED_HALL:
            break
        if state.room_id != ROOM_SPEED:
            break
        if is_knockback(state):
            escape_knockback_spin(
                session,
                prefer_dir="LEFT",
                run_frames=6,
                spin_frames=20,
                label=label,
                stop_room_id=ROOM_SPEED_HALL,
            )
            continue
        if state.pose in (137, 138):
            unmorph(session)
            continue

        # Near the door: shoot + push through.
        if state.samus_x <= _SPEED_DOOR_X and state.velocity_y == 0:
            phase = frame % 16
            if phase < 4:
                hold(session, 1, "LEFT", "X", reason=f"{label}_door_shot")
            elif phase < 10:
                hold(session, 1, "LEFT", "B", reason=f"{label}_door_push")
            else:
                hold(session, 1, "LEFT", "B", "A", reason=f"{label}_door_spin")
            continue

        phase = frame % 18
        if phase < 10:
            hold(session, 1, "LEFT", "B", reason=f"{label}_exit_run")
        elif phase < 14:
            hold(session, 1, "LEFT", reason=f"{label}_exit_walk")
        else:
            hold(session, 1, "LEFT", "X", reason=f"{label}_exit_shot")
    else:
        state = session.state
        raise TimeoutError(
            f"{label}: left Speed door missed; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y})"
        )

    return wait_ordinary_room(
        session, ROOM_SPEED_HALL, settle_frames=280, label=label
    )


def _dash_speed_hall_return(
    session: ControllerSession, label: str
) -> SuperMetroidState:
    """Right lip → LEFT+B across incline → ordinary Bat Cave top-right."""
    require_room(session, ROOM_SPEED_HALL, label)

    for _ in range(30):
        state = hold(session, 1, reason=f"{label}_hall_land")
        if state.velocity_y == 0 and state.pose in _STANDING_POSES:
            break

    select_weapon(session, 0)
    min_x = session.state.samus_x
    for frame in range(_HALL_DASH_FRAMES):
        state = hold(session, 1, "LEFT", "B", reason=f"{label}_hall_dash")
        min_x = min(min_x, state.samus_x)
        if state.room_id == ROOM_BAT_CAVE:
            break
        if state.room_id != ROOM_SPEED_HALL:
            break
        # At left door band: add shot pulses for the blue door.
        if state.samus_x <= _HALL_LEFT_DOOR_X:
            if frame % 12 < 3:
                hold(session, 1, "LEFT", "X", reason=f"{label}_hall_door_shot")
            else:
                hold(session, 1, "LEFT", "B", reason=f"{label}_hall_door_push")
            if session.state.room_id == ROOM_BAT_CAVE:
                break
    else:
        state = session.state
        raise TimeoutError(
            f"{label}: Speed Hall left door missed; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) min_x={min_x}"
        )

    if session.state.room_id != ROOM_BAT_CAVE:
        # Extra door attempt if still on left lip.
        for frame in range(200):
            state = hold(session, 1, "LEFT", "B", reason=f"{label}_hall_retry")
            if state.room_id == ROOM_BAT_CAVE:
                break
            if frame % 20 == 0:
                hold(session, 2, "LEFT", "X", reason=f"{label}_hall_retry_shot")
        else:
            state = session.state
            raise TimeoutError(
                f"{label}: Bat Cave not reached from Hall; "
                f"room=0x{state.room_id:04X} pose={state.pose} "
                f"xy=({state.samus_x},{state.samus_y})"
            )

    return wait_ordinary_room(
        session, ROOM_BAT_CAVE, settle_frames=_BAT_SETTLE, label=label
    )


def _in_hole_band(state: SuperMetroidState) -> bool:
    return _HOLE_X[0] <= state.samus_x <= _HOLE_X[1]


def _in_cavity(state: SuperMetroidState) -> bool:
    return _CAVITY_Y[0] <= state.samus_y <= _CAVITY_Y[1]


def _ensure_morph(session: ControllerSession, label: str) -> None:
    """Double-tap DOWN morph (morph poses 27–70 range / ball)."""
    for _ in range(4):
        pose = session.state.pose
        # Morph ball / spring / related crouch-ball poses.
        if pose in (27, 28, 49, 50, 55, 56, 65, 129, 130, 131, 132):
            return
        hold(session, 5, "DOWN", reason=f"{label}_morph1")
        hold(session, 4, reason=f"{label}_morph_gap")
        hold(session, 5, "DOWN", reason=f"{label}_morph2")
        hold(session, 8, reason=f"{label}_morph_settle")


def _shelf_bomb_to_cavity(session: ControllerSession, label: str) -> None:
    """Door shelf → morph bombs at x≈168 → mid cavity ~(168,251)."""
    # Walk into bomb band from top-right door (~x=200+).
    for _ in range(120):
        state = session.state
        if state.room_id != ROOM_BAT_CAVE:
            return
        if state.samus_y >= _CAVITY_Y[0]:
            return
        if is_knockback(state):
            escape_knockback_spin(
                session,
                prefer_dir="LEFT",
                run_frames=4,
                spin_frames=14,
                label=label,
                stop_room_id=ROOM_BUBBLE,
            )
            continue
        if (
            _SHELF_BOMB_X[0] <= state.samus_x <= _SHELF_BOMB_X[1]
            and state.velocity_y == 0
        ):
            break
        face = "LEFT" if state.samus_x > _SHELF_BOMB_TARGET else "RIGHT"
        hold(session, 1, face, reason=f"{label}_to_bomb")

    _ensure_morph(session, label)
    for i in range(_MORPH_BOMB_FRAMES):
        state = session.state
        if state.room_id != ROOM_BAT_CAVE:
            return
        if state.samus_y >= _CAVITY_Y[0]:
            return
        if is_knockback(state):
            unmorph(session)
            escape_knockback_spin(
                session,
                prefer_dir="LEFT",
                run_frames=4,
                spin_frames=12,
                label=label,
                stop_room_id=ROOM_BUBBLE,
            )
            _ensure_morph(session, label)
            continue
        # Bomb every ~40f (fuse); stay near bomb band.
        if i % 42 < 2:
            hold(session, 1, "X", reason=f"{label}_shelf_bomb")
        elif state.samus_x < _SHELF_BOMB_X[0]:
            hold(session, 1, "RIGHT", reason=f"{label}_bomb_r")
        elif state.samus_x > _SHELF_BOMB_X[1]:
            hold(session, 1, "LEFT", reason=f"{label}_bomb_l")
        else:
            hold(session, 1, reason=f"{label}_bomb_wait")


def _cavity_hole_to_floor(session: ControllerSession, label: str) -> None:
    """Cavity pocket → power DOWN+X at tight hole → floor y≈395."""
    unmorph(session)
    select_weapon(session, 0)
    for _ in range(25):
        hold(session, 1, reason=f"{label}_cavity_stand")
        if (
            session.state.velocity_y == 0
            and session.state.pose in _STANDING_POSES
        ):
            break

    for i in range(_HOLE_OPEN_FRAMES):
        state = session.state
        if state.room_id != ROOM_BAT_CAVE:
            return
        if state.samus_y >= _BAT_FLOOR_Y:
            return
        if is_knockback(state):
            escape_knockback_spin(
                session,
                prefer_dir="LEFT",
                run_frames=4,
                spin_frames=12,
                label=label,
                stop_room_id=ROOM_BUBBLE,
            )
            continue
        if state.velocity_y == 0 and state.samus_x < _HOLE_X[0]:
            hold(session, 1, "RIGHT", reason=f"{label}_hole_r")
            continue
        if state.velocity_y == 0 and state.samus_x > _HOLE_X[1]:
            hold(session, 1, "LEFT", reason=f"{label}_hole_l")
            continue
        phase = i % 20
        if phase < 10:
            hold(session, 1, "DOWN", "X", reason=f"{label}_hole_shot")
        elif phase < 14:
            hold(session, 1, "A", reason=f"{label}_hole_hop")
        else:
            hold(session, 1, "DOWN", "X", reason=f"{label}_hole_fall")


def _floor_to_bubble_door(session: ControllerSession, label: str) -> None:
    """Floor under hole → LEFT hop rhythm over lava → bottom-left blue door."""
    unmorph(session)
    select_weapon(session, 0)
    for _ in range(40):
        hold(session, 1, reason=f"{label}_floor_land")
        if (
            session.state.velocity_y == 0
            and session.state.samus_y < _LAVA_Y
            and session.state.pose in _STANDING_POSES
        ):
            break

    for frame in range(_FLOOR_TO_DOOR_FRAMES):
        state = session.state
        if state.room_id == ROOM_BUBBLE:
            return
        if state.room_id != ROOM_BAT_CAVE:
            return
        if is_knockback(state):
            escape_knockback_spin(
                session,
                prefer_dir="LEFT",
                run_frames=4,
                spin_frames=14,
                label=label,
                stop_room_id=ROOM_BUBBLE,
            )
            continue

        # Lava recover: spin-jump left toward door ledge.
        if state.samus_y >= _LAVA_Y:
            hold(session, 1, "LEFT", "B", "A", reason=f"{label}_lava")
            continue

        grounded = state.velocity_y == 0 and state.pose in _STANDING_POSES
        # Near left wall: drop to door sill if high, else shoot+push blue door.
        if state.samus_x <= _BAT_DOOR_X + 15:
            if state.samus_y < 380 and grounded:
                # Mid ledge near door — step/drop down to sill y≈395.
                hold(session, 1, "LEFT", reason=f"{label}_to_sill")
                if frame % 20 == 10:
                    hold(session, 2, "DOWN", reason=f"{label}_sill_drop")
                continue
            if state.samus_x <= _BAT_DOOR_X:
                phase = frame % 16
                if phase < 5:
                    hold(session, 1, "LEFT", "X", reason=f"{label}_bot_shot")
                elif phase < 11:
                    hold(session, 1, "LEFT", "B", reason=f"{label}_bot_push")
                else:
                    hold(session, 1, "LEFT", "B", "A", reason=f"{label}_bot_spin")
                continue

        # Live pure: hop rhythm clears lava gaps (flat LEFT falls in).
        phase = frame % 25
        if phase < 6:
            hold(session, 1, "LEFT", "B", "A", reason=f"{label}_floor_hop")
        elif phase < 14:
            hold(session, 1, "LEFT", "B", reason=f"{label}_floor_run")
        else:
            hold(session, 1, "LEFT", reason=f"{label}_floor_walk")


def _descend_bat_to_bubble(
    session: ControllerSession, label: str
) -> SuperMetroidState:
    """Top-right door shelf → cavity bombs → hole drop → hop door → Bubble.

    Live pure (2026-08-06): morph bombs open shelf→cavity; power DOWN+X at
    hole x∈[148,154] opens cavity→floor; LEFT hop rhythm reaches door.
    """
    require_room(session, ROOM_BAT_CAVE, label)
    unmorph(session)

    if session.state.samus_y < _CAVITY_Y[0]:
        _shelf_bomb_to_cavity(session, label)
    if (
        session.state.room_id == ROOM_BAT_CAVE
        and session.state.samus_y < _BAT_FLOOR_Y
    ):
        # If still on shelf, one more bomb pass then hole open.
        if session.state.samus_y < _CAVITY_Y[0]:
            _shelf_bomb_to_cavity(session, label)
        if session.state.samus_y < _BAT_FLOOR_Y:
            _cavity_hole_to_floor(session, label)

    if session.state.room_id == ROOM_BAT_CAVE:
        _floor_to_bubble_door(session, label)

    if session.state.room_id != ROOM_BUBBLE:
        state = session.state
        raise TimeoutError(
            f"{label}: Bat Cave bottom door missed; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"door_transition={state.door_transition}"
        )

    return wait_ordinary_room(
        session, ROOM_BUBBLE, settle_frames=_BUBBLE_SETTLE, label=label
    )


def play_speed_return_to_bubble(session: ControllerSession) -> SuperMetroidState:
    """Speed Room post-collect → ordinary Bubble Mountain via Hall + Bat.

    Path: left blue out of Speed → LEFT+B reverse Hall → Bat top-right drop
    → bottom-left blue into Bubble. Caps: Morph, Bombs, Missiles, Supers,
    Hi-Jump, Varia, Speed.
    """
    label = "speed_return_to_bubble"
    require_room(session, ROOM_SPEED, label)
    start = session.frame

    _leave_speed_room(session, label)
    _dash_speed_hall_return(session, label)
    state = _descend_bat_to_bubble(session, label)

    if state.room_id != ROOM_BUBBLE:
        raise TimeoutError(
            f"{label}: finished outside Bubble; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"frames={session.frame - start}"
        )
    if session.frame - start > _TOTAL_BUDGET:
        pass  # soft budget; room success is the gate
    return state


__all__ = [
    "play_speed_return_to_bubble",
    "ROOM_BAT_CAVE",
    "ROOM_BUBBLE",
    "ROOM_SPEED",
    "ROOM_SPEED_HALL",
    "ITEM_SPEED",
]

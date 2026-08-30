"""Bat Cave → Speed Booster Hall pure controller (K4.5).

Source: ``scratch/post_bat_cave_continuous`` / ``post_bubble_to_bat_pure``
(room ``0xB07A``, left blue lip ~x=39 y=395).

Geometry (1×2 heated shaft, sm-json-data + live recon 2026-08-05):

* Bottom-left blue door (from Bubble) → top-right blue door (Speed Hall).
* Floor is segmented with lava gaps; door ledge ends ~x=66–80.
* Ceiling shot-block hole sits ~x=140–160 (underside ~y=323).
* Hi-Jump through hole lands mid pocket ~(171,251) — **not** the door shelf.
  That pocket has a solid ceiling at ~y=211 until another UP+X clear opens it.
* After cavity ceiling clear (~60f UP+X), jump peaks y≈99 onto door shelf
  (y≤160, x≥180 open-loops RIGHT → ordinary Speed Hall ``0xACF0``).

Caps: Morph, Bombs, Missiles, Supers, Hi-Jump, Varia — no Speed.
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
from super_metroid.routes.kpdr.rooms import (
    ITEM_SPEED,
    ROOM_BAT_CAVE,
    ROOM_SPEED,
    ROOM_SPEED_HALL,
)
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import (
    escape_knockback_spin,
    is_knockback,
)

# Live pins from post_bat_cave_continuous + pure recon (2026-08-05).
_DOOR_LEDGE_X_MAX = 66
_MID_FLOOR_X = (100, 170)
# Shot-block hole is narrow: pure source at x≈157 stays solid (peak y≈331);
# x∈[148,154] opens (peak y≈228 → cavity). Prefer center ~151.
_HOLE_X = (148, 154)
_HOLE_TARGET_X = 151
_HOLE_CEILING_Y = 323
_DOOR_SHELF_Y = 160
_CAVITY_Y = (230, 290)
_FLOOR_Y = 395
_LAVA_Y = 430

_LAND_FRAMES = 60
_SHOT_BURSTS = 14
_SHOT_TRAVEL = 3
_GAP_RUNUP = 6
_GAP_JUMP_FRAMES = 55
_HOLE_SPAM_FRAMES = 180
_HOLE_JUMP_FRAMES = 90
_CAVITY_CEILING_SPAM = 90  # opens solid above (171,251); 60f min live
_CAVITY_JUMP_FRAMES = 100
_CLIMB_ATTEMPTS = 6
_UPPER_DOOR_FRAMES = 500
_SETTLE_FRAMES = 320
_TOTAL_BUDGET = 4500


def _land_door_ledge(session: ControllerSession, label: str) -> SuperMetroidState:
    """Settle standing on the bottom-left door ledge."""
    for _ in range(_LAND_FRAMES):
        state = hold(session, 1, reason=f"{label}_land")
        if (
            state.velocity_y == 0
            and state.pose in _STANDING_POSES
            and state.samus_y < _LAVA_Y
        ):
            return state
    return session.state


def _face_right_no_walk(session: ControllerSession, label: str) -> None:
    """Face right without walking off the short door ledge."""
    hold(session, 2, "RIGHT", reason=f"{label}_face")
    hold(session, 3, reason=f"{label}_face_settle")


def _shoot_ceiling_hole(session: ControllerSession, label: str) -> None:
    """Angle-up / up beam pressure on the lower ceiling shot block from door."""
    select_weapon(session, 0)
    _face_right_no_walk(session, label)
    for i in range(_SHOT_BURSTS):
        state = session.state
        if state.samus_x > _DOOR_LEDGE_X_MAX:
            hold(session, 1, "LEFT", reason=f"{label}_ledge_back")
            continue
        if i % 2 == 0:
            hold(session, 1, "R", "X", reason=f"{label}_angle_shot")
        else:
            hold(session, 1, "UP", "X", reason=f"{label}_up_shot")
        hold(session, _SHOT_TRAVEL, reason=f"{label}_shot_travel")


def _gap_skip_to_mid(session: ControllerSession, label: str) -> SuperMetroidState:
    """Door ledge → mid floor segment (~x 100–170, y≈395) over the lava gap."""
    for _ in range(_GAP_RUNUP):
        hold(session, 1, "RIGHT", "B", reason=f"{label}_gap_runup")
    for frame in range(_GAP_JUMP_FRAMES):
        state = hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_gap_jump")
        if state.room_id != ROOM_BAT_CAVE:
            return state
        if (
            frame > 20
            and state.velocity_y == 0
            and state.samus_y < _LAVA_Y
            and state.samus_x > _MID_FLOOR_X[0]
        ):
            break
    for _ in range(40):
        state = hold(session, 1, reason=f"{label}_gap_land")
        if (
            state.velocity_y == 0
            and state.samus_y < _LAVA_Y
            and state.pose in _STANDING_POSES
        ):
            break
    return session.state


def _clear_shot_block_under_hole(session: ControllerSession, label: str) -> None:
    """Stand under the hole and spam UP+X until the lower shot block yields.

    Re-center if drift leaves the narrow open band (x>154 fails pure source).
    """
    select_weapon(session, 0)
    for _ in range(_HOLE_SPAM_FRAMES):
        state = session.state
        if state.samus_y >= _LAVA_Y or state.room_id != ROOM_BAT_CAVE:
            return
        if is_knockback(state):
            escape_knockback_spin(
                session,
                prefer_dir="LEFT",
                run_frames=3,
                spin_frames=12,
                label=label,
            )
            continue
        if state.velocity_y == 0 and state.samus_x < _HOLE_X[0]:
            hold(session, 1, "RIGHT", reason=f"{label}_hole_recenter")
            continue
        if state.velocity_y == 0 and state.samus_x > _HOLE_X[1]:
            hold(session, 1, "LEFT", reason=f"{label}_hole_recenter")
            continue
        hold(session, 1, "UP", "X", reason=f"{label}_hole_spam")


def _lava_recover(session: ControllerSession, label: str) -> None:
    """Spin-escape left toward the door ledge from lava (assist energy)."""
    for _ in range(40):
        state = hold(session, 1, "LEFT", "B", "A", reason=f"{label}_lava")
        if state.samus_y < _FLOOR_Y + 10 and state.velocity_y == 0:
            break
    for _ in range(20):
        hold(session, 1, reason=f"{label}_lava_settle")


def _walk_to_hole_band(session: ControllerSession, label: str) -> bool:
    """Grounded walk under hole center without dashing into lava. True if banded."""
    for _ in range(70):
        state = session.state
        if state.samus_y >= _LAVA_Y:
            return False
        if (
            _HOLE_X[0] <= state.samus_x <= _HOLE_X[1]
            and state.velocity_y == 0
            and state.samus_y < _LAVA_Y
        ):
            return True
        if state.velocity_y != 0:
            hold(session, 1, reason=f"{label}_hole_air")
            continue
        # No B dash — short mid platforms; dash walks off into lava.
        face = "RIGHT" if state.samus_x < _HOLE_TARGET_X else "LEFT"
        hold(session, 1, face, reason=f"{label}_to_hole")
    st = session.state
    return _HOLE_X[0] <= st.samus_x <= _HOLE_X[1] and st.samus_y < _LAVA_Y


def _on_door_shelf(state: SuperMetroidState) -> bool:
    return (
        state.room_id == ROOM_BAT_CAVE
        and state.velocity_y == 0
        and state.samus_y <= _DOOR_SHELF_Y
        and state.pose in _STANDING_POSES | {9, 10, 11}
    )


def _in_cavity(state: SuperMetroidState) -> bool:
    return (
        state.room_id == ROOM_BAT_CAVE
        and _CAVITY_Y[0] <= state.samus_y <= _CAVITY_Y[1]
        and state.samus_x >= 140
    )


def _jump_through_hole(
    session: ControllerSession, label: str
) -> tuple[SuperMetroidState, int]:
    """Jump through lower hole; natural land is cavity ~(171,251).

    Door shelf is a second climb from cavity after clearing the upper ceiling.
    """
    min_y = session.state.samus_y
    for frame in range(_HOLE_JUMP_FRAMES):
        state = session.state
        min_y = min(min_y, state.samus_y)
        if state.room_id != ROOM_BAT_CAVE:
            return state, min_y
        if is_knockback(state):
            escape_knockback_spin(
                session,
                prefer_dir="RIGHT",
                run_frames=4,
                spin_frames=16,
                label=label,
                stop_room_id=ROOM_SPEED_HALL,
            )
            continue
        if _on_door_shelf(state):
            return state, min_y
        if (
            frame > 40
            and state.velocity_y == 0
            and _in_cavity(state)
        ):
            return state, min_y
        if frame < 16:
            hold(session, 1, "A", reason=f"{label}_hole_up")
        else:
            hold(session, 1, "RIGHT", "B", "A", "X", reason=f"{label}_hole_R")
    return session.state, min_y


def _settle_cavity(session: ControllerSession, label: str) -> SuperMetroidState:
    """Escape gamet knockback and stand in the mid pocket."""
    for _ in range(60):
        state = session.state
        if state.room_id != ROOM_BAT_CAVE:
            return state
        if is_knockback(state):
            escape_knockback_spin(
                session,
                prefer_dir="LEFT",
                run_frames=4,
                spin_frames=14,
                label=label,
            )
            continue
        if state.velocity_y == 0 and state.pose in _STANDING_POSES | {9, 10, 11, 230}:
            # pose 230 = hit-land settle; one idle then done
            hold(session, 2, reason=f"{label}_cavity_settle")
            return session.state
        hold(session, 1, reason=f"{label}_cavity_fall")
    return session.state


def _cavity_to_door_shelf(
    session: ControllerSession, label: str
) -> tuple[SuperMetroidState, int]:
    """Clear shot-block ceiling above cavity, jump to door shelf (y≤160).

    Live recon (2026-08-05): without UP+X, cavity jumps peak y≈211 (solid).
    After ≥60f UP+X from ~(171,251), jump peaks y≈99 and lands shelf ~x=192.
    """
    min_y = session.state.samus_y
    select_weapon(session, 0)

    for _ in range(_CAVITY_CEILING_SPAM):
        state = session.state
        if state.room_id != ROOM_BAT_CAVE:
            return state, min_y
        if state.samus_y >= _LAVA_Y:
            return state, min_y
        if is_knockback(state):
            escape_knockback_spin(
                session,
                prefer_dir="LEFT",
                run_frames=3,
                spin_frames=12,
                label=label,
            )
            continue
        # Stay roughly under the ceiling hole band.
        if state.velocity_y == 0 and state.samus_x < 150:
            hold(session, 1, "RIGHT", reason=f"{label}_cavity_center")
            continue
        if state.velocity_y == 0 and state.samus_x > 180:
            hold(session, 1, "LEFT", reason=f"{label}_cavity_center")
            continue
        hold(session, 1, "UP", "X", reason=f"{label}_cavity_ceil")

    for frame in range(_CAVITY_JUMP_FRAMES):
        state = session.state
        min_y = min(min_y, state.samus_y)
        if state.room_id == ROOM_SPEED_HALL:
            return state, min_y
        if state.room_id != ROOM_BAT_CAVE:
            return state, min_y
        if is_knockback(state):
            escape_knockback_spin(
                session,
                prefer_dir="RIGHT",
                run_frames=4,
                spin_frames=14,
                label=label,
                stop_room_id=ROOM_SPEED_HALL,
            )
            continue
        if _on_door_shelf(state) and frame > 25:
            return state, min_y
        if frame < 20:
            hold(session, 1, "A", reason=f"{label}_shelf_up")
        else:
            hold(session, 1, "RIGHT", "B", "A", "X", reason=f"{label}_shelf_R")
    return session.state, min_y


def _climb_toward_upper(
    session: ControllerSession, label: str
) -> tuple[SuperMetroidState, int, int]:
    """Hole clear → cavity land → cavity ceiling clear → door shelf.

    Returns (state, min_y, attempts).
    """
    min_y = session.state.samus_y
    for attempt in range(_CLIMB_ATTEMPTS):
        state = session.state
        if state.room_id != ROOM_BAT_CAVE:
            return state, min_y, attempt
        if _on_door_shelf(state):
            return state, min_y, attempt
        if state.room_id == ROOM_SPEED_HALL:
            return state, min_y, attempt

        if state.samus_y >= _LAVA_Y:
            _lava_recover(session, label)
            if session.state.samus_x < 80 and session.state.samus_y < _LAVA_Y:
                _gap_skip_to_mid(session, label)
            continue

        # Already in cavity: second-stage climb.
        if _in_cavity(state) or (
            state.samus_y < 320 and state.samus_y > _DOOR_SHELF_Y and state.samus_x >= 140
        ):
            _settle_cavity(session, label)
            state, jump_min = _cavity_to_door_shelf(session, label)
            min_y = min(min_y, jump_min)
            if state.room_id != ROOM_BAT_CAVE or _on_door_shelf(state):
                return state, min_y, attempt
            continue

        # Lower floor: align under hole and first jump.
        if state.samus_y >= 340:
            if not _walk_to_hole_band(session, label):
                if session.state.samus_y >= _LAVA_Y:
                    continue
                if session.state.samus_x < 80:
                    _gap_skip_to_mid(session, label)
                continue
            for _ in range(12):
                st = hold(session, 1, reason=f"{label}_hole_settle")
                if st.velocity_y == 0 and st.pose in _STANDING_POSES:
                    break
            _clear_shot_block_under_hole(session, label)
            state, jump_min = _jump_through_hole(session, label)
            min_y = min(min_y, jump_min)
            if state.room_id != ROOM_BAT_CAVE or _on_door_shelf(state):
                return state, min_y, attempt
            _settle_cavity(session, label)
            continue

        # Mid-air / odd band: idle a few frames then reclassify.
        hold(session, 1, reason=f"{label}_reclass")

    return session.state, min_y, _CLIMB_ATTEMPTS


def _upper_to_speed_hall(
    session: ControllerSession, label: str
) -> SuperMetroidState:
    """Door shelf (y≤160) → right blue door → ordinary Speed Hall."""
    select_weapon(session, 0)
    min_y = session.state.samus_y
    max_x = session.state.samus_x
    for frame in range(_UPPER_DOOR_FRAMES):
        state = session.state
        if state.room_id == ROOM_SPEED_HALL:
            break
        if state.room_id != ROOM_BAT_CAVE:
            break
        min_y = min(min_y, state.samus_y)
        max_x = max(max_x, state.samus_x)

        if state.samus_y > 350:
            raise TimeoutError(
                f"{label}: fell from upper band before door; "
                f"room=0x{state.room_id:04X} pose={state.pose} "
                f"xy=({state.samus_x},{state.samus_y}) min_y={min_y} max_x={max_x} "
                f"door_transition={state.door_transition}"
            )

        if is_knockback(state):
            escape_knockback_spin(
                session,
                prefer_dir="RIGHT",
                run_frames=6,
                spin_frames=24,
                label=label,
                stop_room_id=ROOM_SPEED_HALL,
            )
            continue

        # Dropped back into cavity — re-clear and re-jump.
        if state.samus_y > 200:
            if _in_cavity(state) or state.velocity_y == 0:
                _settle_cavity(session, label)
                state, jump_min = _cavity_to_door_shelf(session, label)
                min_y = min(min_y, jump_min)
                if state.room_id == ROOM_SPEED_HALL:
                    break
            else:
                hold(session, 1, "UP", "X", reason=f"{label}_reclear")
            continue

        # Door shelf: shoot-run right to blue door.
        phase = frame % 20
        if phase < 8:
            hold(session, 1, "RIGHT", "B", "X", reason=f"{label}_upper_run")
        elif phase < 12:
            hold(session, 1, "RIGHT", "B", "A", "X", reason=f"{label}_upper_hop")
        elif phase < 16:
            hold(session, 1, "RIGHT", "X", reason=f"{label}_upper_shoot")
        else:
            hold(session, 1, "R", "X", reason=f"{label}_upper_ang")
    else:
        state = session.state
        raise TimeoutError(
            f"{label}: top-right door missed before room "
            f"0x{ROOM_SPEED_HALL:04X}; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"min_y={min_y} max_x={max_x} door_transition={state.door_transition}"
        )

    return wait_ordinary_room(
        session,
        ROOM_SPEED_HALL,
        settle_frames=_SETTLE_FRAMES,
        label=label,
    )


def play_bat_cave_to_speed_hall(session: ControllerSession) -> SuperMetroidState:
    """Bat Cave left lip → ordinary Speed Booster Hall via top-right door.

    Path: door shots → gap-skip → hole clear → cavity land → cavity ceiling
    clear → door shelf → right blue door.
    """
    label = "bat_cave_to_speed_hall"
    require_room(session, ROOM_BAT_CAVE, label)

    start_frame = session.frame
    _land_door_ledge(session, label)
    _shoot_ceiling_hole(session, label)
    _gap_skip_to_mid(session, label)

    state, min_y, attempts = _climb_toward_upper(session, label)

    if state.room_id == ROOM_SPEED_HALL:
        return wait_ordinary_room(
            session,
            ROOM_SPEED_HALL,
            settle_frames=_SETTLE_FRAMES,
            label=label,
        )

    if state.room_id == ROOM_BAT_CAVE and (
        _on_door_shelf(state) or state.samus_y < 300
    ):
        return _upper_to_speed_hall(session, label)

    if session.frame - start_frame > _TOTAL_BUDGET:
        state = session.state
        raise TimeoutError(
            f"{label}: budget exceeded; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"min_y={min_y} attempts={attempts} "
            f"door_transition={state.door_transition}"
        )

    state = session.state
    raise TimeoutError(
        f"{label}: upper band / Speed Hall missed; room=0x{state.room_id:04X} "
        f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
        f"min_y={min_y} attempts={attempts} hole_ceiling_y={_HOLE_CEILING_Y} "
        f"door_transition={state.door_transition} "
        f"frames={session.frame - start_frame}"
    )


# --- Speed Hall (0xACF0) → Speed Booster room collect (0xAD1B) ---------------
# Live pure from post_bat_cave_to_speed_hall_pure (2026-08-05):
# * Hold RIGHT+B the whole incline — crumble bridges hold under dash.
# * Right red Super door ~x≥2950, y≈395 (bottom-right of 12-screen hall).
# * Speed room: left lip ~(39,139) → hop onto chozo shelf ~x=165–175 → item.

_HALL_DASH_FRAMES = 900
_HALL_DOOR_X = 2950
_HALL_DOOR_BACKOFF = 10
_HALL_SUPER_FUSE = 70
_HALL_ENTER_FRAMES = 500
_SPEED_SETTLE = 320
_SPEED_COLLECT_FRAMES = 280
_SPEED_FANFARE = 500
_HALL_TO_SPEED_BUDGET = 3500


def _dash_speed_hall(session: ControllerSession, label: str) -> SuperMetroidState:
    """Hold RIGHT+B across the entire Speed Hall incline to the red door band."""
    for _ in range(30):
        state = hold(session, 1, reason=f"{label}_land")
        if state.velocity_y == 0 and state.pose in _STANDING_POSES:
            break

    max_x = session.state.samus_x
    for frame in range(_HALL_DASH_FRAMES):
        state = hold(session, 1, "RIGHT", "B", reason=f"{label}_dash")
        max_x = max(max_x, state.samus_x)
        if state.room_id == ROOM_SPEED:
            return state
        if state.room_id != ROOM_SPEED_HALL:
            break
        if (
            state.samus_x >= _HALL_DOOR_X
            and state.velocity_y == 0
            and state.pose in _STANDING_POSES
        ):
            return state
    else:
        state = session.state
        raise TimeoutError(
            f"{label}: hall dash missed door band; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) max_x={max_x}"
        )
    return session.state


def _open_speed_hall_super_door(
    session: ControllerSession, label: str
) -> SuperMetroidState:
    """Back off, Super the right red door, enter ordinary Speed Booster Room."""
    if session.state.room_id == ROOM_SPEED:
        return wait_ordinary_room(
            session, ROOM_SPEED, settle_frames=_SPEED_SETTLE, label=label
        )

    unmorph(session)
    hold(session, _HALL_DOOR_BACKOFF, "LEFT", reason=f"{label}_door_back")
    hold(session, 10, reason=f"{label}_door_settle")
    select_weapon(session, 2)
    hold(session, 4, "RIGHT", reason=f"{label}_face_door")
    hold(session, 4, reason=f"{label}_face_release")
    hold(session, 2, "RIGHT", "X", reason=f"{label}_super")
    hold(session, _HALL_SUPER_FUSE, reason=f"{label}_fuse")

    for frame in range(_HALL_ENTER_FRAMES):
        state = hold(session, 1, "RIGHT", "B", reason=f"{label}_enter")
        if state.room_id == ROOM_SPEED:
            break
        # Second Super pulse if still blocked at the wall.
        if frame > 0 and frame % 120 == 0:
            select_weapon(session, 2)
            hold(session, 2, "RIGHT", "X", reason=f"{label}_super_retry")
            hold(session, 40, reason=f"{label}_fuse_retry")
    else:
        state = session.state
        raise TimeoutError(
            f"{label}: right Super door missed before room "
            f"0x{ROOM_SPEED:04X}; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"supers={state.super_missiles} selected={state.selected_item}"
        )

    return wait_ordinary_room(
        session, ROOM_SPEED, settle_frames=_SPEED_SETTLE, label=label
    )


def _collect_speed_booster(
    session: ControllerSession, label: str
) -> SuperMetroidState:
    """Left lip → walk/shoot into chozo Speed Booster PLM.

    Live pure (2026-08-05): walk RIGHT with periodic beam shots collects at
    ~(171,123). Continuous jump pins on the statue face and misses the orb.
    """
    require_room(session, ROOM_SPEED, label)
    if session.state.collected_items & ITEM_SPEED:
        return session.state

    select_weapon(session, 0)
    unmorph(session)

    for frame in range(_SPEED_COLLECT_FRAMES):
        state = session.state
        if state.collected_items & ITEM_SPEED:
            break
        if state.room_id != ROOM_SPEED:
            raise TimeoutError(
                f"{label}: left Speed room during collect; "
                f"room=0x{state.room_id:04X} xy=({state.samus_x},{state.samus_y})"
            )
        if state.pose in (137, 138):
            unmorph(session)
            continue
        # Past the statue without collect: step back and re-approach.
        if state.samus_x > 190 and state.velocity_y == 0:
            hold(session, 12, "LEFT", reason=f"{label}_rebound")
            continue
        if frame % 12 == 0:
            hold(session, 1, "X", reason=f"{label}_shot")
        else:
            hold(session, 1, "RIGHT", reason=f"{label}_walk")
    else:
        state = session.state
        raise TimeoutError(
            f"{label}: Speed Booster PLM not collected; "
            f"items=0x{state.collected_items:04X} pose={state.pose} "
            f"xy=({state.samus_x},{state.samus_y})"
        )

    hold(session, _SPEED_FANFARE, reason=f"{label}_fanfare")
    # Standing handoff for human record / next pure (not mid-fanfare morph).
    unmorph(session)
    for _ in range(30):
        state = hold(session, 1, reason=f"{label}_stand")
        if state.velocity_y == 0 and state.pose in _STANDING_POSES:
            break
    hold(session, 4, "LEFT", reason=f"{label}_face_exit")
    hold(session, 6, reason=f"{label}_exit_settle")
    return session.state


def play_speed_hall_to_speed(session: ControllerSession) -> SuperMetroidState:
    """Speed Hall left lip → right Super door → natural Speed Booster collect.

    Path: hold RIGHT+B across crumble incline → Super red door → hop into
    chozo PLM. Caps: Morph, Bombs, Missiles, Supers, Hi-Jump, Varia.
    """
    label = "speed_hall_to_speed"
    require_room(session, ROOM_SPEED_HALL, label)
    start = session.frame

    state = _dash_speed_hall(session, label)
    if state.room_id != ROOM_SPEED:
        state = _open_speed_hall_super_door(session, label)
    else:
        state = wait_ordinary_room(
            session, ROOM_SPEED, settle_frames=_SPEED_SETTLE, label=label
        )

    state = _collect_speed_booster(session, label)

    if not (state.collected_items & ITEM_SPEED):
        raise TimeoutError(
            f"{label}: finished without Speed bit; "
            f"items=0x{state.collected_items:04X} room=0x{state.room_id:04X} "
            f"xy=({state.samus_x},{state.samus_y}) "
            f"frames={session.frame - start}"
        )
    if session.frame - start > _HALL_TO_SPEED_BUDGET:
        # Soft warning path — still success if item bit is set.
        pass
    return session.state


__all__ = [
    "play_bat_cave_to_speed_hall",
    "play_speed_hall_to_speed",
    "ROOM_BAT_CAVE",
    "ROOM_SPEED",
    "ROOM_SPEED_HALL",
    "ITEM_SPEED",
]

"""Warehouse with Hi-Jump → Zeela → Kihunter → Baby Kraid → Kraid."""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    ensure_morph,
    hold,
    play_run_shoot_exit,
    require_room,
    select_weapon,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.rooms import (
    ITEM_HI_JUMP,
    ROOM_BABY_KRAID,
    ROOM_BAT,
    ROOM_BELOW_SPAZER,
    ROOM_BIG_PINK,
    ROOM_BUSINESS,
    ROOM_EAST_TUNNEL,
    ROOM_GHZ,
    ROOM_GLASS,
    ROOM_HJ,
    ROOM_HJ_SHAFT,
    ROOM_KRAID,
    ROOM_KRAID_EYE,
    ROOM_NOOB,
    ROOM_RED_TOWER,
    ROOM_WAREHOUSE,
    ROOM_WAREHOUSE_KIHUNTER,
    ROOM_WEST_TUNNEL,
    ROOM_ZEELA,
)
from super_metroid.routes.runtime import ControllerSession

_hold = hold
_require_room = require_room
_select_weapon = select_weapon
_unmorph = unmorph
_wait_ordinary_room = wait_ordinary_room

from super_metroid.routes.kpdr.warehouse import play_warehouse_wall_to_lower_lip

def play_warehouse_to_zeela_with_hijump(
    session: ControllerSession,
) -> SuperMetroidState:
    """Open Warehouse Super stack, Hi-Jump to the ledge, and enter Zeela."""
    play_warehouse_wall_to_lower_lip(session)
    _unmorph(session)
    _select_weapon(session, 0)
    _hold(session, 12, reason="warehouse_hj_release")
    for _ in range(120):
        if session.state.samus_x <= 445:
            break
        _hold(session, 1, "LEFT", reason="warehouse_hj_backoff")
    _hold(session, 5, "RIGHT", reason="warehouse_hj_brake")
    _hold(session, 8, reason="warehouse_hj_jump_release")
    for frame in range(180):
        buttons = ("A",) if frame < 25 else ("RIGHT", "B", "A")
        state = _hold(session, 1, *buttons, reason="warehouse_hj_climb")
        if state.samus_x >= 720 and state.samus_y <= 160:
            break
    _hold(session, 30, reason="warehouse_hj_door_settle")
    _unmorph(session)
    _select_weapon(session, 0)
    _hold(session, 3, "RIGHT", reason="warehouse_face_zeela")
    _hold(session, 3, reason="warehouse_face_zeela_release")
    _hold(session, 2, "RIGHT", "X", reason="warehouse_zeela_door_shot")
    _hold(session, 30, reason="warehouse_zeela_door_open")
    for _ in range(420):
        state = _hold(session, 1, "RIGHT", "B", "A", reason="warehouse_enter_zeela")
        if state.room_id == ROOM_ZEELA:
            break
    else:
        raise TimeoutError(f"warehouse_to_zeela: {state}")
    return _wait_ordinary_room(
        session, ROOM_ZEELA, settle_frames=280, label="warehouse_to_zeela"
    )



def play_zeela_to_kihunter(session: ControllerSession) -> SuperMetroidState:
    """Warehouse Zeela Room top-left → upper door to Kihunter room."""
    _require_room(session, ROOM_ZEELA, "zeela_to_kihunter")
    _unmorph(session)
    _select_weapon(session, 0)
    _hold(session, 10, reason="zeela_entry_release")
    _hold(session, 10, "A", reason="zeela_first_drop_jump")
    _hold(session, 1, "DOWN", reason="zeela_first_drop_aim")
    _hold(session, 2, "X", reason="zeela_first_drop_shot")
    _hold(session, 80, reason="zeela_first_drop")
    ensure_morph(session)
    for _ in range(300):
        state = _hold(session, 1, "RIGHT", reason="zeela_middle_roll")
        if state.samus_x >= 105 and state.samus_y >= 325:
            break
    _unmorph(session)
    _hold(session, 30, reason="zeela_middle_land")
    _select_weapon(session, 0)
    _hold(session, 8, "A", reason="zeela_second_drop_jump")
    _hold(session, 1, "DOWN", reason="zeela_second_drop_aim")
    _hold(session, 2, "X", reason="zeela_second_drop_shot")
    for _ in range(180):
        state = _hold(session, 1, "LEFT", reason="zeela_second_drop")
        if state.samus_y >= 395:
            break
    _hold(session, 40, reason="zeela_bottom_land")
    ensure_morph(session)
    for frame in range(700):
        buttons = ("RIGHT", "X") if frame % 45 < 2 else ("RIGHT",)
        state = _hold(session, 1, *buttons, reason="zeela_bottom_bomb_roll")
        if state.samus_x >= 400:
            break
    else:
        raise TimeoutError(f"zeela_to_kihunter: tunnel stalled: {state}")
    _unmorph(session)
    _hold(session, 40, reason="zeela_up_door_stand")
    _select_weapon(session, 0)
    _hold(session, 2, "UP", reason="zeela_up_door_aim")
    _hold(session, 2, "UP", "X", reason="zeela_up_door_shot")
    _hold(session, 35, reason="zeela_up_door_open")
    for _ in range(400):
        state = _hold(session, 1, "A", reason="zeela_enter_kihunter")
        if state.room_id == ROOM_WAREHOUSE_KIHUNTER:
            break
    else:
        raise TimeoutError(f"zeela_to_kihunter: up door failed: {state}")
    return _wait_ordinary_room(
        session,
        ROOM_WAREHOUSE_KIHUNTER,
        settle_frames=280,
        label="zeela_to_kihunter",
    )



def play_kihunter_to_baby_kraid(session: ControllerSession) -> SuperMetroidState:
    """Drop through the Warehouse Kihunter floor and take the lower door."""
    _require_room(session, ROOM_WAREHOUSE_KIHUNTER, "kihunter_to_baby")
    _hold(session, 80, reason="kihunter_entry_floor")
    _unmorph(session)
    _select_weapon(session, 0)
    for _ in range(300):
        if session.state.samus_x >= 350:
            break
        _hold(session, 1, "RIGHT", "B", reason="kihunter_drop_position")
    _hold(session, 6, "LEFT", reason="kihunter_drop_brake")
    _hold(session, 10, reason="kihunter_drop_settle")
    _hold(session, 3, "RIGHT", reason="kihunter_drop_exact")
    _hold(session, 2, "LEFT", reason="kihunter_drop_exact_brake")
    _hold(session, 10, reason="kihunter_drop_exact_settle")
    ensure_morph(session)
    _hold(session, 2, "X", reason="kihunter_floor_bomb")
    _hold(session, 55, reason="kihunter_floor_bomb_wait")
    _hold(session, 2, "X", reason="kihunter_floor_bomb2")
    for _ in range(180):
        state = _hold(session, 1, reason="kihunter_floor_drop")
        if state.samus_y >= 310:
            break
    ensure_morph(session)
    for _ in range(160):
        state = _hold(session, 1, "LEFT", reason="kihunter_shaft_align")
        if state.samus_y >= 350:
            break
    for _ in range(360):
        state = _hold(session, 1, "RIGHT", reason="kihunter_lower_roll")
        if state.samus_x >= 470:
            break
    _unmorph(session)
    _select_weapon(session, 0)
    for frame in range(500):
        buttons = ("RIGHT", "B", "X") if frame % 25 < 5 else ("RIGHT", "B", "A")
        state = _hold(session, 1, *buttons, reason="kihunter_enter_baby")
        if state.room_id == ROOM_BABY_KRAID:
            break
    else:
        raise TimeoutError(f"kihunter_to_baby: {state}")
    return _wait_ordinary_room(
        session, ROOM_BABY_KRAID, settle_frames=280, label="kihunter_to_baby"
    )



def _baby_kraid_sweep(
    session: ControllerSession,
    direction: str,
    target_x: int,
    *,
    limit: int,
    label: str,
) -> None:
    for frame in range(limit):
        phase = frame % 24
        if phase < 3:
            buttons = (direction, "X")
        elif phase >= 14:
            buttons = (direction, "B", "A")
        else:
            buttons = (direction, "B")
        state = _hold(session, 1, *buttons, reason=label)
        if direction == "RIGHT" and state.samus_x >= target_x:
            return
        if direction == "LEFT" and state.samus_x <= target_x:
            return
    raise TimeoutError(f"{label}: {session.state}")



def play_baby_kraid_to_eye(session: ControllerSession) -> SuperMetroidState:
    """Kill the three pirates and Mini-Kraid, then take the right gray door."""
    _require_room(session, ROOM_BABY_KRAID, "baby_kraid_to_eye")
    _hold(session, 100, reason="baby_kraid_entry_floor")
    _unmorph(session)
    _select_weapon(session, 2)
    _baby_kraid_sweep(session, "RIGHT", 1490, limit=1700, label="baby_kraid_forward")
    if session.state.enemies_killed < session.state.num_enemies:
        _baby_kraid_sweep(session, "LEFT", 50, limit=1900, label="baby_kraid_cleanup")
    _baby_kraid_sweep(session, "RIGHT", 1490, limit=1900, label="baby_kraid_return")
    for _ in range(600):
        state = _hold(session, 1, "RIGHT", "B", "A", reason="baby_kraid_enter_eye")
        if state.room_id == ROOM_KRAID_EYE:
            break
    else:
        raise TimeoutError(f"baby_kraid_to_eye: gray door failed: {state}")
    return _wait_ordinary_room(
        session, ROOM_KRAID_EYE, settle_frames=300, label="baby_kraid_to_eye"
    )



def play_eye_to_kraid(session: ControllerSession) -> SuperMetroidState:
    """Cross Kraid Eye Door Room and open the eye door with Supers."""
    _require_room(session, ROOM_KRAID_EYE, "eye_to_kraid")
    _hold(session, 100, reason="kraid_eye_entry_floor")
    _unmorph(session)
    _select_weapon(session, 2)
    for frame in range(1800):
        phase = frame % 28
        if phase < 3:
            buttons = ("RIGHT", "X")
        elif phase >= 16:
            buttons = ("RIGHT", "B", "A")
        else:
            buttons = ("RIGHT", "B")
        state = _hold(session, 1, *buttons, reason="kraid_eye_run")
        if state.room_id == ROOM_KRAID:
            break
    else:
        raise TimeoutError(f"eye_to_kraid: eye door failed: {state}")
    return _wait_ordinary_room(
        session, ROOM_KRAID, settle_frames=340, label="eye_to_kraid"
    )



def play_warehouse_to_kraid_with_hijump(
    session: ControllerSession,
) -> SuperMetroidState:
    """Natural Warehouse anchor with Hi-Jump → natural Kraid-room entry."""
    if not session.state.collected_items & ITEM_HI_JUMP:
        raise RuntimeError("warehouse_to_kraid_with_hijump: Hi-Jump not collected")
    play_warehouse_to_zeela_with_hijump(session)
    play_zeela_to_kihunter(session)
    play_kihunter_to_baby_kraid(session)
    play_baby_kraid_to_eye(session)
    return play_eye_to_kraid(session)



def play_warehouse_hijump_kraid(session: ControllerSession) -> SuperMetroidState:
    """Composed safer route: Warehouse → Hi-Jump → Warehouse → Kraid."""
    play_warehouse_to_hijump(session)
    play_hijump_to_warehouse(session)
    return play_warehouse_to_kraid_with_hijump(session)


"""Hi-Jump room → shaft → Business → Warehouse after collect."""

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
    ITEM_HI_JUMP,
    ROOM_BUSINESS,
    ROOM_HJ,
    ROOM_HJ_SHAFT,
    ROOM_WAREHOUSE,
)
from super_metroid.routes.runtime import ControllerSession

_hold = hold
_require_room = require_room
_select_weapon = select_weapon
_unmorph = unmorph
_wait_ordinary_room = wait_ordinary_room

from super_metroid.routes.kpdr.business_climb import play_business_to_warehouse

def play_hj_room_to_shaft(session: ControllerSession) -> SuperMetroidState:
    """Collected Hi-Jump alcove → natural E-Tank-room left-door spawn."""
    _require_room(session, ROOM_HJ, "hj_room_to_shaft")
    _unmorph(session)
    _hold(session, 20, reason="hj_room_return_settle")
    for _ in range(80):
        state = session.state
        if state.samus_x <= 80:
            break
        _hold(session, 1, "LEFT", "B", reason="hj_room_return_backoff")
    _hold(session, 8, "RIGHT", reason="hj_room_return_brake")
    _hold(session, 10, reason="hj_room_return_release")
    _hold(session, 12, "RIGHT", "B", reason="hj_room_return_runup")
    for _ in range(120):
        state = _hold(session, 1, "RIGHT", "B", "A", reason="hj_room_return_cross")
        if state.samus_x >= 181:
            break
    _hold(session, 80, reason="hj_room_return_land")

    _unmorph(session)
    _select_weapon(session, 0)
    for _ in range(80):
        state = session.state
        if state.samus_x <= 185:
            break
        _hold(session, 1, "LEFT", reason="hj_room_return_door_backoff")
    _hold(session, 8, "RIGHT", reason="hj_room_return_door_brake")
    _hold(session, 8, reason="hj_room_return_door_settle")
    _hold(session, 3, "RIGHT", reason="hj_room_return_face_door")
    _hold(session, 3, reason="hj_room_return_face_release")
    _hold(session, 1, "X", reason="hj_room_return_door_shot")
    _hold(session, 40, reason="hj_room_return_door_open")
    for _ in range(420):
        state = _hold(session, 1, "RIGHT", "B", "A", reason="hj_room_return_enter")
        if state.room_id == ROOM_HJ_SHAFT:
            break
    else:
        raise TimeoutError(f"hj_room_to_shaft: {state}")
    return _wait_ordinary_room(
        session, ROOM_HJ_SHAFT, settle_frames=280, label="hj_room_to_shaft"
    )




def play_hj_shaft_to_business(session: ControllerSession) -> SuperMetroidState:
    """Use Hi-Jump's intended left climb and bomb tunnel back to Business."""
    _require_room(session, ROOM_HJ_SHAFT, "hj_shaft_to_business")
    _unmorph(session)
    _hold(session, 50, reason="hj_return_bottom_land")

    # Bottom floor → right shelf.
    _hold(session, 10, reason="hj_return_jump_release")
    for frame in range(125):
        buttons = ("A",) if frame < 18 else ("RIGHT", "A")
        _hold(session, 1, *buttons, reason="hj_return_first_jump")
    _hold(session, 80, reason="hj_return_first_land")

    # Right shelf → upper-left slope.
    _unmorph(session)
    _hold(session, 50, reason="hj_return_shelf_stand")
    for _ in range(80):
        state = session.state
        if state.samus_x <= 82:
            break
        _hold(session, 1, "LEFT", reason="hj_return_shelf_position")
    _hold(session, 6, "RIGHT", reason="hj_return_shelf_brake")
    _hold(session, 8, reason="hj_return_shelf_release")
    for frame in range(130):
        buttons = ("A",) if frame < 65 else ("LEFT", "A")
        _hold(session, 1, *buttons, reason="hj_return_second_jump")
    _hold(session, 50, reason="hj_return_second_land")

    # Upper-left slope → one-tile morph tunnel.
    _unmorph(session)
    _hold(session, 40, reason="hj_return_slope_stand")
    for frame in range(110):
        buttons = ("A",) if frame < 18 else ("RIGHT", "B", "A")
        state = _hold(session, 1, *buttons, reason="hj_return_top_jump")
        if frame > 55 and state.samus_y <= 95 and state.pose in (1, 2, 9, 10, 137, 138):
            break
    _hold(session, 40, reason="hj_return_top_land")

    # Bomb through the missile tunnel.  The explosions also naturally kill
    # the Sova, satisfying the gray-door lock.
    ensure_morph(session)
    for frame in range(1100):
        buttons = ("RIGHT", "X") if frame % 30 < 3 else ("RIGHT",)
        state = _hold(session, 1, *buttons, reason="hj_return_bomb_tunnel")
        if state.samus_x >= 350:
            break
    else:
        raise TimeoutError(f"hj_shaft_to_business: tunnel stalled: {state}")
    if state.enemies_killed < 1:
        for frame in range(500):
            buttons = ("RIGHT", "X") if frame % 40 < 2 else ("RIGHT",)
            state = _hold(session, 1, *buttons, reason="hj_return_sova_cleanup")
            if state.enemies_killed >= 1:
                break

    _hold(session, 80, "RIGHT", reason="hj_return_gray_approach")
    _unmorph(session)
    _select_weapon(session, 0)
    for frame in range(600):
        buttons = ("RIGHT", "B", "X") if frame < 4 else ("RIGHT", "B")
        state = _hold(session, 1, *buttons, reason="hj_return_gray_exit")
        if state.room_id == ROOM_BUSINESS:
            break
    else:
        raise TimeoutError(f"hj_shaft_to_business: gray door failed: {state}")
    state = _wait_ordinary_room(
        session, ROOM_BUSINESS, settle_frames=180, label="hj_shaft_to_business"
    )
    for _ in range(60):
        state = _hold(session, 1, reason="hj_return_business_floor")
        if state.samus_y >= 1419 and state.pose in (1, 2, 9, 10, 137, 138):
            break
    for _ in range(60):
        state = session.state
        if state.samus_x >= 88:
            break
        _hold(session, 1, "RIGHT", reason="hj_return_business_climb_anchor")
    _hold(session, 4, "LEFT", reason="hj_return_business_anchor_brake")
    _hold(session, 20, reason="hj_return_business_anchor_settle")
    return session.state




def play_hijump_to_warehouse(session: ControllerSession) -> SuperMetroidState:
    """Natural collected Hi-Jump state → Warehouse upper-left anchor."""
    play_hj_room_to_shaft(session)
    play_hj_shaft_to_business(session)
    return play_business_to_warehouse(session)




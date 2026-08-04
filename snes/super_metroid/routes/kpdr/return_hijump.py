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

from super_metroid.routes.kpdr.business_climb import play_business_to_warehouse

def play_hj_room_to_shaft(session: ControllerSession) -> SuperMetroidState:
    """Collected Hi-Jump alcove → natural E-Tank-room left-door spawn."""
    require_room(session, ROOM_HJ, "hj_room_to_shaft")
    unmorph(session)
    hold(session, 20, reason="hj_room_return_settle")
    for _ in range(80):
        state = session.state
        if state.samus_x <= 80:
            break
        hold(session, 1, "LEFT", "B", reason="hj_room_return_backoff")
    hold(session, 8, "RIGHT", reason="hj_room_return_brake")
    hold(session, 10, reason="hj_room_return_release")
    hold(session, 12, "RIGHT", "B", reason="hj_room_return_runup")
    for _ in range(120):
        state = hold(session, 1, "RIGHT", "B", "A", reason="hj_room_return_cross")
        if state.samus_x >= 181:
            break
    hold(session, 80, reason="hj_room_return_land")

    unmorph(session)
    select_weapon(session, 0)
    for _ in range(80):
        state = session.state
        if state.samus_x <= 185:
            break
        hold(session, 1, "LEFT", reason="hj_room_return_door_backoff")
    hold(session, 8, "RIGHT", reason="hj_room_return_door_brake")
    hold(session, 8, reason="hj_room_return_door_settle")
    hold(session, 3, "RIGHT", reason="hj_room_return_face_door")
    hold(session, 3, reason="hj_room_return_face_release")
    hold(session, 1, "X", reason="hj_room_return_door_shot")
    hold(session, 40, reason="hj_room_return_door_open")
    for _ in range(420):
        state = hold(session, 1, "RIGHT", "B", "A", reason="hj_room_return_enter")
        if state.room_id == ROOM_HJ_SHAFT:
            break
    else:
        raise TimeoutError(f"hj_room_to_shaft: {state}")
    return wait_ordinary_room(
        session, ROOM_HJ_SHAFT, settle_frames=280, label="hj_room_to_shaft"
    )

def play_hj_shaft_to_business(session: ControllerSession) -> SuperMetroidState:
    """Use Hi-Jump's intended left climb and bomb tunnel back to Business."""
    require_room(session, ROOM_HJ_SHAFT, "hj_shaft_to_business")
    unmorph(session)
    hold(session, 50, reason="hj_return_bottom_land")

    # Bottom floor → right shelf.
    hold(session, 10, reason="hj_return_jump_release")
    for frame in range(125):
        buttons = ("A",) if frame < 18 else ("RIGHT", "A")
        hold(session, 1, *buttons, reason="hj_return_first_jump")
    hold(session, 80, reason="hj_return_first_land")

    # Right shelf → upper-left slope.
    unmorph(session)
    hold(session, 50, reason="hj_return_shelf_stand")
    for _ in range(80):
        state = session.state
        if state.samus_x <= 82:
            break
        hold(session, 1, "LEFT", reason="hj_return_shelf_position")
    hold(session, 6, "RIGHT", reason="hj_return_shelf_brake")
    hold(session, 8, reason="hj_return_shelf_release")
    for frame in range(130):
        buttons = ("A",) if frame < 65 else ("LEFT", "A")
        hold(session, 1, *buttons, reason="hj_return_second_jump")
    hold(session, 50, reason="hj_return_second_land")

    # Upper-left slope → one-tile morph tunnel.
    unmorph(session)
    hold(session, 40, reason="hj_return_slope_stand")
    for frame in range(110):
        buttons = ("A",) if frame < 18 else ("RIGHT", "B", "A")
        state = hold(session, 1, *buttons, reason="hj_return_top_jump")
        if frame > 55 and state.samus_y <= 95 and state.pose in (1, 2, 9, 10, 137, 138):
            break
    hold(session, 40, reason="hj_return_top_land")

    # Bomb through the missile tunnel.  The explosions also naturally kill
    # the Sova, satisfying the gray-door lock.
    ensure_morph(session)
    for frame in range(1100):
        buttons = ("RIGHT", "X") if frame % 30 < 3 else ("RIGHT",)
        state = hold(session, 1, *buttons, reason="hj_return_bomb_tunnel")
        if state.samus_x >= 350:
            break
    else:
        raise TimeoutError(f"hj_shaft_to_business: tunnel stalled: {state}")
    if state.enemies_killed < 1:
        for frame in range(500):
            buttons = ("RIGHT", "X") if frame % 40 < 2 else ("RIGHT",)
            state = hold(session, 1, *buttons, reason="hj_return_sova_cleanup")
            if state.enemies_killed >= 1:
                break

    hold(session, 80, "RIGHT", reason="hj_return_gray_approach")
    unmorph(session)
    select_weapon(session, 0)
    for frame in range(600):
        buttons = ("RIGHT", "B", "X") if frame < 4 else ("RIGHT", "B")
        state = hold(session, 1, *buttons, reason="hj_return_gray_exit")
        if state.room_id == ROOM_BUSINESS:
            break
    else:
        raise TimeoutError(f"hj_shaft_to_business: gray door failed: {state}")
    state = wait_ordinary_room(
        session, ROOM_BUSINESS, settle_frames=180, label="hj_shaft_to_business"
    )
    for _ in range(60):
        state = hold(session, 1, reason="hj_return_business_floor")
        if state.samus_y >= 1419 and state.pose in (1, 2, 9, 10, 137, 138):
            break
    for _ in range(60):
        state = session.state
        if state.samus_x >= 88:
            break
        hold(session, 1, "RIGHT", reason="hj_return_business_climb_anchor")
    hold(session, 4, "LEFT", reason="hj_return_business_anchor_brake")
    hold(session, 20, reason="hj_return_business_anchor_settle")
    return session.state

def play_hijump_to_warehouse(session: ControllerSession) -> SuperMetroidState:
    """Natural collected Hi-Jump state → Warehouse upper-left anchor."""
    play_hj_room_to_shaft(session)
    play_hj_shaft_to_business(session)
    return play_business_to_warehouse(session)


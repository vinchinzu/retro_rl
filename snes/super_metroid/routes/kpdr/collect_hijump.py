"""Warehouse → Business → Hi-Jump shaft → Hi-Jump collect."""

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

from super_metroid.routes.kpdr.warehouse_stack import play_warehouse_to_business

def play_business_to_hj_shaft(session: ControllerSession) -> SuperMetroidState:
    """Descend Business Center and enter the lower-left red Hi-Jump door."""
    require_room(session, ROOM_BUSINESS, "business_to_hj_shaft")
    # The room becomes ordinary while Samus is still riding the incoming
    # elevator.  Wait for its center stop, then walk off before descending.
    for _ in range(500):
        state = hold(session, 1, reason="business_incoming_elevator")
        if state.pose == 0 and 675 <= state.samus_y <= 690:
            break
    for _ in range(120):
        state = hold(session, 1, "RIGHT", reason="business_elevator_dismount")
        if state.pose != 0 and state.samus_x >= 145:
            break
    unmorph(session)

    # Descend the lower shaft.  Run off each alternating shelf; Hi-Jump is not
    # owned yet, so do not rely on jump height here.
    direction = "LEFT"
    for frame in range(4200):
        state = session.state
        if state.samus_y >= 1390:
            break
        if state.pose in (137, 138):
            unmorph(session)
        if state.samus_x <= 45:
            direction = "RIGHT"
        elif state.samus_x >= 215:
            direction = "LEFT"
        phase = frame % 90
        if phase < 58:
            buttons = (direction, "B")
        else:
            buttons = (direction, "B", "A")
        hold(session, 1, *buttons, reason="business_descend")
    else:
        raise TimeoutError(f"business_to_hj_shaft: descent stalled: {state}")
    hold(session, 60, reason="business_bottom_settle")

    # The Sova can be tanked.  Approach from its left, then fire the red-door
    # Super explicitly facing left (direction+shot on the same first frame can
    # otherwise use the previous facing).
    for _ in range(320):
        state = session.state
        if state.samus_x <= 70:
            break
        hold(session, 1, "LEFT", "B", reason="business_red_door_approach")
    for _ in range(100):
        state = session.state
        if state.samus_x >= 92:
            break
        hold(session, 1, "RIGHT", reason="business_red_door_standoff")
    hold(session, 5, "LEFT", reason="business_red_door_brake")
    hold(session, 20, reason="business_red_door_settle")
    select_weapon(session, 2)
    hold(session, 3, "LEFT", reason="business_face_red_door")
    hold(session, 3, reason="business_face_red_door_release")
    hold(session, 2, "LEFT", "X", reason="business_red_door_super")
    hold(session, 80, reason="business_red_door_fuse")
    for _ in range(500):
        state = hold(session, 1, "LEFT", "B", "A", reason="business_enter_hj_shaft")
        if state.room_id == ROOM_HJ_SHAFT:
            break
    else:
        raise TimeoutError(f"business_to_hj_shaft: red door failed: {state}")
    return wait_ordinary_room(
        session, ROOM_HJ_SHAFT, settle_frames=280, label="business_to_hj_shaft"
    )

def play_hj_shaft_to_hj_room(session: ControllerSession) -> SuperMetroidState:
    """Hi-Jump E-Tank room right door → lower-left Hi-Jump Boots door."""
    require_room(session, ROOM_HJ_SHAFT, "hj_shaft_to_hj")
    unmorph(session)
    select_weapon(session, 0)

    # Cross the E-Tank plinth, collecting it naturally, and enter the low
    # morph tunnel.
    for _ in range(220):
        state = session.state
        if state.samus_x <= 390:
            break
        hold(session, 1, "LEFT", "B", reason="hj_shaft_etank_approach")
    # The item fanfare holds Samus against the right face of the plinth.
    # Morph back to the right after it finishes, then use that clear runway
    # for the leftward jump over the statue.
    hold(session, 480, reason="hj_shaft_etank_fanfare")
    ensure_morph(session)
    for _ in range(120):
        state = hold(session, 1, "RIGHT", reason="hj_shaft_etank_backoff")
        if state.samus_x >= 470:
            break
    unmorph(session)
    hold(session, 20, reason="hj_shaft_etank_stand")
    hold(session, 20, "LEFT", "B", reason="hj_shaft_etank_runup")
    for _ in range(140):
        state = hold(session, 1, "LEFT", "B", "A", reason="hj_shaft_etank_jump")
        if state.samus_x <= 310 and state.samus_y >= 180:
            break
    hold(session, 30, reason="hj_shaft_etank_jump_land")
    for _ in range(160):
        state = session.state
        if state.samus_x <= 310 and state.samus_y >= 180:
            break
        hold(session, 1, "LEFT", "B", reason="hj_shaft_low_tunnel")
    ensure_morph(session)
    for _ in range(700):
        state = hold(session, 1, "LEFT", reason="hj_shaft_morph_left")
        if state.samus_x <= 40 and state.samus_y >= 450:
            break
    else:
        raise TimeoutError(f"hj_shaft_to_hj: lower tunnel stalled: {state}")

    # Jump the short shaft beside the blue door and shoot it while rising.
    unmorph(session)
    select_weapon(session, 0)
    hold(session, 12, reason="hj_shaft_door_release")
    for _ in range(80):
        state = hold(session, 1, "A", reason="hj_shaft_door_jump")
        if state.samus_y <= 390:
            break
    hold(session, 2, "LEFT", "A", "X", reason="hj_shaft_blue_door_shot")
    for _ in range(420):
        state = hold(session, 1, "LEFT", "A", reason="hj_shaft_enter_hj")
        if state.room_id == ROOM_HJ:
            break
    else:
        raise TimeoutError(f"hj_shaft_to_hj: blue door failed: {state}")
    return wait_ordinary_room(
        session, ROOM_HJ, settle_frames=260, label="hj_shaft_to_hj"
    )

def play_hj_room_collect(session: ControllerSession) -> SuperMetroidState:
    """Destroy both pillar shot-block sets and collect Hi-Jump naturally."""
    require_room(session, ROOM_HJ, "hj_room_collect")
    unmorph(session)
    select_weapon(session, 0)
    hold(session, 20, reason="hj_room_entry_settle")

    # Left-facing down-shot opens the first half of the pillar.
    hold(session, 12, "LEFT", "B", reason="hj_room_first_runup")
    for _ in range(70):
        state = hold(session, 1, "LEFT", "B", "A", reason="hj_room_first_jump")
        if state.samus_y <= 52:
            break
    hold(session, 1, "DOWN", reason="hj_room_first_aim_down")
    hold(session, 1, "X", reason="hj_room_first_downshot")
    hold(session, 80, reason="hj_room_first_land")

    # Face right, jump vertically, and down-shoot the other orientation.
    hold(session, 2, "RIGHT", reason="hj_room_face_right")
    hold(session, 10, reason="hj_room_face_right_settle")
    for _ in range(80):
        state = hold(session, 1, "A", reason="hj_room_second_jump")
        if state.samus_y <= 53:
            break
    hold(session, 1, "DOWN", reason="hj_room_second_aim_down")
    hold(session, 1, "X", reason="hj_room_second_downshot")
    hold(session, 80, reason="hj_room_second_land")

    # Rebuild a leftward run and cross the now-open pillar.
    hold(session, 12, "RIGHT", "B", reason="hj_room_cross_backoff")
    hold(session, 15, "LEFT", "B", reason="hj_room_cross_runup")
    for _ in range(100):
        state = hold(session, 1, "LEFT", "B", "A", reason="hj_room_cross_pillar")
        if state.samus_x < 120:
            break
    hold(session, 80, reason="hj_room_left_land")

    # Shoot the Chozo statue from the right, then walk into the real PLM.
    for _ in range(80):
        state = session.state
        if state.samus_x >= 115:
            break
        hold(session, 1, "RIGHT", reason="hj_room_statue_approach")
    hold(session, 12, "LEFT", reason="hj_room_statue_brake")
    hold(session, 8, reason="hj_room_statue_settle")
    hold(session, 3, "LEFT", reason="hj_room_statue_face")
    hold(session, 3, reason="hj_room_statue_face_release")
    hold(session, 1, "X", reason="hj_room_statue_shot")
    hold(session, 60, reason="hj_room_statue_open")
    for _ in range(180):
        state = hold(session, 1, "LEFT", reason="hj_room_collect_item")
        if state.collected_items & ITEM_HI_JUMP:
            break
    else:
        raise TimeoutError(f"hj_room_collect: Hi-Jump PLM not collected: {state}")
    # Item-room controls remain locked substantially longer than the visible
    # pickup flash.  Let the full fanfare finish so the return composes without
    # relying on a save/load input reset.
    hold(session, 480, reason="hj_room_item_fanfare")
    return session.state

def play_warehouse_to_hijump(session: ControllerSession) -> SuperMetroidState:
    """Natural Warehouse entry → real Hi-Jump Boots collection."""
    play_warehouse_to_business(session)
    play_business_to_hj_shaft(session)
    play_hj_shaft_to_hj_room(session)
    return play_hj_room_collect(session)


"""Warehouse Entrance wall stack and elevator to Business Center."""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    ensure_morph,
    hold,
    play_run_shoot_exit,
    require_room,
    select_weapon,
    settle_hold,
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

def play_warehouse_wall_to_lower_lip(
    session: ControllerSession,
) -> SuperMetroidState:
    """Open Warehouse's three Super blocks and reach the lower-right lip.

    The stack at block x=15 is vertical: crouch-Super hits y=9, standing-Super
    hits y=8, and a five-frame hop-Super hits y=7.  This crosses the stack
    controller-only, but deliberately stops at x≈507/y≈315.  The no-Hi-Jump
    climb from that lower lip to the upper-right ledge is still open, so this
    is not a Warehouse→Zeela clearance.
    """
    _require_room(session, ROOM_WAREHOUSE, "warehouse_wall")
    _unmorph(session)
    _select_weapon(session, 2)
    for _ in range(160):
        state = _hold(session, 1, "RIGHT", "B", reason="warehouse_wall_runup")
        if state.samus_x >= 75:
            break
    _hold(session, 30, reason="warehouse_super_cooldown")

    _hold(session, 8, "DOWN", reason="warehouse_crouch")
    _hold(session, 1, "X", reason="warehouse_bottom_super")
    _hold(session, 30, reason="warehouse_bottom_open")
    _hold(session, 5, "UP", reason="warehouse_stand")
    settle_hold(session, 4, reason="warehouse_stand_settle")
    _hold(session, 1, "X", reason="warehouse_middle_super")
    _hold(session, 30, reason="warehouse_middle_open")
    _hold(session, 5, "A", reason="warehouse_tiny_hop")
    _hold(session, 1, "RIGHT", "X", reason="warehouse_top_super")
    _hold(session, 24, reason="warehouse_top_open")

    for _ in range(500):
        state = _hold(session, 1, "RIGHT", "B", "A", reason="warehouse_cross_stack")
        if state.samus_x >= 500 and state.samus_y >= 300:
            break
    else:
        raise TimeoutError(f"warehouse_wall: lower lip not reached: {state}")
    settle_hold(session, 30, reason="warehouse_lower_lip_settle")
    state = session.state
    if state.samus_x < 500 or state.samus_y < 300:
        raise TimeoutError(f"warehouse_wall: unstable lower lip: {state}")
    return state



def play_warehouse_to_business(session: ControllerSession) -> SuperMetroidState:
    """Warehouse Entrance elevator → natural Business Center spawn."""
    _require_room(session, ROOM_WAREHOUSE, "warehouse_to_business")
    _unmorph(session)
    for _ in range(180):
        state = session.state
        if state.samus_x >= 126:
            break
        _hold(session, 1, "RIGHT", reason="warehouse_elevator_position")
    _hold(session, 5, "LEFT", reason="warehouse_elevator_brake")
    settle_hold(session, 20, reason="warehouse_elevator_settle")
    for _ in range(700):
        state = _hold(session, 1, "DOWN", reason="warehouse_elevator_down")
        if state.room_id == ROOM_BUSINESS:
            break
    else:
        raise TimeoutError(f"warehouse_to_business: {state}")
    return _wait_ordinary_room(
        session, ROOM_BUSINESS, settle_frames=320, label="warehouse_to_business"
    )



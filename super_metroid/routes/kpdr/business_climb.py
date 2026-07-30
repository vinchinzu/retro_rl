"""Hi-Jump platform climb in Business Center (elevator return)."""

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

def _business_high_jump_platforms(session: ControllerSession) -> None:
    """Bottom Business Center floor → center elevator (Hi-Jump route)."""
    # Four forgiving setup jumps land on the first left platform (~y=1339).
    _unmorph(session)
    for direction in ("RIGHT", "LEFT", "LEFT", "RIGHT"):
        _hold(session, 12, reason="business_climb_release")
        _hold(session, 85, direction, "B", "A", reason="business_climb_setup")
        _hold(session, 30, reason="business_climb_setup_land")

    # y1339 → y1227.
    _unmorph(session)
    _hold(session, 20, reason="business_1339_settle")
    for _ in range(80):
        if session.state.samus_x <= 84:
            break
        _hold(session, 1, "LEFT", reason="business_1339_position")
    _hold(session, 4, "RIGHT", reason="business_1339_brake")
    _hold(session, 8, reason="business_1339_release")
    for frame in range(120):
        if frame < 14:
            buttons = ("LEFT", "A")
        elif frame < 24:
            buttons = ("A",)
        else:
            buttons = ("RIGHT", "A")
        state = _hold(session, 1, *buttons, reason="business_to_1227")
        if frame > 45 and state.samus_y == 1227 and state.samus_x >= 120:
            break
    _hold(session, 3, "LEFT", reason="business_1227_brake")
    _hold(session, 20, reason="business_1227_settle")

    # y1227 → right platform y1147.
    _unmorph(session)
    _hold(session, 15, reason="business_1227_release")
    for _ in range(80):
        if session.state.samus_x <= 105:
            break
        _hold(session, 1, "LEFT", reason="business_1227_back")
    _hold(session, 4, "RIGHT", reason="business_1227_brake2")
    _hold(session, 4, reason="business_1227_run_release")
    _hold(session, 8, "RIGHT", "B", reason="business_1227_runup")
    for frame in range(140):
        buttons = ("RIGHT", "B", "A") if frame < 90 else ("LEFT", "A")
        state = _hold(session, 1, *buttons, reason="business_to_1147")
        if frame > 88 and state.samus_y == 1147 and state.samus_x >= 192:
            break
    _hold(session, 3, "LEFT", reason="business_1147_brake")
    _hold(session, 20, reason="business_1147_settle")

    # y1147 → center platform y1067.
    _unmorph(session)
    _hold(session, 16, reason="business_1147_release")
    for frame in range(150):
        buttons = ("LEFT", "B", "A") if frame < 85 else ("RIGHT", "A")
        state = _hold(session, 1, *buttons, reason="business_to_1067")
        if frame > 100 and state.samus_y == 1067 and 95 <= state.samus_x <= 160:
            break
    _hold(session, 30, reason="business_1067_settle")

    # y1067 → y987 through the left edge of the overhead platform.
    _unmorph(session)
    _hold(session, 12, reason="business_1067_release")
    for _ in range(80):
        if session.state.samus_x <= 92:
            break
        _hold(session, 1, "LEFT", reason="business_1067_position")
    _hold(session, 4, "RIGHT", reason="business_1067_brake")
    _hold(session, 8, reason="business_1067_jump_release")
    for frame in range(100):
        buttons = ("A",) if frame < 14 else ("RIGHT", "B", "A")
        state = _hold(session, 1, *buttons, reason="business_to_987")
        if frame > 25 and state.samus_y == 987 and state.pose in (1, 2, 9, 10):
            break
    # This landing is on the extreme left pixel of the three-block platform;
    # nudge inward instead of braking back off its edge.
    _hold(session, 4, "RIGHT", reason="business_987_brake")
    _hold(session, 20, reason="business_987_settle")

    # y987 → right platform y907.
    _unmorph(session)
    _hold(session, 12, reason="business_987_release")
    _hold(session, 8, "RIGHT", "B", reason="business_987_runup")
    for frame in range(90):
        state = _hold(session, 1, "RIGHT", "B", "A", reason="business_to_907")
        if frame > 35 and state.samus_y == 907 and state.samus_x >= 160:
            break
    for _ in range(60):
        if session.state.samus_x <= 165:
            break
        _hold(session, 1, "LEFT", reason="business_907_brake")
    _hold(session, 2, "RIGHT", reason="business_907_brake")
    _hold(session, 20, reason="business_907_settle")

    # y907 → center y843.
    _unmorph(session)
    _hold(session, 12, reason="business_907_release")
    for _ in range(80):
        if session.state.samus_x >= 205:
            break
        _hold(session, 1, "RIGHT", reason="business_907_back")
    _hold(session, 3, "LEFT", reason="business_907_brake2")
    _hold(session, 5, reason="business_907_run_release")
    _hold(session, 8, "LEFT", "B", reason="business_907_runup")
    for frame in range(90):
        state = _hold(session, 1, "LEFT", "B", "A", reason="business_to_843")
        if frame > 35 and state.samus_y == 843 and 108 <= state.samus_x <= 160:
            break
    _hold(session, 2, "RIGHT", reason="business_843_brake")
    _hold(session, 20, reason="business_843_settle")

    # y843 → left y779.
    _unmorph(session)
    _hold(session, 12, reason="business_843_release")
    for _ in range(80):
        if session.state.samus_x >= 145:
            break
        _hold(session, 1, "RIGHT", reason="business_843_position")
    _hold(session, 3, "LEFT", reason="business_843_brake2")
    _hold(session, 6, reason="business_843_jump_release")
    for frame in range(90):
        buttons = ("A",) if frame < 10 else ("LEFT", "B", "A")
        state = _hold(session, 1, *buttons, reason="business_to_779")
        if frame > 25 and state.samus_y == 779 and state.samus_x <= 115:
            break
    _hold(session, 2, "RIGHT", reason="business_779_brake")
    _hold(session, 20, reason="business_779_settle")

    # y779 → center elevator y683.
    _unmorph(session)
    _hold(session, 12, reason="business_779_release")
    for _ in range(80):
        if session.state.samus_x <= 76:
            break
        _hold(session, 1, "LEFT", reason="business_779_position")
    _hold(session, 3, "RIGHT", reason="business_779_brake2")
    _hold(session, 6, reason="business_779_jump_release")
    for frame in range(120):
        buttons = ("A",) if frame < 18 else ("RIGHT", "B", "A")
        state = _hold(session, 1, *buttons, reason="business_to_elevator")
        if frame > 45 and state.samus_y == 683 and 95 <= state.samus_x <= 160:
            break
    _hold(session, 2, "LEFT", reason="business_elevator_brake")
    _hold(session, 20, reason="business_elevator_settle")




def play_business_to_warehouse(session: ControllerSession) -> SuperMetroidState:
    """Hi-Jump-assisted Business Center climb and elevator to Warehouse."""
    _require_room(session, ROOM_BUSINESS, "business_to_warehouse")
    _business_high_jump_platforms(session)
    for _ in range(1000):
        state = _hold(session, 1, "UP", reason="business_elevator_up")
        if state.room_id == ROOM_WAREHOUSE:
            break
    else:
        raise TimeoutError(f"business_to_warehouse: elevator failed: {state}")
    state = _wait_ordinary_room(
        session, ROOM_WAREHOUSE, settle_frames=360, label="business_to_warehouse"
    )
    # Let the Warehouse platform finish rising, then step back to the same
    # upper-left anchor used by the natural East Tunnel entry.
    _hold(session, 30, reason="warehouse_elevator_top")
    for _ in range(160):
        state = session.state
        if state.samus_x <= 40 and state.samus_y <= 145:
            break
        _hold(session, 1, "LEFT", reason="warehouse_elevator_exit")
    _hold(session, 30, reason="warehouse_elevator_exit_settle")
    return session.state





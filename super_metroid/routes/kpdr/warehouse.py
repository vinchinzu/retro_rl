"""Warehouse Entrance wall stack and elevator to Business Center."""

from __future__ import annotations

from typing import Literal

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    hold,
    require_room,
    select_weapon,
    settle_hold,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.rooms import (
    ITEM_HI_JUMP,
    ROOM_BUSINESS,
    ROOM_WAREHOUSE,
)
from super_metroid.routes.runtime import ControllerSession

WarehouseEntryMode = Literal["auto", "left_elevator", "right_reverse_stack"]


def _open_warehouse_stack(
    session: ControllerSession,
    *,
    face: Literal["LEFT", "RIGHT"],
    label: str,
) -> None:
    """Open Warehouse's three Super blocks while staged on the far side of ``face``.

    ``face="LEFT"``: fire leftward (right-ledge reverse lineage).
    ``face="RIGHT"``: fire rightward (ordinary left-platform power-on lineage).
    """
    select_weapon(session, 2)
    hold(session, 6, face, reason=f"{label}_face")
    hold(session, 8, "DOWN", reason=f"{label}_crouch")
    hold(session, 1, "X", reason=f"{label}_bottom_super")
    hold(session, 30, reason=f"{label}_bottom_open")
    hold(session, 5, "UP", reason=f"{label}_stand")
    settle_hold(session, 4, reason=f"{label}_stand_settle")
    hold(session, 1, "X", reason=f"{label}_middle_super")
    hold(session, 30, reason=f"{label}_middle_open")
    hold(session, 5, "A", reason=f"{label}_tiny_hop")
    hold(session, 1, face, "X", reason=f"{label}_top_super")
    hold(session, 30 if face == "LEFT" else 24, reason=f"{label}_top_open")


def _play_warehouse_reverse_stack(
    session: ControllerSession,
) -> SuperMetroidState:
    """Natural right ledge → left elevator platform, with Hi-Jump.

    The post-Varia return enters on the Zeela-door ledge (x≈722/y≈160),
    while the ordinary Warehouse→Business hop expects its usual left-side
    platform.  The floor stack does not cross directly: drop to the lower
    lip, clear the lower three-Super stack, climb to the mid and upper lips,
    then clear the same stack from the upper left-facing lip.  This preserves
    the normal power-on elevator path below.
    """
    label = "warehouse_reverse"
    if not (session.state.collected_items & ITEM_HI_JUMP):
        raise TimeoutError(f"{label}: right-ledge return requires Hi-Jump")

    # Leave the Zeela ledge, then reverse the fall at the lower-right lip.
    # A rightward correction after the 120-frame left spin lands at
    # x≈498/y≈315; holding left all the way instead pins x≈309/y≈251.
    for _ in range(120):
        hold(session, 1, "LEFT", "B", "A", reason=f"{label}_drop_left")
    for _ in range(100):
        state = hold(session, 1, "RIGHT", reason=f"{label}_lower_lip")
        if (
            445 <= state.samus_x <= 510
            and 300 <= state.samus_y <= 320
            and state.velocity_y == 0
        ):
            break
    else:
        raise TimeoutError(f"{label}: lower lip missed: {session.state}")

    # Clear the stack from the lower lip, then use the two Hi-Jump landings
    # which lead to the upper left-facing firing lip.
    _open_warehouse_stack(session, face="LEFT", label=f"{label}_lower_stack")
    for _ in range(180):
        hold(session, 1, "LEFT", "B", "A", reason=f"{label}_lower_cross")
    hold(session, 8, "UP", reason=f"{label}_mid_stand")
    settle_hold(session, 20, reason=f"{label}_mid_settle")
    for _ in range(105):
        hold(session, 1, "LEFT", "B", "A", reason=f"{label}_mid_climb")
    hold(session, 8, "UP", reason=f"{label}_upper_stand")
    settle_hold(session, 20, reason=f"{label}_upper_settle")
    for _ in range(105):
        hold(session, 1, "LEFT", "B", "A", reason=f"{label}_upper_climb")

    # The block stack is still closed at this upper lip.  Reopen it from the
    # left-facing side, then land on the regular elevator approach (x≈37).
    _open_warehouse_stack(session, face="LEFT", label=f"{label}_upper_stack")
    hold(session, 8, "UP", reason=f"{label}_exit_stand")
    settle_hold(session, 20, reason=f"{label}_exit_settle")
    for _ in range(180):
        state = hold(session, 1, "LEFT", "B", "A", reason=f"{label}_exit")
        if state.samus_x <= 40 and state.samus_y <= 150:
            break
    else:
        raise TimeoutError(f"{label}: left elevator platform missed: {session.state}")
    return session.state


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
    require_room(session, ROOM_WAREHOUSE, "warehouse_wall")
    unmorph(session)
    for _ in range(160):
        state = hold(session, 1, "RIGHT", "B", reason="warehouse_wall_runup")
        if state.samus_x >= 75:
            break
    hold(session, 30, reason="warehouse_super_cooldown")
    _open_warehouse_stack(session, face="RIGHT", label="warehouse_wall")

    for _ in range(500):
        state = hold(session, 1, "RIGHT", "B", "A", reason="warehouse_cross_stack")
        if state.samus_x >= 500 and state.samus_y >= 300:
            break
    else:
        raise TimeoutError(f"warehouse_wall: lower lip not reached: {state}")
    settle_hold(session, 30, reason="warehouse_lower_lip_settle")
    state = session.state
    if state.samus_x < 500 or state.samus_y < 300:
        raise TimeoutError(f"warehouse_wall: unstable lower lip: {state}")
    return state


def resolve_warehouse_entry_mode(
    state: SuperMetroidState,
    *,
    entry_mode: WarehouseEntryMode = "auto",
) -> Literal["left_elevator", "right_reverse_stack"]:
    """Choose Warehouse→Business lineage once at hop start.

    ``auto`` maps right-ledge returns (x>400, post-Zeela) to reverse stack;
    left-platform / ordinary power-on entries stay on the elevator path.
    Explicit modes skip the pose heuristic entirely.
    """
    if entry_mode == "auto":
        return "right_reverse_stack" if state.samus_x > 400 else "left_elevator"
    return entry_mode


def play_warehouse_to_business(
    session: ControllerSession,
    *,
    entry_mode: WarehouseEntryMode = "auto",
) -> SuperMetroidState:
    """Warehouse Entrance → natural Business Center spawn.

    Entry lineage is selected **once** at hop start via ``entry_mode``:

    - ``left_elevator`` — ordinary power-on left platform → elevator down
    - ``right_reverse_stack`` — post-Varia Zeela ledge → Hi-Jump reverse stack
    - ``auto`` (default) — resolve from pose at hop start (x>400 → reverse)

    Continuous composition may pass an explicit mode; do not re-discover
    lineage mid-frame inside nested helpers.
    """
    require_room(session, ROOM_WAREHOUSE, "warehouse_to_business")
    unmorph(session)
    mode = resolve_warehouse_entry_mode(session.state, entry_mode=entry_mode)
    if mode == "right_reverse_stack":
        _play_warehouse_reverse_stack(session)
    for _ in range(180):
        state = session.state
        if state.samus_x >= 126:
            break
        hold(session, 1, "RIGHT", reason="warehouse_elevator_position")
    hold(session, 5, "LEFT", reason="warehouse_elevator_brake")
    settle_hold(session, 20, reason="warehouse_elevator_settle")
    for _ in range(700):
        state = hold(session, 1, "DOWN", reason="warehouse_elevator_down")
        if state.room_id == ROOM_BUSINESS:
            break
    else:
        raise TimeoutError(f"warehouse_to_business: {state}")
    return wait_ordinary_room(
        session, ROOM_BUSINESS, settle_frames=320, label="warehouse_to_business"
    )

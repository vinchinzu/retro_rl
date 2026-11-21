"""Restore-free movement out of Draygon's lair after collecting Space Jump."""

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    hold,
    hold_until,
    require_room,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.maridia.halfie_climb import (
    ROOM_COLOSSEUM,
    ROOM_DRAYGON,
    ROOM_HALFIE_CLIMB,
    ROOM_PRECIOUS,
)
from super_metroid.routes.runtime import ControllerSession

ROOM_SPACE_JUMP = 0xD9AA
ROOM_EAST_CACTUS = 0xDA2B
ROOM_WEST_CACTUS = 0xD9FE
ROOM_BUTTERFLY = 0xD5EC
ITEM_SPACE_JUMP = 0x0200
SPIN_JUMP = 3
RISING = 1
FALLING = 2


def _space_jump(
    state: SuperMetroidState, jump_held: bool, *, fall_speed: int = 2,
) -> bool:
    if state.movement_type == SPIN_JUMP:
        # Keep the takeoff button down on the rise, release it at the apex,
        # then press again after Samus has gained enough downward speed.
        return state.vertical_direction == RISING or (
            state.vertical_direction == FALLING and state.velocity_y >= fall_speed and not jump_held
        )
    return state.vertical_direction == 0 and not jump_held


def _climb(
    session: ControllerSession, *,
    shaft_x: int, ledge_y: int, label: str, fall_speed: int = 2,
) -> None:
    if not session.state.equipped_items & ITEM_SPACE_JUMP:
        raise RuntimeError(f"{label}: Space Jump is required")
    jump_held = False
    for _ in range(340):
        state = session.state
        if state.samus_y <= ledge_y:
            return
        if state.samus_x < shaft_x - 3:
            steer = ("RIGHT",)
        elif state.samus_x > shaft_x + 3:
            steer = ("LEFT",)
        else:
            steer = ()
        jump = _space_jump(state, jump_held, fall_speed=fall_speed)
        if jump and state.vertical_direction == 0 and not steer:
            steer = ("RIGHT",) if state.samus_x <= shaft_x else ("LEFT",)
        keys = (*steer, "B") if steer else ()
        hold(session, 1, *keys, *(("A",) if jump else ()), reason=label)
        jump_held = jump
    raise TimeoutError(f"{label}: failed to climb to y={ledge_y}: {session.state}")


def _beam_exit(
    session: ControllerSession, target_room: int, direction: str, *, label: str,
) -> SuperMetroidState:
    for frame in range(120):
        state = session.state
        if state.game_state != 8 or state.door_transition:
            return wait_ordinary_room(session, target_room, settle_frames=280, label=label)
        shoot = 118 <= state.samus_y <= 144 and frame % 7 == 0
        hold(session, 1, direction, "B", *(("X",) if shoot else ()), reason=label)
    raise TimeoutError(f"{label}: missed the door to 0x{target_room:04X}: {session.state}")


def play_space_jump_room_exit(session: ControllerSession) -> SuperMetroidState:
    """Run off the item pedestal and jump straight to the east hatch."""
    label = "space_jump_room_exit"
    require_room(session, ROOM_SPACE_JUMP, label)
    if not session.state.equipped_items & ITEM_SPACE_JUMP:
        raise RuntimeError(f"{label}: Space Jump is required")
    if session.state.selected_item:
        hold(session, 1, "Y", reason=f"{label}_beam")
    # The short ceiling over the pedestal cuts off a jump from the item.
    # Drop to the lower floor and take off beneath the right ledge.
    hold_until(
        session, lambda s: s.samus_x >= 168 and s.samus_y >= 187 and s.vertical_direction == 0,
        "RIGHT", "B", timeout=95, reason=f"{label}_drop",
    )
    hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_takeoff")
    hold_until(
        session, lambda s: s.samus_y <= 128 and s.samus_x >= 192,
        "RIGHT", "B", "A", timeout=45, reason=f"{label}_jump_to_hatch",
    )
    return _beam_exit(session, ROOM_DRAYGON, "RIGHT", label=label)


def play_draygon_room_exit(session: ControllerSession) -> SuperMetroidState:
    """Cross the empty boss arena and chain Space Jumps to the top hatch."""
    label = "draygon_room_exit"
    require_room(session, ROOM_DRAYGON, label)
    hold_until(
        session, lambda s: s.samus_x >= 390,
        "RIGHT", "B", timeout=145, reason=f"{label}_runway",
    )
    _climb(session, shaft_x=390, ledge_y=138, label=f"{label}_climb")
    return _beam_exit(session, ROOM_PRECIOUS, "RIGHT", label=label)


def play_precious_space_jump(session: ControllerSession) -> SuperMetroidState:
    """Climb the gap west of the middle platform and leave for Colosseum."""
    label = "precious_space_jump"
    require_room(session, ROOM_PRECIOUS, label)
    _climb(session, shaft_x=90, ledge_y=114, fall_speed=1, label=f"{label}_climb")
    return _beam_exit(session, ROOM_COLOSSEUM, "LEFT", label=label)


def play_halfie_return_drop(session: ControllerSession) -> SuperMetroidState:
    """Drop from Colosseum to the west hatch without landing in the pit."""
    label = "halfie_return_drop"
    require_room(session, ROOM_HALFIE_CLIMB, label)
    for frame in range(180):
        state = session.state
        if state.game_state != 8 or state.door_transition:
            return wait_ordinary_room(session, ROOM_EAST_CACTUS, settle_frames=280, label=label)
        # Ease off near the left wall to stay outside the spike columns.
        keys = ("LEFT", "B") if state.samus_x > 77 or state.samus_y >= 310 else ()
        shoot = state.samus_x < 105 and 300 <= state.samus_y <= 400 and frame % 7 == 0
        hold(session, 1, *keys, *(("X",) if shoot else ()), reason=label)
    raise TimeoutError(f"{label}: missed the hatch: {session.state}")


def play_west_cactus_drop(session: ControllerSession) -> SuperMetroidState:
    """Drop onto the middle shelf and descend to the Butterfly Room hatch."""
    label = "west_cactus_drop"
    require_room(session, ROOM_WEST_CACTUS, label)
    for frame in range(180):
        state = session.state
        if state.game_state != 8 or state.door_transition:
            return wait_ordinary_room(session, ROOM_BUTTERFLY, settle_frames=280, label=label)
        # Stay over the shelf while falling, then move left below its overhang.
        if state.samus_y == 171 and state.vertical_direction == 0 and state.samus_x <= 135:
            hold_until(
                session, lambda s: s.samus_x >= 143,
                "RIGHT", timeout=40, reason=f"{label}_clear_lip",
            )
            continue
        keys = ("LEFT", "B") if state.samus_x > 135 or state.samus_y >= 291 else ()
        shoot = state.samus_x < 110 and state.samus_y > 354 and frame % 7 == 0
        hold(session, 1, *keys, *(("X",) if shoot else ()), reason=label)
    raise TimeoutError(f"{label}: missed the hatch: {session.state}")

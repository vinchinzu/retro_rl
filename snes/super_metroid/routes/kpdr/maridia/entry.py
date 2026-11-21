"""Early Maridia entry from the west side of the intact tube."""

from super_metroid.ram import ADDR_EVENT_FLAGS, SuperMetroidState, read_wram_u8
from super_metroid.routes.controller_common import (
    hold,
    hold_until,
    require_room,
    select_weapon,
    wait_ordinary_room,
    wait_until,
)
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.shinespark import SPARK_POSES
from super_metroid.routes.skills.shinespark_plans import short_charge_plan

ROOM_TUBE = 0xCEFB
ROOM_MAIN_STREET = 0xCFC9
ROOM_EVEREST = 0xD0B9
ROOM_CRAB_SHAFT = 0xD1A3
ROOM_AQUEDUCT = 0xD5A7
ROOM_BOTWOON_HALLWAY = 0xD617
ROOM_BOTWOON = 0xD95E
_TUBE_BROKEN_MASK = 0x08


def play_tube_direct(session: ControllerSession) -> SuperMetroidState:
    """Break the tube and climb to Main Street straight from the west entry."""
    label = "maridia_tube_direct"
    require_room(session, ROOM_TUBE, label)
    if session.state.power_bombs < 1:
        raise RuntimeError(f"{label}: need one Power Bomb")
    select_weapon(session, 3)

    # Lay the bomb over the right-hand ledge so Samus can land after the
    # glass breaks. Jump into the glass while the power bomb is active.
    hold(session, 30, "RIGHT", "B", reason=f"{label}_run")
    hold(session, 2, "DOWN", reason=f"{label}_crouch")
    hold(session, 5, reason=f"{label}_release_down")
    hold(session, 2, "DOWN", reason=f"{label}_morph")
    hold(session, 15, reason=f"{label}_settle_morph")
    hold(session, 1, "X", reason=f"{label}_power_bomb")
    hold(session, 180, reason=f"{label}_detonate")
    hold(session, 8, "UP", reason=f"{label}_unmorph")
    hold(session, 7, reason=f"{label}_settle")
    hold(session, 1, "A", reason=f"{label}_jump_into_glass")
    wait_until(
        session,
        lambda _: bool(read_wram_u8(session.env, ADDR_EVENT_FLAGS + 1) & _TUBE_BROKEN_MASK),
        timeout=400,
        reason=f"{label}_glass_break",
    )
    hold(session, 60, reason=f"{label}_land")

    # Climb from the right platform through the two left ledges to the
    # platform directly under the ceiling door.
    hold(session, 13, "A", reason=f"{label}_first_jump")
    hold(session, 29, "LEFT", "A", reason=f"{label}_first_jump")
    hold(session, 24, "LEFT", reason=f"{label}_first_jump")
    hold(session, 40, reason=f"{label}_land_middle")
    hold(session, 52, "LEFT", "B", "A", reason=f"{label}_second_jump")
    hold(session, 35, reason=f"{label}_land_left")
    hold(session, 1, "Y", reason=f"{label}_beam")
    hold(session, 38, "RIGHT", "A", reason=f"{label}_third_jump")
    hold(session, 24, "RIGHT", reason=f"{label}_line_up")
    hold(session, 9, reason=f"{label}_stop")
    hold(session, 8, "A", reason=f"{label}_jump_to_shoot")
    hold(session, 4, "UP", "A", reason=f"{label}_aim_at_ceiling")
    hold(session, 1, "UP", "A", "X", reason=f"{label}_shoot_door")
    hold(session, 16, "UP", "A", reason=f"{label}_clear_shot")
    hold(session, 45, reason=f"{label}_land_top")
    hold(session, 1, "A", reason=f"{label}_exit_jump")
    hold(session, 55, "UP", "A", reason=f"{label}_enter_door")
    return wait_until(
        session,
        lambda state: (
            state.room_id == ROOM_MAIN_STREET and state.game_state == 8
            and not state.door_transition and state.samus_y == 1979
        ),
        timeout=240,
        reason=f"{label}_land_main_street",
    )


def _vertical_short_spark(session: ControllerSession, label: str) -> SuperMetroidState:
    for names in short_charge_plan(stutter=True):
        hold(session, 1, *names, reason=f"{label}_short_charge")
    if not session.state.speed_boosting:
        raise RuntimeError(f"{label}: short charge did not reach full boost")
    state = hold(session, 1, "DOWN", reason=f"{label}_store")
    if not state.shinesparking:
        raise RuntimeError(f"{label}: crouch did not store a shinespark")
    hold(session, 7, reason=f"{label}_settle_crouch")
    hold(session, 1, "A", reason=f"{label}_activate")
    for _ in range(12):
        hold(session, 1, "UP", "A", reason=f"{label}_aim_up")
    if session.state.pose not in SPARK_POSES:
        raise RuntimeError(f"{label}: vertical spark did not start: {session.state}")
    for _ in range(168):
        if session.state.pose not in SPARK_POSES:
            return session.state
        hold(session, 1, "UP", "A", reason=f"{label}_spark")
    raise TimeoutError(f"{label}: spark did not finish: {session.state}")


def play_everest_short_sparks(session: ControllerSession) -> SuperMetroidState:
    """Reach Crab Shaft via two vertical sparks from the lower hills."""
    label = "maridia_everest"
    require_room(session, ROOM_EVEREST, label)
    if session.state.selected_item:
        hold(session, 1, "Y", reason=f"{label}_beam")

    # The stutter runway begins in the valley near x=952. Let the running
    # momentum settle before the short charge onto the far slope.
    for frame in range(540):
        state = session.state
        if state.samus_x >= 952 and state.samus_y >= 770 and state.vertical_direction == 0:
            break
        names = ("RIGHT", "B", "X") if frame % 9 == 2 else ("RIGHT", "B")
        hold(session, 1, *names, reason=f"{label}_lower_hills")
    else:
        raise TimeoutError(f"{label}: missed the lower runway: {session.state}")
    hold(session, 12, reason=f"{label}_brake")
    _vertical_short_spark(session, f"{label}_first")

    # Fall right onto the plateau, then stutter charge to x≈1410. To the
    # right of that window the spark hits the underside of the door ledge.
    for _ in range(150):
        state = session.state
        if state.samus_x >= 1228 and state.vertical_direction == 0:
            break
        hold(session, 1, "RIGHT", "B", reason=f"{label}_land_plateau")
    else:
        raise TimeoutError(f"{label}: missed the right plateau: {session.state}")
    hold(session, 12, reason=f"{label}_brake_plateau")
    _vertical_short_spark(session, f"{label}_second")
    for frame in range(165):
        state = session.state
        if state.room_id == ROOM_CRAB_SHAFT:
            return wait_ordinary_room(session, ROOM_CRAB_SHAFT, settle_frames=320, label=label)
        names = ("RIGHT", "B", "X") if state.samus_x > 1460 and frame % 7 == 2 else ("RIGHT", "B")
        hold(session, 1, *names, reason=f"{label}_exit")
    raise TimeoutError(f"{label}: missed the Crab Shaft door: {session.state}")


def play_crab_shaft_direct(session: ControllerSession) -> SuperMetroidState:
    """Shoot the shaft floor with beam and leave east through the green door."""
    label = "maridia_crab_shaft"
    require_room(session, ROOM_CRAB_SHAFT, label)
    hold(session, 1, "Y", reason=f"{label}_beam")
    hold(session, 61, reason=f"{label}_brake")
    hold(session, 15, "RIGHT", reason=f"{label}_approach")
    hold(session, 14, reason=f"{label}_align")
    hold(session, 2, "A", reason=f"{label}_hop")
    hold(session, 9, "RIGHT", "A", reason=f"{label}_hop")
    hold(session, 18, "RIGHT", reason=f"{label}_align_shot")
    hold(session, 1, "RIGHT", "DOWN", "X", reason=f"{label}_shoot_floor")
    hold(session, 7, "DOWN", "X", reason=f"{label}_shoot_floor")
    hold(session, 26, "DOWN", reason=f"{label}_release")
    hold(session, 12, "DOWN", "A", reason=f"{label}_jump_to_shoot")
    hold(session, 7, "DOWN", "X", reason=f"{label}_shoot_floor")
    hold(session, 10, "DOWN", reason=f"{label}_release")
    hold(session, 5, "DOWN", "X", reason=f"{label}_shoot_floor")
    hold(session, 17, "DOWN", reason=f"{label}_fall")
    wait_until(session, lambda s: s.samus_y == 731, timeout=65, reason=f"{label}_first_ledge")

    # Slip through the staggered one-tile gaps, right first, then left.
    hold(session, 14, "RIGHT", reason=f"{label}_first_gap")
    hold(session, 3, "RIGHT", "DOWN", reason=f"{label}_first_gap")
    hold(session, 8, "DOWN", reason=f"{label}_fall")
    hold(session, 23, "LEFT", reason=f"{label}_second_gap")
    hold(session, 5, reason=f"{label}_fall")
    hold(session, 11, "RIGHT", reason=f"{label}_center_gap")
    hold_until(session, lambda s: s.samus_x >= 313, "RIGHT", "B", timeout=140, reason=f"{label}_bottom")
    hold(session, 9, "RIGHT", "B", "A", reason=f"{label}_jump_gap")
    hold(session, 7, "RIGHT", "B", reason=f"{label}_jump_gap")
    hold(session, 26, "RIGHT", reason=f"{label}_door_approach")
    hold(session, 10, "RIGHT", "A", reason=f"{label}_door_ledge")
    hold(session, 4, "RIGHT", reason=f"{label}_door_ledge")
    wait_until(
        session, lambda s: s.samus_y == 907 and s.vertical_direction == 0,
        timeout=75, reason=f"{label}_land",
    )
    select_weapon(session, 2)
    hold(session, 1, "RIGHT", "X", reason=f"{label}_open_green")
    hold_until(
        session, lambda s: s.room_id == ROOM_AQUEDUCT, "RIGHT", "B",
        timeout=160, reason=f"{label}_enter_aqueduct",
    )
    return wait_ordinary_room(session, ROOM_AQUEDUCT, settle_frames=320, label=label)


def play_aqueduct_short_spark(session: ControllerSession) -> SuperMetroidState:
    """Power-bomb the floor and spark up the left passage to Botwoon Hallway."""
    label = "maridia_aqueduct"
    require_room(session, ROOM_AQUEDUCT, label)
    if session.state.power_bombs < 1:
        raise RuntimeError(f"{label}: need one Power Bomb")
    hold_until(session, lambda state: state.samus_x >= 101, "RIGHT", "B", timeout=80, reason=f"{label}_approach")
    hold(session, 45, reason=f"{label}_land")
    select_weapon(session, 3)
    hold(session, 2, "DOWN", reason=f"{label}_crouch")
    hold(session, 5, reason=f"{label}_release_down")
    hold(session, 3, "DOWN", reason=f"{label}_morph")
    hold(session, 12, reason=f"{label}_settle_morph")
    hold(session, 1, "X", reason=f"{label}_power_bomb")
    wait_until(
        session,
        lambda state: state.samus_y == 665 and state.vertical_direction == 0,
        timeout=275,
        reason=f"{label}_drop_through_blocks",
    )
    hold_until(session, lambda state: state.samus_x >= 135, "RIGHT", timeout=30, reason=f"{label}_align_floor")
    hold(session, 1, "Y", reason=f"{label}_beam")
    hold(session, 8, "UP", reason=f"{label}_unmorph")
    hold(session, 8, reason=f"{label}_settle")
    hold(session, 8, "UP", reason=f"{label}_stand")
    hold(session, 8, reason=f"{label}_settle")
    _vertical_short_spark(session, label)

    # Drift left over the lip of the y=155 walkway before dropping back
    # through the passage, then jump to the pedestal beneath the hatch.
    hold_until(session, lambda state: state.samus_x <= 184, "LEFT", "B", timeout=100, reason=f"{label}_top_walk")
    hold(session, 26, "LEFT", "A", reason=f"{label}_to_pedestal")
    hold(session, 4, "UP", "A", reason=f"{label}_aim")
    hold(session, 1, "UP", "A", "X", reason=f"{label}_open_hatch")
    hold(session, 20, reason=f"{label}_land_pedestal")
    hold(session, 1, "A", reason=f"{label}_jump_to_hatch")
    hold(session, 4, "UP", "A", reason=f"{label}_aim")
    hold(session, 1, "UP", "A", "X", reason=f"{label}_shoot_hatch")
    hold(session, 45, "UP", "A", reason=f"{label}_enter_hatch")
    return wait_ordinary_room(session, ROOM_BOTWOON_HALLWAY, settle_frames=320, label=label)


def play_botwoon_hallway_direct(session: ControllerSession) -> SuperMetroidState:
    """Shoot the hallway's block gates and short-charge across the speed blocks."""
    label = "maridia_botwoon_hallway"
    require_room(session, ROOM_BOTWOON_HALLWAY, label)
    hold(session, 16, "UP", reason=f"{label}_aim")
    hold(session, 4, "UP", "X", reason=f"{label}_clear_shaft")
    hold(session, 12, reason=f"{label}_release")
    hold(session, 36, "RIGHT", "A", reason=f"{label}_jump_to_walkway")
    for frame in range(225):
        if session.state.samus_x >= 500:
            break
        names = ("RIGHT", "B", "X") if frame % 7 == 3 else ("RIGHT", "B")
        hold(session, 1, *names, reason=f"{label}_first_corridor")
    else:
        raise TimeoutError(f"{label}: missed the speed runway: {session.state}")
    hold(session, 12, reason=f"{label}_brake")
    for names in short_charge_plan(stutter=True):
        hold(session, 1, *names, reason=f"{label}_short_charge")
    if not session.state.speed_boosting:
        raise RuntimeError(f"{label}: short charge did not reach full boost")
    hold_until(
        session, lambda s: s.samus_x >= 802, "RIGHT", "B",
        timeout=80, reason=f"{label}_break_speed_blocks",
    )
    hold(session, 22, "RIGHT", "B", "A", reason=f"{label}_jump_to_exit")
    hold(session, 10, "RIGHT", "B", reason=f"{label}_release_jump")
    hold(session, 20, "RIGHT", "B", "A", reason=f"{label}_jump_to_door")
    for frame in range(250):
        if session.state.room_id == ROOM_BOTWOON:
            return wait_ordinary_room(session, ROOM_BOTWOON, settle_frames=320, label=label)
        names = ("RIGHT", "B", "X") if frame % 7 == 3 else ("RIGHT", "B")
        hold(session, 1, *names, reason=f"{label}_exit")
    raise TimeoutError(f"{label}: missed the Botwoon door: {session.state}")

"""Restore-free wall jumps and Full Halfie shinespark for early Maridia."""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState, read_wram_u16
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

ROOM_BOTWOON_ETANK = 0xD7E4
ROOM_HALFIE_CLIMB = 0xD913
ROOM_COLOSSEUM = 0xD72A
ROOM_PRECIOUS = 0xD78F
ROOM_DRAYGON = 0xDA60

_POSE_LATCH_RIGHT = 83
_POSE_LATCH_LEFT = 84
_PLM_HEADERS = 0x1C37
_PLM_INSTRUCTIONS = 0x1D27
_PLM_TIMERS = 0xDE1C
_DRAYGON_EYE_PLM = 0xDB48
_DRAYGON_EYE_OPEN = 0xD97D


def play_halfie_climb_walljump(session: ControllerSession) -> SuperMetroidState:
    """Climb Halfie Climb Room (0xD913) to Colosseum (0xD72A) via single-wall jumps.

    Requires Gravity + Hi-Jump (no Grapple, no Space Jump). Runs right to x=112
    (well clear of the x=219 floor spikes), spin-jumps onto the right vertical
    shaft wall (x=187 lower / x=175 upper), chains 5 single-wall kicks to the
    y=139 exit ledge, and shoots through the blue door into Colosseum.
    """
    label = "halfie_climb_wj"
    require_room(session, ROOM_HALFIE_CLIMB, label)

    # 1) Approach the vertical shaft bottom (stop left of the x=219 spike pit).
    for _ in range(70):
        s = session.state
        if s.samus_x >= 112:
            break
        hold(session, 1, "RIGHT", "B", reason=f"{label}_approach")

    # 2) Initial spin-jump up-right onto the right shaft wall.
    hold(session, 24, "RIGHT", "B", "A", reason=f"{label}_init_jump")

    prev_x = -1
    wall_touch = 0
    for f in range(800):
        s = session.state
        if s.room_id == ROOM_COLOSSEUM:
            return wait_ordinary_room(session, ROOM_COLOSSEUM, settle_frames=320, label=label)

        # Top-right exit ledge (y <= 155, x >= 165): dash right and shoot blue door!
        if s.samus_y <= 155 and s.samus_x >= 165:
            if f % 8 == 3:
                hold(session, 1, "RIGHT", "B", "X", reason=f"{label}_door_shoot")
            else:
                hold(session, 1, "RIGHT", "B", reason=f"{label}_door_run")
            continue

        # Grounded recovery if Samus lands on a ledge or bottom floor.
        if s.vertical_direction == 0 and s.velocity_y == 0:
            if s.samus_x > 160 and s.samus_y > 580:
                hold(session, 18, "LEFT", "B", "A", reason=f"{label}_escape_pit")
            else:
                hold(session, 20, "RIGHT", "B", "A", reason=f"{label}_rejump")
            wall_touch = 0
            prev_x = -1
            continue

        # Live right-wall contact detection (tracks x=187 lower wall & x=175 upper wall).
        wall_touch = wall_touch + 1 if (s.samus_x == prev_x and s.samus_x >= 165) else 0
        prev_x = s.samus_x
        if s.pose == _POSE_LATCH_RIGHT or wall_touch >= 2:
            if s.pose != _POSE_LATCH_RIGHT:
                hold(session, 2, "LEFT", reason=f"{label}_latch_r")
            hold(session, 2, "LEFT", "A", reason=f"{label}_kick_r")
            hold(session, 16, "RIGHT", "B", "A", reason=f"{label}_return_r")
            wall_touch = 0
            prev_x = -1
        elif s.pose == _POSE_LATCH_LEFT:
            hold(session, 2, "RIGHT", "A", reason=f"{label}_kick_l")
            hold(session, 16, "RIGHT", "B", "A", reason=f"{label}_return_l")
            wall_touch = 0
            prev_x = -1
        else:
            hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_steer_r")

    raise TimeoutError(f"{label}: failed to reach Colosseum 0xD72A: {session.state}")


def play_precious_room_direct(session: ControllerSession) -> SuperMetroidState:
    """Bomb through the floor, then open Draygon's eye door with one Super."""
    label = "precious_room_direct"
    require_room(session, ROOM_PRECIOUS, label)
    if session.state.super_missiles < 1:
        raise RuntimeError(f"{label}: need one Super for the eye door")
    if session.state.selected_item:
        hold(session, 1, "Y", reason=f"{label}_beam")
    for _ in range(180):
        s = session.state
        if s.samus_y >= 340 and s.velocity_y == 0:
            break
        if s.samus_y < 185:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_step_off")
        elif s.samus_x > 115:
            hold(session, 1, "LEFT", "B", reason=f"{label}_align_l")
        elif s.samus_x < 98:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_align_r")
        else:
            hold(session, 1, reason=f"{label}_fall")

    # Morph & break floor bomb-blocks at y=347 -> drop to y=491 -> roll right off ledge to y=651.
    hold(session, 2, "DOWN", reason=f"{label}_m1")
    hold(session, 2, reason=f"{label}_m1s")
    hold(session, 2, "DOWN", reason=f"{label}_m2")
    hold(session, 2, reason=f"{label}_m2s")
    hold(session, 2, "X", reason=f"{label}_bomb")
    for _ in range(120):
        s = hold(session, 1, reason=f"{label}_wait_bomb")
        if s.samus_y >= 480:
            break
    for _ in range(120):
        s = hold(session, 1, "RIGHT", reason=f"{label}_roll_r") if session.state.samus_y < 600 else hold(session, 1, reason=f"{label}_drop_bot")
        if s.samus_y >= 635 and s.velocity_y == 0:
            break
    hold(session, 2, "UP", reason=f"{label}_u1")
    hold(session, 2, reason=f"{label}_u1s")
    hold(session, 2, "UP", reason=f"{label}_u2")
    hold(session, 2, reason=f"{label}_u2s")

    # Wait within projectile range for the eye to open. Once the Super kills
    # it, the remaining shell takes a beam shot.
    select_weapon(session, 2)
    hold_until(session, lambda s: s.samus_x <= 78, "LEFT", "B", timeout=180, reason=f"{label}_to_eye")
    eye_slot = next(
        (
            offset for offset in range(0, 0x50, 2)
            if read_wram_u16(session.env, _PLM_HEADERS + offset) == _DRAYGON_EYE_PLM
        ),
        None,
    )
    if eye_slot is None:
        raise RuntimeError(f"{label}: missing Draygon eye door")
    wait_until(
        session,
        lambda _: (
            read_wram_u16(session.env, _PLM_INSTRUCTIONS + eye_slot) == _DRAYGON_EYE_OPEN
            and read_wram_u16(session.env, _PLM_TIMERS + eye_slot) >= 8
        ),
        timeout=300,
        reason=f"{label}_eye_open",
    )
    hold(session, 1, "LEFT", "X", reason=f"{label}_shoot_eye")
    hold(session, 8, "LEFT", "B", reason=f"{label}_projectile_travel")
    hold(session, 1, "Y", reason=f"{label}_beam")
    for frame in range(200):
        if session.state.room_id == ROOM_DRAYGON:
            return wait_ordinary_room(session, ROOM_DRAYGON, settle_frames=320, label=label)
        names = ("LEFT", "X") if frame % 8 == 3 else ("LEFT", "B")
        hold(session, 1, *names, reason=f"{label}_enter")
    raise TimeoutError(f"{label}: failed to enter Draygon's Room 0xDA60: {session.state}")


def play_full_halfie_chain(session: ControllerSession) -> SuperMetroidState:
    """Carry Speed Booster through Halfie, then spark across Colosseum (r206–208).

    Requires Gravity, Hi-Jump, Speed Booster, Wave Beam, a Super and enough
    energy for the ceiling spark. The chain uses no Grapple or Space Jump.
    """
    require_room(session, ROOM_BOTWOON_ETANK, "full_halfie")
    _carry_speed_into_halfie(session)
    _carry_charge_into_colosseum(session)
    return _spark_across_colosseum(session)


def _carry_speed_into_halfie(session: ControllerSession) -> None:
    label = "full_halfie_r206"
    if session.state.selected_item:
        hold(session, 1, "Y", reason=f"{label}_beam")
    # Turn back from the door and settle on the last ledge, leaving enough
    # runway to get four boost ticks before crossing the open door.
    for _ in range(300):
        if session.state.samus_x >= 1740:
            break
        hold(session, 1, "RIGHT", "B", reason=f"{label}_cross")
    for _ in range(90):
        if session.state.samus_x <= 1632:
            break
        hold(session, 1, "LEFT", "B", reason=f"{label}_back")
    hold(session, 8, reason=f"{label}_stop")
    hold(session, 1, "RIGHT", "X", reason=f"{label}_face")
    hold(session, 6, reason=f"{label}_settle")
    for frame, names in enumerate(short_charge_plan(stutter=True)):
        if frame == 55:
            names += ("X",)
        hold(session, 1, *names, reason=f"{label}_short_charge")
    for _ in range(274):
        state = session.state
        if state.room_id == ROOM_HALFIE_CLIMB and state.game_state == 8 and not state.door_transition:
            if not state.speed_boosting:
                raise RuntimeError(f"{label}: entered Halfie without full boost")
            return
        hold(session, 1, "RIGHT", "B", reason=f"{label}_enter")
    raise TimeoutError(f"{label}: failed to enter Halfie Climb: {session.state}")


def _carry_charge_into_colosseum(session: ControllerSession) -> None:
    label = "full_halfie_r207"
    for _ in range(40):
        if session.state.samus_x >= 145:
            break
        hold(session, 1, "RIGHT", "B", reason=f"{label}_approach")
    state = hold(session, 1, "DOWN", reason=f"{label}_store")
    if not state.shinesparking:
        raise RuntimeError(f"{label}: crouch did not store a shinespark")
    for _ in range(12):
        hold(session, 1, "RIGHT", "B", reason=f"{label}_stand")
        if session.state.pose == 9 and session.state.samus_x >= 149:
            break
    hold(session, 22, "RIGHT", "B", "A", reason=f"{label}_jump")
    prev_x = -1
    fired = False
    for _ in range(260):
        state = session.state
        if state.room_id == ROOM_COLOSSEUM:
            state = wait_ordinary_room(session, ROOM_COLOSSEUM, settle_frames=320, label=label)
            if state.shinespark_timer < 12:
                raise RuntimeError(f"{label}: only {state.shinespark_timer} charge frames left in Colosseum")
            return
        if state.door_transition:
            hold(session, 1, reason=f"{label}_door_wait")
            continue
        if state.samus_y <= 155 and state.samus_x >= 165:
            names = ("RIGHT", "B") if fired else ("RIGHT", "B", "X")
            hold(session, 1, *names, reason=f"{label}_run_door")
            fired = True
            continue
        at_wall = state.samus_x == prev_x and state.samus_x >= 165
        prev_x = state.samus_x
        if state.pose == _POSE_LATCH_RIGHT or at_wall:
            if state.pose != _POSE_LATCH_RIGHT:
                hold(session, 2, "LEFT", reason=f"{label}_latch")
            hold(session, 1, "LEFT", "A", reason=f"{label}_kick")
            if state.samus_y <= 205 and not fired:
                # Shoot during the final kick so the hatch opens before landing.
                hold(session, 5, "RIGHT", "B", "A", reason=f"{label}_return")
                hold(session, 1, "RIGHT", "B", "A", "X", reason=f"{label}_shoot_door")
                hold(session, 5, "RIGHT", "B", "A", reason=f"{label}_return")
                fired = True
            else:
                hold(session, 11, "RIGHT", "B", "A", reason=f"{label}_return")
            prev_x = -1
        else:
            hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_steer")
    raise TimeoutError(f"{label}: failed to reach Colosseum: {session.state}")


def _spark_across_colosseum(session: ControllerSession) -> SuperMetroidState:
    label = "full_halfie_r208"
    # A full spark costs 127 energy and cuts out at 29. The corridor is at
    # y=102–109, between the ceiling overhangs and the tops of the pillars.
    if session.state.health < 157 or session.state.super_missiles < 1:
        raise RuntimeError(f"{label}: need 157 energy and one Super at Colosseum entry")
    hold(session, 3, "RIGHT", "B", reason=f"{label}_run")
    hold(session, 7, "RIGHT", "B", "A", reason=f"{label}_jump")
    state = hold(session, 1, "R", reason=f"{label}_unspin")
    if not (102 <= state.samus_y <= 109 and state.pose in SPARK_POSES):
        raise RuntimeError(f"{label}: missed spark corridor: {state}")
    hold(session, 1, "A", reason=f"{label}_activate")
    for _ in range(220):
        if session.state.pose not in SPARK_POSES:
            break
        hold(session, 1, "RIGHT", "A", reason=f"{label}_spark")
    if session.state.samus_x < 1740 or session.state.pose in SPARK_POSES:
        raise RuntimeError(f"{label}: spark failed to clear the corridor: {session.state}")

    # Drop left of the y=336 overhang, then hop right onto the green door ledge.
    for _ in range(220):
        s = session.state
        if s.samus_y == 395 and s.samus_x >= 1720 and s.velocity_y == 0:
            break
        if s.samus_y < 355:
            if s.samus_x > 1705:
                hold(session, 1, "LEFT", "B", reason=f"{label}_shaft_left")
            elif s.samus_x < 1690:
                hold(session, 1, "RIGHT", "B", reason=f"{label}_shaft_right")
            else:
                hold(session, 1, reason=f"{label}_shaft_fall")
        elif s.samus_y >= 402:
            hold(session, 14, "RIGHT", "B", "A", reason=f"{label}_hop_out")
        else:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_land_door")
    else:
        raise TimeoutError(f"{label}: missed the green door ledge: {session.state}")

    hold(session, 1, "SELECT", reason=f"{label}_missiles")
    hold(session, 1, reason=f"{label}_release_select")
    state = hold(session, 1, "SELECT", reason=f"{label}_supers")
    if state.selected_item != 2:
        raise RuntimeError(f"{label}: expected a Super selected, got slot {state.selected_item}")
    hold(session, 1, "RIGHT", "X", reason=f"{label}_shoot_green")
    for _ in range(150):
        if session.state.room_id == ROOM_PRECIOUS:
            return wait_ordinary_room(session, ROOM_PRECIOUS, settle_frames=320, label=label)
        hold(session, 1, "RIGHT", "B", reason=f"{label}_enter")
    raise TimeoutError(f"{label}: failed to reach Precious Room 0xD78F: {session.state}")


__all__ = [
    "play_full_halfie_chain",
    "play_halfie_climb_walljump",
    "play_precious_room_direct",
]

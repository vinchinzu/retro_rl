"""Tourian escape, ship launch, and post-launch credits."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from retro_harness.actions import idle_action, snes_action
from super_metroid.combat.mother_brain import ROOM_MOTHER_BRAIN, play_mother_brain_closeout
from super_metroid.ram import (
    ADDR_GAME_STATE,
    GameplayPhase,
    SuperMetroidState,
    credits_complete,
    read_wram_u16,
)
from super_metroid.routes.kpdr.room_ids import ROOM_CLIMB, ROOM_LANDING_SITE, ROOM_PARLOR
from super_metroid.routes.runtime import ControllerSession


ROOM_ESCAPE_1 = 0xDE4D
ROOM_ESCAPE_2 = 0xDE7A
ROOM_ESCAPE_3 = 0xDEA7
ROOM_ESCAPE_4 = 0xDEDE
# Timer states from initialization through the countdown.
ACTIVE_ESCAPE_TIMER_STATES = frozenset({3, 4, 5, 6})

ESCAPE_ROOM_IDS = (
    ROOM_ESCAPE_1,
    ROOM_ESCAPE_2,
    ROOM_ESCAPE_3,
    ROOM_ESCAPE_4,
)
ESCAPE_CHAIN_ROOM_IDS = ESCAPE_ROOM_IDS + (ROOM_LANDING_SITE,)


def _space_jump(state: SuperMetroidState, frame: int) -> bool:
    # Keep A through the rise. Re-press it once Samus is falling fast enough.
    return state.vertical_direction == 1 or (
        frame % 2 == 0 and (state.vertical_direction == 0 or state.velocity_y >= 2)
    )


def _shaft_x(y: int, waypoints: tuple[tuple[int, int], ...], top_x: int) -> int:
    return next((x for min_y, x in waypoints if y >= min_y), top_x)


CLIMB_WAYPOINTS = (
    (2120, 362), (2070, 395), (2005, 322), (1865, 373),
    # The platforms repeat up the central shaft.
    (1740, 322), (1610, 373), (1485, 322), (1360, 373),
    (1230, 322), (1105, 373), (975, 322), (850, 373),
    (720, 322), (595, 373), (460, 322), (335, 373),
    (200, 322), (108, 343),
)
PARLOR_WAYPOINTS = (
    (994, 435), (928, 402), (856, 435), (795, 406), (710, 395),
    (585, 405), (505, 430), (429, 407), (325, 373), (236, 424),
)
ESCAPE_FOUR_WAYPOINTS = ((875, 400), (806, 433), (690, 405), (498, 430), (455, 369))


def _play_exit(
    session: ControllerSession,
    room: int,
    destination: int,
    choose: Callable[[SuperMetroidState, int], tuple[str, ...]],
    *,
    timeout_frames: int,
) -> None:
    """Drive one escape room up to the next settled door entry."""
    if session.state.room_id != room or session.state.timer_type not in ACTIVE_ESCAPE_TIMER_STATES:
        raise RuntimeError(f"escape controller requires room 0x{room:04X} with the timer running")
    for elapsed in range(timeout_frames + 1):
        state = session.state
        if state.health <= 0 or state.phase is GameplayPhase.DEATH_OR_GAME_OVER:
            raise RuntimeError(f"escape failed in room 0x{state.room_id:04X} at frame {elapsed}")
        ordinary = state.phase is GameplayPhase.ORDINARY_GAMEPLAY
        if ordinary and state.room_id == destination:
            return
        if ordinary and state.room_id != room:
            raise RuntimeError(f"escape reached unexpected room 0x{state.room_id:04X}")
        if elapsed == timeout_frames:
            break
        names = choose(state, elapsed) if ordinary else ()
        session.step(snes_action(*names, dtype=np.int8), "tourian_escape")
    raise TimeoutError(
        f"escape from 0x{room:04X} timed out at ({state.samus_x}, {state.samus_y})"
    )


def _escape_one_buttons(state: SuperMetroidState, frame: int) -> tuple[str, ...]:
    x, y = state.samus_x, state.samus_y
    jump = ("A",) if _space_jump(state, frame) else ()
    fire = ("X",) if frame % 2 == 0 else ()
    if x > 220:
        return ("LEFT", "B") + fire + (jump if x >= 440 else ())
    if x > 145 and y < 158:
        return ("LEFT",) + (jump if x < 200 else ())
    # Aim down through the shot block, then center on the floor hatch.
    target = 108 if y < 204 else 119
    steer = ("RIGHT",) if x < target - 2 else ("LEFT",) if x > target + 2 else ()
    return steer + (jump if state.vertical_direction == 0 else ("DOWN",)) + fire


def play_tourian_escape_one(session: ControllerSession) -> None:
    _play_exit(session, ROOM_ESCAPE_1, ROOM_ESCAPE_2, _escape_one_buttons, timeout_frames=1000)


def _escape_two_buttons(state: SuperMetroidState, frame: int) -> tuple[str, ...]:
    x, y = state.samus_x, state.samus_y
    fire = ("X",) if frame % 2 == 0 else ()
    if y >= 320:
        # Jump back to hatch height if the initial drop passed below it.
        jump = _space_jump(state, frame) and (y >= 383 or state.vertical_direction == 1)
        return ("RIGHT",) + fire + (("A",) if jump else ())
    if x < 211:
        return ("RIGHT",)
    if x > 218:
        return ("LEFT",)
    return ("DOWN",) + fire


def play_tourian_escape_two(session: ControllerSession) -> None:
    _play_exit(session, ROOM_ESCAPE_2, ROOM_ESCAPE_3, _escape_two_buttons, timeout_frames=900)


def play_tourian_escape_three(session: ControllerSession) -> None:
    section = "entrance"

    def choose(state: SuperMetroidState, frame: int) -> tuple[str, ...]:
        nonlocal section
        x, y = state.samus_x, state.samus_y
        jump = ("A",) if _space_jump(state, frame) else ()
        fire = ("X",) if frame % 2 == 0 else ()
        if section == "entrance" and y < 350 and x > 75:
            section = "lower corridor"
        if section == "lower corridor" and x > 1345:
            section = "right shaft"
        if section == "right shaft" and y < 280 and x < 1330:
            section = "middle corridor"
        if section == "middle corridor" and x < 548:
            section = "upper shaft"
        if section == "upper shaft" and y < 132 and x >= 530:
            section = "exit corridor"

        # The lower and middle corridors connect on the right. The opening
        # from the middle corridor to the exit is back on the left.
        if section == "entrance":
            steer = ("RIGHT",) if x < 88 else ("LEFT",) if x > 95 else ()
            return steer + (jump if y >= 399 or state.vertical_direction == 1 else ())
        if section == "lower corridor":
            return ("RIGHT", "B") + fire
        if section == "right shaft":
            target = 1300 if y < 290 else 1370
            return (("RIGHT",) if x <= target else ("LEFT",)) + jump
        if section == "middle corridor":
            return ("LEFT", "B") if x > 605 else ("LEFT",)
        if section == "upper shaft":
            return (("RIGHT",) if x <= 543 else ("LEFT",)) + jump
        return (
            ("RIGHT", "B") + (fire if x >= 580 else ())
            + (jump if x < 630 or 864 <= x < 1000 else ())
        )

    _play_exit(session, ROOM_ESCAPE_3, ROOM_ESCAPE_4, choose, timeout_frames=1750)


def play_tourian_escape_four(session: ControllerSession) -> None:
    section = "west approach"

    def choose(state: SuperMetroidState, frame: int) -> tuple[str, ...]:
        nonlocal section
        x, y = state.samus_x, state.samus_y
        jump = ("A",) if _space_jump(state, frame) else ()
        if section == "west approach" and x >= 126:
            section = "west shaft"
        if section == "west shaft" and y < 130:
            section = "upper ledge"
        if section == "upper ledge" and y > 208 and x > 550:
            section = "shaft entrance"
        if section == "shaft entrance" and y > 300:
            section = "drop"
        if section == "drop" and y > 1360:
            section = "shaft exit"
        if section == "shaft exit" and x > 385:
            section = "east shaft"
        if section == "east shaft" and y < 395 and state.vertical_direction == 1:
            section = "exit ledge"
        if section == "exit ledge" and y > 420:
            section = "east shaft"

        # The narrow center shaft connects the upper ledge to the base of
        # the east shaft. Head right only once below the dividing wall.
        if section == "west approach":
            # Walk under the entrance overhang before jumping up the shaft.
            return ("RIGHT",)
        if section == "west shaft":
            return (("RIGHT",) if x <= 157 else ("LEFT",)) + jump
        if section == "upper ledge":
            return ("RIGHT", "B") + (jump if 174 <= x < 270 else ()) if x < 617 else ()
        if section == "shaft entrance":
            if x > 278:
                return ("LEFT", "B")
            return ("LEFT",) if x > 227 else ("RIGHT",) if x < 216 else ()
        if section == "drop":
            return ("LEFT",) if x > 226 else ("RIGHT",) if x < 218 else ()
        if section == "shaft exit":
            return ("RIGHT",)
        if section == "east shaft":
            target = _shaft_x(y, ESCAPE_FOUR_WAYPOINTS, 310)
            return (("RIGHT",) if x <= target else ("LEFT",)) + jump
        fire = ("X",) if frame % 2 == 0 else ()
        return ("RIGHT", "B") + (jump if x < 345 else fire)

    _play_exit(session, ROOM_ESCAPE_4, ROOM_CLIMB, choose, timeout_frames=3400)


def _climb_buttons(state: SuperMetroidState, frame: int) -> tuple[str, ...]:
    x, y = state.samus_x, state.samus_y
    jump = ("A",) if _space_jump(state, frame) else ()
    if x < 260 and y > 2130:
        # Dash across the flat bottom corridor, then walk into the first jump
        # so momentum does not carry Samus into the right wall below the lip.
        return ("RIGHT", "B")
    if x < 305 and y > 2130:
        return ("RIGHT",)
    if y < 104:
        # Open the ceiling hatch with Hyper and go straight up through it.
        steer = ("RIGHT",) if x < 365 else ("LEFT",) if x > 369 else ()
        shot = ("X",) if frame % 2 == 0 else ()
        return ("UP",) + steer + shot + jump
    target = _shaft_x(y, CLIMB_WAYPOINTS, 367)
    return (("RIGHT",) if x <= target else ("LEFT",)) + jump


def play_escape_climb(session: ControllerSession) -> None:
    _play_exit(session, ROOM_CLIMB, ROOM_PARLOR, _climb_buttons, timeout_frames=2200)


def play_escape_parlor(session: ControllerSession) -> None:
    across = False

    def choose(state: SuperMetroidState, frame: int) -> tuple[str, ...]:
        nonlocal across
        x, y = state.samus_x, state.samus_y
        across |= y < 164
        if not across and y <= 176 and state.vertical_direction == 1:
            return ("RIGHT", "A")
        if not across and 260 < x < 475:
            target = _shaft_x(y, PARLOR_WAYPOINTS, 382)
            jump = ("A",) if _space_jump(state, frame) else ()
            return (("RIGHT",) if x <= target else ("LEFT",)) + jump
        # Sprint across the upper corridor and jump at the step before the exit.
        jump = ("A",) if 657 <= x < 850 else ()
        return ("B", "RIGHT", "X") + jump

    _play_exit(session, ROOM_PARLOR, ROOM_LANDING_SITE, choose, timeout_frames=2200)


def _ship_approach(
    state: SuperMetroidState, *, jumped: bool, down_held: bool
) -> tuple[str, ...]:
    x, y = state.samus_x, state.samus_y
    vertical = state.vertical_direction

    # Run from the west door and jump off the last stretch of low ground.
    if y >= 1160 and x < 738:
        return ("B", "RIGHT")
    if (y >= 1160 and not jumped) or (jumped and y > 975 and vertical == 1 and x < 1040):
        return ("B", "RIGHT", "A")

    # Coast toward the ship and brake in the air before landing on its deck.
    if x < 1000:
        return ("RIGHT",)
    if vertical:
        return ("LEFT",) if 1103 <= x <= 1200 and state.pose in (0x81, 0x19) else ()

    # Walk in short pulses so a DOWN press lands inside the boarding window.
    if 1100 <= x <= 1200 and y in (1091, 1096):
        if state.pose in (39, 40):
            return ("UP",)
        if state.velocity_x_sub >= 14000 or state.momentum_x:
            return ()
        if abs(x - 1152) <= 2 and y == 1091 and not state.velocity_x and not state.velocity_x_sub:
            return () if down_held else ("DOWN",)
    if x < 1150:
        return ("RIGHT",)
    if x > 1154:
        return ("LEFT",)
    return ()


def play_ship_launch(session: ControllerSession, *, timeout_frames: int = 1800) -> None:
    """Board from the escape-side Landing Site door and wait for the ending."""
    state = session.state
    if state.room_id != ROOM_LANDING_SITE or state.timer_type != 6:
        raise RuntimeError("ship launch requires Landing Site with the Zebes escape timer running")
    jumped = down_held = False
    for elapsed in range(timeout_frames + 1):
        state = session.state
        if state.room_id != ROOM_LANDING_SITE or state.health <= 0 or state.phase is GameplayPhase.DEATH_OR_GAME_OVER:
            raise RuntimeError(f"ship approach failed at frame {elapsed}: {state}")
        if state.phase is GameplayPhase.ENDING_OR_CREDITS:
            return
        if elapsed == timeout_frames:
            break
        # The ship takes control as soon as the boarding pose is set.
        names = ()
        if state.phase is GameplayPhase.ORDINARY_GAMEPLAY and state.pose not in (0x9A, 0x9B):
            names = _ship_approach(state, jumped=jumped, down_held=down_held)
            if state.samus_y >= 1160:
                jumped = "A" in names or (jumped and state.samus_x >= 738)
        down_held = "DOWN" in names
        session.step(snes_action(*names, dtype=np.int8), "escape_ship_launch")
    raise TimeoutError(f"ship launch did not reach the ending within {timeout_frames} frames")


ESCAPE_ROW_CONTROLLERS = {
    ("0xDD58", "0xDE4D", None): play_mother_brain_closeout,
    ("0xDE4D", "0xDE7A", None): play_tourian_escape_one,
    ("0xDE7A", "0xDEA7", None): play_tourian_escape_two,
    ("0xDEA7", "0xDEDE", None): play_tourian_escape_three,
    ("0xDEDE", "0x96BA", None): play_tourian_escape_four,
    ("0x96BA", "0x92FD", None): play_escape_climb,
    ("0x92FD", "0x91F8", None): play_escape_parlor,
    ("0x91F8", None, "ending_or_credits"): play_ship_launch,
}


def finish_controller_for(
    state: SuperMetroidState, room: str, destination: str | None, goal_phase: str | None
) -> Callable[[ControllerSession], None] | None:
    """Select the late-game controller for a live escape or Mother Brain leave."""
    if state.room_id == ROOM_MOTHER_BRAIN or state.timer_type in ACTIVE_ESCAPE_TIMER_STATES:
        return ESCAPE_ROW_CONTROLLERS.get((room, destination, goal_phase))
    return None


def play_credits_tail(env: Any, *, timeout_frames: int = 24_000) -> np.ndarray:
    """Play neutral inputs from the ending until the final screen is displayed."""
    idle = idle_action(dtype=np.int8)
    for elapsed in range(timeout_frames + 1):
        if credits_complete(env):
            return np.repeat(idle[np.newaxis, :], elapsed, axis=0)
        if read_wram_u16(env, ADDR_GAME_STATE) not in (38, 39):
            raise RuntimeError("game left the ending before the final credits screen")
        if elapsed < timeout_frames:
            env.step(idle)
    raise RuntimeError(f"final credits screen did not appear within {timeout_frames} frames")

# These are planning bounds only. They do not drive emulator frames yet.
DEFAULT_ESCAPE_TIMEOUTS: dict[int, int] = {
    ROOM_ESCAPE_1: 1_800,
    ROOM_ESCAPE_2: 1_800,
    ROOM_ESCAPE_3: 1_800,
    ROOM_ESCAPE_4: 1_800,
    ROOM_LANDING_SITE: 2_400,
}


def _room_evidence(room_id: int, timeout_frames: int) -> dict[str, object]:
    """Describe one deferred room controller without touching a session."""
    return {
        "roomId": room_id,
        "roomIdHex": f"0x{room_id:04X}",
        "timeoutFrames": timeout_frames,
        "status": "stub",
        "success": False,
        "implemented": False,
    }


def play_escape_chain_scaffold(
    session: Any,
    *,
    room_timeouts: Mapping[int, int] | None = None,
) -> dict[str, object]:
    """Return bounded escape-chain evidence for future controller work.

    ``session`` is accepted to match the eventual segment surface, but is only
    inspected for an optional current state. This stub never steps, warps,
    places Samus, or writes RAM. The returned evidence is not a clear.
    """
    timeouts = dict(DEFAULT_ESCAPE_TIMEOUTS)
    if room_timeouts is not None:
        timeouts.update(room_timeouts)
    invalid = {
        room_id: timeout
        for room_id, timeout in timeouts.items()
        if room_id in ESCAPE_CHAIN_ROOM_IDS and timeout <= 0
    }
    if invalid:
        raise ValueError(f"escape room timeouts must be positive: {invalid}")

    state = getattr(session, "state", None)
    current_room = getattr(state, "room_id", None)
    return {
        "developmentOnly": True,
        "outcome": "scaffold_only",
        "success": False,
        "currentRoomId": current_room,
        "rooms": [
            _room_evidence(room_id, timeouts[room_id])
            for room_id in ESCAPE_CHAIN_ROOM_IDS
        ],
    }


def play_escape_room_1(session: Any, *, timeout_frames: int = 1_800) -> dict[str, object]:
    """Deferred bounded stub for Tourian escape room 1."""
    return _play_escape_room_stub(session, ROOM_ESCAPE_1, timeout_frames)


def _play_escape_room_stub(
    session: Any, room_id: int, timeout_frames: int
) -> dict[str, object]:
    if timeout_frames <= 0:
        raise ValueError("escape room timeout must be positive")
    state = getattr(session, "state", None)
    return {
        **_room_evidence(room_id, timeout_frames),
        "currentRoomId": getattr(state, "room_id", None),
    }

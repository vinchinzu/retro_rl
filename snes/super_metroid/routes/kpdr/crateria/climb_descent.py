"""Crateria Climb (0x96BA), ascent to Parlor and descent toward Morph.

Public policy: https://wiki.supermetroid.run/Climb and
https://wiki.supermetroid.run/Moonwalk (Climb is the biggest moonfall save,
~7.40s regular fall → ~3.45s).

Enter from Parlor's bottom-left vertical door (top of Climb). Land on the
start ledge, face right, moonwalk left to the lip (~x=349), spinning
moonfall, hold LEFT down the shaft (skips the pirate floater at ~395,107),
aim-down to clip the bottom platform, then run RIGHT into the Pit door
(~x=493, y=2187).

Moonwalk is a file option (``$09E4``). This hop pokes it **on** at entry and
**off** after Pit settle so later hash-pinned seeds (pit / elev / morph)
keep their moonwalk-off inputs.

Assisted product still plays the Climb seed. Flip
:data:`CLIMB_MOONFALL_ON_CLEAN` only after the pin dual is green and faster.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from retro_harness.actions import buttons, idle_action
from super_metroid.ram import FACING_RIGHT, parse_state, set_moonwalk
from super_metroid.routes.kpdr.room_ids import ROOM_CLIMB, ROOM_PARLOR, ROOM_PIT
from super_metroid.routes.runtime import ControllerSession, hold
from super_metroid.routes.skills.knockback import is_knockback
from super_metroid.routes.skills.moonfall import (
    WIKI_URL,
    initiate_moonfall,
    is_airborne,
    is_moonfalling,
    is_moonwalking,
    require_moonwalk_on,
)
from super_metroid.takeoff import walk_toward_x

Phase = Literal[
    "plant",
    "face",
    "moonwalk",
    "jump",
    "fall",
    "bottom",
    "exit",
    "done",
]

# Climb 3×9; start ledge y=91 x=348–372. Pirate floater ~(395,107).
# Left-lip moonfall holds LEFT down the shaft. Pit door is RIGHT at
# floor ~(493, 2187) — seed exit, not the map-left node name.
FALL_X = 300
LIP_X = 349
TOP_Y_MAX = 120
AIM_Y = 1600
BOTTOM_Y = 2100
DOOR_X = 490
PIT_SETTLE = 180
JUMP_HOLD = 3
SPIN_HOLD = 4

# Default off: seed remains the assisted/clean product hop until the probe
# dual-greens from a natural Climb enter pin.
CLIMB_MOONFALL_ON_CLEAN = False


@dataclass(frozen=True)
class ClimbLedge:
    x: int
    y: int
    step: int = 0
    launch: Literal["toward", "away", "vertical"] = "toward"
    steer_after: int = 6
    shoot: bool = False


# Platforms along the Pit-to-Parlor ascent in room coordinates.
_ASCENT_LEDGES = (
    ClimbLedge(420, 2131, -6, "away", shoot=True),
    ClimbLedge(378, 2011, 3),
    ClimbLedge(350, 1915, launch="away"),
    ClimbLedge(330, 1843),
    ClimbLedge(349, 1755, launch="away"),
    ClimbLedge(343, 1659, 3, "away", shoot=True),
    ClimbLedge(320, 1603),
    ClimbLedge(357, 1499),
    ClimbLedge(332, 1403, launch="away"),
    ClimbLedge(319, 1347, launch="away"),
    ClimbLedge(360, 1243),
    ClimbLedge(346, 1147, 6, "away", 12),
    ClimbLedge(323, 1091),
    ClimbLedge(353, 987, launch="vertical", steer_after=36),
    ClimbLedge(341, 891, launch="away", steer_after=12),
    ClimbLedge(316, 835, 3, "away", shoot=True),
    ClimbLedge(361, 731),
    ClimbLedge(341, 635, launch="away"),
    ClimbLedge(322, 579),
    ClimbLedge(361, 475, launch="away"),
    ClimbLedge(353, 379, launch="away"),
    ClimbLedge(375, 267, launch="away"),
    ClimbLedge(351, 211, 3, "away", 12),
    ClimbLedge(322, 139),
    ClimbLedge(366, 91),
)


def _jump_to_ledge(session: ControllerSession, ledge: ClimbLedge) -> None:
    if ledge.step:
        side = "LEFT" if ledge.step < 0 else "RIGHT"
        hold(session, abs(ledge.step), side, reason="climb_takeoff_seat")
    hold(session, 1, reason="climb_jump_release")
    toward = "RIGHT" if ledge.x >= session.state.samus_x else "LEFT"
    away = "LEFT" if toward == "RIGHT" else "RIGHT"
    airborne = False
    for tick in range(115):
        state = session.state
        if airborne and state.vertical_direction == 0 and not state.velocity_y:
            if abs(state.samus_y - ledge.y) <= 3 and abs(state.samus_x - ledge.x) <= 22:
                return
            break
        airborne |= bool(state.vertical_direction)
        names = ("A",)
        if tick >= ledge.steer_after:
            names += walk_toward_x(state.samus_x, ledge.x, slack=2)
        elif ledge.launch != "vertical":
            names += (away if ledge.launch == "away" else toward,)
        if (ledge.launch == "away" and tick < 3) or (ledge.launch == "toward" and tick == 0):
            names += ("B",)
        if ledge.shoot and tick >= 10 and tick % 6 == 0:
            names += ("X",)
        hold(session, 1, *names, reason="climb_ledge")
    raise TimeoutError(f"Climb missed the ledge at y={ledge.y}: {session.state}")


def play_climb_to_parlor(session: ControllerSession) -> None:
    """Climb from Pit using the live platform landings and beam-opened hatch."""
    state = session.state
    if state.room_id != ROOM_CLIMB or state.samus_y < 2110:
        raise RuntimeError("Climb ascent requires the lower Pit entrance")
    if state.selected_item:
        hold(session, 1, "Y", reason="climb_select_beam")
    for _ in range(55):
        state = session.state
        if not state.vertical_direction and state.samus_x <= 418:
            break
        names = ("LEFT", "B") if state.samus_x > 418 else ()
        hold(session, 1, *names, reason="climb_pit_run")
    for ledge in _ASCENT_LEDGES:
        if abs(session.state.samus_y - ledge.y) > 4:
            _jump_to_ledge(session, ledge)

    for tick in range(330):
        state = session.state
        if state.room_id == ROOM_PARLOR and state.game_state == 8 and not state.door_transition:
            return
        names = ()
        if state.room_id == ROOM_CLIMB and state.game_state == 8:
            names = ("UP",) + walk_toward_x(state.samus_x, 368, slack=2)
            if tick % 2 == 0:
                names += ("X",)
            if tick % 65 >= 10:
                names += ("A",)
        hold(session, 1, *names, reason="climb_ceiling_hatch")
    raise TimeoutError(f"Climb missed Parlor: {session.state}")


@dataclass(frozen=True)
class ClimbMoonfallTrack:
    phase: Phase = "plant"
    held: int = 0


def climb_moonfall_enabled(session: ControllerSession) -> bool:
    """Whether this session should replace the Climb seed with moonfall."""
    override = getattr(session, "climb_moonfall", None)
    if override is not None:
        return bool(override)
    if not CLIMB_MOONFALL_ON_CLEAN:
        return False
    assist = getattr(session, "assist", None)
    if assist is None:
        return False
    # Clean morph: ammo assist is off (energy already off on morph).
    return not bool(getattr(assist, "enabled", True))


def climb_moonfall_action(
    state,
    track: ClimbMoonfallTrack,
) -> tuple[tuple[str, ...], ClimbMoonfallTrack]:
    """One-frame Climb moonfall policy (ROM-free).

    Live probe (warp pin): left-lip moonfall, LEFT down the shaft, RIGHT
    along the floor into Pit. First floater + pirate sit at ~(395,107);
    jumping right from the start ledge lands on them.
    """
    x = int(state.samus_x)
    y = int(state.samus_y)
    room = int(state.room_id)
    phase = track.phase
    held = track.held
    grounded = not is_airborne(state)

    if room == ROOM_PIT:
        return (), replace(track, phase="done", held=0)
    if room != ROOM_CLIMB and phase != "exit":
        return ("RIGHT", "X"), replace(track, phase="exit", held=0)

    if is_knockback(state) and phase not in ("exit", "done", "bottom"):
        if held > 24:
            return ("LEFT",), replace(track, phase="fall" if y > 200 else "plant", held=0)
        return (), replace(track, held=held + 1)

    if phase == "plant":
        if y > 400:
            return ("LEFT",), replace(track, phase="fall", held=0)
        if is_airborne(state):
            # No d-pad during drop-in (RIGHT walks onto the pirate floater).
            return ("X", "L"), replace(track, held=held + 1)
        return ("RIGHT",), replace(track, phase="face", held=0)

    if phase == "face":
        if grounded and int(state.facing) == FACING_RIGHT:
            held += 1
            if held >= 2:
                return ("LEFT", "X", "L"), replace(track, phase="moonwalk", held=0)
            return (), replace(track, held=held)
        return ("RIGHT",), replace(track, held=0)

    if phase == "moonwalk":
        if is_airborne(state):
            return ("LEFT", "A"), replace(track, phase="fall", held=0)
        moonwalking = is_moonwalking(state)
        if moonwalking and x <= LIP_X:
            return ("LEFT", "X", "L", "A"), replace(track, phase="jump", held=0)
        if held > 40:
            return ("LEFT", "X", "L", "A"), replace(track, phase="jump", held=0)
        return ("LEFT", "X", "L"), replace(track, held=held + 1)

    if phase == "jump":
        held += 1
        if is_moonfalling(state) and held >= JUMP_HOLD:
            return ("LEFT", "A"), replace(track, phase="fall", held=0)
        if held <= JUMP_HOLD:
            return ("LEFT", "X", "L", "A"), replace(track, held=held)
        if held <= JUMP_HOLD + SPIN_HOLD:
            return ("LEFT", "A"), replace(track, held=held)
        return ("LEFT",), replace(track, phase="fall", held=0)

    if phase == "fall":
        floor = grounded and y >= BOTTOM_Y - 50
        if floor or y >= BOTTOM_Y:
            return ("A",) if y > 2190 else ("RIGHT", "X"), replace(
                track, phase="bottom", held=0
            )
        if grounded and y < BOTTOM_Y - 50:
            return ("LEFT", "X", "L", "A"), replace(track, held=held + 1)
        if y >= AIM_Y:
            return ("LEFT", "L", "X"), replace(track, held=held + 1)
        return ("LEFT",), replace(track, held=held + 1)

    if phase == "bottom":
        if y > 2192 and held < 8:
            return ("A",), replace(track, held=held + 1)
        if x >= DOOR_X:
            return ("RIGHT", "X"), replace(track, phase="exit", held=0)
        return ("RIGHT", "X"), replace(track, held=held + 1)

    if phase == "exit":
        if room == ROOM_PIT:
            return (), replace(track, phase="done", held=0)
        return ("RIGHT", "X"), replace(track, held=held + 1)

    return (), track


def play_climb_to_pit_moonfall(
    session: ControllerSession,
    *,
    max_frames: int = 1200,
    restore_moonwalk: bool = True,
    use_menu_setting: bool = False,
) -> None:
    """Drop from Climb to Pit using spinning Moonfall."""
    env = None
    if not use_menu_setting:
        env = getattr(session, "env", None)
        if env is None:
            raise RuntimeError("climb moonfall needs session.env for $09E4 poke")
        set_moonwalk(env, True)
        session.state = parse_state(env.get_ram(), frame=session.frame)
    require_moonwalk_on(session.state, label="climb_moonfall")
    if session.state.room_id != ROOM_CLIMB and session.state.game_state not in (8, 11):
        raise RuntimeError(
            f"climb moonfall: expected Climb 0x{ROOM_CLIMB:04X}, got {session.state}"
        )

    track = ClimbMoonfallTrack()
    for tick in range(max_frames):
        state = session.state
        if use_menu_setting and state.room_id == ROOM_CLIMB and state.samus_y >= 1640:
            # Start steering toward the Pit hatch during the fall to carry
            # rightward momentum into the run across the floor.
            names = ("RIGHT", "B")
            if state.samus_y >= 2174 and tick % 4 == 0:
                names += ("X",)
        else:
            names, track = climb_moonfall_action(state, track)
        action = buttons(*names) if names else idle_action()
        session.step(action, f"climb_moonfall_{track.phase}")
        if track.phase == "done" or session.state.room_id == ROOM_PIT:
            break
    else:
        raise TimeoutError(
            f"climb moonfall missed Pit after {max_frames}f: {session.state} "
            f"phase={track.phase} ({WIKI_URL})"
        )

    if session.state.room_id != ROOM_PIT or session.state.game_state == 11:
        for _ in range(PIT_SETTLE):
            state = session.step(idle_action(), "climb_moonfall_pit_settle")
            if state.room_id == ROOM_PIT and state.game_state == 8:
                break
        else:
            raise TimeoutError(f"moonfall leave did not settle: {session.state}")
    if env is not None and restore_moonwalk:
        set_moonwalk(env, False)


def play_climb_to_pit_moonfall_clean(session: ControllerSession) -> None:
    """Run the descent with Moon Walk enabled through the menu."""
    play_climb_to_pit_moonfall(session, use_menu_setting=True)


# Optional convenience: skill initiate used by the probe "setup" dump.
def setup_then_fall(session: ControllerSession) -> None:
    """In-room initiate_moonfall then idle-steer (practice / dump)."""
    env = getattr(session, "env", None)
    if env is not None:
        set_moonwalk(env, True)
    initiate_moonfall(session, reason="climb_setup")


__all__ = [
    "CLIMB_MOONFALL_ON_CLEAN",
    "ClimbMoonfallTrack",
    "BOTTOM_Y",
    "DOOR_X",
    "FALL_X",
    "LIP_X",
    "climb_moonfall_action",
    "climb_moonfall_enabled",
    "play_climb_to_parlor",
    "play_climb_to_pit_moonfall",
    "setup_then_fall",
]

"""Climb (0x96BA) first descent toward Morph — moonfall policy.

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
from super_metroid.routes.kpdr.room_ids import ROOM_CLIMB, ROOM_PIT
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import is_knockback
from super_metroid.routes.skills.moonfall import (
    WIKI_URL,
    initiate_moonfall,
    is_airborne,
    is_moonfalling,
    is_moonwalking,
    require_moonwalk_on,
)

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
) -> None:
    """RAM-driven Climb → Pit using spinning moonfall. Pokes $09E4 on."""
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
    for _ in range(max_frames):
        names, track = climb_moonfall_action(session.state, track)
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
        session.wait_until(  # type: ignore[attr-defined]
            lambda s: s.room_id == ROOM_PIT and s.game_state == 8,
            timeout=PIT_SETTLE,
            reason="climb_moonfall_pit_settle",
        )
    if restore_moonwalk:
        set_moonwalk(env, False)


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
    "play_climb_to_pit_moonfall",
    "setup_then_fall",
]

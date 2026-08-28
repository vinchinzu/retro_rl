"""Climb (0x96BA) first descent toward Morph — moonfall policy.

Public policy: https://wiki.supermetroid.run/Climb and
https://wiki.supermetroid.run/Moonwalk (Climb is the biggest moonfall save,
~7.40s regular fall → ~3.45s).

Enter from Parlor's bottom-left vertical door (top of Climb). Face left,
moonwalk right, jump, release shot (spinning moonfall), fall through the
shaft, open the bottom door to Pit.

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
from super_metroid.ram import FACING_LEFT, parse_state, set_moonwalk
from super_metroid.routes.kpdr.room_ids import ROOM_CLIMB, ROOM_PIT
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import is_knockback
from super_metroid.routes.skills.moonfall import (
    WIKI_URL,
    initiate_moonfall,
    is_airborne,
    is_moonfalling,
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

# Climb 3×9; enter top ~(393,41), fall x~475, floor ~(493,2187).
FALL_X = 510
FALL_X_LO = 460
FALL_X_HI = 560
TOP_Y_MAX = 280
BOTTOM_Y = 2050
DOOR_X = 48
PIT_SETTLE = 180

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


def _steer_fall_x(x: int) -> tuple[str, ...]:
    if x < FALL_X_LO:
        return ("RIGHT",)
    if x > FALL_X_HI:
        return ("LEFT",)
    return ()


def climb_moonfall_action(
    state,
    track: ClimbMoonfallTrack,
) -> tuple[tuple[str, ...], ClimbMoonfallTrack]:
    """One-frame Climb moonfall policy (ROM-free)."""
    x = int(state.samus_x)
    y = int(state.samus_y)
    room = int(state.room_id)
    phase = track.phase
    held = track.held

    if room == ROOM_PIT:
        return (), replace(track, phase="done", held=0)
    if room != ROOM_CLIMB and phase != "exit":
        return (), replace(track, phase="exit", held=0)

    if is_knockback(state) and phase not in ("exit", "done"):
        if held > 24:
            return (), replace(track, phase="plant", held=0)
        return (), replace(track, held=held + 1)

    if phase == "plant":
        if y > TOP_Y_MAX + 80:
            return _steer_fall_x(x), replace(track, phase="fall", held=0)
        if is_airborne(state):
            # Wiki: buffer Angle+Shot+Jump with NO d-pad during the drop-in,
            # then hold right after landing and release shot.
            return ("X", "L", "A"), replace(track, held=held + 1)
        if int(state.facing) != FACING_LEFT:
            return ("LEFT",), replace(track, phase="face", held=0)
        return ("RIGHT", "X", "L"), replace(track, phase="moonwalk", held=0)

    if phase == "face":
        if int(state.facing) == FACING_LEFT:
            return ("RIGHT", "X", "L"), replace(track, phase="moonwalk", held=0)
        return ("LEFT",), replace(track, held=held + 1)

    if phase == "moonwalk":
        # Jump at the right lip (~372) so the first floater at ~x=390 is skipped.
        if is_airborne(state) or x >= 362 or held >= 8:
            return ("RIGHT", "X", "L", "A"), replace(track, phase="jump", held=0)
        return ("RIGHT", "X", "L"), replace(track, held=held + 1)

    if phase == "jump":
        if is_moonfalling(state) or (
            is_airborne(state) and int(state.vertical_direction) == 0
        ):
            return ("RIGHT", "A"), replace(track, phase="fall", held=0)
        if is_airborne(state) and y > TOP_Y_MAX:
            return _steer_fall_x(x), replace(track, phase="fall", held=0)
        if held >= 20:
            return ("X", "L", "A"), replace(track, phase="plant", held=0)
        if held < 3:
            return ("RIGHT", "X", "L", "A"), replace(track, held=held + 1)
        # Release shot (keep angle briefly, then spin).
        return ("RIGHT", "A"), replace(track, held=held + 1)

    if phase == "fall":
        if y < TOP_Y_MAX and held > 45 and not is_moonfalling(state):
            return ("RIGHT", "X", "L", "A"), replace(track, phase="plant", held=0)
        grounded_bottom = (not is_airborne(state)) and y >= BOTTOM_Y
        if grounded_bottom or y >= BOTTOM_Y + 80:
            return ("LEFT",), replace(track, phase="bottom", held=0)
        if not is_airborne(state) and y < BOTTOM_Y:
            # Mid-shaft plant after a cancelled moonfall. Walk to the fall column.
            if x < FALL_X - 10:
                return ("RIGHT",), replace(track, held=held + 1)
            if x > FALL_X + 10:
                return ("LEFT",), replace(track, held=held + 1)
            return ("RIGHT",), replace(track, held=held + 1)
        if y < 500:
            return ("LEFT",), replace(track, held=held + 1)
        return _steer_fall_x(x), replace(track, held=held + 1)

    if phase == "bottom":
        if x <= DOOR_X + 24:
            return ("LEFT", "X"), replace(track, phase="exit", held=0)
        return ("LEFT",), replace(track, held=held + 1)

    if phase == "exit":
        if room == ROOM_PIT:
            return (), replace(track, phase="done", held=0)
        return ("LEFT", "X") if held % 18 < 6 else ("LEFT",), replace(
            track, held=held + 1
        )

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
    "climb_moonfall_action",
    "climb_moonfall_enabled",
    "play_climb_to_pit_moonfall",
    "setup_then_fall",
]

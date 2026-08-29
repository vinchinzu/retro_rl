"""Parlor (0x92FD) first descent toward Climb — moonfall policy.

Public policy: https://wiki.supermetroid.run/Parlor_and_Alcatraz and
https://wiki.supermetroid.run/Moonwalk (Parlor fall ~8.10s regular →
~7.50s moonfall; listed save 0.20s). Then a double set of downbacks into
the floor Climb door.

Enter from Landing Site (Parlor node 4, top-right). Dash left across the
top corridor (jump at x≈1127 clears the first Geemer), spinning moonfall
at the left-shaft lip (x≤360), LEFT off the grass platforms, LEFT+X+L
into the floor Climb door (node 7) ~(393, 1248).

Moonwalk is a file option (``$09E4``). This hop pokes it **on** at entry
and **off** after Climb settle so the Climb seed keeps moonwalk-off inputs.

Assisted product still plays the Parlor seed. Flip
:data:`PARLOR_MOONFALL_ON_CLEAN` only after the pin dual is green and faster,
and Climb still clears from the new seat.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from retro_harness.actions import buttons, idle_action
from super_metroid.ram import FACING_RIGHT, parse_state, set_moonwalk
from super_metroid.routes.kpdr.room_ids import ROOM_CLIMB, ROOM_PARLOR
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
    "run",
    "face",
    "moonwalk",
    "jump",
    "fall",
    "downback",
    "exit",
    "done",
]

# Parlor 5×5. Top corridor y<220. Left shaft map col 1 (x 256–512).
# Seed: dash LEFT+B from the landing door, jump at x≈1127 to clear the
# first Geemer ledge, then moonfall from the left-shaft lip ~x=360.
LEDGE_X = 390
LIP_X = 365
SHAFT_LIP_X = 360
GEEMER_JUMP_X = 1130
SHAFT_X = 425
TOP_Y_MAX = 220
AIM_Y = 1300
BOTTOM_Y = 1180
DOOR_X = 393
DOOR_X_LO = 360
DOOR_X_HI = 430
CLIMB_SETTLE = 180
JUMP_HOLD = 3
SPIN_HOLD = 4

# Default off: seed remains the assisted/clean product hop until the probe
# dual-greens from a Parlor enter pin and Climb still leaves from the seat.
PARLOR_MOONFALL_ON_CLEAN = False


@dataclass(frozen=True)
class ParlorMoonfallTrack:
    phase: Phase = "plant"
    held: int = 0


def parlor_moonfall_enabled(session: ControllerSession) -> bool:
    """Whether this session should replace the Parlor seed with moonfall."""
    override = getattr(session, "parlor_moonfall", None)
    if override is not None:
        return bool(override)
    if not PARLOR_MOONFALL_ON_CLEAN:
        return False
    assist = getattr(session, "assist", None)
    if assist is None:
        return False
    return not bool(getattr(assist, "enabled", True))


def _steer_x(x: int, target: int) -> tuple[str, ...]:
    if x < target - 12:
        return ("RIGHT",)
    if x > target + 12:
        return ("LEFT",)
    return ()


def parlor_moonfall_action(
    state,
    track: ParlorMoonfallTrack,
) -> tuple[tuple[str, ...], ParlorMoonfallTrack]:
    """One-frame Parlor moonfall policy (ROM-free).

    Live probe: run left on the top, left-lip moonfall down the Alcatraz
    shaft, downback into the floor Climb door. First-descent planet is
    typically not awake (no Ripper/Geemer).
    """
    x = int(state.samus_x)
    y = int(state.samus_y)
    room = int(state.room_id)
    phase = track.phase
    held = track.held
    grounded = not is_airborne(state)

    if room == ROOM_CLIMB:
        return (), replace(track, phase="done", held=0)
    if room != ROOM_PARLOR and phase != "exit":
        return ("LEFT",), replace(track, phase="exit", held=0)

    if is_knockback(state) and phase not in ("done", "exit"):
        nxt: Phase = "fall" if y > TOP_Y_MAX else "run"
        names = ("LEFT", "B", "A") if held % 2 == 0 else ("LEFT", "B")
        return names, replace(track, phase=nxt, held=held + 1)

    if phase == "plant":
        if int(state.game_state) == 11:
            # Match seed: dash LEFT+B through the landing door.
            return ("LEFT", "B"), replace(track, held=held + 1)
        if y > 400:
            return ("LEFT",), replace(track, phase="fall", held=0)
        if grounded:
            return ("LEFT", "B"), replace(track, phase="run", held=0)
        return ("LEFT", "B"), replace(track, held=held + 1)

    if phase == "run":
        if y > TOP_Y_MAX + 50:
            return ("LEFT", "A"), replace(track, phase="fall", held=0)
        if x <= LEDGE_X and y < TOP_Y_MAX + 40:
            if grounded:
                return ("RIGHT",), replace(track, phase="face", held=0)
            return (), replace(track, held=held + 1)
        # Seed jump at x≈1127 y≈163 (pose 26) clears the Geemer ledge.
        jumping = is_airborne(state) and x > 900
        want_jump = (grounded and 1000 < x <= GEEMER_JUMP_X) or jumping
        if want_jump:
            return ("LEFT", "B", "A"), replace(track, held=held + 1)
        return ("LEFT", "B"), replace(track, held=held + 1)

    if phase == "face":
        if y > TOP_Y_MAX + 50:
            return ("LEFT",), replace(track, phase="fall", held=0)
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
        lip = LIP_X if y < 160 else SHAFT_LIP_X
        if moonwalking and x <= lip:
            return ("LEFT", "X", "L", "A"), replace(track, phase="jump", held=0)
        if held > 90 and x <= lip + 12:
            return ("LEFT", "X", "L", "A"), replace(track, phase="jump", held=0)
        return ("LEFT", "X", "L"), replace(track, held=held + 1)

    if phase == "jump":
        held += 1
        if is_moonfalling(state) and held >= JUMP_HOLD:
            return ("RIGHT", "A"), replace(track, phase="fall", held=0)
        if held <= JUMP_HOLD:
            return ("LEFT", "X", "L", "A"), replace(track, held=held)
        if held <= JUMP_HOLD + SPIN_HOLD:
            return ("RIGHT", "A"), replace(track, held=held)
        return ("RIGHT",), replace(track, phase="fall", held=0)

    if phase == "fall":
        if grounded and y >= BOTTOM_Y - 80:
            return ("LEFT", "X", "L"), replace(track, phase="downback", held=0)
        if y >= BOTTOM_Y:
            return ("LEFT", "X", "L"), replace(track, phase="downback", held=0)
        if grounded and y < BOTTOM_Y - 80:
            # Seed: downback RIGHT off the y≈173 ledge, LEFT off later
            # grass platforms. Do not re-moonfall on every seat.
            if 160 <= y <= 190 and x < 420:
                return ("RIGHT", "DOWN", "B"), replace(track, held=held + 1)
            return ("LEFT",), replace(track, held=held + 1)
        if y >= AIM_Y:
            steer = _steer_x(x, DOOR_X)
            return steer + ("L", "X"), replace(track, held=held + 1)
        steer = _steer_x(x, SHAFT_X)
        return steer if steer else ("RIGHT",), replace(track, held=held + 1)

    if phase == "downback":
        if room == ROOM_CLIMB:
            return (), replace(track, phase="done", held=0)
        # Live pin: LEFT+X+L from the falling y≈1183 seat clips the floor door.
        if y >= 1170:
            return ("LEFT", "X", "L"), replace(track, phase="exit", held=0)
        if x < 420:
            return ("RIGHT",), replace(track, held=held + 1)
        if x > 435:
            return ("LEFT",), replace(track, held=held + 1)
        return ("LEFT",), replace(track, held=held + 1)

    if phase == "exit":
        if room == ROOM_CLIMB:
            return (), replace(track, phase="done", held=0)
        return ("LEFT", "X", "L"), replace(track, held=held + 1)

    return (), track


def play_parlor_to_climb_moonfall(
    session: ControllerSession,
    *,
    max_frames: int = 1800,
    restore_moonwalk: bool = True,
) -> None:
    """RAM-driven Parlor → Climb using spinning moonfall. Pokes $09E4 on."""
    env = getattr(session, "env", None)
    if env is None:
        raise RuntimeError("parlor moonfall needs session.env for $09E4 poke")
    set_moonwalk(env, True)
    session.state = parse_state(env.get_ram(), frame=session.frame)
    require_moonwalk_on(session.state, label="parlor_moonfall")
    if session.state.room_id != ROOM_PARLOR and session.state.game_state not in (8, 11):
        raise RuntimeError(
            f"parlor moonfall: expected Parlor 0x{ROOM_PARLOR:04X}, got {session.state}"
        )

    track = ParlorMoonfallTrack()
    for _ in range(max_frames):
        names, track = parlor_moonfall_action(session.state, track)
        action = buttons(*names) if names else idle_action()
        session.step(action, f"parlor_moonfall_{track.phase}")
        if track.phase == "done" or session.state.room_id == ROOM_CLIMB:
            break
    else:
        raise TimeoutError(
            f"parlor moonfall missed Climb after {max_frames}f: {session.state} "
            f"phase={track.phase} ({WIKI_URL})"
        )

    if session.state.room_id != ROOM_CLIMB or session.state.game_state == 11:
        session.wait_until(  # type: ignore[attr-defined]
            lambda s: s.room_id == ROOM_CLIMB and s.game_state == 8,
            timeout=CLIMB_SETTLE,
            reason="parlor_moonfall_climb_settle",
        )
    if restore_moonwalk:
        set_moonwalk(env, False)
        session.state = parse_state(env.get_ram(), frame=session.frame)


def setup_then_fall(session: ControllerSession) -> None:
    """In-room initiate_moonfall then idle-steer (practice / dump)."""
    env = getattr(session, "env", None)
    if env is not None:
        set_moonwalk(env, True)
    initiate_moonfall(session, reason="parlor_setup")


__all__ = [
    "PARLOR_MOONFALL_ON_CLEAN",
    "ParlorMoonfallTrack",
    "BOTTOM_Y",
    "DOOR_X",
    "GEEMER_JUMP_X",
    "LEDGE_X",
    "LIP_X",
    "SHAFT_LIP_X",
    "SHAFT_X",
    "parlor_moonfall_action",
    "parlor_moonfall_enabled",
    "play_parlor_to_climb_moonfall",
    "setup_then_fall",
]

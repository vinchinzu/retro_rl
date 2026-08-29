"""Moonwalk + moonfall builders (project-core, even though Map Rando is Hard/VH).

Public policy (wiki.supermetroid.run/Moonwalk):

* Moonwalk is a **file option** (Special Setting Mode). Default off.
  Gameplay flag is WRAM ``$09E4`` (1 = on). :func:`super_metroid.ram.set_moonwalk`
  pokes that copy; menuing the option is equivalent but slower.
* Moonwalk input: shot + the direction **opposite** Samus's facing, plus an
  angle button (L aim-down / R aim-up in this harness).
* Moonfall: hold angle, start a moonwalk, press jump. Vertical direction
  (``$0B36``) stays 0 in air so fall speed is uncapped (normally ~5 px/f).
* Spinning moonfall: release the angle button during the turnaround.

These are builders, not a room AI. Climb's first descent (Parlor → Pit)
composes them in :mod:`super_metroid.routes.kpdr.climb_descent`. Parlor's
first descent (Landing → Climb) composes them in
:mod:`super_metroid.routes.kpdr.parlor_descent`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from super_metroid.ram import (
    FACING_LEFT,
    FACING_RIGHT,
    SuperMetroidState,
)
from super_metroid.routes.controller_common import hold

if TYPE_CHECKING:
    from super_metroid.routes.runtime import ControllerSession

Direction = Literal["LEFT", "RIGHT"]
Aim = Literal["UP", "DOWN"]

# PJBoy movement type $0A1F.
MOVEMENT_JUMPING = 0x02
MOVEMENT_SPIN = 0x03
MOVEMENT_FALLING = 0x06
MOVEMENT_MOONWALKING = 0x10
AIR_MOVEMENT = frozenset({MOVEMENT_JUMPING, MOVEMENT_SPIN, MOVEMENT_FALLING})

# Moonwalk turnaround/jump poses $C0–$C4 (facing-dependent + aim variants).
MOONWALK_TURN_POSES = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC4})

# Ordinary fall caps near 5 px/frame; moonfall underflows past that.
NORMAL_FALL_CAP_PX = 5

WIKI_URL = "https://wiki.supermetroid.run/Moonwalk"


def moonwalk_direction(facing: int) -> Direction:
    """Walk direction for a moonwalk given current facing."""
    return "LEFT" if facing == FACING_RIGHT else "RIGHT"


def angle_button(aim: Aim = "DOWN") -> str:
    """Harness shoulders: R = aim-up, L = aim-down (same as product shots)."""
    return "R" if aim == "UP" else "L"


def moonwalk_buttons(
    facing: int,
    *,
    aim: Aim = "DOWN",
    extra: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Shot + opposite-of-facing + angle. Optional extras (usually A)."""
    names = [moonwalk_direction(facing), "X", angle_button(aim), *extra]
    return tuple(names)


def is_moonwalking(state: SuperMetroidState) -> bool:
    return int(state.movement_type) == MOVEMENT_MOONWALKING


def is_airborne(state: SuperMetroidState) -> bool:
    return int(state.movement_type) in AIR_MOVEMENT


def is_moonfalling(state: SuperMetroidState) -> bool:
    """Moonfall: airborne with vertical direction 0 (wiki / $0B36).

    Grounded standing also has direction 0 — require an air movement type.
    """
    return int(state.vertical_direction) == 0 and is_airborne(state)


def uncapped_fall(state: SuperMetroidState) -> bool:
    """True once vy has passed the ordinary ~5 px/f cap."""
    return is_moonfalling(state) and int(state.velocity_y) > NORMAL_FALL_CAP_PX


def require_moonwalk_on(state: SuperMetroidState, *, label: str = "moonfall") -> None:
    if not state.moonwalk_enabled:
        raise RuntimeError(
            f"{label}: moonwalk flag $09E4 is off (need Special Setting Mode "
            f"or ram.set_moonwalk). {WIKI_URL}"
        )


def initiate_moonfall(
    session: ControllerSession,
    *,
    aim: Aim = "DOWN",
    spin: bool = True,
    walk_frames: int = 10,
    jump_frames: int = 2,
    release_frames: int = 2,
    timeout: int = 40,
    reason: str = "moonfall",
) -> SuperMetroidState:
    """Angle + moonwalk + jump. Optional spin release of the angle button.

    Does not poke the option flag — callers enable moonwalk first.
    """
    require_moonwalk_on(session.state, label=reason)
    facing = int(session.state.facing) or FACING_LEFT
    walk = moonwalk_direction(facing)
    angle = angle_button(aim)
    hold(session, walk_frames, walk, "X", angle, reason=f"{reason}_moonwalk")
    hold(session, jump_frames, walk, "X", angle, "A", reason=f"{reason}_jump")
    if spin:
        hold(session, release_frames, walk, "A", reason=f"{reason}_spin")
    else:
        hold(session, release_frames, walk, angle, "A", reason=f"{reason}_held_angle")
    state = session.state
    for _ in range(timeout):
        if is_moonfalling(state):
            return state
        state = hold(session, 1, walk, reason=f"{reason}_wait")
    return state


def fall_until(
    session: ControllerSession,
    done: Callable[[SuperMetroidState], bool],
    *,
    steer: Callable[[SuperMetroidState], Direction | None] | None = None,
    timeout: int = 400,
    reason: str = "moonfall_fall",
) -> SuperMetroidState:
    """Hold optional steer while falling until ``done`` or timeout."""
    state = session.state
    for _ in range(timeout):
        if done(state):
            return state
        names: tuple[str, ...] = ()
        if steer is not None:
            d = steer(state)
            if d:
                names = (d,)
        state = hold(session, 1, *names, reason=reason)
    raise TimeoutError(f"{reason} timed out at frame {session.frame}: {session.state}")


__all__ = [
    "AIR_MOVEMENT",
    "MOONWALK_TURN_POSES",
    "MOVEMENT_FALLING",
    "MOVEMENT_MOONWALKING",
    "NORMAL_FALL_CAP_PX",
    "WIKI_URL",
    "angle_button",
    "fall_until",
    "initiate_moonfall",
    "is_airborne",
    "is_moonfalling",
    "is_moonwalking",
    "moonwalk_buttons",
    "moonwalk_direction",
    "require_moonwalk_on",
    "uncapped_fall",
]

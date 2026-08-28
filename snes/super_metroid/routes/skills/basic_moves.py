"""Basic / Medium Map Rando movement builders (room-agnostic).

Thin, reusable primitives for room optimization. Tech names match
Map Rando / sm-json-data (see :mod:`super_metroid.rooms.tech_catalog`):

* ``canCrouchJump`` — crouch then jump for a higher short hop
* ``canDownGrab`` — hold DOWN near a ledge to grab
* ``canSpeedyJump`` — dash + jump for longer horizontal jump
* ``canStopOnADime`` — angle-hold to kill horizontal momentum
* ``canDash`` — hold dash + direction (delegates to runway when possible)
* ``canMidAirMorph`` / ``canTrivialMidAirMorph`` — morph mid-air

These are **builders**, not full room clears: callers supply budgets and
success predicates. Prefer composing them inside hop controllers or hill-climb
mutation operators rather than hardcoding room IDs here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import ensure_morph, hold

if TYPE_CHECKING:
    from super_metroid.routes.runtime import ControllerSession

Direction = Literal["LEFT", "RIGHT"]
GuardFn = Callable[[SuperMetroidState], None]


def stop_on_a_dime(
    session: ControllerSession,
    *,
    frames: int = 4,
    aim: Literal["UP", "DOWN"] = "UP",
    reason: str = "stop_on_a_dime",
) -> SuperMetroidState:
    """Kill horizontal momentum with angle hold and no direction (Implicit).

    Map Rando: ``canStopOnADime`` — angle up/down with no D-pad LEFT/RIGHT
    stops Samus. Harness maps shoulder **R** = aim-up, **L** = aim-down
    (same as product angle shots). Useful for elevator pads and precise
    door seating.
    """
    angle = "R" if aim == "UP" else "L"
    hold(session, frames, angle, reason=reason)
    return session.state


def crouch_jump(
    session: ControllerSession,
    *,
    crouch_frames: int = 3,
    jump_frames: int = 8,
    direction: Direction | None = None,
    reason: str = "crouch_jump",
) -> SuperMetroidState:
    """Crouch then jump (Medium ``canCrouchJump``).

    Brief DOWN, then A (+ optional horizontal). Higher short hop than standing
    jump in many geometries; common ledge-into-door builder.
    """
    hold(session, crouch_frames, "DOWN", reason=f"{reason}_crouch")
    buttons = ["A"]
    if direction:
        buttons.append(direction)
    hold(session, jump_frames, *buttons, reason=f"{reason}_jump")
    return session.state


def down_grab(
    session: ControllerSession,
    *,
    frames: int = 20,
    direction: Direction | None = None,
    reason: str = "down_grab",
) -> SuperMetroidState:
    """Hold DOWN near a ledge to grab (Medium ``canDownGrab``).

    Optional horizontal to stay against the wall/lip while grabbing.
    """
    buttons = ["DOWN"]
    if direction:
        buttons.append(direction)
    hold(session, frames, *buttons, reason=reason)
    return session.state


def speedy_jump(
    session: ControllerSession,
    *,
    direction: Direction,
    dash_frames: int = 12,
    jump_frames: int = 10,
    keep_dash: bool = True,
    reason: str = "speedy_jump",
) -> SuperMetroidState:
    """Dash then jump (Medium ``canSpeedyJump``).

    Builds horizontal speed with B+direction, then A. When ``keep_dash`` is
    true, dash stays held through the jump for carry.
    """
    hold(session, dash_frames, direction, "B", reason=f"{reason}_dash")
    jump_btns = [direction, "A"]
    if keep_dash:
        jump_btns.append("B")
    hold(session, jump_frames, *jump_btns, reason=f"{reason}_jump")
    return session.state


def mid_air_morph(
    session: ControllerSession,
    *,
    reason: str = "mid_air_morph",
) -> SuperMetroidState:
    """Morph in air (Basic ``canMidAirMorph`` / Implicit trivial)."""
    ensure_morph(session)
    # ensure_morph may not accept reason; touch state for API parity
    _ = reason
    return session.state


def dash(
    session: ControllerSession,
    *,
    direction: Direction,
    frames: int,
    reason: str = "dash",
) -> SuperMetroidState:
    """Hold dash + direction (Implicit ``canDash``)."""
    hold(session, frames, direction, "B", reason=reason)
    return session.state


def shoot_up_action() -> tuple[str, ...]:
    """Vertical beam: D-pad UP + fire. Not shoulder R (diagonal)."""
    return ("UP", "X")


def shoot_up(
    session: ControllerSession,
    *,
    frames: int = 2,
    reason: str = "shoot_up",
) -> SuperMetroidState:
    """Fire a vertical shot to break overhead shot / Wave blocks.

    Hops were inlining ``UP+X``. Shoulder ``R`` is aim-up diagonal; ``L``
    is aim-down. Neither is a vertical shot.
    """
    hold(session, frames, *shoot_up_action(), reason=reason)
    return session.state


__all__ = [
    "crouch_jump",
    "dash",
    "down_grab",
    "mid_air_morph",
    "shoot_up",
    "shoot_up_action",
    "speedy_jump",
    "stop_on_a_dime",
]

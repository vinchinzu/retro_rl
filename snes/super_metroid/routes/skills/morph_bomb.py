"""Morph bomb-jump and morph-roll micro-skills (room-agnostic budgets).

Used by Kraid return (Kihunter bomb hole → upper band) and similar vertical
morph climbs. Callers pass geometry bands and success predicates; this module
does not hardcode room IDs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import ensure_morph, hold

if TYPE_CHECKING:
    from super_metroid.routes.runtime import ControllerSession

GuardFn = Callable[[SuperMetroidState], None]


def align_x(
    session: ControllerSession,
    *,
    x_lo: int,
    x_hi: int,
    label: str,
    max_frames: int = 50,
    settle_frames: int = 0,
    guard: GuardFn | None = None,
    reason: str = "align",
) -> SuperMetroidState:
    """Walk LEFT/RIGHT until ``samus_x`` is in ``[x_lo, x_hi]``."""
    for _ in range(max_frames):
        state = session.state
        if guard is not None:
            guard(state)
        if x_lo <= state.samus_x <= x_hi:
            break
        direction = "RIGHT" if state.samus_x < x_lo else "LEFT"
        hold(session, 1, direction, reason=f"{label}_{reason}")
    if settle_frames > 0:
        hold(session, settle_frames, reason=f"{label}_{reason}_settle")
    return session.state


def morph_bomb_hole_climb(
    session: ControllerSession,
    *,
    label: str,
    hole_x_lo: int = 372,
    hole_x_hi: int = 382,
    success_y: int = 200,
    peak_y: int = 210,
    settle_y: int = 240,
    firm_y: int = 195,
    max_cycles: int = 90,
    guard: GuardFn | None = None,
    best_min_y: list[int] | None = None,
) -> int:
    """Morph bomb-jump through a floor hole until ``samus_y < success_y``.

    Recenter under the hole each cycle, lay bomb (X), wait by height class,
    then rapid top bombs once peaking. Returns best (minimum) ``samus_y``
    observed. Raises ``TimeoutError`` if the climb does not firm on top.

    ``best_min_y`` is an optional single-element list seeded by the caller so
    pre-climb min_y can be shared with the product hop error string.
    """
    ensure_morph(session)
    min_y = best_min_y[0] if best_min_y else session.state.samus_y
    climbed = False

    def _guard(state: SuperMetroidState) -> None:
        nonlocal min_y
        min_y = min(min_y, state.samus_y)
        if guard is not None:
            guard(state)

    for _cycle in range(max_cycles):
        state = session.state
        _guard(state)
        if state.samus_x < hole_x_lo:
            hold(session, 2, "RIGHT", reason=f"{label}_hole_recenter")
        elif state.samus_x > hole_x_hi:
            hold(session, 2, "LEFT", reason=f"{label}_hole_recenter")
        hold(session, 2, "X", reason=f"{label}_hole_bomb")
        if state.samus_y < 260:
            wait = 22
        elif state.samus_y < 280:
            wait = 30
        else:
            wait = 50
        for _ in range(wait):
            state = hold(session, 1, reason=f"{label}_hole_bomb_wait")
            _guard(state)
        # Firm upper: current settle height, not only a peak boost.
        if session.state.samus_y < success_y:
            climbed = True
            break
        # Once peaking through the hole, keep rapid bombs to settle on top.
        if min_y < peak_y and session.state.samus_y < settle_y:
            hold(session, 2, "X", reason=f"{label}_hole_top_bomb")
            for _ in range(20):
                state = hold(session, 1, reason=f"{label}_hole_top_wait")
                _guard(state)
                if state.samus_y < firm_y:
                    climbed = True
                    break
            if climbed:
                break
    if not climbed:
        raise TimeoutError(
            f"{label}: bomb-hole climb timed out: {session.state}; "
            f"best_min_y={min_y}"
        )
    if best_min_y is not None:
        best_min_y[0] = min_y
    return min_y


def morph_upper_plant(
    session: ControllerSession,
    *,
    label: str,
    plant_y: int = 190,
    max_bombs: int = 8,
    wait_frames: int = 22,
    settle_frames: int = 10,
    fail_y: int = 230,
    guard: GuardFn | None = None,
    best_min_y: list[int] | None = None,
) -> int:
    """Extra bombs to plant on the upper floor after a hole climb."""
    ensure_morph(session)
    min_y = best_min_y[0] if best_min_y else session.state.samus_y
    for _ in range(max_bombs):
        if session.state.samus_y < plant_y:
            break
        hold(session, 2, "X", reason=f"{label}_upper_plant_bomb")
        for _wait in range(wait_frames):
            state = hold(session, 1, reason=f"{label}_upper_plant_wait")
            min_y = min(min_y, state.samus_y)
            if guard is not None:
                guard(state)
    hold(session, settle_frames, reason=f"{label}_upper_morph_settle")
    if session.state.samus_y >= fail_y:
        raise TimeoutError(
            f"{label}: fell off upper after hole climb: {session.state}; "
            f"best_min_y={min_y}"
        )
    if best_min_y is not None:
        best_min_y[0] = min_y
    return min_y


def morph_roll_to_window(
    session: ControllerSession,
    *,
    label: str,
    x_lo: int,
    x_hi: int,
    y_max: int,
    max_frames: int = 500,
    sink_y: int = 210,
    fall_y: int = 300,
    boost_wait: int = 18,
    source_room: int | None = None,
    forbidden_rooms: frozenset[int] = frozenset(),
    guard: GuardFn | None = None,
) -> SuperMetroidState:
    """Morph-roll toward an x-window, bomb-boosting if starting to sink.

    Success: ``x_lo <= samus_x <= x_hi`` and ``samus_y < y_max``.
    """
    for _ in range(max_frames):
        state = session.state
        if state.room_id in forbidden_rooms:
            raise TimeoutError(
                f"{label}: upper traverse crossed wrong door: {session.state}"
            )
        if source_room is not None and state.room_id != source_room:
            raise TimeoutError(
                f"{label}: upper traverse left source room: {session.state}"
            )
        if state.samus_y > fall_y:
            raise TimeoutError(f"{label}: fell during upper traverse: {session.state}")
        if guard is not None:
            guard(state)
        # Bomb-boost if starting to sink through residual floor tiles.
        if state.samus_y > sink_y:
            hold(session, 2, "X", reason=f"{label}_traverse_boost")
            for _wait in range(boost_wait):
                state = hold(session, 1, reason=f"{label}_traverse_boost_wait")
                if guard is not None:
                    guard(state)
        if state.samus_x < x_lo:
            hold(session, 1, "RIGHT", reason=f"{label}_window_recover")
            continue
        if state.samus_x <= x_hi and state.samus_y < y_max:
            return session.state
        hold(session, 1, "LEFT", reason=f"{label}_upper_roll")
    raise TimeoutError(f"{label}: x-window approach timed out: {session.state}")


__all__ = [
    "align_x",
    "morph_bomb_hole_climb",
    "morph_upper_plant",
    "morph_roll_to_window",
]

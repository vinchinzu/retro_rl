"""Shared Spazer hop helpers: lag break, weapon select; re-export play_script."""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import hold, select_weapon
from super_metroid.routes.kpdr.rooms import ROOM_BELOW_SPAZER
from super_metroid.routes.kpdr.spazer.geometry import (
    is_lag_pose,
    is_true_ground_pose,
)
from super_metroid.routes.rle import StopWhen, play_script
from super_metroid.routes.runtime import ControllerSession


def break_lag(
    session: ControllerSession,
    *,
    reason: str = "spazer_lag",
    budget: int = 40,
) -> None:
    """Clear knockback / item-grab lag poses that freeze open-loop RLE."""
    for _ in range(budget):
        if not is_lag_pose(session.state):
            return
        hold(session, 1, "A", reason=reason)
        hold(session, 2, reason=reason)


def wait_true_ground(
    session: ControllerSession,
    *,
    reason: str = "spazer_ground",
    budget: int = 80,
    room_id: int | None = ROOM_BELOW_SPAZER,
) -> SuperMetroidState:
    """Idle (with lag break) until true-ground pose or room leave."""
    for _ in range(budget):
        if room_id is not None and int(session.state.room_id) != room_id:
            return session.state
        if is_true_ground_pose(session.state):
            return session.state
        if is_lag_pose(session.state):
            break_lag(session, reason=reason)
            continue
        hold(session, 1, reason=reason)
    return session.state


def try_select_weapon(session: ControllerSession, slot: int) -> None:
    """Select weapon; ignore transient HUD/phase failures at hop edges."""
    try:
        select_weapon(session, slot)
    except RuntimeError:
        pass


__all__ = [
    "StopWhen",
    "break_lag",
    "play_script",
    "try_select_weapon",
    "wait_true_ground",
]

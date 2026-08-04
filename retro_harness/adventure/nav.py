"""Shared navigation primitives for scripted adventure controllers.

Game modules keep geometry and button timing; this module owns the boring
waypoint / direction math that was copy-pasted across ALTTP, Zelda I, and SM.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Waypoint:
    """An absolute in-room (or on-screen) navigation target."""

    x: int
    y: int
    tolerance: int = 5
    label: str = ""
    room: int | None = None


class PositionLike(Protocol):
    """Minimal position surface (Link/Samus snapshot fields)."""

    @property
    def x(self) -> int: ...

    @property
    def y(self) -> int: ...


def manhattan(ax: int, ay: int, bx: int, by: int) -> int:
    return abs(ax - bx) + abs(ay - by)


def reached_waypoint(
    x: int,
    y: int,
    waypoint: Waypoint,
    *,
    tolerance: int | None = None,
) -> bool:
    tol = waypoint.tolerance if tolerance is None else tolerance
    return abs(x - waypoint.x) <= tol and abs(y - waypoint.y) <= tol


def direction_toward(
    x: int,
    y: int,
    target_x: int,
    target_y: int,
    *,
    x_tol: int = 2,
    y_tol: int = 2,
    prefer_horizontal: bool = True,
) -> str | None:
    """Return a single cardinal button name, or None if within tolerance."""
    dx = target_x - x
    dy = target_y - y
    if abs(dx) <= x_tol and abs(dy) <= y_tol:
        return None
    if prefer_horizontal:
        if abs(dx) > x_tol:
            return "RIGHT" if dx > 0 else "LEFT"
        if abs(dy) > y_tol:
            return "DOWN" if dy > 0 else "UP"
    else:
        if abs(dy) > y_tol:
            return "DOWN" if dy > 0 else "UP"
        if abs(dx) > x_tol:
            return "RIGHT" if dx > 0 else "LEFT"
    return None


def direction_to_waypoint(
    x: int,
    y: int,
    waypoint: Waypoint,
    *,
    x_tol: int | None = None,
    y_tol: int | None = None,
    prefer_horizontal: bool = True,
) -> str | None:
    xt = waypoint.tolerance if x_tol is None else x_tol
    yt = waypoint.tolerance if y_tol is None else y_tol
    return direction_toward(
        x,
        y,
        waypoint.x,
        waypoint.y,
        x_tol=xt,
        y_tol=yt,
        prefer_horizontal=prefer_horizontal,
    )


@dataclass
class WaypointFollower:
    """Advance through an ordered waypoint list."""

    waypoints: Sequence[Waypoint]
    index: int = 0

    @property
    def done(self) -> bool:
        return self.index >= len(self.waypoints)

    @property
    def current(self) -> Waypoint | None:
        if self.done:
            return None
        return self.waypoints[self.index]

    def step(
        self,
        x: int,
        y: int,
        *,
        prefer_horizontal: bool = True,
    ) -> str | None:
        """Return next direction, or None if finished / on waypoint idle.

        Advances ``index`` when the current waypoint is reached.
        """
        while not self.done:
            wp = self.waypoints[self.index]
            if reached_waypoint(x, y, wp):
                self.index += 1
                continue
            return direction_to_waypoint(
                x, y, wp, prefer_horizontal=prefer_horizontal
            )
        return None

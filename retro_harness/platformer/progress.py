"""Progress tracking strategies for side-scroller levels.

Each tracker converts raw RAM values (camera_x, camera_y, etc.) into a
single progress float. The evaluator calls update() each frame and uses
the returned value for fitness and stall detection.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from retro_harness.platformer.level_config import LevelConfig


class ProgressTracker(ABC):
    """Base class for progress tracking strategies."""

    @abstractmethod
    def reset(self) -> None:
        """Reset state for a new evaluation run."""

    @abstractmethod
    def update(self, ram_values: dict[str, int]) -> float:
        """Process a new frame's RAM values and return current progress.

        Returns the highest progress achieved so far (high-water mark).
        """

    @property
    @abstractmethod
    def max_progress(self) -> float:
        """Highest progress value seen so far."""

    @property
    @abstractmethod
    def is_stalled(self) -> bool:
        """Whether progress has stalled (for backtrack-aware trackers)."""


class MonotonicAxisTracker(ProgressTracker):
    """Track progress along a single axis (camera_x or camera_y).

    Covers ~80% of platformer levels: standard left-to-right or
    bottom-to-top scrolling. Progress = high-water mark of the
    tracked axis relative to starting position.
    """

    def __init__(self, axis: str = "camera_x", direction: int = 1) -> None:
        self._axis = axis
        self._direction = direction
        self._initial: float | None = None
        self._max_progress: float = 0.0

    def reset(self) -> None:
        self._initial = None
        self._max_progress = 0.0

    def update(self, ram_values: dict[str, int]) -> float:
        if self._axis not in ram_values:
            return self._max_progress  # no new observation for this axis
        value = float(ram_values[self._axis])
        if self._initial is None:
            self._initial = value
        progress = (value - self._initial) * self._direction
        if progress > self._max_progress:
            self._max_progress = progress
        return self._max_progress

    @property
    def max_progress(self) -> float:
        return self._max_progress

    @property
    def is_stalled(self) -> bool:
        return False  # stall is handled by the evaluator's frame counter


class CompositeAxisTracker(ProgressTracker):
    """Track progress as weighted combination of X and Y axes.

    Useful for diagonal scrolling levels where progress goes both
    right and up/down simultaneously.
    """

    def __init__(
        self,
        x_weight: float = 1.0,
        y_weight: float = 1.0,
        x_direction: int = 1,
        y_direction: int = 1,
    ) -> None:
        self._x_weight = x_weight
        self._y_weight = y_weight
        self._x_direction = x_direction
        self._y_direction = y_direction
        self._initial_x: float | None = None
        self._initial_y: float | None = None
        self._max_progress: float = 0.0

    def reset(self) -> None:
        self._initial_x = None
        self._initial_y = None
        self._max_progress = 0.0

    def update(self, ram_values: dict[str, int]) -> float:
        x = float(ram_values.get("camera_x", 0))
        y = float(ram_values.get("camera_y", 0))
        if self._initial_x is None:
            self._initial_x = x
            self._initial_y = y
        dx = (x - self._initial_x) * self._x_direction * self._x_weight
        dy = (y - (self._initial_y or 0)) * self._y_direction * self._y_weight
        progress = dx + dy
        if progress > self._max_progress:
            self._max_progress = progress
        return self._max_progress

    @property
    def max_progress(self) -> float:
        return self._max_progress

    @property
    def is_stalled(self) -> bool:
        return False


class HighWaterWithBacktrack(ProgressTracker):
    """Like MonotonicAxisTracker but tolerates backtracking.

    For maze levels (e.g. DKC aquatic levels) where the optimal path
    may require going backwards. The stall detector ignores regression
    within backtrack_tolerance, so the GA doesn't get early-terminated
    when the optimal path backtracks.

    Progress = highest point reached (not current position).
    """

    def __init__(
        self,
        axis: str = "camera_x",
        direction: int = 1,
        backtrack_tolerance: float = 200.0,
    ) -> None:
        self._axis = axis
        self._direction = direction
        self._backtrack_tolerance = backtrack_tolerance
        self._initial: float | None = None
        self._max_progress: float = 0.0
        self._current_progress: float = 0.0

    def reset(self) -> None:
        self._initial = None
        self._max_progress = 0.0
        self._current_progress = 0.0

    def update(self, ram_values: dict[str, int]) -> float:
        if self._axis not in ram_values:
            return self._max_progress  # no new observation for this axis
        value = float(ram_values[self._axis])
        if self._initial is None:
            self._initial = value
        self._current_progress = (value - self._initial) * self._direction
        if self._current_progress > self._max_progress:
            self._max_progress = self._current_progress
        return self._max_progress

    @property
    def max_progress(self) -> float:
        return self._max_progress

    @property
    def is_stalled(self) -> bool:
        """Stalled only if regression exceeds tolerance."""
        regression = self._max_progress - self._current_progress
        return regression > self._backtrack_tolerance


class WaypointTracker(ProgressTracker):
    """Track progress through a user-defined list of waypoints.

    For complex non-linear paths where simple axis tracking doesn't work.
    Progress = index of furthest waypoint reached + fractional distance
    to the next waypoint.
    """

    def __init__(self, waypoints: list[tuple[float, float]], capture_radius: float = 32.0) -> None:
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints")
        self._waypoints = waypoints
        self._capture_radius = capture_radius
        self._furthest_wp: int = 0
        self._max_progress: float = 0.0

    def reset(self) -> None:
        self._furthest_wp = 0
        self._max_progress = 0.0

    def update(self, ram_values: dict[str, int]) -> float:
        px = float(ram_values.get("player_x", ram_values.get("camera_x", 0)))
        py = float(ram_values.get("player_y", 0))

        # Check if we've reached the next waypoint(s)
        while self._furthest_wp < len(self._waypoints) - 1:
            wx, wy = self._waypoints[self._furthest_wp + 1]
            dist = math.hypot(px - wx, py - wy)
            if dist <= self._capture_radius:
                self._furthest_wp += 1
            else:
                break

        # Fractional progress toward next waypoint
        if self._furthest_wp < len(self._waypoints) - 1:
            wx, wy = self._waypoints[self._furthest_wp]
            nx, ny = self._waypoints[self._furthest_wp + 1]
            seg_len = math.hypot(nx - wx, ny - wy)
            if seg_len > 0:
                # Project current position onto segment
                dx, dy = nx - wx, ny - wy
                t = max(0, min(1, ((px - wx) * dx + (py - wy) * dy) / (seg_len * seg_len)))
                frac = t
            else:
                frac = 0.0
        else:
            frac = 0.0

        progress = float(self._furthest_wp) + frac
        if progress > self._max_progress:
            self._max_progress = progress
        return self._max_progress

    @property
    def max_progress(self) -> float:
        return self._max_progress

    @property
    def is_stalled(self) -> bool:
        return False


def make_progress_tracker(config: LevelConfig) -> ProgressTracker:
    """Factory: create the right tracker for a level config."""
    if config.progress_axis == "camera_x":
        if config.backtrack_tolerance > 0:
            return HighWaterWithBacktrack(
                axis="camera_x",
                direction=config.progress_direction,
                backtrack_tolerance=config.backtrack_tolerance,
            )
        return MonotonicAxisTracker(axis="camera_x", direction=config.progress_direction)

    if config.progress_axis == "camera_y":
        if config.backtrack_tolerance > 0:
            return HighWaterWithBacktrack(
                axis="camera_y",
                direction=config.progress_direction,
                backtrack_tolerance=config.backtrack_tolerance,
            )
        return MonotonicAxisTracker(axis="camera_y", direction=config.progress_direction)

    if config.progress_axis == "player_x":
        if config.backtrack_tolerance > 0:
            return HighWaterWithBacktrack(
                axis="player_x",
                direction=config.progress_direction,
                backtrack_tolerance=config.backtrack_tolerance,
            )
        return MonotonicAxisTracker(axis="player_x", direction=config.progress_direction)

    if config.progress_axis == "player_y":
        if config.backtrack_tolerance > 0:
            return HighWaterWithBacktrack(
                axis="player_y",
                direction=config.progress_direction,
                backtrack_tolerance=config.backtrack_tolerance,
            )
        return MonotonicAxisTracker(axis="player_y", direction=config.progress_direction)

    if config.progress_axis == "composite":
        return CompositeAxisTracker(x_direction=config.progress_direction)

    if config.progress_axis == "waypoints":
        if not config.waypoints:
            raise ValueError(f"Level {config.level_id} uses waypoint progress but has no waypoints")
        return WaypointTracker(
            waypoints=config.waypoints,
            capture_radius=getattr(config, "waypoint_capture_radius", 64.0),
        )

    raise ValueError(f"Unknown progress_axis: {config.progress_axis!r}")

"""Stuck detection for long-horizon oneshot runs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from snes_oneshot.game_state import GameState


class WatchdogEvent(Enum):
    """Signals emitted when the agent appears stuck."""

    NONE = auto()
    POSITION_STALLED = auto()
    CAMERA_STALLED = auto()
    HEALTH_DRAINING = auto()
    ENEMY_STALLED = auto()


@dataclass
class StuckDetector:
    """Track progress signals and report stall conditions.

    Args:
        position_window: Frames without player movement before stall.
        camera_window: Frames without camera movement before stall.
        health_window: Frames of health decline without progress.
        enemy_window: Frames with living enemies and no health drop.
        move_epsilon: Minimum player delta counted as movement.
    """

    position_window: int = 180
    camera_window: int = 300
    health_window: int = 240
    enemy_window: int = 360
    move_epsilon: int = 1

    def __post_init__(self) -> None:
        self._last_x: int | None = None
        self._last_y: int | None = None
        self._last_cam_x: int | None = None
        self._pos_stall = 0
        self._cam_stall = 0
        self._health_stall = 0
        self._enemy_stall = 0
        self._last_health: int | None = None
        self._last_enemy_health_sum: int | None = None

    def reset(self) -> None:
        """Clear stall counters."""
        self.__post_init__()

    def update(self, state: GameState) -> WatchdogEvent:
        """Feed a frame of state; return the highest-priority stall event."""
        moved = False
        if self._last_x is not None and self._last_y is not None:
            dx = abs(state.player_x - self._last_x)
            dy = abs(state.player_y - self._last_y)
            moved = dx >= self.move_epsilon or dy >= self.move_epsilon
            self._pos_stall = 0 if moved else self._pos_stall + 1
        self._last_x = state.player_x
        self._last_y = state.player_y

        if self._last_cam_x is not None:
            cam_moved = abs(state.camera_x - self._last_cam_x) >= self.move_epsilon
            self._cam_stall = 0 if cam_moved or moved else self._cam_stall + 1
        self._last_cam_x = state.camera_x

        if self._last_health is not None and state.health < self._last_health:
            self._health_stall += 1
        else:
            self._health_stall = 0
        self._last_health = state.health

        enemy_sum = sum(e.health for e in state.living_enemies)
        if state.living_enemies:
            if (
                self._last_enemy_health_sum is not None
                and enemy_sum >= self._last_enemy_health_sum
            ):
                self._enemy_stall += 1
            else:
                self._enemy_stall = 0
        else:
            self._enemy_stall = 0
        self._last_enemy_health_sum = enemy_sum

        if self._pos_stall >= self.position_window:
            return WatchdogEvent.POSITION_STALLED
        if self._enemy_stall >= self.enemy_window:
            return WatchdogEvent.ENEMY_STALLED
        if self._health_stall >= self.health_window:
            return WatchdogEvent.HEALTH_DRAINING
        if self._cam_stall >= self.camera_window:
            return WatchdogEvent.CAMERA_STALLED
        return WatchdogEvent.NONE

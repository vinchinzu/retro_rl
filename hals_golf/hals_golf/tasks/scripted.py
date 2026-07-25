"""Frame-script task for recorded or hand-authored button sequences."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from retro_harness.protocol import (
    ActionResult,
    TaskResult,
    TaskStatus,
    WorldState,
)

from hals_golf.core.actions import idle


@dataclass
class ScriptedTask:
    """Play a fixed list of per-frame actions to completion."""

    name: str
    frames: list[np.ndarray] = field(default_factory=list)
    _index: int = 0

    def reset(self, world: WorldState) -> None:
        del world
        self._index = 0

    def can_start(self, world: WorldState) -> bool:
        del world
        return bool(self.frames)

    def step(self, world: WorldState) -> TaskResult:
        del world
        if self._index >= len(self.frames):
            return TaskResult(status=TaskStatus.SUCCESS)
        action = self.frames[self._index]
        self._index += 1
        done = self._index >= len(self.frames)
        return TaskResult(
            status=TaskStatus.SUCCESS if done else TaskStatus.RUNNING,
            action=ActionResult(action=action, reason=self.name),
        )


@dataclass
class IdleTask:
    """Hold idle for a fixed number of frames."""

    name: str = "idle"
    frames: int = 60
    _left: int = 0

    def reset(self, world: WorldState) -> None:
        del world
        self._left = self.frames

    def can_start(self, world: WorldState) -> bool:
        del world
        return True

    def step(self, world: WorldState) -> TaskResult:
        del world
        if self._left <= 0:
            return TaskResult(status=TaskStatus.SUCCESS)
        self._left -= 1
        status = TaskStatus.SUCCESS if self._left <= 0 else TaskStatus.RUNNING
        return TaskResult(
            status=status,
            action=ActionResult(action=idle(), reason=self.name),
        )

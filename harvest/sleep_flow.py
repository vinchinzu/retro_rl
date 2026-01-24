"""Sleep interaction flow using recorded task if available."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from harness import Task, TaskResult, TaskStatus, WorldState
from recorded_task import RecordedTask


@dataclass
class SleepInteractionTask(Task):
    name: str = "go_to_sleep"
    _task: Optional[RecordedTask] = None

    def reset(self, world: WorldState) -> None:
        if self._task is None:
            self._task = RecordedTask.load(self.name)
        self._task.reset(world)

    def can_start(self, world: WorldState) -> bool:
        try:
            if self._task is None:
                self._task = RecordedTask.load(self.name)
            return True
        except FileNotFoundError:
            return False

    def step(self, world: WorldState) -> TaskResult:
        if self._task is None:
            return TaskResult(status=TaskStatus.BLOCKED, reason="recorded task missing")
        return self._task.step(world)

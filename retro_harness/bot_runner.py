"""
Autopilot framework for running Task-based bots.

Bridges the Task protocol with PlaySession by building WorldState
from env output and converting TaskResult actions into numpy arrays.
"""
from __future__ import annotations

import numpy as np
from retro_harness.protocol import Task, TaskStatus, TaskResult, WorldState, ActionResult


class BotRunner:
    """Wraps a Task to provide actions for PlaySession's bot interface.

    Callable as runner(obs, info) -> action_array or None, compatible
    with PlaySession's bot parameter.
    """

    def __init__(self, task, *, ram_schema=None, action_size=12):
        self.task = task
        self.ram_schema = ram_schema
        self.action_size = action_size
        self._frame = 0
        self._initialized = False

    def __call__(self, obs, info) -> np.ndarray | None:
        """Called by PlaySession each frame. Returns action or None."""
        world = self._build_world(obs, info)
        if not self._initialized:
            self.task.reset(world)
            self._initialized = True

        result = self.task.step(world)
        self._frame += 1

        if result.status in (TaskStatus.SUCCESS, TaskStatus.FAILURE):
            return None  # done, fall to human
        elif result.action is not None:
            return np.array(result.action.action, dtype=np.int8)
        else:
            return np.zeros(self.action_size, dtype=np.int8)  # idle

    def reset(self):
        """Reset the runner for a new episode."""
        self._frame = 0
        self._initialized = False

    def _build_world(self, obs, info):
        ram = info.get("ram", np.array([], dtype=np.uint8))
        meta = self.ram_schema.read(ram) if self.ram_schema is not None else {}
        return WorldState(
            frame=self._frame, ram=ram, info=info, obs=obs, meta=meta,
        )


class TaskSequencer:
    """Run a sequence of Tasks in order. Implements the Task protocol.

    Each task runs to SUCCESS, then the next starts. If any task
    returns FAILURE, the sequencer fails.
    """

    name: str = "TaskSequencer"

    def __init__(self, tasks):
        self.tasks = list(tasks)
        self._idx = 0

    def reset(self, world):
        self._idx = 0
        if self.tasks:
            self.tasks[0].reset(world)

    def can_start(self, world):
        return bool(self.tasks) and self.tasks[0].can_start(world)

    def step(self, world) -> TaskResult:
        if self._idx >= len(self.tasks):
            return TaskResult(status=TaskStatus.SUCCESS)

        result = self.tasks[self._idx].step(world)

        if result.status == TaskStatus.SUCCESS:
            self._idx += 1
            if self._idx >= len(self.tasks):
                return TaskResult(status=TaskStatus.SUCCESS, action=result.action)
            self.tasks[self._idx].reset(world)
            return TaskResult(status=TaskStatus.RUNNING, action=result.action)

        return result

    @property
    def current_task_index(self) -> int:
        return self._idx

    @property
    def current_task(self) -> Task | None:
        if self._idx < len(self.tasks):
            return self.tasks[self._idx]
        return None


class TaskRepeater:
    """Repeat a task N times (or indefinitely if times=None).

    Implements the Task protocol.
    """

    name: str = "TaskRepeater"

    def __init__(self, task, *, times=None):
        self.task = task
        self.max_times = times
        self._count = 0

    def reset(self, world):
        self._count = 0
        self.task.reset(world)

    def can_start(self, world):
        return self.task.can_start(world)

    def step(self, world) -> TaskResult:
        result = self.task.step(world)

        if result.status == TaskStatus.SUCCESS:
            self._count += 1
            if self.max_times is not None and self._count >= self.max_times:
                return TaskResult(status=TaskStatus.SUCCESS, action=result.action)
            self.task.reset(world)
            return TaskResult(status=TaskStatus.RUNNING, action=result.action)

        return result

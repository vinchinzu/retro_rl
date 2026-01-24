"""Daily chain: weather check -> go home -> sleep (scaffold)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harness_graph import TaskGraph, TaskNode


@dataclass
class WaitFramesTask(Task):
    name: str
    frames: int
    _remaining: int = 0

    def reset(self, world: WorldState) -> None:
        self._remaining = self.frames

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        if self._remaining <= 0:
            return TaskResult(status=TaskStatus.SUCCESS)
        self._remaining -= 1
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(np.zeros(12, dtype=np.int32)),
        )


@dataclass
class WeatherCheckTask(Task):
    name: str = "weather_check"

    def reset(self, world: WorldState) -> None:
        pass

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        weather = world.info.get("weather") or world.info.get("forecast")
        if weather is None:
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(np.zeros(12, dtype=np.int32)),
                reason="weather unknown",
                checkpoint="weather_check",
            )
        return TaskResult(status=TaskStatus.SUCCESS, checkpoint="weather_check", meta={"weather": weather})


@dataclass
class GoHomeTask(WaitFramesTask):
    name: str = "go_home"
    frames: int = 60


@dataclass
class SleepTask(WaitFramesTask):
    name: str = "sleep"
    frames: int = 30


def build_daily_chain(
    weather_task: Optional[Task] = None,
    go_home_task: Optional[Task] = None,
    sleep_task: Optional[Task] = None,
) -> TaskGraph:
    weather_task = weather_task or WeatherCheckTask()
    go_home_task = go_home_task or GoHomeTask()
    sleep_task = sleep_task or SleepTask()

    nodes = [
        TaskNode(name="weather", task=weather_task, on_success="go_home", checkpoint="weather_check"),
        TaskNode(name="go_home", task=go_home_task, on_success="sleep", checkpoint="home_reached"),
        TaskNode(name="sleep", task=sleep_task, checkpoint="sleeping"),
    ]
    return TaskGraph(nodes=nodes, start="weather")

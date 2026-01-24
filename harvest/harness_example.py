"""Minimal harness wiring example (mock env + simple tasks)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harness_graph import LinearChain
from harness_runtime import HarnessRunner
from day_planner import DayPlanner, PlanRule


class MockEnv:
    """Tiny env shim for exercising the harness without an emulator."""

    def __init__(self, action_shape=(12,)):
        self.action_space = type("ActionSpace", (), {"shape": action_shape, "dtype": np.int32})()
        self._frame = 0

    def reset(self, state: Optional[str] = None):
        self._frame = 0
        obs = np.zeros((32, 32, 3), dtype=np.uint8)
        info = {"hour": 6, "stamina": 100, "weather": "sun"}
        return obs, info

    def step(self, action):
        self._frame += 1
        obs = np.zeros((32, 32, 3), dtype=np.uint8)
        info = {"hour": 6 + (self._frame // 600), "stamina": max(0, 100 - self._frame // 30), "weather": "sun"}
        return obs, 0.0, False, info

    def get_ram(self):
        return np.zeros(2048, dtype=np.uint8)


@dataclass
class WaitTask(Task):
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
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(np.zeros(12, dtype=np.int32)))


def build_demo_chain() -> LinearChain:
    wake = WaitTask(name="wake", frames=30)
    clear = WaitTask(name="clear", frames=60)
    ship = WaitTask(name="ship", frames=30)
    return LinearChain.from_tasks([wake, clear, ship])


def build_demo_planner(sleep_task: Task) -> DayPlanner:
    water = WaitTask(name="water", frames=40)
    ship = WaitTask(name="ship", frames=20)
    rules = [
        PlanRule(name="water", task=water, predicate=lambda w: True, priority=50),
        PlanRule(name="ship", task=ship, predicate=lambda w: True, priority=40),
    ]
    return DayPlanner(name="demo_plan", rules=rules, sleep_task=sleep_task)


def run_demo():
    env = MockEnv()
    runner = HarnessRunner(env)
    world = runner.reset()

    chain = build_demo_chain()
    while True:
        result = chain.step(world)
        if result.status in (TaskStatus.SUCCESS, TaskStatus.FAILURE):
            break
        world = runner.step_env(result.action.action)

    sleep_task = WaitTask(name="sleep", frames=10)
    planner = build_demo_planner(sleep_task)
    planner.reset(world)

    for _ in range(120):
        result = planner.step(world)
        if result.status == TaskStatus.SUCCESS:
            break
        if result.action is None:
            world = runner.step_env(np.zeros(12, dtype=np.int32))
        else:
            world = runner.step_env(result.action.action)


if __name__ == "__main__":
    run_demo()

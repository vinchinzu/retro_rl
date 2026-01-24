"""Watering + refill behavior using recorded tasks and RAM water level."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from recorded_task import RecordedTask
WATER_CAN_ADDR = 0x0926


def _water_level(ram: np.ndarray) -> int:
    return int(ram[WATER_CAN_ADDR]) if WATER_CAN_ADDR < len(ram) else 0


@dataclass
class WateringAndRefillTask(Task):
    name: str = "water_and_refill"
    water_task_name: str = "water_planted_tile"
    refill_task_name: str = "go_to_water_source"
    min_level: int = 1
    _water_task: Optional[RecordedTask] = None
    _refill_task: Optional[RecordedTask] = None
    _mode: str = "watering"
    _last_level: int = 0

    def reset(self, world: WorldState) -> None:
        if self._water_task is None:
            self._water_task = RecordedTask.load(self.water_task_name)
        if self._refill_task is None:
            self._refill_task = RecordedTask.load(self.refill_task_name)
        self._water_task.reset(world)
        self._refill_task.reset(world)
        self._mode = "watering"
        self._last_level = _water_level(world.ram)

    def can_start(self, world: WorldState) -> bool:
        try:
            if self._water_task is None:
                self._water_task = RecordedTask.load(self.water_task_name)
            if self._refill_task is None:
                self._refill_task = RecordedTask.load(self.refill_task_name)
            return True
        except FileNotFoundError:
            return False

    def step(self, world: WorldState) -> TaskResult:
        water_level = _water_level(world.ram)

        # Switch to refill if low
        if self._mode == "watering" and water_level <= self.min_level:
            self._mode = "refill"
            if self._refill_task is not None:
                self._refill_task.reset(world)

        # Switch back to watering when refill increases level
        if self._mode == "refill" and water_level > self._last_level:
            self._mode = "watering"
            if self._water_task is not None:
                self._water_task.reset(world)

        self._last_level = water_level

        if self._mode == "watering":
            if self._water_task is None:
                return TaskResult(status=TaskStatus.BLOCKED, reason="water task missing")
            return self._water_task.step(world)

        if self._mode == "refill":
            if self._refill_task is None:
                return TaskResult(status=TaskStatus.BLOCKED, reason="refill task missing")
            return self._refill_task.step(world)

        return TaskResult(status=TaskStatus.FAILURE, reason="unknown mode")

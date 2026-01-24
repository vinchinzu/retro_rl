"""Building navigation tasks (barn/coop) using recorded sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set

import numpy as np

from harness import Task, TaskResult, TaskStatus, WorldState
from recorded_task import RecordedTask
from farm_clearer import ADDR_TILEMAP


MAP_BARN = 0x27
MAP_COOP = 0x28
FARM_MAP_IDS: Set[int] = {0x00, 0x04}  # 0x00 normal, 0x04 winter (from HM-Decomp)


def _get_tilemap(ram: np.ndarray) -> int:
    return int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0


@dataclass
class NavigateToBuildingTask(Task):
    name: str
    target_map_id: int
    recorded_name: str
    farm_map_ids: Set[int] = None
    _task: Optional[RecordedTask] = None

    def reset(self, world: WorldState) -> None:
        if self._task is None:
            self._task = RecordedTask.load(self.recorded_name)
        self._task.reset(world)

    def can_start(self, world: WorldState) -> bool:
        try:
            if self._task is None:
                self._task = RecordedTask.load(self.recorded_name)
            return True
        except FileNotFoundError:
            return False

    def step(self, world: WorldState) -> TaskResult:
        tilemap = _get_tilemap(world.ram)
        if tilemap == self.target_map_id:
            return TaskResult(status=TaskStatus.SUCCESS)

        farm_maps = self.farm_map_ids or FARM_MAP_IDS
        if tilemap not in farm_maps:
            return TaskResult(status=TaskStatus.BLOCKED, reason=f"unexpected map 0x{tilemap:02X}")

        if self._task is None:
            return TaskResult(status=TaskStatus.BLOCKED, reason="recorded task missing")

        return self._task.step(world)


def go_to_barn_task() -> NavigateToBuildingTask:
    return NavigateToBuildingTask(
        name="go_to_barn",
        target_map_id=MAP_BARN,
        recorded_name="go_to_barn",
    )


def go_to_coop_task() -> NavigateToBuildingTask:
    return NavigateToBuildingTask(
        name="go_to_coop",
        target_map_id=MAP_COOP,
        recorded_name="go_to_coop",
    )

"""Return-home and sleep tasks used by the day planner.

Compatibility barrel: implementations live in focused modules
(:mod:`home_return`, :mod:`home_sleep`, plus geometry/policy in
:mod:`home_approach` / :mod:`home_recover`). Existing
``from harvest.planner.tasks.home import X`` imports keep working.
"""

from __future__ import annotations

from dataclasses import dataclass

from retro_harness import Task, TaskResult, TaskStatus, WorldState

from harvest.planner.tasks.home_return import (
    HOUSE_DOOR_FRONT_PX,
    HOUSE_FRONT_PX,
    ReturnHomeTask,
)
from harvest.planner.tasks.home_sleep import (
    HOUSE_BED_STAND_PX,
    HOUSE_BED_STAND_TOLERANCE,
    HOUSE_SLEEP_TRANSITION_TILEMAP,
    GoToSleepTask,
)


@dataclass
class ReadyToGoHomeTask(Task):
    """Marker task: town/day work is done; planner should end the day.

    Success is the go-home flag. The day planner records it and advances into
    ``RETURN_HOME`` / ``GO_TO_SLEEP`` (or appends them if missing).
    """

    name: str = "ready_to_go_home"

    def reset(self, world: WorldState) -> None:
        return None

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        return TaskResult(
            status=TaskStatus.SUCCESS,
            reason="ready_to_go_home",
            checkpoint="ready_to_go_home",
            meta={"ready_to_go_home": True},
        )



__all__ = [
    "HOUSE_FRONT_PX",
    "HOUSE_DOOR_FRONT_PX",
    "HOUSE_BED_STAND_PX",
    "HOUSE_SLEEP_TRANSITION_TILEMAP",
    "HOUSE_BED_STAND_TOLERANCE",
    "ReadyToGoHomeTask",
    "ReturnHomeTask",
    "GoToSleepTask",
]

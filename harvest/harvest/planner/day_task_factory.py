"""Factory for turning phase specs into runnable tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from retro_harness import Task, WorldState

from harvest.planner.day_phase_registry import TaskBuildContext, build_phase_task
from harvest.planner.day_phase_types import PhaseSpec
from harvest.planner.day_plan_status import TASKS_DIR


@dataclass(frozen=True)
class DayTaskFactory:
    seed_type: str = "potato"
    tasks_dir: str = TASKS_DIR
    state_name: Optional[str] = None

    def make_task(self, spec: PhaseSpec, world: WorldState) -> Optional[Task]:
        """Create the sub-task for a phase spec."""
        ctx = TaskBuildContext(
            seed_type=self.seed_type,
            tasks_dir=self.tasks_dir,
            state_name=self.state_name,
        )
        return build_phase_task(ctx, spec, world)


__all__ = ["DayTaskFactory"]

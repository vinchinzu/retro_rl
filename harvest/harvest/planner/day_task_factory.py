"""Factory for turning phase specs into runnable tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from retro_harness import Task, WorldState

from harvest.core.world_context import WorldContext
from harvest.planner.day_phase_registry import TaskBuildContext, build_phase_task
from harvest.planner.day_phase_types import DayPlannerPolicy, PhaseSpec
from harvest.planner.day_plan_status import TASKS_DIR


@dataclass
class DayTaskFactory:
    seed_type: str = "potato"
    tasks_dir: str = TASKS_DIR
    state_name: Optional[str] = None
    policy: Optional[DayPlannerPolicy] = None
    world_context: WorldContext = field(default_factory=WorldContext)

    def make_task(self, spec: PhaseSpec, world: WorldState) -> Optional[Task]:
        """Create the sub-task for a phase spec."""
        self.world_context.bind(world)
        ctx = TaskBuildContext(
            seed_type=self.seed_type,
            tasks_dir=self.tasks_dir,
            state_name=self.state_name,
            policy=self.policy,
            world_context=self.world_context,
        )
        return build_phase_task(ctx, spec, world)


__all__ = ["DayTaskFactory"]

"""Day planner core: wake -> tasks -> bed with simple branching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from harness import ActionResult, Plan, Task, TaskResult, TaskStatus, WorldState


@dataclass
class PlanRule:
    name: str
    task: Task
    predicate: Callable[[WorldState], bool]
    priority: int = 0


def _zero_action() -> np.ndarray:
    return np.zeros(12, dtype=np.int32)


def _get_info_int(world: WorldState, key: str, default: int = 0) -> int:
    try:
        return int(world.info.get(key, default))
    except Exception:
        return default


def is_night(world: WorldState, hour_cutoff: int) -> bool:
    hour = _get_info_int(world, "hour", 6)
    return hour >= hour_cutoff


def is_raining(world: WorldState) -> bool:
    for key in ("is_raining", "raining", "rain"):
        if key in world.info:
            return bool(world.info.get(key))
    weather = str(world.info.get("weather", "")).lower()
    return "rain" in weather


def low_stamina(world: WorldState, threshold: int) -> bool:
    stamina = _get_info_int(world, "stamina", 0)
    return stamina <= threshold


class DayPlanner(Plan):
    """Simple planner that selects tasks based on rules and bedtime."""

    def __init__(
        self,
        name: str,
        rules: Optional[List[PlanRule]] = None,
        sleep_task: Optional[Task] = None,
        bedtime_hour: int = 22,
        stamina_floor: int = 10,
    ):
        self.name = name
        self.rules: List[PlanRule] = rules or []
        self.sleep_task = sleep_task
        self.bedtime_hour = bedtime_hour
        self.stamina_floor = stamina_floor
        self._current: Optional[Task] = None
        self._completed: Dict[str, bool] = {}

    def reset(self, world: WorldState) -> None:
        self._current = None
        self._completed = {}

    def next_tasks(self, world: WorldState) -> List[Task]:
        if self.sleep_task and is_night(world, self.bedtime_hour):
            return [self.sleep_task]

        ordered = sorted(self.rules, key=lambda r: r.priority, reverse=True)
        for rule in ordered:
            if self._completed.get(rule.name):
                continue
            if rule.predicate(world):
                return [rule.task]
        return []

    def step(self, world: WorldState) -> TaskResult:
        # Decide on a task if we don't have one or it just finished.
        if self._current is None:
            next_tasks = self.next_tasks(world)
            self._current = next_tasks[0] if next_tasks else None
            if self._current:
                self._current.reset(world)

        if self._current is None:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(_zero_action()), reason="idle")

        result = self._current.step(world)

        if result.status == TaskStatus.RUNNING:
            return result

        # Mark completed and clear current task.
        for rule in self.rules:
            if rule.task is self._current:
                self._completed[rule.name] = True
                break

        if self.sleep_task is self._current and result.status == TaskStatus.SUCCESS:
            return TaskResult(status=TaskStatus.SUCCESS, reason="day complete")

        self._current = None
        return TaskResult(status=TaskStatus.RUNNING, checkpoint=result.checkpoint)


def make_default_rules(
    tasks: Dict[str, Task],
    stamina_floor: int = 10,
) -> List[PlanRule]:
    """Helper to build common rules from named tasks."""

    rules: List[PlanRule] = []

    if "water" in tasks:
        rules.append(PlanRule(
            name="water",
            task=tasks["water"],
            predicate=lambda w: not is_raining(w),
            priority=50,
        ))

    if "ship" in tasks:
        rules.append(PlanRule(
            name="ship",
            task=tasks["ship"],
            predicate=lambda w: True,
            priority=40,
        ))

    if "clear" in tasks:
        rules.append(PlanRule(
            name="clear",
            task=tasks["clear"],
            predicate=lambda w: not low_stamina(w, stamina_floor),
            priority=20,
        ))

    return rules

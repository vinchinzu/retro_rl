"""Tests for task progress snapshots."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from harvest.core.task_progress import ProgressSnapshot, task_progress_snapshot
from harvest.planner.day_plan_orchestrator import DayPlanTask, MultiDayPlannerTask


class TaskProgressTests(unittest.TestCase):
    def test_day_plan_task_progress_snapshot_uses_public_accessors(self) -> None:
        task = DayPlanTask(phase_sequence=[])
        task._phase_index = 0
        task._step_count = 3
        task._current_task = SimpleNamespace(
            progress_snapshot=lambda: ProgressSnapshot(
                task_name="NavTask",
                phase_text="nav",
                details=(("target", (1, 2)),),
            )
        )

        snap = task.progress_snapshot()
        self.assertEqual(snap.task_name, "DayPlanTask")
        self.assertEqual(snap.phase_index, 0)
        self.assertEqual(snap.step_count, 3)
        self.assertIsNotNone(snap.child)
        self.assertEqual(snap.child.task_name, "NavTask")

    def test_multi_day_planner_progress_snapshot_includes_days_completed(self) -> None:
        task = MultiDayPlannerTask()
        task._phase = "plan_day"
        task._days_completed = 2
        task._step_count = 5

        snap = task.progress_snapshot()
        self.assertEqual(snap.task_name, "MultiDayPlannerTask")
        self.assertEqual(snap.phase_text, "PLAN_DAY")
        self.assertEqual(snap.step_count, 5)
        self.assertIn(("days_completed", 2), snap.details)

    def test_task_progress_snapshot_falls_back_without_method(self) -> None:
        leaf = SimpleNamespace(
            phase_text="water",
            phase_index=2,
            step_count=10,
        )
        snap = task_progress_snapshot(leaf)
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertEqual(snap.task_name, "SimpleNamespace")
        self.assertEqual(snap.phase_text, "water")
        self.assertEqual(snap.phase_index, 2)


if __name__ == "__main__":
    unittest.main()

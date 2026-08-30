"""Tests for task progress snapshots."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from harvest.core.task_progress import (
    GOAL_STALL_FRAMES,
    MOTION_STALL_FRAMES,
    ProgressSnapshot,
    stalled,
    task_progress_snapshot,
)
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

    def test_signature_ignores_step_count_field(self) -> None:
        base = dict(
            task_name="DayPlanTask",
            phase_text="CLEAR",
            phase_index=1,
            details=(("cleared", 4),),
        )
        first = ProgressSnapshot(step_count=3, **base)
        second = ProgressSnapshot(step_count=300, **base)
        self.assertEqual(first.signature(), second.signature())
        self.assertNotEqual(first.step_count, second.step_count)

    def test_day_plan_step_count_ticks_do_not_change_signature(self) -> None:
        task = DayPlanTask(phase_sequence=[])
        task._phase_index = 0
        task._current_task = SimpleNamespace(
            progress_snapshot=lambda: ProgressSnapshot(
                task_name="NavTask",
                phase_text="nav",
                details=(("target", (1, 2)),),
            )
        )
        task._step_count = 3
        first = task.progress_snapshot().signature()
        task._step_count = 300
        second = task.progress_snapshot().signature()
        self.assertEqual(first, second)
        self.assertEqual(task.progress_snapshot().step_count, 300)

    def test_signature_ignores_step_count_detail_pair(self) -> None:
        first = ProgressSnapshot(
            task_name="Leaf",
            details=(("target", (1, 2)), ("step_count", 3)),
        )
        second = ProgressSnapshot(
            task_name="Leaf",
            details=(("target", (1, 2)), ("step_count", 300)),
        )
        self.assertEqual(first.signature(), second.signature())

    def test_signature_includes_target_detail(self) -> None:
        first = ProgressSnapshot(task_name="Leaf", details=(("target", (1, 2)),))
        second = ProgressSnapshot(task_name="Leaf", details=(("target", (3, 4)),))
        self.assertNotEqual(first.signature(), second.signature())

    def test_motion_stall_window(self) -> None:
        self.assertTrue(stalled(0, 360, MOTION_STALL_FRAMES))
        self.assertFalse(stalled(0, 359, MOTION_STALL_FRAMES))

    def test_goal_stall_window(self) -> None:
        self.assertTrue(stalled(0, 24000, GOAL_STALL_FRAMES))


if __name__ == "__main__":
    unittest.main()

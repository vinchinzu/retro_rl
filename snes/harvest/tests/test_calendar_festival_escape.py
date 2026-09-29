"""Festival roam-and-talk escape (Star Night / New Year forced festivals)."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from harvest.core.tile_catalog import ADDR_TILEMAP
from harvest.planner.multi_day_planner import MultiDayPlannerTask
from harvest.planner.tasks.festival_escape import FESTIVAL_MAPS, FestivalEscapeTask
from harvest.tasks.nav import make_action
from retro_harness import ActionResult, TaskResult, TaskStatus


def _world(tilemap: int):
    ram = np.zeros(0x24000, dtype=np.uint8)
    ram[ADDR_TILEMAP] = tilemap
    return SimpleNamespace(ram=ram)


class FestivalEscapeTests(unittest.TestCase):
    def test_scripted_phase_then_logged_random_fallback(self) -> None:
        import contextlib
        import io

        t = FestivalEscapeTask(scripted_budget=50, seed=1)
        w = _world(0x0A)
        t.reset(w)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            for _ in range(120):
                self.assertEqual(t.step(w).status, TaskStatus.RUNNING)
        self.assertIn("random roam fallback", buf.getvalue())
        self.assertEqual(buf.getvalue().count("fallback"), 1)

    def test_runs_and_is_deterministic(self) -> None:
        a, b = FestivalEscapeTask(seed=3), FestivalEscapeTask(seed=3)
        w = _world(0x0A)
        a.reset(w)
        b.reset(w)
        for _ in range(500):
            ra, rb = a.step(w), b.step(w)
            self.assertEqual(ra.status, TaskStatus.RUNNING)
            self.assertTrue((ra.action.action == rb.action.action).all())

    def test_succeeds_once_the_map_changes(self) -> None:
        t = FestivalEscapeTask()
        t.reset(_world(0x0A))
        self.assertEqual(t.step(_world(0x15)).status, TaskStatus.SUCCESS)

    def test_new_year_and_star_night_maps_covered(self) -> None:
        self.assertIn(0x3A, FESTIVAL_MAPS)
        self.assertIn(0x0A, FESTIVAL_MAPS)

    def test_planner_takes_over_on_a_festival_map_only(self) -> None:
        task = MultiDayPlannerTask()
        task._sync_festival_phase(_world(0x15))
        self.assertNotEqual(task._phase, "festival")
        self.assertIsNone(task._festival_escape)
        task._sync_festival_phase(_world(0x0A))
        self.assertEqual(task._phase, "festival")
        result = task._current_task.step(_world(0x0A))
        self.assertEqual(result.status, TaskStatus.RUNNING)
        task._sync_festival_phase(_world(0x15))
        self.assertNotEqual(task._phase, "festival")
        self.assertIsNone(task._festival_escape)
        self.assertEqual(task._phase, "plan_day")

    def test_failed_escape_falls_through_to_the_interrupted_day(self) -> None:
        task = MultiDayPlannerTask()
        seen = {"n": 0}

        class _Day:
            def step(self, world):
                seen["n"] += 1
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        task._phase = "return_home"
        task._current_task = _Day()
        task._sync_festival_phase(_world(0x0A))
        task._festival_escape.timeout = 0
        result = task._step_inner(_world(0x0A))
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "return_home")
        self.assertEqual(seen["n"], 1)
        self.assertIs(task._current_task.__class__, _Day)


if __name__ == "__main__":
    unittest.main()

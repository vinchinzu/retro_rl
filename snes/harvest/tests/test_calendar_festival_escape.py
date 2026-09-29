"""Festival roam-and-talk escape (Star Night / New Year forced festivals)."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from harvest.core.tile_catalog import ADDR_TILEMAP
from harvest.planner.multi_day_planner import MultiDayPlannerTask
from harvest.planner.tasks.festival_escape import FESTIVAL_MAPS, FestivalEscapeTask
from retro_harness import TaskStatus


def _world(tilemap: int):
    ram = np.zeros(0x24000, dtype=np.uint8)
    ram[ADDR_TILEMAP] = tilemap
    return SimpleNamespace(ram=ram)


class FestivalEscapeTests(unittest.TestCase):
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
        self.assertIsNone(task._step_festival_escape(_world(0x15)))
        result = task._step_festival_escape(_world(0x0A))
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNone(task._step_festival_escape(_world(0x15)))
        self.assertIsNone(task._festival_escape)


if __name__ == "__main__":
    unittest.main()

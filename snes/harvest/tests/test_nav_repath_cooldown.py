"""A sealed nav goal must not cost two failed searches every frame.

run19 D6: once the crop pocket filled in, NAV_CROP could not path to
(15,29). Both `find_path` and `find_frontier_path` returned empty, the
navigator's path stayed empty, and the next frame paid for both searches
again -- for up to the 9000f phase timeout. The run crawled to ~66 f/s
against a ~344 f/s norm.
"""
from __future__ import annotations

import unittest

import numpy as np

from harvest.core.tile_catalog import ADDR_INPUT_LOCK, ADDR_MAP, ADDR_TILEMAP
from harvest.tasks.nav import MAP_WIDTH, Point
from harvest.planner.tasks.navigation import (
    NAV_REPATH_COOLDOWN_FRAMES,
    NavTask,
)
from retro_harness import TaskStatus, WorldState

from tests.crop_planter_test_helpers import set_player_tile, set_tile

_SOLID = 0x05
_PLAYER = (10, 10)


def _sealed_world() -> WorldState:
    """Player boxed in by solid tiles, so no goal is ever reachable.

    Only the tile-map region is filled -- writing 0x05 across all of RAM also
    sets the shipping-scene flag, and NavTask short-circuits into dismissing
    that before it ever reaches the repath block.
    """
    ram = np.zeros(ADDR_MAP + MAP_WIDTH * MAP_WIDTH, dtype=np.uint8)
    ram[ADDR_MAP : ADDR_MAP + MAP_WIDTH * MAP_WIDTH] = _SOLID
    ram[ADDR_TILEMAP] = 0x00
    ram[ADDR_INPUT_LOCK] = 1  # unlocked; NavTask idles on a locked input
    set_tile(ram, _PLAYER[0], _PLAYER[1], 0x00)
    set_player_tile(ram, _PLAYER)
    return WorldState(frame=0, ram=ram, info={}, obs=None)


class RepathCooldownTests(unittest.TestCase):
    def _task(self) -> NavTask:
        task = NavTask(name="nav_sealed", target_px=Point(248, 472), radius=8, timeout=9000)
        world = _sealed_world()
        task.reset(world)
        return task

    def test_a_failed_search_arms_the_cooldown(self) -> None:
        task = self._task()
        world = _sealed_world()
        task.step(world)
        self.assertGreater(task._repath_cooldown, 0)

    def test_cooldown_frames_do_not_research(self) -> None:
        task = self._task()
        world = _sealed_world()
        task.step(world)

        calls = {"n": 0}
        real = task._pathfinder.find_path

        def counting(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        task._pathfinder.find_path = counting
        for _ in range(NAV_REPATH_COOLDOWN_FRAMES):
            result = task.step(world)
            self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(calls["n"], 0, "no search may run during the cooldown")

    def test_the_task_still_moves_while_cooling_down(self) -> None:
        task = self._task()
        world = _sealed_world()
        task.step(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action, "must still emit a fallback action")

    def test_search_resumes_after_the_cooldown(self) -> None:
        task = self._task()
        world = _sealed_world()
        task.step(world)

        calls = {"n": 0}
        real = task._pathfinder.find_path

        def counting(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        task._pathfinder.find_path = counting
        for _ in range(NAV_REPATH_COOLDOWN_FRAMES + 1):
            task.step(world)
        self.assertEqual(calls["n"], 1, "exactly one search once the cooldown lapses")

    def test_reset_clears_the_cooldown(self) -> None:
        task = self._task()
        world = _sealed_world()
        task.step(world)
        task.reset(world)
        self.assertEqual(task._repath_cooldown, 0)


if __name__ == "__main__":
    unittest.main()

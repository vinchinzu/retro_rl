"""A wedged walk must end in seconds, never on the 9,000-frame timeout.

rec1_20261114: 16 NAV_CROP phases ran the whole 9,000f timeout with the farmer inside a few pixels
(Y1 D12-13 and Summer D4-10 from 06:00, Y2/Y3 at 18:00). NavTask and MultiMapNavTask now watch a 24 px
box: one clean repath after NAV_WEDGE_FRAMES, then FAILURE.
"""
from __future__ import annotations

import unittest

import numpy as np

from harvest.core.task_progress import (
    NAV_NO_PROGRESS_NEAR_FRAMES,
    NAV_WAYPOINT_NO_PROGRESS_FRAMES,
    NAV_WEDGE_FRAMES,
    MotionBox,
)
from harvest.core.tile_catalog import ADDR_INPUT_LOCK, ADDR_MAP, ADDR_TILEMAP
from harvest.maps.map_config import Waypoint
from harvest.planner.tasks.multi_nav import MultiMapNavTask
from harvest.planner.tasks.navigation import NavTask
from harvest.tasks.nav import ADDR_X, MAP_WIDTH, Point
from retro_harness import TaskStatus, WorldState

from tests.crop_planter_test_helpers import set_player_tile, set_tile

_SOLID = 0x05


def _world(player=(10, 10), solid=True) -> WorldState:
    ram = np.zeros(ADDR_MAP + MAP_WIDTH * MAP_WIDTH, dtype=np.uint8)
    if solid:
        ram[ADDR_MAP : ADDR_MAP + MAP_WIDTH * MAP_WIDTH] = _SOLID
    ram[ADDR_TILEMAP] = 0x00
    ram[ADDR_INPUT_LOCK] = 1
    set_tile(ram, player[0], player[1], 0x00)
    set_player_tile(ram, player)
    return WorldState(frame=0, ram=ram, info={}, obs=None)


class MotionBoxTests(unittest.TestCase):
    def test_counts_while_inside_the_box_and_resets_on_leaving_or_a_map_change(self) -> None:
        box = MotionBox()
        self.assertEqual(box.observe(0, 0, 100, 100), 0)
        self.assertEqual(box.observe(50, 0, 110, 90), 50)       # 10 px: still inside
        self.assertEqual(box.observe(60, 0, 140, 100), 0)       # 40 px: a new box
        self.assertEqual(box.observe(90, 0, 140, 100), 30)
        self.assertEqual(box.observe(95, 1, 140, 100), 0)       # map change
        box.restart(200)
        self.assertEqual(box.observe(260, 1, 141, 100), 60)


class NavTaskWedgeTests(unittest.TestCase):
    def test_each_new_stall_earns_its_own_repath_after_the_farmer_moves(self) -> None:
        task = NavTask(name="nav_NAV_CROP", target_px=Point(728, 440), radius=8, timeout=9000)
        world = _world(solid=False)
        task.reset(world)
        for _ in range(NAV_WEDGE_FRAMES + 5):
            self.assertNotIn("wedged", task.step(world).reason or "")
        self.assertEqual(task._unwedges, 1)
        set_player_tile(world.ram, (30, 10))  # real movement: a new box
        task.step(world)
        self.assertEqual(task._unwedges, 0)

    def test_a_sealed_farmer_gets_one_repath_then_fails_long_before_the_timeout(self) -> None:
        task = NavTask(name="nav_NAV_CROP", target_px=Point(728, 440), radius=8, timeout=9000)
        world = _world()
        task.reset(world)
        result = None
        for step in range(9000):
            result = task.step(world)
            if result.status != TaskStatus.RUNNING:
                break
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("nav wedged", result.reason)
        self.assertLessEqual(step, 2 * NAV_WEDGE_FRAMES + 30)
        self.assertEqual(task._unwedges, 1)

    def test_circling_a_target_it_cannot_reach_fails_on_no_progress(self) -> None:
        """rec1 Y1 D12: 30-38 px from the plot centre for 9,000 f, never inside one 24 px box."""
        task = NavTask(name="nav_NAV_CROP", target_px=Point(216, 456), radius=8, soft_radius=20, timeout=9000)
        world = _world(solid=False)
        task.reset(world)
        spots = [(214, 494), (246, 462), (216, 494)]  # 38, 30, 38 px from the target
        result = None
        for step in range(3000):
            x, y = spots[(step // 100) % 3]
            for addr, v in ((ADDR_X, x), (ADDR_X + 2, y)):
                world.ram[addr] = v & 0xFF
                world.ram[addr + 1] = (v >> 8) & 0xFF
            result = task.step(world)
            if result.status != TaskStatus.RUNNING:
                break
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("no progress", result.reason)
        self.assertLessEqual(step, 100 + NAV_NO_PROGRESS_NEAR_FRAMES + 30)

    def test_a_stalled_walk_inside_the_soft_radius_arrives_instead_of_failing(self) -> None:
        """Det1 (2026-11-15): the shed-shelf walk stood 14 px from its stand every day and the old code soft-arrived.

        The guard failed it ("nav no progress") and cost the snow-flower berry on 57 mornings.
        """
        task = NavTask(name="nav_shed_shelf_0x17", target_px=Point(48, 185), radius=4, timeout=3000)
        world = _world(player=(3, 11), solid=False)
        task.reset(world)
        pos = [(62, 183), (66, 183)]  # wobbling across the tile line at x=64, 14-18 px from the stand
        result = None
        for step in range(3000):
            x, y = pos[(step // 20) % 2]
            for addr, v in ((ADDR_X, x), (ADDR_X + 2, y)):
                world.ram[addr] = v & 0xFF
                world.ram[addr + 1] = (v >> 8) & 0xFF
            result = task.step(world)
            if result.status != TaskStatus.RUNNING:
                break
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("soft arrived", result.reason)
        self.assertLessEqual(step, NAV_WEDGE_FRAMES + 30)

    def test_a_farmer_who_keeps_walking_is_never_called_wedged(self) -> None:
        task = NavTask(name="nav_far", target_px=Point(900, 168), radius=8, timeout=9000)
        world = _world(player=(5, 10), solid=False)
        task.reset(world)
        x = 5 * 16 + 8
        for _ in range(2 * NAV_WEDGE_FRAMES):
            result = task.step(world)
            self.assertNotIn("wedged", result.reason or "")
            x += 1  # one pixel a frame, slower than any real walk
            world.ram[ADDR_X] = x & 0xFF
            world.ram[ADDR_X + 1] = (x >> 8) & 0xFF


class MultiMapNavWedgeTests(unittest.TestCase):
    def test_a_wiggling_walk_gets_one_repath_then_fails(self) -> None:
        """A 6 px wiggle resets the pixel-stuck count (it needs under 4 px), so only the box catches it.

        Drives ``_wedge_check`` directly: a static fake RAM also trips the corridor's own no-path failure
        first, which is not what this test is about.
        """
        wp = [Waypoint(tilemap=0x00, target_px=(300, 168), radius=8)]
        task = MultiMapNavMapTask(wp)
        world = _world(solid=False)
        task.reset(world)
        base = 10 * 16 + 8
        results = []
        for step in range(1, 2 * NAV_WEDGE_FRAMES + 40):
            x = base + (6 if (step // 20) % 2 else 0)
            world.ram[ADDR_X] = x & 0xFF
            world.ram[ADDR_X + 1] = (x >> 8) & 0xFF
            task._navigator.update(world.ram)
            task._step_count = step
            result = task._wedge_check(world)
            if result is not None:
                results.append((step, result))
                break
        self.assertTrue(results, "the wiggle must end in a FAILURE")
        step, result = results[0]
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("nav wedged", result.reason)
        self.assertEqual(task._unwedges, 1)
        self.assertGreaterEqual(step, 2 * NAV_WEDGE_FRAMES)
        self.assertLessEqual(step, 2 * NAV_WEDGE_FRAMES + 30)

    def test_a_wide_wobble_that_never_nears_the_waypoint_fails_on_no_progress(self) -> None:
        """fdet_20261115: the exit walk from the fishing camp swung 50 px up and down for 11,001 f, outside any 24 px box."""
        wp = [Waypoint(tilemap=0x00, target_px=(300, 168), radius=8)]
        task = MultiMapNavMapTask(wp)
        world = _world(solid=False)
        task.reset(world)
        base_x, ys = 10 * 16 + 8, (10 * 16 + 8, 10 * 16 + 58)
        failed = None
        for step in range(1, NAV_WAYPOINT_NO_PROGRESS_FRAMES + 200):
            y = ys[(step // 120) % 2]  # 50 px swings every 120 f: never 600 f inside one box
            for addr, v in ((ADDR_X, base_x), (ADDR_X + 2, y)):
                world.ram[addr] = v & 0xFF
                world.ram[addr + 1] = (v >> 8) & 0xFF
            task._navigator.update(world.ram)
            task._step_count = step
            result = task._wedge_check(world)
            if result is not None:
                failed = (step, result)
                break
        self.assertIsNotNone(failed)
        step, result = failed
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("nav no progress", result.reason)
        self.assertGreaterEqual(step, NAV_WAYPOINT_NO_PROGRESS_FRAMES)
        self.assertLessEqual(step, NAV_WAYPOINT_NO_PROGRESS_FRAMES + 30)


def MultiMapNavMapTask(waypoints):
    return MultiMapNavTask(name="route", waypoints=waypoints, timeout=4000, initial_settle_frames=0)


if __name__ == "__main__":
    unittest.main()

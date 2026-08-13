"""CoopChoresTask navigation / routing tests — egg routes, ship nav, pathfinding."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from coop_task_test_helpers import (  # noqa: E402
    add_chicken_object,
    add_egg_object,
    block_tiles,
    make_coop_ram,
    make_world,
    set_chicken_slot_position,
)
from harvest.tasks.coop_task import (  # noqa: E402
    COOP_FALSE_OPEN_COLUMN_X,
    COOP_MAIN_AISLE_TOP,
    CoopChoresTask,
    EXIT_PREP_ESCAPE_ROUTE,
    MAX_EXIT_PREP_FRAMES,
    SHIP_BIN_INTERACT_STAND,
    SHIP_BIN_STAND,
)
from harvest.core.tile_catalog import ADDR_X, ADDR_Y  # noqa: E402
from retro_harness import TaskStatus  # noqa: E402


class CoopChoresTaskNavTests(unittest.TestCase):
    """Routing, obstacle blocking, feed-bin approach, exit_prep, false-open tiles."""

    def test_extended_egg_flags_use_spawn_table_pickup_spots(self):
        ram = make_coop_ram(
            adults=0,
            hay=50,
            egg_available=0x001C,
            incubating=True,
            player_tile=(13, 9),
        )
        add_egg_object(ram, (14, 11))
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._navigator.update(ram)

        self.assertEqual(task._phase, "egg_nav")
        self.assertEqual(task._egg_tiles(ram), [])
        # Prefer below-egg stand: (5,9) sits on the false-open column.
        self.assertEqual(task._egg_pickup_spot(ram), ((4, 10), "up"))

    def test_lower_egg_route_does_not_force_top_aisle(self):
        ram = make_coop_ram(
            adults=0,
            hay=50,
            egg_available=0x001C,
            player_tile=(13, 9),
        )
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._navigator.update(ram)

        route = task._left_top_route((4, 10))

        self.assertEqual(route, ((4, 10),))
        self.assertNotIn(COOP_MAIN_AISLE_TOP, route)

    def test_lower_egg_route_from_ship_bin_uses_left_lane(self):
        ram = make_coop_ram(
            adults=0,
            hay=50,
            egg_available=0x001C,
            player_tile=(1, 10),
        )
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._navigator.update(ram)

        route = task._left_top_route((4, 10))

        self.assertEqual(route, ((2, 10), (2, 9), (3, 9), (4, 10)))
        self.assertNotIn(COOP_MAIN_AISLE_TOP, route)

    def test_flag01_egg_avoids_unreachable_default_stand(self):
        """Regression: stand (2,4) is an island when (2,5)/(2,3)/(3,4) are walls."""
        ram = make_coop_ram(
            adults=0,
            hay=0,
            egg_available=0x0001,
            player_tile=(1, 10),
        )
        block_tiles(ram, [(2, 3), (2, 5), (3, 4)])
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._navigator.update(ram)

        spot = task._egg_pickup_spot(ram)

        self.assertIsNotNone(spot)
        self.assertNotEqual(spot[0], (2, 4))
        self.assertIn(spot[0], {(0, 4), (1, 5)})
        self.assertEqual(task._current_egg_flag, 0x0001)

    def test_upper_egg_route_from_ship_climbs_service_lane(self):
        ram = make_coop_ram(
            adults=0,
            hay=0,
            egg_available=0x0001,
            player_tile=(1, 10),
        )
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._navigator.update(ram)

        route = task._left_top_route((0, 4))

        self.assertEqual(route[0], (0, 6))
        self.assertEqual(route[-1], (0, 4))

    def test_ship_nav_sidesteps_left_from_approach_stand(self):
        ram = make_coop_ram(adults=1, holding_egg=True, player_tile=SHIP_BIN_STAND, shipping_money=0)
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._phase = "ship_nav"
        task._navigator.update(ram)

        result = task.step(make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "ship_nav")
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[6]), 1)  # left

    def test_ship_nav_uses_recorded_left_lane_from_egg_area(self):
        ram = make_coop_ram(adults=1, holding_egg=True, player_tile=(2, 5), shipping_money=0)
        ram[ADDR_X] = 38
        ram[ADDR_X + 1] = 0
        ram[ADDR_Y] = 85
        ram[ADDR_Y + 1] = 0
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._phase = "ship_nav"
        task._navigator.update(ram)

        result = task.step(make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[5]), 1)
        self.assertEqual(int(result.action.action[0]), 1)

    def test_ship_nav_from_mid_coop_paths_before_pixel_lane(self):
        ram = make_coop_ram(
            adults=0,
            egg_available=0x0030,
            holding_egg=True,
            player_tile=(6, 7),
            shipping_money=0,
        )
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._phase = "ship_nav"
        task._navigator.update(ram)

        result = task.step(make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "ship_nav")
        self.assertIsNotNone(result.action)
        self.assertTrue(task._navigator.path)
        self.assertEqual(task._navigator.path[-1], SHIP_BIN_STAND)
        # Far approach uses coop_nav_to_shipping_bin_skill
        self.assertIsNotNone(task._active_skill)
        self.assertEqual(task._active_skill.name, "coop_nav_ship_bin")

    def test_ship_nav_from_lower_right_corner_avoids_bin_corner_dead_edge(self):
        ram = make_coop_ram(
            adults=0,
            egg_available=0x0020,
            holding_egg=True,
            player_tile=(3, 11),
            shipping_money=0,
        )
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._phase = "ship_nav"
        task._navigator.update(ram)
        task._navigator.path = [(2, 11), (2, 10)]

        result = task.step(make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "ship_nav")
        self.assertIsNotNone(result.action)
        self.assertTrue(task._navigator.path)
        self.assertEqual(task._navigator.path[-1], (3, 10))

    def test_ship_nav_from_row_ten_uses_right_lane_corner(self):
        ram = make_coop_ram(
            adults=0,
            egg_available=0x0038,
            holding_egg=True,
            player_tile=(5, 10),
            shipping_money=0,
        )
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._phase = "ship_nav"
        task._navigator.update(ram)
        task._navigator.path = [(5, 11), (4, 11), (3, 11), (2, 11), (2, 10)]

        result = task.step(make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "ship_nav")
        self.assertIsNotNone(result.action)
        self.assertTrue(task._navigator.path)
        self.assertEqual(task._navigator.path[-1], (3, 10))

    def test_ship_nav_from_right_lane_corner_uses_pixel_slide(self):
        ram = make_coop_ram(
            adults=0,
            egg_available=0x0038,
            holding_egg=True,
            player_tile=(3, 10),
            shipping_money=0,
        )
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._phase = "ship_nav"
        task._navigator.update(ram)

        result = task.step(make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "ship_nav")
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[6]), 1)  # slide left toward bin
        self.assertEqual(int(result.action.action[0]), 1)

    def test_ship_nav_queues_press_from_interact_stand(self):
        ram = make_coop_ram(adults=1, holding_egg=True, player_tile=SHIP_BIN_INTERACT_STAND, shipping_money=0)
        ram[ADDR_X] = 22
        ram[ADDR_X + 1] = 0
        ram[ADDR_Y] = 169
        ram[ADDR_Y + 1] = 0
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._phase = "ship_nav"
        task._navigator.update(ram)

        result = task.step(make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "ship_verify")
        self.assertGreater(len(task._action_queue), 0)
        first = task._action_queue[0]
        self.assertEqual(int(first[5]), 1)  # face down into the egg bin
        self.assertEqual(int(first[4]), 0)
        second = task._action_queue[1]
        self.assertEqual(int(second[8]), 1)  # press A without walking
        self.assertEqual(int(second[5]), 0)

    def test_navigation_routes_around_live_chicken_object(self):
        ram = make_coop_ram(adults=1, egg_available=False, player_tile=(2, 7))
        add_chicken_object(ram, (3, 7))
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._navigator.update(ram)

        action = task._navigate_to_tile(ram, (4, 7))

        self.assertIsNotNone(action)
        self.assertNotIn((3, 7), task._navigator.path)

    def test_navigation_blocks_adult_chicken_slot_without_live_object(self):
        ram = make_coop_ram(adults=1, egg_available=False, player_tile=(2, 7))
        set_chicken_slot_position(ram, 0, (3, 7))
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._navigator.update(ram)

        action = task._navigate_to_tile(ram, (4, 7))

        self.assertIsNotNone(action)
        self.assertNotIn((3, 7), task._navigator.path)

    def test_navigation_blocks_egg_slot_without_live_object(self):
        ram = make_coop_ram(adults=0, slot_eggs=1, egg_available=False, player_tile=(13, 11))
        set_chicken_slot_position(ram, 0, (14, 11))
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._navigator.update(ram)

        action = task._navigate_to_tile(ram, (15, 11))

        self.assertIsNotNone(action)
        self.assertNotIn((14, 11), task._navigator.path)

    def test_navigation_blocks_visible_egg_object(self):
        ram = make_coop_ram(adults=0, egg_available=False, player_tile=(13, 11))
        add_egg_object(ram, (14, 11))
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._navigator.update(ram)

        action = task._navigate_to_tile(ram, (15, 11))

        self.assertIsNotNone(action)
        self.assertNotIn((14, 11), task._navigator.path)

    def test_navigation_blocks_flagged_egg_spawn_tiles(self):
        ram = make_coop_ram(adults=0, egg_available=0x0002, player_tile=(4, 5))
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._navigator.update(ram)

        action = task._navigate_to_tile(ram, (2, 4))

        self.assertIsNotNone(action)
        self.assertNotIn((3, 5), task._navigator.path)

    def test_feed_nav_from_entrance_uses_center_aisle(self):
        ram = make_coop_ram(adults=2, hay=50, egg_available=True, player_tile=(8, 12))
        task = CoopChoresTask()
        task.reset(make_world(ram))

        result = task.step(make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[4]), 1)
        self.assertEqual(int(result.action.action[6]), 0)
        self.assertEqual(task._left_top_route_points[:2], ((8, 6), (4, 5)))
        # Production path: feed_nav steps coop_nav_to_feed_bin_skill
        self.assertIsNotNone(task._active_skill)
        self.assertEqual(task._active_skill.name, "coop_nav_feed_bin")
        snap = task.progress_snapshot()
        self.assertEqual(snap.phase_text, "feed_nav")
        self.assertIsNotNone(snap.child)
        self.assertEqual(snap.child.task_name, "coop_nav_feed_bin")

    def test_feed_nav_recovers_from_lower_left_coop_corner(self):
        ram = make_coop_ram(adults=2, hay=50, egg_available=True, player_tile=(2, 12))
        task = CoopChoresTask()
        task.reset(make_world(ram))

        result = task.step(make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[7]), 1)
        self.assertEqual(int(result.action.action[4]), 0)
        self.assertEqual(task._left_top_route_points[:2], ((8, 12), (8, 6)))
        self.assertIsNotNone(task._active_skill)
        self.assertEqual(task._active_skill.name, "coop_nav_feed_bin")

    def test_coop_navigation_strictly_centers_before_vertical_step(self):
        ram = make_coop_ram(adults=0, egg_available=False, player_tile=(2, 12))
        ram[ADDR_X] = 42
        ram[ADDR_Y] = 198
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._navigator.update(ram)
        task._navigator.path = [(2, 11), (2, 10)]

        action = task._navigate_to_tile(ram, (2, 6))

        self.assertIsNotNone(action)
        self.assertEqual(int(action[6]), 1)
        self.assertEqual(int(action[4]), 0)

    def test_navigation_does_not_block_baby_chick_slot(self):
        ram = make_coop_ram(adults=0, chicks=1, egg_available=False, player_tile=(2, 7))
        set_chicken_slot_position(ram, 0, (3, 7))
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._navigator.update(ram)

        action = task._navigate_to_tile(ram, (4, 7))

        self.assertIsNotNone(action)
        self.assertIn((3, 7), task._navigator.path)

    def test_navigation_waits_when_goal_is_occupied_by_chicken(self):
        ram = make_coop_ram(adults=1, egg_available=False, player_tile=(2, 7))
        add_chicken_object(ram, (4, 7))
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._navigator.update(ram)

        action = task._navigate_to_tile(ram, (4, 7))

        self.assertIsNotNone(action)
        self.assertEqual(int(np.sum(action)), 0)
        self.assertEqual(task._navigator.path, [])

    def test_exit_prep_escapes_false_open_column_instead_of_stalling(self):
        """Regression: long runs stuck at (5,11)/(86,183) in exit_prep_nav."""
        ram = make_coop_ram(adults=0, hay=0, egg_available=False, player_tile=(5, 11))
        task = CoopChoresTask()
        task.reset(make_world(ram))
        self.assertEqual(task._phase, "exit_prep_nav")
        self.assertEqual(COOP_FALSE_OPEN_COLUMN_X, 5)
        self.assertEqual(EXIT_PREP_ESCAPE_ROUTE[0], (5, 12))

        result = task.step(make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        # Climb above the false-open band before crossing east (B = run).
        self.assertEqual(int(result.action.action[4]), 1)
        self.assertEqual(int(result.action.action[0]), 1)

    def test_exit_prep_timeout_hands_off_instead_of_watchdog(self):
        ram = make_coop_ram(adults=0, hay=0, egg_available=False, player_tile=(5, 11))
        task = CoopChoresTask()
        task.reset(make_world(ram))
        task._exit_prep_started_step = 1
        task._step_count = MAX_EXIT_PREP_FRAMES + 2

        result = task.step(make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "done")

    def test_false_open_tiles_are_blocked_in_pathfinding(self):
        ram = make_coop_ram(adults=0, hay=0, egg_available=False, player_tile=(2, 11))
        task = CoopChoresTask()
        task.reset(make_world(ram))
        blocked = task._coop_false_open_tiles()
        self.assertIn((5, 11), blocked)
        path = task._find_path_around_chickens(ram, (2, 11), (8, 12))
        if path is not None:
            self.assertNotIn((5, 11), path)


if __name__ == "__main__":
    unittest.main()

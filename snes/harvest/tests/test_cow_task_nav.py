"""CowChoresTask navigation/geometry/shipping tests.

Split from test_cow_task monofile.
"""
from __future__ import annotations

from pathlib import Path
import sys

# Path-stable import of sibling helpers (works under unittest and pytest importlib mode).
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
from cow_test_helpers import (
    _make_barn_ram,
    _make_world,
    _set_cow_daily,
    _set_cow_slot,
    _set_player_px,
    _set_player_tile,
    _write_u16,
    _write_u24,
)

import unittest
from unittest.mock import patch

from harvest.tasks.cow_task import (
    ADDR_TOOL_BACKPACK,
    ADDR_TOOL_SELECTED,
    ADDR_FED_COWS_N,
    ADDR_HELD_ITEM,
    ADDR_NUM_COWS,
    ADDR_PLAYER_ACTION,
    ADDR_STORED_GRASS,
    BRUSH_TOOL_ID,
    MILKER_TOOL_ID,
    BARN_SHIP_BIN_INTERACT_STAND,
    BARN_SHIP_BIN_STAND,
    COW_DAILY_BRUSHED_FLAG,
    COW_DAILY_MILKED_FLAG,
    COW_DAILY_TALKED_FLAG,
    COW_FEED_SPOTS,
    COW_INTERACT_X_OFFSET,
    COW_LEFT_INTERACT_X,
    COW_TALK_ROUTE,
    COW_TALK_STAND,
    FEED_TROUGH_INTERACT_PX,
    FEED_TROUGH_STAND,
    FEED_TROUGH_ROUTE,
    FODDER_STAND,
    FODDER_ROUTE,
    FODDER_TROUGH_ROUTE,
    ITEM_FODDER,
    COW_EXIT_PREP_STAND,
    MAX_CARE_DEFERRALS,
    MAX_COW_SLOT_CARE_FRAMES,
    MAX_COW_SLOT_MILK_FRAMES,
    MAX_COW_NAV_FAILURES,
    MAX_PIXEL_NAV_STALLS,
    MAX_TALK_ATTEMPTS,
    MILK_SHIP_ROUTE,
    PIXEL_NAV_STALL_FRAMES,
    CowChoresTask,
)
from harvest.core.animal_status import ADDR_FED_COWS_FLAGS, COW_STATUS_BABY_FLAG
from harvest.core.tile_catalog import (
    ADDR_INPUT_LOCK,
    ADDR_MAP,
    ADDR_TILEMAP,
    ADDR_X,
    ADDR_Y,
)
from harvest.tasks.nav import make_action
from retro_harness import TaskStatus


class CowChoresNavTests(unittest.TestCase):
    def test_talk_nav_exits_left_trough_before_crossing_stall(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20)
        _set_player_px(ram, COW_FEED_SPOTS[3].interact_px)
        _set_cow_slot(ram, 0, (9, 17))
        task = CowChoresTask(brush=False, milk=False, feed=False)
        task.reset(_make_world(ram))

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[5]), 1)
        self.assertEqual(int(result.action.action[7]), 0)
        self.assertEqual(int(result.action.action[0]), 1)

    def test_brush_nav_exits_left_trough_before_crossing_stall(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, tool_selected=BRUSH_TOOL_ID)
        _set_player_px(ram, COW_FEED_SPOTS[3].interact_px)
        _set_cow_slot(ram, 0, (9, 17))
        _set_cow_daily(ram, 0, flags=COW_DAILY_TALKED_FLAG, happiness=1)
        task = CowChoresTask(milk=False, feed=False)
        task.reset(_make_world(ram))

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[5]), 1)
        self.assertEqual(int(result.action.action[7]), 0)
        self.assertEqual(int(result.action.action[0]), 1)

    def test_milk_ship_nav_uses_barn_bin_interact_tile(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, held_item=0x16, player_tile=BARN_SHIP_BIN_STAND)
        task = CowChoresTask(talk=False, brush=False, milk=False, feed=False)
        task.reset(_make_world(ram))
        task._phase = "milk_ship_nav"

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNone(result.action)
        self.assertEqual(task._phase, "milk_ship_verify")

    def test_milk_ship_nav_uses_lower_aisle_before_crossing_left(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, held_item=0x17, player_tile=(11, 15))
        task = CowChoresTask(talk=False, brush=False, milk=False, feed=False)
        task.reset(_make_world(ram))
        task._phase = "milk_ship_nav"

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[5]), 1)
        self.assertEqual(int(result.action.action[6]), 0)

    def test_milk_ship_nav_uses_right_aisle_before_dropping_from_upper_stalls(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, held_item=0x17, player_tile=(11, 10))
        task = CowChoresTask(talk=False, brush=False, milk=False, feed=False)
        task.reset(_make_world(ram))
        task._phase = "milk_ship_nav"

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[7]), 1)
        self.assertEqual(int(result.action.action[5]), 0)
        self.assertEqual(int(result.action.action[6]), 0)

    def test_milk_ship_nav_uses_recorded_right_aisle_from_x9_gap(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, held_item=0x17, player_tile=(9, 11))
        task = CowChoresTask(talk=False, brush=False, milk=False, feed=False)
        task.reset(_make_world(ram))
        task._phase = "milk_ship_nav"

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[7]), 1)
        self.assertEqual(int(result.action.action[0]), 1)

    def test_milk_ship_nav_does_not_cross_left_at_blocked_stall_row(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, held_item=0x17, player_tile=(11, 12))
        task = CowChoresTask(talk=False, brush=False, milk=False, feed=False)
        task.reset(_make_world(ram))
        task._phase = "milk_ship_nav"

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[5]), 1)
        self.assertEqual(int(result.action.action[6]), 0)

    def test_milk_ship_nav_escapes_cow_interaction_pin(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, held_item=0x15, player_tile=(10, 17))
        task = CowChoresTask(talk=False, brush=False, milk=False, feed=False)
        task.reset(_make_world(ram))
        task._phase = "milk_ship_nav"

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[7]), 1)
        self.assertEqual(int(result.action.action[0]), 1)

        _set_player_tile(ram, (11, 18))
        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[5]), 1)
        self.assertEqual(int(result.action.action[0]), 1)

    def test_milk_ship_verify_finishes_after_money_increases(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, held_item=0x16, player_tile=BARN_SHIP_BIN_INTERACT_STAND)
        task = CowChoresTask(talk=False, brush=False, milk=False, feed=False)
        task.reset(_make_world(ram))
        task._phase = "milk_ship_verify"
        task._ship_money_before = 0
        _write_u24(ram, 0x11F07, 8)

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "exit_prep_nav")
        self.assertEqual(task.milk_shipped_count, 1)

    def test_milk_ship_verify_counts_inside_barn_crate_drop_when_money_lags(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, held_item=0, player_tile=BARN_SHIP_BIN_INTERACT_STAND)
        task = CowChoresTask(talk=False, brush=False, milk=False, feed=False)
        task.reset(_make_world(ram))
        task._phase = "milk_ship_verify"
        task._ship_money_before = 0

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "exit_prep_nav")
        self.assertEqual(task.milk_shipped_count, 1)

    def test_talk_nav_uses_recorded_left_lane_from_barn_bin(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, player_tile=BARN_SHIP_BIN_STAND)
        _set_player_px(ram, (38, 361))
        _set_cow_slot(ram, 0, (9, 17))
        task = CowChoresTask(brush=False, milk=False, feed=False)
        task.reset(_make_world(ram))

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[7]), 1)
        self.assertEqual(int(result.action.action[0]), 1)

    def test_talk_route_avoids_right_side_cow_tile(self) -> None:
        ram = _make_barn_ram(cows=1, player_tile=(11, 18))
        _set_cow_slot(ram, 0, (11, 17))
        task = CowChoresTask()

        task.reset(_make_world(ram))
        task._talk_route_index = len(COW_TALK_ROUTE) - 1
        result = task.step(_make_world(ram))

        self.assertEqual(task._talk_stand, (12, 17))
        self.assertEqual(task._talk_face, "left")
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[6]), 0)

    def test_talk_approach_uses_accessible_side_for_current_cow_tile(self) -> None:
        ram = _make_barn_ram(cows=1, player_tile=(11, 21))
        task = CowChoresTask()

        _set_cow_slot(ram, 0, (10, 17))
        task.reset(_make_world(ram))
        self.assertEqual(task._talk_stand, (11, 17))
        self.assertEqual(task._talk_face, "left")

        _set_cow_slot(ram, 0, (12, 17))
        task._refresh_talk_approach(ram)
        self.assertEqual(task._talk_stand, (11, 17))
        self.assertEqual(task._talk_face, "right")

    def test_left_face_interact_pixel_keeps_right_side_cow_offset(self) -> None:
        """Regression: clamp to x=163 must not pull through right-side cows."""
        ram = _make_barn_ram(cows=1, player_tile=(14, 14))
        _set_cow_slot(ram, 0, (13, 14))
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._talk_face = "left"

        target = task._cow_interact_pixel(ram, tool=False)

        self.assertIsNotNone(target)
        cow_px = task._target_cow_pixel(ram)
        assert cow_px is not None
        self.assertEqual(target[0], cow_px[0] + COW_INTERACT_X_OFFSET)
        self.assertGreater(target[0], COW_LEFT_INTERACT_X)

    def test_left_face_interact_pixel_still_clamps_left_aisle_cows(self) -> None:
        ram = _make_barn_ram(cows=1, player_tile=(10, 15))
        _set_cow_slot(ram, 0, (9, 15))
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._talk_face = "left"

        target = task._cow_interact_pixel(ram, tool=False)

        self.assertEqual(target, (COW_LEFT_INTERACT_X, 249))

    def test_left_face_interact_pixel_clamps_mid_left_stall_cows(self) -> None:
        ram = _make_barn_ram(cows=1, player_tile=(11, 17))
        _set_cow_slot(ram, 0, (10, 17))
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._talk_face = "left"

        target = task._cow_interact_pixel(ram, tool=True)

        self.assertEqual(target, (COW_LEFT_INTERACT_X, 278))

    def test_talk_approach_prefers_body_side_for_wall_cows(self) -> None:
        ram = _make_barn_ram(cows=1, player_tile=(3, 5))
        _set_cow_slot(ram, 0, (0, 3))
        task = CowChoresTask(brush=False, milk=False, feed=False)
        task.reset(_make_world(ram))

        self.assertEqual(task._talk_stand, (1, 4))
        self.assertEqual(task._talk_face, "left")
        self.assertTrue(task._is_adjacent_to_target_cow(ram, (1, 4), "left"))

    def test_left_side_vertical_nav_stays_on_interact_column(self) -> None:
        ram = _make_barn_ram(cows=1, player_tile=(1, 3))
        _set_player_px(ram, (27, 57))
        _set_cow_slot(ram, 0, (0, 4))
        task = CowChoresTask(brush=False, milk=False, feed=False)
        task.reset(_make_world(ram))
        task._talk_face = "left"
        task._navigator.update(ram)

        action = task._left_side_vertical_nav_action(27, 57, 27, 76, going_down=True)

        self.assertIsNotNone(action)
        self.assertEqual(int(action[6]), 0)  # not left
        self.assertEqual(int(action[7]), 0)  # not right
        self.assertEqual(int(action[5]), 1)  # down toward target

    def test_left_side_vertical_nav_climbs_lane_before_cutting_to_wall(self) -> None:
        ram = _make_barn_ram(cows=1, player_tile=(3, 21))
        _set_player_px(ram, (54, 345))
        _set_cow_slot(ram, 0, (0, 7))
        task = CowChoresTask(brush=False, milk=False, feed=False)
        task.reset(_make_world(ram))
        task._navigator.update(ram)

        action = task._left_side_vertical_nav_action(54, 345, 27, 122, going_down=False)

        self.assertIsNotNone(action)
        self.assertEqual(int(action[6]), 0)  # not left into bottom dead-end
        self.assertEqual(int(action[4]), 1)  # up the left lane first

    def test_care_trough_exit_does_not_block_lower_left_wall_targets(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, player_tile=(8, 21))
        _set_player_px(ram, (129, 345))
        _set_cow_slot(ram, 0, (0, 9))
        task = CowChoresTask(brush=False, milk=False, feed=False)
        task.reset(_make_world(ram))
        task._talk_face = "left"
        task._navigator.update(ram)

        self.assertIsNone(task._care_trough_exit_action(ram))

    def test_talk_recorded_nav_yields_when_already_beside_cow(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, player_tile=(1, 9))
        _set_player_px(ram, (27, 153))
        _set_cow_slot(ram, 0, (2, 8))
        task = CowChoresTask(brush=False, milk=False, feed=False)
        task.reset(_make_world(ram))
        task._talk_stand = (1, 9)
        task._talk_face = "right"
        task._navigator.update(ram)

        self.assertTrue(task._is_adjacent_to_target_cow(ram, (1, 9), "right"))
        self.assertIsNone(task._recorded_interact_nav_action(ram, tool=False))
        self.assertIsNotNone(task._recorded_interact_nav_action(ram, tool=True))

    def test_talk_approach_prefers_pinned_escape_tile(self) -> None:
        ram = _make_barn_ram(cows=1, player_tile=(11, 17))
        _set_cow_slot(ram, 0, (11, 14))
        ram[ADDR_MAP + 14 * 64 + 12] = 0x00
        task = CowChoresTask()

        task.reset(_make_world(ram))

        self.assertEqual(task._talk_stand, (10, 14))
        self.assertEqual(task._talk_face, "right")

    def test_path_around_cows_respects_temporary_blocks(self) -> None:
        ram = _make_barn_ram(cows=1, player_tile=(11, 16))
        _set_cow_slot(ram, 0, (11, 14))
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._pathfinder.temp_blocked.add((10, 16))

        path = task._find_path_around_cows(ram, (11, 16), (10, 14))

        self.assertIsNotNone(path)
        self.assertNotIn((10, 16), path)

    def test_barn_chore_routes_use_recorded_corridors(self) -> None:
        self.assertEqual(COW_TALK_ROUTE, ((11, 21), COW_TALK_STAND))
        self.assertNotIn((11, 17), COW_TALK_ROUTE)
        self.assertEqual(FODDER_ROUTE, ((11, 11), FODDER_STAND))
        self.assertEqual(FODDER_TROUGH_ROUTE, ((9, 11), (11, 11), FODDER_STAND))
        self.assertEqual(FEED_TROUGH_ROUTE, ((9, 11), FEED_TROUGH_STAND))
        self.assertEqual(FEED_TROUGH_INTERACT_PX, (113, 149))
        self.assertEqual(BARN_SHIP_BIN_STAND, (2, 22))


if __name__ == "__main__":
    unittest.main()

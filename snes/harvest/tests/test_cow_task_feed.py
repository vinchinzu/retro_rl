"""CowChoresTask feed/fodder tests.

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


class CowChoresFeedTests(unittest.TestCase):
    def test_fodder_verify_uses_held_item_and_hay_delta(self) -> None:
        ram = _make_barn_ram(cows=1, fed=0, hay=20, player_tile=FODDER_STAND)
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._phase = "fodder_verify"
        task._grass_before = 20

        ram[ADDR_HELD_ITEM] = ITEM_FODDER
        _write_u16(ram, ADDR_STORED_GRASS, 19)
        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "feed_place_nav")

    def test_feed_verify_succeeds_when_fodder_clears(self) -> None:
        ram = _make_barn_ram(cows=1, fed=0, hay=19, held_item=ITEM_FODDER, player_tile=FEED_TROUGH_STAND)
        task = CowChoresTask(talk=False, brush=False, milk=False)
        task.reset(_make_world(ram))
        task._phase = "feed_verify"
        task._feed_remaining = 1

        ram[ADDR_HELD_ITEM] = 0
        ram[ADDR_FED_COWS_N] = 1
        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task.fed_count, 1)
        self.assertEqual(task._phase, "exit_prep_nav")

    def test_feed_remaining_uses_trough_flags_for_multiple_cows(self) -> None:
        ram = _make_barn_ram(cows=4, fed=1, hay=19, held_item=ITEM_FODDER)
        _write_u16(ram, ADDR_FED_COWS_FLAGS, COW_FEED_SPOTS[0].flag | COW_FEED_SPOTS[1].flag)
        task = CowChoresTask(talk=False, brush=False, milk=False)

        task.reset(_make_world(ram))

        self.assertEqual(task._fed_before, 2)
        self.assertEqual(task._feed_remaining, 2)
        self.assertEqual(task._next_feed_spot(ram), COW_FEED_SPOTS[2])

    def test_calf_cow_is_included_in_feed_goal(self) -> None:
        ram = _make_barn_ram(cows=4, fed=1, hay=19, held_item=ITEM_FODDER)
        _set_cow_slot(ram, 3, (13, 9), status=0x01 | COW_STATUS_BABY_FLAG)
        _write_u16(ram, ADDR_FED_COWS_FLAGS, COW_FEED_SPOTS[0].flag | COW_FEED_SPOTS[1].flag)
        task = CowChoresTask(talk=False, brush=False, milk=False)

        task.reset(_make_world(ram))

        self.assertEqual(task._feed_goal_count, 4)
        self.assertEqual(task._feed_remaining, 2)
        self.assertEqual(task._next_feed_spot(ram), COW_FEED_SPOTS[2])

    def test_after_feed_keeps_looping_until_cow_feed_goal_reached(self) -> None:
        ram = _make_barn_ram(cows=4, fed=1, hay=19, held_item=0)
        _write_u16(ram, ADDR_FED_COWS_FLAGS, COW_FEED_SPOTS[0].flag | COW_FEED_SPOTS[1].flag)
        task = CowChoresTask(talk=False, brush=False, milk=False)
        task.reset(_make_world(ram))
        task._feed_remaining = 3
        task._fed_before = 1

        result = task._after_feed(ram)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._feed_remaining, 2)
        self.assertEqual(task._phase, "fodder_nav")

    def test_after_feed_does_not_count_unchanged_trough_flags(self) -> None:
        ram = _make_barn_ram(cows=1, fed=0, hay=19, held_item=0)
        task = CowChoresTask(talk=False, brush=False, milk=False)
        task.reset(_make_world(ram))
        task._feed_remaining = 1
        task._fed_before = 0

        result = task._after_feed(ram)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task.fed_count, 0)
        self.assertEqual(task._feed_remaining, 1)
        self.assertEqual(task._phase, "fodder_nav")

    def test_feed_starts_at_trough_when_already_holding_fodder(self) -> None:
        ram = _make_barn_ram(cows=1, fed=0, hay=19, held_item=ITEM_FODDER)
        task = CowChoresTask(talk=False)

        task.reset(_make_world(ram))

        self.assertEqual(task._phase, "feed_place_nav")

    def test_feed_place_nav_left_trough_aligns_across_row_before_descending(self) -> None:
        ram = _make_barn_ram(cows=1, fed=0, hay=19, held_item=ITEM_FODDER, player_tile=(13, 11))
        task = CowChoresTask(talk=False, brush=False, milk=False)
        task.reset(_make_world(ram))
        task._phase = "feed_place_nav"

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[6]), 1)
        self.assertEqual(int(result.action.action[5]), 0)
        self.assertEqual(int(result.action.action[0]), 1)

    def test_feed_place_nav_top_trough_climbs_after_x_alignment(self) -> None:
        ram = _make_barn_ram(cows=1, fed=0, hay=19, held_item=ITEM_FODDER)
        _set_player_px(ram, (COW_FEED_SPOTS[0].interact_px[0], 185))
        task = CowChoresTask(talk=False, brush=False, milk=False)
        task.reset(_make_world(ram))
        task._phase = "feed_place_nav"

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[4]), 1)
        self.assertEqual(int(result.action.action[6]), 0)
        self.assertEqual(int(result.action.action[0]), 1)

    def test_fodder_nav_returns_from_left_trough_on_recorded_lane(self) -> None:
        ram = _make_barn_ram(cows=1, fed=0, hay=19, held_item=0)
        _set_player_px(ram, COW_FEED_SPOTS[0].interact_px)
        task = CowChoresTask(talk=False, brush=False, milk=False)
        task.reset(_make_world(ram))
        task._phase = "fodder_nav"

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[5]), 1)
        self.assertEqual(int(result.action.action[0]), 1)

    def test_fodder_route_uses_lower_corridor_only_from_trough_area(self) -> None:
        ram = _make_barn_ram(cows=1, fed=0, hay=20, player_tile=(9, 19))
        task = CowChoresTask(talk=False, brush=False, milk=False)
        task.reset(_make_world(ram))

        self.assertEqual(task._fodder_route(), FODDER_TROUGH_ROUTE)

        ram = _make_barn_ram(cows=1, fed=0, hay=20, player_tile=(10, 17))
        task = CowChoresTask(talk=False, brush=False, milk=False)
        task.reset(_make_world(ram))

        self.assertEqual(task._fodder_route(), ((11, 17),) + FODDER_ROUTE)

        ram = _make_barn_ram(cows=1, fed=0, hay=20, player_tile=(10, 9))
        task = CowChoresTask(talk=False, brush=False, milk=False)
        task.reset(_make_world(ram))

        self.assertEqual(task._fodder_route(), FODDER_ROUTE)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for CowChoresTask care phases -- no ROM needed.

Split monofile: feed in test_cow_task_feed, nav in test_cow_task_nav.
Helpers: cow_test_helpers.
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


class CowChoresTaskTests(unittest.TestCase):
    def test_reset_counts_cows_and_starts_feed_when_milker_missing(self) -> None:
        ram = _make_barn_ram(cows=2, fed=0, hay=20)
        task = CowChoresTask(talk=False, brush=False, milk=False)

        task.reset(_make_world(ram))

        self.assertEqual(task._cow_count, 2)
        self.assertEqual(task._feed_remaining, 2)
        self.assertEqual(task._phase, "fodder_nav")
        self.assertEqual(task.timeout, 30000)

    def test_reset_finishes_when_no_cows(self) -> None:
        ram = _make_barn_ram(cows=0, hay=20)
        task = CowChoresTask()

        task.reset(_make_world(ram))

        self.assertEqual(task._phase, "done")

    def test_can_start_requires_barn(self) -> None:
        ram = _make_barn_ram()
        task = CowChoresTask()
        self.assertTrue(task.can_start(_make_world(ram)))
        ram[ADDR_TILEMAP] = 0x00
        self.assertFalse(task.can_start(_make_world(ram)))

    def test_talk_verify_advances_to_fodder_after_cow_ram_changes(self) -> None:
        ram = _make_barn_ram(cows=1, fed=0, hay=20)
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._phase = "talk_verify"
        task._interaction_started = True
        task._talk_flags_before = 0
        task._talk_happiness_before = 0
        task._navigator.path = [COW_TALK_STAND]
        task._navigator.stasis = 120
        task._pathfinder.temp_blocked.add(COW_TALK_STAND)

        _set_cow_daily(ram, 0, flags=COW_DAILY_TALKED_FLAG, happiness=1)
        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertTrue(task.talked)
        self.assertEqual(task._phase, "fodder_nav")
        self.assertEqual(task._navigator.path, [])
        self.assertEqual(task._navigator.stasis, 0)
        self.assertEqual(task._pathfinder.temp_blocked, set())

    def test_talk_verify_starts_brushing_without_idle_gap_when_adjacent(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, tool_selected=BRUSH_TOOL_ID)
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._phase = "talk_verify"
        task._interaction_started = True
        task._talk_flags_before = 0
        task._talk_happiness_before = 0

        _set_cow_daily(ram, 0, flags=COW_DAILY_TALKED_FLAG, happiness=1)
        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "brush_verify")
        self.assertEqual(task._brush_route_index, len(task._talk_route()) - 1)
        self.assertIsNotNone(result.action)

    def test_talk_verify_pulses_a_while_dialog_is_open(self) -> None:
        ram = _make_barn_ram(cows=1, fed=0, hay=20)
        ram[ADDR_INPUT_LOCK] = 2
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._phase = "talk_verify"
        task._interaction_started = True

        presses = []
        for _ in range(14):
            result = task.step(_make_world(ram))
            self.assertIsNotNone(result.action)
            presses.append(int(result.action.action[8]))

        self.assertEqual(presses[:5], [0, 0, 0, 0, 0])
        self.assertIn(1, presses)
        self.assertIn(0, presses[6:])

    def test_talk_verify_timeout_continues_to_feed(self) -> None:
        ram = _make_barn_ram(cows=1, fed=0, hay=20)
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._phase = "talk_verify"
        task._verify_count = 90
        task._talk_attempts = MAX_TALK_ATTEMPTS
        task._talk_flags_before = 0
        task._talk_happiness_before = 0

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertFalse(task.talked)
        self.assertEqual(task._phase, "fodder_nav")

    def test_talk_verify_retries_before_skipping_when_no_dialog_starts(self) -> None:
        ram = _make_barn_ram(cows=1, fed=0, hay=20)
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._phase = "talk_verify"
        task._verify_count = 90
        task._talk_attempts = 1
        task._talk_flags_before = 0
        task._talk_happiness_before = 0

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertFalse(task.talked)
        self.assertEqual(task._phase, "talk_nav")
        self.assertNotIn(0, task._skipped_talk_slots)

    def test_talk_verify_waits_for_dialog_to_close_after_ram_changes(self) -> None:
        ram = _make_barn_ram(cows=1, fed=0, hay=20)
        ram[ADDR_INPUT_LOCK] = 2
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._phase = "talk_verify"
        task._interaction_started = True
        task._talk_flags_before = 0
        task._talk_happiness_before = 0

        _set_cow_daily(ram, 0, flags=COW_DAILY_TALKED_FLAG, happiness=1)
        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertTrue(task.talked)
        self.assertEqual(task._phase, "talk_verify")

        ram[ADDR_INPUT_LOCK] = 1
        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "fodder_nav")

    def test_reset_starts_brush_when_talk_already_done_and_brush_selected(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, tool_selected=BRUSH_TOOL_ID)
        _set_cow_daily(ram, 0, flags=COW_DAILY_TALKED_FLAG, happiness=1)
        task = CowChoresTask()

        task.reset(_make_world(ram))

        self.assertEqual(task._phase, "brush_nav")

    def test_reset_starts_milk_when_adult_talked_brushed_and_milker_selected(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, tool_selected=MILKER_TOOL_ID)
        _set_cow_slot(ram, 0, (9, 17), status=0x09)
        _set_cow_daily(ram, 0, flags=COW_DAILY_TALKED_FLAG | COW_DAILY_BRUSHED_FLAG, happiness=96)
        task = CowChoresTask()

        task.reset(_make_world(ram))

        self.assertEqual(task._phase, "milk_nav")
        self.assertEqual(task._target_cow_slot, 0)
        self.assertEqual(task._milk_slots, [0])

    def test_reset_skips_milk_for_young_cow(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, tool_selected=MILKER_TOOL_ID)
        _set_cow_daily(ram, 0, flags=COW_DAILY_TALKED_FLAG | COW_DAILY_BRUSHED_FLAG, happiness=96)
        task = CowChoresTask()

        task.reset(_make_world(ram))

        self.assertEqual(task._phase, "exit_prep_nav")

    def test_milk_select_cycles_when_milker_is_in_backpack_slot(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, tool_backpack=MILKER_TOOL_ID)
        _set_cow_slot(ram, 0, (9, 17), status=0x09)
        _set_cow_daily(ram, 0, flags=COW_DAILY_TALKED_FLAG | COW_DAILY_BRUSHED_FLAG, happiness=96)
        task = CowChoresTask()
        task.reset(_make_world(ram))

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "milk_select")
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[9]), 1)

    def test_milk_select_waits_during_tool_swap_animation(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, tool_backpack=MILKER_TOOL_ID)
        ram[ADDR_PLAYER_ACTION] = 28
        _set_cow_slot(ram, 0, (9, 17), status=0x09)
        _set_cow_daily(ram, 0, flags=COW_DAILY_TALKED_FLAG | COW_DAILY_BRUSHED_FLAG, happiness=96)
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._phase = "milk_select"
        task._milk_select_frames = 60

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "milk_select")
        self.assertEqual(task._milk_select_frames, 60)
        self.assertEqual(task._skipped_milk_slots, set())
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[9]), 0)

    def test_after_milk_feeds_before_optional_talk_and_brush(self) -> None:
        ram = _make_barn_ram(
            cows=1,
            fed=0,
            hay=20,
            tool_selected=MILKER_TOOL_ID,
            tool_backpack=BRUSH_TOOL_ID,
        )
        _set_cow_slot(ram, 0, (9, 17), status=0x09)
        task = CowChoresTask()
        task.reset(_make_world(ram))
        _set_cow_daily(ram, 0, flags=COW_DAILY_MILKED_FLAG, happiness=96)
        task._milk_slots = []
        task.milked_count = 1

        result = task._after_milk(ram)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._target_cow_slot, 0)
        self.assertEqual(task._phase, "fodder_nav")

    def test_brush_select_cycles_when_brush_is_in_backpack_slot(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, tool_backpack=BRUSH_TOOL_ID)
        _set_cow_daily(ram, 0, flags=COW_DAILY_TALKED_FLAG, happiness=1)
        task = CowChoresTask()
        task.reset(_make_world(ram))

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "brush_select")
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[9]), 1)

    def test_brush_select_replans_when_only_facing_neighbor_body_tile(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, tool_selected=BRUSH_TOOL_ID, player_tile=(11, 17))
        _set_cow_slot(ram, 0, (9, 17), status=0x09)
        _set_cow_daily(ram, 0, flags=COW_DAILY_TALKED_FLAG, happiness=1)
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._phase = "brush_select"
        task._brush_route_index = 0

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "brush_nav")
        self.assertEqual(task._brush_route_index, len(task._talk_route()) - 1)

    def test_milk_nav_uses_aligned_pinned_interaction_pixel(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, tool_selected=MILKER_TOOL_ID, player_tile=(10, 17))
        _set_cow_slot(ram, 0, (9, 17), status=0x09)
        _set_player_px(ram, (163, 278))
        _set_cow_daily(ram, 0, flags=COW_DAILY_TALKED_FLAG | COW_DAILY_BRUSHED_FLAG, happiness=96)
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._phase = "milk_nav"
        task._target_cow_slot = 0
        task._talk_stand = (10, 17)
        task._talk_face = "left"
        task._recent_pin_slot = 0
        task._recent_pin_stand = (10, 17)
        task._recent_pin_face = "left"
        task._brush_route_index = len(task._talk_route()) - 1

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "milk_verify")
        self.assertEqual(task._navigator.current_tile, (10, 17))
        queued_actions = [result.action.action] + list(task._action_queue)
        self.assertTrue(any(int(action[1]) == 1 for action in queued_actions))
        self.assertFalse(any(int(action[8]) == 1 for action in queued_actions))

    def test_talk_approach_can_pin_from_current_lower_cow_body_tile(self) -> None:
        ram = _make_barn_ram(cows=1, fed=0, hay=20, player_tile=(11, 18))
        _set_cow_slot(ram, 0, (10, 17), status=0x09)
        task = CowChoresTask()
        task.reset(_make_world(ram))

        self.assertEqual(task._talk_stand, (11, 18))
        self.assertEqual(task._talk_face, "left")
        self.assertTrue(task._is_adjacent_to_target_cow(ram, (11, 18), "left"))
        self.assertEqual(task._face_for_target_cow(ram, (11, 18)), "left")

    def test_brush_select_timeout_continues_to_feed(self) -> None:
        ram = _make_barn_ram(cows=1, fed=0, hay=20, tool_backpack=BRUSH_TOOL_ID)
        _set_cow_daily(ram, 0, flags=COW_DAILY_TALKED_FLAG, happiness=1)
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._phase = "brush_select"
        task._brush_select_frames = 60

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertFalse(task.brushed)
        self.assertEqual(task._phase, "fodder_nav")

    def test_brush_select_waits_during_tool_swap_animation(self) -> None:
        ram = _make_barn_ram(cows=1, fed=0, hay=20, tool_backpack=BRUSH_TOOL_ID)
        ram[ADDR_PLAYER_ACTION] = 28
        _set_cow_daily(ram, 0, flags=COW_DAILY_TALKED_FLAG, happiness=1)
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._phase = "brush_select"
        task._brush_select_frames = 60

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "brush_select")
        self.assertEqual(task._brush_select_frames, 60)
        self.assertEqual(task._skipped_brush_slots, set())
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[9]), 0)

    def test_brush_verify_advances_after_cow_ram_changes(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, tool_selected=BRUSH_TOOL_ID)
        _set_cow_daily(ram, 0, flags=COW_DAILY_TALKED_FLAG, happiness=1)
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._phase = "brush_verify"
        task._brush_flags_before = COW_DAILY_TALKED_FLAG
        task._brush_happiness_before = 1

        _set_cow_daily(
            ram,
            0,
            flags=COW_DAILY_TALKED_FLAG | COW_DAILY_BRUSHED_FLAG,
            happiness=4,
        )
        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertTrue(task.brushed)
        self.assertEqual(task._phase, "exit_prep_nav")

    def test_brush_verify_samples_ram_before_draining_tool_queue(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, tool_selected=BRUSH_TOOL_ID)
        _set_cow_daily(ram, 0, flags=COW_DAILY_TALKED_FLAG, happiness=1)
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._phase = "brush_verify"
        task._brush_flags_before = COW_DAILY_TALKED_FLAG
        task._brush_happiness_before = 1
        task._action_queue.append(make_action(y=True))

        _set_cow_daily(
            ram,
            0,
            flags=COW_DAILY_TALKED_FLAG | COW_DAILY_BRUSHED_FLAG,
            happiness=4,
        )
        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertTrue(task.brushed)
        self.assertEqual(task._phase, "exit_prep_nav")
        self.assertEqual(len(task._action_queue), 0)

    def test_begin_brush_verify_refaces_before_tool_use(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, tool_selected=BRUSH_TOOL_ID)
        _set_cow_daily(ram, 0, flags=COW_DAILY_TALKED_FLAG, happiness=1)
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._target_cow_slot = 0
        task._talk_face = "left"

        result = task._begin_brush_verify(ram)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        queued_actions = [result.action.action] + list(task._action_queue)
        self.assertEqual([int(action[6]) for action in queued_actions[:4]], [1, 1, 1, 1])
        self.assertTrue(any(int(action[1]) == 1 for action in queued_actions[4:]))

    def test_brush_verify_retries_when_animation_ends_without_flag(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, tool_selected=BRUSH_TOOL_ID)
        _set_cow_daily(ram, 0, flags=COW_DAILY_TALKED_FLAG, happiness=1)
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._phase = "brush_verify"
        task._interaction_started = True
        task._verify_count = 21
        task._brush_attempts = 1
        task._brush_flags_before = COW_DAILY_TALKED_FLAG
        task._brush_happiness_before = 1

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertFalse(task.brushed)
        self.assertEqual(task._phase, "brush_nav")
        self.assertEqual(task._brush_route_index, len(task._talk_route()) - 1)

    def test_milk_verify_advances_to_ship_after_flag_and_milk_item(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, tool_selected=MILKER_TOOL_ID)
        _set_cow_slot(ram, 0, (9, 17), status=0x09)
        flags = COW_DAILY_TALKED_FLAG | COW_DAILY_BRUSHED_FLAG
        _set_cow_daily(ram, 0, flags=flags, happiness=96)
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._phase = "milk_verify"
        task._target_cow_slot = 0
        task._milk_slots = [0]
        task._milk_flags_before = flags

        _set_cow_daily(ram, 0, flags=flags | COW_DAILY_MILKED_FLAG, happiness=96)
        ram[ADDR_HELD_ITEM] = 0x16
        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "milk_ship_nav")
        self.assertEqual(task.milked_count, 1)
        self.assertEqual(task._milk_slots, [])

    def test_milk_verify_clears_tool_queue_before_barn_shipping(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, tool_selected=MILKER_TOOL_ID)
        _set_cow_slot(ram, 0, (9, 17), status=0x09)
        flags = COW_DAILY_TALKED_FLAG | COW_DAILY_BRUSHED_FLAG
        _set_cow_daily(ram, 0, flags=flags, happiness=96)
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._phase = "milk_verify"
        task._target_cow_slot = 0
        task._milk_slots = [0]
        task._milk_flags_before = flags
        task._action_queue.append(make_action(y=True))

        _set_cow_daily(ram, 0, flags=flags | COW_DAILY_MILKED_FLAG, happiness=96)
        ram[ADDR_HELD_ITEM] = 0x16
        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "milk_ship_nav")
        self.assertEqual(len(task._action_queue), 0)

    def test_brush_retry_refreshes_to_pinned_escape_tile(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, tool_selected=BRUSH_TOOL_ID, player_tile=(11, 17))
        _set_cow_daily(ram, 0, flags=COW_DAILY_TALKED_FLAG, happiness=1)
        _set_cow_slot(ram, 0, (11, 14))
        ram[ADDR_MAP + 14 * 64 + 12] = 0x00
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._phase = "brush_verify"
        task._interaction_started = True
        task._verify_count = 21
        task._brush_attempts = 1
        task._brush_flags_before = COW_DAILY_TALKED_FLAG
        task._brush_happiness_before = 1

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "brush_nav")
        self.assertEqual(task._talk_stand, (10, 14))
        self.assertEqual(task._talk_face, "right")

    def test_brush_nav_does_not_repick_stand_when_already_positioned(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, tool_selected=BRUSH_TOOL_ID, player_tile=(11, 17))
        _set_cow_daily(ram, 0, flags=COW_DAILY_TALKED_FLAG, happiness=1)
        _set_cow_slot(ram, 0, (10, 17))
        _set_player_px(ram, (163, 278))
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._phase = "brush_nav"
        task._brush_route_index = len(task._talk_route()) - 1
        task._talk_stand = (11, 17)
        task._talk_face = "left"

        with patch.object(task, "_refresh_talk_approach", wraps=task._refresh_talk_approach) as refresh:
            result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(refresh.call_count, 0)
        self.assertEqual(task._phase, "brush_verify")

    def test_brush_completion_moves_to_next_cow_attention(self) -> None:
        ram = _make_barn_ram(cows=2, fed=2, hay=20, tool_selected=BRUSH_TOOL_ID)
        _set_cow_daily(ram, 0, flags=COW_DAILY_TALKED_FLAG, happiness=1)
        _set_cow_daily(ram, 1, flags=0, happiness=0)
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._phase = "brush_verify"
        task._target_cow_slot = 0
        task._care_slots = [0, 1]
        task._brush_flags_before = COW_DAILY_TALKED_FLAG
        task._brush_happiness_before = 1
        task.brushed = False

        _set_cow_daily(
            ram,
            0,
            flags=COW_DAILY_TALKED_FLAG | COW_DAILY_BRUSHED_FLAG,
            happiness=4,
        )
        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._target_cow_slot, 1)
        self.assertEqual(task._phase, "talk_nav")

    def test_brush_completion_milks_same_ready_cow_before_next_slot(self) -> None:
        ram = _make_barn_ram(cows=2, fed=2, hay=20, tool_selected=BRUSH_TOOL_ID, tool_backpack=MILKER_TOOL_ID)
        _set_cow_slot(ram, 0, (9, 17), status=0x09)
        _set_cow_slot(ram, 1, (11, 14), status=0x09)
        _set_cow_daily(ram, 0, flags=COW_DAILY_TALKED_FLAG, happiness=1)
        _set_cow_daily(ram, 1, flags=0, happiness=0)
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._phase = "brush_verify"
        task._target_cow_slot = 0
        task._care_slots = [0, 1]
        task._brush_flags_before = COW_DAILY_TALKED_FLAG
        task._brush_happiness_before = 1
        task.brushed = False

        _set_cow_daily(
            ram,
            0,
            flags=COW_DAILY_TALKED_FLAG | COW_DAILY_BRUSHED_FLAG,
            happiness=4,
        )
        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._target_cow_slot, 0)
        self.assertEqual(task._phase, "milk_select")

    def test_brush_failure_does_not_block_milking_ready_cow(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, tool_selected=BRUSH_TOOL_ID, tool_backpack=MILKER_TOOL_ID)
        _set_cow_slot(ram, 0, (9, 17), status=0x09)
        _set_cow_daily(ram, 0, flags=COW_DAILY_TALKED_FLAG, happiness=96)
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._phase = "brush_verify"
        task._target_cow_slot = 0
        task._brush_attempts = 3
        task._interaction_started = True
        task._verify_count = 91
        task._brush_flags_before = COW_DAILY_TALKED_FLAG
        task._brush_happiness_before = 96

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIn(0, task._skipped_brush_slots)
        self.assertEqual(task._phase, "milk_select")

    def test_calf_cow_is_included_in_talk_and_brush_care(self) -> None:
        ram = _make_barn_ram(cows=1, fed=0, hay=19, tool_selected=BRUSH_TOOL_ID)
        _set_cow_slot(ram, 0, (10, 17), status=0x01 | COW_STATUS_BABY_FLAG)
        task = CowChoresTask()

        task.reset(_make_world(ram))

        self.assertEqual(task._care_slots, [0])
        self.assertEqual(task._phase, "fodder_nav")

    def test_cow_slot_timeout_skips_to_feed_instead_of_chasing_forever(self) -> None:
        ram = _make_barn_ram(cows=1, fed=0, hay=19, tool_selected=BRUSH_TOOL_ID)
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._phase = "talk_nav"
        task._target_cow_slot = 0
        task._care_slot_started_step = 1
        task._deferred_care_counts[0] = MAX_CARE_DEFERRALS
        task._step_count = MAX_COW_SLOT_CARE_FRAMES + 2

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIn(0, task._skipped_talk_slots)
        self.assertIn(0, task._skipped_brush_slots)
        self.assertEqual(task._phase, "fodder_nav")

    def test_cow_slot_timeout_log_includes_player_and_cow_coordinates(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20)
        _set_player_px(ram, COW_FEED_SPOTS[3].interact_px)
        _set_cow_slot(ram, 0, (9, 17))
        task = CowChoresTask(brush=False, milk=False, feed=False)
        task.reset(_make_world(ram))
        task._phase = "talk_nav"
        task._target_cow_slot = 0
        task._care_slot_started_step = 1
        task._deferred_care_counts[0] = MAX_CARE_DEFERRALS
        task._step_count = MAX_COW_SLOT_CARE_FRAMES + 2

        with patch("builtins.print") as mocked_print:
            result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        messages = [str(call.args[0]) for call in mocked_print.call_args_list if call.args]
        skip_messages = [message for message in messages if "[COW] Care skipped" in message]
        self.assertEqual(len(skip_messages), 1)
        self.assertIn("pos=(113,277)", skip_messages[0])
        self.assertIn("tile=(7, 17)", skip_messages[0])
        self.assertIn("cow_tile=(9, 17)", skip_messages[0])
        self.assertIn("interact_px=", skip_messages[0])

    def test_begin_next_cow_care_reuses_current_horizontal_stand(self) -> None:
        ram = _make_barn_ram(
            cows=2,
            fed=2,
            hay=20,
            tool_selected=BRUSH_TOOL_ID,
            player_tile=(10, 17),
        )
        _set_cow_slot(ram, 0, (9, 17))
        _set_cow_daily(
            ram,
            0,
            flags=COW_DAILY_TALKED_FLAG | COW_DAILY_BRUSHED_FLAG,
            happiness=20,
        )
        _set_cow_slot(ram, 1, (11, 17))
        task = CowChoresTask()

        task.reset(_make_world(ram))

        self.assertEqual(task._target_cow_slot, 1)
        self.assertEqual(task._talk_stand, (10, 17))
        self.assertEqual(task._talk_face, "right")
        self.assertEqual(task._talk_route_index, len(task._talk_route()) - 1)

    def test_begin_next_cow_care_uses_direct_route_to_nearby_stall(self) -> None:
        ram = _make_barn_ram(
            cows=2,
            fed=2,
            hay=20,
            tool_selected=BRUSH_TOOL_ID,
            player_tile=(11, 17),
        )
        _set_cow_slot(ram, 0, (10, 17))
        _set_cow_daily(
            ram,
            0,
            flags=COW_DAILY_TALKED_FLAG | COW_DAILY_BRUSHED_FLAG,
            happiness=20,
        )
        _set_cow_slot(ram, 1, (10, 15))
        task = CowChoresTask()

        task.reset(_make_world(ram))

        self.assertEqual(task._target_cow_slot, 1)
        self.assertEqual(task._talk_stand, (11, 15))
        self.assertEqual(task._talk_route_index, len(task._talk_route()) - 1)

    def test_cow_nav_failure_skips_slot_before_long_wall_run(self) -> None:
        ram = _make_barn_ram(cows=1, fed=0, hay=20)
        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._phase = "talk_nav"
        task._target_cow_slot = 0
        task._care_slot_started_step = 1
        task._deferred_care_counts[0] = MAX_CARE_DEFERRALS
        task._nav_failures = MAX_COW_NAV_FAILURES + 1

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIn(0, task._skipped_talk_slots)

    def test_pixel_nav_stall_helper_escalates_after_repeated_no_progress(self) -> None:
        """Regression: milk_nav could oscillate with tile stasis stuck at 0."""
        ram = _make_barn_ram(
            cows=1,
            fed=1,
            hay=20,
            tool_selected=MILKER_TOOL_ID,
            player_tile=(12, 4),
        )
        _set_player_px(ram, (202, 68))
        _set_cow_slot(ram, 0, (9, 4), status=0x09)
        task = CowChoresTask(talk=False, brush=False, feed=False)
        task.reset(_make_world(ram))
        task._target_cow_slot = 0
        task._milk_slots = [0]
        task._talk_stand = (10, 4)
        task._talk_face = "left"
        task._phase = "milk_nav"
        task._deferred_milk_counts[0] = 2
        task._navigator.update(ram)
        target = task._cow_interact_pixel(ram, tool=True)
        self.assertIsNotNone(target)

        # No progress toward the interact pixel for PIXEL_NAV_STALL_FRAMES.
        for _ in range(PIXEL_NAV_STALL_FRAMES):
            self.assertFalse(task._pixel_nav_stalled(target))
        self.assertTrue(task._pixel_nav_stalled(target))

        task._pixel_nav_stall_count = MAX_PIXEL_NAV_STALLS - 1
        task._pixel_nav_stale_frames = PIXEL_NAV_STALL_FRAMES
        with patch("builtins.print"):
            result = task._handle_pixel_nav_action(
                ram,
                make_action(up=True, b=True),
                tool=True,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIn(0, task._skipped_milk_slots)
        self.assertEqual(task._phase, "exit_prep_nav")

    def test_milk_slot_timeout_is_shorter_than_external_watchdog(self) -> None:
        self.assertLessEqual(MAX_COW_SLOT_MILK_FRAMES * 3, 3600)

    def test_exit_prep_stages_at_lower_aisle_before_barn_exit(self) -> None:
        ram = _make_barn_ram(cows=1, fed=1, hay=20, player_tile=(13, 17))
        task = CowChoresTask(talk=False, brush=False, milk=False, feed=False)
        task.reset(_make_world(ram))
        self.assertEqual(task._phase, "exit_prep_nav")
        self.assertEqual(COW_EXIT_PREP_STAND, (11, 21))

        result = task.step(_make_world(ram))

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)


if __name__ == "__main__":
    unittest.main()

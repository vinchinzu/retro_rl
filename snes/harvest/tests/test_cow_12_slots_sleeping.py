"""Unit tests for 12 cow slot tracking and sleeping cow recovery in CowChoresTask (rr-rbk)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
import numpy as np

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from cow_test_helpers import (
    _make_barn_ram,
    _make_world,
    set_cow_daily,
    set_cow_slot,
    write_u16,
)
from harvest.core.animal_status import (
    COW_DAILY_BRUSHED_FLAG,
    COW_DAILY_MILKED_FLAG,
    COW_DAILY_TALKED_FLAG,
    COW_SLOT_COUNT,
)
from harvest.core.tile_catalog import ADDR_INPUT_LOCK
from harvest.planner.day_phase_cow import (
    COW_CHORES_PHASE,
    ENTER_BARN_PHASE,
    EXIT_BARN_PHASE,
    NAV_TO_BARN_PHASE,
)
from harvest.tasks.cow_nav_ops import _skip_current_cow_care
from harvest.tasks.cow_target import _barn_cow_slots, _target_cow_pixel, _target_cow_tile
from harvest.tasks.cow_task import (
    ADDR_DIALOG_TEXT_ID,
    COW_SLEEPING_DIALOG_ID,
    CowChoresTask,
    CowPhase,
    MAX_CARE_DEFERRALS,
)
from retro_harness import TaskStatus


class TestCow12Slots(unittest.TestCase):
    def test_all_12_cow_slots_detected_in_barn(self):
        """When 12 cows exist in the barn, _barn_cow_slots must return all 12 slots."""
        ram = _make_barn_ram(cows=12, fed=0, hay=50)
        # Position all 12 cows
        for slot in range(12):
            set_cow_slot(ram, slot, (6 + (slot % 4) * 2, 8 + (slot // 4) * 3))

        task = CowChoresTask()
        task.reset(_make_world(ram))

        slots = _barn_cow_slots(task, ram)
        self.assertEqual(len(slots), 12)
        self.assertEqual(slots, list(range(12)))

    def test_target_cow_tile_and_pixel_fallback_for_all_slots(self):
        """_target_cow_tile and _target_cow_pixel should resolve even if snapshot coordinates
        were filtered out, falling back to direct RAM read."""
        ram = _make_barn_ram(cows=12, fed=0, hay=50)
        # Put slot 11 at (10, 14)
        set_cow_slot(ram, 11, (10, 14))

        task = CowChoresTask()
        task.reset(_make_world(ram))
        task._target_cow_slot = 11

        tile = _target_cow_tile(task, ram)
        self.assertEqual(tile, (10, 14))

        pixel = _target_cow_pixel(task, ram)
        self.assertEqual(pixel, (10 * 16 + 8, 14 * 16 + 8))


class TestCowSleepingRecovery(unittest.TestCase):
    def test_sleeping_dialog_0x035f_defers_care_without_brush_or_milk_retries(self):
        """When a cow is asleep (dialog text ID 0x035F), talk verify should detect it,
        defer care to the end of the queue, and proceed to the next cow without
        wasting frames on brush or milk retries."""
        ram = _make_barn_ram(cows=2, fed=0, hay=50)
        set_cow_slot(ram, 0, (6, 8))
        set_cow_slot(ram, 1, (8, 8))

        task = CowChoresTask(talk=True, brush=True, milk=True, feed=False)
        world = _make_world(ram)
        task.reset(world)

        self.assertEqual(task._target_cow_slot, 0)
        self.assertEqual(task._care_slots, [0, 1])

        # Simulate talking to cow 0: dialog opens with Sleeping text ID 0x035F
        task._phase = CowPhase.TALK_VERIFY
        task._interaction_started = False
        ram[ADDR_INPUT_LOCK] = 0  # dialog active
        write_u16(ram, ADDR_DIALOG_TEXT_ID, COW_SLEEPING_DIALOG_ID)

        # Step 1: Dialog is open, interaction started, _cow_sleeping flag set
        task.step(world)
        self.assertTrue(task._interaction_started)
        self.assertTrue(task._cow_sleeping)

        # Step 2: Dialog closes (input_lock back to 1)
        ram[ADDR_INPUT_LOCK] = 1
        res = task.step(world)

        # Care should have deferred slot 0 to end of _care_slots, moved to next cow (slot 1)
        self.assertEqual(task._deferred_care_counts.get(0), 1)
        self.assertEqual(task._target_cow_slot, 1)
        self.assertEqual(task._care_slots, [1, 0])
        # Did NOT transition to brush or milk on the sleeping cow!
        self.assertNotIn(task._phase, (CowPhase.BRUSH_NAV, CowPhase.BRUSH_VERIFY, CowPhase.MILK_NAV, CowPhase.MILK_VERIFY))

    def test_sleeping_cow_exceeding_max_deferrals_skips_all_care(self):
        """If a sleeping cow has reached MAX_CARE_DEFERRALS, it should be marked
        as skipped for talk, brush, and milk together, avoiding infinite deferral."""
        ram = _make_barn_ram(cows=1, fed=0, hay=50)
        set_cow_slot(ram, 0, (6, 8))

        task = CowChoresTask(talk=True, brush=True, milk=True, feed=False)
        world = _make_world(ram)
        task.reset(world)
        task._deferred_care_counts[0] = MAX_CARE_DEFERRALS

        task._phase = CowPhase.TALK_VERIFY
        task._interaction_started = True
        task._cow_sleeping = True
        ram[ADDR_INPUT_LOCK] = 1
        write_u16(ram, ADDR_DIALOG_TEXT_ID, COW_SLEEPING_DIALOG_ID)

        res = task.step(world)
        self.assertIn(0, task._skipped_talk_slots)
        self.assertIn(0, task._skipped_brush_slots)
        self.assertIn(0, task._skipped_milk_slots)


class TestDayPhaseCowContracts(unittest.TestCase):
    def test_cow_chores_phase_spec_contracts(self):
        """COW_CHORES_PHASE must declare required_maps, estimated_frames, and failure_modes."""
        self.assertEqual(COW_CHORES_PHASE.phase, "COW_CHORES")
        self.assertEqual(COW_CHORES_PHASE.contract.required_maps, (0x27,))
        self.assertEqual(COW_CHORES_PHASE.contract.estimated_frames, 8000)
        self.assertIn("sleeping_animal", COW_CHORES_PHASE.contract.failure_modes)
        self.assertIn("slot_timeout", COW_CHORES_PHASE.contract.failure_modes)
        self.assertIn("nav_unreachable", COW_CHORES_PHASE.contract.failure_modes)

    def test_barn_nav_and_exit_phase_contracts(self):
        self.assertEqual(NAV_TO_BARN_PHASE.contract.required_maps, (0x00,))
        self.assertEqual(ENTER_BARN_PHASE.contract.required_maps, (0x00,))
        self.assertEqual(EXIT_BARN_PHASE.contract.required_maps, (0x27,))


if __name__ == "__main__":
    unittest.main()

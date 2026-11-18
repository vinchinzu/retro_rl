"""Pre-bed barn retry when a barn adult is still unfed (y23_final cow deaths)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
from cow_test_helpers import make_barn_ram, set_cow_slot, write_u16
from harvest.core.animal_status import (
    ADDR_FED_COWS_FLAGS,
    unfed_adult_cow_slots,
)
from harvest.core.tile_catalog import ADDR_TILEMAP
from harvest.planner.multi_day_planner import MultiDayPlannerTask
from harvest.planner.tasks.home_recover import barn_feed_check_needed, barn_feed_check_phases
from retro_harness import WorldState

ADULT = 0x09
CHILD = 0x05
PREGNANT_ADULT = 0x49


def _barn(*statuses: int, fed: int = 0, hay: int = 10, tilemap: int = 0x00) -> np.ndarray:
    ram = make_barn_ram(cows=0, hay=hay)
    for slot, status in enumerate(statuses):
        set_cow_slot(ram, slot, (9 + slot, 17), status=status)
    write_u16(ram, ADDR_FED_COWS_FLAGS, fed)
    ram[ADDR_TILEMAP] = tilemap
    return ram


class UnfedAdultCowTests(unittest.TestCase):
    def test_slot_bit_marks_a_barn_adult_fed(self) -> None:
        self.assertEqual(unfed_adult_cow_slots(_barn(ADULT, ADULT, fed=0x0001)), [1])
        self.assertEqual(unfed_adult_cow_slots(_barn(ADULT, ADULT, fed=0x0003)), [])

    def test_children_skip_the_night_feed_check(self) -> None:
        self.assertEqual(unfed_adult_cow_slots(_barn(CHILD)), [])

    def test_pregnant_adult_reads_the_shared_bit(self) -> None:
        self.assertEqual(unfed_adult_cow_slots(_barn(PREGNANT_ADULT, fed=0x0001)), [0])
        self.assertEqual(unfed_adult_cow_slots(_barn(PREGNANT_ADULT, fed=0x1000)), [])


class BarnFeedCheckTests(unittest.TestCase):
    def test_needed_only_with_hay_near_the_barn(self) -> None:
        self.assertTrue(barn_feed_check_needed(_barn(ADULT)))
        self.assertFalse(barn_feed_check_needed(_barn(ADULT, hay=0)))
        self.assertFalse(barn_feed_check_needed(_barn(ADULT, tilemap=0x04)))
        self.assertFalse(barn_feed_check_needed(_barn(ADULT, fed=0x0001)))

    def test_retry_skips_milking(self) -> None:
        names = [spec.phase for spec in barn_feed_check_phases()]
        self.assertEqual(names, ["EXIT_TO_FARM", "NAV_TO_BARN", "ENTER_BARN", "COW_CHORES", "EXIT_BARN"])
        chores = barn_feed_check_phases()[3]
        self.assertTrue(chores.params["feed"])
        self.assertFalse(chores.params["milk"])

    def test_planner_retries_once_per_day(self) -> None:
        planner = MultiDayPlannerTask()
        world = WorldState(frame=0, ram=_barn(ADULT), info={}, obs=None)
        self.assertEqual(planner._home_phase(world), "feed_check")
        self.assertEqual(planner._home_phase(world), "return_home")
        fed = WorldState(frame=0, ram=_barn(ADULT, fed=0x0001), info={}, obs=None)
        self.assertEqual(MultiDayPlannerTask()._home_phase(fed), "return_home")

    def test_finished_check_goes_home(self) -> None:
        planner = MultiDayPlannerTask()
        planner._phase = "feed_check"
        world = WorldState(frame=0, ram=_barn(ADULT), info={}, obs=None)
        planner._feed_check_done(world, "failure", "feed check over 9000f")
        self.assertEqual(planner._phase, "return_home")
        self.assertEqual(planner._last_day_phase_results[-1]["phase"], "BARN_FEED_CHECK")


if __name__ == "__main__":
    unittest.main()

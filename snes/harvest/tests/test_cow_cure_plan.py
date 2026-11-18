"""Cure planning: when a sick adult cow gets a medicine errand, and in what order."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
from day_plan_test_helpers import make_date_world, set_live_cow_slot, set_money

from harvest.core.animal_status import COW_STATUS_SICK_FLAG
from harvest.core.ram_catalog import COW_SLOT_BASE, COW_SLOT_SIZE, field_spec
from harvest.planner.cow_cure import (
    LAST_BUY_HOUR,
    URGENT_NIGHTS,
    cure_day_phases,
    nights_left,
    sick_adult_slots,
)
from harvest.planner.day_plan import ADDR_WEEKDAY, build_day_phases_from_ram
from harvest.tasks.buy_seeds import TOOL_SHED_ROW_ADDR

LIVE = 0x4000
TUESDAY = 2
SUNDAY = 0
CURE_TAIL = ["ENSURE_MEDICINE", "NAV_TO_BARN", "ENTER_BARN", "COW_CURE", "EXIT_BARN"]


def _world(*, weekday: int = TUESDAY, hour: int = 7, money: int = 5000, sick=(1,), status: int = 0x09, timer: int = 5,
           cows: int = 3, medicine: str = "none", tilemap: int = 0x15):
    world = make_date_world(tilemap, season=1, day=5, hour=hour)
    ram = world.ram
    ram[ADDR_WEEKDAY + LIVE] = weekday
    set_money(ram, money)
    ram[field_spec("num_cows").address + LIVE] = cows
    for slot in range(cows):
        set_live_cow_slot(ram, slot, tile=(9, 13 + slot), flags=0, happiness=120)
        ram[LIVE + COW_SLOT_BASE + slot * COW_SLOT_SIZE] = 0x09
    for slot in sick:
        base = LIVE + COW_SLOT_BASE + slot * COW_SLOT_SIZE
        ram[base] = status | COW_STATUS_SICK_FLAG
        ram[base + 3] = timer
    if medicine == "carried":
        ram[field_spec("tool_selected").address] = 9  # low WRAM: no live offset
    elif medicine == "shed":
        ram[TOOL_SHED_ROW_ADDR + LIVE] |= 0x01
    return world


def _names(phases):
    return [phase.phase for phase in phases]


class CureErrandTests(unittest.TestCase):
    def test_a_healthy_herd_plans_nothing(self) -> None:
        self.assertEqual(cure_day_phases(_world(sick=()).ram), [])

    def test_medicine_in_hand_goes_straight_to_the_barn(self) -> None:
        self.assertEqual(_names(cure_day_phases(_world(medicine="carried").ram)), CURE_TAIL)
        self.assertEqual(_names(cure_day_phases(_world(medicine="shed").ram)), CURE_TAIL)

    def test_no_medicine_buys_it_first_on_a_weekday_morning(self) -> None:
        self.assertEqual(_names(cure_day_phases(_world().ram)), ["BUY_MEDICINE", *CURE_TAIL])

    def test_the_buy_is_blocked_by_wallet_closed_shop_or_late_hour(self) -> None:
        self.assertEqual(cure_day_phases(_world(money=990).ram), [])
        self.assertEqual(cure_day_phases(_world(weekday=SUNDAY).ram), [])
        self.assertEqual(cure_day_phases(_world(hour=LAST_BUY_HOUR).ram), [])
        # 1,000G exactly pays for it
        self.assertEqual(_names(cure_day_phases(_world(money=1000).ram))[0], "BUY_MEDICINE")

    def test_a_calf_is_not_dosed_because_the_rom_apply_wants_an_adult(self) -> None:
        world = _world(status=0x05)
        self.assertEqual(sick_adult_slots(world.ram), [])
        self.assertEqual(cure_day_phases(world.ram), [])

    def test_another_town_trip_defers_the_buy_until_the_cow_is_close_to_dying(self) -> None:
        self.assertEqual(cure_day_phases(_world(timer=URGENT_NIGHTS + 1).ram, other_town_trip=True), [])
        urgent = cure_day_phases(_world(timer=URGENT_NIGHTS).ram, other_town_trip=True)
        self.assertEqual(_names(urgent)[0], "BUY_MEDICINE")
        # An owned dose needs no trip, so a shared town trip never delays it.
        self.assertEqual(
            _names(cure_day_phases(_world(timer=6, medicine="carried").ram, other_town_trip=True)), CURE_TAIL
        )

    def test_nights_left_reads_the_shortest_death_timer(self) -> None:
        world = _world(sick=(0, 2), timer=6)
        world.ram[LIVE + COW_SLOT_BASE + 2 * COW_SLOT_SIZE + 3] = 3
        self.assertEqual(nights_left(world.ram), 3)
        self.assertEqual(nights_left(_world(sick=()).ram), 99)


class CureInTheDayPlanTests(unittest.TestCase):
    def test_the_cure_runs_after_the_exit_and_before_the_ordinary_chores(self) -> None:
        names = _names(build_day_phases_from_ram(_world(medicine="carried").ram))
        self.assertEqual(names[0], "EXIT_TO_FARM")
        cure = names.index("COW_CURE")
        chores = names.index("COW_CHORES")
        self.assertLess(names.index("ENSURE_MEDICINE"), cure)
        self.assertLess(cure, chores)
        # the ordinary chores already carry their own EXIT_TO_FARM; the cure must not add a third
        healthy = _names(build_day_phases_from_ram(_world(sick=()).ram))
        self.assertEqual(names.count("EXIT_TO_FARM"), healthy.count("EXIT_TO_FARM"))

    def test_a_healthy_day_is_unchanged(self) -> None:
        names = _names(build_day_phases_from_ram(_world(sick=()).ram))
        self.assertNotIn("COW_CURE", names)
        self.assertNotIn("BUY_MEDICINE", names)


if __name__ == "__main__":
    unittest.main()

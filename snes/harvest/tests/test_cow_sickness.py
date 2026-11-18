"""Unit tests for cow sickness mechanics, death countdown, and medicine logic."""
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
    ADDR_COW_COUNT,
    ADDR_FED_COWS_FLAGS,
    ADDR_FED_COWS_N,
    COW_CRANKY_HAPPINESS_PENALTY,
    COW_CRANKY_OUTSIDE_ROLL_DENOMINATOR,
    COW_CRANKY_START_TIMER,
    COW_MEDICINE_PRICE_G,
    COW_SICK_HAPPINESS_PENALTY,
    COW_SICK_ROLL_DENOMINATOR,
    COW_SICKNESS_START_TIMER,
    COW_STATUS_ADULT_FLAG,
    COW_STATUS_CRANKY_FLAG,
    COW_STATUS_SICK_FLAG,
    COW_UNFED_HAPPINESS_PENALTY,
    COW_UNFED_SICK_OVERNIGHT_DELTA,
    ITEM_COW_MEDICINE,
    WEATHER_WILD_DOG_BREACH_FLAG,
    cow_days_until_death,
    cow_is_cranky,
    cow_is_sick,
    cranky_cow_slots,
    ram_has_cranky_cows,
    ram_has_sick_cows,
    ram_needs_cow_chores,
    ram_needs_cow_medicine,
    read_cow_cranky_timer,
    read_cow_timer,
    sick_cow_slots,
    unfed_adult_cow_slots,
)
from harvest.core.ram_catalog import COW_SLOT_BASE, COW_SLOT_SIZE
from harvest.planner.livestock_econ import (
    apply_cow_medicine,
    herd_count_after_night,
    outside_cow_cranky_probability,
)

HEALTHY_ADULT = 0x09
SICK_ADULT = 0x29    # exists (0x01) | adult (0x08) | sick (0x20)
CRANKY_ADULT = 0x19  # exists (0x01) | adult (0x08) | cranky (0x10)


def _set_cow_raw(ram: np.ndarray, slot: int, status: int, timer: int = 0) -> None:
    offset = COW_SLOT_BASE + slot * COW_SLOT_SIZE
    ram[offset] = status
    ram[offset + 2] = 0x27  # barn
    ram[offset + 3] = timer


class CowSicknessMechanicsTests(unittest.TestCase):
    def test_sickness_constants_match_rom_decomp(self) -> None:
        """ROM verification of bank_83.asm CowFeedingandStatus ($83BC5A)."""
        self.assertEqual(COW_UNFED_HAPPINESS_PENALTY, -8)  # bank_83.asm:6816 LDA #$FFF8
        self.assertEqual(COW_SICK_HAPPINESS_PENALTY, -12)  # bank_83.asm:6875 LDA #$FFF4
        # Total overnight drop when an unfed cow rolls sick: -8 + -12 = -20
        self.assertEqual(COW_UNFED_SICK_OVERNIGHT_DELTA, -20)
        self.assertEqual(COW_UNFED_HAPPINESS_PENALTY + COW_SICK_HAPPINESS_PENALTY, COW_UNFED_SICK_OVERNIGHT_DELTA)
        # RNGReturn0toA with Y=4 is a 1 in 4 (25%) sickness roll
        self.assertEqual(COW_SICK_ROLL_DENOMINATOR, 4)
        # 7-day death countdown
        self.assertEqual(COW_SICKNESS_START_TIMER, 7)
        # Medicine item and price
        self.assertEqual(ITEM_COW_MEDICINE, 0x09)
        self.assertEqual(COW_MEDICINE_PRICE_G, 1000)

    def test_cranky_constants_match_rom_decomp(self) -> None:
        """ROM verification of bank_83.asm Cow Feeding outside cranky mechanics."""
        self.assertEqual(COW_CRANKY_HAPPINESS_PENALTY, -8)  # bank_83.asm:6902 LDA #$FFF8
        self.assertEqual(COW_CRANKY_START_TIMER, 3)         # bank_83.asm:6938 LDA #$03
        self.assertEqual(COW_CRANKY_OUTSIDE_ROLL_DENOMINATOR, 16) # bank_83.asm:6898 LDA #$10
        self.assertEqual(COW_STATUS_CRANKY_FLAG, 0x10)
        self.assertEqual(WEATHER_WILD_DOG_BREACH_FLAG, 0x0400) # bank_82.asm:2499 ORA #$0400

    def test_cranky_cow_detection(self) -> None:
        ram = make_barn_ram(cows=2, hay=10)
        _set_cow_raw(ram, 0, HEALTHY_ADULT, timer=0)
        _set_cow_raw(ram, 1, CRANKY_ADULT, timer=COW_CRANKY_START_TIMER)

        self.assertFalse(cow_is_cranky(ram, 0))
        self.assertTrue(cow_is_cranky(ram, 1))
        self.assertEqual(cranky_cow_slots(ram), [1])
        self.assertTrue(ram_has_cranky_cows(ram))
        self.assertIsNone(read_cow_cranky_timer(ram, 0))
        self.assertEqual(read_cow_cranky_timer(ram, 1), 3)

    def test_outside_cow_cranky_probability(self) -> None:
        # Wild dog breach ($0196 bit 0x0400) -> 100% cranky for outdoor cows
        self.assertEqual(outside_cow_cranky_probability(wild_dog_breach=True), 1.0)
        # Normal outdoor night with flag 1f5a:1000 -> 1/16 (6.25%)
        self.assertEqual(outside_cow_cranky_probability(wild_dog_breach=False, flag_1f5a_1000=True), 1.0 / 16)
        # If flag clear -> 0%
        self.assertEqual(outside_cow_cranky_probability(wild_dog_breach=False, flag_1f5a_1000=False), 0.0)

    def test_cow_sickness_detection(self) -> None:
        ram = make_barn_ram(cows=2, hay=10)
        _set_cow_raw(ram, 0, HEALTHY_ADULT, timer=0)
        _set_cow_raw(ram, 1, SICK_ADULT, timer=COW_SICKNESS_START_TIMER)

        self.assertFalse(cow_is_sick(ram, 0))
        self.assertTrue(cow_is_sick(ram, 1))
        self.assertEqual(sick_cow_slots(ram), [1])
        self.assertTrue(ram_has_sick_cows(ram))
        self.assertTrue(ram_needs_cow_medicine(ram))

        self.assertIsNone(cow_days_until_death(ram, 0))
        self.assertEqual(cow_days_until_death(ram, 1), 7)
        self.assertEqual(read_cow_timer(ram, 1), 7)

    def test_apply_cow_medicine(self) -> None:
        status, timer = apply_cow_medicine(SICK_ADULT, 5)
        self.assertEqual(status, HEALTHY_ADULT)
        self.assertEqual(timer, 0)
        self.assertFalse(status & COW_STATUS_SICK_FLAG)

    def test_herd_count_after_death_timer(self) -> None:
        # Sick cow with timer 1 reaching 0 overnight in clear weather -> dies
        count = herd_count_after_night(
            cows=2,
            in_barn=True,
            sick_timer=1,
            weather_flags=0,
            sick=True,
        )
        self.assertEqual(count, 1)

        # Sick cow with timer > 1 -> does not die tonight
        count = herd_count_after_night(
            cows=2,
            in_barn=True,
            sick_timer=4,
            weather_flags=0,
            sick=True,
        )
        self.assertEqual(count, 2)

        # Cured cow (sick=False, timer=0) -> does not die
        count = herd_count_after_night(
            cows=2,
            in_barn=True,
            sick_timer=0,
            weather_flags=0,
            sick=False,
        )
        self.assertEqual(count, 2)

    def test_ram_needs_cow_chores_triggers_on_sick_cow(self) -> None:
        ram = make_barn_ram(cows=1, hay=10)
        # Fully fed cow with daily flags set (no normal care needed)
        _set_cow_raw(ram, 0, SICK_ADULT, timer=5)
        write_u16(ram, ADDR_FED_COWS_FLAGS, 0x0001)
        ram[ADDR_FED_COWS_N] = 1
        ram[ADDR_COW_COUNT] = 1
        # Daily flags set to talked & brushed
        ram[COW_SLOT_BASE + 1] = 0x05

        # Even though fed and talked/brushed, chore is needed because cow is sick and needs medicine
        self.assertTrue(ram_needs_cow_chores(ram))


if __name__ == "__main__":
    unittest.main()

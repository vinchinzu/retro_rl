"""Unit tests for Ranch Master evaluation matching SNES bank_83.asm."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest
import numpy as np

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from harvest.core.ram_catalog import field_spec
from harvest.core.ranch_master import (
    RANCH_MASTER_SCORE_THRESHOLD,
    evaluate_ranch_mastery_score,
)


def _write_u8(ram: np.ndarray, addr: int, value: int) -> None:
    ram[addr] = value & 0xFF


def _write_u16(ram: np.ndarray, addr: int, value: int) -> None:
    ram[addr] = value & 0xFF
    ram[addr + 1] = (value >> 8) & 0xFF


def _write_u24(ram: np.ndarray, addr: int, value: int) -> None:
    ram[addr] = value & 0xFF
    ram[addr + 1] = (value >> 8) & 0xFF
    ram[addr + 2] = (value >> 16) & 0xFF


def _set_field(ram: np.ndarray, name: str, value: int) -> None:
    spec = field_spec(name)
    if spec.kind == "u8":
        _write_u8(ram, spec.address, value)
    elif spec.kind == "u16":
        _write_u16(ram, spec.address, value)
    elif spec.kind == "u24":
        _write_u24(ram, spec.address, value)


class TestRanchMaster(unittest.TestCase):
    def test_default_empty_farm_is_novice(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        _set_field(ram, "max_stamina", 100)
        breakdown = evaluate_ranch_mastery_score(ram)
        self.assertFalse(breakdown.is_ranch_master)
        self.assertEqual(breakdown.total_score, 0)
        self.assertEqual(breakdown.rating_title, "Novice Rancher")
        self.assertGreater(len(breakdown.missing_milestones), 5)

    def test_score_components_match_bank_83(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        # Max stamina 200 (10 berries): (200 - 100) >> 1 = 50 pts
        _set_field(ram, "max_stamina", 200)
        # 12 cows: 12 * 3 = 36 pts
        _set_field(ram, "num_cows", 12)
        # 12 chickens: 12 * 3 = 36 pts
        _set_field(ram, "num_chickens", 12)
        # House upgrades: 0x0040 | 0x0080 = +32 pts
        _set_field(ram, "upgrade_flags", 0x00C0)
        # Marriage: 0x0001 (Maria) = +32 pts
        _set_field(ram, "marriage_flags", 0x0001)
        # Children: 0x0008 | 0x0004 = +32 pts
        _set_field(ram, "incubator_flags", 0x000C)

        breakdown = evaluate_ranch_mastery_score(ram)
        self.assertEqual(breakdown.power_berries_score, 50)
        self.assertEqual(breakdown.cows_score, 36)
        self.assertEqual(breakdown.chickens_score, 36)
        self.assertEqual(breakdown.house_upgrades_score, 32)
        self.assertEqual(breakdown.marriage_score, 32)
        self.assertEqual(breakdown.children_score, 32)

        # 50 + 36 + 36 + 32 + 32 + 32 = 218 pts -> Already clears Ranch Master!
        self.assertGreaterEqual(breakdown.total_score, RANCH_MASTER_SCORE_THRESHOLD)
        self.assertTrue(breakdown.is_ranch_master)
        self.assertEqual(breakdown.rating_title, "Ranch Master")

    def test_max_score_capped_at_999(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        _set_field(ram, "max_stamina", 200)
        _set_field(ram, "num_cows", 12)
        _set_field(ram, "num_chickens", 12)
        _set_field(ram, "money_raw", 999999)
        _set_field(ram, "upgrade_flags", 0x00C0)
        _set_field(ram, "marriage_flags", 0x0001)
        _set_field(ram, "incubator_flags", 0x400C)  # Clock + children
        _set_field(ram, "family_event_flags", 0x1000)  # Turtle shell
        _set_field(ram, "happiness", 999)
        _set_field(ram, "ranch_development", 100)
        for girl in ("maria", "ann", "nina", "ellen", "eve"):
            _set_field(ram, f"{girl}_hearts", 999)
        for crop in ("shipped_tomatoes", "shipped_corn", "shipped_potatoes", "shipped_turnips"):
            _set_field(ram, crop, 999)
        # 12 cows with max happiness: 12 * 25 = 300 pts
        from cow_test_helpers import set_cow_daily, set_cow_slot
        for slot in range(12):
            set_cow_slot(ram, slot, (10, 10 + slot), status=0x05)
            set_cow_daily(ram, slot, flags=0, happiness=255)

        breakdown = evaluate_ranch_mastery_score(ram)
        self.assertEqual(breakdown.cow_happiness_score, 300)
        self.assertGreaterEqual(breakdown.total_score, 900)
        self.assertEqual(breakdown.rating_title, "Supreme Ranch Master (Near Perfect)")

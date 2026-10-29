"""Unit tests for Ranch Master evaluation matching SNES bank_83.asm."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from harvest.core.ram_catalog import COW_SLOT_BASE, COW_SLOT_SIZE, field_spec
from harvest.core.ranch_master import (
    BERRY_COUNTER_TARGET,
    BLUE_FEATHER_TEXT_ID,
    FARM_MAP_WRAM,
    MONEY_SCORE_CAP,
    PERFECT_SCORE_THRESHOLD,
    RANCH_MASTER_SCORE_THRESHOLD,
    CounterTarget,
    WindowGuard,
    apply_f0_d30_drift,
    berry_window_guard,
    ending_ranch_report,
    evaluate_ranch_mastery_score,
    morning_score_ledger,
    rating_text_id_for_score,
    score_ledger,
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


def _set_cow(
    ram: np.ndarray,
    slot: int,
    *,
    status: int,
    happiness: int,
    x: int = 0x10,
    y: int = 0x10,
) -> None:
    offset = COW_SLOT_BASE + slot * COW_SLOT_SIZE
    ram[offset] = status & 0xFF
    ram[offset + 4] = happiness & 0xFF
    _write_u16(ram, offset + 8, x)
    _write_u16(ram, offset + 10, y)


def _blank() -> np.ndarray:
    ram = np.zeros(0x20000, dtype=np.uint8)
    _set_field(ram, "max_stamina", 100)
    return ram


class TestRanchMaster(unittest.TestCase):
    def test_default_empty_farm_is_lowest_rating_text(self) -> None:
        breakdown = evaluate_ranch_mastery_score(_blank())
        self.assertFalse(breakdown.is_ranch_master)
        self.assertEqual(breakdown.total_score, 0)
        self.assertEqual(breakdown.rating_text_id, 0x0483)
        self.assertEqual(breakdown.rating_title, "0x0483")
        self.assertGreater(len(breakdown.missing_milestones), 5)
        marriage = next(line for line in breakdown.missing_milestones if line.startswith("Marriage:"))
        self.assertIn(f"0x{BLUE_FEATHER_TEXT_ID:04X}", marriage)
        self.assertNotIn("Blue Feather", marriage)
        self.assertNotIn("Novice", breakdown.rating_title)

    def test_score_components_match_bank_83(self) -> None:
        ram = _blank()
        _set_field(ram, "max_stamina", 200)
        _set_field(ram, "num_cows", 12)
        _set_field(ram, "num_chickens", 12)
        _set_field(ram, "upgrade_flags", 0x00C0)
        _set_field(ram, "marriage_flags", 0x0001)
        _set_field(ram, "incubator_flags", 0x000C)

        breakdown = evaluate_ranch_mastery_score(ram)
        self.assertEqual(breakdown.power_berries_score, 50)
        self.assertEqual(breakdown.cows_score, 36)
        self.assertEqual(breakdown.chickens_score, 36)
        self.assertEqual(breakdown.house_upgrades_score, 32)
        self.assertEqual(breakdown.marriage_score, 32)
        self.assertEqual(breakdown.children_score, 32)
        self.assertEqual(breakdown.total_score, 218)
        self.assertGreaterEqual(breakdown.total_score, RANCH_MASTER_SCORE_THRESHOLD)
        self.assertTrue(breakdown.is_ranch_master)
        self.assertEqual(breakdown.rating_text_id, 0x0488)
        self.assertEqual(breakdown.rating_title, "0x0488")

    def test_money_uses_low_16_bits(self) -> None:
        ram = _blank()
        _set_field(ram, "money_raw", 9984)
        self.assertEqual(evaluate_ranch_mastery_score(ram).money_score, MONEY_SCORE_CAP)
        _set_field(ram, "money_raw", 9983)
        self.assertEqual(evaluate_ranch_mastery_score(ram).money_score, 77)
        # 24-bit 65536 >> 7 caps at 78. The ROM low word is 0.
        _set_field(ram, "money_raw", 0x10000)
        self.assertEqual(evaluate_ranch_mastery_score(ram).money_score, 0)
        _set_field(ram, "money_raw", 0x10000 + 128)
        self.assertEqual(evaluate_ranch_mastery_score(ram).money_score, 1)
        _set_field(ram, "money_raw", 0x10000 + 9984)
        self.assertEqual(evaluate_ranch_mastery_score(ram).money_score, MONEY_SCORE_CAP)
        _set_field(ram, "money_raw", 999999)
        low = 999999 & 0xFFFF
        self.assertEqual(evaluate_ranch_mastery_score(ram).money_score, min(MONEY_SCORE_CAP, low >> 7))

    def test_nine_bit_wrap_on_hearts_and_crops(self) -> None:
        ram = _blank()
        _set_field(ram, "maria_hearts", 511)
        _set_field(ram, "ann_hearts", 496)
        _set_field(ram, "nina_hearts", 495)
        _set_field(ram, "ellen_hearts", 512)
        _set_field(ram, "eve_hearts", 999)
        _set_field(ram, "shipped_tomatoes", 511)
        _set_field(ram, "shipped_corn", 496)
        _set_field(ram, "shipped_potatoes", 495)
        _set_field(ram, "shipped_turnips", 512 + 511)

        breakdown = evaluate_ranch_mastery_score(ram)
        self.assertEqual(breakdown.bachelorette_details["maria"], 31)
        self.assertEqual(breakdown.bachelorette_details["ann"], 31)
        self.assertEqual(breakdown.bachelorette_details["nina"], 30)
        self.assertEqual(breakdown.bachelorette_details["ellen"], 0)
        self.assertEqual(breakdown.bachelorette_details["eve"], 30)
        self.assertEqual(breakdown.crop_ship_details["tomatoes"], 31)
        self.assertEqual(breakdown.crop_ship_details["corn"], 31)
        self.assertEqual(breakdown.crop_ship_details["potatoes"], 30)
        self.assertEqual(breakdown.crop_ship_details["turnips"], 31)
        self.assertEqual(breakdown.bachelorettes_score, 31 + 31 + 30 + 0 + 30)
        self.assertEqual(breakdown.shipped_crops_score, 31 + 31 + 30 + 31)

    def test_all_twelve_cow_slots(self) -> None:
        ram = _blank()
        # points = min(25, happiness >> 3); bit 0 clear scores 0; x=y=0 still counts.
        specs = (
            (0x01, 0, 0),
            (0x01, 7, 0),
            (0x01, 8, 1),
            (0x01, 199, 24),
            (0x01, 200, 25),
            (0x01, 255, 25),
            (0x00, 255, 0),
            (0x05, 80, 10),
            (0x01, 16, 2),
            (0x01, 24, 3),
            (0x01, 40, 5),
            (0x01, 100, 12),
        )
        for slot, (status, happiness, _points) in enumerate(specs):
            x = y = 0 if slot == 7 else 0x20 + slot
            _set_cow(ram, slot, status=status, happiness=happiness, x=x, y=y)
        _set_field(ram, "num_cows", 11)

        breakdown = evaluate_ranch_mastery_score(ram)
        expected = [row[2] for row in specs]
        self.assertEqual(len(breakdown.cow_happiness_details), 12)
        self.assertEqual(breakdown.cow_happiness_details, expected)
        self.assertEqual(breakdown.cow_happiness_score, sum(expected))
        self.assertEqual(breakdown.cows_score, 33)

    def test_remaining_terms(self) -> None:
        ram = _blank()
        _set_field(ram, "max_stamina", 210)
        _set_field(ram, "happiness", 991)
        _set_field(ram, "ranch_development", 99)
        _set_field(ram, "upgrade_flags", 0x0040)
        _set_field(ram, "marriage_flags", 0x0020)
        _set_field(ram, "incubator_flags", 0x4008)
        _set_field(ram, "family_event_flags", 0x1000)
        breakdown = evaluate_ranch_mastery_score(ram)
        self.assertEqual(breakdown.power_berries_score, 55)
        self.assertEqual(breakdown.happiness_score, 30)
        self.assertEqual(breakdown.ranch_development_score, 49)
        self.assertEqual(breakdown.house_upgrades_score, 16)
        self.assertEqual(breakdown.marriage_score, 0)
        self.assertEqual(breakdown.children_score, 16)
        self.assertEqual(breakdown.special_items_score, 22 + 21)

        _set_field(ram, "happiness", 992)
        _set_field(ram, "ranch_development", 100)
        _set_field(ram, "upgrade_flags", 0x0080)
        _set_field(ram, "marriage_flags", 0x0010)
        _set_field(ram, "incubator_flags", 0x0004)
        _set_field(ram, "family_event_flags", 0)
        breakdown = evaluate_ranch_mastery_score(ram)
        self.assertEqual(breakdown.happiness_score, 31)
        self.assertEqual(breakdown.ranch_development_score, 50)
        self.assertEqual(breakdown.house_upgrades_score, 16)
        self.assertEqual(breakdown.marriage_score, 32)
        self.assertEqual(breakdown.children_score, 16)
        self.assertEqual(breakdown.special_items_score, 0)

    def test_rating_band_edges(self) -> None:
        self.assertEqual(rating_text_id_for_score(0), 0x0483)
        self.assertEqual(rating_text_id_for_score(199), 0x0483)
        self.assertEqual(rating_text_id_for_score(200), 0x0488)
        self.assertEqual(rating_text_id_for_score(399), 0x0488)
        self.assertEqual(rating_text_id_for_score(400), 0x0489)
        self.assertEqual(rating_text_id_for_score(599), 0x0489)
        self.assertEqual(rating_text_id_for_score(600), 0x048A)
        self.assertEqual(rating_text_id_for_score(799), 0x048A)
        self.assertEqual(rating_text_id_for_score(800), 0x048B)
        self.assertEqual(rating_text_id_for_score(999), 0x048B)

        ram = _blank()
        _set_field(ram, "max_stamina", 200)
        _set_field(ram, "num_cows", 12)
        _set_field(ram, "num_chickens", 12)
        _set_field(ram, "marriage_flags", 0x0001)
        _set_field(ram, "upgrade_flags", 0x0040)
        _set_field(ram, "happiness", 928)
        low = evaluate_ranch_mastery_score(ram)
        self.assertEqual(low.happiness_score, 29)
        self.assertEqual(low.total_score, 199)
        self.assertEqual(low.rating_text_id, 0x0483)
        _set_field(ram, "happiness", 960)
        high = evaluate_ranch_mastery_score(ram)
        self.assertEqual(high.happiness_score, 30)
        self.assertEqual(high.total_score, 200)
        self.assertEqual(high.rating_text_id, 0x0488)

    def test_windowed_terms_sum_to_999(self) -> None:
        ram = _blank()
        _set_field(ram, "money_raw", 9984)
        _set_field(ram, "num_cows", 12)
        _set_field(ram, "num_chickens", 12)
        _set_field(ram, "max_stamina", 200)
        _set_field(ram, "upgrade_flags", 0x00C0)
        _set_field(ram, "marriage_flags", 0x0001)
        _set_field(ram, "incubator_flags", 0x400C)
        _set_field(ram, "family_event_flags", 0x1000)
        _set_field(ram, "happiness", 992)
        _set_field(ram, "ranch_development", 100)
        for girl in ("maria", "ann", "nina", "ellen", "eve"):
            _set_field(ram, f"{girl}_hearts", 511)
        for crop in ("shipped_tomatoes", "shipped_corn", "shipped_potatoes", "shipped_turnips"):
            _set_field(ram, crop, 496)
        for slot in range(12):
            _set_cow(ram, slot, status=0x05, happiness=200)

        breakdown = evaluate_ranch_mastery_score(ram)
        self.assertEqual(breakdown.money_score, 78)
        self.assertEqual(breakdown.cows_score, 36)
        self.assertEqual(breakdown.chickens_score, 36)
        self.assertEqual(breakdown.power_berries_score, 50)
        self.assertEqual(breakdown.bachelorettes_score, 155)
        self.assertEqual(breakdown.shipped_crops_score, 124)
        self.assertEqual(breakdown.happiness_score, 31)
        self.assertEqual(breakdown.house_upgrades_score, 32)
        self.assertEqual(breakdown.marriage_score, 32)
        self.assertEqual(breakdown.children_score, 32)
        self.assertEqual(breakdown.special_items_score, 43)
        self.assertEqual(breakdown.ranch_development_score, 50)
        self.assertEqual(breakdown.cow_happiness_score, 300)
        self.assertEqual(breakdown.total_score, PERFECT_SCORE_THRESHOLD)
        self.assertEqual(breakdown.rating_text_id, 0x048B)

    def test_max_score_capped_at_999(self) -> None:
        ram = _blank()
        _set_field(ram, "max_stamina", 255)
        _set_field(ram, "num_cows", 12)
        _set_field(ram, "num_chickens", 12)
        _set_field(ram, "money_raw", 9984)
        _set_field(ram, "upgrade_flags", 0x00C0)
        _set_field(ram, "marriage_flags", 0x0001)
        _set_field(ram, "incubator_flags", 0x400C)
        _set_field(ram, "family_event_flags", 0x1000)
        _set_field(ram, "happiness", 999)
        _set_field(ram, "ranch_development", 100)
        for girl in ("maria", "ann", "nina", "ellen", "eve"):
            _set_field(ram, f"{girl}_hearts", 511)
        for crop in ("shipped_tomatoes", "shipped_corn", "shipped_potatoes", "shipped_turnips"):
            _set_field(ram, crop, 511)
        for slot in range(12):
            _set_cow(ram, slot, status=0x05, happiness=255)

        breakdown = evaluate_ranch_mastery_score(ram)
        self.assertEqual(breakdown.power_berries_score, (255 - 100) >> 1)
        self.assertGreater(
            breakdown.power_berries_score
            + 78
            + 36
            + 36
            + 155
            + 124
            + 31
            + 32
            + 32
            + 32
            + 43
            + 50
            + 300,
            PERFECT_SCORE_THRESHOLD,
        )
        self.assertEqual(breakdown.total_score, PERFECT_SCORE_THRESHOLD)
        self.assertEqual(breakdown.rating_text_id, 0x048B)
        self.assertEqual(breakdown.rating_title, "0x048B")


def _paint_tiles(ram: np.ndarray, count: int, tile: int) -> None:
    ram[FARM_MAP_WRAM : FARM_MAP_WRAM + count] = tile & 0xFF


class TestScoreLedger(unittest.TestCase):
    def test_each_term_has_value_max_gap_and_window(self) -> None:
        ledger = score_ledger(_blank())
        self.assertGreaterEqual(len(ledger["rows"]), 14)
        for row in ledger["rows"]:
            for key in ("term", "value", "max", "gap", "window_state"):
                self.assertIn(key, row)
            self.assertEqual(row["gap"], row["max"] - row["value"])
            self.assertIn(row["window_state"], {"below", "in_window", "overshoot"})
        self.assertEqual(ledger["uncapped_total"], ledger["total_score"])
        berries = ledger["terms"]["power_berries"]
        self.assertEqual(berries["hi"], 10)
        self.assertEqual(berries["window_state"], "below")

    def test_windows_and_berry_overshoot(self) -> None:
        ram = _blank()
        _set_field(ram, "money_raw", 9984)
        _set_field(ram, "maria_hearts", 500)
        _set_field(ram, "ann_hearts", 512)
        _set_field(ram, "ellen_hearts", 1008)
        _set_field(ram, "shipped_corn", 495)
        _set_field(ram, "power_berry_count", 11)
        _set_field(ram, "max_stamina", 210)
        _set_field(ram, "happiness", 992)
        _set_field(ram, "num_cows", 13)
        ledger = score_ledger(ram)
        self.assertEqual(ledger["terms"]["money"]["window_state"], "in_window")
        self.assertEqual(ledger["terms"]["money"]["value"], 78)
        self.assertEqual(ledger["terms"]["maria"]["window_state"], "in_window")
        self.assertEqual(ledger["terms"]["ann"]["window_state"], "overshoot")
        self.assertEqual(ledger["terms"]["ann"]["value"], 0)
        self.assertEqual(ledger["terms"]["ellen"]["window_state"], "in_window")
        self.assertEqual(ledger["terms"]["ellen"]["value"], 31)
        self.assertEqual(ledger["terms"]["corn"]["window_state"], "below")
        self.assertEqual(ledger["terms"]["power_berries"]["window_state"], "overshoot")
        self.assertEqual(ledger["terms"]["power_berries"]["value"], 55)
        self.assertEqual(ledger["terms"]["power_berries"]["gap"], -5)
        self.assertEqual(ledger["terms"]["happiness"]["window_state"], "in_window")
        self.assertEqual(ledger["terms"]["cows"]["window_state"], "overshoot")
        wrapped = _blank()
        _set_field(wrapped, "money_raw", 0x10000)
        self.assertEqual(score_ledger(wrapped)["terms"]["money"]["window_state"], "overshoot")
        self.assertEqual(score_ledger(wrapped)["terms"]["money"]["value"], 0)

    def test_f0_projection_wife_money_and_tiles(self) -> None:
        ram = _blank()
        _set_field(ram, "marriage_flags", 0x0001)
        _set_field(ram, "maria_hearts", 509)
        _set_field(ram, "ann_hearts", 509)
        _set_field(ram, "money_raw", 1000)
        _set_field(ram, "shipping_money_raw", 500)
        _paint_tiles(ram, 80, 0x1D)
        ram[FARM_MAP_WRAM + 80] = 0x07
        projected, drift = apply_f0_d30_drift(ram)
        self.assertEqual(drift["wife"], "maria")
        self.assertEqual(drift["wife_hearts_delta"], -2)
        self.assertEqual(drift["tiles"], 80)
        self.assertEqual(drift["development_byte"], 2)
        self.assertEqual(drift["money_raw"], 1500)
        hearts_addr = field_spec("maria_hearts").address
        self.assertEqual(int(ram[hearts_addr]) | (int(ram[hearts_addr + 1]) << 8), 509)
        self.assertIs(projected is ram, False)
        ledger = morning_score_ledger(ram)
        self.assertEqual(ledger["terms"]["maria"]["value"], 31)
        self.assertEqual(ledger["d30_projection"]["terms"]["maria"]["raw"], 507)
        self.assertEqual(ledger["d30_projection"]["terms"]["maria"]["value"], 31)
        self.assertEqual(ledger["d30_projection"]["terms"]["ann"]["raw"], 509)
        self.assertEqual(ledger["d30_projection"]["terms"]["money"]["value"], 1500 >> 7)
        self.assertEqual(ledger["d30_projection"]["terms"]["development"]["value"], 1)
        self.assertEqual(ledger["d30_projection"]["terms"]["development"]["raw"], 2)
        self.assertEqual(ledger["d30_projection"]["drift"]["model"], "F0")

        eve = _blank()
        _set_field(eve, "marriage_flags", 0x0001)
        _set_field(eve, "maria_hearts", 496)
        dropped = morning_score_ledger(eve)
        self.assertEqual(dropped["d30_projection"]["terms"]["maria"]["raw"], 494)
        self.assertEqual(dropped["d30_projection"]["terms"]["maria"]["value"], 30)
        self.assertEqual(dropped["d30_projection"]["terms"]["maria"]["window_state"], "below")

        held = _blank()
        _set_field(held, "marriage_flags", 0x0001)
        _set_field(held, "maria_hearts", 498)
        self.assertEqual(morning_score_ledger(held)["d30_projection"]["terms"]["maria"]["value"], 31)

        blocked = _blank()
        _set_field(blocked, "marriage_flags", 0x0001)
        _set_field(blocked, "incubator_flags", 0x0002)
        _set_field(blocked, "maria_hearts", 509)
        self.assertEqual(apply_f0_d30_drift(blocked)[1]["wife_hearts_delta"], 0)

        low = _blank()
        _set_field(low, "marriage_flags", 0x0002)
        _set_field(low, "ann_hearts", 199)
        _set_field(low, "maria_hearts", 400)
        drift_low = apply_f0_d30_drift(low)[1]
        self.assertEqual(drift_low["wife"], "ann")
        self.assertEqual(drift_low["wife_hearts_delta"], 0)

        covered = _blank()
        _paint_tiles(covered, 2950, 0x1D)
        cover = morning_score_ledger(covered)
        self.assertEqual(cover["d30_projection"]["drift"]["development_byte"], 100)
        self.assertEqual(cover["d30_projection"]["terms"]["development"]["value"], 50)
        self.assertEqual(cover["d30_projection"]["terms"]["development"]["window_state"], "in_window")

        capped = _blank()
        _set_field(capped, "money_raw", 999990)
        _set_field(capped, "shipping_money_raw", 100)
        self.assertEqual(apply_f0_d30_drift(capped)[1]["money_raw"], 999999)

    def test_ending_report_prefers_0x20_inputs(self) -> None:
        early = _blank()
        _set_field(early, "ranch_mastery", 0)
        _set_field(early, "ranch_development", 2)
        late = _blank()
        _set_field(late, "ranch_mastery", 1)
        _set_field(late, "ranch_development", 2)
        report = ending_ranch_report(ram_at_0x20=early, ram_at_0x25=late)
        self.assertEqual(report["ranch_mastery_0x20"], 0)
        self.assertEqual(report["ranch_mastery_0x25"], 1)
        self.assertEqual(report["oracle"]["total_score"], 1)
        self.assertEqual(report["rating_text_id"], 0x0483)
        self.assertEqual(report["rating_text_id_0x25"], 0x0483)
        berry = ending_ranch_report(ram_at_0x20=None, ram_at_0x25=late)
        self.assertIsNone(berry["ranch_mastery_0x20"])
        self.assertEqual(berry["oracle"]["ranch_development_score"], 1)

    def test_morning_journal_writes_ledger_rows(self) -> None:
        from day_plan_test_helpers import make_date_world
        from harvest.planner.multi_day_planner import MultiDayPlannerTask

        planner = MultiDayPlannerTask(target_days=1, max_days=1)
        world = make_date_world(0x15, season=0, day=1)
        planner.reset(world)
        planner._journal_day_complete(world, sleep_reason="test")
        ledger = planner.day_journal[-1]["score_ledger"]
        self.assertTrue(ledger["rows"])
        self.assertIn("d30_projection", ledger)
        self.assertEqual(ledger["d30_projection"]["model"], "F0")
        for row in ledger["rows"]:
            self.assertIn(row["window_state"], {"below", "in_window", "overshoot"})


class TestWindowGuard(unittest.TestCase):
    def test_synthetic_deltas_and_f0_automatic(self) -> None:
        deltas = {"talk": 1, "gift": 5, "poison": -6}
        wife = WindowGuard(CounterTarget(496, 511), automatic_delta=-2)
        self.assertTrue(wife.allow_measured(506, "gift", deltas))
        self.assertFalse(wife.allow_measured(510, "gift", deltas))
        self.assertFalse(wife.allow_measured(510, "gift", {"gift": 6}))
        self.assertTrue(wife.allow(510, 3))
        self.assertFalse(wife.allow(510, 4))
        self.assertFalse(wife.allow_measured(500, "unknown", deltas))
        self.assertEqual(wife.window_state(498), "in_window")
        self.assertEqual(wife.window_state(497), "below")
        self.assertTrue(wife.allow(513, 0))
        self.assertEqual(wife.landing(513, 0), 511)

    def test_berry_guard_high_side_is_10(self) -> None:
        guard = berry_window_guard()
        self.assertEqual(guard.target, BERRY_COUNTER_TARGET)
        self.assertEqual(guard.target.hi, 10)
        self.assertEqual(guard.target.lo, 10)
        self.assertTrue(guard.allow(9, 1))
        self.assertTrue(guard.allow(10, 0))
        self.assertFalse(guard.allow(10, 1))
        self.assertFalse(guard.allow(8, 3))
        self.assertEqual(guard.window_state(10), "in_window")
        self.assertEqual(guard.window_state(11), "overshoot")
        with self.assertRaises(ValueError):
            CounterTarget(11, 10)

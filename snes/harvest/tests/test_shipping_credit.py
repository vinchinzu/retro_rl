"""Unit tests for harvest shipping-credit journal helpers (rr-53g / rr-y8n)."""

from __future__ import annotations

import unittest

import numpy as np

from retro_harness import TaskStatus, WorldState

from harvest.core.ram_catalog import field_spec
from harvest.core.shipping_credit import (
    SHIPPING_SCENE_HOUR,
    acceptance_ok,
    money_rose_after_shipping_window,
    shipping_credit_journal_row,
)
from harvest.planner.tasks.inventory import FarmShippingWaitTask
from harvest.scripts.run_to_day2 import _summarize_journal
from harvest.tasks.farm_clearer import ADDR_INPUT_LOCK, ADDR_TILEMAP


class ShippingCreditTests(unittest.TestCase):
    def test_shipping_scene_hour_is_5pm(self) -> None:
        self.assertEqual(SHIPPING_SCENE_HOUR, 17)

    def test_money_rose_requires_ship_and_wallet_delta(self) -> None:
        self.assertTrue(
            money_rose_after_shipping_window(
                money_pre=150, money_post=390, shipped_count=3
            )
        )
        self.assertFalse(
            money_rose_after_shipping_window(
                money_pre=150, money_post=150, shipped_count=3
            )
        )
        self.assertFalse(
            money_rose_after_shipping_window(
                money_pre=150, money_post=390, shipped_count=0
            )
        )

    def test_money_rose_allows_shipping_money_proxy_when_shipped_count_zero(self) -> None:
        # Skip-harvest fixtures may only have shipping_money > 0.
        self.assertTrue(
            money_rose_after_shipping_window(
                money_pre=1260,
                money_post=3180,
                shipped_count=0,
                shipping_money_pre=1920,
            )
        )

    def test_journal_row_and_acceptance(self) -> None:
        row = shipping_credit_journal_row(
            shipped_count=3,
            harvested_count=3,
            money_pre_5pm=150,
            money_post_5pm=150,
            money_post_sleep=390,
            shipping_money_pre_5pm=240,
            shipping_money_post_5pm=240,
            shipping_money_post_sleep=0,
            hour_pre_5pm=11,
            hour_post_5pm=17,
            day_pre=8,
            day_post_sleep=9,
            pre_5pm_state="Y1_Harvest_Ship_Pre5pm",
            post_5pm_state="Y1_Harvest_Ship_Post5pm",
            post_sleep_state="Y1_Harvest_Ship_PostSleep",
        )
        self.assertEqual(row["kind"], "harvest_ship_5pm_credit")
        self.assertEqual(row["money_delta"], 240)
        self.assertTrue(row["money_rose_after_5pm_window"])
        self.assertTrue(acceptance_ok(row))

    def test_acceptance_fails_without_shipped_count(self) -> None:
        row = shipping_credit_journal_row(
            shipped_count=0,
            money_pre_5pm=100,
            money_post_5pm=100,
            money_post_sleep=100,
        )
        self.assertFalse(acceptance_ok(row))

    def test_bin_drop_without_wallet_credit_is_not_acceptance(self) -> None:
        """Bin drop alone (shipping_money up, wallet flat) is NOT done."""
        row = shipping_credit_journal_row(
            shipped_count=3,
            money_pre_5pm=150,
            money_post_5pm=150,
            money_post_sleep=150,
            shipping_money_pre_5pm=240,
            shipping_money_post_sleep=240,
        )
        self.assertFalse(acceptance_ok(row))


def _shipping_wait_world(
    *,
    hour: int,
    minute: int = 0,
    tilemap: int = 0x00,
    input_lock: int = 1,
) -> WorldState:
    ram = np.zeros(0x20000, dtype=np.uint8)
    hour_addr = field_spec("hour").address
    minute_addr = field_spec("minute").address
    # live day-time helpers may use WRAM offset; write both raw and +0x4000.
    for base in (0, 0x4000):
        if hour_addr + base < len(ram):
            ram[hour_addr + base] = hour
        if minute_addr + base < len(ram):
            ram[minute_addr + base] = minute
    ram[ADDR_TILEMAP] = tilemap
    if ADDR_INPUT_LOCK < len(ram):
        ram[ADDR_INPUT_LOCK] = input_lock
    return WorldState(frame=0, ram=ram, info={}, obs=None)


class FarmShippingWaitTests(unittest.TestCase):
    def test_wait_runs_before_5pm_on_farm(self) -> None:
        task = FarmShippingWaitTask(timeout=100)
        world = _shipping_wait_world(hour=12, minute=0, tilemap=0x00)
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)

    def test_wait_succeeds_after_5pm_on_farm(self) -> None:
        task = FarmShippingWaitTask(timeout=100)
        world = _shipping_wait_world(hour=17, minute=5, tilemap=0x00)
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("farm shipping window done", result.reason or "")

    def test_wait_succeeds_off_farm_so_night_reset_can_still_credit(self) -> None:
        task = FarmShippingWaitTask(timeout=100)
        world = _shipping_wait_world(hour=12, minute=0, tilemap=0x15)  # house
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("off-farm", result.reason or "")


class GateAJournalSummaryTests(unittest.TestCase):
    def test_gate_a_economy_ok_when_money_and_harvest(self) -> None:
        journal = [
            {
                "plan_day": 9,
                "money": 3180,
                "shipped_count": 24,
                "harvested_count": 24,
                "establish_planted": 6,
                "phase_results": [
                    {
                        "phase": "HARVEST_ROUTE",
                        "status": "success",
                        "reason": "harvested=24 shipped=24",
                        "shipped_count": 24,
                        "harvested_count": 24,
                    },
                    {
                        "phase": "CROP_ESTABLISH",
                        "status": "success",
                        "reason": "planted=6 watered=0 passes=1",
                    },
                    {
                        "phase": "WAIT_FARM_SHIPPING",
                        "status": "success",
                        "reason": "farm shipping window done 17:05",
                    },
                ],
            }
        ]
        summary = _summarize_journal(journal)
        self.assertEqual(summary["final_money"], 3180)
        self.assertTrue(summary["harvest_phases_present"])
        self.assertTrue(summary["crop_establish_nonzero"])
        self.assertTrue(summary["gate_a_economy_ok"])
        self.assertEqual(summary["total_shipped"], 24)
        self.assertEqual(summary["total_planted"], 6)
        self.assertEqual(summary["phase_success_counts"].get("WAIT_FARM_SHIPPING"), 1)

    def test_gate_a_fails_calendar_only_money_floor(self) -> None:
        journal = [
            {
                "plan_day": 30,
                "money": 100,
                "shipped_count": 0,
                "phase_results": [
                    {"phase": "EXIT_TO_FARM", "status": "success", "reason": "SUCCESS"},
                    {"phase": "CLEAR_FIELD", "status": "success", "reason": "SUCCESS"},
                ],
            }
        ]
        summary = _summarize_journal(journal)
        self.assertFalse(summary["gate_a_economy_ok"])
        self.assertFalse(summary["harvest_phases_present"])


if __name__ == "__main__":
    unittest.main()

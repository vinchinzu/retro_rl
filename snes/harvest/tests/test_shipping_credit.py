"""Unit tests for harvest shipping-credit journal helpers (rr-53g)."""

from __future__ import annotations

import unittest

from harvest.core.shipping_credit import (
    SHIPPING_SCENE_HOUR,
    acceptance_ok,
    money_rose_after_shipping_window,
    shipping_credit_journal_row,
)


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


if __name__ == "__main__":
    unittest.main()

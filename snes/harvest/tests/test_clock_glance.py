"""Clock/tilemap glance checks: shop miss, frozen clock, house vs mountain. No emulator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from harvest.clock_glance import (
    FARM_TILEMAP,
    HOUSE_TILEMAP,
    HopSpec,
    LeaveSpec,
    MOUNTAIN_TILEMAP,
    SHOP_TILEMAP,
    glance_bench,
    grade_final,
    grade_report,
    parse_tilemap,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures"

BUY_SEEDS_D2 = LeaveSpec(
    hop="buy_seeds_d2",
    tilemap=FARM_TILEMAP,
    hour=(6, 11),
    money_delta=-200,
    require_shop_interior=True,
)

MOUNTAIN_LEAVE = LeaveSpec(
    hop="house_to_mountain",
    tilemap=MOUNTAIN_TILEMAP,
    forbid_tilemaps=(HOUSE_TILEMAP,),
    clock_must_advance=False,
)

PLOT_CLEAR = LeaveSpec(
    hop="clear_plot",
    tilemap=FARM_TILEMAP,
    require_plot_cleared=True,
    clock_must_advance=False,
)


def _load(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


class ClockGlanceTests(unittest.TestCase):
    def test_buy_seeds_d2_fixture_glances_farm_after_shop(self) -> None:
        misses = grade_report(_load("clock_glance_buy_seeds_d2.json"), BUY_SEEDS_D2)
        self.assertEqual(misses, [])

    def test_crossmap_origin_return_without_shop_is_a_glance_miss(self) -> None:
        misses = grade_report(_load("clock_glance_crossmap_origin_return.json"), BUY_SEEDS_D2)
        self.assertTrue(any("shop miss" in m for m in misses))

    def test_still_in_house_when_hop_wants_mountain(self) -> None:
        final = {"tilemap": "0x15", "hour": 6, "minute": 8}
        misses = grade_final(final, MOUNTAIN_LEAVE)
        self.assertTrue(any(m.startswith("tilemap ") for m in misses))
        self.assertTrue(any("forbidden" in m for m in misses))
        self.assertEqual(parse_tilemap("0x15"), HOUSE_TILEMAP)
        self.assertEqual(MOUNTAIN_TILEMAP, 0x10)

    def test_clock_frozen_across_distinct_frames_is_a_glance_miss(self) -> None:
        final = {
            "tilemap": FARM_TILEMAP,
            "hour": 6,
            "minute": 8,
            "clock_samples": [
                {"frame": 0, "hour": 6, "minute": 8, "tilemap": 0x00, "x": 132, "y": 212},
                {"frame": 600, "hour": 6, "minute": 8, "tilemap": 0x00, "x": 140, "y": 220},
            ],
        }
        misses = grade_final(final, LeaveSpec(hop="farm_stand", tilemap=FARM_TILEMAP))
        self.assertTrue(any(m.startswith("clock frozen") for m in misses))

    def test_plot_not_cleared_is_a_glance_miss(self) -> None:
        final = {"map_id": "0x00", "hour": 8, "minute": 0, "plot_cleared": False}
        misses = grade_final(final, PLOT_CLEAR)
        self.assertIn("plot not cleared", misses)

    def test_money_delta_missing_is_a_glance_miss(self) -> None:
        final = {
            "tilemap": FARM_TILEMAP,
            "hour": 8,
            "minute": 0,
            "money_before": 300,
            "money_after": 300,
            "shop_seen": True,
            "maps_seen": [SHOP_TILEMAP, FARM_TILEMAP],
        }
        misses = grade_final(final, BUY_SEEDS_D2)
        self.assertTrue(any(m.startswith("money delta") for m in misses))

    def test_shipping_delta_missing_is_a_glance_miss(self) -> None:
        spec = LeaveSpec(
            hop="ship_bin",
            tilemap=FARM_TILEMAP,
            shipping_delta=50,
            clock_must_advance=False,
        )
        final = {"tilemap": 0x00, "shipping_money": 0, "shipping_money_before": 0}
        misses = grade_final(final, spec)
        self.assertTrue(any(m.startswith("shipping delta") for m in misses))

    def test_aliases_map_id_and_timeline_grade(self) -> None:
        report = {
            "success": True,
            "map_id": "0x00",
            "hour": 7,
            "minute": 4,
            "money": {"before": 300, "after": 100},
            "shop_seen": True,
            "timeline": [
                {"frame": 0, "hour": 6, "minute": 8, "tilemap": "0x1C", "x": 182, "y": 342},
                {"frame": 200, "hour": 7, "minute": 4, "tilemap": "0x00", "x": 135, "y": 456},
            ],
        }
        self.assertEqual(grade_report(report, BUY_SEEDS_D2), [])

    def test_glance_bench_returns_frames_seconds_clock(self) -> None:
        table = glance_bench(0, 120)
        self.assertEqual(table["frames"], 120)
        self.assertEqual(table["seconds"], 2.0)
        self.assertIsNotNone(table["clock"])
        self.assertEqual(table["after"]["clock"], table["clock"])

    def test_hopspec_is_leavespec(self) -> None:
        self.assertIs(HopSpec, LeaveSpec)

    def test_failed_run_is_a_glance_miss(self) -> None:
        report = {
            "success": False,
            "runs": [
                {
                    "success": False,
                    "final": {
                        "tilemap": "0x00",
                        "hour": 8,
                        "minute": 20,
                        "money_before": 300,
                        "money_after": 100,
                        "shop_seen": True,
                        "maps_seen": ["0x1C", "0x00"],
                    },
                }
            ],
        }
        misses = grade_report(report, BUY_SEEDS_D2)
        self.assertIn("success is false", misses)
        self.assertIn("run 1 success is false", misses)

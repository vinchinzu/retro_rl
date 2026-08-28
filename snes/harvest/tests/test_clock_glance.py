"""Clock/tilemap glance checks: shop miss, frozen clock, house vs mountain. No emulator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from harvest.clock_glance import (
    FARM_TILEMAP,
    FENCE_DUMP,
    FENCE_DUMP_DONE,
    FENCE_STAND,
    HOUSE_TILEMAP,
    HopSpec,
    LeaveSpec,
    MOUNTAIN_TILEMAP,
    SHOP_TILEMAP,
    d2_leftover_spec,
    glance_bench,
    grade_final,
    grade_leftover,
    grade_report,
    leftover_from_snapshot,
    leftover_json,
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


def _fence_snapshot(*, fences: int = 80, tilemap: str = "0x0", hour: int = 18, minute: int = 6) -> dict:
    """Hand-built probe ``_snapshot`` shape. No emulator."""
    return {
        "tilemap": tilemap,
        "pos": [86, 69],
        "tile": [5, 4],
        "clock": {"hour": hour, "minute": minute, "clock": f"{hour:02d}:{minute:02d}"},
        "stamina": {"current": 65, "maximum": 100},
        "carry": {"selected": 16, "backpack": 2},
        "debris": {
            "weeds": 0,
            "stones": 185,
            "small_rocks": 0,
            "large_rocks": 51,
            "stumps": 38,
            "fences": fences,
        },
        "samples": {"fences": [[2, 13, "0xa6"]]},
    }


class FenceLeftoverGlanceTests(unittest.TestCase):
    def test_named_specs_are_farm_stand_not_frozen_clock(self) -> None:
        self.assertIs(FENCE_DUMP, FENCE_STAND)
        self.assertEqual(FENCE_STAND.tilemap, FARM_TILEMAP)
        self.assertFalse(FENCE_STAND.clock_must_advance)
        self.assertFalse(FENCE_STAND.require_plot_cleared)
        self.assertIsNone(FENCE_STAND.money_delta)
        self.assertEqual(FENCE_DUMP_DONE.require_empty, ("fences",))
        self.assertFalse(FENCE_DUMP_DONE.clock_must_advance)
        self.assertIs(d2_leftover_spec("fences"), FENCE_STAND)
        self.assertIs(d2_leftover_spec("fences", done=True), FENCE_DUMP_DONE)
        self.assertEqual(d2_leftover_spec("stones", done=True).require_empty, ("stones",))
        self.assertEqual(
            d2_leftover_spec("stumps", done=True).require_empty, ("stumps",)
        )
        self.assertEqual(
            d2_leftover_spec("all", done=True).require_empty,
            ("fences", "stones", "large_rocks", "stumps"),
        )

    def test_leftover_from_snapshot_flattens_probe_clock_and_pos(self) -> None:
        leftover = leftover_from_snapshot(_fence_snapshot())
        self.assertEqual(leftover["tilemap"], FARM_TILEMAP)
        self.assertEqual(leftover["hour"], 18)
        self.assertEqual(leftover["minute"], 6)
        self.assertEqual(leftover["x"], 86)
        self.assertEqual(leftover["y"], 69)
        self.assertEqual(leftover["tile"], [5, 4])
        self.assertEqual(leftover["carry"]["selected"], 16)
        self.assertEqual(leftover["debris"]["fences"], 80)
        self.assertNotIn("samples", leftover)

    def test_farm_stand_with_remaining_posts_is_not_a_location_miss(self) -> None:
        glance = grade_leftover(_fence_snapshot(fences=80), FENCE_STAND)
        self.assertTrue(glance.ok)
        self.assertEqual(glance.misses, [])
        self.assertEqual(glance.leftover["debris"]["fences"], 80)

    def test_dump_done_requires_posts_gone(self) -> None:
        remaining = grade_leftover(_fence_snapshot(fences=18), FENCE_DUMP_DONE)
        self.assertFalse(remaining.ok)
        self.assertTrue(remaining.leftover)
        self.assertTrue(any("fences remaining" in m for m in remaining.misses))
        gone = grade_leftover(_fence_snapshot(fences=0), FENCE_DUMP_DONE)
        self.assertTrue(gone.ok)
        self.assertEqual(gone.misses, [])

    def test_hour_18_leftover_is_not_a_frozen_clock_miss_for_fence_stand(self) -> None:
        still = leftover_from_snapshot(_fence_snapshot(hour=18, minute=6))
        still["clock_samples"] = [
            {"frame": 0, "hour": 18, "minute": 6, "tilemap": FARM_TILEMAP},
            {"frame": 600, "hour": 18, "minute": 6, "tilemap": FARM_TILEMAP},
        ]
        self.assertEqual(grade_final(still, FENCE_STAND), [])
        frozen = grade_final(still, LeaveSpec(hop="farm_stand", tilemap=FARM_TILEMAP))
        self.assertTrue(any(m.startswith("clock frozen") for m in frozen))

    def test_house_still_is_a_stand_miss_but_leftover_is_present(self) -> None:
        glance = grade_leftover(_fence_snapshot(tilemap="0x15"), FENCE_STAND)
        self.assertFalse(glance.ok)
        self.assertEqual(glance.leftover["tilemap"], HOUSE_TILEMAP)
        self.assertTrue(any(m.startswith("tilemap ") for m in glance.misses))

    def test_fail_payload_always_has_leftover_and_glance_misses(self) -> None:
        fail = leftover_json(
            _fence_snapshot(),
            FENCE_STAND,
            ok=False,
            journal=[{"phase": "CLEAR_FENCES", "status": "failed"}],
            partial=True,
        )
        self.assertFalse(fail["ok"])
        self.assertIn("leftover", fail)
        self.assertIn("final", fail)
        self.assertIn("glance_misses", fail)
        self.assertEqual(fail["leftover"], fail["final"])
        self.assertEqual(fail["glance_misses"], [])
        empty = leftover_json({}, FENCE_STAND, ok=False, journal=[{"phase": "exit_to_farm"}])
        self.assertIn("leftover", empty)
        self.assertIn("glance_misses", empty)
        self.assertTrue(empty["glance_misses"])
        success = leftover_json(_fence_snapshot(fences=0), FENCE_DUMP_DONE, ok=True)
        self.assertTrue(success["ok"])
        self.assertEqual(success["glance_misses"], [])

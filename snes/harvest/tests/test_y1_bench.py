"""Year-1 scorecard: economy snapshot, journal hooks, and the report reader."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from coop_task_test_helpers import make_coop_ram
from cow_test_helpers import write_u16, write_u24
from retro_harness import WorldState

from harvest.core.ram_catalog import field_spec
from harvest.planner.livestock_econ import CHICKEN_PRICE, COW_PRICE
from harvest.planner.multi_day_planner import MultiDayPlannerTask
from harvest.tools import y1_bench
from harvest.tools.pr999_report import (
    SHED_ROW_2_ADDR,
    SHED_ROW_2_BRUSH_BIT,
    SHED_ROW_2_MILKER_BIT,
    day_end_economy,
    economy_snapshot,
    journal_census,
)


def _addr(name: str) -> int:
    return field_spec(name).address


def synthetic_ram() -> np.ndarray:
    """Coop RAM (2 hens, 1 chick, hay 7, 1 fed) plus the economy fields."""

    ram = make_coop_ram(adults=2, chicks=1, hay=7, fed_chickens=1, egg_available=3)
    write_u24(ram, _addr("money"), 234)  # stored /10: 2,340 G
    write_u24(ram, _addr("shipping_money"), 45)  # 450 G in the bin
    ram[_addr("num_cows")] = 2
    ram[_addr("fed_cows_n")] = 2
    write_u16(ram, _addr("planted_grass"), 33)
    ram[_addr("cow_feed")] = 4
    ram[_addr("chicken_feed")] = 5
    ram[_addr("potato_seeds")] = 3
    ram[_addr("grass_seeds")] = 1
    write_u16(ram, _addr("shipped_potatoes"), 11)
    write_u16(ram, _addr("shipped_corn"), 2)
    return ram


class EconomySnapshotTests(unittest.TestCase):
    def test_reads_every_field(self) -> None:
        ram = synthetic_ram()
        ram[SHED_ROW_2_ADDR] = SHED_ROW_2_BRUSH_BIT
        snap = economy_snapshot(ram)
        self.assertEqual(snap["money"], 2340)
        self.assertEqual(snap["bin_gold"], 450)
        self.assertEqual((snap["cows"], snap["chickens"], snap["hens"]), (2, 3, 2))
        self.assertEqual((snap["fed_cows"], snap["fed_chickens"]), (2, 1))
        self.assertEqual((snap["planted_grass"], snap["hay"]), (33, 7))
        self.assertEqual((snap["cow_feed"], snap["chicken_feed"]), (4, 5))
        self.assertEqual(snap["egg_available"], 3)
        self.assertEqual(snap["seeds"], {"grass": 1, "corn": 0, "tomato": 0, "potato": 3, "turnip": 0})
        self.assertEqual(snap["shipped"], {"corn": 2, "tomato": 0, "turnip": 0, "potato": 11})
        self.assertTrue(snap["brush"])
        self.assertFalse(snap["milker"])

    def test_tools_count_on_shelf_or_in_carry_pair(self) -> None:
        ram = synthetic_ram()
        self.assertEqual((economy_snapshot(ram)["brush"], economy_snapshot(ram)["milker"]), (False, False))
        ram[SHED_ROW_2_ADDR] = SHED_ROW_2_MILKER_BIT
        ram[_addr("tool_backpack")] = 0x0F  # brush in the backpack slot
        snap = economy_snapshot(ram)
        self.assertEqual((snap["brush"], snap["milker"]), (True, True))

    def test_is_read_only_and_json_ready(self) -> None:
        ram = synthetic_ram()
        before = ram.copy()
        json.dumps(economy_snapshot(ram))
        self.assertTrue(np.array_equal(ram, before))


class JournalHookTests(unittest.TestCase):
    def test_eod_is_cleared_at_plan_day_and_held_after_the_first_end_phase(self) -> None:
        ram = synthetic_ram()
        self.assertIsNone(day_end_economy("plan_day", {"stale": 1}, ram))
        first = day_end_economy("return_home", None, ram)
        self.assertEqual(first["bin_gold"], 450)
        later = synthetic_ram()
        write_u24(later, _addr("shipping_money"), 0)
        self.assertIs(day_end_economy("sleep", first, later), first)

    def test_census_carries_economy_and_eod(self) -> None:
        ram = synthetic_ram()
        eod = {"bin_gold": 450}
        census = journal_census(ram, eod=eod)
        self.assertEqual(census["economy"]["cows"], 2)
        self.assertEqual(census["eod"], eod)
        self.assertIsNone(journal_census(ram)["eod"])

    def test_planner_journals_both_reads(self) -> None:
        ram = synthetic_ram()
        world = WorldState(frame=0, ram=ram, info={})
        planner = MultiDayPlannerTask(target_days=1, max_days=1)
        planner.reset(world)
        planner._eod_economy = {"bin_gold": 450, "cows": 2}
        with contextlib.redirect_stdout(io.StringIO()):
            planner._journal_day_complete(world, sleep_reason="test")
        row = planner.day_journal[-1]
        self.assertEqual(row["eod"], {"bin_gold": 450, "cows": 2})
        self.assertEqual(row["economy"]["chickens"], 3)
        self.assertEqual(row["money"], 2340)
        self.assertIn("development_census", row)
        self.assertIn("cows", row)


def eco(*, cows=0, chickens=0, hens=0, brush=False, milker=False, shipped=None, **extra) -> dict:
    base = {
        "money": 0, "bin_gold": 0, "cows": cows, "chickens": chickens, "hens": hens,
        "planted_grass": 0, "hay": 0, "cow_feed": 0, "chicken_feed": 0,
        "fed_cows": 0, "fed_chickens": 0, "egg_available": 0,
        "brush": brush, "milker": milker, "seeds": {},
        "shipped": shipped or {"corn": 0, "tomato": 0, "turnip": 0, "potato": 0},
    }
    base.update(extra)
    return base


def phase(name: str, reason: str, status: str = "success") -> dict:
    return {"phase": name, "status": status, "reason": reason}


def day(season, day_no, money, *, economy=None, eod=None, phases=(), harvested=0, bad=False,
        floor="", rescues=0, cows=None) -> dict:
    row = {
        "plan_season": season, "plan_day": day_no, "money": money, "shipping_money": 0,
        "shipped_count": harvested, "harvested_count": harvested, "establish_planted": 0,
        "bad_day": bad, "floor_reason": floor, "rescues": rescues,
        "phase_results": list(phases),
    }
    if economy is not None:
        row["economy"] = economy
    if eod is not None:
        row["eod"] = eod
    if cows is not None:
        row["cows"] = [{"slot": i, "exists": i < cows, "happiness": 0, "status": 1} for i in range(12)]
    return row


def synthetic_report(*, extra_cow: int = 0) -> dict:
    potato2_tomato1 = {"corn": 0, "tomato": 1, "turnip": 0, "potato": 2}
    corn1_more = {"corn": 1, "tomato": 1, "turnip": 0, "potato": 2}
    journal = [
        day(0, 29, 1000, economy=eco(), eod=eco(bin_gold=80)),
        day(0, 30, 1200, economy=eco(), eod=eco(bin_gold=0)),
        # Summer D1: cow bought, brush owned; 2 potatoes + 1 tomato dropped (260 G) of a 410 G bin.
        day(1, 1, 400, economy=eco(cows=1, brush=True, shipped=potato2_tomato1),
            eod=eco(cows=1, brush=True, fed_cows=1, bin_gold=410),
            phases=[phase("BUY_COW_VENDOR", "cow 0->1 money 5400->400 shop=0x24"),
                    phase("COW_CHORES", "talk=True brush=True milk=1 ship=1 fed=1 flags=0x0001 happy=0:30")],
            harvested=3),
        # Summer D2: hen bought, milker owned; nothing but eggs and milk in the 300 G bin; hen unfed.
        day(1, 2, 500, economy=eco(cows=1, chickens=1, hens=1, brush=True, milker=True, shipped=potato2_tomato1),
            eod=eco(cows=1, chickens=1, hens=1, brush=True, milker=True, fed_cows=1, fed_chickens=0, bin_gold=300),
            phases=[phase("BUY_COW_VENDOR", "chicken 0->1 money 1500->500"),
                    phase("COW_CHORES", "talk=True brush=True milk=2 ship=2 fed=1"),
                    phase("COOP_CHORES", "fed=0 egg=True ship=True incub=False")]),
        # Summer D3: no chores, a corn drop (120 G) plus a mushroom (150 G) in a 270 G bin, and a bad day.
        day(1, 3, 640, economy=eco(cows=1, chickens=1, hens=1, brush=True, milker=True,
                                   shipped=corn1_more),
            eod=eco(cows=1, chickens=1, hens=1, brush=True, milker=True, fed_cows=1, fed_chickens=1, bin_gold=270),
            phases=[phase("BUY_SEEDS", "bought corn_seeds 0->2 money 640->40 bags=2/2"),
                    {**phase("MUSHROOM_GATHER", "mountain mushroom 1/1 shipped"),
                     "shipping_deposit": {"shipping_money_before": 120, "shipping_money_after": 270}},
                    phase("CROP_WATER", "stuck", "failure"), phase("CROP_WATER", "stuck", "failure"),
                    phase("HARVEST_ROUTE", "late", "blocked")],
            bad=True, floor="day budget", rescues=1),
        day(3, 30, 8000, economy=eco(cows=2, chickens=3, hens=2, brush=True, milker=True, shipped=corn1_more),
            eod=eco(cows=2, chickens=3, hens=2, brush=True, milker=True, fed_cows=2, fed_chickens=2, bin_gold=0),
            phases=[phase("COW_CHORES", "talk=True brush=True milk=0 ship=0 fed=2"),
                    phase("COOP_CHORES", "fed=2 egg=False ship=False incub=False")]),
    ]
    return {
        "state": None,
        "power_on": {"completed": True},
        "success": True, "terminal": True, "reason": "goal reached", "end_date": "Y2S0D1",
        "frames": 1_200_000, "planner_frames": 1_194_000, "wall_seconds": 1500.4,
        "days_completed": 5,
        "start": {"year": 0, "season": 0, "day": 1, "money": 300},
        "end": {"year": 1, "season": 0, "day": 1, "money": 8000 + extra_cow,
                "num_cows": 2, "num_chickens": 3},
        "clean_run": {"intervention_class": "Clean", "initial_state_loads": 0,
                      "mid_run_state_loads": 0, "ram_writes": 0, "assists": {}},
        "floor_report": {"total_bad_days": 1},
        "day_failures": [{"phase": "return_home", "status": "rescue"}, {"phase": "plan_day", "status": "floor"},
                         {"phase": "plan_day", "status": "floor"}],
        "day_journal": journal,
    }


class ScorecardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.card = y1_bench.scorecard(synthetic_report(), "synthetic")

    def test_headline_states_both_parts(self) -> None:
        head = self.card["headline"]
        self.assertEqual((head["planted_grass"], head["hay"]), (0, 0))  # last morning economy read
        self.assertEqual(head["net_worth"], 8000 + 3 * CHICKEN_PRICE + 2 * COW_PRICE)
        self.assertEqual((head["chicken_gold"], head["cow_gold"]), (3000, 10000))
        self.assertTrue(head["year_complete"])
        text = y1_bench.render(self.card)
        self.assertIn("HEADLINE net worth 21,000", text)
        self.assertIn("chickens 3 x 1,000 = 3,000", text)
        self.assertIn("cows 2 x 5,000 = 10,000", text)
        # A hen sells for 500G, so at resale the same wallet, hens and cows are worth 1,500 less.
        self.assertEqual(head["resale_net_worth"], 8000 + 3 * 500 + 2 * COW_PRICE)
        self.assertIn("at resale (500 per hen) 19,500", text)

    def test_end_grass_and_hay_come_from_the_end_fields(self) -> None:
        report = synthetic_report()
        report["end"].update(planted_grass=30, stored_grass=48)
        head = y1_bench.scorecard(report, "grass")["headline"]
        self.assertEqual((head["planted_grass"], head["hay"]), (30, 48))

    def test_skips_are_counted_apart_from_failures(self) -> None:
        report = synthetic_report()
        report["day_journal"][0]["phase_results"] = [phase("HARVEST_ROUTE", "x", "skipped")] * 2
        rel = y1_bench.scorecard(report, "skips")["reliability"]
        self.assertEqual(rel["phase_skips_top"], [("HARVEST_ROUTE", 2)])
        self.assertEqual(rel["phase_failures_total"], 3)

    def test_integrity_and_cost(self) -> None:
        self.assertEqual(self.card["integrity"]["verdict"], "CLEAN")
        self.assertTrue(self.card["integrity"]["comparable"])
        self.assertEqual(self.card["cost"]["game_days"], 120)
        self.assertEqual(self.card["cost"]["frames_per_day"], 10_000)

    def test_dirty_or_pinned_runs_are_not_comparable(self) -> None:
        report = synthetic_report()
        report["clean_run"].update(ram_writes=3, assists={"x": 1})
        verdict = y1_bench.integrity(report)
        self.assertFalse(verdict["comparable"])
        self.assertIn("3 RAM write(s)", verdict["verdict"])
        pinned = synthetic_report()
        pinned.update(power_on=None, state="LN_herd")
        pinned["clean_run"]["initial_state_loads"] = 1
        self.assertFalse(y1_bench.integrity(pinned)["comparable"])
        self.assertIn("pin start LN_herd", y1_bench.integrity(pinned)["verdict"])

    def test_seasons_wallet_and_gold(self) -> None:
        spring, summer, winter = self.card["seasons"]
        self.assertEqual((spring["label"], spring["wallet"], spring["gold"]), ("Y1 Sp", 1200, 900))
        self.assertEqual((summer["label"], summer["days"], summer["wallet"], summer["gold"]), ("Y1 Su", 3, 640, -560))
        self.assertEqual(winter["label"], "Y1 Wi")
        self.assertEqual((summer["harvested"], summer["shipped"]), (3, 3))
        self.assertEqual((summer["cows"], summer["chickens"]), (1, 1))
        self.assertEqual(summer["bad_days"], 1)
        self.assertIsNone(summer["planted"])  # harvested but no planting recorded: n/a

    def test_year_wrap_advances_the_label(self) -> None:
        report = synthetic_report()
        report["day_journal"].append(day(0, 1, 9000))
        rows = y1_bench.day_rows(report)
        self.assertEqual(rows[-1]["label"], "Y2 Sp D1")
        self.assertEqual(rows[-2]["label"], "Y1 Wi D30")

    def test_animal_timeline(self) -> None:
        animals = self.card["animals"]
        self.assertEqual(animals["first_cow"]["label"], "Y1 Su D1")
        self.assertEqual(animals["first_hen"]["label"], "Y1 Su D2")
        self.assertEqual(animals["first_in_coop"]["label"], "Y1 Su D2")
        self.assertEqual(animals["brush"]["label"], "Y1 Su D1")
        self.assertEqual(animals["milker"]["label"], "Y1 Su D2")
        self.assertEqual(animals["milk_shipped"], 3)
        self.assertEqual(animals["egg_ship_days"], 1)

    def test_animal_gold_is_bin_minus_crop_counters_on_chore_days(self) -> None:
        animals = self.card["animals"]
        # Su D1: 410 - (2 potato 160 + 1 tomato 100) = 150; Su D2: 300 - 0; Wi D30: 0 - 0.
        self.assertEqual(animals["animal_gold_est"], 150 + 300 + 0)
        self.assertEqual(animals["animal_gold_days"], 3)
        self.assertEqual(animals["forage_gold"], 150)
        self.assertEqual(animals["unexplained_bin_gold"], 0)  # Su D3: 270 = corn 120 + mushroom 150
        self.assertEqual(animals["bin_gold_anomalies"], 0)
        self.assertEqual(self.card["totals"]["bin_gold"], 80 + 0 + 410 + 300 + 270 + 0)
        self.assertEqual((self.card["eod_days"], self.card["journal_days"]), (6, 6))

    def test_forage_in_the_bin_is_not_animal_gold(self) -> None:
        report = synthetic_report()
        row = report["day_journal"][2]  # Su D1 chore day: add a 150 G mushroom to the bin
        row["eod"]["bin_gold"] += 150
        row["phase_results"].append({**phase("MUSHROOM_GATHER", "m"), "shipping_deposit": {
            "shipping_money_before": 0, "shipping_money_after": 150}})
        animals = y1_bench.scorecard(report, "forage")["animals"]
        self.assertEqual(animals["animal_gold_est"], 450)
        self.assertEqual(animals["forage_gold"], 300)

    def test_unexplained_bin_gold_is_flagged(self) -> None:
        report = synthetic_report()
        report["day_journal"][1]["eod"]["bin_gold"] = 200  # Sp D30: bin with no drop and no forage
        card = y1_bench.scorecard(report, "odd")
        self.assertEqual(card["animals"]["unexplained_bin_gold"], 200)
        self.assertIn("WARNING 200G of bin gold", y1_bench.render(card))

    def test_care_counts_days_with_every_animal_fed(self) -> None:
        care = self.card["care"]
        # Su D1 ok, Su D2 hen unfed, Su D3 ok, Wi D30 ok.
        self.assertEqual((care["animal_days"], care["fully_fed_days"]), (4, 3))
        self.assertEqual(care["pct"], 75.0)
        self.assertEqual(care["unfed_days"], ["Y1 Su D2"])
        self.assertTrue(care["hens_known"])

    def test_reliability_and_spending(self) -> None:
        rel = self.card["reliability"]
        self.assertEqual((rel["bad_days"], rel["floor_trips"], rel["rescues"]), (1, 1, 1))
        self.assertEqual(rel["floor_trip_days"], ["Y1 Su D3"])
        self.assertEqual(rel["phase_failures_top"], [("CROP_WATER", 2), ("HARVEST_ROUTE", 1)])
        self.assertEqual(rel["day_failures"], 3)
        self.assertEqual(rel["day_failure_kinds"], [("plan_day/floor", 2), ("return_home/rescue", 1)])
        self.assertEqual(rel["phase_skips_total"], 0)
        spend = self.card["spending"]
        self.assertEqual(spend["seed_bags"], {"corn": 2})
        self.assertEqual(spend["seed_gold_total"], 600)

    def test_text_fits_one_screen(self) -> None:
        text = y1_bench.render(self.card)
        self.assertLessEqual(len(text.splitlines()), 40)
        self.assertNotIn("Traceback", text)


class DegradeTests(unittest.TestCase):
    def legacy_report(self) -> dict:
        report = synthetic_report()
        for row in report["day_journal"]:
            row.pop("economy", None)
            row.pop("eod", None)
        report["day_journal"][2]["cows"] = [{"slot": 0, "exists": True}]
        report["end"].pop("num_chickens")
        return report

    def test_legacy_journal_prints_na_not_a_crash(self) -> None:
        card = y1_bench.scorecard(self.legacy_report(), "legacy")
        text = y1_bench.render(card)
        self.assertIn("n/a", text)
        self.assertIn("(no economy fields; first used: brush Y1 Su D1", text)
        self.assertIn("~ flock from hen purchase receipts", text)
        self.assertEqual(card["seasons"][1]["chickens"], 1)
        self.assertTrue(card["seasons"][1]["chickens_from_log"])
        self.assertIsNone(card["headline"]["net_worth"])
        self.assertIsNone(card["animals"]["animal_gold_est"])
        self.assertEqual(card["animals"]["first_hen"]["label"], "Y1 Su D2")  # still from the phase reason
        self.assertEqual(card["care"]["hens_known"], False)

    def test_cow_without_a_receipt_is_dated_from_the_census(self) -> None:
        report = self.legacy_report()
        for row in report["day_journal"]:
            for phase_row in row["phase_results"]:
                if phase_row["phase"] == "BUY_COW_VENDOR" and phase_row["reason"].startswith("cow"):
                    phase_row["reason"] = "sequence complete"
        report["day_journal"][2]["cows"] = [{"slot": 0, "exists": True}]
        card = y1_bench.scorecard(report, "census")
        self.assertEqual(card["animals"]["first_cow"]["label"], "Y1 Su D1")
        self.assertEqual(card["animals"]["first_cow"]["source"], "census")
        self.assertIn("Y1 Su D1 (day 31), census", y1_bench.render(card))

    def test_empty_journal_and_bare_report(self) -> None:
        for report in ({}, {"day_journal": [], "success": False}, {"start": {}, "end": {}}):
            text = y1_bench.render(y1_bench.scorecard(report, "bare"))
            self.assertIn("n/a", text)
        self.assertEqual(y1_bench.render(y1_bench.scorecard({}, "x")).count("Traceback"), 0)


class CompareAndCliTests(unittest.TestCase):
    def test_second_report_prints_delta(self) -> None:
        first = y1_bench.scorecard(synthetic_report(), "a")
        second = y1_bench.scorecard(synthetic_report(extra_cow=500), "b")
        delta = y1_bench.deltas(first, second)
        self.assertEqual(delta["wallet"], 500)
        self.assertEqual(delta["net worth"], 500)
        self.assertEqual(delta["bad days"], 0)
        text = y1_bench.render_compare(first, second)
        self.assertIn("net worth", text)
        self.assertIn("+500", text)

    def test_cli_writes_markdown_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = root / "a.json", root / "b.json"
            a.write_text(json.dumps(synthetic_report()))
            b.write_text(json.dumps(synthetic_report(extra_cow=500)))
            out_md, out_json = root / "card.md", root / "card.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(y1_bench.main([str(a), str(b), "--out", str(out_md)]), 0)
                self.assertEqual(y1_bench.main([str(a), "--out", str(out_json)]), 0)
            self.assertIn("Y1 BENCH COMPARE", stdout.getvalue())
            self.assertIn("Y1 BENCH COMPARE", out_md.read_text())
            self.assertEqual(json.loads(out_json.read_text())["headline"]["net_worth"], 21000)


if __name__ == "__main__":
    unittest.main()

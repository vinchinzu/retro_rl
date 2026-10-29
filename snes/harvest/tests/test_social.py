"""Social window, gifts, talks, happiness, and one-trip packing."""

from __future__ import annotations

import unittest

from harvest.core.npc_catalog import ROMANCE_NPCS
from harvest.tasks.social import (
    F9_EVENTS,
    FESTIVALS,
    HEART_HI,
    HEART_LO,
    MINUS_30,
    choose_gift,
    climb_and_hold,
    event_decision,
    four_girl_whatif,
    gift_delta,
    minus_30_skipped,
    pack_social_trip,
    positive_cursor,
    project_pin,
    sample_targets,
    talk_matches,
    wife_entry_holds,
    write_report,
)


class GiftTableTests(unittest.TestCase):
    def test_measured_flower_and_poison_on_two_girls(self) -> None:
        self.assertEqual(gift_delta("maria", 0x05), 5)
        self.assertEqual(gift_delta("ann", 0x05), 6)
        self.assertEqual(gift_delta("maria", 0x02), -8)
        self.assertEqual(gift_delta("ann", 0x02), -8)
        self.assertEqual(gift_delta("nina", 0x05), 8)
        self.assertEqual(gift_delta("nina", 0x02), -4)
        self.assertEqual(gift_delta("ellen", 0x14), 8)
        self.assertEqual(gift_delta("eve", 0x07), -8)

    def test_disliked_only_when_above_the_window(self) -> None:
        for girl in ROMANCE_NPCS:
            climb = choose_gift(girl, 470, lo=HEART_LO, hi=HEART_HI)
            self.assertIsNotNone(climb, girl)
            assert climb is not None
            self.assertGreater(climb[1], 0, girl)
            hold = choose_gift(girl, 500, lo=HEART_LO, hi=HEART_HI)
            self.assertIsNone(hold, girl)
            down = choose_gift(girl, 520, lo=HEART_LO, hi=HEART_HI)
            self.assertIsNotNone(down, girl)
            assert down is not None
            self.assertLess(down[1], 0, girl)
            self.assertGreaterEqual(520 + down[1], HEART_LO, girl)


class TalkTests(unittest.TestCase):
    def test_f9_lines(self) -> None:
        self.assertTrue(talk_matches(0x00F9, 7, 0, dialogue=True))
        self.assertTrue(talk_matches(0x0103, 7, 0, dialogue=True))
        self.assertTrue(talk_matches(0x0087, 10, 8, dialogue=True))
        self.assertTrue(talk_matches(0x0088, 10, 2, dialogue=True))
        self.assertFalse(talk_matches(0x0087, 10, 5, dialogue=True))
        self.assertTrue(talk_matches(0x0087, 18, 0, dialogue=True))
        self.assertFalse(talk_matches(0x0087, 18, 8, dialogue=True))
        self.assertFalse(talk_matches(0x00F9, 7, 0, dialogue=False))

    def test_positive_cursor_is_the_larger_add(self) -> None:
        self.assertEqual(positive_cursor("maria", 0), 0)
        self.assertEqual(positive_cursor("maria", 508), 1)


class HeartWindowTests(unittest.TestCase):
    def test_whatif_from_470_holds_ten_days(self) -> None:
        plans = four_girl_whatif(470, days=10, wife="maria")
        self.assertEqual(set(plans), {"ann", "nina", "ellen", "eve"})
        for name, plan in plans.items():
            self.assertTrue(plan.held, name)
            self.assertEqual(len(plan.daily), 10)
            self.assertGreaterEqual(plan.landed, HEART_LO, name)
            self.assertLessEqual(plan.landed, HEART_HI, name)
            for value in plan.daily:
                self.assertGreaterEqual(value, HEART_LO, name)
                self.assertLessEqual(value, HEART_HI, name)
        wife = climb_and_hold("maria", 470, days=10, wife="maria")
        self.assertTrue(wife.held)
        self.assertGreaterEqual(wife.landed, 498)
        self.assertLessEqual(wife.landed, 511)
        self.assertEqual(wife.daily[-1], wife.landed - 2)
        self.assertGreaterEqual(wife.daily[-1], HEART_LO)
        path = write_report(
            "pr072_hearts_whatif.json",
            {
                "start": 470,
                "days": 10,
                "wife": "maria",
                "girls": {
                    name: {
                        "landed": plan.landed,
                        "daily": list(plan.daily),
                        "gifts": [list(pair) for pair in plan.gifts],
                    }
                    for name, plan in {**plans, "maria": wife}.items()
                },
                "emulator_frames": 0,
                "ram_writes": 0,
            },
        )
        self.assertTrue(path.is_file())

    def test_wife_hour18_entry_band(self) -> None:
        self.assertFalse(wife_entry_holds(496))
        self.assertFalse(wife_entry_holds(497))
        self.assertTrue(wife_entry_holds(498))
        self.assertTrue(wife_entry_holds(511))
        self.assertFalse(wife_entry_holds(512))


class EventTests(unittest.TestCase):
    def test_every_f9_class_has_a_decision_and_minus_30_is_skipped(self) -> None:
        for row in F9_EVENTS:
            decision = event_decision(row["id"])
            if row["kind"] == "avoid":
                self.assertEqual(decision, "skip")
                self.assertEqual(event_decision(row["id"], wife_descent=True), "take")
            else:
                self.assertNotEqual(decision, "skip")
        for address, _girl in MINUS_30:
            self.assertTrue(minus_30_skipped(address))
            self.assertFalse(minus_30_skipped(address, wife_descent=True))


class FestivalPolicyTests(unittest.TestCase):
    def test_six_festivals_avoid_the_minus_30(self) -> None:
        names = {row["name"] for row in FESTIVALS}
        self.assertEqual(
            names,
            {
                "Flower Festival",
                "Harvest Festival",
                "Egg Festival",
                "Thanksgiving",
                "Star Night",
                "New Year",
            },
        )


class HappinessTests(unittest.TestCase):
    def test_y2_spring_checkpoint_projects_at_least_992(self) -> None:
        report = project_pin("LN_calendar_Y2_Spring_D1")
        self.assertEqual(report["snap"]["year"], 1)
        self.assertEqual(report["snap"]["season"], 0)
        self.assertEqual(report["snap"]["day"], 1)
        self.assertEqual(report["snap"]["power_berry_count"], 0)
        berry = dict(report["sources"])["berries"]
        self.assertEqual(berry, 100)
        self.assertTrue(report["reaches_992"])
        self.assertGreaterEqual(report["projected"], 992)
        self.assertLessEqual(report["projected"], 999)
        for _name, applied in report["sinks"]:
            self.assertEqual(applied, 0)
        self.assertEqual(report["emulator_frames"], 0)
        path = write_report("pr073_y2_projection.json", report)
        self.assertTrue(path.is_file())


class LocatorScheduleTests(unittest.TestCase):
    def test_three_sample_days_name_five_girls(self) -> None:
        for sample in ("weekday", "saturday", "sunday"):
            rows = sample_targets(sample)
            self.assertEqual(tuple(row.name for row in rows), ROMANCE_NPCS)


class TripPackTests(unittest.TestCase):
    def test_weekday_social_work_is_one_trip_inside_budget(self) -> None:
        trip = pack_social_trip(year=0, season=0, day=4, weekday=3, hour=10)
        self.assertEqual(trip["trips"], 1)
        self.assertTrue(trip["fits"])
        self.assertLessEqual(trip["frames"], trip["budget"])
        kinds = [stop["kind"] for stop in trip["stops"]]
        self.assertGreaterEqual(kinds.count("shop"), 1)
        self.assertEqual(kinds.count("girl"), 5)
        self.assertTrue(all(stop.get("gift") for stop in trip["stops"] if stop["kind"] == "girl"))
        self.assertNotIn("festival", kinds)


if __name__ == "__main__":
    unittest.main()

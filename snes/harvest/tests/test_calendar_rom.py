"""ROM-verified calendar facts and off-season day policy (rr-bu62.2)."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from harvest.planner.day_phase_calendar import (
    FORECAST_FESTIVAL_CODES,
    SEASON_FALL,
    SEASON_SPRING,
    SEASON_SUMMER,
    SEASON_WINTER,
    festival_name_for_date,
    forecast_code_for_date,
    is_festival_day,
    is_storm_weather,
)
from harvest.planner.day_phase_types import DayPlannerPolicy, day_planner_policy_for_season
from harvest.planner.day_plan_phases import _uses_d2_exhaustive_clear, build_day_phases
from harvest.planner.multi_day_planner import MultiDayPlannerTask
from harvest.scripts.calendar_probe import eve_of, parse_spec
from tests.test_day_plan_calendar_events import _make_ram


class RomCalendarTests(unittest.TestCase):
    def test_new_year_only_from_year_two(self) -> None:
        self.assertFalse(is_festival_day(SEASON_SPRING, 1, year=0))
        self.assertTrue(is_festival_day(SEASON_SPRING, 1, year=1))
        self.assertEqual(festival_name_for_date(SEASON_SPRING, 1, 2), "New Year")

    def test_forecast_codes_match_weather_tomorrow(self) -> None:
        self.assertEqual(forecast_code_for_date(0, SEASON_SPRING, 23), 0x06)
        self.assertEqual(forecast_code_for_date(0, SEASON_FALL, 12), 0x07)
        self.assertEqual(forecast_code_for_date(0, SEASON_WINTER, 10), 0x08)
        self.assertEqual(forecast_code_for_date(0, SEASON_WINTER, 24), 0x09)
        self.assertEqual(forecast_code_for_date(1, SEASON_SPRING, 1), 0x0A)
        self.assertEqual(forecast_code_for_date(0, SEASON_FALL, 20), 0x0B)
        self.assertEqual(forecast_code_for_date(0, SEASON_SPRING, 1), 0)
        self.assertEqual(len(FORECAST_FESTIVAL_CODES), 6)

    def test_no_blizzard_only_hurricane_forecast(self) -> None:
        self.assertTrue(is_storm_weather(3))
        self.assertFalse(is_storm_weather(2))


class OffSeasonPolicyTests(unittest.TestCase):
    def test_fall_and_winter_drop_watering_and_clear(self) -> None:
        for season in (SEASON_FALL, SEASON_WINTER):
            p = day_planner_policy_for_season(season, DayPlannerPolicy())
            self.assertFalse(p.include_watering)
            self.assertFalse(p.include_field_clear)
            self.assertFalse(p.include_planting)
        p = day_planner_policy_for_season(SEASON_SUMMER, DayPlannerPolicy())
        self.assertTrue(p.include_watering)

    def test_winter_keeps_town_trips_and_hen_sales(self) -> None:
        # The Winter path is tilemap 0x0F and its one-off horse name grid is
        # answered by the scene guard (pin probe, Winter D5), so the town stays open.
        w = day_planner_policy_for_season(SEASON_WINTER, DayPlannerPolicy())
        self.assertTrue(w.include_shop_run)
        self.assertTrue(w.include_chicken_sales)

    def test_horse_scene_is_met_alone_on_the_first_winter_morning(self) -> None:
        """Pin probe, Winter D5: the scene's actors sealed the road west and the
        tool trip failed; walking home east was clear. Flags 0x00B7 before, 0x01F7 after."""
        import numpy as np
        from harvest.core.ram_catalog import field_spec
        from harvest.planner.day_phase_calendar import horse_event_due
        from harvest.planner.day_plan_phases import build_day_phases_from_ram

        def ram(season, flags, year=0):
            data = np.zeros(0x20000, dtype=np.uint8)
            for key, value in (("season", season), ("year", year), ("day", 5), ("hour", 6),
                               ("weekday", 4), ("event_flags_1f68", flags), ("tilemap", 0x15),
                               ("num_chickens", 1), ("stored_grass", 10)):
                spec = field_spec(key)
                raw = spec.to_storage(value)
                for shift in range(spec.width):
                    data[spec.address + shift] = (raw >> (8 * shift)) & 255
            return data

        self.assertTrue(horse_event_due(ram(SEASON_WINTER, 0x00B7)))
        self.assertTrue(horse_event_due(ram(SEASON_SPRING, 0x00B7, year=1)))
        self.assertFalse(horse_event_due(ram(SEASON_FALL, 0x00B7)))
        self.assertFalse(horse_event_due(ram(SEASON_WINTER, 0x01F7)))
        self.assertFalse(horse_event_due(ram(SEASON_WINTER, 0x00A6)))
        names = [p.phase for p in build_day_phases_from_ram(ram(SEASON_WINTER, 0x00B7))]
        self.assertEqual(names[:2], ["EXIT_TO_FARM", "MEET_HORSE"])
        names = [p.phase for p in build_day_phases_from_ram(ram(SEASON_WINTER, 0x01F7))]
        self.assertNotIn("MEET_HORSE", names)

    def test_church_macro_off_by_default(self) -> None:
        self.assertFalse(DayPlannerPolicy().include_sunday_church)

    def test_berry_run_follows_the_rom_forage_calendar(self) -> None:
        """Grapes in Spring, the green fruit in Summer, every year; Fall and Winter have no berry run."""
        base = DayPlannerPolicy()
        for year in (0, 1, 2):
            self.assertTrue(day_planner_policy_for_season(SEASON_SPRING, base, year=year).include_berry_run)
            self.assertTrue(day_planner_policy_for_season(SEASON_SUMMER, base, year=year).include_berry_run)
            self.assertFalse(day_planner_policy_for_season(SEASON_FALL, base, year=year).include_berry_run)
            self.assertFalse(day_planner_policy_for_season(SEASON_WINTER, base, year=year).include_berry_run)

    def test_d2_intro_tactic_is_year_one_only(self) -> None:
        self.assertTrue(_uses_d2_exhaustive_clear(0, 2, 0))
        self.assertFalse(_uses_d2_exhaustive_clear(0, 2, 1))

    def test_winter_morning_plans_nothing(self) -> None:
        from harvest.planner.day_plan_phases import build_day_phases_from_ram

        ram = _make_ram(season=SEASON_WINTER, day=13, tilemap=0x15)
        self.assertEqual(build_day_phases_from_ram(ram), [])

    def test_idle_off_season_day_sleeps_at_once(self) -> None:
        task = MultiDayPlannerTask()
        task._last_day_decision = SimpleNamespace(facts=SimpleNamespace(is_storm=False))
        task._active_day = (SEASON_FALL, 5)
        task._last_day_phase_results = [{"phase": "EXIT_TO_FARM", "status": "success"}]
        self.assertEqual(task._build_sleep_task().earliest_sleep_hour, 0)
        task._last_day_phase_results.append({"phase": "HARVEST_ROUTE", "status": "success"})
        task.sleep_when_done = False
        self.assertEqual(task._build_sleep_task().earliest_sleep_hour, 18)
        task._active_day = (SEASON_SUMMER, 5)
        task._last_day_phase_results = []
        self.assertEqual(task._build_sleep_task().earliest_sleep_hour, 18)


class ProbeHelperTests(unittest.TestCase):
    def test_eve_wraps_season_and_year(self) -> None:
        self.assertEqual(eve_of(1, 1, 1), (1, 0, 30))
        self.assertEqual(eve_of(2, 0, 1), (1, 3, 30))
        self.assertEqual(eve_of(1, 2, 12), (1, 2, 11))

    def test_parse_spec(self) -> None:
        self.assertEqual(parse_spec("Y2:fall:12"), (2, 2, 12, None))
        self.assertEqual(parse_spec("Y1:winter:8:fair"), (1, 3, 8, "fair"))


if __name__ == "__main__":
    unittest.main()

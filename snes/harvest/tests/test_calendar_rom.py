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

    def test_winter_blocks_town_trips_but_fall_keeps_them(self) -> None:
        w = day_planner_policy_for_season(SEASON_WINTER, DayPlannerPolicy())
        self.assertFalse(w.include_shop_run)
        self.assertFalse(w.include_chicken_sales)

    def test_church_macro_off_by_default(self) -> None:
        self.assertFalse(DayPlannerPolicy().include_sunday_church)

    def test_berry_run_is_year_one_spring_only(self) -> None:
        base = DayPlannerPolicy()
        self.assertTrue(day_planner_policy_for_season(SEASON_SPRING, base).include_berry_run)
        self.assertFalse(
            day_planner_policy_for_season(SEASON_SPRING, base, year=1).include_berry_run
        )

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

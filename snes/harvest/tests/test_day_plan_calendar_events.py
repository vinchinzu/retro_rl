"""Tests for seasonal crop rotations, festivals, Sundays, rain, and storms.

Verifies beads rr-buo1 and rr-1vc:
- Seasonal crop rotations (Spring potato/turnip, Summer corn/tomato, Fall eggplant)
- Sunday closures, church visits, and mountain foraging
- All 9 festival days with early farm chores, closed shops, and optional attendance
- Rainy day water skip and indoor animal care
- Storm confinement (hurricanes/blizzards) and post-storm recovery (debris clear + repair crops)
- Multi-day planner integration of crop rotation and storm tracking
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
import numpy as np

from harvest.core.ram_catalog import LIVE_RAM_WRAM_OFFSET
from harvest.planner.day_phase_calendar import (
    FESTIVAL_CALENDAR,
    SEASON_FALL,
    SEASON_SPRING,
    SEASON_SUMMER,
    SEASON_WINTER,
    WEATHER_HURRICANE,
    WEATHER_RAIN,
    WEATHER_SUNNY,
    attend_festival_phase,
    default_crop_for_season,
    festival_name_for_date,
    is_festival_day,
    is_storm_day,
    is_storm_weather,
    next_crop_in_rotation,
    post_storm_recovery_phases,
    seasonal_crops_for_season,
)
from harvest.planner.day_plan import (
    ADDR_DAY,
    ADDR_HOUR,
    ADDR_MINUTE,
    ADDR_SEASON,
    ADDR_TILEMAP,
    ADDR_WEATHER,
    ADDR_WEEKDAY,
    DayPlannerPolicy,
    auto_day_phases,
    build_day_phases,
    build_outdoor_day_phases,
)
from harvest.planner.day_plan_decision import (
    auto_day_plan_decision,
    build_day_plan_decision,
    planning_facts,
)
from harvest.planner.day_phase_catalog import (
    ATTEND_FESTIVAL_PHASE,
    BUY_SEEDS_PHASE,
    COOP_CHORES_PHASE,
    COW_CHORES_PHASE,
    CROP_WATER_PHASE,
    GO_TO_SLEEP_PHASE,
    HARVEST_ROUTE_PHASE,
    REPAIR_CROPS_PHASE,
    SUNDAY_CHURCH_PHASE,
    SUNDAY_MOUNTAIN_PHASE,
)
from harvest.planner.multi_day_planner import MultiDayPlannerTask
from harvest.planner.world_probe import WorldProbe


def _make_ram(
    *,
    season: int = SEASON_SPRING,
    day: int = 10,
    weekday: int = 2,
    hour: int = 6,
    minute: int = 0,
    weather: int = WEATHER_SUNNY,
    tilemap: int = 0x00,
    money: int = 1000,
) -> np.ndarray:
    from harvest.core.ram_catalog import field_spec

    ram = np.zeros(0x24000, dtype=np.uint8)
    ram[ADDR_TILEMAP] = tilemap
    base = LIVE_RAM_WRAM_OFFSET
    ram[ADDR_SEASON + base] = season
    ram[ADDR_DAY + base] = day
    ram[ADDR_WEEKDAY + base] = weekday
    ram[ADDR_HOUR + base] = hour
    ram[ADDR_MINUTE + base] = minute
    ram[ADDR_WEATHER + base] = weather

    money_spec = field_spec("money")
    storage = money_spec.to_storage(money)
    money_addr = money_spec.address + base
    for i, byte in enumerate(int(storage).to_bytes(3, "little")):
        ram[money_addr + i] = byte
    return ram


from harvest.planner.day_plan_status import SUNDAY_WEEKDAY


class CalendarCropRotationTests(unittest.TestCase):
    """Test seasonal crop rotation mappings and phase generation."""

    def test_seasonal_crop_definitions(self) -> None:
        self.assertEqual(seasonal_crops_for_season(SEASON_SPRING), ("potato", "turnip"))
        self.assertEqual(seasonal_crops_for_season(SEASON_SUMMER), ("corn", "tomato"))
        self.assertEqual(seasonal_crops_for_season(SEASON_FALL), ("eggplant",))
        self.assertEqual(seasonal_crops_for_season(SEASON_WINTER), ())

        self.assertEqual(default_crop_for_season(SEASON_SPRING), "potato")
        self.assertEqual(default_crop_for_season(SEASON_SUMMER), "corn")
        self.assertEqual(default_crop_for_season(SEASON_FALL), "eggplant")
        self.assertIsNone(default_crop_for_season(SEASON_WINTER))

    def test_next_crop_in_rotation(self) -> None:
        self.assertEqual(next_crop_in_rotation(SEASON_SPRING, "potato"), "turnip")
        self.assertEqual(next_crop_in_rotation(SEASON_SPRING, "turnip"), "potato")
        self.assertEqual(next_crop_in_rotation(SEASON_SUMMER, "corn"), "tomato")
        self.assertEqual(next_crop_in_rotation(SEASON_SUMMER, "tomato"), "corn")
        self.assertEqual(next_crop_in_rotation(SEASON_FALL, "eggplant"), "eggplant")

    def test_summer_crop_rotation_in_build_day_phases(self) -> None:
        # Summer with corn
        policy_corn = DayPlannerPolicy(
            include_shop_run=True,
            include_planting=True,
            summer_crop="corn",
        )
        phases_corn = build_day_phases(
            None,
            season=SEASON_SUMMER,
            day=5,
            hour=6,
            has_seeds=False,
            policy=policy_corn,
        )
        corn_buy = [p for p in phases_corn if p.phase == "BUY_SEEDS"]
        self.assertTrue(corn_buy)
        self.assertEqual(corn_buy[0].params.get("seed_type"), "corn")

        # Summer with tomato
        policy_tomato = DayPlannerPolicy(
            include_shop_run=True,
            include_planting=True,
            summer_crop="tomato",
        )
        phases_tomato = build_day_phases(
            None,
            season=SEASON_SUMMER,
            day=5,
            hour=6,
            has_seeds=False,
            policy=policy_tomato,
        )
        tomato_buy = [p for p in phases_tomato if p.phase == "BUY_SEEDS"]
        self.assertTrue(tomato_buy)
        self.assertEqual(tomato_buy[0].params.get("seed_type"), "tomato")

    def test_fall_eggplant_rotation_in_build_day_phases(self) -> None:
        policy_fall = DayPlannerPolicy(
            include_shop_run=True,
            include_planting=True,
            fall_crop="eggplant",
        )
        phases_fall = build_day_phases(
            None,
            season=SEASON_FALL,
            day=3,
            hour=6,
            has_seeds=False,
            policy=policy_fall,
        )
        eggplant_buy = [p for p in phases_fall if p.phase == "BUY_SEEDS"]
        self.assertTrue(eggplant_buy)
        self.assertEqual(eggplant_buy[0].params.get("seed_type"), "eggplant")


class SundayScheduleTests(unittest.TestCase):
    """Test Sunday closures, church visits, and mountain foraging."""

    def test_sunday_suppresses_shops_and_adds_church_mountain(self) -> None:
        policy = DayPlannerPolicy(
            include_shop_run=True,
            include_planting=True,
            include_sunday_church=True,
            include_sunday_mountain=True,
        )
        phases = build_day_phases(
            None,
            season=SEASON_SPRING,
            day=7,
            weekday=SUNDAY_WEEKDAY,
            hour=6,
            has_seeds=False,
            policy=policy,
        )
        names = [p.phase for p in phases]
        self.assertNotIn("BUY_SEEDS", names)
        self.assertIn("SUNDAY_CHURCH", names)
        self.assertIn("SUNDAY_MOUNTAIN", names)

        # Sunday phases have optional failure policy
        church_phase = next(p for p in phases if p.phase == "SUNDAY_CHURCH")
        self.assertEqual(church_phase.failure_policy, "optional")
        mountain_phase = next(p for p in phases if p.phase == "SUNDAY_MOUNTAIN")
        self.assertEqual(mountain_phase.failure_policy, "optional")

    def test_sunday_policy_flags_suppress_church_or_mountain(self) -> None:
        policy_no_church = DayPlannerPolicy(
            include_sunday_church=False,
            include_sunday_mountain=True,
        )
        phases = build_day_phases(
            None,
            season=SEASON_SPRING,
            day=7,
            weekday=SUNDAY_WEEKDAY,
            hour=6,
            policy=policy_no_church,
        )
        names = [p.phase for p in phases]
        self.assertNotIn("SUNDAY_CHURCH", names)
        self.assertIn("SUNDAY_MOUNTAIN", names)

    def test_sunday_defers_seeds_with_shop_closed_sunday(self) -> None:
        ram = _make_ram(weekday=SUNDAY_WEEKDAY, hour=6)
        decision = build_day_plan_decision(
            ram=ram,
            policy=DayPlannerPolicy(include_shop_run=True, include_planting=True),
        )
        deferred_reasons = {item.phase: item.reason for item in decision.deferred}
        self.assertEqual(deferred_reasons.get("BUY_SEEDS"), "shop_closed_sunday")


class FestivalCalendarTests(unittest.TestCase):
    """Test all 9 festivals across the four seasons."""

    def test_all_nine_festivals_detected(self) -> None:
        expected = [
            (SEASON_SPRING, 8, "Flower Festival"),
            (SEASON_SPRING, 23, "Horse Race"),
            (SEASON_SUMMER, 1, "Fireworks"),
            (SEASON_SUMMER, 20, "Cow Festival"),
            (SEASON_FALL, 12, "Harvest Festival"),
            (SEASON_FALL, 20, "Egg Festival"),
            (SEASON_WINTER, 10, "Thanksgiving"),
            (SEASON_WINTER, 24, "Star Night"),
            (SEASON_WINTER, 30, "New Year's Eve"),
        ]
        self.assertEqual(len(FESTIVAL_CALENDAR), 9)
        for season, day, expected_name in expected:
            self.assertTrue(is_festival_day(season, day), f"Expected festival at {season}:{day}")
            self.assertEqual(festival_name_for_date(season, day), expected_name)

        # Non-festival days
        self.assertFalse(is_festival_day(SEASON_SPRING, 1))
        self.assertFalse(is_festival_day(SEASON_SUMMER, 15))
        self.assertFalse(is_festival_day(SEASON_FALL, 5))
        self.assertFalse(is_festival_day(SEASON_WINTER, 1))

    def test_festival_day_schedules_early_chores_and_attendance(self) -> None:
        # Flower festival: Spring Day 8
        phases = build_day_phases(
            None,
            season=SEASON_SPRING,
            day=8,
            hour=6,
            has_harvest=True,
            has_waterable=True,
            has_chickens=True,
            policy=DayPlannerPolicy(
                include_festivals=True,
                include_harvest=True,
                include_watering=True,
                include_chickens=True,
                include_shop_run=True,
            ),
        )
        names = [p.phase for p in phases]

        # Farm chores happen early
        self.assertIn("COOP_CHORES", names)
        self.assertIn("HARVEST_ROUTE", names)
        self.assertIn("CROP_WATER", names)

        # Shops are closed
        self.assertNotIn("BUY_SEEDS", names)

        # Festival attendance scheduled
        self.assertIn("ATTEND_FESTIVAL", names)
        attend_idx = names.index("ATTEND_FESTIVAL")
        harvest_idx = names.index("HARVEST_ROUTE")
        water_idx = names.index("CROP_WATER")
        self.assertLess(harvest_idx, attend_idx)
        self.assertLess(water_idx, attend_idx)

        # Optional failure policy
        attend_phase = next(p for p in phases if p.phase == "ATTEND_FESTIVAL")
        self.assertEqual(attend_phase.failure_policy, "optional")
        self.assertEqual(attend_phase.params.get("festival_name"), "Flower Festival")

    def test_festival_day_defers_seeds_with_shop_closed_festival(self) -> None:
        ram = _make_ram(season=SEASON_SPRING, day=8, hour=6)
        decision = build_day_plan_decision(
            ram=ram,
            policy=DayPlannerPolicy(include_shop_run=True, include_planting=True),
        )
        self.assertTrue(decision.facts.is_festival)
        self.assertEqual(decision.facts.festival_name, "Flower Festival")
        deferred_reasons = {item.phase: item.reason for item in decision.deferred}
        self.assertEqual(deferred_reasons.get("BUY_SEEDS"), "shop_closed_festival")


class RainyDayTests(unittest.TestCase):
    """Test rainy day phase ordering."""

    def test_rainy_day_skips_crop_watering(self) -> None:
        phases = build_day_phases(
            None,
            hour=6,
            has_waterable=True,
            is_rainy=True,
            policy=DayPlannerPolicy(include_watering=True),
        )
        names = [p.phase for p in phases]
        self.assertNotIn("CROP_WATER", names)

    def test_rainy_day_decision_defers_water(self) -> None:
        ram = _make_ram(weather=WEATHER_RAIN, hour=6)
        decision = build_day_plan_decision(
            ram=ram,
            policy=DayPlannerPolicy(include_watering=True),
        )
        self.assertTrue(decision.facts.is_rainy)
        self.assertIn("rainy day suppresses watering work", decision.notes)


class StormAndRecoveryTests(unittest.TestCase):
    """Test storm confinement and post-storm recovery phases."""

    def test_storm_weather_identification(self) -> None:
        self.assertTrue(is_storm_weather(WEATHER_HURRICANE))
        self.assertFalse(is_storm_weather(WEATHER_SUNNY))
        self.assertFalse(is_storm_weather(WEATHER_RAIN))
        self.assertTrue(is_storm_day(SEASON_SUMMER, WEATHER_HURRICANE))
        self.assertTrue(is_storm_day(SEASON_WINTER, WEATHER_HURRICANE))

    def test_storm_confinement_phases(self) -> None:
        phases = build_day_phases(
            None,
            season=SEASON_SUMMER,
            day=15,
            hour=6,
            is_storm=True,
            has_harvest=True,
            has_waterable=True,
            has_chickens=True,
        )
        names = [p.phase for p in phases]
        # Farmer cannot leave house: only sleeps
        self.assertEqual(names, ["GO_TO_SLEEP"])

    def test_storm_confinement_decision_defers_outdoor_work(self) -> None:
        ram = _make_ram(season=SEASON_SUMMER, day=15, weather=WEATHER_HURRICANE)
        decision = build_day_plan_decision(
            ram=ram,
            policy=DayPlannerPolicy(include_watering=True, include_chickens=True),
        )
        self.assertTrue(decision.facts.is_storm)
        self.assertIn("storm confinement restricts farmer indoors", decision.notes)
        deferred_reasons = {item.phase: item.reason for item in decision.deferred}
        if "CROP_WATER" in deferred_reasons:
            self.assertEqual(deferred_reasons["CROP_WATER"], "storm_confinement")

    def test_post_storm_recovery_phases(self) -> None:
        phases = build_day_phases(
            None,
            season=SEASON_SUMMER,
            day=16,
            hour=6,
            is_post_storm=True,
            has_debris=True,
            has_waterable=True,
            policy=DayPlannerPolicy(include_field_clear=True, include_watering=True),
        )
        names = [p.phase for p in phases]
        self.assertIn("CLEAR_FIELD", names)
        self.assertIn("REPAIR_CROPS", names)

        # Debris clear and crop repair happen first before watering
        clear_idx = names.index("CLEAR_FIELD")
        repair_idx = names.index("REPAIR_CROPS")
        water_idx = names.index("CROP_WATER")
        self.assertLess(clear_idx, water_idx)
        self.assertLess(repair_idx, water_idx)

        # Repair crops is optional
        repair_phase = next(p for p in phases if p.phase == "REPAIR_CROPS")
        self.assertEqual(repair_phase.failure_policy, "optional")

    def test_post_storm_decision_notes(self) -> None:
        ram = _make_ram(season=SEASON_SUMMER, day=16)
        decision = build_day_plan_decision(
            ram=ram,
            is_post_storm=True,
        )
        self.assertTrue(decision.facts.is_post_storm)
        self.assertIn(
            "post-storm recovery prioritizes field clearing and crop repair",
            decision.notes,
        )


class MultiDayPlannerCalendarIntegrationTests(unittest.TestCase):
    """Test MultiDayPlannerTask crop rotation and overnight storm tracking."""

    def test_multi_day_planner_crop_rotation(self) -> None:
        task = MultiDayPlannerTask(
            crop_rotation={
                SEASON_SPRING: "potato",
                SEASON_SUMMER: "corn",
                SEASON_FALL: "eggplant",
            }
        )
        # Spring world
        world_spring = SimpleNamespace(ram=_make_ram(season=SEASON_SPRING, day=5))
        day_task_spring = task._build_day_task(world_spring)
        self.assertEqual(day_task_spring.seed_type, "potato")

        # Summer world
        world_summer = SimpleNamespace(ram=_make_ram(season=SEASON_SUMMER, day=5))
        day_task_summer = task._build_day_task(world_summer)
        self.assertEqual(day_task_summer.seed_type, "corn")

        # Fall world
        world_fall = SimpleNamespace(ram=_make_ram(season=SEASON_FALL, day=5))
        day_task_fall = task._build_day_task(world_fall)
        self.assertEqual(day_task_fall.seed_type, "eggplant")

    def test_multi_day_planner_overnight_storm_tracking(self) -> None:
        task = MultiDayPlannerTask()
        world_storm = SimpleNamespace(
            ram=_make_ram(season=SEASON_SUMMER, day=10, weather=WEATHER_HURRICANE)
        )
        task.reset(world_storm)
        self.assertFalse(task._was_storm_yesterday)

        # Plan the storm day: multi-day daytime work is empty, delegating sleep to orchestrator
        day_task = task._build_day_task(world_storm)
        self.assertTrue(task.last_day_decision.facts.is_storm)
        self.assertEqual(day_task.phases, [])
        # Standalone decision includes GO_TO_SLEEP
        standalone = build_day_plan_decision(ram=world_storm.ram)
        self.assertEqual(list(standalone.phase_names), ["GO_TO_SLEEP"])

        # Complete overnight sleep
        task._journal_day_complete(world_storm, sleep_reason="storm night")
        self.assertTrue(task._was_storm_yesterday)
        last_journal = task.day_journal[-1]
        self.assertTrue(last_journal["is_storm"])

        # Next day (sunny) should be recognized as post_storm
        world_next = SimpleNamespace(
            ram=_make_ram(season=SEASON_SUMMER, day=11, weather=WEATHER_SUNNY)
        )
        next_day_task = task._build_day_task(world_next)
        self.assertTrue(task.last_day_decision.facts.is_post_storm)
        phase_names = [p.phase for p in next_day_task.phases]
        self.assertIn("REPAIR_CROPS", phase_names)

        # Overnight from sunny day clears _was_storm_yesterday
        task._journal_day_complete(world_next, sleep_reason="regular night")
        self.assertFalse(task._was_storm_yesterday)
        next_journal = task.day_journal[-1]
        self.assertFalse(next_journal["is_storm"])
        self.assertTrue(next_journal["is_post_storm"])


if __name__ == "__main__":
    unittest.main()

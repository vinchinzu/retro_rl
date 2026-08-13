"""Crop day-plan phase builder tests (build_day_phases, outdoor dynamic).

Split from test_day_plan_crop monofile (LOC soft-max).
"""
from __future__ import annotations

from pathlib import Path
import sys

# Path-stable import of sibling helpers (works under unittest and pytest importlib mode).
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
from day_plan_test_helpers import (
    DayPlanPhaseHelpers,
    make_date_world,
)

import unittest
from unittest.mock import patch

from harvest.planner.day_plan import (
    ADDR_CORN_SEEDS,
    ADDR_WEEKDAY,
    ADDR_WEATHER,
    ADDR_WEATHER_FLAGS,
    DayPlannerPolicy,
    DayPlanTask,
    PhaseSpec,
    TaskStatus,
    auto_day_phases,
    build_day_phases,
    build_day_phases_from_ram,
    build_outdoor_day_phases,
    build_outdoor_day_phases_from_ram,
    is_rainy_weather,
    ram_has_waterable_crops,
)
from harvest.tasks.crop_planter import is_rainy_weather as crop_task_is_rainy_weather
from harvest.core.tile_catalog import ADDR_MAP


class BuildDayPhasesCropTests(DayPlanPhaseHelpers):
    """Tests for the dynamic day plan builder."""

    def test_harvest_before_water(self) -> None:
        phases = build_day_phases(None, hour=16, has_harvest=True, has_waterable=True)
        names = self._phase_names(phases)
        self.assertIn("HARVEST_ROUTE", names)
        self.assertIn("CROP_WATER", names)
        harvest_idx = names.index("HARVEST_ROUTE")
        water_idx = names.index("CROP_WATER")
        self.assertLess(harvest_idx, water_idx)

    def test_harvest_and_water_share_nav_crop(self) -> None:
        phases = build_day_phases(None, hour=16, has_harvest=True, has_waterable=True)
        names = self._phase_names(phases)
        # NAV_CROP should appear once (harvest needs it, water reuses position)
        self.assertEqual(names.count("NAV_CROP"), 1)

    def test_water_only_gets_nav_crop(self) -> None:
        phases = build_day_phases(None, hour=16, has_waterable=True)
        names = self._phase_names(phases)
        self.assertIn("NAV_CROP", names)
        self.assertIn("ENSURE_WATERING_CAN", names)
        self.assertIn("CROP_WATER", names)

    def test_morning_plant_day_orders_establish_ensure_can_then_water(self) -> None:
        """A3: after hoe+seeds plant, re-fetch can before same-day water.

        Carry holds only two tools. Establish uses hoe+seeds; seed bag frees a
        slot once spent, so ENSURE_WATERING_CAN must sit between establish and
        water (no RAM can poke).
        """
        phases = build_day_phases(
            None,
            weekday=3,
            hour=6,
            has_seeds=True,
            has_waterable=False,
            has_harvest=False,
            is_rainy=False,
        )
        names = self._phase_names(phases)

        self.assertIn("ENSURE_CROP_SEEDS", names)
        self.assertIn("CROP_ESTABLISH", names)
        self.assertIn("ENSURE_WATERING_CAN", names)
        self.assertIn("CROP_WATER", names)

        establish_idx = names.index("CROP_ESTABLISH")
        ensure_can_idx = names.index("ENSURE_WATERING_CAN")
        water_idx = names.index("CROP_WATER")
        self.assertLess(names.index("ENSURE_CROP_SEEDS"), establish_idx)
        self.assertLess(establish_idx, ensure_can_idx)
        self.assertLess(ensure_can_idx, water_idx)

        # Contiguous plant→water crop block (nav may sit between ensure can and water).
        crop_block = names[establish_idx : water_idx + 1]
        self.assertEqual(crop_block[0], "CROP_ESTABLISH")
        self.assertEqual(crop_block[1], "ENSURE_WATERING_CAN")
        self.assertEqual(crop_block[-1], "CROP_WATER")
        self.assertNotIn("CROP_ESTABLISH", crop_block[1:])

        water = phases[water_idx]
        self.assertEqual(water.params.get("work_mode"), "water")
        self.assertEqual(water.params.get("refill_bounds"), (3, 10, 62, 60))

        establish = phases[establish_idx]
        self.assertEqual(establish.params.get("work_mode"), "establish")

    def test_plant_with_existing_waterable_still_re_ensures_can_after_establish(self) -> None:
        """Plant pass still steals carry slots; water pass must re-ensure can."""
        phases = build_day_phases(
            None,
            weekday=3,
            hour=8,
            has_seeds=True,
            has_waterable=True,
            has_harvest=False,
            is_rainy=False,
        )
        names = self._phase_names(phases)
        establish_idx = names.index("CROP_ESTABLISH")
        ensure_can_idx = names.index("ENSURE_WATERING_CAN")
        water_idx = names.index("CROP_WATER")
        self.assertLess(establish_idx, ensure_can_idx)
        self.assertLess(ensure_can_idx, water_idx)
        # Only one ensure-can in the crop block (always before water, after plant).
        self.assertEqual(names.count("ENSURE_WATERING_CAN"), 1)

    def test_water_phase_scheduled_for_dry_plots_regardless_of_can_fill(self) -> None:
        """Empty can must not drop CROP_WATER — refill lives inside CropWaterTask."""
        phases = build_day_phases(
            None,
            hour=10,
            has_seeds=False,
            has_waterable=True,
            is_rainy=False,
        )
        names = self._phase_names(phases)
        self.assertIn("ENSURE_WATERING_CAN", names)
        self.assertIn("CROP_WATER", names)
        water = phases[names.index("CROP_WATER")]
        self.assertEqual(water.params.get("work_mode"), "water")
        self.assertEqual(water.params.get("refill_bounds"), (3, 10, 62, 60))

    def test_rainy_day_skips_water_only_crop_phase(self) -> None:
        phases = build_day_phases(None, hour=16, has_waterable=True, is_rainy=True)
        names = self._phase_names(phases)

        self.assertNotIn("ENSURE_WATERING_CAN", names)
        self.assertNotIn("CROP_WATER", names)

    def test_rainy_day_can_still_start_seed_crop_phase_without_watering_can(self) -> None:
        phases = build_day_phases(None, hour=16, has_seeds=True, is_rainy=True)
        names = self._phase_names(phases)

        self.assertIn("ENSURE_CROP_SEEDS", names)
        self.assertIn("NAV_CROP", names)
        self.assertNotIn("ENSURE_WATERING_CAN", names)
        self.assertIn("CROP_ESTABLISH", names)
        self.assertNotIn("CROP_WATER", names)

    def test_seeded_crop_phase_checks_tool_shed_before_field(self) -> None:
        phases = build_day_phases(None, hour=16, has_seeds=True, is_rainy=True)
        names = self._phase_names(phases)

        self.assertLess(names.index("ENSURE_CROP_SEEDS"), names.index("NAV_CROP"))
        self.assertLess(names.index("NAV_CROP"), names.index("CROP_ESTABLISH"))

    def test_rainy_weather_ignores_forecast_weather_field(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=6)
        world.ram[ADDR_WEATHER + 0x4000] = 1

        self.assertFalse(is_rainy_weather(world.ram))
        self.assertFalse(crop_task_is_rainy_weather(world.ram))

    def test_rainy_weather_reads_current_weather_flags(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=16)
        world.ram[ADDR_WEATHER + 0x4000] = 0
        world.ram[ADDR_WEATHER_FLAGS] = 0x02

        self.assertTrue(is_rainy_weather(world.ram))
        self.assertTrue(crop_task_is_rainy_weather(world.ram))

    def test_berries_enabled_by_default_on_sunday_with_seeds(self) -> None:
        phases = build_day_phases(None, weekday=0, hour=8, has_seeds=True)
        names = self._phase_names(phases)
        self.assertIn("BERRY_RUN_WINDOW", names)
        self.assertIn("OPEN_FENCE_GAP", names)
        self.assertIn("SHIP_BERRY_1", names)
        self.assertIn("SHIP_BERRY_2", names)
        self.assertLess(names.index("OPEN_FENCE_GAP"), names.index("SHIP_BERRY_1"))
        self.assertNotIn("EXIT_FARM_WEST", names)
        self.assertNotIn("BUY_SEEDS", names)

    def test_berries_can_be_disabled_by_policy(self) -> None:
        phases = build_day_phases(
            None,
            weekday=0,
            hour=8,
            has_seeds=True,
            policy=DayPlannerPolicy(include_berry_run=False),
        )
        names = self._phase_names(phases)
        self.assertNotIn("BERRY_RUN_WINDOW", names)
        self.assertNotIn("SHIP_BERRY_1", names)

    def test_berries_default_windows_on_sunday_with_seeds(self) -> None:
        phases = build_day_phases(
            None,
            weekday=0,
            hour=8,
            has_seeds=True,
            policy=DayPlannerPolicy(include_berry_run=True),
        )
        names = self._phase_names(phases)
        self.assertIn("BERRY_RUN_WINDOW", names)
        self.assertIn("OPEN_FENCE_GAP", names)
        self.assertIn("SHIP_BERRY_1", names)
        self.assertIn("SHIP_BERRY_2", names)
        ship = phases[names.index("SHIP_BERRY_1")]
        self.assertEqual(ship.kind, "berry_ship")
        self.assertEqual(ship.params["route"], "berry_ship")
        ship2 = phases[names.index("SHIP_BERRY_2")]
        self.assertEqual(ship2.params["route"], "berry_ship_repeat")
        fence = phases[names.index("OPEN_FENCE_GAP")]
        self.assertEqual(fence.kind, "fence_clear")
        self.assertTrue(fence.params.get("corridor_only"))
        self.assertNotIn("BUY_SEEDS", names)
        self.assertEqual(phases[names.index("BERRY_RUN_WINDOW")].params["latest_hour"], 14)

    def test_early_empty_day_berries_before_field_clear(self) -> None:
        """No animals/crops: berry ship first; no daytime bush clear."""
        phases = build_day_phases(
            None,
            weekday=2,
            hour=7,
            has_seeds=False,
            has_debris=True,
            has_chickens=False,
            has_cows=False,
            has_harvest=False,
            has_waterable=False,
            money=0,
        )
        names = self._phase_names(phases)
        self.assertIn("SHIP_BERRY_1", names)
        # Bushes only at night — empty D2 must not thrash CLEAR all day.
        self.assertNotIn("CLEAR_FIELD", names)

    def test_spring_d2_empty_berries_then_potato_if_money(self) -> None:
        """Spring D2: berries ship first, then potato seeds when wallet can pay."""
        rich = build_day_phases(
            None,
            weekday=2,
            hour=6,
            season=0,
            day=2,
            has_seeds=False,
            has_debris=True,
            has_chickens=False,
            has_cows=False,
            has_harvest=False,
            has_waterable=False,
            money=300,
        )
        rich_names = self._phase_names(rich)
        self.assertIn("OPEN_FENCE_GAP", rich_names)
        self.assertIn("SHIP_BERRY_1", rich_names)
        self.assertIn("BUY_SEEDS", rich_names)
        self.assertLess(rich_names.index("OPEN_FENCE_GAP"), rich_names.index("SHIP_BERRY_1"))
        self.assertLess(
            rich_names.index("SHIP_BERRY_1"),
            rich_names.index("BUY_SEEDS"),
        )
        self.assertNotIn("CLEAR_FIELD", rich_names)

        poor = build_day_phases(
            None,
            weekday=2,
            hour=6,
            season=0,
            day=2,
            has_seeds=False,
            has_debris=True,
            has_chickens=False,
            has_cows=False,
            has_harvest=False,
            has_waterable=False,
            money=100,
        )
        poor_names = self._phase_names(poor)
        self.assertIn("SHIP_BERRY_1", poor_names)
        self.assertNotIn("BUY_SEEDS", poor_names)
        self.assertNotIn("CLEAR_FIELD", poor_names)

    def test_evening_clears_bushes_after_shipping_window(self) -> None:
        """Late day: CLEAR_FIELD only after 5pm, before sleep."""
        phases = build_day_phases(
            None,
            weekday=2,
            hour=17,
            season=0,
            day=2,
            has_seeds=False,
            has_debris=True,
            has_chickens=False,
            has_cows=False,
            has_harvest=False,
            has_waterable=False,
            money=300,
        )
        names = self._phase_names(phases)
        self.assertNotIn("SHIP_BERRY_1", names)
        self.assertIn("CLEAR_FIELD", names)
        self.assertIn("RETURN_HOME", names)
        self.assertLess(names.index("CLEAR_FIELD"), names.index("RETURN_HOME"))

    def test_berries_skipped_when_late(self) -> None:
        phases = build_day_phases(
            None,
            weekday=0,
            hour=16,
            has_seeds=True,
            policy=DayPlannerPolicy(include_berry_run=True),
        )
        names = self._phase_names(phases)
        self.assertNotIn("SHIP_BERRY_1", names)

    def test_weekday_morning_no_seeds_buys_seeds(self) -> None:
        phases = build_day_phases(None, weekday=3, hour=6, has_seeds=False, money=300)
        names = self._phase_names(phases)
        self.assertIn("BUY_SEEDS_WINDOW", names)
        self.assertIn("NAV_FARM_EXIT", names)
        self.assertIn("BUY_SEEDS", names)

    def test_summer_morning_buys_summer_seed_recording(self) -> None:
        phases = build_day_phases(
            None,
            weekday=3,
            hour=6,
            season=1,
            day=7,
            has_seeds=False,
            money=500,
        )
        names = self._phase_names(phases)
        self.assertIn("BUY_SEEDS", names)
        buy = phases[names.index("BUY_SEEDS")]
        self.assertEqual(buy.params["recording_name"], "buy_summer")
        self.assertEqual(buy.params["recording_start"], 0)

    def test_fall_morning_skips_seed_shop_and_planting(self) -> None:
        phases = build_day_phases(
            None,
            weekday=3,
            hour=6,
            season=2,
            day=5,
            has_seeds=True,
            has_waterable=False,
            has_harvest=False,
        )
        names = self._phase_names(phases)
        self.assertNotIn("BUY_SEEDS", names)
        self.assertNotIn("ENSURE_CROP_SEEDS", names)
        self.assertNotIn("CROP_WATER", names)

    def test_winter_policy_disables_planting(self) -> None:
        from harvest.planner.day_phase_types import day_planner_policy_for_season

        policy = day_planner_policy_for_season(3)
        self.assertFalse(policy.include_planting)
        self.assertIsNone(policy.seed_purchase_recording)

    def test_seed_buy_runs_after_berries_on_empty_outdoor_day(self) -> None:
        phases = build_outdoor_day_phases(
            weekday=3,
            hour=6,
            has_harvest=False,
            has_waterable=False,
            has_seeds=False,
            has_debris=True,
            money=300,
        )
        names = [phase.phase for phase in phases]
        self.assertIn("SHIP_BERRY_1", names)
        self.assertIn("BUY_SEEDS", names)
        self.assertLess(names.index("SHIP_BERRY_1"), names.index("BUY_SEEDS"))
        # Empty outdoor morning: no day CLEAR (bushes wait for evening).
        self.assertNotIn("CLEAR_FIELD", names)

    def test_full_day_priority_order(self) -> None:
        phases = build_day_phases(
            None,
            weekday=3,
            hour=8,
            has_chickens=True,
            has_cows=True,
            has_harvest=True,
            has_waterable=True,
            has_seeds=True,
            policy=DayPlannerPolicy(include_berry_run=False),
        )
        names = self._phase_names(phases)
        # Verify ordering: exit -> cows -> chickens -> harvest -> crop work.
        exit_idx = names.index("EXIT_TO_FARM")
        cow_idx = names.index("NAV_TO_BARN")
        coop_idx = names.index("NAV_TO_COOP")
        harvest_idx = names.index("HARVEST_ROUTE")
        water_idx = names.index("CROP_WATER")
        self.assertLess(exit_idx, cow_idx)
        self.assertLess(cow_idx, coop_idx)
        self.assertLess(coop_idx, harvest_idx)
        self.assertLess(harvest_idx, water_idx)
        self.assertIn("CROP_WATER", names)
        self.assertNotIn("SHIP_BERRY_1", names)

    def test_early_berry_window_runs_after_crop_work_when_enabled(self) -> None:
        phases = build_day_phases(
            None,
            weekday=6,
            hour=6,
            has_chickens=True,
            has_harvest=True,
            has_waterable=True,
            has_seeds=True,
            policy=DayPlannerPolicy(include_berry_run=True),
        )
        names = self._phase_names(phases)

        self.assertIn("HARVEST_ROUTE", names)
        self.assertIn("SHIP_BERRY_1", names)
        self.assertIn("CROP_WATER", names)
        self.assertLess(names.index("CROP_WATER"), names.index("SHIP_BERRY_1"))

    def test_auto_day_phases_uses_ram_priority_instead_of_resume_water_shortcut(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=6)
        world.ram[ADDR_WEEKDAY + 0x4000] = 6
        world.ram[0x092A + 0x4000] = 5
        world.ram[ADDR_MAP + 34 * 64 + 7] = 0x60

        names = self._phase_names(
            auto_day_phases(
                "fake",
                ram=world.ram,
                policy=DayPlannerPolicy(include_berry_run=False),
            )
        )

        self.assertIn("HARVEST_ROUTE", names)
        self.assertIn("CROP_WATER", names)
        self.assertNotIn("SHIP_BERRY_1", names)

    def test_late_day_forecast_rain_still_waters_dry_crops(self) -> None:
        world = make_date_world(0x00, season=0, day=29, hour=18)
        world.ram[ADDR_WEEKDAY + 0x4000] = 1
        world.ram[ADDR_WEATHER + 0x4000] = 1
        world.ram[ADDR_MAP + 34 * 64 + 7] = 0x5A

        phases = build_day_phases_from_ram(world.ram)
        names = self._phase_names(phases)

        self.assertIn("ENSURE_WATERING_CAN", names)
        self.assertIn("CROP_WATER", names)
        self.assertLess(names.index("CROP_WATER"), names.index("RETURN_HOME"))

    def test_late_day_seed_stock_only_goes_home_to_sleep(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=18)
        world.ram[ADDR_WEEKDAY + 0x4000] = 6
        world.ram[0x092A + 0x4000] = 5

        phases = build_day_phases_from_ram(world.ram)
        names = self._phase_names(phases)

        self.assertNotIn("ENSURE_CROP_SEEDS", names)
        self.assertNotIn("CROP_WATER", names)
        self.assertEqual(names, ["RETURN_HOME", "GO_TO_SLEEP"])

    def test_waterable_detection_skips_known_harvestable_tiles(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=18)
        world.ram[ADDR_MAP + 34 * 64 + 7] = 0x60

        with patch("harvest.planner.day_plan_status.live_harvestable_crop_tiles", return_value=[(7, 34)]):
            self.assertFalse(ram_has_waterable_crops(world.ram, state_name="fake"))

    def test_waterable_detection_ignores_unplanted_and_mature_tiles(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=18)
        world.ram[ADDR_MAP + 34 * 64 + 7] = 0x07
        world.ram[ADDR_MAP + 35 * 64 + 7] = 0x60
        world.ram[ADDR_MAP + 36 * 64 + 7] = 0x61

        self.assertFalse(ram_has_waterable_crops(world.ram))

    def test_day_plan_expands_dynamic_outdoor_phase_from_live_farm_state(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=6)
        world.ram[ADDR_WEEKDAY + 0x4000] = 6
        world.ram[0x092A + 0x4000] = 5
        world.ram[ADDR_MAP + 34 * 64 + 7] = 0x60
        world.ram[ADDR_MAP + 35 * 64 + 7] = 0x58

        plan = DayPlanTask(phase_sequence=[PhaseSpec("DYNAMIC_OUTDOOR_PLAN", "dynamic_outdoor_plan")])
        plan.reset(world)

        result = plan.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(plan.phase_text, "NAV_CROP")
        self.assertEqual([phase.phase for phase in plan.phases], ["DYNAMIC_OUTDOOR_PLAN"])
        self.assertNotEqual([phase.phase for phase in plan.runtime_phases], ["DYNAMIC_OUTDOOR_PLAN"])

    def test_day_plan_expands_dynamic_outdoor_phase_from_seasonal_farm_state(self) -> None:
        world = make_date_world(0x01, season=1, day=1, hour=6)
        world.ram[ADDR_WEEKDAY + 0x4000] = 3
        # Summer needs corn/tomato stock; leftover potato seeds are ignored.
        world.ram[ADDR_CORN_SEEDS + 0x4000] = 5
        world.ram[ADDR_MAP + 34 * 64 + 7] = 0x60
        world.ram[ADDR_MAP + 35 * 64 + 7] = 0x58

        plan = DayPlanTask(phase_sequence=[PhaseSpec("DYNAMIC_OUTDOOR_PLAN", "dynamic_outdoor_plan")])
        plan.reset(world)

        result = plan.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertNotEqual(plan.phase_text, "EXIT_TO_FARM")
        self.assertEqual(plan.phase_text, "NAV_CROP")
        self.assertEqual([phase.phase for phase in plan.phases], ["DYNAMIC_OUTDOOR_PLAN"])

    def test_dynamic_outdoor_phase_uses_state_harvest_fallback(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=6)
        world.ram[ADDR_WEEKDAY + 0x4000] = 3
        world.ram[0x092A + 0x4000] = 5

        with patch("harvest.planner.world_probe.ram_has_harvestable_crops", return_value=False), patch(
            "harvest.planner.world_probe.state_has_harvestable_crops",
            return_value=True,
        ):
            phases = build_outdoor_day_phases_from_ram(world.ram, state_name="pinned_fixture")

        names = self._phase_names(phases)
        self.assertIn("HARVEST_ROUTE", names)
        self.assertIn("CROP_WATER", names)
        self.assertLess(names.index("HARVEST_ROUTE"), names.index("CROP_WATER"))


if __name__ == "__main__":
    unittest.main()

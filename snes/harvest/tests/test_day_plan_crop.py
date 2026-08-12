"""Crop / plant / water / harvest day-plan sequences.

Split from test_day_plan_sequences monofile.
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
    make_navigation_ram,
    make_time_world,
    make_transition_world,
    make_world,
    set_live_chicken_slots,
    set_live_cow_slot,
    set_live_u16,
    set_money,
    set_player_pos,
)

import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from harvest.core.animal_status import CHICKEN_SLOT_BASE, CHICKEN_SLOT_SIZE, COW_DAILY_TALKED_FLAG
from harvest.planner.day_plan import (
    ADDR_CHICKEN_COUNT,
    ADDR_CORN_SEEDS,
    ADDR_COW_COUNT,
    ADDR_DAY,
    ADDR_EGG_AVAILABLE,
    ADDR_FED_CHICKENS_N,
    ADDR_FED_COWS_N,
    ADDR_SEASON,
    ADDR_TILEMAP,
    ADDR_HAY_COUNT,
    ADDR_HOUR,
    ADDR_MINUTE,
    ADDR_MONEY,
    ADDR_WEEKDAY,
    ADDR_WEATHER,
    ADDR_WEATHER_FLAGS,
    ActionResult,
    CHICKEN_PHASES,
    ChickenSaleEventTask,
    ChickenSaleFollowupTask,
    ChickenSaleRequestTask,
    COOP_TILEMAP,
    CoopPickupChickenTask,
    COW_PHASES,
    COW_PURCHASE_COST,
    CrossMapRecordedTask,
    DayPlannerPolicy,
    DayPlanTask,
    DeadlineCheckTask,
    DirectionalTransitionTask,
    DropCarriedChickenTask,
    EnsureAnimalToolsTask,
    EnsureCropSeedsTask,
    EnsureCarryToolTask,
    ExitBuildingTask,
    ExitToFarmTask,
    ShedFetchItemTask,
    EveTalkLoopTask,
    EVE_TALK_LOOP_PHASES,
    GoToSleepTask,
    HARVEST_ROUTE_PHASE,
    ReturnHomeTask,
    ShedShelfToolTask,
    SwapCarrySlotsTask,
    MultiMapNavTask,
    NavTask,
    MultiDayPlannerTask,
    PHASE_SEQUENCES,
    PhaseSpec,
    RecordedTransitionTask,
    SHED_TOOL_SPECS,
    SHED_SEED_SPECS,
    TaskResult,
    TaskStatus,
    WaitUntilTimeTask,
    auto_day_phases,
    auto_day_plan_name_for_ram,
    auto_day_plan_name_for_state,
    auto_day_plan_name_for_weekday,
    build_day_phases,
    build_day_phases_from_ram,
    build_outdoor_day_phases,
    build_outdoor_day_phases_from_ram,
    is_farm_tilemap,
    is_house_tilemap,
    is_rainy_weather,
    make_action,
    romance_points_for_hearts,
    ram_has_waterable_crops,
    state_has_chickens,
    state_has_cows,
)
from harvest.planner.day_task_factory import DayTaskFactory
from harvest.planner.day_plan_decision import DayPlanDecision, DeferredPlan, PlanningFacts
from harvest.planner.world_probe import WorldProbe
from harvest.tasks.crop_planter import is_rainy_weather as crop_task_is_rainy_weather
from harvest.core.tile_catalog import (
    ADDR_INPUT_LOCK,
    ADDR_MAP,
    ADDR_X,
    ADDR_Y,
    Tool,
)
from harvest.tasks.nav import (
    Navigator,
    Pathfinder,
    Point,
)
from harvest.tasks.farm_clearer import TileScanner
from harvest.tasks.harvest_task import ADDR_SHIPPING_MONEY
from harvest.maps.map_config import ROUTES, Waypoint
from harvest.core.ram_catalog import COW_SLOT_BASE, COW_SLOT_SIZE, field_spec


class DayPlanSequenceCropTests(unittest.TestCase):
    def test_berries_water_sequence_keeps_recorded_berry_route_disabled(self) -> None:
        phases = PHASE_SEQUENCES["berries_water"]

        self.assertEqual(
            [phase.phase for phase in phases],
            ["EXIT_TO_FARM", "ENSURE_WATERING_CAN", "NAV_CROP", "CROP_WATER"],
        )
        self.assertEqual(
            [phase.kind for phase in phases],
            ["farm_building_exit", "ensure_tool", "nav", "crop"],
        )
        self.assertNotIn("GET_BERRIES_AND_SHIP", [phase.phase for phase in phases])
        self.assertNotIn("BUY_SEEDS", [phase.phase for phase in phases])

    def test_sunday_sequence_keeps_recorded_berry_route_disabled(self) -> None:
        phases = PHASE_SEQUENCES["sunday"]

        self.assertEqual(
            [phase.phase for phase in phases],
            ["EXIT_TO_FARM", "ENSURE_WATERING_CAN", "NAV_CROP", "CROP_WATER"],
        )
        self.assertEqual(
            [phase.kind for phase in phases],
            ["farm_building_exit", "ensure_tool", "nav", "crop"],
        )
        self.assertNotIn("GET_BERRIES_AND_SHIP", [phase.phase for phase in phases])
        self.assertNotIn("BUY_SEEDS", [phase.phase for phase in phases])

    def test_resume_water_sequence_exits_house_then_routes_to_crop_watering(self) -> None:
        phases = PHASE_SEQUENCES["resume_water"]

        self.assertEqual(
            [phase.phase for phase in phases],
            ["EXIT_TO_FARM", "ENSURE_WATERING_CAN", "NAV_CROP", "CROP_WATER"],
        )
        self.assertEqual(
            [phase.kind for phase in phases],
            ["farm_building_exit", "ensure_tool", "nav", "crop"],
        )
        self.assertEqual(phases[2].params["target_px"], (248, 472))
        self.assertEqual(phases[3].params["refill_bounds"], (3, 10, 62, 60))

    def test_harvest_sequence_uses_trimmed_recording_only(self) -> None:
        phases = PHASE_SEQUENCES["harvest"]

        self.assertEqual(
            [phase.phase for phase in phases],
            ["EXIT_TO_FARM", "NAV_CROP", "HARVEST_ROUTE"],
        )
        self.assertEqual(
            [phase.kind for phase in phases],
            ["farm_building_exit", "nav", "harvest"],
        )

    def test_auto_day_plan_name_uses_sunday_for_sunday_weekday(self) -> None:
        self.assertEqual(auto_day_plan_name_for_weekday(0), "sunday")
        self.assertEqual(auto_day_plan_name_for_weekday(4), "day1")

    def test_auto_day_plan_name_prefers_berry_route_when_seed_stock_exists(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        ram[ADDR_WEEKDAY] = 1
        ram[ADDR_HOUR] = 6
        ram[ADDR_MINUTE] = 0
        ram[0x092A] = 3
        fake_state = SimpleNamespace(ram=ram)

        with patch("harvest.planner.day_plan_status.resolve_state_path", return_value="/tmp/fake.state"), patch(
            "harvest.planner.day_plan_status.parse_save_state",
            return_value=fake_state,
        ):
            self.assertEqual(auto_day_plan_name_for_state("fake"), "berries_water")

    def test_auto_day_plan_name_uses_resume_water_for_non_morning_state(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        ram[ADDR_WEEKDAY] = 1
        ram[ADDR_HOUR] = 15
        ram[ADDR_MINUTE] = 0
        ram[0x092A] = 3
        fake_state = SimpleNamespace(ram=ram)

        with patch("harvest.planner.day_plan_status.resolve_state_path", return_value="/tmp/fake.state"), patch(
            "harvest.planner.day_plan_status.parse_save_state",
            return_value=fake_state,
        ):
            self.assertEqual(auto_day_plan_name_for_state("fake"), "resume_water")

    def test_auto_day_plan_name_prefers_harvest_when_mature_crops_exist(self) -> None:
        with patch.object(WorldProbe, "day_time", return_value=(0, 6, 0)), patch.object(
            WorldProbe, "has_harvestable_crops", return_value=True
        ), patch.object(WorldProbe, "has_waterable_crops", return_value=False):
            self.assertEqual(auto_day_plan_name_for_state("fake"), "harvest")

    def test_auto_day_plan_name_prefers_harvest_over_resume_water(self) -> None:
        with patch.object(WorldProbe, "day_time", return_value=(0, 6, 0)), patch.object(
            WorldProbe, "has_harvestable_crops", return_value=True
        ), patch.object(WorldProbe, "has_waterable_crops", return_value=True), patch.object(
            WorldProbe, "is_rainy", return_value=False
        ):
            self.assertEqual(auto_day_plan_name_for_state("fake"), "harvest")

    def test_auto_day_plan_name_keeps_harvest_priority_later_in_day_before_cutoff(self) -> None:
        with patch.object(WorldProbe, "day_time", return_value=(0, 8, 30)), patch.object(
            WorldProbe, "has_harvestable_crops", return_value=True
        ), patch.object(WorldProbe, "has_waterable_crops", return_value=False):
            self.assertEqual(auto_day_plan_name_for_state("fake"), "harvest")

    def test_auto_day_plan_name_for_ram_prefers_resume_water_when_tiles_need_water(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        ram[ADDR_WEEKDAY + 0x4000] = 2
        ram[ADDR_HOUR + 0x4000] = 18
        ram[ADDR_MINUTE + 0x4000] = 5
        ram[ADDR_MAP + 35 * 64 + 11] = 0x58

        self.assertEqual(auto_day_plan_name_for_ram(ram, fallback_state_name="fake"), "resume_water")

    def test_day_plan_skips_berry_route_on_harvest_failure(self) -> None:
        class FailTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.FAILURE, reason="incomplete harvest")

        class ActionTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(right=True)))

        class TestPlan(DayPlanTask):
            def __init__(self) -> None:
                super().__init__(
                    phase_sequence=[
                        HARVEST_ROUTE_PHASE,
                        PhaseSpec("BERRY_RUN_WINDOW", "deadline"),
                        PhaseSpec("EXIT_FARM_WEST", "directional_transition"),
                        PhaseSpec("ENSURE_WATERING_CAN", "ensure_tool"),
                    ]
                )
                self._tasks = [FailTask(), ActionTask()]

            def _make_task(self, spec, world):
                return self._tasks.pop(0)

        plan = TestPlan()
        world = make_world(0x00)
        plan.reset(world)

        result = plan.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(plan.phase_text, "ENSURE_WATERING_CAN")
        self.assertEqual(
            [item.phase for item in plan.deferred_plans],
            ["HARVEST_ROUTE", "BERRY_RUN_WINDOW", "EXIT_FARM_WEST"],
        )
        self.assertTrue(all(item.reason == "incomplete harvest" for item in plan.deferred_plans))

    def test_day_plan_retries_crop_water_with_targeted_watering_can_recovery(self) -> None:
        class MissingWateringCanTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.FAILURE, reason="watering can not in carry pair")

        class TestPlan(DayPlanTask):
            def __init__(self) -> None:
                super().__init__(phase_sequence=[PhaseSpec("CROP_WATER", "crop")])

            def _make_task(self, spec, world):
                return MissingWateringCanTask()

        plan = TestPlan()
        world = make_world(0x00)
        world.ram[0x0921] = int(Tool.BRUSH)
        world.ram[0x0923] = int(Tool.MILKER)
        plan.reset(world)

        result = plan.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsInstance(plan._recovery_task, EnsureCarryToolTask)
        self.assertEqual(plan.phase_text, "CROP_WATER")

    def test_nav_crop_factory_uses_dynamic_crop_target(self) -> None:
        spec = PhaseSpec("NAV_CROP", "nav", {"target_px": (136, 520)})

        with patch(
            "harvest.planner.day_phase_registry.crop_nav_target_px",
            return_value=(72, 296),
        ):
            task = DayTaskFactory(state_name="latest").make_task(spec, make_transition_world(0x00))

        self.assertIsInstance(task, NavTask)
        self.assertEqual((task.target_px.x, task.target_px.y), (72, 296))

    def test_watering_can_shed_spec_uses_shelf_coordinates(self) -> None:
        spec = SHED_TOOL_SPECS[int(Tool.WATERING_CAN)]

        self.assertEqual(spec.farm_route, "farm_to_shed")
        self.assertEqual(spec.nav_target_px, (422, 474))
        self.assertEqual(spec.inside_stand_px, (96, 168))
        self.assertEqual(spec.inside_face, "up")
        self.assertIsNone(spec.inside_recording)

    def test_ensure_crop_seeds_is_noop_when_seed_and_hoe_held(self) -> None:
        world = make_world(0x00)
        world.ram[0x0921] = 0x07
        world.ram[0x0923] = int(Tool.HOE)
        task = EnsureCropSeedsTask(seed_type="potato")

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "seed tool 0x07 ready")

    def test_ensure_crop_seeds_grabs_hoe_first_when_stock_exists(self) -> None:
        world = make_world(0x00)
        world.ram[0x0921] = 0x00
        world.ram[0x0923] = 0x00
        world.ram[0x092A] = 10
        task = EnsureCropSeedsTask(seed_type="potato")

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "ensure_hoe")
        self.assertIsInstance(task._task, EnsureCarryToolTask)
        self.assertEqual(task._task.tool_id, int(Tool.HOE))

    def test_ensure_crop_seeds_swaps_hoe_off_selected_before_seed_fetch(self) -> None:
        """Shelf A replaces selected; hoe must be backpack before seed grab (rr-6byj)."""
        world = make_world(0x00)
        world.ram[0x0921] = int(Tool.HOE)
        world.ram[0x0923] = int(Tool.WATERING_CAN)
        world.ram[0x092A] = 10
        task = EnsureCropSeedsTask(seed_type="potato")

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "swap_preserve_hoe")
        self.assertIsInstance(task._task, SwapCarrySlotsTask)

    def test_ensure_crop_seeds_swaps_seed_off_selected_before_hoe_fetch(self) -> None:
        """If seeds are already held and selected, preserve them before hoe shelf."""
        world = make_world(0x00)
        world.ram[0x0921] = 0x07  # potato selected
        world.ram[0x0923] = int(Tool.WATERING_CAN)
        world.ram[0x092A] = 10
        task = EnsureCropSeedsTask(seed_type="potato")

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "swap_preserve_seed")
        self.assertIsInstance(task._task, SwapCarrySlotsTask)

    def test_ensure_crop_seeds_fetch_when_hoe_in_backpack(self) -> None:
        """Hoe already in backpack → selected is disposable → go straight to seed route."""
        world = make_world(0x00)
        world.ram[0x0921] = int(Tool.WATERING_CAN)
        world.ram[0x0923] = int(Tool.HOE)
        world.ram[0x092A] = 10
        task = EnsureCropSeedsTask(seed_type="potato")

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "fetch_seed")
        self.assertIsInstance(task._task, ShedFetchItemTask)
        self.assertEqual(task._task._phase, "route")
        self.assertIsInstance(task._task._task, MultiMapNavTask)

    def test_ensure_crop_seeds_caps_shed_trips_against_carry_thrash(self) -> None:
        """Re-fetch loop must fail cleanly instead of multi_nav hang (rr-6byj)."""
        world = make_world(0x00)
        world.ram[0x0921] = 0x00
        world.ram[0x0923] = 0x00
        world.ram[0x092A] = 10
        task = EnsureCropSeedsTask(seed_type="potato", max_shed_trips=2)
        task.reset(world)
        # Simulate prior trips without completing plant tools.
        task._shed_trips = 2
        result = task._start_next_phase(world)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("carry thrash", result.reason or "")

    def test_ensure_crop_seeds_uses_tool_shed_route_when_hoe_ready(self) -> None:
        world = make_world(0x00)
        world.ram[0x0921] = int(Tool.HOE)
        world.ram[0x0923] = 0x00
        world.ram[0x092A] = 10
        task = EnsureCropSeedsTask(seed_type="potato")

        task.reset(world)
        result = task.step(world)

        # Hoe selected + empty backpack: no X-swap (empty cannot be selected);
        # open backpack slot absorbs shelf item → go straight to seed route.
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "fetch_seed")
        self.assertIsInstance(task._task, ShedFetchItemTask)
        self.assertEqual(task._task._phase, "route")
        self.assertIsInstance(task._task._task, MultiMapNavTask)

    def test_ensure_crop_seeds_uses_field_route_after_harvest_shipping(self) -> None:
        world = make_world(0x00)
        set_player_pos(world.ram, 11 * 16 + 8, 30 * 16 + 8)
        world.ram[0x0921] = int(Tool.HOE)
        world.ram[0x0923] = 0x00
        world.ram[0x092A] = 10
        task = EnsureCropSeedsTask(seed_type="potato")

        task.reset(world)
        result = task._start_shed_seed_fetch(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "fetch_seed")
        self.assertIsInstance(task._task, ShedFetchItemTask)
        self.assertEqual(task._task._phase, "route")
        self.assertIsInstance(task._task._task, MultiMapNavTask)
        self.assertEqual(task._task._task.waypoints, ROUTES["field_to_shed"])

    def test_ensure_crop_seeds_uses_upper_route_from_coop_frontage(self) -> None:
        world = make_world(0x00)
        set_player_pos(world.ram, 28 * 16 + 8, 22 * 16 + 8)
        world.ram[0x0921] = int(Tool.HOE)
        world.ram[0x0923] = 0x00
        world.ram[0x092A] = 10
        task = EnsureCropSeedsTask(seed_type="potato")
        result = task._start_shed_seed_fetch(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsInstance(task._task, ShedFetchItemTask)
        self.assertIsInstance(task._task._task, MultiMapNavTask)
        self.assertEqual(task._task._task.waypoints, ROUTES["upper_farm_to_shed"])

    def test_ensure_crop_seeds_uses_field_route_from_lower_farm_edge(self) -> None:
        world = make_world(0x00)
        set_player_pos(world.ram, 1 * 16 + 8, 29 * 16 + 8)
        world.ram[0x0921] = int(Tool.HOE)
        world.ram[0x0923] = 0x00
        world.ram[0x092A] = 10
        task = EnsureCropSeedsTask(seed_type="potato")
        result = task._start_shed_seed_fetch(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsInstance(task._task, ShedFetchItemTask)
        self.assertIsInstance(task._task._task, MultiMapNavTask)
        self.assertEqual(task._task._task.waypoints, ROUTES["field_to_shed"])

    def test_potato_seed_shed_spec_uses_upper_shelf_stand(self) -> None:
        spec = SHED_SEED_SPECS["potato"]

        self.assertEqual(spec.farm_route, "farm_to_shed")
        self.assertEqual(spec.nav_target_px, (422, 474))
        self.assertEqual(spec.inside_stand_px, (190, 118))
        self.assertIsNone(spec.inside_recording)

    def test_ensure_crop_seeds_does_not_restore_watering_can(self) -> None:
        """Seeds must stay in the carry pair for the following plant phase."""
        world = make_world(0x00)
        world.ram[0x0921] = 0x07
        world.ram[0x0923] = int(Tool.HOE)
        task = EnsureCropSeedsTask(seed_type="potato")
        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertNotEqual(getattr(task, "_phase", ""), "restore_watering_can")
        self.assertEqual(int(world.ram[0x0921]), 0x07)

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

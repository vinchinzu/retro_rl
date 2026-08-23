"""Crop day-plan sequence tests (PHASE_SEQUENCES, ensure seeds, recovery).

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
    make_date_world,
    make_transition_world,
    make_world,
    set_player_pos,
)

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from harvest.planner.day_plan import (
    ADDR_HOUR,
    ADDR_MINUTE,
    ADDR_POTATO_SEEDS,
    ADDR_WEEKDAY,
    ActionResult,
    BUY_SEEDS_PHASE,
    CLEAR_FIELD_PHASE,
    DayPlannerPolicy,
    DayPlanTask,
    EnsureCarryToolTask,
    EnsureCropSeedsTask,
    HARVEST_ROUTE_PHASE,
    MultiMapNavTask,
    NavTask,
    PHASE_SEQUENCES,
    PhaseSpec,
    SHED_SEED_SPECS,
    SHED_TOOL_SPECS,
    SwapCarrySlotsTask,
    ShedFetchItemTask,
    TaskResult,
    TaskStatus,
    auto_day_plan_name_for_ram,
    auto_day_plan_name_for_state,
    auto_day_plan_name_for_weekday,
    make_action,
)
from harvest.core.ram_catalog import live_wram_base
from harvest.planner.day_task_factory import DayTaskFactory
from harvest.planner.world_probe import WorldProbe
from harvest.core.tile_catalog import (
    ADDR_MAP,
    Tool,
)
from harvest.maps.map_config import ROUTES
from harvest.planner.tasks.inventory_shed import shed_farm_route_name


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

    def test_buy_seeds_success_splices_same_day_plant(self) -> None:
        """rr-20w.1: morning outdoor plan is frozen before the bag exists."""

        class InstantBuy:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(
                    status=TaskStatus.SUCCESS,
                    reason="bought potato_seeds 0->1 money 300->100",
                )

        class HoldTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        class TestPlan(DayPlanTask):
            def __init__(self) -> None:
                super().__init__(
                    phase_sequence=[BUY_SEEDS_PHASE, CLEAR_FIELD_PHASE],
                    policy=DayPlannerPolicy(include_end_day=False),
                )

            def _make_task(self, spec, world):
                if spec.phase == "BUY_SEEDS":
                    return InstantBuy()
                return HoldTask()

        world = make_date_world(0x00, season=0, day=2, hour=12)
        world.ram[ADDR_POTATO_SEEDS + live_wram_base(world.ram)] = 1
        plan = TestPlan()
        plan.reset(world)
        result = plan.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        names = [phase.phase for phase in plan._schedule.active]
        self.assertEqual(
            names[:5],
            [
                "BUY_SEEDS",
                "ENSURE_CROP_SEEDS",
                "CLEAR_PLOT",
                "NAV_CROP",
                "CROP_ESTABLISH",
            ],
        )
        self.assertIn("CROP_WATER", names)
        self.assertNotIn("CLEAR_FIELD", names)
        self.assertLess(names.index("ENSURE_CROP_SEEDS"), names.index("CLEAR_PLOT"))
        self.assertLess(names.index("CLEAR_PLOT"), names.index("CROP_ESTABLISH"))
        self.assertLess(names.index("CROP_ESTABLISH"), names.index("CROP_WATER"))
        self.assertLess(names.index("CROP_WATER"), names.index("CLEAR_BUSHES"))
        self.assertLess(names.index("CLEAR_BUSHES"), names.index("ENSURE_HAMMER"))
        self.assertLess(names.index("ENSURE_HAMMER"), names.index("CLEAR_ROCKS"))
        self.assertLess(names.index("CLEAR_ROCKS"), names.index("ENSURE_AXE"))
        self.assertLess(names.index("ENSURE_AXE"), names.index("CLEAR_STUMPS"))
        rocks = next(p for p in plan._schedule.active if p.phase == "CLEAR_ROCKS")
        self.assertEqual(rocks.params.get("handoff"), "quota")
        self.assertEqual(rocks.params.get("quota", {}).get("small_rocks"), 10)
        self.assertEqual(rocks.params.get("quota", {}).get("large_rocks"), 4)
        stumps = next(p for p in plan._schedule.active if p.phase == "CLEAR_STUMPS")
        self.assertEqual(stumps.params.get("quota", {}).get("stumps"), 2)
        self.assertEqual(plan.phase_text, "ENSURE_CROP_SEEDS")
        clear = next(p for p in plan._schedule.active if p.phase == "CLEAR_PLOT")
        self.assertEqual(clear.params.get("farm_bounds"), (3, 14, 28, 30))
        self.assertTrue(
            all(phase.failure_policy == "optional" for phase in plan._schedule.active[1:])
        )

    def test_buy_seeds_does_not_splice_plant_without_bag(self) -> None:
        class InstantBuy:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.SUCCESS, reason="shop miss")

        class TestPlan(DayPlanTask):
            def __init__(self) -> None:
                super().__init__(
                    phase_sequence=[BUY_SEEDS_PHASE, CLEAR_FIELD_PHASE],
                    policy=DayPlannerPolicy(include_end_day=False),
                )

            def _make_task(self, spec, world):
                if spec.phase == "BUY_SEEDS":
                    return InstantBuy()
                return None

        world = make_date_world(0x00, season=0, day=2, hour=12)
        plan = TestPlan()
        plan.reset(world)
        plan.step(world)
        names = [phase.phase for phase in plan._schedule.active]
        self.assertEqual(names, ["BUY_SEEDS", "CLEAR_FIELD"])

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
        self.assertFalse(task._task.exit_when_done)

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

    def test_ensure_crop_seeds_fetches_seed_inside_when_already_in_shed(self) -> None:
        """Hoe already in carry inside the shed → seed shelf without exiting."""
        world = make_world(0x26)
        world.ram[0x0921] = int(Tool.HOE)
        world.ram[0x0923] = 0x00
        world.ram[0x092A] = 10
        task = EnsureCropSeedsTask(seed_type="potato")

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "fetch_seed")
        self.assertIsInstance(task._task, ShedFetchItemTask)
        self.assertEqual(task._task._phase, "inside")

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

    def test_shed_farm_route_skips_field_path_north_of_fence(self) -> None:
        world = make_world(0x00)
        set_player_pos(world.ram, 11 * 16 + 8, 30 * 16 + 8)
        self.assertEqual(shed_farm_route_name(world.ram, "farm_to_shed"), "farm_to_shed")
        set_player_pos(world.ram, 11 * 16 + 8, 33 * 16 + 8)
        self.assertEqual(shed_farm_route_name(world.ram, "farm_to_shed"), "field_to_shed")

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

    def test_ensure_crop_seeds_uses_house_route_from_plant_pocket(self) -> None:
        """y=30 is still north of the y=31 fence — field_to_shed L/R-thrashes."""
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
        self.assertEqual(task._task._task.waypoints, ROUTES["farm_to_shed"])

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

    def test_ensure_crop_seeds_uses_house_route_from_west_pocket_edge(self) -> None:
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
        self.assertEqual(task._task._task.waypoints, ROUTES["farm_to_shed"])

    def test_ensure_crop_seeds_uses_field_route_south_of_fence(self) -> None:
        world = make_world(0x00)
        set_player_pos(world.ram, 11 * 16 + 8, 33 * 16 + 8)
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


if __name__ == "__main__":
    unittest.main()

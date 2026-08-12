"""Coop / barn / chicken-sale day-plan sequences.

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


class DayPlanSequenceCoopTests(unittest.TestCase):
    def test_buy_cow_sequence_splits_nav_menu_name_and_barn_chores(self) -> None:
        phases = PHASE_SEQUENCES["buy_cow"]

        self.assertEqual(
            [phase.phase for phase in phases],
            [
                "EXIT_TO_FARM",
                "NAV_TO_ANIMAL_SHOP",
                "BUY_COW_VENDOR",
                "EXIT_ANIMAL_SHOP",
                "RETURN_FARM_AFTER_COW_PURCHASE",
                "NAME_COW",
                "ENSURE_ANIMAL_TOOLS",
                "NAV_TO_BARN",
                "ENTER_BARN",
                "COW_CHORES",
            ],
        )
        self.assertEqual(phases[2].kind, "cow_purchase")
        self.assertEqual(phases[2].params["start_frame"], 1631)
        self.assertEqual(phases[5].params["end_frame"], 5001)

    def test_sell_chicken_test_sequence_is_only_required_sale_work(self) -> None:
        phases = PHASE_SEQUENCES["sell_chicken_test"]

        self.assertEqual(
            [phase.phase for phase in phases],
            [
                "NAV_TO_COOP_FOR_SALE_TEST",
                "ENTER_COOP_FOR_SALE_TEST",
                "PICKUP_CHICKEN_FOR_SALE_TEST",
                "EXIT_COOP_FOR_SALE_TEST",
                "DROP_CHICKEN_FOR_SALE_TEST",
                "NAV_TO_ANIMAL_SHOP_FOR_SALE_TEST",
                "REQUEST_CHICKEN_SALE_TEST",
                "EXIT_ANIMAL_SHOP_AFTER_CHICKEN_SALE_TEST",
                "RETURN_FARM_AFTER_CHICKEN_SALE_TEST",
                "SELL_CHICKEN_TEST",
            ],
        )
        self.assertEqual(
            [phase.kind for phase in phases],
            [
                "multi_nav",
                "directional_transition",
                "pickup_chicken",
                "directional_transition",
                "drop_chicken",
                "multi_nav",
                "chicken_sale_request",
                "multi_nav",
                "multi_nav",
                "chicken_sale_event",
            ],
        )
        self.assertTrue(all(phase.failure_policy == "required" for phase in phases))

    def test_sell_three_chickens_test_repeats_required_sale_cycle(self) -> None:
        phases = PHASE_SEQUENCES["sell_three_chickens_test"]

        self.assertEqual(len(phases), 30)
        self.assertEqual([phase.phase for phase in phases[0:10]], [
            "NAV_TO_COOP_FOR_SALE_TEST_1",
            "ENTER_COOP_FOR_SALE_TEST_1",
            "PICKUP_CHICKEN_FOR_SALE_TEST_1",
            "EXIT_COOP_FOR_SALE_TEST_1",
            "DROP_CHICKEN_FOR_SALE_TEST_1",
            "NAV_TO_ANIMAL_SHOP_FOR_SALE_TEST_1",
            "REQUEST_CHICKEN_SALE_TEST_1",
            "EXIT_ANIMAL_SHOP_AFTER_CHICKEN_SALE_TEST_1",
            "RETURN_FARM_AFTER_CHICKEN_SALE_TEST_1",
            "SELL_CHICKEN_TEST_1",
        ])
        self.assertEqual([phase.phase for phase in phases[-10:]], [
            "NAV_TO_COOP_FOR_SALE_TEST_3",
            "ENTER_COOP_FOR_SALE_TEST_3",
            "PICKUP_CHICKEN_FOR_SALE_TEST_3",
            "EXIT_COOP_FOR_SALE_TEST_3",
            "DROP_CHICKEN_FOR_SALE_TEST_3",
            "NAV_TO_ANIMAL_SHOP_FOR_SALE_TEST_3",
            "REQUEST_CHICKEN_SALE_TEST_3",
            "EXIT_ANIMAL_SHOP_AFTER_CHICKEN_SALE_TEST_3",
            "RETURN_FARM_AFTER_CHICKEN_SALE_TEST_3",
            "SELL_CHICKEN_TEST_3",
        ])

    def test_sell_three_chickens_batch_test_stages_before_shop(self) -> None:
        phases = PHASE_SEQUENCES["sell_three_chickens_batch_test"]

        self.assertEqual(len(phases), 22)
        self.assertEqual([phase.phase for phase in phases[:15:5]], [
            "NAV_TO_COOP_FOR_SALE_BATCH_TEST_STAGE_1",
            "NAV_TO_COOP_FOR_SALE_BATCH_TEST_STAGE_2",
            "NAV_TO_COOP_FOR_SALE_BATCH_TEST_STAGE_3",
        ])
        self.assertEqual(
            [phase.phase for phase in phases[15:]],
            [
                "NAV_TO_ANIMAL_SHOP_FOR_SALE_BATCH_TEST",
                "REQUEST_CHICKEN_SALE_BATCH_TEST_1",
                "REQUEST_CHICKEN_SALE_BATCH_TEST_2",
                "REQUEST_CHICKEN_SALE_BATCH_TEST_3",
                "EXIT_ANIMAL_SHOP_AFTER_CHICKEN_SALE_BATCH_TEST",
                "RETURN_FARM_AFTER_CHICKEN_SALE_BATCH_TEST",
                "SELL_CHICKEN_BATCH_TEST",
            ],
        )
        self.assertEqual(phases[-1].params["target_sales"], 3)
        self.assertEqual(phases[-1].params["event_hour"], 0)
        self.assertTrue(all(phase.failure_policy == "required" for phase in phases))

    def test_chicken_sale_phase_factory_builds_subtasks(self) -> None:
        factory = DayTaskFactory()
        world = make_world(0x00)
        phases = PHASE_SEQUENCES["sell_chicken_test"]

        self.assertIsInstance(factory.make_task(phases[0], world), MultiMapNavTask)
        self.assertIsInstance(factory.make_task(phases[2], world), CoopPickupChickenTask)
        self.assertIsInstance(factory.make_task(phases[4], world), DropCarriedChickenTask)
        self.assertIsInstance(factory.make_task(phases[5], world), MultiMapNavTask)
        self.assertIsInstance(factory.make_task(phases[6], world), ChickenSaleRequestTask)
        self.assertIsInstance(factory.make_task(phases[8], world), MultiMapNavTask)
        self.assertIsInstance(factory.make_task(phases[9], world), ChickenSaleEventTask)

        batch_event = factory.make_task(PHASE_SEQUENCES["sell_three_chickens_batch_test"][-1], world)
        self.assertIsInstance(batch_event, ChickenSaleEventTask)
        self.assertEqual(batch_event.target_sales, 3)

    def test_chicken_sale_request_ignores_stale_request_text(self) -> None:
        world = make_world(0x24)
        set_player_pos(world.ram, 201, 158)
        text_addr = field_spec("dialog_text_id").address
        world.ram[text_addr] = 0x0B
        world.ram[text_addr + 1] = 0x03

        task = ChickenSaleRequestTask(timeout=40)
        task.reset(world)

        result = None
        for _ in range(20):
            result = task.step(world)

        self.assertIsNotNone(result)
        self.assertNotEqual(result.status, TaskStatus.SUCCESS)

    def test_chicken_sale_followup_stops_after_verified_sale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with open(f"{tmp}/sell_chicken.json", "w") as f:
                json.dump({"name": "sell_chicken", "frames": [[0] * 12 for _ in range(5)]}, f)

            world = make_date_world(0x00, season=0, day=10)
            set_player_pos(world.ram, 60, 480)
            base = 0x4000
            world.ram[ADDR_CHICKEN_COUNT + base] = 6
            world.ram[ADDR_SHIPPING_MONEY + base] = 65

            task = ChickenSaleFollowupTask(tasks_dir=tmp, start_frame=0, success_settle_frames=2)
            task.reset(world)

            world.ram[ADDR_CHICKEN_COUNT + base] = 5
            world.ram[ADDR_SHIPPING_MONEY + base] = 80
            first = task.step(world)
            second = task.step(world)

            self.assertEqual(first.status, TaskStatus.RUNNING)
            self.assertEqual(second.status, TaskStatus.SUCCESS)
            self.assertIn("chickens 6->5", second.reason)

    def test_chicken_sale_event_stops_after_verified_sale_money(self) -> None:
        world = make_date_world(0x00, season=0, day=25, hour=15, minute=4)
        base = 0x4000
        world.ram[ADDR_CHICKEN_COUNT + base] = 6
        set_money(world.ram, 7830)

        task = ChickenSaleEventTask(success_settle_frames=2)
        task.reset(world)

        world.ram[ADDR_CHICKEN_COUNT + base] = 5
        set_money(world.ram, 8330)
        first = task.step(world)
        second = task.step(world)

        self.assertEqual(first.status, TaskStatus.RUNNING)
        self.assertEqual(second.status, TaskStatus.SUCCESS)
        self.assertIn("chickens 6->5", second.reason)
        self.assertIn("money 7830->8330", second.reason)

    def test_barn_exit_uses_right_aisle_from_upper_stalls(self) -> None:
        world = make_transition_world(0x27)
        set_player_pos(world.ram, 163, 153)
        task = DirectionalTransitionTask(
            direction="down",
            origin_tilemap=0x27,
            target_tilemap=0x00,
            stand_tile=(8, 22),
            stand_tolerance=1,
        )

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[7]), 1)
        self.assertEqual(int(result.action.action[5]), 0)
        self.assertEqual(int(result.action.action[6]), 0)
        self.assertEqual(int(result.action.action[0]), 1)

    def test_barn_exit_moves_from_lower_cow_lane_to_door(self) -> None:
        world = make_transition_world(0x27)
        set_player_pos(world.ram, 163, 344)
        task = DirectionalTransitionTask(
            direction="down",
            origin_tilemap=0x27,
            target_tilemap=0x00,
            stand_tile=(8, 22),
            stand_tolerance=1,
        )

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[6]), 1)
        self.assertEqual(int(result.action.action[5]), 0)
        self.assertEqual(int(result.action.action[0]), 1)

    def test_barn_exit_turns_left_from_right_aisle_crossing(self) -> None:
        world = make_transition_world(0x27)
        set_player_pos(world.ram, 201, 329)
        task = DirectionalTransitionTask(
            direction="down",
            origin_tilemap=0x27,
            target_tilemap=0x00,
            stand_tile=(8, 22),
            stand_tolerance=1,
        )

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[6]), 1)
        self.assertEqual(int(result.action.action[5]), 0)
        self.assertEqual(int(result.action.action[0]), 1)

    def test_coop_exit_climbs_above_false_open_before_crossing_east(self) -> None:
        world = make_transition_world(0x28)
        set_player_pos(world.ram, 57, 184)
        task = DirectionalTransitionTask(
            direction="down",
            origin_tilemap=0x28,
            target_tilemap=0x00,
            stand_tile=(8, 12),
            stand_tolerance=1,
        )

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        # From left service lane, climb above the false-open band first.
        self.assertEqual(int(result.action.action[4]), 1)
        self.assertEqual(int(result.action.action[0]), 1)

    def test_coop_exit_climbs_out_of_shipping_bin_alcove(self) -> None:
        world = make_transition_world(0x28)
        set_player_pos(world.ram, 22, 169)
        task = DirectionalTransitionTask(
            direction="down",
            origin_tilemap=0x28,
            target_tilemap=0x00,
            stand_tile=(8, 12),
            stand_tolerance=1,
        )

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        # First regain the left lane at x=38, then climb.
        self.assertEqual(int(result.action.action[7]), 1)
        self.assertEqual(int(result.action.action[0]), 1)

    def test_barn_exit_uses_bypass_when_right_aisle_is_blocked(self) -> None:
        world = make_transition_world(0x27)
        set_player_pos(world.ram, 201, 267)
        task = DirectionalTransitionTask(
            direction="down",
            origin_tilemap=0x27,
            target_tilemap=0x00,
            stand_tile=(8, 22),
            stand_tolerance=1,
        )

        task.reset(world)
        task._navigator.current_pos = Point(201, 267)
        task._navigator.stasis = 46
        first = task.step(world)

        self.assertEqual(first.status, TaskStatus.RUNNING)
        self.assertIsNotNone(first.action)
        self.assertEqual(int(first.action.action[7]), 1)
        self.assertEqual(int(first.action.action[5]), 0)
        self.assertTrue(task._barn_exit_bypass)

        set_player_pos(world.ram, 216, 267)
        second = task.step(world)

        self.assertEqual(second.status, TaskStatus.RUNNING)
        self.assertIsNotNone(second.action)
        self.assertEqual(int(second.action.action[5]), 1)
        self.assertEqual(int(second.action.action[7]), 0)

    def test_farm_to_barn_route_starts_south_of_house_exit(self) -> None:
        route = ROUTES["farm_to_barn"]

        self.assertEqual(route[0].target_px, (137, 375))
        self.assertNotIn((137, 300), [waypoint.target_px for waypoint in route])
        self.assertTrue(all(px != 137 or py >= 344 for px, py in [waypoint.target_px for waypoint in route]))

    def test_barn_to_coop_phase_skips_initial_settle_backtrack(self) -> None:
        plan = DayPlanTask(
            phase_sequence=[
                PhaseSpec(
                    "NAV_TO_COOP",
                    "multi_nav",
                    {"route": "barn_to_coop", "initial_settle_frames": 0},
                )
            ]
        )
        world = make_world(0x00)
        set_player_pos(world.ram, 20 * 16 + 8, 22 * 16 + 8)

        plan.reset(world)
        plan.step(world)

        task = plan.current_task
        self.assertIsInstance(task, MultiMapNavTask)
        self.assertEqual(task.initial_settle_frames, 0)
        self.assertEqual(task.waypoints, ROUTES["barn_to_coop"])

    def test_animal_tool_shed_specs_use_shelf_coordinates(self) -> None:
        milker = SHED_TOOL_SPECS[int(Tool.MILKER)]
        brush = SHED_TOOL_SPECS[int(Tool.BRUSH)]

        self.assertEqual(int(Tool.MILKER), 0x0E)
        self.assertEqual(milker.inside_stand_px, (64, 168))
        self.assertEqual(milker.inside_face, "up")
        self.assertIsNone(milker.inside_recording)
        self.assertEqual(brush.inside_stand_px, (80, 168))
        self.assertEqual(brush.inside_face, "up")
        self.assertIsNone(brush.inside_recording)

    def test_ensure_carry_tool_uses_dynamic_shed_shelf_task_for_brush(self) -> None:
        world = make_world(0x26)
        task = EnsureCarryToolTask(tool_id=int(Tool.BRUSH))

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "inside")
        self.assertIsInstance(task._task, ShedShelfToolTask)
        self.assertEqual(task._task.stand_px, (80, 168))
        self.assertEqual(task._task.radius, 2)

    def test_ensure_animal_tools_is_noop_when_milker_and_brush_carried(self) -> None:
        world = make_world(0x00)
        world.ram[0x0921] = int(Tool.BRUSH)
        world.ram[0x0923] = int(Tool.MILKER)
        task = EnsureAnimalToolsTask()

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "animal tools ready")

    def test_ensure_animal_tools_swaps_selected_target_before_getting_missing_tool(self) -> None:
        world = make_world(0x00)
        world.ram[0x0921] = int(Tool.BRUSH)
        world.ram[0x0923] = int(Tool.WATERING_CAN)
        task = EnsureAnimalToolsTask()

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "swap")
        self.assertIsInstance(task._task, SwapCarrySlotsTask)

    def test_ensure_animal_tools_gets_brush_first_when_neither_target_carried(self) -> None:
        world = make_world(0x00)
        world.ram[0x0921] = int(Tool.WATERING_CAN)
        world.ram[0x0923] = 0x07
        task = EnsureAnimalToolsTask()

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "ensure_0x0F")
        self.assertIsInstance(task._task, EnsureCarryToolTask)
        self.assertEqual(task._task.tool_id, int(Tool.BRUSH))

class BuildDayPhasesCoopTests(DayPlanPhaseHelpers):
    """Tests for the dynamic day plan builder."""

    def test_chickens_come_first(self) -> None:
        phases = build_day_phases(None, has_chickens=True)
        names = self._phase_names(phases)
        chicken_names = [p.phase for p in CHICKEN_PHASES]
        chicken_start = names.index("NAV_TO_COOP")
        self.assertEqual(names[chicken_start:chicken_start + len(chicken_names)], chicken_names)

    def test_oversupplied_chickens_schedule_sale_and_limited_chores(self) -> None:
        phases = build_day_phases(None, hour=6, has_chickens=True, adult_chickens=6)
        names = self._phase_names(phases)

        self.assertLess(names.index("SELL_CHICKEN"), names.index("NAV_TO_COOP"))
        coop_phase = phases[names.index("COOP_CHORES")]
        self.assertEqual(coop_phase.params["egg_mode"], "ship")
        self.assertEqual(coop_phase.params["max_feed_adults"], 2)

    def test_chicken_sale_skips_after_cutoff(self) -> None:
        phases = build_day_phases(None, hour=14, has_chickens=True, adult_chickens=6)
        names = self._phase_names(phases)

        self.assertNotIn("SELL_CHICKEN", names)
        self.assertIn("COOP_CHORES", names)

    def test_chicken_sale_skips_sunday(self) -> None:
        phases = build_day_phases(None, weekday=0, hour=6, has_chickens=True, adult_chickens=6)
        names = self._phase_names(phases)

        self.assertNotIn("SELL_CHICKEN", names)
        self.assertIn("COOP_CHORES", names)

    def test_cows_run_before_chickens_when_both_need_chores(self) -> None:
        phases = build_day_phases(None, has_chickens=True, has_cows=True)
        names = self._phase_names(phases)
        chicken_names = [p.phase for p in CHICKEN_PHASES]
        cow_names = [p.phase for p in COW_PHASES]
        cow_start = names.index("ENSURE_ANIMAL_TOOLS")
        chicken_start = names.index("NAV_TO_COOP")

        self.assertEqual(names[cow_start:cow_start + len(cow_names)], cow_names)
        self.assertEqual(names[chicken_start:chicken_start + len(chicken_names)], chicken_names)
        self.assertEqual(phases[chicken_start].params["route"], "barn_to_coop")
        self.assertLess(cow_start, chicken_start)
        self.assertEqual(phases[chicken_start].params["initial_settle_frames"], 0)

    def test_affordable_first_cow_purchase_runs_before_chickens(self) -> None:
        phases = build_day_phases(
            None,
            weekday=2,
            hour=6,
            has_chickens=True,
            has_harvest=True,
            has_seeds=True,
            should_buy_cow=True,
        )
        names = self._phase_names(phases)

        self.assertLess(names.index("BUY_COW_VENDOR"), names.index("NAV_TO_COOP"))
        self.assertLess(names.index("EXIT_BARN"), names.index("NAV_TO_COOP"))
        self.assertEqual(phases[names.index("NAV_TO_COOP")].params["route"], "barn_to_coop")
        self.assertEqual(names[0], "EXIT_TO_FARM")

    def test_state_has_chickens_detects_chicken_count(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        ram[ADDR_CHICKEN_COUNT] = 2
        fake_state = SimpleNamespace(ram=ram)

        with patch("harvest.planner.day_plan_status.resolve_state_path", return_value="/tmp/fake.state"), patch(
            "harvest.planner.day_plan_status.parse_save_state",
            return_value=fake_state,
        ):
            self.assertTrue(state_has_chickens("fake"))

    def test_state_has_chickens_false_when_zero(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        fake_state = SimpleNamespace(ram=ram)

        with patch("harvest.planner.day_plan_status.resolve_state_path", return_value="/tmp/fake.state"), patch(
            "harvest.planner.day_plan_status.parse_save_state",
            return_value=fake_state,
        ):
            self.assertFalse(state_has_chickens("fake"))

    def test_state_has_cows_detects_cow_count(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        ram[ADDR_COW_COUNT] = 1
        fake_state = SimpleNamespace(ram=ram)

        with patch("harvest.planner.day_plan_status.resolve_state_path", return_value="/tmp/fake.state"), patch(
            "harvest.planner.day_plan_status.parse_save_state",
            return_value=fake_state,
        ):
            self.assertTrue(state_has_cows("fake"))

    def test_chicken_phases_structure(self) -> None:
        names = [p.phase for p in CHICKEN_PHASES]
        kinds = [p.kind for p in CHICKEN_PHASES]
        self.assertEqual(names, ["NAV_TO_COOP", "ENTER_COOP", "COOP_CHORES", "EXIT_COOP"])
        self.assertEqual(kinds, ["multi_nav", "directional_transition", "coop_chores", "directional_transition"])
        self.assertEqual(CHICKEN_PHASES[-1].params["stand_tile"], (8, 12))
        self.assertEqual(CHICKEN_PHASES[-1].params["settle_frames"], 5)
        self.assertEqual(len(ROUTES["barn_to_coop"]), 1)
        self.assertEqual(ROUTES["barn_to_coop"][0].target_px, (454, 360))
        self.assertIsNone(ROUTES["barn_to_coop"][0].run_direction)

    def test_cow_phases_structure(self) -> None:
        names = [p.phase for p in COW_PHASES]
        kinds = [p.kind for p in COW_PHASES]
        self.assertEqual(names, ["ENSURE_ANIMAL_TOOLS", "EXIT_TO_FARM", "NAV_TO_BARN", "ENTER_BARN", "COW_CHORES", "EXIT_BARN"])
        self.assertEqual(kinds, ["ensure_animal_tools", "farm_building_exit", "multi_nav", "directional_transition", "cow_chores", "directional_transition"])
        self.assertEqual(COW_PHASES[2].params["route"], "farm_to_barn")
        self.assertTrue(COW_PHASES[4].params.get("milk", True))
        self.assertTrue(COW_PHASES[4].params.get("feed", True))
        self.assertTrue(COW_PHASES[4].params.get("talk", True))
        self.assertTrue(COW_PHASES[4].params.get("brush", True))
        self.assertEqual(COW_PHASES[-1].params["stand_tile"], (8, 22))

    def test_build_day_phases_from_ram_sells_when_adult_chickens_exceed_target(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=6)
        world.ram[ADDR_WEEKDAY + 0x4000] = 2
        world.ram[ADDR_CHICKEN_COUNT + 0x4000] = 6
        world.ram[ADDR_HAY_COUNT + 0x4000] = 50
        set_live_chicken_slots(world.ram, adults=6)

        phases = build_day_phases_from_ram(world.ram)
        names = self._phase_names(phases)

        self.assertIn("SELL_CHICKEN", names)
        self.assertLess(names.index("SELL_CHICKEN"), names.index("NAV_TO_COOP"))
        self.assertEqual(phases[names.index("COOP_CHORES")].params["max_feed_adults"], 2)

    def test_next_morning_ram_plan_runs_cow_then_coop_chores(self) -> None:
        world = make_date_world(0x15, season=0, day=14, hour=6)
        world.ram[ADDR_WEEKDAY + 0x4000] = 0
        world.ram[ADDR_CHICKEN_COUNT + 0x4000] = 1
        world.ram[ADDR_EGG_AVAILABLE + 0x4000] = 1
        world.ram[ADDR_COW_COUNT + 0x4000] = 1
        world.ram[ADDR_FED_COWS_N + 0x4000] = 0
        world.ram[ADDR_HAY_COUNT + 0x4000] = 20
        set_live_cow_slot(world.ram, 0)

        phases = build_day_phases_from_ram(world.ram)
        names = self._phase_names(phases)

        self.assertLess(names.index("NAV_TO_BARN"), names.index("NAV_TO_COOP"))
        self.assertEqual(phases[names.index("NAV_TO_COOP")].params["route"], "barn_to_coop")
        self.assertIn("COOP_CHORES", names)
        self.assertIn("COW_CHORES", names)

    def test_build_day_phases_from_ram_includes_cow_chores_when_unfed(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=6)
        world.ram[ADDR_COW_COUNT + 0x4000] = 1
        world.ram[ADDR_FED_COWS_N + 0x4000] = 0
        world.ram[ADDR_HAY_COUNT + 0x4000] = 20

        phases = build_day_phases_from_ram(world.ram)
        names = self._phase_names(phases)

        self.assertIn("NAV_TO_BARN", names)
        self.assertIn("COW_CHORES", names)

    def test_build_day_phases_from_ram_includes_cow_chores_when_brush_missing(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=6)
        world.ram[ADDR_COW_COUNT + 0x4000] = 1
        world.ram[ADDR_FED_COWS_N + 0x4000] = 1
        set_live_cow_slot(world.ram, 0, flags=COW_DAILY_TALKED_FLAG, happiness=1)

        phases = build_day_phases_from_ram(world.ram)
        names = self._phase_names(phases)

        self.assertIn("NAV_TO_BARN", names)
        self.assertIn("COW_CHORES", names)

    def test_build_day_phases_from_ram_starts_cows_in_barn_when_tools_ready(self) -> None:
        world = make_date_world(0x27, season=0, day=13, hour=6)
        world.ram[ADDR_COW_COUNT + 0x4000] = 1
        world.ram[ADDR_FED_COWS_N + 0x4000] = 0
        world.ram[ADDR_HAY_COUNT + 0x4000] = 20
        world.ram[field_spec("tool_selected").address + 0x4000] = int(Tool.BRUSH)
        world.ram[field_spec("tool_backpack").address + 0x4000] = int(Tool.MILKER)
        set_live_cow_slot(world.ram, 0)

        phases = build_day_phases_from_ram(world.ram)

        self.assertEqual(
            self._phase_names(phases)[:3],
            ["COW_CHORES", "EXIT_BARN", "DYNAMIC_OUTDOOR_PLAN"],
        )
        self.assertNotIn("NAV_TO_BARN", self._phase_names(phases))

    def test_build_day_phases_from_ram_starts_chickens_in_coop(self) -> None:
        world = make_date_world(COOP_TILEMAP, season=0, day=13, hour=6)
        world.ram[ADDR_CHICKEN_COUNT + 0x4000] = 4
        world.ram[ADDR_FED_CHICKENS_N + 0x4000] = 0
        world.ram[ADDR_EGG_AVAILABLE + 0x4000] = 1
        world.ram[ADDR_COW_COUNT + 0x4000] = 1
        world.ram[ADDR_FED_COWS_N + 0x4000] = 0
        world.ram[ADDR_HAY_COUNT + 0x4000] = 20

        phases = build_day_phases_from_ram(world.ram)
        names = self._phase_names(phases)

        self.assertEqual(names[:2], ["COOP_CHORES", "EXIT_COOP"])
        self.assertNotIn("NAV_TO_COOP", names)
        self.assertLess(names.index("EXIT_COOP"), names.index("NAV_TO_BARN"))
        self.assertEqual(names[-1], "DYNAMIC_OUTDOOR_PLAN")

    def test_build_day_phases_from_ram_uses_coop_exit_when_coop_chores_done(self) -> None:
        world = make_date_world(COOP_TILEMAP, season=0, day=13, hour=6)
        world.ram[ADDR_CHICKEN_COUNT + 0x4000] = 1
        world.ram[ADDR_FED_CHICKENS_N + 0x4000] = 1
        world.ram[ADDR_EGG_AVAILABLE + 0x4000] = 0

        phases = build_day_phases_from_ram(world.ram)

        self.assertEqual(self._phase_names(phases), ["EXIT_COOP", "DYNAMIC_OUTDOOR_PLAN"])

    def test_build_day_phases_from_ram_buys_first_cow_before_coop(self) -> None:
        world = make_date_world(0x15, season=0, day=16, hour=6)
        world.ram[ADDR_WEEKDAY + 0x4000] = 2
        world.ram[ADDR_CHICKEN_COUNT + 0x4000] = 1
        world.ram[ADDR_EGG_AVAILABLE + 0x4000] = 1
        set_money(world.ram, COW_PURCHASE_COST)

        phases = build_day_phases_from_ram(world.ram)
        names = self._phase_names(phases)

        self.assertLess(names.index("BUY_COW_VENDOR"), names.index("NAV_TO_COOP"))
        self.assertLess(names.index("EXIT_BARN"), names.index("NAV_TO_COOP"))
        self.assertEqual(names[-1], "DYNAMIC_OUTDOOR_PLAN")

    def test_build_day_phases_from_ram_skips_cow_purchase_when_already_owned(self) -> None:
        world = make_date_world(0x15, season=0, day=16, hour=6)
        world.ram[ADDR_WEEKDAY + 0x4000] = 2
        world.ram[ADDR_COW_COUNT + 0x4000] = 1
        world.ram[ADDR_FED_COWS_N + 0x4000] = 1
        set_money(world.ram, COW_PURCHASE_COST)

        phases = build_day_phases_from_ram(world.ram)
        names = self._phase_names(phases)

        self.assertNotIn("BUY_COW_VENDOR", names)

    def test_late_day_ram_plan_skips_done_coop_waters_then_sleeps(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=18)
        world.ram[ADDR_WEEKDAY + 0x4000] = 6
        world.ram[ADDR_CHICKEN_COUNT + 0x4000] = 1
        world.ram[ADDR_FED_CHICKENS_N + 0x4000] = 1
        world.ram[ADDR_COW_COUNT + 0x4000] = 1
        world.ram[ADDR_FED_COWS_N + 0x4000] = 1
        world.ram[ADDR_EGG_AVAILABLE + 0x4000] = 0
        world.ram[0x092A + 0x4000] = 5
        world.ram[ADDR_MAP + 34 * 64 + 7] = 0x60
        world.ram[ADDR_MAP + 35 * 64 + 7] = 0x58

        phases = build_day_phases_from_ram(world.ram)
        names = self._phase_names(phases)

        self.assertNotIn("NAV_TO_COOP", names)
        self.assertNotIn("NAV_TO_BARN", names)
        self.assertNotIn("HARVEST_ROUTE", names)
        self.assertIn("ENSURE_WATERING_CAN", names)
        self.assertNotIn("ENSURE_CROP_SEEDS", names)
        self.assertIn("CROP_WATER", names)
        self.assertEqual(names[-2:], ["RETURN_HOME", "GO_TO_SLEEP"])


if __name__ == "__main__":
    unittest.main()

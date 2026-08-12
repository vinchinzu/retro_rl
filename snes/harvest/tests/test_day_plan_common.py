"""Core day-plan orchestration, nav, transitions, shed tools.

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
    WEED,
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


class DayPlanSequenceCommonTests(unittest.TestCase):
    def test_weekday_sequence_keeps_town_first_route(self) -> None:
        phases = PHASE_SEQUENCES["day1"]

        self.assertEqual(
            [phase.phase for phase in phases],
            [
                "EXIT_TO_FARM",
                "CLEAR_FIELD",
                "NAV_FARM_EXIT",
                "BUY_SEEDS",
                "ENSURE_CROP_SEEDS",
                "NAV_CROP",
                "CROP_ESTABLISH",
                "ENSURE_WATERING_CAN",
                "NAV_CROP",
                "CROP_WATER",
                "TOWN_EXPLORE",
                "READY_TO_GO_HOME",
                "RETURN_HOME",
                "GO_TO_SLEEP",
            ],
        )
        self.assertIn("BUY_SEEDS", [phase.phase for phase in phases])
        self.assertIn("TOWN_EXPLORE", [phase.phase for phase in phases])
        self.assertIn("RETURN_HOME", [phase.phase for phase in phases])
        self.assertNotIn("GET_BERRIES_AND_SHIP", [phase.phase for phase in phases])

    def test_eve_loop_sequence_replays_recording_from_bar_exterior(self) -> None:
        phases = PHASE_SEQUENCES["eve_loop"]

        self.assertIs(phases, EVE_TALK_LOOP_PHASES)
        self.assertEqual(
            [phase.phase for phase in phases],
            ["EVE_TALK_LOOP"],
        )
        self.assertEqual(phases[0].kind, "eve_talk_loop")
        self.assertEqual(phases[0].params["target_hearts"], 10)

    def test_deadline_check_reads_live_time_from_offset_ram(self) -> None:
        task = DeadlineCheckTask(latest_hour=17, latest_minute=0)
        world = make_time_world(0x00, day=9, hour=11, minute=10, live_offset=True)

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "before cutoff at 11:10")

    def test_deadline_check_fails_at_shipping_cutoff(self) -> None:
        task = DeadlineCheckTask(latest_hour=17, latest_minute=0)
        world = make_time_world(0x00, day=9, hour=17, minute=0)

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertEqual(result.reason, "cutoff reached at 17:00")

    def test_recorded_transition_requires_target_tilemap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/leave_farm_west.json"
            with open(path, "w") as f:
                json.dump(
                    {
                        "name": "leave_farm_west",
                        "frames": [[0] * 12, [0] * 12],
                        "start_state": "pinned_fixture",
                    },
                    f,
                )

            task = RecordedTransitionTask(
                task_name="leave_farm_west",
                tasks_dir=tmpdir,
                origin_tilemap=0x00,
                target_tilemap=0x0C,
                min_frames_before_success=1,
            )
            task.reset(make_world(0x00))

            first = task.step(make_world(0x00))
            self.assertEqual(first.status, TaskStatus.RUNNING)

            second = task.step(make_world(0x0C))
            self.assertEqual(second.status, TaskStatus.SUCCESS)

    def test_recorded_transition_accepts_seasonal_farm_target(self) -> None:
        world = make_world(0x01)
        task = RecordedTransitionTask(target_tilemap=0x00, min_frames_before_success=0)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "tilemap=0x01")

    def test_cross_map_exit_waits_on_seasonal_farm_origin(self) -> None:
        world = make_world(0x01)
        task = CrossMapRecordedTask(origin_tilemap=0x00, exit_direction="left")

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "exit")
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[6]), 1)

    def test_eve_talk_loop_replays_talk_chunk_until_heart_gain_then_exits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/talk_eve_loop.json"
            with open(path, "w") as f:
                json.dump(
                    {
                        "name": "talk_eve_loop",
                        "frames": [[0] * 12, [1] + [0] * 11],
                    },
                    f,
                )

            world = make_world(0x1E)
            set_player_pos(world.ram, 69, 450)
            set_live_u16(world.ram, "eve_hearts", 40)
            task = EveTalkLoopTask(tasks_dir=tmpdir, target_hearts=2, max_loops=2)
            task.reset(world)
            task._phase = "talk"
            task._attempt_start_points = 40

            first = task.step(world)
            self.assertEqual(first.status, TaskStatus.RUNNING)
            self.assertEqual(first.reason, "talk")

            set_live_u16(world.ram, "eve_hearts", 50)
            second = task.step(world)
            self.assertEqual(second.status, TaskStatus.RUNNING)
            self.assertEqual(second.reason, "heart_gain")
            self.assertEqual(task._phase, "exit_bar")

    def test_eve_talk_loop_ignores_small_yes_gain_and_resets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/talk_eve_loop.json"
            with open(path, "w") as f:
                json.dump({"name": "talk_eve_loop", "frames": [[0] * 12]}, f)

            world = make_world(0x1E)
            set_player_pos(world.ram, 69, 450)
            set_live_u16(world.ram, "eve_hearts", 40)
            task = EveTalkLoopTask(tasks_dir=tmpdir, target_hearts=2)
            task.reset(world)
            task._phase = "talk"
            task._attempt_start_points = 40

            set_live_u16(world.ram, "eve_hearts", 44)
            result = task.step(world)
            self.assertEqual(result.status, TaskStatus.RUNNING)
            self.assertEqual(result.reason, "small_heart_gain")
            self.assertEqual(task._loop_count, 0)
            self.assertEqual(task._phase, "exit_bar")

    def test_eve_talk_loop_overrides_locked_choice_to_no(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/talk_eve_loop.json"
            with open(path, "w") as f:
                json.dump({"name": "talk_eve_loop", "frames": [[0] * 12]}, f)

            world = make_world(0x1E)
            world.ram[ADDR_INPUT_LOCK] = 2
            world.ram[field_spec("player_action").address] = 9
            set_player_pos(world.ram, 69, 450)
            set_live_u16(world.ram, "eve_hearts", 40)
            task = EveTalkLoopTask(tasks_dir=tmpdir, target_hearts=2)
            task.reset(world)
            task._phase = "talk"
            task._attempt_start_points = 40

            result = task.step(world)
            self.assertEqual(result.status, TaskStatus.RUNNING)
            self.assertEqual(result.reason, "choose_no")
            self.assertIsNotNone(result.action)
            self.assertEqual(int(result.action.action[5]), 1)

    def test_eve_talk_loop_resets_entry_when_talk_stops_working(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/talk_eve_loop.json"
            with open(path, "w") as f:
                json.dump({"name": "talk_eve_loop", "frames": [[0] * 12]}, f)

            world = make_world(0x1E)
            set_player_pos(world.ram, 69, 450)
            set_live_u16(world.ram, "eve_hearts", 40)
            task = EveTalkLoopTask(tasks_dir=tmpdir, target_hearts=2)
            task.reset(world)
            task._phase = "talk"
            task._attempt_start_points = 40

            task.step(world)
            result = task.step(world)
            self.assertEqual(result.status, TaskStatus.RUNNING)
            self.assertEqual(result.reason, "reset_after_missed_talk")
            self.assertEqual(task._phase, "exit_bar")

    def test_eve_talk_loop_is_noop_when_target_already_met(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/talk_eve_loop.json"
            with open(path, "w") as f:
                json.dump({"name": "talk_eve_loop", "frames": [[0] * 12]}, f)

            world = make_world(0x1E)
            set_live_u16(world.ram, "eve_hearts", 999)
            task = EveTalkLoopTask(tasks_dir=tmpdir, target_hearts=10)
            task.reset(world)

            result = task.step(world)
            self.assertEqual(result.status, TaskStatus.SUCCESS)
            self.assertIn("eve hearts 999/999", result.reason)

    def test_eve_talk_loop_stops_at_daily_question_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/talk_eve_loop.json"
            with open(path, "w") as f:
                json.dump({"name": "talk_eve_loop", "frames": [[0] * 12]}, f)

            world = make_world(0x1E)
            set_live_u16(world.ram, "eve_hearts", 120)
            task = EveTalkLoopTask(tasks_dir=tmpdir, target_hearts=10)
            task.reset(world)

            result = task.step(world)
            self.assertEqual(result.status, TaskStatus.SUCCESS)
            self.assertIn("eve daily question cap 120/999", result.reason)

    def test_eve_talk_loop_exits_bar_between_replays(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/talk_eve_loop.json"
            with open(path, "w") as f:
                json.dump({"name": "talk_eve_loop", "frames": [[0] * 12]}, f)

            world = make_world(0x1E)
            set_player_pos(world.ram, 69, 450)
            set_live_u16(world.ram, "eve_hearts", 40)
            task = EveTalkLoopTask(tasks_dir=tmpdir, target_hearts=1)
            task.reset(world)
            task._phase = "exit_bar"

            result = task.step(world)
            self.assertEqual(result.status, TaskStatus.RUNNING)
            self.assertEqual(result.reason, "exit_bar")
            self.assertIsNotNone(result.action)
            self.assertEqual(int(result.action.action[7]), 1)
            self.assertEqual(int(result.action.action[0]), 1)

    def test_eve_talk_loop_rejects_wrong_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/talk_eve_loop.json"
            with open(path, "w") as f:
                json.dump({"name": "talk_eve_loop", "frames": [[0] * 12]}, f)

            world = make_world(0x00)
            set_player_pos(world.ram, 152, 872)
            set_live_u16(world.ram, "eve_hearts", 40)
            task = EveTalkLoopTask(tasks_dir=tmpdir, target_hearts=1, align_timeout=2)
            task.reset(world)

            task.step(world)
            task.step(world)
            result = task.step(world)
            self.assertEqual(result.status, TaskStatus.FAILURE)
            self.assertIn("expected Eve loop origin", result.reason)

    def test_eve_talk_loop_aligns_to_recording_origin_on_same_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/talk_eve_loop.json"
            with open(path, "w") as f:
                json.dump({"name": "talk_eve_loop", "frames": [[0] * 12]}, f)

            world = make_world(0x04)
            set_player_pos(world.ram, 152, 953)
            set_live_u16(world.ram, "eve_hearts", 40)
            task = EveTalkLoopTask(tasks_dir=tmpdir, target_hearts=1)
            task.reset(world)

            result = task.step(world)
            self.assertEqual(result.status, TaskStatus.RUNNING)
            self.assertIsNotNone(result.action)
            self.assertEqual(int(result.action.action[4]), 1)

    def test_day_plan_advances_to_next_phase_without_idle_frame(self) -> None:
        class ImmediateSuccessTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.SUCCESS, reason="done")

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
                        PhaseSpec("FIRST", "custom"),
                        PhaseSpec("SECOND", "custom"),
                    ]
                )
                self._tasks = [ImmediateSuccessTask(), ActionTask()]

            def _make_task(self, spec, world):
                return self._tasks.pop(0)

        plan = TestPlan()
        world = make_world(0x00)
        plan.reset(world)

        result = plan.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[7]), 1)
        self.assertEqual(plan.phase_text, "SECOND")

    def test_day_plan_skips_optional_route_on_deadline_failure(self) -> None:
        class FailTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.FAILURE, reason="missed cutoff")

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
                        PhaseSpec("BERRY_RUN_WINDOW", "deadline"),
                        PhaseSpec("EXIT_FARM_WEST", "directional_transition"),
                        PhaseSpec("GET_BERRIES_AND_SHIP", "recorded"),
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
            ["BERRY_RUN_WINDOW", "EXIT_FARM_WEST", "GET_BERRIES_AND_SHIP"],
        )
        self.assertTrue(all(item.reason == "missed cutoff" for item in plan.deferred_plans))

    def test_day_plan_skips_optional_blocked_phase(self) -> None:
        class BlockedTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.BLOCKED, reason="precondition missing")

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
                        PhaseSpec("OPTIONAL_SETUP", "test", failure_policy="optional"),
                        PhaseSpec("ENSURE_WATERING_CAN", "ensure_tool"),
                    ]
                )
                self._tasks = [BlockedTask(), ActionTask()]

            def _make_task(self, spec, world):
                return self._tasks.pop(0)

        plan = TestPlan()
        world = make_world(0x00)
        plan.reset(world)

        result = plan.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(plan.phase_text, "ENSURE_WATERING_CAN")
        self.assertEqual([item.phase for item in plan.deferred_plans], ["OPTIONAL_SETUP"])
        self.assertEqual(plan.deferred_plans[0].reason, "precondition missing")

    def test_day_plan_recovers_required_blocked_phase_then_retries(self) -> None:
        class BlockedTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.BLOCKED, reason="precondition missing")

        class SuccessTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.SUCCESS, reason="done")

        class ActionTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(right=True)))

        class RecoverySuccessTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.SUCCESS, reason="recovered")

        class TestPlan(DayPlanTask):
            def __init__(self) -> None:
                super().__init__(
                    phase_sequence=[
                        PhaseSpec("EXIT_TO_FARM", "farm_building_exit"),
                        PhaseSpec("ENSURE_WATERING_CAN", "ensure_tool"),
                    ]
                )
                self.phase_attempts = 0
                self.recovery_attempts = 0

            def _make_task(self, spec, world):
                if spec.phase == "EXIT_TO_FARM":
                    self.phase_attempts += 1
                    if self.phase_attempts == 1:
                        return BlockedTask()
                    return SuccessTask()
                return ActionTask()

            def _make_recovery_task(self, spec, status, reason, world):
                self.recovery_attempts += 1
                return RecoverySuccessTask()

        plan = TestPlan()
        world = make_world(0x00)
        plan.reset(world)

        result = plan.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(plan.phase_text, "ENSURE_WATERING_CAN")
        self.assertEqual(plan.phase_attempts, 2)
        self.assertEqual(plan.recovery_attempts, 1)

    def test_day_plan_aborts_required_blocked_phase_after_recovery_blocks(self) -> None:
        class BlockedTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.BLOCKED, reason="precondition missing")

        class RecoveryBlockedTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.BLOCKED, reason="recovery stuck")

        class TestPlan(DayPlanTask):
            def __init__(self) -> None:
                super().__init__(
                    phase_sequence=[
                        PhaseSpec("EXIT_TO_FARM", "farm_building_exit"),
                        PhaseSpec("ENSURE_WATERING_CAN", "ensure_tool"),
                    ]
                )

            def _make_task(self, spec, world):
                return BlockedTask()

            def _make_recovery_task(self, spec, status, reason, world):
                return RecoveryBlockedTask()

        plan = TestPlan()
        world = make_world(0x00)
        plan.reset(world)

        result = plan.step(world)

        self.assertEqual(result.status, TaskStatus.BLOCKED)
        self.assertIn("required phase EXIT_TO_FARM failed after recovery: precondition missing", result.reason)
        self.assertIn("recovery blocked: recovery stuck", result.reason)
        self.assertEqual(plan.phase_text, "EXIT_TO_FARM")

    def test_day_plan_aborts_required_missing_task(self) -> None:
        plan = DayPlanTask(phase_sequence=[PhaseSpec("ENTER_BARN", "unknown_kind")])
        world = make_world(0x00)
        plan.reset(world)

        result = plan.step(world)

        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("required phase ENTER_BARN failed: no task", result.reason)
        self.assertEqual(plan.phase_text, "ENTER_BARN")

    def test_navigator_treats_transition_tile_0x76_as_stale_and_repaths(self) -> None:
        ram = make_navigation_ram()
        navigator = Navigator(Pathfinder(TileScanner()))
        navigator.update(ram)
        navigator.path = [(14, 8)]

        action = navigator.follow_path(ram)

        self.assertIsNone(action)
        self.assertEqual(navigator.path, [])

    def test_navigator_clears_path_for_known_nonwalkable_tile(self) -> None:
        ram = make_navigation_ram(blocked_tile=(14, 8), blocked_id=0xA6)
        navigator = Navigator(Pathfinder(TileScanner()))
        navigator.update(ram)
        navigator.path = [(14, 8)]

        action = navigator.follow_path(ram)

        self.assertIsNone(action)
        self.assertEqual(navigator.path, [])
        self.assertIn((14, 8), navigator.pathfinder.temp_blocked)

    def test_directional_transition_uses_stand_tile_before_holding_direction(self) -> None:
        world = make_transition_world(0x28, current_tile=(8, 11))
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
        self.assertEqual(int(result.action.action[5]), 1)

    def test_directional_transition_routes_to_nonwalkable_exit_tile(self) -> None:
        ram = make_navigation_ram(current_tile=(2, 1), blocked_tile=(1, 1), blocked_id=0xA6)
        ram[ADDR_INPUT_LOCK] = 1
        ram[ADDR_MAP + 0 * 64 + 0] = 0xC0
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = DirectionalTransitionTask(
            direction="left",
            origin_tilemap=0x00,
            target_tilemap=0x0C,
            stand_tile=(0, 0),
        )

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[4]), 1)
        self.assertEqual(int(result.action.action[6]), 0)

    def test_directional_transition_can_settle_after_tilemap_change(self) -> None:
        origin = make_transition_world(0x28, current_tile=(8, 12))
        target = make_transition_world(0x00, current_tile=(8, 13))
        set_player_pos(target.ram, 456, 360)
        task = DirectionalTransitionTask(
            direction="down",
            origin_tilemap=0x28,
            target_tilemap=0x00,
            min_frames_before_success=1,
            settle_frames=3,
        )

        task.reset(origin)
        self.assertEqual(task.step(origin).status, TaskStatus.RUNNING)
        self.assertEqual(task.step(target).status, TaskStatus.RUNNING)
        self.assertEqual(task.step(target).status, TaskStatus.RUNNING)
        result = task.step(target)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "tilemap=0x00")

    def test_directional_transition_waits_for_target_stand_tile(self) -> None:
        origin = make_transition_world(0x00)
        transitional = make_transition_world(0x27)
        landed = make_transition_world(0x27)
        set_player_pos(origin.ram, 20 * 16 + 8, 22 * 16 + 8)
        set_player_pos(transitional.ram, 20 * 16 + 8, 21 * 16 + 8)
        set_player_pos(landed.ram, 8 * 16 + 8, 22 * 16 + 8)
        task = DirectionalTransitionTask(
            direction="up",
            origin_tilemap=0x00,
            target_tilemap=0x27,
            min_frames_before_success=1,
            settle_frames=2,
            target_stand_tile=(8, 22),
            target_stand_tolerance=1,
        )

        task.reset(origin)
        self.assertEqual(task.step(origin).status, TaskStatus.RUNNING)
        waiting = task.step(transitional)
        self.assertEqual(waiting.status, TaskStatus.RUNNING)
        self.assertEqual(waiting.reason, "transition target settle")
        self.assertEqual(int(waiting.action.action[4]), 1)
        self.assertEqual(task.step(landed).status, TaskStatus.RUNNING)
        result = task.step(landed)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "tilemap=0x27")

    def test_directional_transition_factory_passes_target_stand_tile(self) -> None:
        spec = PhaseSpec(
            "ENTER_BARN",
            "directional_transition",
            {
                "direction": "up",
                "origin_tilemap": 0x00,
                "target_tilemap": 0x27,
                "target_stand_tile": (8, 22),
                "target_stand_tolerance": 1,
            },
        )

        task = DayTaskFactory().make_task(spec, make_transition_world(0x00))

        self.assertIsInstance(task, DirectionalTransitionTask)
        self.assertEqual(task.target_stand_tile, (8, 22))
        self.assertEqual(task.target_stand_tolerance, 1)

    def test_directional_transition_factory_passes_door_entry_params(self) -> None:
        spec = PhaseSpec(
            "ENTER_COOP",
            "directional_transition",
            {
                "direction": "up",
                "origin_tilemap": 0x00,
                "target_tilemap": 0x28,
                "stand_tile": (28, 22),
                "door_align_px": 456,
                "overshoot_limit_px": 330,
                "require_empty_hands": True,
            },
        )

        task = DayTaskFactory().make_task(spec, make_transition_world(0x00))

        self.assertIsInstance(task, DirectionalTransitionTask)
        self.assertEqual(task.door_align_px, 456)
        self.assertEqual(task.overshoot_limit_px, 330)
        self.assertTrue(task.require_empty_hands)

    def test_directional_transition_clears_hands_before_door(self) -> None:
        from harvest.core.animal_status import ADDR_HELD_ITEM
        from harvest.core.ram_catalog import live_wram_base

        world = make_transition_world(0x00, current_tile=(8, 26))
        base = live_wram_base(world.ram)
        world.ram[ADDR_HELD_ITEM + base] = 0x0D
        world.ram[field_spec("player_state").address + base] = 0x03
        task = DirectionalTransitionTask(
            direction="up",
            origin_tilemap=0x00,
            target_tilemap=0x15,
            stand_tile=(8, 26),
            require_empty_hands=True,
            door_align_px=136,
        )

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(result.reason, "clear hands for door")
        self.assertEqual(task._hands_attempts, 1)
        self.assertTrue(task._action_queue)

    def test_directional_transition_overshoot_restands(self) -> None:
        world = make_transition_world(0x00, current_tile=(8, 20))
        set_player_pos(world.ram, 136, 310)
        task = DirectionalTransitionTask(
            direction="up",
            origin_tilemap=0x00,
            target_tilemap=0x15,
            stand_tile=(8, 26),
            overshoot_limit_px=328,
            door_align_px=136,
        )

        task.reset(world)
        task._stand_reached = True
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertFalse(task._stand_reached)
        # Re-stand now strafes off the embedded door column before moving down.
        self.assertEqual(int(result.action.action[6]), 1)

    def test_directional_transition_aligns_door_column_before_push(self) -> None:
        world = make_transition_world(0x00, current_tile=(8, 26))
        set_player_pos(world.ram, 120, 424)
        task = DirectionalTransitionTask(
            direction="up",
            origin_tilemap=0x00,
            target_tilemap=0x15,
            stand_tile=(8, 26),
            door_align_px=136,
        )

        task.reset(world)
        task._stand_reached = True
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(int(result.action.action[7]), 1)
        self.assertEqual(int(result.action.action[4]), 0)

    def test_nav_task_soft_arrives_when_stuck_near_target(self) -> None:
        world = make_world(0x00)
        set_player_pos(world.ram, 121, 519)
        task = NavTask(
            target_px=Point(136, 520),
            radius=12,
            soft_radius=48,
            soft_stasis=5,
            timeout=200,
        )
        task.reset(world)
        task._navigator.stasis = 10

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("soft arrived", result.reason or "")

    def test_directional_transition_accepts_seasonal_farm_origin(self) -> None:
        world = make_world(0x01)
        set_player_pos(world.ram, 20 * 16 + 8, 22 * 16 + 8)
        task = DirectionalTransitionTask(
            direction="up",
            origin_tilemap=0x00,
            target_tilemap=0x27,
            min_frames_before_success=1,
        )

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertNotIn("expected origin", result.reason or "")

    def test_multi_nav_treats_seasonal_farm_as_farm_waypoint(self) -> None:
        world = make_world(0x01)
        set_player_pos(world.ram, 137, 344)
        task = MultiMapNavTask(
            waypoints=[Waypoint(tilemap=0x00, target_px=(137, 344), radius=12)],
            initial_settle_frames=0,
        )

        task.reset(world)
        first = task.step(world)
        result = task.step(world)

        self.assertEqual(first.status, TaskStatus.RUNNING)
        self.assertEqual(result.status, TaskStatus.SUCCESS)

    def test_directional_transition_does_not_recenter_after_stand_reached(self) -> None:
        world = make_transition_world(0x00, current_tile=(8, 12))
        task = DirectionalTransitionTask(
            direction="up",
            origin_tilemap=0x00,
            target_tilemap=0x28,
            stand_tile=(8, 12),
        )

        task.reset(world)
        first = task.step(world)
        self.assertEqual(int(first.action.action[4]), 1)

        set_player_pos(world.ram, 8 * 16 + 8, 12 * 16 + 6)
        second = task.step(world)

        self.assertEqual(int(second.action.action[4]), 1)
        self.assertEqual(int(second.action.action[5]), 0)

    def test_multi_nav_blocks_wrong_tilemap_before_farm_route(self) -> None:
        task = MultiMapNavTask(
            waypoints=[Waypoint(tilemap=0x00, target_px=(137, 375), run_direction="down")],
            initial_settle_frames=0,
        )
        world = make_transition_world(0x15, current_tile=(8, 7))

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("expected tilemap 0x00", result.reason or "")

    def test_multi_nav_run_direction_corrects_off_lane_before_running(self) -> None:
        task = MultiMapNavTask(
            waypoints=[Waypoint(tilemap=0x00, target_px=(200, 200), radius=12, run_direction="right")],
            initial_settle_frames=0,
        )
        world = make_transition_world(0x00, current_tile=(10, 15))

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[4]), 1)
        self.assertEqual(int(result.action.action[7]), 0)

    def test_multi_nav_bfs_routes_around_farm_weeds(self) -> None:
        world = make_transition_world(0x00, current_tile=(10, 10))
        world.ram[ADDR_MAP + 10 * 64 + 11] = WEED
        task = MultiMapNavTask(
            waypoints=[Waypoint(tilemap=0x00, target_px=(16 * 16 + 8, 10 * 16 + 8), radius=8)],
            initial_settle_frames=0,
        )

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIn((11, 10), task._farm_weed_blocks)
        self.assertNotIn((11, 10), task._navigator.path)
        # First move must take the clear vertical detour, not step right onto weed.
        self.assertEqual(int(result.action.action[7]), 0)

    def test_farm_exit_uses_south_return_route_after_berry_ship(self) -> None:
        from harvest.planner.tasks.inventory_exit import FarmExitTask

        world = make_transition_world(0x00, current_tile=(62, 60))
        task = FarmExitTask(timeout=10_000)

        task.reset(world)

        self.assertIsInstance(task._nav, MultiMapNavTask)
        self.assertEqual(task._nav.waypoints, ROUTES["farm_south_to_west_gate"])

    def test_berry_ship_fails_closed_without_shipping_money_delta(self) -> None:
        from harvest.tasks.berry_ship import BerryShipTask

        class DoneNav:
            def step(self, world):
                return TaskResult(status=TaskStatus.SUCCESS, reason="route done")

        world = make_transition_world(0x00, current_tile=(62, 60))
        task = BerryShipTask(waypoints=[Waypoint(0x00, (1001, 969))])
        task._shipping_before = 0
        task._nav = DoneNav()

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("unverified", result.reason or "")

        world.ram[ADDR_SHIPPING_MONEY] = 15
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("0->150", result.reason or "")

    def test_day_plan_resume_after_hotswap_clears_active_task_state(self) -> None:
        plan = DayPlanTask(phase_sequence=[PhaseSpec("NAV_TO_COOP", "multi_nav", {"route": "farm_to_coop"})])
        world = make_transition_world(0x00, current_tile=(8, 13))
        plan.reset(world)
        plan.step(world)
        task = plan.current_task
        self.assertIsInstance(task, MultiMapNavTask)
        task._action_queue.append(make_action(a=True))
        task._phase = "action_drain"
        task._navigator.path = [(9, 13)]

        plan.resume_after_hotswap(world)

        self.assertEqual(task._phase, "nav")
        self.assertEqual(len(task._action_queue), 0)
        self.assertEqual(task._navigator.path, [])

    def test_ensure_carry_tool_is_noop_when_tool_already_held(self) -> None:
        world = make_world(0x00)
        world.ram[0x0921] = 0x10
        task = EnsureCarryToolTask(tool_id=0x10)

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "tool 0x10 ready")

    def test_ensure_carry_tool_uses_farmside_shed_approach_target(self) -> None:
        world = make_world(0x00)
        task = EnsureCarryToolTask(tool_id=int(Tool.WATERING_CAN))

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "route")
        self.assertIsInstance(task._task, MultiMapNavTask)
        self.assertEqual(task._task.initial_settle_frames, 0)
        self.assertEqual(task._task.waypoints[0].target_px, (137, 375))
        self.assertEqual(task._task.waypoints[1].target_px, (244, 375))
        self.assertEqual(task._task.waypoints[-1].target_px, (424, 489))

    def test_shed_routes_stop_below_doorway_for_transition(self) -> None:
        self.assertEqual(ROUTES["farm_to_shed"][-1].target_px, (424, 489))
        self.assertEqual(ROUTES["upper_farm_to_shed"][-1].target_px, (424, 489))
        self.assertEqual(ROUTES["field_to_shed"][-1].target_px, (424, 489))
        self.assertNotIn((422, 478), [waypoint.target_px for waypoint in ROUTES["farm_to_shed"]])
        self.assertIn((354, 489), [waypoint.target_px for waypoint in ROUTES["farm_to_shed"]])
        self.assertNotIn((456, 424), [waypoint.target_px for waypoint in ROUTES["farm_to_shed"]])
        self.assertNotIn((456, 489), [waypoint.target_px for waypoint in ROUTES["farm_to_shed"]])

    def test_shed_fetch_exits_shed_when_item_missing_after_shelf(self) -> None:
        class DoneTask:
            def step(self, world):
                return TaskResult(status=TaskStatus.SUCCESS, reason="done")

        world = make_world(0x26)
        world.ram[0x0921] = int(Tool.WATERING_CAN)
        world.ram[0x0923] = int(Tool.BRUSH)
        task = ShedFetchItemTask(
            item_id=0x07,
            shelf=SHED_SEED_SPECS["potato"],
        )
        task._phase = "inside"
        task._task = DoneTask()

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "exit_after_failure")
        self.assertIsInstance(task._task, ExitToFarmTask)

    def test_shed_fetch_fails_after_shed_exit_without_item(self) -> None:
        class DoneTask:
            def step(self, world):
                return TaskResult(status=TaskStatus.SUCCESS, reason="done")

        world = make_world(0x00)
        world.ram[0x0921] = int(Tool.WATERING_CAN)
        world.ram[0x0923] = int(Tool.BRUSH)
        task = ShedFetchItemTask(item_id=0x07, shelf=SHED_SEED_SPECS["potato"])
        task._phase = "exit_after_failure"
        task._task = DoneTask()
        task._failed_reason = "tool 0x07 missing after shed task"

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertEqual(result.reason, "tool 0x07 missing after shed task")

    def test_ensure_carry_tool_exits_shed_when_already_holding_tool(self) -> None:
        """Watering-can ensure must leave the shed so NAV_CROP can run outdoors."""
        world = make_world(0x26)
        world.ram[0x0921] = int(Tool.WATERING_CAN)
        world.ram[0x0923] = 0x00
        task = EnsureCarryToolTask(tool_id=int(Tool.WATERING_CAN))

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "exit_after_ready")
        self.assertIsInstance(task._task, ExitToFarmTask)

class BuildDayPhasesCommonTests(DayPlanPhaseHelpers):
    """Tests for the dynamic day plan builder."""

    def test_minimal_day_has_only_exit(self) -> None:
        phases = build_day_phases(None, hour=16)
        self.assertEqual(self._phase_names(phases), ["EXIT_TO_FARM"])

    def test_eve_loop_target_uses_ten_heart_threshold(self) -> None:
        self.assertEqual(romance_points_for_hearts(10), 999)
        self.assertEqual(romance_points_for_hearts(1), 49)

    def test_build_day_phases_from_ram_defers_outdoor_work_until_after_exit(self) -> None:
        world = make_date_world(0x15, season=0, day=13, hour=6)
        world.ram[ADDR_WEEKDAY + 0x4000] = 6
        world.ram[0x092A + 0x4000] = 5
        world.ram[ADDR_CHICKEN_COUNT + 0x4000] = 1
        world.ram[ADDR_EGG_AVAILABLE + 0x4000] = 1

        phases = build_day_phases_from_ram(world.ram)

        self.assertEqual(
            [phase.phase for phase in phases],
            [
                "EXIT_TO_FARM",
                "NAV_TO_COOP",
                "ENTER_COOP",
                "COOP_CHORES",
                "EXIT_COOP",
                "DYNAMIC_OUTDOOR_PLAN",
            ],
        )


if __name__ == "__main__":
    unittest.main()

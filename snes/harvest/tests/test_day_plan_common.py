"""Core day-plan orchestration, transitions, eve loop, shed tools.

Nav / multi-map / directional transition suites live in
``tests.test_day_plan_common_nav``. Split from test_day_plan_sequences monofile.
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
    make_time_world,
    make_transition_world,
    make_world,
    set_live_u16,
    set_player_pos,
)

import json
import tempfile
import unittest

from harvest.planner.day_plan import (
    ADDR_CHICKEN_COUNT,
    ADDR_EGG_AVAILABLE,
    ADDR_WEEKDAY,
    ActionResult,
    CrossMapRecordedTask,
    DayPlanTask,
    DeadlineCheckTask,
    EnsureCarryToolTask,
    ExitToFarmTask,
    ShedFetchItemTask,
    EveTalkLoopTask,
    EVE_TALK_LOOP_PHASES,
    MultiMapNavTask,
    PHASE_SEQUENCES,
    PhaseSpec,
    RecordedTransitionTask,
    SHED_SEED_SPECS,
    TaskResult,
    TaskStatus,
    build_day_phases,
    build_day_phases_from_ram,
    make_action,
    romance_points_for_hearts,
)
from harvest.core.tile_catalog import (
    ADDR_INPUT_LOCK,
    Tool,
)
from harvest.maps.map_config import ROUTES
from harvest.core.ram_catalog import field_spec


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

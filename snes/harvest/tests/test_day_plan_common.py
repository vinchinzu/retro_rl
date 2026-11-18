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
    set_money,
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
    ShedShelfToolTask,
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

    def test_eve_loop_sequence_runs_from_bar_exterior(self) -> None:
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
                        PhaseSpec("MOUNTAIN_BERRY", "mountain_berry"),
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
            ["BERRY_RUN_WINDOW", "MOUNTAIN_BERRY"],
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

    def test_establish_that_fails_after_recovery_is_skipped_not_fatal(self) -> None:
        """rings_d3 Spring D19-D24: one ring that would not plant cost the day's watering."""

        class FailTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.FAILURE, reason="nav_pocket_plant: nav timeout")

        class ActionTask(FailTask):
            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(right=True)))

        class RecoverySuccessTask(FailTask):
            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.SUCCESS, reason="recovered")

        class TestPlan(DayPlanTask):
            def __init__(self) -> None:
                super().__init__(
                    phase_sequence=[
                        PhaseSpec("CROP_ESTABLISH", "crop", {"work_mode": "establish"}),
                        PhaseSpec("CROP_WATER", "crop", {"work_mode": "water"}),
                    ]
                )
                self.recovery_attempts = 0

            def _make_task(self, spec, world):
                return FailTask() if spec.phase == "CROP_ESTABLISH" else ActionTask()

            def _make_recovery_task(self, spec, status, reason, world):
                self.recovery_attempts += 1
                return RecoverySuccessTask()

        plan = TestPlan()
        world = make_world(0x00)
        plan.reset(world)

        result = None
        for _ in range(6):
            result = plan.step(world)
            if plan.phase_text == "CROP_WATER":
                break

        self.assertEqual(plan.recovery_attempts, 1)
        self.assertEqual(plan.phase_text, "CROP_WATER")
        self.assertEqual(result.status, TaskStatus.RUNNING)

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
        # CROP_WATER, not ENTER_BARN: the animal block is an optional money
        # route now, so it no longer demonstrates required-phase abort.
        plan = DayPlanTask(phase_sequence=[PhaseSpec("CROP_WATER", "unknown_kind")])
        world = make_world(0x00)
        plan.reset(world)

        result = plan.step(world)

        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("required phase CROP_WATER failed: no task", result.reason)
        self.assertEqual(plan.phase_text, "CROP_WATER")

    def test_phase_watchdog_fails_a_stuck_crop_phase(self) -> None:
        """run25 D30: CROP_WATER's pond-refill retry loop had no cap and ran
        ~560k frames with the clock frozen. A looping phase must fail at the
        watchdog so recovery and the rest of the day can run."""
        from harvest.planner import day_plan_orchestrator as orch
        from retro_harness import ActionResult, Task, TaskResult

        class Spinner(Task):
            name = "spinner"

            def reset(self, world) -> None:
                pass

            def can_start(self, world) -> bool:
                return True

            def step(self, world):
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action()),
                )

        class SpinPlan(DayPlanTask):
            def _make_task(self, spec, world):
                return Spinner()

        plan = SpinPlan(phase_sequence=[PhaseSpec("CROP_WATER", "crop", {"watchdog_frames": 50})])
        world = make_world(0x00)
        plan.reset(world)
        status = TaskStatus.RUNNING
        for _ in range(200):
            result = plan.step(world)
            status = result.status
            if status != TaskStatus.RUNNING:
                break
        self.assertIn(status, (TaskStatus.FAILURE, TaskStatus.BLOCKED))
        self.assertIn("phase watchdog", result.reason)
        self.assertEqual(orch._phase_watchdog_frames(PhaseSpec("CROP_WATER", "crop")), 24_000)
        # Every other phase gets the 120k stall backstop (the day floor is off), so nothing can loop for millions of frames.
        from harvest.planner.day_phase_types import PHASE_BACKSTOP_FRAMES
        self.assertEqual(orch._phase_watchdog_frames(PhaseSpec("NAV_CROP", "nav")), PHASE_BACKSTOP_FRAMES)

    def test_day_plan_defers_the_animal_block_instead_of_losing_the_day(self) -> None:
        """run16 D26/D27: the cow-purchase day replaced the income plan with
        required phases, NAV_TO_ANIMAL_SHOP failed, and both days earned
        nothing. Buying a cow is discretionary; the farm work behind it is not."""
        plan = DayPlanTask(
            phase_sequence=[
                PhaseSpec("NAV_TO_ANIMAL_SHOP", "unknown_kind"),
                PhaseSpec("BUY_COW_VENDOR", "unknown_kind"),
                PhaseSpec("COW_CHORES", "unknown_kind"),
            ]
        )
        world = make_world(0x00)
        plan.reset(world)

        result = plan.step(world)

        self.assertNotEqual(result.status, TaskStatus.FAILURE)
        self.assertEqual(
            [d.phase for d in plan.deferred_plans],
            ["NAV_TO_ANIMAL_SHOP", "BUY_COW_VENDOR", "COW_CHORES"],
        )

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
        self.assertEqual(ROUTES["pocket_to_shed"][-1].target_px, (424, 489))
        self.assertEqual(ROUTES["upper_farm_to_shed"][-1].target_px, (424, 489))
        self.assertEqual(ROUTES["field_to_shed"][-1].target_px, (424, 489))
        self.assertEqual(ROUTES["pocket_to_shed"][0].target_px, (244, 375))
        self.assertNotIn((137, 375), [waypoint.target_px for waypoint in ROUTES["pocket_to_shed"]])
        self.assertNotIn((422, 478), [waypoint.target_px for waypoint in ROUTES["farm_to_shed"]])
        self.assertIn((354, 489), [waypoint.target_px for waypoint in ROUTES["farm_to_shed"]])
        self.assertNotIn((456, 424), [waypoint.target_px for waypoint in ROUTES["farm_to_shed"]])
        self.assertNotIn((456, 489), [waypoint.target_px for waypoint in ROUTES["farm_to_shed"]])
        # Field/near approaches must not B-run the stump line or the door.
        self.assertIn((400, 504), [waypoint.target_px for waypoint in ROUTES["field_to_shed"]])
        self.assertTrue(all(wp.run_direction is None for wp in ROUTES["field_to_shed"]))
        self.assertTrue(all(wp.run_direction is None for wp in ROUTES["near_shed_to_shed"]))

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

    def test_shed_shelf_fails_after_take_attempts(self) -> None:
        world = make_world(0x26)
        world.ram[0x0921] = 0x00
        world.ram[0x0923] = 0x00
        task = ShedShelfToolTask(
            tool_id=int(Tool.HOE),
            settle_frames=0,
            take_attempts=2,
        )
        task.reset(world)
        task._phase = "take"
        task._take_tries = 2

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("missing after shelf", result.reason or "")


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

    def test_spring_d23_schedules_the_flower_festival_for_eve(self) -> None:
        world = make_date_world(0x15, season=0, day=23, hour=6)
        world.ram[ADDR_WEEKDAY + 0x4000] = 2

        phases = build_day_phases_from_ram(world.ram)
        names = [phase.phase for phase in phases]

        self.assertEqual(names[0], "EXIT_TO_FARM")
        festival = next(phase for phase in phases if phase.phase == "ATTEND_FESTIVAL")
        self.assertEqual(festival.kind, "flower_festival")
        self.assertTrue(festival.params.get("ends_day"))
        self.assertLess(names.index("EXIT_TO_FARM"), names.index("ATTEND_FESTIVAL"))

    def test_eve_proposal_replaces_the_day_once_the_feather_is_carried(self) -> None:
        world = make_date_world(0x00, season=0, day=27, hour=8)
        world.ram[ADDR_WEEKDAY + 0x4000] = 6
        world.ram[0x0921] = 0x18
        set_live_u16(world.ram, "eve_hearts", 250)

        phases = build_day_phases_from_ram(world.ram)

        self.assertEqual([phase.phase for phase in phases], ["PROPOSE_EVE"])
        self.assertEqual(phases[0].kind, "propose_eve")

    def test_eve_proposal_exits_the_house_before_the_mountain(self) -> None:
        world = make_date_world(0x15, season=0, day=27, hour=8)
        world.ram[ADDR_WEEKDAY + 0x4000] = 6
        world.ram[0x0921] = 0x18
        set_live_u16(world.ram, "eve_hearts", 250)

        phases = build_day_phases_from_ram(world.ram)

        self.assertEqual([phase.phase for phase in phases], ["EXIT_TO_FARM", "PROPOSE_EVE"])

    def test_sunday_peddler_is_scheduled_when_eve_can_marry(self) -> None:
        world = make_date_world(0x15, season=0, day=28, hour=8)
        world.ram[ADDR_WEEKDAY + 0x4000] = 0
        set_live_u16(world.ram, "eve_hearts", 250)
        set_live_u16(world.ram, "upgrade_flags", 0x40)
        set_money(world.ram, 1000)

        phases = build_day_phases_from_ram(world.ram)
        names = [phase.phase for phase in phases]

        self.assertEqual(names[0], "EXIT_TO_FARM")
        self.assertEqual(names[1], "BUY_FEATHER")
        self.assertEqual(phases[1].kind, "feather_buy")

    def test_sunday_peddler_waits_for_the_first_house_upgrade(self) -> None:
        world = make_date_world(0x15, season=0, day=28, hour=8)
        world.ram[ADDR_WEEKDAY + 0x4000] = 0
        set_live_u16(world.ram, "eve_hearts", 250)
        set_money(world.ram, 1000)

        names = [phase.phase for phase in build_day_phases_from_ram(world.ram)]

        self.assertEqual(names[0], "EXIT_TO_FARM")
        self.assertNotIn("BUY_FEATHER", names)

        set_live_u16(world.ram, "upgrade_flags", 0x40)
        set_live_u16(world.ram, "family_event_flags", 0x40)
        names = [phase.phase for phase in build_day_phases_from_ram(world.ram)]
        self.assertNotIn("BUY_FEATHER", names)

    def test_shed_fetch_is_scheduled_after_the_shelf_bit(self) -> None:
        world = make_date_world(0x15, season=0, day=29, hour=8)
        world.ram[ADDR_WEEKDAY + 0x4000] = 1
        world.ram[0x11F02 + 0x4000] = 0x80
        set_live_u16(world.ram, "eve_hearts", 250)

        phases = build_day_phases_from_ram(world.ram)
        names = [phase.phase for phase in phases]

        self.assertEqual(names[0], "EXIT_TO_FARM")
        fetch = next(phase for phase in phases if phase.phase == "FETCH_FEATHER")
        self.assertEqual(fetch.params.get("tool_id"), 0x18)
        self.assertNotIn("PROPOSE_EVE", names)

    def test_open_bar_replaces_a_late_house_sleep(self) -> None:
        world = make_date_world(0x15, season=0, day=10, hour=18)
        world.ram[ADDR_WEEKDAY + 0x4000] = 2

        phases = build_day_phases_from_ram(world.ram)

        self.assertEqual(
            [phase.phase for phase in phases],
            ["EVE_TALK_LOOP", "RETURN_HOME", "GO_TO_SLEEP"],
        )
        self.assertEqual(phases[0].params.get("target_points"), 108)

    def test_dead_zone_night_does_not_schedule_the_bar(self) -> None:
        world = make_date_world(0x15, season=0, day=10, hour=18)
        world.ram[ADDR_WEEKDAY + 0x4000] = 2
        set_live_u16(world.ram, "eve_hearts", 130)

        phases = build_day_phases_from_ram(world.ram)

        self.assertEqual([phase.phase for phase in phases], ["GO_TO_SLEEP"])


if __name__ == "__main__":
    unittest.main()

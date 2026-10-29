"""Power-on / multi-day / sleep planner sequences.

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


class DayPlanSequencePowerOnTests(unittest.TestCase):
    def test_boot_to_day2_sequence_chains_macros_and_sleep(self) -> None:
        phases = PHASE_SEQUENCES["boot_to_day2"]
        names = [phase.phase for phase in phases]

        self.assertEqual(names[0], "EXIT_TO_FARM")
        self.assertIn("GET_HAMMER", names)
        self.assertIn("BUY_SEEDS", names)
        self.assertIn("TOWN_EXPLORE", names)
        self.assertIn("READY_TO_GO_HOME", names)
        self.assertEqual(names[-2:], ["RETURN_HOME", "GO_TO_SLEEP"])

    def test_complete_outdoor_morning_intro_already_ready(self) -> None:
        """Gate B: CompleteOutdoorMorningIntroTask no-ops when flags+free ready."""
        from harvest.core.ram_catalog import live_wram_base
        from harvest.planner.tasks.inventory import CompleteOutdoorMorningIntroTask

        world = make_world(0x00)
        set_player_pos(world.ram, 136, 424)
        base = live_wram_base(world.ram)
        world.ram[base + 0x00D2] = 0x01
        world.ram[base + 0x00D3] = 0x40  # free-move
        world.ram[base + 0x11F68] = 0xB1
        world.ram[base + 0x11F69] = 0x00
        task = CompleteOutdoorMorningIntroTask()
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("already ready", result.reason or "")

    def test_complete_outdoor_morning_intro_name_entry_presses(self) -> None:
        """Gate B: dog name tilemap 0x5F uses PowerOn-style AAAA + OK path."""
        from harvest.planner.tasks.inventory import CompleteOutdoorMorningIntroTask

        world = make_world(0x5F)
        # input_lock @ ADDR_INPUT_LOCK
        world.ram[ADDR_INPUT_LOCK] = 5
        world.ram[0x099F] = 3  # dog name kind
        world.ram[0x0994] = 0  # name length
        world.ram[0x0991] = 0  # cursor
        # free-move off, intro incomplete
        from harvest.core.ram_catalog import live_wram_base

        base = live_wram_base(world.ram)
        world.ram[base + 0x00D2] = 0x01
        world.ram[base + 0x00D3] = 0x00
        world.ram[base + 0x11F68] = 0x31
        world.ram[base + 0x11F69] = 0x00

        task = CompleteOutdoorMorningIntroTask()
        task.reset(world)
        r1 = task.step(world)
        self.assertEqual(r1.status, TaskStatus.RUNNING)
        self.assertIsNotNone(r1.action)
        self.assertIn("dog name char", r1.reason or "")

        # After 4 chars, cursor 0 → LEFT toward OK
        world.ram[0x0994] = 4
        world.ram[0x0991] = 0
        task._last_input_step = -100
        r2 = task.step(world)
        self.assertEqual(r2.status, TaskStatus.RUNNING)
        self.assertIn("move to OK", r2.reason or "")

        world.ram[0x0991] = 40
        task._last_input_step = -100
        r3 = task.step(world)
        self.assertIn("move to OK", r3.reason or "")

        world.ram[0x0991] = 70
        task._last_input_step = -100
        r4 = task.step(world)
        self.assertIn("confirm dog name", r4.reason or "")
        self.assertTrue(task._name_submitted)

    def test_name_screen_is_submitted_before_nav_or_rescue(self) -> None:
        """rr-bu62.18: tilemap 0x5F at (0, 0) must not time out into a rescue walk.

        Horse/calf/dog share the grid. The scripted world applies the same
        cursor moves PowerOnStartTask expects. B deletes a character.
        """
        from harvest.core.scene import SceneGuard, name_entry_press, unowned_scene_press
        from harvest.planner.day_phase_catalog import EXIT_TO_FARM_PHASE
        from harvest.planner.tasks.inventory_exit import ExitToFarmTask

        b_index = int(np.argmax(make_action(b=True)))

        def arm(tilemap: int = 0x5F) -> object:
            world = make_world(tilemap)
            world.ram[ADDR_INPUT_LOCK] = 5
            world.ram[0x099F] = 4  # horse; calf/dog use the same grid
            world.ram[0x0994] = 0
            world.ram[0x0991] = 0
            set_player_pos(world.ram, 0, 0)
            return world

        for tilemap in (0x5C, 0x5E):
            boot = arm(tilemap)
            self.assertIsNone(
                name_entry_press(boot.ram, step=0, last_press_step=-(10**9))
            )

        world = arm()
        guard = SceneGuard()
        # child_pressed is how DayPlanTask calls the guard; the name screen
        # still has to win, or the phase task navs from (0, 0).
        self.assertIsNotNone(
            unowned_scene_press(world.ram, guard, step=0, child_pressed=True)
        )
        self.assertNotIn(
            "b", unowned_scene_press(world.ram, guard, step=0, child_pressed=True)[0]
        )

        direct = ExitToFarmTask(cutscene_mash_limit=2, dismiss_mash_limit=2)
        direct.reset(world)
        self.assertIsNone(direct._task)
        self.assertEqual(direct._blocked_reason, "")

        plan = DayPlanTask(
            phase_sequence=[EXIT_TO_FARM_PHASE],
            policy=DayPlannerPolicy(include_end_day=False),
        )
        plan.reset(world)
        presses: list[str] = []
        for _ in range(500):
            if int(world.ram[ADDR_TILEMAP]) != 0x5F:
                break
            result = plan.step(world)
            self.assertEqual(result.status, TaskStatus.RUNNING)
            self.assertIsNone(plan._current_task)
            reason = result.reason or ""
            self.assertNotIn("timeout", reason)
            self.assertNotIn("rescue", reason)
            self.assertNotIn("unknown-map", reason)
            action = result.action.action
            self.assertEqual(int(np.asarray(action)[b_index]), 0)
            if np.array_equal(action, make_action(a=True)):
                if int(world.ram[0x0994]) < 4:
                    world.ram[0x0994] = int(world.ram[0x0994]) + 1
                    presses.append("a")
                elif int(world.ram[0x0991]) == 70:
                    world.ram[ADDR_TILEMAP] = 0x00
                    world.ram[ADDR_INPUT_LOCK] = 1
                    set_player_pos(world.ram, 136, 424)
                    presses.append("confirm")
            elif np.array_equal(action, make_action(left=True)):
                self.assertEqual(int(world.ram[0x0991]), 0)
                world.ram[0x0991] = 40
                presses.append("left")
            elif np.array_equal(action, make_action(up=True)):
                self.assertEqual(int(world.ram[0x0991]), 40)
                world.ram[0x0991] = 70
                presses.append("up")
        else:
            self.fail(f"still on name screen after scripted edges {presses}")

        self.assertEqual(presses, ["a", "a", "a", "a", "left", "up", "confirm"])
        self.assertNotEqual(int(world.ram[ADDR_TILEMAP]), 0x5F)
        pos_x = int(world.ram[ADDR_X]) | (int(world.ram[ADDR_X + 1]) << 8)
        pos_y = int(world.ram[ADDR_Y]) | (int(world.ram[ADDR_Y + 1]) << 8)
        self.assertNotEqual((pos_x, pos_y), (0, 0))
        self.assertIsNone(name_entry_press(world.ram, step=0, last_press_step=-(10**9)))

        # The exit composer itself must not unknown-map-timeout while naming.
        stuck = arm()
        exit_task = ExitToFarmTask(cutscene_mash_limit=2, dismiss_mash_limit=2)
        exit_task.reset(stuck)
        exit_presses: list[str] = []
        for _ in range(500):
            if int(stuck.ram[ADDR_TILEMAP]) != 0x5F:
                break
            result = exit_task.step(stuck)
            self.assertEqual(result.status, TaskStatus.RUNNING)
            self.assertIsNone(exit_task._task)
            self.assertNotIn("timeout", result.reason or "")
            action = result.action.action
            self.assertEqual(int(np.asarray(action)[b_index]), 0)
            if np.array_equal(action, make_action(a=True)):
                if int(stuck.ram[0x0994]) < 4:
                    stuck.ram[0x0994] = int(stuck.ram[0x0994]) + 1
                    exit_presses.append("a")
                elif int(stuck.ram[0x0991]) == 70:
                    stuck.ram[ADDR_TILEMAP] = 0x04
                    stuck.ram[ADDR_INPUT_LOCK] = 1
                    set_player_pos(stuck.ram, 200, 160)
                    exit_presses.append("confirm")
            elif np.array_equal(action, make_action(left=True)):
                stuck.ram[0x0991] = 40
                exit_presses.append("left")
            elif np.array_equal(action, make_action(up=True)):
                stuck.ram[0x0991] = 70
                exit_presses.append("up")
        else:
            self.fail(f"ExitToFarm stayed on 0x5F {exit_presses}")
        self.assertEqual(exit_presses, ["a", "a", "a", "a", "left", "up", "confirm"])
        self.assertNotEqual(int(stuck.ram[ADDR_TILEMAP]), 0x5F)
        self.assertNotEqual(
            (
                int(stuck.ram[ADDR_X]) | (int(stuck.ram[ADDR_X + 1]) << 8),
                int(stuck.ram[ADDR_Y]) | (int(stuck.ram[ADDR_Y + 1]) << 8),
            ),
            (0, 0),
        )

class SleepAndPlannerTests(unittest.TestCase):
    def test_go_to_sleep_succeeds_once_date_advances(self) -> None:
        task = GoToSleepTask(morning_ready_frames=3)
        start = make_date_world(0x15, season=0, day=12)
        task.reset(start)

        advanced = make_date_world(0x15, season=0, day=13)
        # Morning settle: wait for house-ready frames after day roll (Gate B).
        result = None
        for _ in range(5):
            result = task.step(advanced)
            if result.status != TaskStatus.RUNNING:
                break

        self.assertIsNotNone(result)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("day advanced", result.reason or "")

    def test_go_to_sleep_starts_return_home_when_outside_house(self) -> None:
        task = GoToSleepTask()
        world = make_date_world(0x00, season=0, day=12)
        set_player_pos(world.ram, 200, 500)
        task.reset(world)

        self.assertEqual(task._phase, "ensure_house")
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "ensure_house")
        self.assertIsNotNone(task._return_home)

    def test_go_to_sleep_succeeds_on_ending_credits(self) -> None:
        task = GoToSleepTask()
        start = make_date_world(0x15, season=3, day=30)
        task.reset(start)

        ending = make_date_world(0x15, season=3, day=30)
        set_live_u16(ending.ram, "ending_scene_index", 0x04)
        result = task.step(ending)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "ending reached")

    def test_go_to_sleep_dismisses_during_sleep_transition(self) -> None:
        task = GoToSleepTask()
        world = make_date_world(0x0F, season=0, day=12)
        task.reset(world)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(result.reason, "sleep transition")
        self.assertIsNotNone(result.action)

    def test_go_to_sleep_routes_stuck_left_wall_back_to_lower_lane(self) -> None:
        task = GoToSleepTask()
        world = make_date_world(0x15, season=0, day=12, hour=18)
        set_player_pos(world.ram, 77, 182)
        task.reset(world)

        result = task.step(world)
        action = result.action.action

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(action[5], 1)  # down
        self.assertEqual(action[0], 1)  # run
        self.assertEqual(action[4], 0)  # not up into the bed/wall edge
        self.assertEqual(action[6], 0)  # not left into the bed/wall edge

    def test_go_to_sleep_requires_tight_bed_column_before_interacting(self) -> None:
        task = GoToSleepTask()
        world = make_date_world(0x15, season=0, day=12, hour=18)
        set_player_pos(world.ram, 75, 86)
        task.reset(world)

        result = task.step(world)
        action = result.action.action

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "nav_bed")
        self.assertEqual(action[6], 1)  # left toward the proven x=70 stand point
        self.assertEqual(action[8], 0)  # do not press A from the loose column

    def test_go_to_sleep_confirmation_faces_up_then_plain_a(self) -> None:
        task = GoToSleepTask()
        # Bed A is a no-op before evening; tests force hour>=17.
        world = make_date_world(0x15, season=0, day=12, hour=18)
        set_player_pos(world.ram, 70, 86)
        task.reset(world)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "sleep_verify")
        # Face up at the bed stand, then A alone (not up+A).
        self.assertTrue(any(action[4] == 1 and action[8] == 0 for action in task._action_queue))
        self.assertTrue(
            any(action[8] == 1 and action[4] == 0 for action in task._action_queue)
        )
        self.assertFalse(
            any(action[8] == 1 and action[4] == 1 for action in task._action_queue)
        )
        # Recording taps B before the sleep confirmation.
        self.assertTrue(any(action[0] == 1 for action in task._action_queue))
        # Never face-left + A (walks into mattress / misses prompt — rr-m0wq).
        self.assertFalse(
            any(action[8] == 1 and action[6] == 1 for action in task._action_queue)
        )

    def test_go_to_sleep_waits_for_evening_before_bed_a(self) -> None:
        """Gate B: midday sleep exits outdoors (house clock freezes)."""
        task = GoToSleepTask()
        world = make_date_world(0x15, season=0, day=4, hour=13)
        set_player_pos(world.ram, 70, 86)
        task.reset(world)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._sleep_attempts, 0)
        # Must not mash bed A before evening — start exit-to-farm instead.
        self.assertIsNotNone(task._exit_for_evening)
        self.assertTrue(
            "evening" in (result.reason or "")
            or task._exit_for_evening is not None
        )

    def test_go_to_sleep_courts_eve_at_the_open_bar(self) -> None:
        task = GoToSleepTask()
        world = make_date_world(0x00, season=0, day=10, hour=17)
        world.ram[ADDR_WEEKDAY + 0x4000] = 2
        set_player_pos(world.ram, 136, 424)
        task.reset(world)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(task._eve)
        self.assertEqual(task._eve.target_points, 108)
        self.assertEqual(task._sleep_attempts, 0)

        world.ram[ADDR_HOUR + 0x4000] = 18
        capped = task.step(world)
        self.assertEqual(capped.status, TaskStatus.RUNNING)
        self.assertIsNotNone(task._eve)
        self.assertEqual(task._sleep_attempts, 0)

    def test_go_to_sleep_courts_eve_after_the_clock_caps(self) -> None:
        task = GoToSleepTask()
        world = make_date_world(0x00, season=0, day=10, hour=18)
        world.ram[ADDR_WEEKDAY + 0x4000] = 2
        set_player_pos(world.ram, 136, 424)
        task.reset(world)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(task._eve)
        self.assertEqual(task._eve.target_points, 108)
        self.assertEqual(task._phase, "ensure_house")

    def test_go_to_sleep_later_attempts_stay_face_up_no_post_a_b(self) -> None:
        """rr-m0wq: left-face A and B-after-A cancel Yes; stay face-up + A-only."""
        task = GoToSleepTask()
        world = make_date_world(0x15, season=0, day=12, hour=18)
        set_player_pos(world.ram, 70, 86)
        task.reset(world)
        task._sleep_attempts = 6
        task._phase = "sleep_attempt"
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "sleep_verify")
        queue = list(task._action_queue)
        # Still face up somewhere in the attempt.
        self.assertTrue(any(action[4] == 1 for action in queue))
        # No left+A combo on the confirm press.
        self.assertFalse(any(action[8] == 1 and action[6] == 1 for action in queue))
        # After the first A appears, no B should follow (B = No on sleep confirm).
        first_a = next(i for i, a in enumerate(queue) if a[8] == 1)
        self.assertFalse(any(a[0] == 1 for a in queue[first_a:]))

    def test_go_to_sleep_level2_uses_recorded_wife_bed_position(self) -> None:
        task = GoToSleepTask()
        world = make_date_world(0x17, season=1, day=6, hour=18)
        set_player_pos(world.ram, 294, 102)
        task.reset(world)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "sleep_verify")
        # Face up toward the wife bed, then A alone (not up+A).
        self.assertTrue(any(action[4] == 1 and action[8] == 0 for action in task._action_queue))
        self.assertTrue(any(action[8] == 1 and action[4] == 0 for action in task._action_queue))
        self.assertFalse(
            any(action[8] == 1 and action[4] == 1 for action in task._action_queue)
        )

    def test_go_to_sleep_level2_does_not_use_base_bed_position(self) -> None:
        task = GoToSleepTask()
        world = make_date_world(0x17, season=1, day=6, hour=18)
        set_player_pos(world.ram, 70, 86)
        task.reset(world)

        result = task.step(world)
        action = result.action.action

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "nav_bed")
        self.assertEqual(action[8], 0)
        self.assertEqual(action[6], 1)

    def test_go_to_sleep_level2_routes_entry_toward_recorded_bed_lane(self) -> None:
        task = GoToSleepTask()
        world = make_date_world(0x17, season=1, day=6, hour=18)
        set_player_pos(world.ram, 136, 330)
        task.reset(world)

        result = task.step(world)
        action = result.action.action

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "nav_bed")
        self.assertEqual(action[4], 1)
        self.assertEqual(action[0], 1)

    def test_multi_day_planner_rebuilds_day_task_after_sleep(self) -> None:
        class ImmediateSuccessTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.SUCCESS, reason="done")

        class WaitForNextDayTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                day = int(world.ram[ADDR_DAY + 0x4000])
                if day >= 2:
                    return TaskResult(status=TaskStatus.SUCCESS, reason="slept")
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        class TestPlanner(MultiDayPlannerTask):
            def __init__(self) -> None:
                super().__init__(until_season=0, until_day=2, max_days=3)
                self.day_builds = 0

            def _build_day_task(self, world):
                self.day_builds += 1
                return ImmediateSuccessTask()

            def _build_return_home_task(self):
                return ImmediateSuccessTask()

            def _build_sleep_task(self):
                return WaitForNextDayTask()

        planner = TestPlanner()
        planner.morning_settle_frames = 2
        day1 = make_date_world(0x15, season=0, day=1)
        planner.reset(day1)

        planner.step(day1)
        day2 = make_date_world(0x15, season=0, day=2)
        for _ in range(16):
            planner.step(day2)
            if planner.day_builds >= 2 and planner.phase_text == "PLAN_DAY":
                break

        self.assertEqual(planner.day_builds, 2)
        self.assertEqual(planner.phase_text, "PLAN_DAY")

    def test_multi_day_planner_settles_morning_before_rebuild(self) -> None:
        class ImmediateSuccessTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.SUCCESS, reason="done")

        class AdvancingSleepTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                world.ram[ADDR_DAY + 0x4000] = int(world.ram[ADDR_DAY + 0x4000]) + 1
                return TaskResult(status=TaskStatus.SUCCESS, reason="day advanced")

        class TestPlanner(MultiDayPlannerTask):
            def __init__(self) -> None:
                super().__init__(target_days=3, max_days=3, morning_settle_frames=3)
                self.day_builds = 0

            def _build_day_task(self, world):
                self.day_builds += 1
                return ImmediateSuccessTask()

            def _build_return_home_task(self):
                return ImmediateSuccessTask()

            def _build_sleep_task(self):
                return AdvancingSleepTask()

        planner = TestPlanner()
        world = make_date_world(0x15, season=0, day=1)
        planner.reset(world)

        # Drive through first day plan → return_home → sleep.
        for _ in range(6):
            planner.step(world)
            if planner.phase_text == "SETTLE_MORNING":
                break

        self.assertEqual(planner.phase_text, "SETTLE_MORNING")
        builds_before = planner.day_builds

        for _ in range(3):
            planner.step(world)

        self.assertEqual(planner.phase_text, "PLAN_DAY")
        self.assertEqual(planner.day_builds, builds_before)

    def test_multi_day_planner_stops_on_ending_after_sleep(self) -> None:
        class ImmediateSuccessTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.SUCCESS, reason="done")

        class EndingSleepTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                set_live_u16(world.ram, "ending_scene_index", 0x25)
                return TaskResult(status=TaskStatus.SUCCESS, reason="ending reached")

        class TestPlanner(MultiDayPlannerTask):
            def _build_day_task(self, world):
                return ImmediateSuccessTask()

            def _build_return_home_task(self):
                return ImmediateSuccessTask()

            def _build_sleep_task(self):
                return EndingSleepTask()

        planner = TestPlanner(target_days=10, max_days=10)
        world = make_date_world(0x15, season=3, day=30)
        planner.reset(world)

        result = TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
        for _ in range(8):
            result = planner.step(world)
            if result.status == TaskStatus.SUCCESS:
                break

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "ending reached")

    def test_multi_day_planner_detects_overnight_while_return_home_running(self) -> None:
        class ImmediateSuccessTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.SUCCESS, reason="done")

        class OvernightWhileRunningTask:
            def __init__(self) -> None:
                self.steps = 0

            def reset(self, world) -> None:
                self.steps = 0

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                self.steps += 1
                if self.steps >= 2:
                    world.ram[ADDR_DAY + 0x4000] = int(world.ram[ADDR_DAY + 0x4000]) + 1
                    world.ram[ADDR_TILEMAP] = 0x0F
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action()),
                    reason="still exiting",
                )

        class TestPlanner(MultiDayPlannerTask):
            def __init__(self) -> None:
                super().__init__(target_days=1, max_days=1, morning_settle_frames=1)

            def _build_day_task(self, world):
                return ImmediateSuccessTask()

            def _build_return_home_task(self):
                return OvernightWhileRunningTask()

        planner = TestPlanner()
        world = make_date_world(0x05, season=1, day=10)
        planner.reset(world)
        planner.step(world)  # plan_day success → return_home

        result = planner.step(world)  # return_home running day 10
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(planner.phase_text, "RETURN_HOME")

        result = planner.step(world)  # overnight while RUNNING → settle morning
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(planner.phase_text, "SETTLE_MORNING")
        self.assertEqual(planner.day_failures[0]["phase"], "return_home")

        # Fake overnight leaves sleep-transition tilemap; land in morning house.
        world.ram[ADDR_TILEMAP] = 0x15
        set_player_pos(world.ram, 136, 120)
        world.ram[ADDR_HOUR + 0x4000] = 6
        for _ in range(4):
            result = planner.step(world)
            if result.status == TaskStatus.SUCCESS:
                break

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "target date reached")

    def test_multi_day_planner_forces_sleep_route_when_day_plan_blocks(self) -> None:
        class BlockedDayTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.BLOCKED, reason="required phase EXIT_TO_FARM failed")

        class ImmediateSuccessTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.SUCCESS, reason="done")

        class TestPlanner(MultiDayPlannerTask):
            def _build_day_task(self, world):
                return BlockedDayTask()

            def _build_return_home_task(self):
                return ImmediateSuccessTask()

        planner = TestPlanner(until_season=0, until_day=2, max_days=3)
        world = make_date_world(0x15, season=0, day=1)
        planner.reset(world)

        result = planner.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(planner.phase_text, "RETURN_HOME")
        self.assertEqual(len(planner.day_failures), 1)
        self.assertEqual(planner.day_failures[0]["reason"], "required phase EXIT_TO_FARM failed")

        result = planner.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(planner.phase_text, "SLEEP")

    def test_multi_day_planner_days_target_completes_after_requested_sleeps(self) -> None:
        class ImmediateSuccessTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.SUCCESS, reason="done")

        class AdvancingSleepTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                world.ram[ADDR_DAY + 0x4000] = int(world.ram[ADDR_DAY + 0x4000]) + 1
                return TaskResult(status=TaskStatus.SUCCESS, reason="slept")

        class TestPlanner(MultiDayPlannerTask):
            def __init__(self) -> None:
                super().__init__(target_days=2, max_days=2)

            def _build_day_task(self, world):
                return ImmediateSuccessTask()

            def _build_return_home_task(self):
                return ImmediateSuccessTask()

            def _build_sleep_task(self):
                return AdvancingSleepTask()

        planner = TestPlanner()
        planner.morning_settle_frames = 1
        world = make_date_world(0x15, season=0, day=1)
        planner.reset(world)

        result = TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
        for _ in range(32):
            result = planner.step(world)
            if result.status == TaskStatus.SUCCESS:
                break

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "target date reached")
        # After two overnights from day 1, morning is day 3; settle updates active day.
        self.assertEqual(planner.progress_text, "date=0:3 completed=2/2")

    def test_multi_day_journal_keeps_planning_and_runtime_deferrals(self) -> None:
        planner = MultiDayPlannerTask(target_days=1, max_days=1)
        world = make_date_world(0x15, season=0, day=4)
        planner.reset(world)
        planner._last_day_decision = DayPlanDecision(
            phases=(),
            facts=PlanningFacts(source="test", weekday=1, hour=6),
            deferred=(
                DeferredPlan(
                    phase="CROP_WATER",
                    kind="crop",
                    reason="rainy_day",
                    params={"work_mode": "water"},
                ),
            ),
        )
        # The runtime can report the same deferral as the initial plan plus a
        # later route miss.  Preserve decision metadata and do not duplicate it.
        planner._last_day_deferred = [
            {"phase": "CROP_WATER", "reason": "rainy_day", "retry": "tomorrow"},
            {"phase": "BUY_SEEDS", "reason": "seed_shop_cutoff", "retry": "tomorrow"},
        ]

        planner._journal_day_complete(world, sleep_reason="test")

        deferred = planner.day_journal[-1]["deferred"]
        self.assertEqual([item["phase"] for item in deferred], ["CROP_WATER", "BUY_SEEDS"])
        self.assertEqual(deferred[0]["params"], {"work_mode": "water"})

    def test_multi_day_planner_counts_event_transition_during_return_home(self) -> None:
        class ImmediateSuccessTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.SUCCESS, reason="done")

        class EventTransitionTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                world.ram[ADDR_DAY + 0x4000] = int(world.ram[ADDR_DAY + 0x4000]) + 1
                world.ram[ADDR_TILEMAP] = 0x0F
                return TaskResult(status=TaskStatus.FAILURE, reason="expected tilemap 0x05, got 0x0F")

        class TestPlanner(MultiDayPlannerTask):
            def __init__(self) -> None:
                super().__init__(target_days=1, max_days=1, morning_settle_frames=1)

            def _build_day_task(self, world):
                return ImmediateSuccessTask()

            def _build_return_home_task(self):
                return EventTransitionTask()

        planner = TestPlanner()
        world = make_date_world(0x05, season=1, day=10)
        planner.reset(world)
        planner.step(world)

        result = planner.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(planner.phase_text, "SETTLE_MORNING")
        self.assertEqual(planner.day_failures[0]["phase"], "return_home")

        world.ram[ADDR_TILEMAP] = 0x15
        set_player_pos(world.ram, 136, 120)
        world.ram[ADDR_HOUR + 0x4000] = 6
        for _ in range(4):
            result = planner.step(world)
            if result.status == TaskStatus.SUCCESS:
                break

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "target date reached")
        self.assertEqual(planner.progress_text, "date=1:11 completed=1/1")


if __name__ == "__main__":
    unittest.main()

"""Home / exit-to-farm / return-home day-plan sequences.

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


class DayPlanSequenceHomeTests(unittest.TestCase):
    def test_day_plan_appends_return_home_after_late_optional_route_skip(self) -> None:
        class FailTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.FAILURE, reason="cutoff reached at 18:06")

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
                        PhaseSpec("BERRY_RECORDING_WINDOW", "deadline"),
                        PhaseSpec("GET_BERRIES_AND_SHIP", "recorded"),
                    ]
                )
                self._tasks = [FailTask(), ActionTask()]

            def _make_task(self, spec, world):
                return self._tasks.pop(0)

        plan = TestPlan()
        world = make_date_world(0x00, season=0, day=13, hour=18, minute=6)
        plan.reset(world)

        result = plan.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(plan.phase_text, "RETURN_HOME")
        self.assertEqual(
            [phase.phase for phase in plan.phases],
            ["BERRY_RUN_WINDOW", "EXIT_FARM_WEST", "BERRY_RECORDING_WINDOW", "GET_BERRIES_AND_SHIP"],
        )
        self.assertEqual([phase.phase for phase in plan.runtime_phases[-2:]], ["RETURN_HOME", "GO_TO_SLEEP"])

    def test_day_plan_advances_exit_phase_when_recovery_reaches_target_map(self) -> None:
        class FailingExitTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.FAILURE, reason="exit blocked")

        class ActionTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(right=True)))

        class RecoveryToFarmTask:
            def reset(self, world) -> None:
                return None

            def can_start(self, world) -> bool:
                return True

            def step(self, world) -> TaskResult:
                world.ram[ADDR_TILEMAP] = 0x01
                set_player_pos(world.ram, 328, 360)
                return TaskResult(status=TaskStatus.SUCCESS, reason="recovered to farm")

        class TestPlan(DayPlanTask):
            def __init__(self) -> None:
                super().__init__(
                    phase_sequence=[
                        PhaseSpec(
                            "EXIT_BARN",
                            "directional_transition",
                            {"target_tilemap": 0x00},
                        ),
                        PhaseSpec("ENSURE_WATERING_CAN", "ensure_tool"),
                    ]
                )
                self.exit_attempts = 0

            def _make_task(self, spec, world):
                if spec.phase == "EXIT_BARN":
                    self.exit_attempts += 1
                    return FailingExitTask()
                return ActionTask()

            def _make_recovery_task(self, spec, status, reason, world):
                return RecoveryToFarmTask()

        plan = TestPlan()
        world = make_world(0x27)
        plan.reset(world)

        result = plan.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(plan.phase_text, "ENSURE_WATERING_CAN")
        self.assertEqual(plan.exit_attempts, 1)

    def test_exit_to_farm_uses_return_route_from_path(self) -> None:
        task = ExitToFarmTask()
        world = make_world(0x0C)

        task.reset(world)

        self.assertIsInstance(task._task, MultiMapNavTask)
        self.assertEqual(task._task.name, "return_path_to_farm")
        self.assertEqual(task._task.waypoints[-1].target_px, (244, 128))

    def test_exit_to_farm_uses_event_town_return_route(self) -> None:
        task = ExitToFarmTask()
        world = make_world(0x05)
        set_player_pos(world.ram, 440, 160)

        task.reset(world)

        self.assertIsInstance(task._task, MultiMapNavTask)
        self.assertEqual(task._task.name, "return_event_town_to_farm")
        self.assertEqual(task._task.waypoints[0].tilemap, 0x05)
        self.assertEqual(task._task.waypoints[-1].target_px, (244, 128))

    def test_exit_to_farm_succeeds_on_seasonal_farm_tilemaps(self) -> None:
        for tilemap in (0x00, 0x01, 0x02, 0x03):
            with self.subTest(tilemap=tilemap):
                task = ExitToFarmTask()
                world = make_world(tilemap)

                task.reset(world)
                result = task.step(world)

                self.assertTrue(is_farm_tilemap(tilemap))
                self.assertIsNone(task._task)
                self.assertEqual(result.status, TaskStatus.SUCCESS)
                self.assertEqual(result.reason, f"tilemap=0x{tilemap:02X}")

    def test_exit_to_farm_uses_scripted_house_exit(self) -> None:
        task = ExitToFarmTask()
        world = make_world(0x15)

        task.reset(world)

        self.assertIsInstance(task._task, ExitBuildingTask)
        self.assertEqual(task._task.target_tilemap, 0x00)
        self.assertEqual(task._task.timeout, 2200)

    def test_exit_to_farm_accepts_remodeled_house_tilemaps(self) -> None:
        for tilemap in (0x16, 0x17):
            with self.subTest(tilemap=tilemap):
                task = ExitToFarmTask()
                world = make_world(tilemap)

                task.reset(world)

                self.assertIsInstance(task._task, ExitBuildingTask)
                self.assertTrue(is_house_tilemap(tilemap))

    def test_exit_to_farm_mashes_then_blocks_stuck_cutscene(self) -> None:
        task = ExitToFarmTask(cutscene_mash_limit=2, dismiss_mash_limit=2)
        world = make_world(0xFE)
        set_player_pos(world.ram, 136, 424)

        task.reset(world)
        first = task.step(world)
        second = task.step(world)
        blocked = task.step(world)

        self.assertEqual(first.status, TaskStatus.RUNNING)
        self.assertEqual(second.status, TaskStatus.RUNNING)
        # rr-uru1: FAILURE (not BLOCKED) so ReturnHome can recover-mash/retry.
        self.assertEqual(blocked.status, TaskStatus.FAILURE)
        self.assertTrue(
            "timeout" in (blocked.reason or "")
            or "cutscene" in (blocked.reason or "")
            or "unknown" in (blocked.reason or "")
        )

    def test_exit_to_farm_step_mashes_cutscene_after_lazy_reset(self) -> None:
        task = ExitToFarmTask(cutscene_mash_limit=1, dismiss_mash_limit=1)
        world = make_world(0xFE)
        set_player_pos(world.ram, 136, 424)

        first = task.step(world)
        blocked = task.step(world)

        self.assertEqual(first.status, TaskStatus.RUNNING)
        self.assertEqual(blocked.status, TaskStatus.FAILURE)
        self.assertTrue(
            "timeout" in (blocked.reason or "")
            or "cutscene" in (blocked.reason or "")
        )

    def test_exit_to_farm_dialogue_dismiss_times_out(self) -> None:
        """rr-uru1: dialogue@unknown must not mash forever until outer timeout."""
        task = ExitToFarmTask(dismiss_mash_limit=3, cutscene_mash_limit=3)
        world = make_world(0x08)
        set_player_pos(world.ram, 281, 357)
        world.ram[ADDR_INPUT_LOCK] = 2
        # Non-zero dialog text so classifier prefers DIALOGUE over INPUT_LOCKED.
        text_addr = field_spec("dialog_text_id").address
        world.ram[text_addr] = 0x65
        world.ram[text_addr + 1] = 0x03

        task.reset(world)
        statuses = []
        last = None
        for _ in range(8):
            last = task.step(world)
            statuses.append(last.status)
            if last.status != TaskStatus.RUNNING:
                break

        self.assertIsNotNone(last)
        assert last is not None
        self.assertEqual(last.status, TaskStatus.FAILURE)
        self.assertIn("timeout", last.reason or "")
        self.assertIn("0x08", last.reason or "")
        self.assertTrue(any(s == TaskStatus.RUNNING for s in statuses[:-1]))

    def test_exit_to_farm_intermittent_dialogue_still_times_out(self) -> None:
        """rr-uru1 root cause: free frames must not reset the sticky budget."""
        task = ExitToFarmTask(dismiss_mash_limit=4, cutscene_mash_limit=4)
        world = make_world(0x08)
        set_player_pos(world.ram, 281, 361)
        text_addr = field_spec("dialog_text_id").address
        world.ram[text_addr] = 0x0A
        world.ram[text_addr + 1] = 0x00

        task.reset(world)
        last = None
        for i in range(12):
            # Alternate locked dialogue and free input on the same unknown map.
            world.ram[ADDR_INPUT_LOCK] = 2 if i % 2 == 0 else 1
            last = task.step(world)
            if last.status != TaskStatus.RUNNING:
                break

        self.assertIsNotNone(last)
        assert last is not None
        self.assertEqual(last.status, TaskStatus.FAILURE)
        self.assertIn("timeout", last.reason or "")
        self.assertIn("0x08", last.reason or "")

    def test_exit_to_farm_does_not_false_success_on_unknown_map(self) -> None:
        """Unregistered map must never SUCCESS as if already on farm."""
        task = ExitToFarmTask(cutscene_mash_limit=2, dismiss_mash_limit=2)
        world = make_world(0x08)
        set_player_pos(world.ram, 281, 357)
        world.ram[ADDR_INPUT_LOCK] = 1  # free; unregistered → cutscene_event

        task.reset(world)
        last = None
        for _ in range(6):
            last = task.step(world)
            if last.status != TaskStatus.RUNNING:
                break

        self.assertIsNotNone(last)
        assert last is not None
        self.assertNotEqual(last.status, TaskStatus.SUCCESS)
        self.assertIn(last.status, {TaskStatus.BLOCKED, TaskStatus.FAILURE})

    def test_return_home_retries_exit_to_farm_after_failure(self) -> None:
        """rr-uru1: ExitToFarm FAILURE triggers recover mash then re-route."""
        world = make_world(0x08)
        set_player_pos(world.ram, 281, 357)
        world.ram[ADDR_INPUT_LOCK] = 2
        task = ReturnHomeTask(exit_to_farm_retry_limit=2)
        task.reset(world)
        task._phase = "exit_to_farm"
        task._task = SimpleNamespace(
            step=lambda _w: TaskResult(
                status=TaskStatus.FAILURE,
                reason="dialogue dismiss timeout from dialogue@unknown",
            )
        )

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "exit_to_farm_recover")
        self.assertEqual(task._exit_to_farm_retries, 1)
        self.assertGreater(len(task._action_queue), 0)

    def test_house_exit_relocalizes_from_live_position(self) -> None:
        task = ExitBuildingTask(target_tilemap=0x00)
        world = make_transition_world(0x15, current_tile=(8, 7))

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[7]), 1)
        self.assertEqual(int(result.action.action[5]), 0)

    def test_house_exit_recovers_from_doorway_offset(self) -> None:
        task = ExitBuildingTask(target_tilemap=0x00)
        world = make_transition_world(0x15, current_tile=(10, 12))

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[6]), 1)
        self.assertEqual(int(result.action.action[5]), 0)

    def test_house_exit_waits_for_stable_farm_coordinates(self) -> None:
        player_state_addr = field_spec("player_state").address
        task = ExitBuildingTask(target_tilemap=0x00, settle_frames=1)
        world = make_transition_world(0x00, current_tile=(8, 13))
        set_player_pos(world.ram, 137, 212)
        world.ram[player_state_addr] = 0x81

        task.reset(world)
        task._step_count = 30
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        # Mid-warp outdoor (y < 330): hold down+B — neutral settle freezes control (rr-bhr).
        self.assertEqual(result.reason, "exit mid-warp push")
        self.assertIsNotNone(result.action)
        self.assertEqual(int(result.action.action[5]), 1)  # down

        set_player_pos(world.ram, 137, 344)
        world.ram[player_state_addr] = 0x01
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)

    def test_house_exit_accepts_seasonal_farm_tilemap(self) -> None:
        player_state_addr = field_spec("player_state").address
        task = ExitBuildingTask(target_tilemap=0x00, settle_frames=1)
        world = make_world(0x01)
        set_player_pos(world.ram, 137, 344)
        world.ram[player_state_addr] = 0x01

        task.reset(world)
        task._step_count = 30
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(result.reason, "tilemap=0x01")

    def test_farm_free_move_ready_and_house_front_softlock(self) -> None:
        """Gate B: free-move bit 0x4000 distinguishes good outdoor from soft-lock."""
        from harvest.core.ram_catalog import LIVE_RAM_WRAM_OFFSET, live_wram_base
        from harvest.planner.tasks.inventory import (
            EVENT_1F68_OUTDOOR_INTRO_MASK,
            farm_free_move_ready,
            farm_house_front_softlock,
            outdoor_intro_flags_ready,
        )

        world = make_world(0x00)
        set_player_pos(world.ram, 136, 344)
        # game_state u16 @ 0x00D2; base 0 for 0x20000 ram fixtures
        lo = live_wram_base(world.ram) + 0x00D2
        world.ram[lo] = 0x01
        world.ram[lo + 1] = 0x40  # 0x4001 free-move
        self.assertTrue(farm_free_move_ready(world.ram))
        self.assertFalse(farm_house_front_softlock(world.ram))

        world.ram[lo + 1] = 0x00  # 0x0001 control lost
        set_player_pos(world.ram, 133, 425)
        self.assertFalse(farm_free_move_ready(world.ram))
        self.assertTrue(farm_house_front_softlock(world.ram))

        # Live-size snapshot still resolves free-move via WRAM mirror.
        live = make_time_world(0x00, day=2, hour=6, minute=0, live_offset=True)
        set_player_pos(live.ram, 136, 344)
        lo2 = LIVE_RAM_WRAM_OFFSET + 0x00D2
        live.ram[lo2] = 0x01
        live.ram[lo2 + 1] = 0x40
        self.assertTrue(farm_free_move_ready(live.ram))

        # outdoor intro flags: truck 0x0011 incomplete; 0x00A1 / 0x00B1 ready
        f68 = live_wram_base(world.ram) + 0x11F68
        world.ram[f68] = 0x11
        world.ram[f68 + 1] = 0x00
        self.assertFalse(outdoor_intro_flags_ready(world.ram))
        world.ram[f68] = EVENT_1F68_OUTDOOR_INTRO_MASK & 0xFF  # 0xA1
        world.ram[f68 + 1] = 0x00
        self.assertTrue(outdoor_intro_flags_ready(world.ram))
        world.ram[f68] = 0xB1
        self.assertTrue(outdoor_intro_flags_ready(world.ram))

    def test_exit_to_farm_task_exits_shed_with_downward_transition(self) -> None:
        world = make_world(0x26)
        set_player_pos(world.ram, 8 * 16 + 8, 12 * 16 + 8)
        task = ExitToFarmTask()
        task.reset(world)

        first = task.step(world)
        self.assertEqual(first.status, TaskStatus.RUNNING)
        self.assertIsNotNone(first.action)
        self.assertEqual(int(first.action.action[5]), 1)

        for _ in range(13):
            task.step(make_world(0x26))
        target = make_world(0x00)
        set_player_pos(target.ram, 422, 489)
        for _ in range(4):
            self.assertEqual(task.step(target).status, TaskStatus.RUNNING)
        result = task.step(target)
        self.assertEqual(result.status, TaskStatus.SUCCESS)

    def test_exit_to_farm_task_routes_to_shed_door_before_exiting(self) -> None:
        world = make_world(0x26)
        set_player_pos(world.ram, 4 * 16 + 8, 11 * 16 + 8)
        world.ram[ADDR_MAP + 11 * 64 + 4] = 0xA1
        world.ram[ADDR_MAP + 11 * 64 + 5] = 0xA1
        world.ram[ADDR_MAP + 11 * 64 + 6] = 0xA1
        world.ram[ADDR_MAP + 11 * 64 + 7] = 0xA1
        world.ram[ADDR_MAP + 11 * 64 + 8] = 0xA1
        world.ram[ADDR_MAP + 12 * 64 + 8] = 0xA1
        task = ExitToFarmTask()

        task.reset(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsNotNone(result.action)
        self.assertEqual(task._task.stand_tile, (8, 12))

class BuildDayPhasesHomeTests(DayPlanPhaseHelpers):
    """Tests for the dynamic day plan builder."""

    def test_late_day_house_state_goes_directly_to_sleep(self) -> None:
        world = make_date_world(0x15, season=0, day=14, hour=18)
        world.ram[ADDR_WEEKDAY + 0x4000] = 0
        world.ram[ADDR_CHICKEN_COUNT + 0x4000] = 1
        world.ram[ADDR_EGG_AVAILABLE + 0x4000] = 1
        world.ram[0x092A + 0x4000] = 5
        world.ram[ADDR_MAP + 34 * 64 + 7] = 0x60

        phases = build_day_phases_from_ram(world.ram)

        self.assertEqual([phase.phase for phase in phases], ["GO_TO_SLEEP"])

    def test_late_day_remodeled_house_state_goes_directly_to_sleep(self) -> None:
        world = make_date_world(0x16, season=0, day=14, hour=18)

        phases = build_day_phases_from_ram(world.ram)

        self.assertEqual([phase.phase for phase in phases], ["GO_TO_SLEEP"])

    def test_return_home_enters_when_already_at_house_front(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=18)
        set_player_pos(world.ram, 136, 424)
        task = ReturnHomeTask()
        task.reset(world)

        task.step(world)

        self.assertEqual(task._phase, "enter_house")
        self.assertIsInstance(task._task, DirectionalTransitionTask)
        self.assertEqual(task._task.stand_tile, (8, 26))
        self.assertEqual(task._task.overshoot_limit_px, 328)
        self.assertTrue(task._task.require_empty_hands)

    def test_return_home_timeout_fails_cleanly(self) -> None:
        """Outer budget prevents multi-day hang when enter/nav never terminates."""
        world = make_date_world(0x00, season=0, day=13, hour=18)
        set_player_pos(world.ram, 400, 500)
        task = ReturnHomeTask(timeout=5)
        task.reset(world)
        # Child nav keeps RUNNING; outer timeout must still fire.
        task._task = SimpleNamespace(
            step=lambda _w: TaskResult(status=TaskStatus.RUNNING, reason="stuck nav")
        )
        task._phase = "nav_house_front"

        result = None
        for _ in range(12):
            result = task.step(world)
            if result.status != TaskStatus.RUNNING:
                break

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("timeout", result.reason or "")

    def test_return_home_succeeds_when_already_house_mid_exit_to_farm(self) -> None:
        """rr-ws8h: house tilemap short-circuits even if phase is exit_to_farm."""
        world = make_date_world(0x15, season=0, day=23, hour=18)
        task = ReturnHomeTask()
        task.reset(world)
        task._phase = "exit_to_farm"
        task._task = SimpleNamespace(
            step=lambda _w: TaskResult(
                status=TaskStatus.RUNNING, reason="stuck exit_to_farm"
            )
        )

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("already in house", result.reason or "")
        self.assertIn("exit_to_farm", result.reason or "")

    def test_return_home_timeout_succeeds_when_already_house(self) -> None:
        """rr-ws8h: hard timeout must not FAIL if player is already inside."""
        world = make_date_world(0x15, season=0, day=23, hour=18)
        task = ReturnHomeTask(timeout=5)
        task.reset(world)
        task._phase = "exit_to_farm"
        task._task = SimpleNamespace(
            step=lambda _w: TaskResult(
                status=TaskStatus.RUNNING, reason="stuck exit_to_farm"
            )
        )
        # Past timeout with house tilemap — short-circuit wins (via=step).
        task._total_steps = 5

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("already in house", result.reason or "")
        self.assertNotIn("timeout", result.reason or "")

    def test_return_home_remodel_tilemap_short_circuits(self) -> None:
        """Remodeled house tilemaps (0x16/0x17) also count as arrival."""
        world = make_date_world(0x16, season=0, day=10, hour=17)
        task = ReturnHomeTask()
        task.reset(world)
        task._phase = "nav_house_front"
        task._task = SimpleNamespace(
            step=lambda _w: TaskResult(status=TaskStatus.RUNNING, reason="nav")
        )

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("0x16", result.reason or "")

    def test_return_home_level2_tilemap_short_circuits_exit_to_farm(self) -> None:
        """House L2 (0x17) mid exit_to_farm must SUCCESS (not run child forever)."""
        world = make_date_world(0x17, season=0, day=10, hour=17)
        task = ReturnHomeTask()
        task.reset(world)
        task._phase = "exit_to_farm"
        task._task = SimpleNamespace(
            step=lambda _w: TaskResult(
                status=TaskStatus.RUNNING, reason="stuck exit_to_farm"
            )
        )

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("0x17", result.reason or "")
        self.assertIn("exit_to_farm", result.reason or "")
        self.assertIn("via=step", result.reason or "")

    def test_return_home_start_next_phase_short_circuits_when_house(self) -> None:
        """_start_next_phase must not spawn exit_to_farm when already house."""
        world = make_date_world(0x15, season=0, day=23, hour=18)
        task = ReturnHomeTask()
        task.reset(world)
        # Fresh task: phase=start, no child — same path as reset → first phase pick.
        self.assertIsNone(task._task)
        self.assertEqual(task._phase, "start")

        result = task._start_next_phase(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("already in house", result.reason or "")
        self.assertIn("via=start_next_phase", result.reason or "")
        self.assertIsNone(task._task)

    def test_return_home_timeout_exit_to_farm_on_non_house_fails_with_phase(
        self,
    ) -> None:
        """Soak residual: stuck exit_to_farm off-house must FAIL with phase=."""
        # D23 power_on end: dialogue@unknown tilemap=0x08 — not house/farm.
        world = make_date_world(0x08, season=0, day=23, hour=7)
        task = ReturnHomeTask(timeout=5)
        task.reset(world)
        task._phase = "exit_to_farm"
        task._task = SimpleNamespace(
            step=lambda _w: TaskResult(
                status=TaskStatus.RUNNING, reason="stuck exit_to_farm"
            )
        )

        result = None
        for _ in range(12):
            result = task.step(world)
            if result.status != TaskStatus.RUNNING:
                break

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("timeout", result.reason or "")
        self.assertIn("phase=exit_to_farm", result.reason or "")

    def test_return_home_timeout_on_farm_fails_with_phase(self) -> None:
        """Hard timeout on farm (never entered house) stays FAILURE + phase."""
        world = make_date_world(0x00, season=0, day=13, hour=18)
        set_player_pos(world.ram, 400, 500)
        task = ReturnHomeTask(timeout=5)
        task.reset(world)
        task._phase = "nav_house_front"
        task._task = SimpleNamespace(
            step=lambda _w: TaskResult(status=TaskStatus.RUNNING, reason="stuck nav")
        )

        result = None
        for _ in range(12):
            result = task.step(world)
            if result.status != TaskStatus.RUNNING:
                break

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("timeout", result.reason or "")
        self.assertIn("phase=nav_house_front", result.reason or "")

    def test_return_home_renavs_when_stuck_north_of_door_stand(self) -> None:
        """Mid-door tiles (~y=389) must walk south before pushing up."""
        world = make_date_world(0x00, season=0, day=13, hour=18)
        set_player_pos(world.ram, 137, 389)
        task = ReturnHomeTask()
        task.reset(world)

        task.step(world)

        self.assertEqual(task._phase, "nav_house_front")
        self.assertIsInstance(task._task, NavTask)
        self.assertEqual(
            (task._task.target_px.x, task._task.target_px.y),
            (136, 424),
        )

    def test_return_home_enter_uses_catalog_stand_not_overshoot_tile(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=18)
        set_player_pos(world.ram, 136, 424)
        task = ReturnHomeTask()
        enter = task._house_enter_task(world)

        self.assertEqual(enter.stand_tile, (8, 26))
        self.assertEqual(enter.door_align_px, 136)
        self.assertEqual(enter.overshoot_limit_px, 328)

    def test_return_home_navs_when_far_from_house_front(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=18)
        set_player_pos(world.ram, 400, 500)
        task = ReturnHomeTask()
        task.reset(world)

        task.step(world)

        self.assertEqual(task._phase, "nav_house_front")
        self.assertIsInstance(task._task, MultiMapNavTask)
        self.assertEqual(task._task.waypoints[-1].target_px, (136, 424))

    def test_return_home_uses_remodeled_house_waypoint_from_upgrade_flags(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=18)
        world.ram[field_spec("upgrade_flags").address + 0x4000] = 0x40
        set_player_pos(world.ram, 400, 500)
        task = ReturnHomeTask()
        task.reset(world)

        task.step(world)

        self.assertEqual(task._phase, "nav_house_front")
        self.assertIsInstance(task._task, MultiMapNavTask)
        self.assertEqual(task._task.waypoints[-1].target_px, (136, 344))

    def test_return_home_enter_house_accepts_remodel_tilemap(self) -> None:
        world = make_date_world(0x00, season=0, day=13, hour=18)
        task = ReturnHomeTask()
        task._phase = "nav_house_front"
        task._task = SimpleNamespace(step=lambda _world: TaskResult(status=TaskStatus.SUCCESS, reason="arrived"))

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "enter_house")
        self.assertIsInstance(task._task, DirectionalTransitionTask)
        self.assertIn(0x16, task._task.target_tilemaps)
        self.assertEqual(task._task.stand_tile, (8, 26))
        self.assertEqual(task._task.door_align_px, 136)
        self.assertTrue(task._task.require_empty_hands)

    def test_return_home_navs_to_drop_spot_when_hands_full_in_field(self) -> None:
        """rr-6g7g: CLEAR_FIELD may finish holding a stone far from the house."""
        from harvest.core.animal_status import ADDR_HELD_ITEM
        from harvest.core.ram_catalog import live_wram_base

        world = make_date_world(0x00, season=0, day=7, hour=17)
        set_player_pos(world.ram, 89, 726)
        base = live_wram_base(world.ram)
        world.ram[ADDR_HELD_ITEM + base] = 0x0D
        world.ram[field_spec("player_state").address + base] = 0x03
        task = ReturnHomeTask()
        task.reset(world)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "nav_drop_spot")
        self.assertEqual(task._drop_spot_navs, 1)
        # Deep south: densified MultiMapNav ending at drop spot ~(136,480).
        self.assertIsInstance(task._task, MultiMapNavTask)
        self.assertEqual(task._task.waypoints[-1].target_px, (136, 480))
        self.assertGreaterEqual(len(task._task.waypoints), 2)

    def test_return_home_densifies_south_field_approach(self) -> None:
        """rr-5in: mid-wall south of y=31 → east of pond (not x≈248 or pond)."""
        world = make_date_world(0x00, season=0, day=9, hour=14)
        # Mid-wall x (under fence body). Exhaust pre-escape budget so densify
        # multi_nav is exercised (pre-escape now covers mid-south of fence).
        set_player_pos(world.ram, 280, 620)
        task = ReturnHomeTask()
        task.reset(world)
        task._south_escape_attempts = task.south_escape_limit

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "nav_house_front")
        self.assertIsInstance(task._task, MultiMapNavTask)
        wps = task._task.waypoints
        self.assertGreaterEqual(len(wps), 3)
        self.assertEqual(wps[-1].target_px, (136, 424))
        # East of pond free lane (tile x≥36 → px≥576); never pond column 512.
        self.assertGreaterEqual(wps[0].target_px[0], 576)
        self.assertEqual(wps[0].run_direction, "right")
        for wp in wps:
            if wp.run_direction == "up":
                self.assertGreaterEqual(wp.target_px[0], 576)
                # Must not lateral-align through pond body (x≈512).
                self.assertNotEqual(wp.target_px[0], 512)

    def test_return_home_far_east_pond_pre_escapes_before_approach(self) -> None:
        """rr-5in D12: ~(854,527) after water — west+north pre-escape, not pond crawl."""
        world = make_date_world(0x00, season=0, day=12, hour=18)
        set_player_pos(world.ram, 854, 527)
        task = ReturnHomeTask()
        task.reset(world)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "south_escape")
        self.assertEqual(task._south_escape_attempts, 1)
        self.assertIn("far-east", result.reason or "")

    def test_return_home_far_east_densifies_north_of_pond_lane(self) -> None:
        """After pre-escape budget spent, east free lane is east-of-pond then west above wall."""
        world = make_date_world(0x00, season=0, day=12, hour=18)
        set_player_pos(world.ram, 700, 520)
        task = ReturnHomeTask()
        task.reset(world)
        task._south_escape_attempts = task.south_escape_limit

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "nav_house_front")
        self.assertIsInstance(task._task, MultiMapNavTask)
        wps = task._task.waypoints
        self.assertEqual(wps[-1].target_px, (136, 424))
        # Northbound stages stay east of pond (x≥576), never 512.
        up_xs = [wp.target_px[0] for wp in wps if wp.run_direction == "up"]
        self.assertTrue(up_xs)
        for x in up_xs:
            self.assertGreaterEqual(x, 576)
            self.assertLessEqual(x, 640)
        # Eventually slides west above fence.
        self.assertTrue(any(wp.run_direction == "left" for wp in wps))

    def test_return_home_south_escape_on_fence_latitude_timeout(self) -> None:
        """South-of-fence but not deep_south (y=527) still escapes on multi_nav fail."""
        world = make_date_world(0x00, season=0, day=12, hour=18)
        set_player_pos(world.ram, 774, 521)
        task = ReturnHomeTask()
        task.reset(world)
        task._phase = "nav_house_front"
        task._task = SimpleNamespace(
            step=lambda _w: TaskResult(
                status=TaskStatus.FAILURE, reason="multi_nav timeout"
            )
        )

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "south_escape")
        self.assertEqual(task._south_escape_attempts, 1)
        self.assertIn("south escape", result.reason or "")

    def test_return_home_forces_enter_when_mid_yard_south_of_door(self) -> None:
        """D12 residual (118,486): force enter instead of hard multi_nav fail."""
        world = make_date_world(0x00, season=0, day=12, hour=18)
        set_player_pos(world.ram, 118, 486)
        task = ReturnHomeTask()
        task.reset(world)
        task._phase = "nav_house_front"
        task._task = SimpleNamespace(
            step=lambda _w: TaskResult(
                status=TaskStatus.FAILURE, reason="multi_nav timeout"
            )
        )

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "enter_house")
        self.assertIsInstance(task._task, DirectionalTransitionTask)

    def test_return_home_west_of_fence_keeps_near_x_lane(self) -> None:
        """West free side densify uses current x, not forced SW pocket x=96."""
        world = make_date_world(0x00, season=0, day=12, hour=18)
        set_player_pos(world.ram, 122, 518)
        task = ReturnHomeTask()
        task.reset(world)
        # Skip pre-escape so densify multi_nav is exercised.
        task._south_escape_attempts = task.south_escape_limit

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "nav_house_front")
        self.assertIsInstance(task._task, MultiMapNavTask)
        wps = task._task.waypoints
        self.assertEqual(wps[-1].target_px, (136, 424))
        # Northbound corridor stays near player x (not yanked to 96).
        up_xs = [wp.target_px[0] for wp in wps if wp.run_direction == "up"]
        self.assertTrue(up_xs)
        for x in up_xs:
            self.assertGreaterEqual(x, 110)
            self.assertLessEqual(x, 160)

    def test_return_home_west_of_fence_runs_north(self) -> None:
        """West of fence wall (px x<176): densify north on free side, not east."""
        world = make_date_world(0x00, season=0, day=8, hour=15)
        set_player_pos(world.ram, 120, 620)
        task = ReturnHomeTask()
        task.reset(world)
        task._south_escape_attempts = task.south_escape_limit

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "nav_house_front")
        self.assertIsInstance(task._task, MultiMapNavTask)
        wps = task._task.waypoints
        self.assertEqual(wps[-1].target_px, (136, 424))
        # West corridor x≈96, not east 512.
        self.assertLessEqual(wps[0].target_px[0], 160)
        self.assertEqual(wps[0].run_direction, "up")

    def test_return_home_pre_escapes_sw_pocket_before_approach(self) -> None:
        """Deep SW after CLEAR: B-run east first so multi_nav is not born stuck."""
        world = make_date_world(0x00, season=0, day=8, hour=15)
        set_player_pos(world.ram, 37, 715)
        task = ReturnHomeTask()
        task.reset(world)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "south_escape")
        self.assertEqual(task._south_escape_attempts, 1)
        self.assertIn("pre-escape", result.reason or "")

    def test_return_home_south_escape_on_multi_nav_timeout(self) -> None:
        """Far-south multi_nav fail queues B-run escape instead of hard-fail."""
        world = make_date_world(0x00, season=0, day=9, hour=18)
        set_player_pos(world.ram, 102, 726)
        task = ReturnHomeTask()
        task.reset(world)
        task._phase = "nav_house_front"
        task._task = SimpleNamespace(
            step=lambda _w: TaskResult(
                status=TaskStatus.FAILURE, reason="multi_nav timeout"
            )
        )

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "south_escape")
        self.assertEqual(task._south_escape_attempts, 1)
        self.assertTrue(task._action_queue)
        self.assertIn("south escape", result.reason or "")

    def test_return_home_uses_fence_gap_when_wall_confirmed(self) -> None:
        """Open y=31 gap after water: approach through gap, not only east end."""
        from harvest.core.tile_catalog import ADDR_MAP, FENCE, UNTILLED

        world = make_date_world(0x00, season=0, day=9, hour=14)
        set_player_pos(world.ram, 280, 560)
        # Solid fence wall x=11–29 with gap at x=14 (water refill opened it).
        for x in range(11, 30):
            world.ram[ADDR_MAP + 31 * 64 + x] = FENCE
        world.ram[ADDR_MAP + 31 * 64 + 14] = UNTILLED
        task = ReturnHomeTask()
        task.reset(world)
        # Skip mid-south pre-escape so gap densify multi_nav is exercised.
        task._south_escape_attempts = task.south_escape_limit

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIsInstance(task._task, MultiMapNavTask)
        wps = task._task.waypoints
        gap_px = 14 * 16 + 8  # 232
        self.assertTrue(any(abs(wp.target_px[0] - gap_px) <= 8 for wp in wps[:3]))
        self.assertEqual(wps[-1].target_px, (136, 424))

    def test_return_home_tosses_at_drop_spot_when_hands_full(self) -> None:
        from harvest.core.animal_status import ADDR_HELD_ITEM
        from harvest.core.ram_catalog import live_wram_base

        world = make_date_world(0x00, season=0, day=7, hour=17)
        set_player_pos(world.ram, 136, 480)
        base = live_wram_base(world.ram)
        world.ram[ADDR_HELD_ITEM + base] = 0x0D
        world.ram[field_spec("player_state").address + base] = 0x03
        task = ReturnHomeTask()
        task.reset(world)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(result.reason, "drop carried before house")
        self.assertEqual(task._phase, "drop_carried")
        self.assertEqual(task._drop_attempts, 1)
        self.assertTrue(task._action_queue)

    def test_return_home_fails_after_drop_budget_with_held_item(self) -> None:
        from harvest.core.animal_status import ADDR_HELD_ITEM
        from harvest.core.ram_catalog import live_wram_base

        world = make_date_world(0x00, season=0, day=7, hour=17)
        set_player_pos(world.ram, 136, 480)
        base = live_wram_base(world.ram)
        world.ram[ADDR_HELD_ITEM + base] = 0x0D
        world.ram[field_spec("player_state").address + base] = 0x03
        task = ReturnHomeTask(drop_attempt_limit=2)
        task.reset(world)
        task._drop_spot_navs = 3  # skip relocate
        task._drop_attempts = 2

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("could not clear hands before house entry", result.reason or "")
        self.assertIn("held=0x0D", result.reason or "")

    def test_return_home_low_budget_south_of_fence_short_charge(self) -> None:
        """D19 residual: after drop thrash, ~1k frames left at (153,518).

        Must queue compact east→north rather than hard-fail immediately.
        """
        world = make_date_world(0x00, season=0, day=19, hour=18)
        set_player_pos(world.ram, 153, 518)
        task = ReturnHomeTask(timeout=11000)
        task.reset(world)
        task._total_steps = 10000  # remaining ~999f
        task._drop_attempts = 3

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task._phase, "south_escape")
        self.assertIn("low-budget", result.reason or "")
        self.assertTrue(task._action_queue)

    def test_return_home_fails_early_on_stuck_same_held(self) -> None:
        """Power-on D19 residual: held=0x0F rock fragment never clears.

        Same held id across drop_stuck_held_limit observations must hard-fail
        before burning the outer 11k timeout in phase=drop_carried.
        """
        from harvest.core.animal_status import ADDR_HELD_ITEM
        from harvest.core.ram_catalog import live_wram_base

        world = make_date_world(0x00, season=0, day=19, hour=18)
        set_player_pos(world.ram, 136, 480)
        base = live_wram_base(world.ram)
        world.ram[ADDR_HELD_ITEM + base] = 0x0F
        world.ram[field_spec("player_state").address + base] = 0x03
        task = ReturnHomeTask(drop_stuck_held_limit=3, drop_attempt_limit=10)
        task.reset(world)
        task._drop_spot_navs = 3
        task._drop_deep_relocated = True  # deep south already tried
        task._drop_last_held = 0x0F
        task._drop_same_held = 2  # one more observation trips the gate
        task._drop_attempts = 2

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("could not clear hands before house entry", result.reason or "")
        self.assertIn("held=0x0F", result.reason or "")
        self.assertIn("same_held=", result.reason or "")


if __name__ == "__main__":
    unittest.main()

"""Home / exit-to-farm day-plan sequences.

Split from test_day_plan_home monofile (exit / house-exit sequences).
Return-home and late-day sleep live in ``test_day_plan_home_return``.
Discovered via pytest; also aggregated by ``test_day_plan_home`` shim.
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
    make_time_world,
    make_transition_world,
    make_world,
    set_player_pos,
)

import unittest
from types import SimpleNamespace

from harvest.planner.day_plan import (
    ADDR_TILEMAP,
    ActionResult,
    DayPlanTask,
    ExitBuildingTask,
    ExitToFarmTask,
    MultiMapNavTask,
    PhaseSpec,
    ReturnHomeTask,
    TaskResult,
    TaskStatus,
    is_farm_tilemap,
    is_house_tilemap,
    make_action,
)
from harvest.core.tile_catalog import (
    ADDR_INPUT_LOCK,
    ADDR_MAP,
)
from harvest.core.ram_catalog import field_spec


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
        intermediate = make_world(0x00)
        set_player_pos(intermediate.ram, 424, 392)
        self.assertEqual(task.step(intermediate).status, TaskStatus.RUNNING)
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


if __name__ == "__main__":
    unittest.main()

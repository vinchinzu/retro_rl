from __future__ import annotations

from collections import deque
from types import SimpleNamespace
import unittest

import numpy as np

import harvest  # noqa: F401 - ensures monorepo imports are on sys.path
from retro_harness import ActionResult, TaskResult, TaskStatus

from harvest.core.ram_catalog import LIVE_RAM_WRAM_OFFSET, field_spec
from harvest.core.scene import SceneLocation, SceneMode
from harvest.tasks.primitives import (
    PressAndVerifyTask,
    RamCondition,
    RetryTask,
    TaskSequence,
    WaitForRamConditionTask,
    WaitForSceneTask,
    dismiss_dialogue_result,
    drain_action_queue,
    press_a_sequence,
)


def _world() -> SimpleNamespace:
    ram = np.zeros(0x24000, dtype=np.uint8)
    ram[field_spec("tilemap").address] = 0x00
    ram[field_spec("input_lock").address] = 1
    _write_u16(ram, field_spec("player_x").address + LIVE_RAM_WRAM_OFFSET, 136)
    _write_u16(ram, field_spec("player_y").address + LIVE_RAM_WRAM_OFFSET, 424)
    return SimpleNamespace(ram=ram, info={}, obs=None)


def _write_u16(ram: np.ndarray, addr: int, value: int) -> None:
    ram[addr] = value & 0xFF
    ram[addr + 1] = (value >> 8) & 0xFF


class _ScriptedTask:
    def __init__(
        self,
        name: str,
        statuses: list[TaskStatus],
        *,
        can_start: bool = True,
        running_action: bool = False,
    ) -> None:
        self.name = name
        self.statuses = list(statuses)
        self.starts = can_start
        self.running_action = running_action
        self.resets = 0
        self.steps = 0

    def reset(self, world) -> None:
        self.resets += 1

    def can_start(self, world) -> bool:
        return self.starts

    def step(self, world) -> TaskResult:
        self.steps += 1
        status = self.statuses.pop(0) if self.statuses else TaskStatus.SUCCESS
        action = None
        if status == TaskStatus.RUNNING and self.running_action:
            action = ActionResult(np.ones(12, dtype=np.int32))
        return TaskResult(status=status, action=action, reason=f"{self.name} {status.value}")


class TaskPrimitiveTests(unittest.TestCase):
    def test_press_a_sequence_faces_presses_and_settles(self) -> None:
        actions = press_a_sequence(
            "up",
            face_frames=2,
            pre_press_settle_frames=1,
            hold_frames=3,
            settle_frames=1,
            hold_face_with_a=True,
        )

        self.assertEqual(len(actions), 7)
        self.assertEqual(int(actions[0][4]), 1)
        self.assertEqual(int(actions[3][4]), 1)
        self.assertEqual(int(actions[3][8]), 1)
        self.assertFalse(actions[-1].any())

    def test_dismiss_dialogue_result_pulses_on_even_frames(self) -> None:
        odd = dismiss_dialogue_result(1)
        even = dismiss_dialogue_result(2)

        self.assertFalse(odd.action.action.any())
        self.assertEqual(int(even.action.action[8]), 1)

    def test_drain_action_queue_returns_next_running_action(self) -> None:
        queue = deque([np.ones(12, dtype=np.int32)])

        result = drain_action_queue(queue)

        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(int(result.action.action.sum()), 12)
        self.assertIsNone(drain_action_queue(queue))

    def test_wait_for_ram_condition_requires_stable_frames(self) -> None:
        world = _world()
        world.ram[field_spec("day").address + LIVE_RAM_WRAM_OFFSET] = 14
        task = WaitForRamConditionTask(
            condition=RamCondition("day", expected=14),
            stable_frames=2,
            timeout=10,
        )
        task.reset(world)

        self.assertEqual(task.step(world).status, TaskStatus.RUNNING)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("observed 14", result.reason)

    def test_wait_for_ram_condition_timeout_reports_observed_value(self) -> None:
        world = _world()
        task = WaitForRamConditionTask(
            condition=RamCondition("day", expected=14),
            timeout=1,
        )
        task.reset(world)

        task.step(world)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("observed 0", result.reason)

    def test_wait_for_scene_matches_mode_and_location(self) -> None:
        world = _world()
        task = WaitForSceneTask(
            expected_mode=SceneMode.NORMAL,
            expected_location=SceneLocation.FARM,
            stable_frames=2,
            timeout=10,
        )
        task.reset(world)

        self.assertEqual(task.step(world).status, TaskStatus.RUNNING)
        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("normal_map@farm", result.reason)

    def test_press_and_verify_runs_sequence_before_condition(self) -> None:
        world = _world()
        sequence = press_a_sequence("right", face_frames=1, pre_press_settle_frames=0, hold_frames=1, settle_frames=0)
        task = PressAndVerifyTask(
            sequence=sequence,
            condition=RamCondition("input_lock", expected=1),
            timeout=10,
        )
        task.reset(world)

        first = task.step(world)
        second = task.step(world)

        self.assertEqual(first.status, TaskStatus.RUNNING)
        self.assertIsNotNone(first.action)
        self.assertEqual(second.status, TaskStatus.RUNNING)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)

    def test_task_sequence_runs_children_in_order(self) -> None:
        world = _world()
        first = _ScriptedTask("first", [TaskStatus.SUCCESS])
        second = _ScriptedTask("second", [TaskStatus.SUCCESS])
        task = TaskSequence(tasks=[first, second])
        task.reset(world)

        between = task.step(world)
        result = task.step(world)

        self.assertEqual(between.status, TaskStatus.RUNNING)
        self.assertIn("first complete", between.reason)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(first.resets, 1)
        self.assertEqual(second.resets, 1)

    def test_task_sequence_prefixes_child_failures(self) -> None:
        world = _world()
        task = TaskSequence(tasks=[_ScriptedTask("fragile", [TaskStatus.BLOCKED])])
        task.reset(world)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.BLOCKED)
        self.assertIn("fragile:", result.reason)

    def test_retry_task_recreates_child_until_success(self) -> None:
        world = _world()
        scripted = iter(
            [
                _ScriptedTask("attempt_one", [TaskStatus.FAILURE]),
                _ScriptedTask("attempt_two", [TaskStatus.SUCCESS]),
            ]
        )
        task = RetryTask(name="retry_script", task_factory=lambda: next(scripted), max_attempts=2)
        task.reset(world)

        retrying = task.step(world)
        result = task.step(world)

        self.assertEqual(retrying.status, TaskStatus.RUNNING)
        self.assertIn("retrying", retrying.reason)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(task.attempt, 2)

    def test_retry_task_reports_exhausted_attempts(self) -> None:
        world = _world()
        task = RetryTask(
            name="retry_script",
            task_factory=lambda: _ScriptedTask("always_blocked", [TaskStatus.BLOCKED]),
            max_attempts=1,
        )
        task.reset(world)

        result = task.step(world)

        self.assertEqual(result.status, TaskStatus.BLOCKED)
        self.assertIn("exhausted 1 attempts", result.reason)


if __name__ == "__main__":
    unittest.main()

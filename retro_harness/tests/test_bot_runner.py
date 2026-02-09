"""Tests for retro_harness.bot_runner module."""
import numpy as np
import pytest

from retro_harness.bot_runner import BotRunner, TaskRepeater, TaskSequencer
from retro_harness.protocol import ActionResult, TaskResult, TaskStatus, WorldState
from retro_harness.ram_state import RAMSchema


class FakeTask:
    """Minimal Task protocol implementation for testing."""

    name = "FakeTask"

    def __init__(self, steps_to_success=3):
        self.steps_to_success = steps_to_success
        self._step_count = 0
        self._reset_count = 0

    def reset(self, world):
        self._step_count = 0
        self._reset_count += 1

    def can_start(self, world):
        return True

    def step(self, world):
        self._step_count += 1
        if self._step_count >= self.steps_to_success:
            action = ActionResult(action=np.array([1, 0, 0, 0] + [0] * 8, dtype=np.int8))
            return TaskResult(status=TaskStatus.SUCCESS, action=action)
        action = ActionResult(action=np.array([0, 1, 0, 0] + [0] * 8, dtype=np.int8))
        return TaskResult(status=TaskStatus.RUNNING, action=action)


class FailTask:
    name = "FailTask"

    def reset(self, world): pass
    def can_start(self, world): return True

    def step(self, world):
        return TaskResult(status=TaskStatus.FAILURE, reason="always fails")


class TestBotRunner:
    def test_basic_step(self):
        task = FakeTask(steps_to_success=5)
        runner = BotRunner(task)
        obs = np.zeros((224, 256, 3), dtype=np.uint8)
        info = {"ram": np.zeros(2048, dtype=np.uint8)}

        # First call initializes
        action = runner(obs, info)
        assert action is not None
        assert len(action) == 12

    def test_returns_none_on_success(self):
        task = FakeTask(steps_to_success=1)
        runner = BotRunner(task)
        obs = np.zeros((1, 1, 3), dtype=np.uint8)
        info = {"ram": np.zeros(16, dtype=np.uint8)}

        result = runner(obs, info)
        assert result is None  # success -> None

    def test_returns_none_on_failure(self):
        runner = BotRunner(FailTask())
        obs = np.zeros((1, 1, 3), dtype=np.uint8)
        info = {"ram": np.zeros(16, dtype=np.uint8)}

        result = runner(obs, info)
        assert result is None

    def test_reset(self):
        task = FakeTask(steps_to_success=5)
        runner = BotRunner(task)
        obs = np.zeros((1, 1, 3), dtype=np.uint8)
        info = {"ram": np.zeros(16, dtype=np.uint8)}

        runner(obs, info)  # initializes
        runner.reset()
        assert runner._frame == 0
        assert runner._initialized is False

    def test_with_ram_schema(self):
        schema = RAMSchema({"health": (0x10, "u8")})
        task = FakeTask(steps_to_success=5)
        runner = BotRunner(task, ram_schema=schema)

        ram = np.zeros(256, dtype=np.uint8)
        ram[0x10] = 161
        obs = np.zeros((1, 1, 3), dtype=np.uint8)
        info = {"ram": ram}

        runner(obs, info)
        # Task was reset with meta containing health


class TestTaskSequencer:
    def test_runs_tasks_in_order(self):
        t1 = FakeTask(steps_to_success=1)
        t2 = FakeTask(steps_to_success=1)
        seq = TaskSequencer([t1, t2])

        world = WorldState(frame=0, ram=np.zeros(16, dtype=np.uint8), info={})
        seq.reset(world)

        # Step t1 to success
        r = seq.step(world)
        assert r.status == TaskStatus.RUNNING  # t1 done, but t2 started

        # Step t2 to success
        r = seq.step(world)
        assert r.status == TaskStatus.SUCCESS  # all done

    def test_current_task_index(self):
        seq = TaskSequencer([FakeTask(1), FakeTask(1)])
        world = WorldState(frame=0, ram=np.zeros(16, dtype=np.uint8), info={})
        seq.reset(world)
        assert seq.current_task_index == 0

        seq.step(world)  # t1 finishes
        assert seq.current_task_index == 1

    def test_empty_sequence(self):
        seq = TaskSequencer([])
        world = WorldState(frame=0, ram=np.zeros(16, dtype=np.uint8), info={})
        assert not seq.can_start(world)

    def test_current_task_none_after_complete(self):
        seq = TaskSequencer([FakeTask(1)])
        world = WorldState(frame=0, ram=np.zeros(16, dtype=np.uint8), info={})
        seq.reset(world)
        seq.step(world)  # completes
        assert seq.current_task is None


class TestTaskRepeater:
    def test_repeats_n_times(self):
        task = FakeTask(steps_to_success=1)
        rep = TaskRepeater(task, times=3)
        world = WorldState(frame=0, ram=np.zeros(16, dtype=np.uint8), info={})
        rep.reset(world)

        # Each call completes one task iteration
        r = rep.step(world)
        assert r.status == TaskStatus.RUNNING  # 1/3

        r = rep.step(world)
        assert r.status == TaskStatus.RUNNING  # 2/3

        r = rep.step(world)
        assert r.status == TaskStatus.SUCCESS  # 3/3 done

    def test_infinite_repeat(self):
        task = FakeTask(steps_to_success=1)
        rep = TaskRepeater(task, times=None)
        world = WorldState(frame=0, ram=np.zeros(16, dtype=np.uint8), info={})
        rep.reset(world)

        for _ in range(100):
            r = rep.step(world)
            assert r.status == TaskStatus.RUNNING

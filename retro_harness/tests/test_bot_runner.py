"""Tests for retro_harness.bot_runner module."""
import numpy as np

from retro_harness.actions import buttons, idle_action
from retro_harness.bot_runner import (
    ActionNode,
    BotRunner,
    Condition,
    NodeStatus,
    Selector,
    Sequence,
    StuckDetector,
    TaskRepeater,
    TaskSequencer,
    WatchdogEvent,
)
from retro_harness.input_script import FrameAction
from retro_harness.protocol import ActionResult, TaskResult, TaskStatus, WorldState
from retro_harness.ram_state import EnemyState, GameState, RAMSchema


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

    def test_mission_status_uses_current_task_name(self):
        runner = BotRunner(TaskSequencer([FakeTask(2), FakeTask(2)]))
        obs = np.zeros((1, 1, 3), dtype=np.uint8)
        info = {"ram": np.zeros(16, dtype=np.uint8)}

        runner(obs, info)
        status = runner.mission_status()

        assert status.mission_id == "TaskSequencer"
        assert status.phase == "FakeTask"


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


class TestNode:
    def test_condition_true_and_false(self):
        state = GameState(frame=0)
        assert Condition(lambda s: True, name="ok").tick(state).status is NodeStatus.SUCCESS
        assert Condition(lambda s: False, name="no").tick(state).status is NodeStatus.FAILURE

    def test_action_node_runs_then_succeeds(self):
        node = ActionNode(
            lambda s: FrameAction(action=buttons("A"), reason="go"),
            done=lambda s: s.health > 0,
            name="act",
        )
        r = node.tick(GameState(frame=0))
        assert r.status is NodeStatus.RUNNING
        assert r.action.action[8] == 1
        r = node.tick(GameState(frame=1, health=3))
        assert r.status is NodeStatus.SUCCESS


class TestSelector:
    def test_succeeds_on_first_success(self):
        sel = Selector(
            [
                Condition(lambda s: True, name="yes"),
                ActionNode(lambda s: FrameAction(action=buttons("A"), reason="go"), name="go"),
            ]
        )
        result = sel.tick(GameState(frame=0))
        assert result.status is NodeStatus.SUCCESS

    def test_runs_until_running_child(self):
        sel = Selector(
            [
                Condition(lambda s: False, name="no"),
                ActionNode(lambda s: FrameAction(action=buttons("A"), reason="go"), name="go"),
            ]
        )
        result = sel.tick(GameState(frame=0))
        assert result.status is NodeStatus.RUNNING
        assert result.action is not None
        assert result.action.action[8] == 1

    def test_idles_when_all_fail(self):
        sel = Selector(
            [Condition(lambda s: False, name="no"), Condition(lambda s: False, name="no2")],
            name="sel",
        )
        result = sel.tick(GameState(frame=0))
        assert result.status is NodeStatus.FAILURE
        assert result.action.action == idle_action()


class TestSequence:
    def test_succeeds_when_all_children_succeed(self):
        seq = Sequence(
            [Condition(lambda s: True, name="a"), Condition(lambda s: True, name="b")]
        )
        result = seq.tick(GameState(frame=0))
        assert result.status is NodeStatus.SUCCESS

    def test_fails_on_first_failure(self):
        seq = Sequence(
            [Condition(lambda s: True, name="a"), Condition(lambda s: False, name="b")]
        )
        result = seq.tick(GameState(frame=0))
        assert result.status is NodeStatus.FAILURE

    def test_running_short_circuits(self):
        seq = Sequence(
            [
                Condition(lambda s: True, name="a"),
                ActionNode(lambda s: FrameAction(action=buttons("A"), reason="go"), name="go"),
                Condition(lambda s: False, name="never"),  # not reached
            ]
        )
        result = seq.tick(GameState(frame=0))
        assert result.status is NodeStatus.RUNNING
        assert result.action.action[8] == 1


def _stuck_state(frame, x=0, y=0, cam=0, health=100, enemies=()):
    return GameState(
        frame=frame,
        player_x=x,
        player_y=y,
        camera_x=cam,
        health=health,
        enemies=enemies,
    )


class TestStuckDetector:
    def test_position_stall(self):
        detector = StuckDetector(position_window=3)
        st = _stuck_state(0, x=5)
        assert detector.update(st) is WatchdogEvent.NONE
        assert detector.update(st) is WatchdogEvent.NONE
        assert detector.update(st) is WatchdogEvent.NONE
        assert detector.update(st) is WatchdogEvent.POSITION_STALLED

    def test_movement_resets_position_stall(self):
        detector = StuckDetector(position_window=3)
        st = _stuck_state(0, x=5)
        detector.update(st)
        detector.update(st)  # stall 1
        st = _stuck_state(2, x=6)  # moved
        assert detector.update(st) is WatchdogEvent.NONE
        st = _stuck_state(3, x=6)
        detector.update(st)  # stall 1
        detector.update(st)  # stall 2
        assert detector.update(st) is WatchdogEvent.POSITION_STALLED

    def test_camera_stall(self):
        detector = StuckDetector(position_window=1000, camera_window=3)
        st = _stuck_state(0, cam=100)
        assert detector.update(st) is WatchdogEvent.NONE
        assert detector.update(st) is WatchdogEvent.NONE
        assert detector.update(st) is WatchdogEvent.NONE
        assert detector.update(st) is WatchdogEvent.CAMERA_STALLED

    def test_health_drain(self):
        detector = StuckDetector(position_window=1000, camera_window=1000, health_window=3)
        detector.update(_stuck_state(0, health=100))
        detector.update(_stuck_state(1, health=90))
        detector.update(_stuck_state(2, health=80))
        assert detector.update(_stuck_state(3, health=70)) is WatchdogEvent.HEALTH_DRAINING

    def test_health_heal_resets_drain(self):
        detector = StuckDetector(position_window=1000, camera_window=1000, health_window=3)
        detector.update(_stuck_state(0, health=100))
        detector.update(_stuck_state(1, health=90))  # drain 1
        detector.update(_stuck_state(2, health=120))  # heal resets
        detector.update(_stuck_state(3, health=110))  # drain 1
        assert detector.update(_stuck_state(4, health=100)) is WatchdogEvent.NONE

    def test_enemy_stall(self):
        detector = StuckDetector(position_window=1000, health_window=1000, enemy_window=3)
        st = _stuck_state(
            0,
            x=5,
            y=10,
            enemies=(EnemyState(slot=0, x=10, y=10, health=5, active=True),),
        )
        detector.update(st)
        detector.update(st)  # stall 1
        detector.update(st)  # stall 2
        assert detector.update(st) is WatchdogEvent.ENEMY_STALLED

    def test_enemy_damage_resets_enemy_stall(self):
        detector = StuckDetector(position_window=1000, health_window=1000, enemy_window=5)
        def st(hp):
            return _stuck_state(
                0,
                x=5,
                y=10,
                enemies=(EnemyState(slot=0, x=10, y=10, health=hp, active=True),),
            )
        detector.update(st(5))
        detector.update(st(5))  # stall 1
        detector.update(st(3))  # damage resets
        detector.update(st(3))  # stall 1
        detector.update(st(3))  # stall 2
        detector.update(st(3))  # stall 3
        detector.update(st(3))  # stall 4
        assert detector.update(st(3)) is WatchdogEvent.ENEMY_STALLED

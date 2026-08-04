"""
Autopilot framework for running Task-based bots.

Bridges the Task protocol with PlaySession by building WorldState
from env output and converting TaskResult actions into numpy arrays.
"""
from __future__ import annotations

from retro_harness.ram_state import GameState
from retro_harness.input_script import FrameAction
from retro_harness.actions import idle_action
from typing import Callable
from enum import Enum, auto
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from retro_harness.mission_control import MissionSnapshot
from retro_harness.protocol import Task, TaskStatus, TaskResult, WorldState, ActionResult


class BotRunner:
    """Wraps a Task to provide actions for PlaySession's bot interface.

    Callable as runner(obs, info) -> action_array or None, compatible
    with PlaySession's bot parameter.
    """

    def __init__(self, task, *, ram_schema=None, action_size=12):
        self.task = task
        self.ram_schema = ram_schema
        self.action_size = action_size
        self._frame = 0
        self._initialized = False

    def __call__(self, obs, info) -> np.ndarray | None:
        """Called by PlaySession each frame. Returns action or None."""
        world = self._build_world(obs, info)
        if not self._initialized:
            self.task.reset(world)
            self._initialized = True

        result = self.task.step(world)
        self._frame += 1

        if result.status in (TaskStatus.SUCCESS, TaskStatus.FAILURE):
            return None  # done, fall to human
        elif result.action is not None:
            return np.array(result.action.action, dtype=np.int8)
        else:
            return np.zeros(self.action_size, dtype=np.int8)  # idle

    def reset(self):
        """Reset the runner for a new episode."""
        self._frame = 0
        self._initialized = False

    def mission_status(self) -> MissionSnapshot:
        """Expose current task so PlaySession can show mission state."""
        current = getattr(self.task, "current_task", None) or self.task
        phase = getattr(current, "name", current.__class__.__name__)
        objective = f"frame={self._frame}"
        if hasattr(self.task, "current_task_index"):
            objective = f"task={self.task.current_task_index} frame={self._frame}"
        return MissionSnapshot(mission_id=getattr(self.task, "name", "bot"), phase=phase, objective=objective)

    def on_human_takeover(self) -> None:
        """Mission state stays hot while a human is driving."""
        return None

    def on_autopilot_resume(self) -> None:
        """Resume without resetting the task tree."""
        return None

    def _build_world(self, obs, info):
        ram = info.get("ram", np.array([], dtype=np.uint8))
        meta = self.ram_schema.read(ram) if self.ram_schema is not None else {}
        return WorldState(
            frame=self._frame, ram=ram, info=info, obs=obs, meta=meta,
        )


class TaskSequencer:
    """Run a sequence of Tasks in order. Implements the Task protocol.

    Each task runs to SUCCESS, then the next starts. If any task
    returns FAILURE, the sequencer fails.
    """

    name: str = "TaskSequencer"

    def __init__(self, tasks):
        self.tasks = list(tasks)
        self._idx = 0

    def reset(self, world):
        self._idx = 0
        if self.tasks:
            self.tasks[0].reset(world)

    def can_start(self, world):
        return bool(self.tasks) and self.tasks[0].can_start(world)

    def step(self, world) -> TaskResult:
        if self._idx >= len(self.tasks):
            return TaskResult(status=TaskStatus.SUCCESS)

        result = self.tasks[self._idx].step(world)

        if result.status == TaskStatus.SUCCESS:
            self._idx += 1
            if self._idx >= len(self.tasks):
                return TaskResult(status=TaskStatus.SUCCESS, action=result.action)
            self.tasks[self._idx].reset(world)
            return TaskResult(status=TaskStatus.RUNNING, action=result.action)

        return result

    @property
    def current_task_index(self) -> int:
        return self._idx

    @property
    def current_task(self) -> Task | None:
        if self._idx < len(self.tasks):
            return self.tasks[self._idx]
        return None


class TaskRepeater:
    """Repeat a task N times (or indefinitely if times=None).

    Implements the Task protocol.
    """

    name: str = "TaskRepeater"

    def __init__(self, task, *, times=None):
        self.task = task
        self.max_times = times
        self._count = 0

    def reset(self, world):
        self._count = 0
        self.task.reset(world)

    def can_start(self, world):
        return self.task.can_start(world)

    def step(self, world) -> TaskResult:
        result = self.task.step(world)

        if result.status == TaskStatus.SUCCESS:
            self._count += 1
            if self.max_times is not None and self._count >= self.max_times:
                return TaskResult(status=TaskStatus.SUCCESS, action=result.action)
            self.task.reset(world)
            return TaskResult(status=TaskStatus.RUNNING, action=result.action)

        return result


# -- Behavior trees -----------------------------------------------------------


class NodeStatus(Enum):
    """Behavior-tree tick result."""

    RUNNING = auto()
    SUCCESS = auto()
    FAILURE = auto()


@dataclass(frozen=True)
class TickResult:
    """Outcome of one behavior-tree tick."""

    status: NodeStatus
    action: FrameAction | None = None
    reason: str = ""


class BehaviorNode(ABC):
    """Base behavior-tree node."""

    name: str = "BehaviorNode"

    @abstractmethod
    def tick(self, state: GameState) -> TickResult:
        """Evaluate this node against the current game state."""


class Condition(BehaviorNode):
    """Succeeds when predicate(state) is true; otherwise fails."""

    def __init__(
        self,
        predicate: Callable[[GameState], bool],
        *,
        name: str = "Condition",
    ) -> None:
        self._predicate = predicate
        self.name = name

    def tick(self, state: GameState) -> TickResult:
        ok = self._predicate(state)
        return TickResult(
            status=NodeStatus.SUCCESS if ok else NodeStatus.FAILURE,
            reason=self.name,
        )


class ActionNode(BehaviorNode):
    """Emit a single FrameAction producer each tick while RUNNING."""

    def __init__(
        self,
        producer: Callable[[GameState], FrameAction],
        *,
        done: Callable[[GameState], bool] | None = None,
        name: str = "Action",
    ) -> None:
        self._producer = producer
        self._done = done
        self.name = name

    def tick(self, state: GameState) -> TickResult:
        if self._done is not None and self._done(state):
            return TickResult(status=NodeStatus.SUCCESS, reason=self.name)
        return TickResult(
            status=NodeStatus.RUNNING,
            action=self._producer(state),
            reason=self.name,
        )


class Sequence(BehaviorNode):
    """Run children in order; fail on first failure (reactive each tick)."""

    def __init__(self, children: list[BehaviorNode], *, name: str = "Sequence") -> None:
        self.children = list(children)
        self.name = name

    def tick(self, state: GameState) -> TickResult:
        for child in self.children:
            result = child.tick(state)
            if result.status is NodeStatus.RUNNING:
                return TickResult(
                    status=NodeStatus.RUNNING,
                    action=result.action,
                    reason=f"{self.name}:{result.reason}",
                )
            if result.status is NodeStatus.FAILURE:
                return TickResult(
                    status=NodeStatus.FAILURE,
                    action=result.action,
                    reason=f"{self.name}:{result.reason}",
                )
        return TickResult(status=NodeStatus.SUCCESS, reason=self.name)


class Selector(BehaviorNode):
    """Try children until one succeeds or runs; fail if all fail."""

    def __init__(self, children: list[BehaviorNode], *, name: str = "Selector") -> None:
        self.children = list(children)
        self.name = name

    def tick(self, state: GameState) -> TickResult:
        for child in self.children:
            result = child.tick(state)
            if result.status is not NodeStatus.FAILURE:
                return TickResult(
                    status=result.status,
                    action=result.action,
                    reason=f"{self.name}:{result.reason}",
                )
        return TickResult(
            status=NodeStatus.FAILURE,
            action=FrameAction(action=idle_action(), reason="selector_idle"),
            reason=self.name,
        )


# -- Stuck detection ----------------------------------------------------------


class WatchdogEvent(Enum):
    """Signals emitted when the agent appears stuck."""

    NONE = auto()
    POSITION_STALLED = auto()
    CAMERA_STALLED = auto()
    HEALTH_DRAINING = auto()
    ENEMY_STALLED = auto()


@dataclass
class StuckDetector:
    """Track progress signals and report stall conditions.

    Args:
        position_window: Frames without player movement before stall.
        camera_window: Frames without camera movement before stall.
        health_window: Frames of health decline without progress.
        enemy_window: Frames with living enemies and no health drop.
        move_epsilon: Minimum player delta counted as movement.
    """

    position_window: int = 180
    camera_window: int = 300
    health_window: int = 240
    enemy_window: int = 360
    move_epsilon: int = 1

    def __post_init__(self) -> None:
        self._last_x: int | None = None
        self._last_y: int | None = None
        self._last_cam_x: int | None = None
        self._pos_stall = 0
        self._cam_stall = 0
        self._health_stall = 0
        self._enemy_stall = 0
        self._last_health: int | None = None
        self._last_enemy_health_sum: int | None = None

    def reset(self) -> None:
        """Clear stall counters."""
        self.__post_init__()

    def update(self, state: GameState) -> WatchdogEvent:
        """Feed a frame of state; return the highest-priority stall event."""
        moved = False
        if self._last_x is not None and self._last_y is not None:
            dx = abs(state.player_x - self._last_x)
            dy = abs(state.player_y - self._last_y)
            moved = dx >= self.move_epsilon or dy >= self.move_epsilon
            self._pos_stall = 0 if moved else self._pos_stall + 1
        self._last_x = state.player_x
        self._last_y = state.player_y

        if self._last_cam_x is not None:
            cam_moved = abs(state.camera_x - self._last_cam_x) >= self.move_epsilon
            self._cam_stall = 0 if cam_moved or moved else self._cam_stall + 1
        self._last_cam_x = state.camera_x

        if self._last_health is not None and state.health < self._last_health:
            self._health_stall += 1
        else:
            self._health_stall = 0
        self._last_health = state.health

        enemy_sum = sum(e.health for e in state.living_enemies)
        if state.living_enemies:
            if (
                self._last_enemy_health_sum is not None
                and enemy_sum >= self._last_enemy_health_sum
            ):
                self._enemy_stall += 1
            else:
                self._enemy_stall = 0
        else:
            self._enemy_stall = 0
        self._last_enemy_health_sum = enemy_sum

        if self._pos_stall >= self.position_window:
            return WatchdogEvent.POSITION_STALLED
        if self._enemy_stall >= self.enemy_window:
            return WatchdogEvent.ENEMY_STALLED
        if self._health_stall >= self.health_window:
            return WatchdogEvent.HEALTH_DRAINING
        if self._cam_stall >= self.camera_window:
            return WatchdogEvent.CAMERA_STALLED
        return WatchdogEvent.NONE


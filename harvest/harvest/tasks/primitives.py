"""Shared building blocks for autonomous task implementations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from retro_harness.input_script import (
    press_button_sequence as _shared_press_button_sequence,
    repeat_action as _shared_repeat_action,
)

from harvest.core.ram_catalog import read_ram_value
from harvest.core.scene import SceneLocation, SceneMode, classify_scene_from_ram
from harvest.tasks.farm_clearer import make_action

ButtonName = str
QueuedActions = deque[np.ndarray]
TaskFactory = Callable[[], Task]


def button_action(button: Optional[ButtonName] = None, **buttons: bool) -> np.ndarray:
    if button in {None, "", "idle"} and not buttons:
        return make_action()
    if button not in {None, "", "idle"}:
        buttons[button] = True
    return make_action(**buttons)


def repeat_action(action: np.ndarray, frames: int) -> list[np.ndarray]:
    return _shared_repeat_action(action, frames, dtype=np.int32)


def press_button_sequence(
    button: ButtonName,
    *,
    face: Optional[str] = None,
    face_frames: int = 0,
    pre_press_settle_frames: int = 0,
    hold_frames: int = 1,
    settle_frames: int = 0,
    hold_face_with_button: bool = False,
) -> list[np.ndarray]:
    return _shared_press_button_sequence(
        button,
        face=face,
        face_frames=face_frames,
        pre_press_settle_frames=pre_press_settle_frames,
        hold_frames=hold_frames,
        settle_frames=settle_frames,
        hold_face_with_button=hold_face_with_button,
        dtype=np.int32,
    )


def press_a_sequence(
    face: Optional[str] = None,
    *,
    face_frames: int = 2,
    pre_press_settle_frames: int = 4,
    hold_frames: int = 25,
    settle_frames: int = 18,
    hold_face_with_a: bool = False,
) -> list[np.ndarray]:
    return press_button_sequence(
        "a",
        face=face,
        face_frames=face_frames if face else 0,
        pre_press_settle_frames=pre_press_settle_frames,
        hold_frames=hold_frames,
        settle_frames=settle_frames,
        hold_face_with_button=hold_face_with_a,
    )


def drain_action_queue(queue: QueuedActions, *, reason: str = "queued action") -> Optional[TaskResult]:
    if not queue:
        return None
    return TaskResult(
        status=TaskStatus.RUNNING,
        action=ActionResult(queue.popleft()),
        reason=reason,
    )


def idle_result(*, reason: str = "idle") -> TaskResult:
    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()), reason=reason)


def dismiss_dialogue_action(
    frame: int,
    *,
    buttons: Sequence[ButtonName] = ("a",),
    pulse_every: int = 2,
) -> np.ndarray:
    if pulse_every > 1 and frame % pulse_every != 0:
        return make_action()
    if not buttons:
        return make_action()
    pulse_index = frame // max(1, pulse_every)
    button = buttons[(pulse_index - 1) % len(buttons)]
    return button_action(button)


def dismiss_dialogue_result(
    frame: int,
    *,
    buttons: Sequence[ButtonName] = ("a",),
    pulse_every: int = 2,
    reason: str = "dialog",
) -> TaskResult:
    return TaskResult(
        status=TaskStatus.RUNNING,
        action=ActionResult(dismiss_dialogue_action(frame, buttons=buttons, pulse_every=pulse_every)),
        reason=reason,
    )


@dataclass(frozen=True)
class RamCondition:
    ram_field: str
    expected: Optional[int] = None
    predicate: Optional[Callable[[int], bool]] = None
    raw: bool = True
    description: str = ""

    def observe(self, ram: np.ndarray) -> int:
        return read_ram_value(ram, self.ram_field, raw=self.raw)

    def matches(self, ram: np.ndarray) -> bool:
        observed = self.observe(ram)
        if self.predicate is not None:
            return bool(self.predicate(observed))
        return observed == self.expected

    def expected_text(self) -> str:
        if self.description:
            return self.description
        if self.expected is not None:
            return f"{self.ram_field}={self.expected}"
        return f"{self.ram_field} predicate"


@dataclass
class WaitForRamConditionTask(Task):
    name: str = "wait_for_ram"
    condition: RamCondition = field(default_factory=lambda: RamCondition("input_lock", expected=1))
    stable_frames: int = 1
    timeout: int = 120

    _step_count: int = field(default=0, init=False)
    _stable_count: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._stable_count = 0

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        observed = self.condition.observe(world.ram)
        if self.condition.matches(world.ram):
            self._stable_count += 1
            if self._stable_count >= self.stable_frames:
                return TaskResult(
                    status=TaskStatus.SUCCESS,
                    reason=f"{self.condition.expected_text()} observed {observed}",
                )
        else:
            self._stable_count = 0

        if self._step_count > self.timeout:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"timed out waiting for {self.condition.expected_text()}, observed {observed}",
            )
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))


@dataclass
class WaitForSceneTask(Task):
    name: str = "wait_for_scene"
    expected_mode: Optional[SceneMode | str] = SceneMode.NORMAL
    expected_location: Optional[SceneLocation | str] = None
    stable_frames: int = 1
    timeout: int = 180
    dismiss_blocking_dialogue: bool = False

    _step_count: int = field(default=0, init=False)
    _stable_count: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._stable_count = 0

    def can_start(self, world: WorldState) -> bool:
        return True

    def _matches(self, scene_mode: SceneMode, scene_location: SceneLocation) -> bool:
        if self.expected_mode is not None and scene_mode.value != _enum_text(self.expected_mode):
            return False
        if self.expected_location is not None and scene_location.value != _enum_text(self.expected_location):
            return False
        return True

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        scene = classify_scene_from_ram(world.ram)
        if self._matches(scene.mode, scene.location):
            self._stable_count += 1
            if self._stable_count >= self.stable_frames:
                return TaskResult(status=TaskStatus.SUCCESS, reason=scene.summary())
        else:
            self._stable_count = 0

        if self._step_count > self.timeout:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"timed out waiting for scene, observed {scene.summary()}",
            )
        if self.dismiss_blocking_dialogue and scene.mode in {
            SceneMode.DIALOGUE,
            SceneMode.MENU,
            SceneMode.INPUT_LOCKED,
        }:
            return dismiss_dialogue_result(self._step_count)
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))


@dataclass
class PressAndVerifyTask(Task):
    name: str = "press_and_verify"
    sequence: Iterable[np.ndarray] = field(default_factory=tuple)
    condition: RamCondition = field(default_factory=lambda: RamCondition("input_lock", expected=1))
    stable_frames: int = 1
    timeout: int = 180

    _step_count: int = field(default=0, init=False)
    _stable_count: int = field(default=0, init=False)
    _queue: QueuedActions = field(default_factory=deque, init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._stable_count = 0
        self._queue = deque(np.array(action, dtype=np.int32) for action in self.sequence)

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        queued = drain_action_queue(self._queue)
        if queued is not None:
            return queued

        observed = self.condition.observe(world.ram)
        if self.condition.matches(world.ram):
            self._stable_count += 1
            if self._stable_count >= self.stable_frames:
                return TaskResult(
                    status=TaskStatus.SUCCESS,
                    reason=f"{self.condition.expected_text()} observed {observed}",
                )
        else:
            self._stable_count = 0

        if self._step_count > self.timeout:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"verify failed for {self.condition.expected_text()}, observed {observed}",
            )
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))


@dataclass
class TaskSequence(Task):
    """Run child tasks in order, resetting each task as it becomes active."""

    name: str = "task_sequence"
    tasks: Sequence[Task] = field(default_factory=tuple)
    idle_between_tasks: bool = True

    _index: int = field(default=0, init=False)
    _active_started: bool = field(default=False, init=False)

    def reset(self, world: WorldState) -> None:
        self._index = 0
        self._active_started = False

    def can_start(self, world: WorldState) -> bool:
        if not self.tasks:
            return True
        return self.tasks[0].can_start(world)

    @property
    def active_task_name(self) -> str:
        if self._index >= len(self.tasks):
            return "done"
        return getattr(self.tasks[self._index], "name", f"task_{self._index}")

    def _start_active_task(self, world: WorldState) -> Optional[TaskResult]:
        task = self.tasks[self._index]
        if not task.can_start(world):
            return TaskResult(
                status=TaskStatus.BLOCKED,
                reason=f"{getattr(task, 'name', self._index)} cannot start",
            )
        task.reset(world)
        self._active_started = True
        return None

    def step(self, world: WorldState) -> TaskResult:
        while self._index < len(self.tasks):
            if not self._active_started:
                blocked = self._start_active_task(world)
                if blocked is not None:
                    return blocked

            task = self.tasks[self._index]
            result = task.step(world)
            if result.status == TaskStatus.RUNNING:
                return result if result.action is not None else idle_result(reason=result.reason or task.name)
            if result.status in {TaskStatus.FAILURE, TaskStatus.BLOCKED}:
                return TaskResult(
                    status=result.status,
                    action=result.action,
                    reason=f"{getattr(task, 'name', self._index)}: {result.reason or result.status.value}",
                    checkpoint=result.checkpoint,
                    meta=result.meta,
                )

            self._index += 1
            self._active_started = False
            if self.idle_between_tasks and self._index < len(self.tasks):
                return idle_result(reason=f"{getattr(task, 'name', self._index - 1)} complete")

        return TaskResult(status=TaskStatus.SUCCESS, reason="sequence complete")


@dataclass
class RetryTask(Task):
    """Retry a task factory after failure or blocked results."""

    name: str = "retry"
    task_factory: TaskFactory = field(default_factory=lambda: (lambda: WaitForRamConditionTask()))
    max_attempts: int = 2
    retry_statuses: Sequence[TaskStatus] = (TaskStatus.FAILURE, TaskStatus.BLOCKED)
    settle_frames: int = 0

    _attempt: int = field(default=0, init=False)
    _settle_remaining: int = field(default=0, init=False)
    _task: Optional[Task] = field(default=None, init=False)

    def reset(self, world: WorldState) -> None:
        self._attempt = 0
        self._settle_remaining = 0
        self._task = None

    def can_start(self, world: WorldState) -> bool:
        return self.max_attempts > 0

    @property
    def attempt(self) -> int:
        return self._attempt

    def _start_attempt(self, world: WorldState) -> Optional[TaskResult]:
        if self._attempt >= self.max_attempts:
            return TaskResult(status=TaskStatus.FAILURE, reason=f"{self.name} exhausted attempts")
        self._attempt += 1
        self._task = self.task_factory()
        if not self._task.can_start(world):
            return self._handle_retryable_result(
                world,
                TaskResult(status=TaskStatus.BLOCKED, reason=f"{self._task.name} cannot start"),
            )
        self._task.reset(world)
        return None

    def _handle_retryable_result(self, world: WorldState, result: TaskResult) -> TaskResult:
        if result.status not in self.retry_statuses:
            return result
        if self._attempt >= self.max_attempts:
            return TaskResult(
                status=result.status,
                reason=f"{self.name} exhausted {self.max_attempts} attempts: {result.reason or result.status.value}",
                checkpoint=result.checkpoint,
                meta=result.meta,
            )
        self._task = None
        self._settle_remaining = max(0, self.settle_frames)
        return idle_result(reason=f"{self.name} retrying after {result.reason or result.status.value}")

    def step(self, world: WorldState) -> TaskResult:
        if self._settle_remaining > 0:
            self._settle_remaining -= 1
            return idle_result(reason=f"{self.name} settling before retry")

        if self._task is None:
            started = self._start_attempt(world)
            if started is not None:
                return started

        assert self._task is not None
        result = self._task.step(world)
        if result.status == TaskStatus.RUNNING:
            return result if result.action is not None else idle_result(reason=result.reason or self._task.name)
        if result.status in self.retry_statuses:
            return self._handle_retryable_result(world, result)
        return result


def _enum_text(value: SceneMode | SceneLocation | str) -> str:
    return value.value if isinstance(value, Enum) else str(value)


__all__ = [
    "RamCondition",
    "PressAndVerifyTask",
    "RetryTask",
    "TaskFactory",
    "TaskSequence",
    "WaitForRamConditionTask",
    "WaitForSceneTask",
    "button_action",
    "dismiss_dialogue_action",
    "dismiss_dialogue_result",
    "drain_action_queue",
    "idle_result",
    "press_a_sequence",
    "press_button_sequence",
    "repeat_action",
]

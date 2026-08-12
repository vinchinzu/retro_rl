"""Scene-level recovery task for autonomous planner failures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from harvest.core.scene import Scene, SceneLocation, SceneMode, classify_scene_from_ram
from harvest.tasks.nav import make_action
from harvest.tasks.primitives import dismiss_dialogue_result


TaskFactory = Callable[[], Task]


@dataclass
class RecoveryTask(Task):
    """Recover from transient scene states before retrying a planner phase."""

    name: str = "recovery"
    target_location: SceneLocation | str | None = SceneLocation.FARM
    stable_frames: int = 8
    timeout: int = 900
    cutscene_mash_limit: int = 240
    route_to_target_factory: Optional[TaskFactory] = None
    dismiss_buttons: Sequence[str] = ("b", "a")

    _step_count: int = field(default=0, init=False)
    _stable_count: int = field(default=0, init=False)
    _cutscene_mash_count: int = field(default=0, init=False)
    _route_task: Optional[Task] = field(default=None, init=False)
    _route_started_from: str = field(default="", init=False)
    _last_scene: str = field(default="", init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._stable_count = 0
        self._cutscene_mash_count = 0
        self._route_task = None
        self._route_started_from = ""
        self._last_scene = ""

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        scene = classify_scene_from_ram(world.ram)
        self._last_scene = scene.summary()

        if self._target_matches(scene):
            self._stable_count += 1
            self._cutscene_mash_count = 0
            if self._stable_count >= self.stable_frames:
                return TaskResult(status=TaskStatus.SUCCESS, reason=f"recovered at {scene.summary()}")
            return self._idle(f"stabilizing {scene.summary()}")

        self._stable_count = 0
        if self._step_count > self.timeout:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"recovery timeout, last scene {self._last_scene}",
            )

        if scene.mode in {SceneMode.UNKNOWN_TILEMAP, SceneMode.INVALID_COORDINATES, SceneMode.ENDING_CREDITS}:
            return TaskResult(status=TaskStatus.BLOCKED, reason=f"unrecoverable scene {scene.summary()}")

        if scene.needs_input_dismiss:
            if scene.mode == SceneMode.CUTSCENE_EVENT:
                self._cutscene_mash_count += 1
                if self._cutscene_mash_count > self.cutscene_mash_limit:
                    return TaskResult(
                        status=TaskStatus.BLOCKED,
                        reason=f"cutscene did not clear: {scene.summary()}",
                    )
            else:
                self._cutscene_mash_count = 0
            return dismiss_dialogue_result(
                self._step_count,
                buttons=self.dismiss_buttons,
                pulse_every=1,
                reason=f"recovering {scene.mode.value}",
            )

        self._cutscene_mash_count = 0
        if scene.mode in {SceneMode.MAP_TRANSITION, SceneMode.SLEEP_WAKE_TRANSITION}:
            return self._idle(f"waiting for {scene.summary()}")

        if scene.mode != SceneMode.NORMAL:
            return TaskResult(status=TaskStatus.BLOCKED, reason=f"unhandled scene {scene.summary()}")

        return self._route_to_target(world, scene)

    def _target_matches(self, scene: Scene) -> bool:
        if scene.mode != SceneMode.NORMAL:
            return False
        if self.target_location is None:
            return True
        return scene.location.value == _enum_text(self.target_location)

    def _route_to_target(self, world: WorldState, scene: Scene) -> TaskResult:
        if self.route_to_target_factory is None:
            return TaskResult(
                status=TaskStatus.BLOCKED,
                reason=f"normal scene is not target location: {scene.summary()}",
            )
        if self._route_task is None:
            self._route_task = self.route_to_target_factory()
            self._route_task.reset(world)
            self._route_started_from = scene.summary()

        result = self._route_task.step(world)
        if result.status == TaskStatus.RUNNING:
            if result.action is not None:
                return result
            return self._idle(result.reason or "recovery route running")

        if result.status == TaskStatus.SUCCESS:
            scene_after = classify_scene_from_ram(world.ram)
            if self._target_matches(scene_after):
                return TaskResult(
                    status=TaskStatus.SUCCESS,
                    reason=f"route recovered from {self._route_started_from}: {result.reason or scene_after.summary()}",
                )
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=(
                    "recovery route completed before target scene: "
                    f"{scene_after.summary()} ({result.reason or 'no route reason'})"
                ),
            )

        return TaskResult(
            status=result.status,
            reason=f"recovery route {result.status.value}: {result.reason or 'unknown'}",
        )

    def _idle(self, reason: str) -> TaskResult:
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()), reason=reason)


def _enum_text(value: SceneLocation | str) -> str:
    return value.value if isinstance(value, SceneLocation) else str(value)


__all__ = ["RecoveryTask"]

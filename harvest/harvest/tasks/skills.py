"""Composable skill tasks for hierarchical day-plan composition.

Domain tasks (coop, cow, harvest) should become thin composers of skills that
implement the same Task protocol. Skills keep ProgressSnapshot trees precise
for stall detection and make recording → autonomous extraction easier.

Prefer these over growing another 50–100 KB phase-machine file. See
``docs/PLANNING_STACK.md``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from retro_harness import Task, TaskResult, TaskStatus, WorldState

from harvest.core.task_progress import ProgressSnapshot, task_progress_snapshot
from harvest.planner.tasks.navigation import NavTask
from harvest.tasks.farm_clearer import Point
from harvest.tasks.primitives import (
    PressAndVerifyTask,
    QueuedActions,
    RamCondition,
    TaskSequence,
    WaitForRamConditionTask,
    drain_action_queue,
    press_a_sequence,
)

# Re-export composition primitives under the skills namespace.
SequenceSkill = TaskSequence
VerifyRamSkill = WaitForRamConditionTask
InteractSkill = PressAndVerifyTask


@dataclass
class NavSkill(Task):
    """Navigate to a pixel target via the shared viewport-aware NavTask.

    Thin skill wrapper so domain tasks can compose nav without owning a full
    multi-phase state machine.
    """

    name: str = "nav_skill"
    target_px: Tuple[int, int] = (0, 0)
    radius: int = 12
    timeout: int = 1800

    _nav: NavTask = field(init=False)

    def __post_init__(self) -> None:
        self._nav = NavTask(
            name=self.name,
            target_px=Point(self.target_px[0], self.target_px[1]),
            radius=self.radius,
            timeout=self.timeout,
        )

    def reset(self, world: WorldState) -> None:
        self._nav.reset(world)

    def can_start(self, world: WorldState) -> bool:
        return self._nav.can_start(world)

    def progress_snapshot(self) -> ProgressSnapshot:
        child = task_progress_snapshot(self._nav)
        return ProgressSnapshot(
            task_name=self.name,
            phase_text="navigate",
            step_count=getattr(self._nav, "_step_count", None),
            details=(
                ("target_px", self.target_px),
                ("radius", self.radius),
            ),
            child=child,
        )

    def step(self, world: WorldState) -> TaskResult:
        return self._nav.step(world)


@dataclass
class PressAInteractSkill(Task):
    """Face a direction, press A, optionally verify a RAM condition."""

    name: str = "press_a_interact"
    face: Optional[str] = None
    condition: Optional[RamCondition] = None
    face_frames: int = 2
    hold_frames: int = 25
    settle_frames: int = 18
    stable_frames: int = 1
    timeout: int = 180

    _inner: Optional[PressAndVerifyTask] = field(default=None, init=False)
    _queue: QueuedActions = field(default_factory=deque, init=False)
    _step_count: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        sequence = press_a_sequence(
            self.face,
            face_frames=self.face_frames if self.face else 0,
            hold_frames=self.hold_frames,
            settle_frames=self.settle_frames,
        )
        if self.condition is not None:
            self._inner = PressAndVerifyTask(
                name=self.name,
                sequence=sequence,
                condition=self.condition,
                stable_frames=self.stable_frames,
                timeout=self.timeout,
            )
            self._inner.reset(world)
            self._queue = deque()
        else:
            self._inner = None
            self._queue = deque(np.array(a, dtype=np.int32) for a in sequence)

    def can_start(self, world: WorldState) -> bool:
        return True

    def progress_snapshot(self) -> ProgressSnapshot:
        child = task_progress_snapshot(self._inner) if self._inner is not None else None
        return ProgressSnapshot(
            task_name=self.name,
            phase_text="interact",
            step_count=self._step_count,
            details=(("face", self.face or ""),),
            child=child,
        )

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        if self._inner is not None:
            return self._inner.step(world)

        queued = drain_action_queue(self._queue, reason=f"{self.name} press")
        if queued is not None:
            return queued
        return TaskResult(status=TaskStatus.SUCCESS, reason=f"{self.name} complete")


@dataclass
class SkillSequence(TaskSequence):
    """TaskSequence with an explicit skill-oriented name and progress tree."""

    name: str = "skill_sequence"

    def progress_snapshot(self) -> ProgressSnapshot:
        child_task = self.current_task
        child = task_progress_snapshot(child_task) if child_task is not None else None
        return ProgressSnapshot(
            task_name=self.name,
            phase_text=self.active_task_name,
            phase_index=self._index,
            details=(("skill_count", len(self.tasks)),),
            child=child,
        )


def sequence_skills(name: str, *skills: Task, idle_between: bool = True) -> SkillSequence:
    """Convenience constructor for skill composition."""
    return SkillSequence(name=name, tasks=tuple(skills), idle_between_tasks=idle_between)


# ── Domain skill factories ────────────────────────────────────────────
# These pin skill *boundaries* for composition. Production domain tasks
# (CoopChoresTask, HarvestTask, …) remain the live path until each skill
# is fully extracted + replay-covered. Prefer factories over growing mono
# phase machines — see docs/PLANNING_STACK.md.


def coop_nav_to_feed_bin_skill(*, timeout: int = 900) -> NavSkill:
    """Navigate to the coop feed-bin stand tile (2, 6) in pixel space."""
    from harvest.tasks.coop_task import FEED_BIN_STAND
    from harvest.tasks.farm_clearer import TILE_SIZE

    tx, ty = FEED_BIN_STAND
    return NavSkill(
        name="coop_nav_feed_bin",
        target_px=(tx * TILE_SIZE + 8, ty * TILE_SIZE + 8),
        radius=10,
        timeout=timeout,
    )


def coop_press_feed_skill(*, face: str = "left") -> PressAInteractSkill:
    """Press A at the feed bin (no RAM verify — feed flags vary by slot)."""
    return PressAInteractSkill(name="coop_press_feed", face=face)


def coop_nav_to_shipping_bin_skill(*, timeout: int = 900) -> NavSkill:
    """Navigate to the coop shipping-bin stand (egg ship path)."""
    from harvest.tasks.coop_task import SHIP_BIN_STAND
    from harvest.tasks.farm_clearer import TILE_SIZE

    tx, ty = SHIP_BIN_STAND
    return NavSkill(
        name="coop_nav_ship_bin",
        target_px=(tx * TILE_SIZE + 8, ty * TILE_SIZE + 8),
        radius=10,
        timeout=timeout,
    )


def coop_press_ship_skill(*, face: str = "up") -> PressAInteractSkill:
    """Press A at the coop shipping bin (egg disposition)."""
    return PressAInteractSkill(name="coop_press_ship", face=face)


def farm_nav_to_shipping_bin_skill(*, timeout: int = 1800) -> NavSkill:
    """Navigate toward the outdoor farm shipping bin landmark."""
    from harvest.maps.map_config import find_landmark

    hit = find_landmark("shipping_bin", tilemap_id=0x00)
    if hit is not None:
        _tilemap, lm = hit
        target = (int(lm.target_px[0]), int(lm.target_px[1]))
    else:
        # Fallback: farm landmark tile (62, 60) in pixel space.
        target = (62 * 16 + 8, 60 * 16 + 8)
    return NavSkill(
        name="farm_nav_ship_bin",
        target_px=target,
        radius=16,
        timeout=timeout,
    )


def farm_press_ship_skill(*, face: str = "up") -> PressAInteractSkill:
    """Press A at the farm shipping bin (crops/produce drop).

    Money does **not** post instantly — shipping credits at 5pm. Verify with
    pre/post-5pm saves, not an immediate money delta.
    """
    return PressAInteractSkill(name="farm_press_ship", face=face)


def talk_press_skill(
    *,
    name: str = "talk_press",
    face: str | None = "up",
    hold_frames: int = 25,
) -> PressAInteractSkill:
    """Generic face+A talk/interact (D1 town bits, NPC gifts, shop counters)."""
    return PressAInteractSkill(
        name=name,
        face=face,
        hold_frames=hold_frames,
    )


__all__ = [
    "InteractSkill",
    "NavSkill",
    "PressAInteractSkill",
    "SequenceSkill",
    "SkillSequence",
    "VerifyRamSkill",
    "coop_nav_to_feed_bin_skill",
    "coop_nav_to_shipping_bin_skill",
    "coop_press_feed_skill",
    "coop_press_ship_skill",
    "farm_nav_to_shipping_bin_skill",
    "farm_press_ship_skill",
    "sequence_skills",
    "talk_press_skill",
]

"""Composable skill tasks for hierarchical day-plan composition.

Domain tasks (coop, cow, harvest) should become thin composers of skills that
implement the same Task protocol. Skills keep ProgressSnapshot trees precise
for stall detection and make recording → autonomous extraction easier.

**Production path (wired):** ``CoopChoresTask`` feed_nav + ship_nav (far
approach) step ``coop_nav_to_feed_bin_skill`` / ``coop_nav_to_shipping_bin_skill``
with a host ``navigate`` callable so specialized coop routing is preserved.

Prefer these over growing another 50–100 KB phase-machine file. See
``docs/PLANNING_STACK.md``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Union

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from harvest.core.task_progress import ProgressSnapshot, task_progress_snapshot
from harvest.core.tile_catalog import ADDR_TILEMAP
from harvest.planner.tasks.navigation import NavTask
from harvest.tasks.nav import Point
from harvest.tasks.primitives import (
    PressAndVerifyTask,
    QueuedActions,
    RamCondition,
    TaskSequence,
    WaitForRamConditionTask,
    drain_action_queue,
    press_a_sequence,
)

# Host navigate: return a button array while moving, None when arrived.
NavigateFn = Callable[[WorldState], Optional[np.ndarray]]

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
    soft_radius: Optional[int] = None
    require_tilemap: Optional[int] = None

    _nav: NavTask = field(init=False)

    def __post_init__(self) -> None:
        self._nav = NavTask(
            name=self.name,
            target_px=Point(self.target_px[0], self.target_px[1]),
            radius=self.radius,
            soft_radius=self.soft_radius,
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
        if self.require_tilemap is not None:
            tilemap = (
                int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else -1
            )
            if tilemap != int(self.require_tilemap):
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=(
                        f"{self.name} left map "
                        f"0x{int(self.require_tilemap):02X} → 0x{tilemap:02X}"
                    ),
                )
        return self._nav.step(world)


@dataclass
class NavigateUntilArrivedSkill(Task):
    """Host-backed nav skill: call ``navigate(world)`` until it returns None.

    Used when a domain task owns specialized pathfinding (chicken blockers,
    false-open columns, left-top aisle routes) but still wants the skill
    protocol for progress trees and composition. Factories accept an optional
    ``navigate`` callable and return this instead of generic ``NavSkill``.
    """

    name: str = "navigate_until_arrived"
    navigate: Optional[NavigateFn] = None
    timeout: int = 900

    _step_count: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0

    def can_start(self, world: WorldState) -> bool:
        return self.navigate is not None

    def progress_snapshot(self) -> ProgressSnapshot:
        return ProgressSnapshot(
            task_name=self.name,
            phase_text="navigate",
            step_count=self._step_count,
            details=(("host_navigate", self.navigate is not None),),
        )

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        if self.navigate is None:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"{self.name} missing navigate",
            )
        if self._step_count > self.timeout:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"{self.name} timeout",
            )
        action = self.navigate(world)
        if action is not None:
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(action),
                reason=f"{self.name} moving",
            )
        return TaskResult(status=TaskStatus.SUCCESS, reason=f"{self.name} arrived")


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
# Pin skill *boundaries* for composition. Coop feed_nav / ship_nav far
# approach already call these with a host ``navigate`` (specialized routing).
# Generic ``NavSkill`` (no navigate) remains for open-map / unit targets.
# Prefer factories over growing mono phase machines — see PLANNING_STACK.


def coop_nav_to_feed_bin_skill(
    *,
    timeout: int = 900,
    navigate: Optional[NavigateFn] = None,
) -> Union[NavSkill, NavigateUntilArrivedSkill]:
    """Navigate to the coop feed-bin stand tile (2, 6) in pixel space.

    Pass ``navigate`` from ``CoopChoresTask`` to keep chicken/false-open routing.
    """
    if navigate is not None:
        return NavigateUntilArrivedSkill(
            name="coop_nav_feed_bin",
            navigate=navigate,
            timeout=timeout,
        )

    from harvest.tasks.coop_task import FEED_BIN_STAND
    from harvest.tasks.nav import TILE_SIZE

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


def coop_nav_to_shipping_bin_skill(
    *,
    timeout: int = 900,
    navigate: Optional[NavigateFn] = None,
) -> Union[NavSkill, NavigateUntilArrivedSkill]:
    """Navigate to the coop shipping-bin stand (egg ship path).

    Pass ``navigate`` from ``CoopChoresTask`` for lane/corner routing; pixel
    slide + press remain on the host after this skill reports arrived.
    """
    if navigate is not None:
        return NavigateUntilArrivedSkill(
            name="coop_nav_ship_bin",
            navigate=navigate,
            timeout=timeout,
        )

    from harvest.tasks.coop_task import SHIP_BIN_STAND
    from harvest.tasks.nav import TILE_SIZE

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


def farm_nav_to_pond_refill_skill(*, timeout: int = 2400) -> NavSkill:
    """Navigate to the primary main-pond F0 refill stand (map_config corridor)."""
    from harvest.maps.map_config import farm_pond_refill_primary_stand
    from harvest.tasks.nav import TILE_SIZE

    stand, _face = farm_pond_refill_primary_stand()
    tx, ty = stand
    return NavSkill(
        name="farm_nav_pond_refill",
        target_px=(tx * TILE_SIZE + 8, ty * TILE_SIZE + 8),
        radius=10,
        timeout=timeout,
    )


def farm_pond_refill_face() -> str:
    """Face direction for the primary pond refill stand."""
    from harvest.maps.map_config import farm_pond_refill_primary_stand

    _stand, face = farm_pond_refill_primary_stand()
    return face


def farm_fence_jump_toss_skill(*, timeout: int = 300):
    """Toss toward RAM-confirmed open ground; escape only when boxed in."""
    from harvest.tasks.farm_toss import FenceJumpTossSkill

    return FenceJumpTossSkill(timeout=timeout)


def farm_select_carry_skill(tool_id: int, *, timeout: int = 90):
    """X-swap until ``tool_id`` is selected."""
    from harvest.tasks.crop_skills import SelectCarrySkill

    return SelectCarrySkill(
        name=f"select_carry_0x{int(tool_id):02X}",
        wanted=int(tool_id),
        timeout=timeout,
    )


def farm_nav_pocket_plant_skill(*, timeout: int = 4000) -> NavSkill:
    """Stand on the tape plant notch (13, 28)."""
    from harvest.maps.farm_pond import WEST_POCKET_PLANT_CENTER
    from harvest.tasks.nav import TILE_SIZE

    tx, ty = WEST_POCKET_PLANT_CENTER
    return NavSkill(
        name="nav_pocket_plant",
        target_px=(tx * TILE_SIZE + 8, ty * TILE_SIZE + 8),
        radius=8,
        timeout=timeout,
        require_tilemap=0x00,
    )


def farm_nav_pocket_hoe_stand_skill(*, timeout: int = 9000) -> NavSkill:
    """Stand west of (13,29) (fence-lip (13,30) is not a hoe stand).

    Same timeout/radius class as NAV_CROP: this hop often starts at the shed
    outdoor door after ENSURE_CROP_SEEDS. Tight radius + short timeout walked
    UP into the shed when BFS was sealed.
    """
    from harvest.maps.farm_pond import WEST_POCKET_PLANT_CENTER
    from harvest.tasks.crop_skills import hoe_stand_px

    # Approach from the west so the last walk is RIGHT and RAM facing is
    # already 2 when the first hoe (13,29) starts. Fence-lip (13,30) is not
    # a stand.
    cx, cy = WEST_POCKET_PLANT_CENTER
    stand = (cx - 2, cy + 1)
    px, py = hoe_stand_px(stand, "right")
    return NavSkill(
        name="nav_pocket_hoe_stand",
        target_px=(px, py),
        radius=6,
        soft_radius=6,
        timeout=timeout,
        require_tilemap=0x00,
    )


def farm_hoe_tile_skill(*, timeout: int = 240):
    from harvest.maps.farm_pond import WEST_POCKET_PLANT_CENTER
    from harvest.tasks.crop_skills import hoe_until_tilled_skill

    return hoe_until_tilled_skill(
        target_tile=WEST_POCKET_PLANT_CENTER,
        face="up",
        timeout=timeout,
    )


def farm_plant_tile_skill(*, seed_type: str = "potato", timeout: int = 240):
    from harvest.maps.farm_pond import WEST_POCKET_PLANT_CENTER
    from harvest.tasks.crop_skills import plant_until_crop_skill

    return plant_until_crop_skill(
        seed_type=seed_type,
        target_tile=WEST_POCKET_PLANT_CENTER,
        timeout=timeout,
    )


def farm_water_one_skill(*, timeout: int = 240):
    from harvest.tasks.crop_skills import water_until_wet_skill

    return water_until_wet_skill(timeout=timeout)


def farm_pocket_plant_skill(
    *,
    seed_type: str = "potato",
    include_water: bool = False,
    include_plant: bool = True,
    timeout: int = 4000,
):
    """Reactive 3x3 hoe ring → plant from the untilled notch. No tape replay.

    Carry must already hold hoe+seeds (establish pass). Water is a later
    can-pass unless ``include_water`` and the can is in the pair.
    ``include_plant=False`` is the hoe-until-5pm tune: till the ring only.
    """
    from harvest.core.carry import seed_item_id
    from harvest.core.tile_catalog import Tool
    from harvest.maps.farm_pond import WEST_POCKET_PLANT_CENTER
    from harvest.tasks.crop_skills import (
        plant_until_plot_skill,
        pocket_hoe_ring_skills,
    )

    skills: list = [
        farm_fence_jump_toss_skill(),
        # Walk into the pocket before X-swap. Doing the swap at the shed
        # door (post-ENSURE) times out while input_lock is still settling.
        # NavTask leaves (26,30) south then west — do not hoe the notch.
        farm_nav_pocket_hoe_stand_skill(),
        farm_select_carry_skill(int(Tool.HOE)),
        *pocket_hoe_ring_skills(WEST_POCKET_PLANT_CENTER),
    ]
    if include_plant:
        skills.extend(
            [
                farm_nav_pocket_plant_skill(),
                farm_select_carry_skill(seed_item_id(seed_type)),
                plant_until_plot_skill(seed_type=seed_type),
            ]
        )
    if include_water:
        skills.append(farm_pocket_water_skill())
    return sequence_skills("pocket_plant_plot", *skills, idle_between=True)


def farm_pocket_water_skill(*, timeout: int = 4000):
    """Reactive 8-ring water from the untilled notch. No tape replay.

    Can pass: select the watering can, stand on (13,28) for the cardinals,
    corners from right-middle / left-middle. Does not water the notch.
    """
    from harvest.core.tile_catalog import Tool
    from harvest.maps.farm_pond import WEST_POCKET_PLANT_CENTER
    from harvest.tasks.crop_skills import pocket_water_ring_skills

    skills: list = [
        farm_fence_jump_toss_skill(),
        farm_nav_pocket_plant_skill(),
        farm_select_carry_skill(int(Tool.WATERING_CAN)),
        *pocket_water_ring_skills(WEST_POCKET_PLANT_CENTER),
    ]
    return sequence_skills("pocket_water_ring", *skills, idle_between=True)


__all__ = [
    "InteractSkill",
    "NavSkill",
    "NavigateFn",
    "NavigateUntilArrivedSkill",
    "PressAInteractSkill",
    "SequenceSkill",
    "SkillSequence",
    "VerifyRamSkill",
    "coop_nav_to_feed_bin_skill",
    "coop_nav_to_shipping_bin_skill",
    "coop_press_feed_skill",
    "coop_press_ship_skill",
    "farm_nav_to_pond_refill_skill",
    "farm_fence_jump_toss_skill",
    "farm_hoe_tile_skill",
    "farm_nav_pocket_hoe_stand_skill",
    "farm_nav_pocket_plant_skill",
    "farm_nav_to_shipping_bin_skill",
    "farm_plant_tile_skill",
    "farm_pocket_plant_skill",
    "farm_pocket_water_skill",
    "farm_pond_refill_face",
    "farm_press_ship_skill",
    "farm_select_carry_skill",
    "farm_water_one_skill",
    "sequence_skills",
    "talk_press_skill",
]

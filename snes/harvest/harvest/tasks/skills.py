from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Union

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from harvest.core.task_progress import ProgressSnapshot, task_progress_snapshot
from harvest.core.tile_catalog import ADDR_TILEMAP
from harvest.planner.day_plan_status import tilemaps_match
from harvest.planner.tasks.navigation import NavTask
from harvest.tasks.nav import Point, make_action
from harvest.tasks.primitives import (
    PressAndVerifyTask,
    QueuedActions,
    RamCondition,
    TaskSequence,
    WaitForRamConditionTask,
    drain_action_queue,
    press_a_sequence,
)

NavigateFn = Callable[[WorldState], Optional[np.ndarray]]

SequenceSkill = TaskSequence
VerifyRamSkill = WaitForRamConditionTask
InteractSkill = PressAndVerifyTask

@dataclass
class NavSkill(Task):

    name: str = "nav_skill"
    target_px: Tuple[int, int] = (0, 0)
    radius: int = 12
    timeout: int = 1800
    soft_radius: Optional[int] = None
    water_stand: bool = False
    require_tilemap: Optional[int] = None
    _nav: NavTask = field(init=False)
    def __post_init__(self) -> None:
        self._nav = NavTask(
            name=self.name,
            target_px=Point(self.target_px[0], self.target_px[1]),
            radius=self.radius,
            soft_radius=self.soft_radius,
            timeout=self.timeout,
            water_stand=self.water_stand or "water" in self.name.lower(),
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
            details=( ("target_px", self.target_px), ("radius", self.radius), ),
            child=child,
        )
    def step(self, world: WorldState) -> TaskResult:
        if self.require_tilemap is not None:
            tilemap = ( int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else -1 )
            if not tilemaps_match(tilemap, int(self.require_tilemap)):
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=( f"{self.name} left map " f"0x{int(self.require_tilemap):02X} → 0x{tilemap:02X}" ),
                )
        return self._nav.step(world)

@dataclass
class NavigateUntilArrivedSkill(Task):

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
            return TaskResult( status=TaskStatus.FAILURE, reason=f"{self.name} missing navigate", )
        if self._step_count > self.timeout:
            return TaskResult( status=TaskStatus.FAILURE, reason=f"{self.name} timeout", )
        action = self.navigate(world)
        if action is not None:
            return TaskResult( status=TaskStatus.RUNNING, action=ActionResult(action), reason=f"{self.name} moving", )
        return TaskResult(status=TaskStatus.SUCCESS, reason=f"{self.name} arrived")

@dataclass
class PressAInteractSkill(Task):

    name: str = "press_a_interact"
    face: Optional[str] = None
    condition: Optional[RamCondition] = None
    face_frames: int = 2
    pre_press_settle_frames: int = 0
    hold_frames: int = 25
    settle_frames: int = 18
    stable_frames: int = 1
    hold_face_with_a: bool = False
    timeout: int = 180
    _inner: Optional[PressAndVerifyTask] = field(default=None, init=False)
    _queue: QueuedActions = field(default_factory=deque, init=False)
    _step_count: int = field(default=0, init=False)
    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        sequence = press_a_sequence(
            self.face,
            face_frames=self.face_frames if self.face else 0,
            pre_press_settle_frames=self.pre_press_settle_frames,
            hold_frames=self.hold_frames,
            settle_frames=self.settle_frames,
            hold_face_with_a=self.hold_face_with_a,
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

@dataclass
class BestEffortSkill(Task):

    name: str = "best_effort"
    tasks: Tuple[Task, ...] = ()
    skipped: bool = field(default=False, init=False)
    skip_reason: str = field(default="", init=False)
    _inner: Optional[SkillSequence] = field(default=None, init=False)
    def reset(self, world: WorldState) -> None:
        self.skipped = False
        self.skip_reason = ""
        self._inner = SkillSequence(name=self.name, tasks=tuple(self.tasks), idle_between_tasks=True)
        self._inner.reset(world)
    def can_start(self, world: WorldState) -> bool:
        return True
    @property
    def current_task(self) -> Optional[Task]:
        return self._inner.current_task if self._inner is not None else None
    def progress_snapshot(self) -> ProgressSnapshot:
        child = task_progress_snapshot(self._inner) if self._inner is not None else None
        return ProgressSnapshot(task_name=self.name, phase_text="best_effort", child=child)
    def step(self, world: WorldState) -> TaskResult:
        if self._inner is None:
            self.reset(world)
        result = self._inner.step(world)
        if result.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
            self.skipped = True
            self.skip_reason = result.reason or ""
            print(f"[SKILL] {self.name} skipped: {self.skip_reason}")
            return TaskResult( status=TaskStatus.SUCCESS, action=result.action, reason=f"{self.name} skipped: {self.skip_reason}", )
        return result

def sequence_skills(name: str, *skills: Task, idle_between: bool = True) -> SkillSequence:
    return SkillSequence(name=name, tasks=tuple(skills), idle_between_tasks=idle_between)

def coop_nav_to_feed_bin_skill(
    *,
    timeout: int = 900,
    navigate: Optional[NavigateFn] = None,
) -> Union[NavSkill, NavigateUntilArrivedSkill]:
    if navigate is not None:
        return NavigateUntilArrivedSkill( name="coop_nav_feed_bin", navigate=navigate, timeout=timeout, )
    from harvest.tasks.coop_task import FEED_BIN_STAND
    from harvest.tasks.nav import TILE_SIZE
    tx, ty = FEED_BIN_STAND
    return NavSkill( name="coop_nav_feed_bin", target_px=(tx * TILE_SIZE + 8, ty * TILE_SIZE + 8), radius=10, timeout=timeout, )

def coop_press_feed_skill(*, face: str = "left") -> PressAInteractSkill:
    return PressAInteractSkill(name="coop_press_feed", face=face)

def coop_feed_place_sequence(face: str = "up") -> list[np.ndarray]:
    return [
        *(make_action(**{face: True}) for _ in range(4)),
        *(make_action(**{face: True, "a": True}) for _ in range(8)),
        *(make_action(a=True) for _ in range(4)),
        *(make_action(down=True) for _ in range(12)),
        *(make_action() for _ in range(8)),
    ]

@dataclass
class ButtonSequenceSkill(Task):

    name: str = "button_sequence"
    sequence: list[np.ndarray] = field(default_factory=list)
    timeout: int = 180
    _queue: QueuedActions = field(default_factory=deque, init=False)
    _step_count: int = field(default=0, init=False)
    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._queue = deque(np.array(a, dtype=np.int32, copy=True) for a in self.sequence)
    def can_start(self, world: WorldState) -> bool:
        return bool(self.sequence)
    def progress_snapshot(self) -> ProgressSnapshot:
        return ProgressSnapshot(
            task_name=self.name,
            phase_text="press",
            step_count=self._step_count,
            details=(("queue_len", len(self._queue)),),
        )
    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        if self._step_count > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason=f"{self.name} timeout")
        queued = drain_action_queue(self._queue, reason=f"{self.name} step")
        if queued is not None:
            return queued
        return TaskResult(status=TaskStatus.SUCCESS, reason=f"{self.name} complete")

def coop_press_feed_place_skill(*, face: str = "up") -> ButtonSequenceSkill:
    return ButtonSequenceSkill( name="coop_press_feed_place", sequence=coop_feed_place_sequence(face), )

def coop_nav_to_feed_spot_skill(
    *,
    target_stand: Optional[Tuple[int, int]] = None,
    target_px: Optional[Tuple[int, int]] = None,
    timeout: int = 900,
    navigate: Optional[NavigateFn] = None,
) -> Union[NavSkill, NavigateUntilArrivedSkill]:
    if navigate is not None:
        return NavigateUntilArrivedSkill( name="coop_nav_feed_spot", navigate=navigate, timeout=timeout, )
    from harvest.tasks.nav import TILE_SIZE
    if target_px is not None:
        px = target_px
    elif target_stand is not None:
        px = (target_stand[0] * TILE_SIZE + 8, target_stand[1] * TILE_SIZE + 8)
    else:
        px = (2 * TILE_SIZE + 8, 3 * TILE_SIZE + 8)
    return NavSkill( name="coop_nav_feed_spot", target_px=px, radius=6, timeout=timeout, )

def coop_press_egg_pickup_skill(*, face: str = "left") -> PressAInteractSkill:
    return PressAInteractSkill(
        name="coop_press_egg_pickup",
        face=face,
        face_frames=4,
        hold_frames=28,
        settle_frames=30,
        hold_face_with_a=False,
    )

def coop_nav_to_egg_skill(
    *,
    target_stand: Optional[Tuple[int, int]] = None,
    target_px: Optional[Tuple[int, int]] = None,
    timeout: int = 900,
    navigate: Optional[NavigateFn] = None,
) -> Union[NavSkill, NavigateUntilArrivedSkill]:
    if navigate is not None:
        return NavigateUntilArrivedSkill( name="coop_nav_egg", navigate=navigate, timeout=timeout, )
    from harvest.tasks.coop_layout import EGG_PICKUP_STAND
    from harvest.tasks.nav import TILE_SIZE
    if target_px is not None:
        px = target_px
    elif target_stand is not None:
        px = (target_stand[0] * TILE_SIZE + 8, target_stand[1] * TILE_SIZE + 8)
    else:
        px = (EGG_PICKUP_STAND[0] * TILE_SIZE + 8, EGG_PICKUP_STAND[1] * TILE_SIZE + 8)
    return NavSkill( name="coop_nav_egg", target_px=px, radius=6, timeout=timeout, )

def coop_nav_to_incubator_skill(
    *,
    timeout: int = 900,
    navigate: Optional[NavigateFn] = None,
) -> Union[NavSkill, NavigateUntilArrivedSkill]:
    if navigate is not None:
        return NavigateUntilArrivedSkill( name="coop_nav_incubator", navigate=navigate, timeout=timeout, )
    from harvest.tasks.coop_layout import INCUBATOR_STAND
    from harvest.tasks.nav import TILE_SIZE
    tx, ty = INCUBATOR_STAND
    return NavSkill( name="coop_nav_incubator", target_px=(tx * TILE_SIZE + 8, ty * TILE_SIZE + 8), radius=10, timeout=timeout, )

def coop_press_incubator_skill(*, face: str = "right") -> PressAInteractSkill:
    return PressAInteractSkill(
        name="coop_press_incubator",
        face=face,
        face_frames=2,
        hold_frames=20,
        settle_frames=24,
        hold_face_with_a=False,
    )

def coop_nav_to_exit_prep_skill(
    *,
    timeout: int = 900,
    navigate: Optional[NavigateFn] = None,
) -> Union[NavSkill, NavigateUntilArrivedSkill]:
    if navigate is not None:
        return NavigateUntilArrivedSkill( name="coop_nav_exit_prep", navigate=navigate, timeout=timeout, )
    from harvest.tasks.coop_layout import EXIT_PREP_STAND
    from harvest.tasks.nav import TILE_SIZE
    tx, ty = EXIT_PREP_STAND
    return NavSkill( name="coop_nav_exit_prep", target_px=(tx * TILE_SIZE + 8, ty * TILE_SIZE + 8), radius=10, timeout=timeout, )

def coop_nav_to_shipping_bin_skill(
    *,
    timeout: int = 900,
    navigate: Optional[NavigateFn] = None,
) -> Union[NavSkill, NavigateUntilArrivedSkill]:
    if navigate is not None:
        return NavigateUntilArrivedSkill( name="coop_nav_ship_bin", navigate=navigate, timeout=timeout, )
    from harvest.tasks.coop_task import SHIP_BIN_STAND
    from harvest.tasks.nav import TILE_SIZE
    tx, ty = SHIP_BIN_STAND
    return NavSkill( name="coop_nav_ship_bin", target_px=(tx * TILE_SIZE + 8, ty * TILE_SIZE + 8), radius=10, timeout=timeout, )

def coop_press_ship_skill(*, face: str = "up") -> PressAInteractSkill:
    return PressAInteractSkill(name="coop_press_ship", face=face)

def farm_nav_to_shipping_bin_skill(*, timeout: int = 1800) -> NavSkill:
    from harvest.maps.map_config import find_landmark
    hit = find_landmark("shipping_bin", tilemap_id=0x00)
    if hit is not None:
        _tilemap, lm = hit
        target = (int(lm.target_px[0]), int(lm.target_px[1]))
    else:
        target = (62 * 16 + 8, 60 * 16 + 8)
    return NavSkill( name="farm_nav_ship_bin", target_px=target, radius=16, timeout=timeout, )

def farm_press_ship_skill(*, face: str = "up") -> PressAInteractSkill:
    return PressAInteractSkill(name="farm_press_ship", face=face)

def talk_press_skill( *, name: str = "talk_press", face: str | None = "up", hold_frames: int = 25, ) -> PressAInteractSkill:
    return PressAInteractSkill( name=name, face=face, hold_frames=hold_frames, )

def farm_nav_to_pond_refill_skill(*, timeout: int = 2400) -> NavSkill:
    from harvest.maps.map_config import farm_pond_refill_primary_stand
    from harvest.tasks.nav import TILE_SIZE
    stand, _face = farm_pond_refill_primary_stand()
    tx, ty = stand
    return NavSkill( name="farm_nav_pond_refill", target_px=(tx * TILE_SIZE + 8, ty * TILE_SIZE + 8), radius=10, timeout=timeout, )

def farm_pond_refill_face() -> str:
    from harvest.maps.map_config import farm_pond_refill_primary_stand
    _stand, face = farm_pond_refill_primary_stand()
    return face

def farm_fence_jump_toss_skill(*, timeout: int = 300):
    from harvest.tasks.farm_toss import FenceJumpTossSkill
    return FenceJumpTossSkill(timeout=timeout)

def farm_select_carry_skill(tool_id: int, *, timeout: int = 90):
    from harvest.tasks.crop_skills import SelectCarrySkill
    return SelectCarrySkill( name=f"select_carry_0x{int(tool_id):02X}", wanted=int(tool_id), timeout=timeout, )

def farm_nav_pocket_plant_skill(*, center=None, timeout: int = 4000, radius: int = 5) -> NavSkill:
    from harvest.maps.farm_pond import WEST_POCKET_PLANT_CENTER
    from harvest.tasks.nav import TILE_SIZE
    tx, ty = center or WEST_POCKET_PLANT_CENTER
    return NavSkill(
        name="nav_pocket_plant",
        target_px=(tx * TILE_SIZE + 8, ty * TILE_SIZE + 8),
        radius=radius,
        soft_radius=radius,
        timeout=timeout,
        require_tilemap=0x00,
    )

def farm_nav_pocket_hoe_stand_skill(*, center=None, ram=None, timeout: int = 9000) -> NavSkill:
    from harvest.maps.farm_pond import (
        SECOND_POCKET_HOE_STAND,
        SECOND_POCKET_PLANT_CENTER,
        WEST_POCKET_PLANT_CENTER,
    )
    from harvest.tasks.crop_skills import hoe_stand_px, remap_pocket_hoe_stand
    cx, cy = center or WEST_POCKET_PLANT_CENTER
    if (cx, cy) == SECOND_POCKET_PLANT_CENTER:
        stand, face = SECOND_POCKET_HOE_STAND
    else:
        stand = (cx - 2, cy + 1)
        stand, face = remap_pocket_hoe_stand((cx, cy), (cx - 1, cy + 1), stand, "right", ram) if ram is not None else (stand, "right")
    px, py = hoe_stand_px(stand, face)
    return NavSkill( name="nav_pocket_hoe_stand", target_px=(px, py), radius=6, soft_radius=6, timeout=timeout, require_tilemap=0x00, )

def farm_hoe_tile_skill(*, center=None, timeout: int = 240):
    from harvest.maps.farm_pond import WEST_POCKET_PLANT_CENTER
    from harvest.tasks.crop_skills import hoe_until_tilled_skill
    return hoe_until_tilled_skill( target_tile=center or WEST_POCKET_PLANT_CENTER, face="up", timeout=timeout, )

def farm_plant_tile_skill(*, seed_type: str = "potato", center=None, timeout: int = 240):
    from harvest.maps.farm_pond import WEST_POCKET_PLANT_CENTER
    from harvest.tasks.crop_skills import plant_until_crop_skill
    return plant_until_crop_skill( seed_type=seed_type, target_tile=center or WEST_POCKET_PLANT_CENTER, timeout=timeout, )

def farm_water_one_skill(*, timeout: int = 240):
    from harvest.tasks.crop_skills import water_until_wet_skill
    return water_until_wet_skill(timeout=timeout)

def farm_pocket_plant_skill(
    *,
    seed_type: str = "potato",
    layout_name: str = "eight_tile_ring",
    center=None,
    ram=None,
    include_water: bool = False,
    include_plant: bool = True,
    timeout: int = 4000,
):
    from harvest.core.carry import seed_item_id
    from harvest.core.tile_catalog import Tool
    from harvest.maps.farm_pond import WEST_POCKET_PLANT_CENTER
    from harvest.tasks.crop_skills import ( plant_until_plot_skill, pocket_hoe_ring_skills, )
    center = center or WEST_POCKET_PLANT_CENTER
    from harvest.planner.crop_planner import CROP_LAYOUTS
    layout = CROP_LAYOUTS[layout_name]
    opening = next((tile for tile in layout.access_offsets if tile != (0, 0)), None)
    if opening is not None and include_water:
        raise ValueError("side-notch watering uses CropWaterTask water mode")
    skills: list = [
        farm_fence_jump_toss_skill(),
        farm_nav_pocket_hoe_stand_skill(center=center, ram=ram),
        farm_select_carry_skill(int(Tool.HOE)),
        *pocket_hoe_ring_skills(center, ram=ram, opening=opening, plant_center=(0, 0) in layout.crop_offsets),
    ]
    if ram is not None:
        from harvest.tasks.farm_clearer import FarmClearer
        from harvest.core.tile_catalog import DebrisType
        cx, cy = center
        # A bare ring can still have a fence on the initial hoe stand or a
        # weed in its center. Summer disables the whole-field clear. The walk
        # in is a plain nav (any ring, not just the west notch) and the clear
        # is a ``type_clear`` box: the legacy pocket approach only knows the
        # west notch and looped forever for every other centre.
        skills.insert(1, BestEffortSkill(name="clear_plant_stands", tasks=(
            NavSkill(name="nav_ring_approach", target_px=(cx * 16 + 8, cy * 16 + 8), radius=40,
                     soft_radius=40, timeout=4000, require_tilemap=0x00),
            FarmClearer(name="clear_plant_stands", fetch_tools=False,
                priority=[DebrisType.WEED, DebrisType.STONE],
                prefer_lift_for_stones=True, farm_bounds=(cx - 3, cy - 3, cx + 2, cy + 2),
                handoff="type_clear", timeout=6000),)))
    if include_plant:
        if opening is not None:
            # A notch block is entered through its bare notch, sideways onto the hoed centre. A straight
            # walk in over an edge cell stopped a pixel short when an object sat on the next cell
            # (PR_blocks_Y1_Spring_D9: farmer (198,640) under (12,39), sprite 0x0116 on (11,39), two
            # nav timeouts). The bag sows around the farmer's tile, so any pixel in the centre tile will do.
            from harvest.tasks.nav import TILE_SIZE

            nx, ny = center[0] + opening[0], center[1] + opening[1]
            skills.append(NavSkill(name="nav_block_notch", target_px=(nx * TILE_SIZE + 8, ny * TILE_SIZE + 8),
                                   radius=5, soft_radius=5, timeout=4000, require_tilemap=0x00))
        skills.extend(
            [
                farm_nav_pocket_plant_skill(center=center, radius=7 if opening is not None else 5),
                farm_select_carry_skill(seed_item_id(seed_type)),
                plant_until_plot_skill(seed_type=seed_type, center=center, layout_name=layout_name, min_planted=layout.crop_count),
            ]
        )
    if include_water:
        skills.append(farm_pocket_water_skill(center=center))
    return sequence_skills("pocket_plant_plot", *skills, idle_between=True)

def farm_pocket_water_skill(*, center=None, timeout: int = 4000):
    from harvest.core.tile_catalog import Tool
    from harvest.maps.farm_pond import WEST_POCKET_PLANT_CENTER
    from harvest.tasks.crop_skills import EnsureCanFilledTask, pocket_water_ring_skills
    center = center or WEST_POCKET_PLANT_CENTER
    skills: list = [
        farm_fence_jump_toss_skill(),
        farm_select_carry_skill(int(Tool.WATERING_CAN)),
        EnsureCanFilledTask(),
        farm_nav_pocket_plant_skill(center=center),
        *pocket_water_ring_skills(center),
    ]
    return sequence_skills("pocket_water_ring", *skills, idle_between=True)

@dataclass
class CoopFeedAdultsSkill(Task):

    name: str = "coop_feed_adults"
    max_feed_adults: Optional[int] = None
    timeout: int = 6000
    _inner: Any = field(init=False)
    def __post_init__(self) -> None:
        from harvest.tasks.coop_task import CoopChoresTask
        self._inner = CoopChoresTask( name=self.name, max_feed_adults=self.max_feed_adults, timeout=self.timeout, )
    def reset(self, world: WorldState) -> None:
        self._inner.reset(world)
    def can_start(self, world: WorldState) -> bool:
        return self._inner.can_start(world)
    @property
    def fed_count(self) -> int:
        return getattr(self._inner, "fed_count", 0)
    def progress_snapshot(self) -> ProgressSnapshot:
        return self._inner.progress_snapshot()
    def _feed_done(self) -> bool:
        feed_phases = ("feed_nav", "feed_act", "feed_verify", "feed_place_nav", "feed_place_verify")
        return getattr(self._inner, "_feed_remaining", 0) <= 0 and getattr(self._inner, "_phase", "") not in feed_phases
    def step(self, world: WorldState) -> TaskResult:
        if self._feed_done():
            return TaskResult(status=TaskStatus.SUCCESS, reason=f"fed={self.fed_count} adults")
        result = self._inner.step(world)
        if self._feed_done():
            return TaskResult(status=TaskStatus.SUCCESS, reason=f"fed={self.fed_count} adults")
        return result

@dataclass
class CoopCollectEggsSkill(Task):

    name: str = "coop_collect_eggs"
    timeout: int = 3000
    _inner: Any = field(init=False)
    def __post_init__(self) -> None:
        from harvest.tasks.coop_task import CoopChoresTask
        self._inner = CoopChoresTask(name=self.name, timeout=self.timeout)
    def reset(self, world: WorldState) -> None:
        self._inner.reset(world)
        self._inner._phase = "egg_nav"
    def can_start(self, world: WorldState) -> bool:
        return self._inner.can_start(world)
    @property
    def egg_collected(self) -> bool:
        return getattr(self._inner, "egg_collected", False)
    def progress_snapshot(self) -> ProgressSnapshot:
        return self._inner.progress_snapshot()
    def _egg_done(self) -> bool:
        done_phases = ("decide", "incubate_nav", "ship_nav", "exit_prep_nav", "done")
        return self.egg_collected or getattr(self._inner, "_phase", "") in done_phases
    def step(self, world: WorldState) -> TaskResult:
        if self._egg_done():
            return TaskResult(status=TaskStatus.SUCCESS, reason=f"egg_collected={self.egg_collected}")
        result = self._inner.step(world)
        if self._egg_done():
            return TaskResult(status=TaskStatus.SUCCESS, reason=f"egg_collected={self.egg_collected}")
        return result

@dataclass
class CoopEggDispositionSkill(Task):

    name: str = "coop_egg_disposition"
    egg_mode: str = "auto"
    timeout: int = 3000
    _inner: Any = field(init=False)
    def __post_init__(self) -> None:
        from harvest.tasks.coop_task import CoopChoresTask
        self._inner = CoopChoresTask(name=self.name, egg_mode=self.egg_mode, timeout=self.timeout)
    def reset(self, world: WorldState) -> None:
        self._inner.reset(world)
        self._inner.egg_collected = True
        self._inner._phase = "decide"
    def can_start(self, world: WorldState) -> bool:
        return self._inner.can_start(world)
    @property
    def egg_incubated(self) -> bool:
        return getattr(self._inner, "egg_incubated", False)
    @property
    def egg_shipped(self) -> bool:
        return getattr(self._inner, "egg_shipped", False)
    def progress_snapshot(self) -> ProgressSnapshot:
        return self._inner.progress_snapshot()
    def _disp_done(self) -> bool:
        return getattr(self._inner, "_phase", "") in ("exit_prep_nav", "done")
    def step(self, world: WorldState) -> TaskResult:
        if self._disp_done():
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=f"disposition={self.egg_mode} incub={self.egg_incubated} ship={self.egg_shipped}",
            )
        result = self._inner.step(world)
        if self._disp_done():
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=f"disposition={self.egg_mode} incub={self.egg_incubated} ship={self.egg_shipped}",
            )
        return result

@dataclass
class CoopExitStagingSkill(Task):

    name: str = "coop_exit_staging"
    timeout: int = 900
    _inner: Any = field(init=False)
    def __post_init__(self) -> None:
        from harvest.tasks.coop_task import CoopChoresTask
        self._inner = CoopChoresTask(name=self.name, timeout=self.timeout)
    def reset(self, world: WorldState) -> None:
        self._inner.reset(world)
        self._inner._phase = "exit_prep_nav"
    def can_start(self, world: WorldState) -> bool:
        return self._inner.can_start(world)
    def progress_snapshot(self) -> ProgressSnapshot:
        return self._inner.progress_snapshot()
    def step(self, world: WorldState) -> TaskResult:
        if getattr(self._inner, "_phase", "") == "done":
            return TaskResult(status=TaskStatus.SUCCESS, reason="at exit door")
        result = self._inner.step(world)
        if getattr(self._inner, "_phase", "") == "done":
            return TaskResult(status=TaskStatus.SUCCESS, reason="at exit door")
        return result

def coop_chores_composed_task( *, egg_mode: str = "auto", max_feed_adults: Optional[int] = None, ) -> SkillSequence:
    return sequence_skills(
        "coop_chores_sequence",
        CoopFeedAdultsSkill(max_feed_adults=max_feed_adults),
        CoopCollectEggsSkill(),
        CoopEggDispositionSkill(egg_mode=egg_mode),
        CoopExitStagingSkill(),
        idle_between=False,
    )

__all__ = [
    "ButtonSequenceSkill",
    "CoopCollectEggsSkill",
    "CoopEggDispositionSkill",
    "CoopExitStagingSkill",
    "CoopFeedAdultsSkill",
    "InteractSkill",
    "NavSkill",
    "NavigateFn",
    "NavigateUntilArrivedSkill",
    "PressAInteractSkill",
    "SequenceSkill",
    "SkillSequence",
    "VerifyRamSkill",
    "coop_chores_composed_task",
    "coop_feed_place_sequence",
    "coop_nav_to_egg_skill",
    "coop_nav_to_exit_prep_skill",
    "coop_nav_to_feed_bin_skill",
    "coop_nav_to_feed_spot_skill",
    "coop_nav_to_incubator_skill",
    "coop_nav_to_shipping_bin_skill",
    "coop_press_egg_pickup_skill",
    "coop_press_feed_place_skill",
    "coop_press_feed_skill",
    "coop_press_incubator_skill",
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
    "SKILL_SEAMS",
    "register_skill_seam",
    "skill_seam",
]

# Factories later domains register. None means the skill table has no row yet.
SKILL_SEAMS = ("social", "family", "cover")
_SKILL_ROWS: dict[str, Callable | None] = {name: None for name in SKILL_SEAMS}

def register_skill_seam(name: str, factory: Callable | None = None) -> None:
    if name not in _SKILL_ROWS:
        raise KeyError(name)
    _SKILL_ROWS[name] = factory

def skill_seam(name: str) -> Callable | None:
    if name not in _SKILL_ROWS:
        raise KeyError(name)
    return _SKILL_ROWS[name]

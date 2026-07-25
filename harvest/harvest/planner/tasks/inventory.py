"""Inventory, shed, and simple deadline tasks used by the day planner."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.tasks.crop_planter import SEED_ITEM, is_rainy_weather
from harvest.tasks.farm_clearer import (
    Tool,
    Point,
    make_action,
    get_pos_from_ram,
    ADDR_TILEMAP,
    ADDR_INPUT_LOCK,
    TILE_SIZE,
)
from harvest.maps.map_config import ROUTES
from harvest.tasks.primitives import (
    dismiss_dialogue_result,
    drain_action_queue,
    press_a_sequence,
)
from harvest.tasks.recorded_task import RecordedTask
from harvest.planner.day_plan_status import (
    TASKS_DIR,
    FARM_TILEMAP,
    SHED_TILEMAP,
    BARN_TILEMAP,
    COOP_TILEMAP,
    ADDR_TOOL_SELECTED,
    ADDR_TOOL_BACKPACK,
    read_world_day_time,
    ram_seed_count,
    is_farm_tilemap,
    is_house_tilemap,
)
from harvest.core.scene import (
    SceneLocation,
    SceneMode,
    classify_scene_from_ram,
    scene_indicates_ending,
)
from harvest.core.ram_catalog import read_ram_u8
from harvest.planner.tasks.navigation import MultiMapNavTask, NavTask
from harvest.planner.tasks.transitions import (
    DirectionalTransitionTask,
    ExitBuildingTask,
    SHED_ENTER_DOOR_X,
    SHED_ENTER_OVERSHOOT_Y,
    SHED_ENTER_STAND_TILE,
)

def tool_in_carry_pair(ram: np.ndarray, tool_id: int) -> bool:
    if ADDR_TOOL_SELECTED < len(ram) and int(ram[ADDR_TOOL_SELECTED]) == tool_id:
        return True
    if ADDR_TOOL_BACKPACK < len(ram) and int(ram[ADDR_TOOL_BACKPACK]) == tool_id:
        return True
    return False


def seed_in_carry_pair(ram: np.ndarray, seed_type: str = "potato") -> bool:
    seed_item = SEED_ITEM.get(seed_type, SEED_ITEM["potato"])
    return tool_in_carry_pair(ram, seed_item)


FARM_BUILDING_EXIT_STAND_TILES: Dict[int, Tuple[int, int]] = {
    SHED_TILEMAP: (8, 12),
    BARN_TILEMAP: (8, 22),
    COOP_TILEMAP: (8, 12),
}
FARM_BUILDING_EXIT_DOOR_X = 8 * 16 + 8
BARN_EXIT_TROUGH_X = 113
BARN_EXIT_TROUGH_MAX_X = 130
BARN_EXIT_RIGHT_AISLE_X = 204
BARN_EXIT_BYPASS_X = 216
BARN_EXIT_LOWER_Y = 20 * 16 + 8
BARN_EXIT_DOOR_X = 8 * 16 + 8


def shed_enter_transition(*, name: str, timeout: int) -> DirectionalTransitionTask:
    """Enter the tool shed from the outdoor door stand."""
    return DirectionalTransitionTask(
        name=name,
        direction="up",
        origin_tilemap=FARM_TILEMAP,
        target_tilemap=SHED_TILEMAP,
        timeout=timeout,
        min_frames_before_success=10,
        stand_tile=SHED_ENTER_STAND_TILE,
        stand_tolerance=0,
        target_stand_tile=(8, 12),
        target_stand_tolerance=1,
        settle_frames=30,
        door_align_px=SHED_ENTER_DOOR_X,
        overshoot_limit_px=SHED_ENTER_OVERSHOOT_Y,
        require_empty_hands=True,
    )


@dataclass(frozen=True)
class RecordingSliceSpec:
    task_name: str
    start_frame: int = 0
    end_frame: Optional[int] = None


@dataclass(frozen=True)
class ShedToolSpec:
    nav_target_px: Tuple[int, int]
    nav_radius: int
    enter_direction: str
    inside_recording: Optional[RecordingSliceSpec] = None
    farm_route: Optional[str] = None
    inside_stand_px: Optional[Tuple[int, int]] = None
    inside_face: str = "up"
    inside_settle_frames: int = 70
    inside_timeout: int = 900


@dataclass(frozen=True)
class ShedSeedSpec:
    nav_target_px: Tuple[int, int]
    nav_radius: int
    enter_direction: str
    inside_recording: RecordingSliceSpec
    farm_route: Optional[str] = None


SHED_TOOL_SPECS: Dict[int, ShedToolSpec] = {
    int(Tool.MILKER): ShedToolSpec(
        farm_route="farm_to_shed",
        nav_target_px=(422, 474),
        nav_radius=12,
        enter_direction="up",
        # Tool layout from HM-Decomp DATA16_81BE0F:
        # milker sprite at px(0x40, 0x90); stand one tile below it.
        inside_stand_px=(64, 168),
        inside_face="up",
    ),
    int(Tool.BRUSH): ShedToolSpec(
        farm_route="farm_to_shed",
        nav_target_px=(422, 474),
        nav_radius=12,
        enter_direction="up",
        # Brush sprite at px(0x50, 0x90); stand one tile below it.
        inside_stand_px=(80, 168),
        inside_face="up",
    ),
    int(Tool.WATERING_CAN): ShedToolSpec(
        farm_route="farm_to_shed",
        nav_target_px=(422, 474),
        nav_radius=12,
        enter_direction="up",
        # Watering can sprite at px(0x60, 0x90); stand one tile below it.
        inside_stand_px=(96, 168),
        inside_face="up",
    ),
}

SHED_SEED_SPECS: Dict[str, ShedSeedSpec] = {
    # The water-can shelf is where the potato bag is left after the normal
    # morning "swap seeds for watering can" route. Replaying the same inside
    # shed interaction swaps it back if the bag is there.
    "potato": ShedSeedSpec(
        farm_route="farm_to_shed",
        nav_target_px=(422, 474),
        nav_radius=12,
        enter_direction="up",
        inside_recording=RecordingSliceSpec(
            task_name="shed_get_watering_can",
            start_frame=455,
            end_frame=1020,
        ),
    ),
}


def shed_farm_route_name(ram: np.ndarray, default_route: Optional[str]) -> Optional[str]:
    """Select the shed approach route for the current outdoor farm position."""
    if default_route != "farm_to_shed":
        return default_route
    pos = get_pos_from_ram(ram)
    tile = (pos.x // TILE_SIZE, pos.y // TILE_SIZE)
    if tile[0] >= 19 and tile[1] <= 24:
        return "upper_farm_to_shed"
    if tile[0] >= 23 and tile[1] >= 29:
        return "near_shed_to_shed"
    if tile[1] >= 29:
        return "field_to_shed"
    return default_route


def load_recording_slice(spec: RecordingSliceSpec, tasks_dir: str) -> RecordedTask:
    recording = RecordedTask.load(spec.task_name, tasks_dir)
    frames = recording.frames[spec.start_frame:spec.end_frame]
    return RecordedTask(
        name=f"{spec.task_name}[{spec.start_frame}:{spec.end_frame}]",
        frames=frames,
        start_state=recording.start_state,
    )




@dataclass
class DeadlineCheckTask(Task):
    """Require the current in-game time to still be before a cutoff."""

    name: str = "deadline_check"
    latest_hour: int = 17
    latest_minute: int = 0

    def reset(self, world: WorldState) -> None:
        return None

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        _day, hour, minute = read_world_day_time(world.ram)
        if (hour, minute) >= (self.latest_hour, self.latest_minute):
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"cutoff reached at {hour:02d}:{minute:02d}",
            )
        return TaskResult(
            status=TaskStatus.SUCCESS,
            reason=f"before cutoff at {hour:02d}:{minute:02d}",
        )


@dataclass
class WaitUntilTimeTask(Task):
    """Idle until the in-game clock reaches a target time."""

    name: str = "wait_until_time"
    target_hour: int = 12
    target_minute: int = 0
    timeout: int = 4000

    _step_count: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        _day, hour, minute = read_world_day_time(world.ram)
        if (hour, minute) >= (self.target_hour, self.target_minute):
            return TaskResult(status=TaskStatus.SUCCESS, reason=f"reached {hour:02d}:{minute:02d}")
        if self._step_count > self.timeout:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"wait timeout at {hour:02d}:{minute:02d}",
            )
        input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
        if input_lock != 1:
            return dismiss_dialogue_result(self._step_count, buttons=("a", "b"), pulse_every=1)
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))


@dataclass
class ExitToFarmTask(Task):
    """Exit the current farm building without assuming a house-only opener."""

    name: str = "exit_to_farm"
    tasks_dir: str = TASKS_DIR
    house_timeout: int = 2200

    cutscene_mash_limit: int = 240

    _task: Optional[Task] = field(default=None, init=False)
    _blocked_reason: str = field(default="", init=False)
    _step_count: int = field(default=0, init=False)
    _cutscene_mash_count: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._blocked_reason = ""
        self._step_count = 0
        self._cutscene_mash_count = 0
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if is_farm_tilemap(tilemap):
            self._task = None
            return
        if is_house_tilemap(tilemap):
            self._task = ExitBuildingTask(
                target_tilemap=FARM_TILEMAP,
                dialog_frames=120,
                timeout=self.house_timeout,
            )
        elif tilemap in {SHED_TILEMAP, BARN_TILEMAP, COOP_TILEMAP}:
            self._task = DirectionalTransitionTask(
                name="directional_transition_exit_to_farm",
                direction="down",
                origin_tilemap=tilemap,
                target_tilemap=FARM_TILEMAP,
                timeout=1800 if tilemap == BARN_TILEMAP else 1200,
                min_frames_before_success=15,
                stand_tile=FARM_BUILDING_EXIT_STAND_TILES.get(tilemap),
                stand_tolerance=1,
                door_align_px=FARM_BUILDING_EXIT_DOOR_X,
                settle_frames=5,
            )
        elif tilemap in {0x0C, 0x04, 0x05, 0x10}:
            route_name = {
                0x0C: "path_to_farm",
                0x04: "town_to_farm",
                0x05: "event_town_to_farm",
                0x10: "mountain_to_farm",
            }[tilemap]
            waypoints = ROUTES.get(route_name, [])
            self._task = MultiMapNavTask(
                name=f"return_{route_name}",
                waypoints=list(waypoints),
                timeout=6000,
                initial_settle_frames=30,
            )
        else:
            scene = classify_scene_from_ram(world.ram)
            if scene_indicates_ending(scene) or scene.mode == SceneMode.INVALID_COORDINATES:
                self._task = None
                self._blocked_reason = f"cannot exit to farm from {scene.summary()}"
                return
            if (
                scene.mode == SceneMode.CUTSCENE_EVENT
                or scene.location == SceneLocation.UNKNOWN
                or scene.mode == SceneMode.UNKNOWN_TILEMAP
            ):
                # Mash through event cutscenes in step(); do not invent a house exit.
                self._task = None
                return
            self._task = ExitBuildingTask(
                target_tilemap=FARM_TILEMAP,
                dialog_frames=120,
                timeout=self.house_timeout,
            )
        self._task.reset(world)

    def resume_after_hotswap(self, world: WorldState) -> None:
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if is_farm_tilemap(tilemap):
            self._task = None
            return
        if self._task is None:
            self.reset(world)
            return
        resume = getattr(self._task, "resume_after_hotswap", None)
        if callable(resume):
            resume(world)

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        scene = classify_scene_from_ram(world.ram)
        if scene_indicates_ending(scene):
            return TaskResult(
                status=TaskStatus.BLOCKED,
                reason=f"cannot exit to farm from {scene.summary()}",
            )
        if scene.is_transient:
            if scene.mode == SceneMode.CUTSCENE_EVENT:
                self._cutscene_mash_count += 1
                if self._cutscene_mash_count > self.cutscene_mash_limit:
                    return TaskResult(
                        status=TaskStatus.BLOCKED,
                        reason=f"cannot exit to farm from {scene.summary()}",
                    )
                return dismiss_dialogue_result(
                    self._step_count,
                    buttons=("a", "b"),
                    pulse_every=1,
                    reason=f"waiting through {scene.mode.value}",
                )
            self._cutscene_mash_count = 0
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action()),
                reason=f"waiting through {scene.summary()}",
            )
        if scene.needs_input_dismiss:
            self._cutscene_mash_count = 0
            return dismiss_dialogue_result(
                self._step_count,
                buttons=("a", "b"),
                pulse_every=1,
                reason=f"dismiss {scene.mode.value} before exit",
            )
        self._cutscene_mash_count = 0
        if self._blocked_reason:
            return TaskResult(status=TaskStatus.BLOCKED, reason=self._blocked_reason)
        if is_farm_tilemap(tilemap) and self._task is None:
            return TaskResult(status=TaskStatus.SUCCESS, reason=f"tilemap=0x{tilemap:02X}")
        if self._task is None:
            self.reset(world)
            if self._blocked_reason:
                return TaskResult(status=TaskStatus.BLOCKED, reason=self._blocked_reason)
            if self._task is None:
                return TaskResult(status=TaskStatus.SUCCESS, reason=f"tilemap=0x{tilemap:02X}")
        return self._task.step(world)


@dataclass
class SwapCarrySlotsTask(Task):
    """Tap X until the selected tool and backpack tool swap places."""

    name: str = "swap_carry_slots"
    timeout: int = 90

    _step_count: int = field(default=0, init=False)
    _start_selected: int = field(default=0, init=False)
    _start_backpack: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._start_selected = read_ram_u8(world.ram, ADDR_TOOL_SELECTED)
        self._start_backpack = read_ram_u8(world.ram, ADDR_TOOL_BACKPACK)

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        selected = read_ram_u8(world.ram, ADDR_TOOL_SELECTED)
        backpack = read_ram_u8(world.ram, ADDR_TOOL_BACKPACK)
        if selected == self._start_backpack and backpack == self._start_selected:
            return TaskResult(status=TaskStatus.SUCCESS, reason="carry slots swapped")
        if self._step_count > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason="carry slot swap timeout")
        input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
        if input_lock != 1:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(make_action(x=True) if self._step_count % 6 == 1 else make_action()),
        )


@dataclass
class ShedShelfToolTask(Task):
    """Walk to a known tool-shed shelf coordinate and take that tool normally."""

    name: str = "shed_shelf_tool"
    tool_id: int = int(Tool.BRUSH)
    stand_px: Tuple[int, int] = (80, 166)
    face: str = "up"
    settle_frames: int = 70
    timeout: int = 900
    radius: int = 2

    _phase: str = field(default="settle", init=False)
    _step_count: int = field(default=0, init=False)
    _task: Optional[Task] = field(default=None, init=False)
    _action_queue: deque = field(default_factory=deque, init=False)

    def reset(self, world: WorldState) -> None:
        self._phase = "settle"
        self._step_count = 0
        self._task = None
        self._action_queue.clear()

    def can_start(self, world: WorldState) -> bool:
        return True

    def _queue_take_tool(self) -> None:
        self._action_queue.extend(
            press_a_sequence(
                self.face,
                face_frames=6,
                pre_press_settle_frames=0,
                hold_frames=18,
                settle_frames=24,
                hold_face_with_a=True,
            )
        )

    def _activate_nav(self, world: WorldState) -> TaskResult:
        self._phase = "nav"
        self._task = NavTask(
            name=f"nav_shed_shelf_0x{self.tool_id:02X}",
            target_px=Point(self.stand_px[0], self.stand_px[1]),
            radius=self.radius,
            timeout=self.timeout,
            stasis_repath=90,
        )
        self._task.reset(world)
        return self._task.step(world)

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        if self._step_count > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason="shed shelf timeout")
        if tool_in_carry_pair(world.ram, self.tool_id):
            return TaskResult(status=TaskStatus.SUCCESS, reason=f"tool 0x{self.tool_id:02X} ready")

        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if tilemap != SHED_TILEMAP:
            return TaskResult(status=TaskStatus.FAILURE, reason=f"not in shed tilemap=0x{tilemap:02X}")

        input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
        if input_lock != 1:
            return dismiss_dialogue_result(self._step_count, pulse_every=1)

        if self._phase == "settle":
            if self._step_count < self.settle_frames:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            return self._activate_nav(world)

        if self._phase == "nav":
            if self._task is None:
                return self._activate_nav(world)
            result = self._task.step(world)
            if result.status == TaskStatus.RUNNING:
                return result
            if result.status == TaskStatus.FAILURE:
                return TaskResult(status=TaskStatus.FAILURE, reason=f"shed shelf nav failed: {result.reason}")
            self._phase = "take"
            self._task = None
            self._queue_take_tool()

        if self._phase == "take":
            queued = drain_action_queue(self._action_queue)
            if queued is not None:
                return queued
            if tool_in_carry_pair(world.ram, self.tool_id):
                return TaskResult(status=TaskStatus.SUCCESS, reason=f"tool 0x{self.tool_id:02X} ready")
            return TaskResult(status=TaskStatus.FAILURE, reason=f"tool 0x{self.tool_id:02X} missing after shelf")

        return TaskResult(status=TaskStatus.FAILURE, reason=f"unknown shed shelf phase {self._phase}")


@dataclass
class EnsureCarryToolTask(Task):
    """Ensure a desired shed tool is present in the 2-slot carry pair."""

    name: str = "ensure_carry_tool"
    tool_id: int = int(Tool.WATERING_CAN)
    tasks_dir: str = TASKS_DIR
    nav_timeout: int = 8000
    enter_timeout: int = 1500

    _phase: str = field(default="start", init=False)
    _task: Optional[Task] = field(default=None, init=False)
    _fallback_reason: str = field(default="", init=False)

    def reset(self, world: WorldState) -> None:
        self._phase = "start"
        self._task = None
        self._fallback_reason = ""

    def can_start(self, world: WorldState) -> bool:
        return True

    def _shed_spec(self) -> Optional[ShedToolSpec]:
        return SHED_TOOL_SPECS.get(self.tool_id)

    def _inside_task(self, spec: ShedToolSpec) -> Task:
        if spec.inside_recording is not None:
            return load_recording_slice(spec.inside_recording, self.tasks_dir)
        if spec.inside_stand_px is None:
            raise ValueError(f"no inside shed plan for 0x{self.tool_id:02X}")
        return ShedShelfToolTask(
            name=f"shed_shelf_tool_0x{self.tool_id:02X}",
            tool_id=self.tool_id,
            stand_px=spec.inside_stand_px,
            face=spec.inside_face,
            settle_frames=spec.inside_settle_frames,
            timeout=spec.inside_timeout,
        )

    def _activate(self, phase: str, task: Task, world: WorldState) -> None:
        self._phase = phase
        self._task = task
        task.reset(world)

    def _start_next_phase(self, world: WorldState) -> TaskResult:
        if tool_in_carry_pair(world.ram, self.tool_id):
            return TaskResult(status=TaskStatus.SUCCESS, reason=f"tool 0x{self.tool_id:02X} ready")

        spec = self._shed_spec()
        if spec is None:
            return TaskResult(status=TaskStatus.FAILURE, reason=f"no shed tool plan for 0x{self.tool_id:02X}")

        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if tilemap == SHED_TILEMAP:
            self._activate("inside", self._inside_task(spec), world)
            return self._task.step(world)
        if is_farm_tilemap(tilemap):
            farm_route = shed_farm_route_name(world.ram, spec.farm_route)
            if farm_route:
                waypoints = ROUTES.get(farm_route, [])
                if not waypoints:
                    return TaskResult(
                        status=TaskStatus.FAILURE,
                        reason=f"missing shed farm route {farm_route}",
                    )
                self._activate(
                    "route",
                    MultiMapNavTask(
                        name=f"route_shed_tool_0x{self.tool_id:02X}",
                        waypoints=list(waypoints),
                        timeout=self.nav_timeout,
                        initial_settle_frames=0,
                    ),
                    world,
                )
            else:
                self._activate(
                    "nav",
                    NavTask(
                        name=f"nav_shed_tool_0x{self.tool_id:02X}",
                        target_px=Point(spec.nav_target_px[0], spec.nav_target_px[1]),
                        radius=spec.nav_radius,
                        timeout=self.nav_timeout,
                    ),
                    world,
                )
            return self._task.step(world)

        self._activate("exit_to_farm", ExitToFarmTask(tasks_dir=self.tasks_dir), world)
        return self._task.step(world)

    def step(self, world: WorldState) -> TaskResult:
        if self._task is None:
            return self._start_next_phase(world)

        result = self._task.step(world)
        if result.status == TaskStatus.RUNNING:
            return result
        if result.status == TaskStatus.FAILURE:
            tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
            if self._phase in {"route", "nav", "enter"} and tilemap == SHED_TILEMAP:
                spec = self._shed_spec()
                self._activate("inside", self._inside_task(spec), world)
                return self._task.step(world)
            reason = result.reason or "unknown"
            return TaskResult(status=TaskStatus.FAILURE, reason=f"{self._phase} failed: {reason}")

        if self._phase == "exit_to_farm":
            self._task = None
            return self._start_next_phase(world)

        if self._phase in {"route", "nav"}:
            self._activate(
                "enter",
                shed_enter_transition(
                    name=f"enter_shed_tool_0x{self.tool_id:02X}",
                    timeout=self.enter_timeout,
                ),
                world,
            )
            return self._task.step(world)

        if self._phase == "enter":
            spec = self._shed_spec()
            self._activate("inside", self._inside_task(spec), world)
            return self._task.step(world)

        if self._phase == "inside":
            if tool_in_carry_pair(world.ram, self.tool_id):
                return TaskResult(status=TaskStatus.SUCCESS, reason=f"tool 0x{self.tool_id:02X} ready")
            return TaskResult(status=TaskStatus.FAILURE, reason=f"tool 0x{self.tool_id:02X} missing after shed task")

        return TaskResult(status=TaskStatus.FAILURE, reason=f"unknown phase {self._phase}")


@dataclass
class EnsureAnimalToolsTask(Task):
    """Ensure the milker and brush are both in the two carried tool slots."""

    name: str = "ensure_animal_tools"
    tasks_dir: str = TASKS_DIR
    first_tool_id: int = int(Tool.BRUSH)
    second_tool_id: int = int(Tool.MILKER)

    _task: Optional[Task] = field(default=None, init=False)
    _phase: str = field(default="start", init=False)
    _last_reason: str = field(default="", init=False)
    _failed_reason: str = field(default="", init=False)

    def reset(self, world: WorldState) -> None:
        self._task = None
        self._phase = "start"
        self._last_reason = ""
        self._failed_reason = ""

    def can_start(self, world: WorldState) -> bool:
        return True

    def _selected_tool(self, ram: np.ndarray) -> int:
        return read_ram_u8(ram, ADDR_TOOL_SELECTED)

    def _backpack_tool(self, ram: np.ndarray) -> int:
        return read_ram_u8(ram, ADDR_TOOL_BACKPACK)

    def _ready(self, ram: np.ndarray) -> bool:
        return tool_in_carry_pair(ram, self.first_tool_id) and tool_in_carry_pair(ram, self.second_tool_id)

    def _activate(self, phase: str, task: Task, world: WorldState) -> TaskResult:
        self._phase = phase
        self._task = task
        task.reset(world)
        return task.step(world)

    def _start_next_phase(self, world: WorldState) -> TaskResult:
        if self._ready(world.ram):
            tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
            if tilemap == SHED_TILEMAP:
                return self._activate("exit_after_ready", ExitToFarmTask(tasks_dir=self.tasks_dir), world)
            return TaskResult(status=TaskStatus.SUCCESS, reason="animal tools ready")

        selected = self._selected_tool(world.ram)
        backpack = self._backpack_tool(world.ram)
        desired = {self.first_tool_id, self.second_tool_id}
        if selected in desired and backpack not in desired:
            return self._activate("swap", SwapCarrySlotsTask(), world)

        missing = self.first_tool_id if not tool_in_carry_pair(world.ram, self.first_tool_id) else self.second_tool_id
        return self._activate(
            f"ensure_0x{missing:02X}",
            EnsureCarryToolTask(
                name=f"ensure_animal_tool_0x{missing:02X}",
                tool_id=missing,
                tasks_dir=self.tasks_dir,
            ),
            world,
        )

    def step(self, world: WorldState) -> TaskResult:
        if self._task is None:
            return self._start_next_phase(world)

        result = self._task.step(world)
        if result.status == TaskStatus.RUNNING:
            return result
        if self._phase == "exit_after_failure":
            return TaskResult(status=TaskStatus.FAILURE, reason=self._failed_reason or "animal tool prep failed")
        if result.status == TaskStatus.FAILURE:
            reason = result.reason or "unknown"
            tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
            if tilemap == SHED_TILEMAP:
                self._failed_reason = f"{self._phase} failed: {reason}"
                return self._activate("exit_after_failure", ExitToFarmTask(tasks_dir=self.tasks_dir), world)
            return TaskResult(status=TaskStatus.FAILURE, reason=f"{self._phase} failed: {reason}")
        self._last_reason = result.reason or ""
        self._task = None
        return self._start_next_phase(world)


@dataclass
class EnsureCropSeedsTask(Task):
    """Best-effort retrieval of stored crop seeds from the tool shed."""

    name: str = "ensure_crop_seeds"
    seed_type: str = "potato"
    tasks_dir: str = TASKS_DIR
    nav_timeout: int = 4000
    enter_timeout: int = 1500

    _phase: str = field(default="start", init=False)
    _task: Optional[Task] = field(default=None, init=False)
    _fallback_reason: str = field(default="", init=False)

    def reset(self, world: WorldState) -> None:
        self._phase = "start"
        self._task = None
        self._fallback_reason = ""

    def can_start(self, world: WorldState) -> bool:
        return True

    def _seed_item(self) -> int:
        return SEED_ITEM.get(self.seed_type, SEED_ITEM["potato"])

    def _shed_spec(self) -> Optional[ShedSeedSpec]:
        return SHED_SEED_SPECS.get(self.seed_type)

    def _activate(self, phase: str, task: Task, world: WorldState) -> None:
        self._phase = phase
        self._task = task
        task.reset(world)

    def _success_reason(self, reason: str) -> TaskResult:
        return TaskResult(status=TaskStatus.SUCCESS, reason=reason)

    def _needs_watering_can_restore(self, world: WorldState) -> bool:
        return not is_rainy_weather(world.ram) and not tool_in_carry_pair(world.ram, int(Tool.WATERING_CAN))

    def _restore_watering_can(self, world: WorldState) -> TaskResult:
        self._activate(
            "restore_watering_can",
            EnsureCarryToolTask(
                name=f"restore_watering_can_after_seed_{self.seed_type}",
                tool_id=int(Tool.WATERING_CAN),
                tasks_dir=self.tasks_dir,
            ),
            world,
        )
        return self._task.step(world)

    def _complete_after_inside(self, world: WorldState, *, suffix: str = "") -> TaskResult:
        reason = self._fallback_reason or f"{self.seed_type} seed stock ready"
        if suffix:
            reason = f"{reason}{suffix}"
        if self._needs_watering_can_restore(world):
            return self._restore_watering_can(world)
        return self._success_reason(reason)

    def _start_next_phase(self, world: WorldState) -> TaskResult:
        seed_item = self._seed_item()
        if tool_in_carry_pair(world.ram, seed_item):
            return self._success_reason(f"seed tool 0x{seed_item:02X} ready")

        stored = ram_seed_count(world.ram, self.seed_type)
        if stored <= 0:
            return self._success_reason(f"no stored {self.seed_type} seeds")

        spec = self._shed_spec()
        if spec is None:
            return self._success_reason(f"no shed seed plan for {self.seed_type}")

        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if tilemap == SHED_TILEMAP:
            self._activate("inside", load_recording_slice(spec.inside_recording, self.tasks_dir), world)
            return self._task.step(world)
        if is_farm_tilemap(tilemap):
            farm_route = shed_farm_route_name(world.ram, spec.farm_route)
            if farm_route:
                waypoints = ROUTES.get(farm_route, [])
                if not waypoints:
                    return self._success_reason(f"missing shed seed route {farm_route}")
                self._activate(
                    "route",
                    MultiMapNavTask(
                        name=f"route_seed_{self.seed_type}",
                        waypoints=list(waypoints),
                        timeout=self.nav_timeout,
                        initial_settle_frames=0,
                    ),
                    world,
                )
            else:
                self._activate(
                    "nav",
                    NavTask(
                        name=f"nav_seed_{self.seed_type}",
                        target_px=Point(spec.nav_target_px[0], spec.nav_target_px[1]),
                        radius=spec.nav_radius,
                        timeout=self.nav_timeout,
                    ),
                    world,
                )
            return self._task.step(world)

        self._activate("exit_to_farm", ExitToFarmTask(tasks_dir=self.tasks_dir), world)
        return self._task.step(world)

    def step(self, world: WorldState) -> TaskResult:
        if self._task is None:
            return self._start_next_phase(world)

        result = self._task.step(world)
        if result.status == TaskStatus.RUNNING:
            return result
        if result.status == TaskStatus.FAILURE:
            tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
            if self._phase in {"route", "nav", "enter"} and tilemap == SHED_TILEMAP:
                spec = self._shed_spec()
                self._activate("inside", load_recording_slice(spec.inside_recording, self.tasks_dir), world)
                return self._task.step(world)
            reason = result.reason or "unknown"
            if self._phase == "exit_after_inside":
                suffix = f"; shed exit failed: {reason}"
                return self._complete_after_inside(world, suffix=suffix)
            if self._phase == "restore_watering_can":
                return TaskResult(status=TaskStatus.FAILURE, reason=f"restore_watering_can failed: {reason}")
            return self._success_reason(f"{self._phase} did not get {self.seed_type} seeds: {reason}")

        if self._phase == "exit_to_farm":
            self._task = None
            return self._start_next_phase(world)

        if self._phase == "exit_after_inside":
            return self._complete_after_inside(world)

        if self._phase == "restore_watering_can":
            return self._success_reason(self._fallback_reason or f"{self.seed_type} seed stock ready")

        if self._phase in {"route", "nav"}:
            self._activate(
                "enter",
                shed_enter_transition(
                    name=f"enter_shed_seed_{self.seed_type}",
                    timeout=self.enter_timeout,
                ),
                world,
            )
            return self._task.step(world)

        if self._phase == "enter":
            spec = self._shed_spec()
            self._activate("inside", load_recording_slice(spec.inside_recording, self.tasks_dir), world)
            return self._task.step(world)

        if self._phase == "inside":
            seed_item = self._seed_item()
            if tool_in_carry_pair(world.ram, seed_item):
                return self._success_reason(f"seed tool 0x{seed_item:02X} ready")
            stored = ram_seed_count(world.ram, self.seed_type)
            self._fallback_reason = (
                f"{self.seed_type} seed stock={stored}, seed tool 0x{seed_item:02X} not carried"
            )
            tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
            if tilemap == SHED_TILEMAP:
                self._activate("exit_after_inside", ExitToFarmTask(tasks_dir=self.tasks_dir), world)
                return self._task.step(world)
            return self._complete_after_inside(world)

        return self._success_reason(f"unknown seed phase {self._phase}")


@dataclass
class FarmExitTask(Task):
    """Deterministic walk from the farmhouse door area to the west farm exit."""

    name: str = "farm_exit"
    target_px: Point = field(default_factory=lambda: Point(40, 424))
    radius: int = 12
    timeout: int = 3000

    _step_count: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        if self._step_count > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason="farm_exit timeout")

        input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
        if input_lock != 1:
            return dismiss_dialogue_result(self._step_count)

        pos = get_pos_from_ram(world.ram)
        dx = self.target_px.x - pos.x
        dy = self.target_px.y - pos.y
        if abs(dx) <= self.radius and abs(dy) <= self.radius:
            return TaskResult(status=TaskStatus.SUCCESS, reason="arrived")

        if abs(dy) > self.radius:
            direction = "down" if dy > 0 else "up"
        else:
            direction = "right" if dx > 0 else "left"
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(**{direction: True, "b": True})))



__all__ = [
    "RecordingSliceSpec",
    "ShedToolSpec",
    "ShedSeedSpec",
    "SHED_TOOL_SPECS",
    "SHED_SEED_SPECS",
    "tool_in_carry_pair",
    "seed_in_carry_pair",
    "shed_farm_route_name",
    "shed_enter_transition",
    "load_recording_slice",
    "DeadlineCheckTask",
    "WaitUntilTimeTask",
    "ExitToFarmTask",
    "SwapCarrySlotsTask",
    "ShedShelfToolTask",
    "EnsureCarryToolTask",
    "EnsureAnimalToolsTask",
    "EnsureCropSeedsTask",
    "FarmExitTask",
]

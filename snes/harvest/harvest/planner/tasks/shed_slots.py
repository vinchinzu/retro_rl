"""Shed shelf coordinates, farm-approach route selection, and carry-slot swap."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.core.carry import (
    ADDR_TOOL_BACKPACK,
    ADDR_TOOL_SELECTED,
    backpack_tool,
    seed_item_id,
    selected_tool,
    tool_in_carry_pair,
)
from harvest.core.ram_catalog import read_ram_u8
from harvest.core.tile_catalog import ADDR_INPUT_LOCK, ADDR_TILEMAP, Tool
from harvest.maps.map_config import WEST_POCKET_PLANT_CENTER
from harvest.planner.day_plan_status import FARM_TILEMAP, SHED_TILEMAP
from harvest.planner.tasks.navigation import NavTask
from harvest.planner.tasks.transitions import (
    DirectionalTransitionTask,
    SHED_ENTER_DOOR_X,
    SHED_ENTER_OVERSHOOT_Y,
    SHED_ENTER_STAND_TILE,
)
from harvest.tasks.nav import Point, TILE_SIZE, get_pos_from_ram, make_action
from harvest.tasks.primitives import drain_action_queue, press_a_sequence
from harvest.tasks.recorded_task import RecordedTask


_DEFAULT_SHED_NAV_PX = (422, 474)
_DEFAULT_SHED_NAV_RADIUS = 12
_DEFAULT_SHED_FARM_ROUTE = "farm_to_shed"


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
        walk_into_door=True,
    )


@dataclass(frozen=True)
class RecordingSliceSpec:
    task_name: str
    start_frame: int = 0
    end_frame: Optional[int] = None


@dataclass(frozen=True)
class ShedShelfSpec:
    """Where a tool or seed bag sits on the tool-shed shelf.

    Shop-bought bags land on the shelf (``shed_items`` bits), not in the
    2-slot carry pair. Stand under the sprite and press A — same as tools.
    """

    item_id: int
    inside_stand_px: Tuple[int, int]
    nav_target_px: Tuple[int, int] = _DEFAULT_SHED_NAV_PX
    nav_radius: int = _DEFAULT_SHED_NAV_RADIUS
    enter_direction: str = "up"
    farm_route: Optional[str] = _DEFAULT_SHED_FARM_ROUTE
    inside_face: str = "up"
    inside_settle_frames: int = 70
    inside_timeout: int = 900
    inside_recording: Optional[RecordingSliceSpec] = None


ShedToolSpec = ShedShelfSpec
ShedSeedSpec = ShedShelfSpec


def _tool_shelf(
    tool: Tool,
    stand_px: Tuple[int, int],
    *,
    settle: int = 70,
) -> ShedShelfSpec:
    return ShedShelfSpec(
        item_id=int(tool),
        inside_stand_px=stand_px,
        inside_settle_frames=settle,
    )


SHED_TOOL_SPECS: Dict[int, ShedShelfSpec] = {
    # DATA16_81BE0F (tool_id*8: sprite x,y). Stand y = sprite_y+24.
    int(Tool.MILKER): _tool_shelf(Tool.MILKER, (64, 168)),
    int(Tool.BRUSH): _tool_shelf(Tool.BRUSH, (80, 168)),
    int(Tool.WATERING_CAN): _tool_shelf(Tool.WATERING_CAN, (96, 168)),
    int(Tool.SICKLE): _tool_shelf(Tool.SICKLE, (144, 168)),
    int(Tool.HOE): _tool_shelf(Tool.HOE, (168, 166)),
    int(Tool.HAMMER): _tool_shelf(Tool.HAMMER, (176, 168)),
    int(Tool.AXE): _tool_shelf(Tool.AXE, (192, 168)),
}

SHED_SEED_SPECS: Dict[str, ShedShelfSpec] = {
    "potato": ShedShelfSpec(
        item_id=seed_item_id("potato"),
        inside_stand_px=(190, 118),
        inside_settle_frames=40,
    ),
    "grass": ShedShelfSpec(
        item_id=seed_item_id("grass"),
        inside_stand_px=(96, 118),
        inside_settle_frames=40,
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
    cx, cy = WEST_POCKET_PLANT_CENTER
    if abs(tile[0] - cx) <= 3 and abs(tile[1] - cy) <= 3:
        return "pocket_to_shed"
    if tile[1] >= 32:
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


def keep_selected_needs_swap(ram: np.ndarray, keep_id: int) -> bool:
    """True when shelf A would knock ``keep_id`` out of the selected slot."""
    sel = selected_tool(ram)
    bp = backpack_tool(ram)
    keep = int(keep_id)
    return sel == keep and bp not in (0, keep)


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
    take_attempts: int = 3

    _phase: str = field(default="settle", init=False)
    _step_count: int = field(default=0, init=False)
    _task: Optional[Task] = field(default=None, init=False)
    _action_queue: deque = field(default_factory=deque, init=False)
    _take_tries: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._phase = "settle"
        self._step_count = 0
        self._task = None
        self._action_queue.clear()
        self._take_tries = 0

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
        if input_lock != 1 and self._phase != "take":
            from harvest.tasks.primitives import dismiss_dialogue_result

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

        if self._phase == "take":
            if tool_in_carry_pair(world.ram, self.tool_id):
                return TaskResult(status=TaskStatus.SUCCESS, reason=f"tool 0x{self.tool_id:02X} ready")
            queued = drain_action_queue(self._action_queue)
            if queued is not None:
                return queued
            if tool_in_carry_pair(world.ram, self.tool_id):
                return TaskResult(status=TaskStatus.SUCCESS, reason=f"tool 0x{self.tool_id:02X} ready")
            if self._take_tries >= self.take_attempts:
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=f"tool 0x{self.tool_id:02X} missing after shelf",
                )
            if input_lock != 1:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            self._queue_take_tool()
            self._take_tries += 1
            queued = drain_action_queue(self._action_queue)
            if queued is not None:
                return queued
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        return TaskResult(status=TaskStatus.FAILURE, reason=f"unknown shed shelf phase {self._phase}")


__all__ = [
    "RecordingSliceSpec",
    "SHED_SEED_SPECS",
    "SHED_TOOL_SPECS",
    "ShedSeedSpec",
    "ShedShelfSpec",
    "ShedShelfToolTask",
    "ShedToolSpec",
    "SwapCarrySlotsTask",
    "keep_selected_needs_swap",
    "load_recording_slice",
    "shed_enter_transition",
    "shed_farm_route_name",
]

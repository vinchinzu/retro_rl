"""Shed fetch/ensure tasks. Shelf math lives in :mod:`shed_slots`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from retro_harness import Task, TaskResult, TaskStatus, WorldState
from harvest.core.carry import (
    ADDR_TOOL_BACKPACK,
    ADDR_TOOL_SELECTED,
    seed_item_id,
    tool_in_carry_pair,
)
from harvest.core.tile_catalog import Tool
from harvest.tasks.nav import (
    Point,
    get_pos_from_ram,
    get_tile_at,
    TILE_SIZE,
)
from harvest.core.tile_catalog import (
    ADDR_TILEMAP,
    STALE_TILE_IDS,
)
from harvest.maps.map_config import ROUTES
from harvest.planner.day_plan_status import (
    TASKS_DIR,
    SHED_TILEMAP,
    ram_seed_count,
    is_farm_tilemap,
)
from harvest.core.ram_catalog import read_ram_u8, read_ram_value
from harvest.planner.tasks.navigation import MultiMapNavTask, NavTask
from harvest.planner.tasks.inventory_common import (
    farm_free_move_ready,
    outdoor_intro_flags_ready,
    farm_house_front_softlock,
)
from harvest.planner.tasks.inventory_exit import ExitToFarmTask
from harvest.tasks.farm_ops import LOADED_FARM_STAND, SHED_DOOR_TILE
from harvest.planner.tasks.shed_slots import (
    SHED_SEED_SPECS,
    SHED_TOOL_SPECS,
    RecordingSliceSpec,
    ShedSeedSpec,
    ShedShelfSpec,
    ShedShelfToolTask,
    ShedToolSpec,
    SwapCarrySlotsTask,
    keep_selected_needs_swap,
    load_recording_slice,
    shed_enter_transition,
    shed_farm_route_name,
)

__all__ = [
    "RecordingSliceSpec",
    "ShedShelfSpec",
    "ShedToolSpec",
    "ShedSeedSpec",
    "SHED_TOOL_SPECS",
    "SHED_SEED_SPECS",
    "shed_farm_route_name",
    "shed_enter_transition",
    "load_recording_slice",
    "SwapCarrySlotsTask",
    "ShedShelfToolTask",
    "ShedFetchItemTask",
    "EnsureCarryToolTask",
    "EnsureAnimalToolsTask",
    "EnsureCropSeedsTask",
]


@dataclass
class ShedFetchItemTask(Task):
    """Fetch one shed-shelf item into the carry pair and leave the shed.

    Shared ladder: exit_to_farm → approach route → enter → shelf → exit_to_farm.
    X cannot pull shelf bags; this task is the only way to equip them.
    """

    name: str = "shed_fetch_item"
    item_id: int = int(Tool.WATERING_CAN)
    shelf: Optional[ShedShelfSpec] = None
    tasks_dir: str = TASKS_DIR
    nav_timeout: int = 8000
    enter_timeout: int = 1500
    # When True, success leaves the player on the farm (not inside the shed).
    exit_when_done: bool = True

    _phase: str = field(default="start", init=False)
    _task: Optional[Task] = field(default=None, init=False)
    _failed_reason: str = field(default="", init=False)
    _farm_frames: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._phase = "start"
        self._task = None
        self._failed_reason = ""
        self._farm_frames = 0

    def can_start(self, world: WorldState) -> bool:
        return True

    def _spec(self) -> Optional[ShedShelfSpec]:
        if self.shelf is not None:
            return self.shelf
        return SHED_TOOL_SPECS.get(self.item_id)

    def _inside_task(self, spec: ShedShelfSpec) -> Task:
        if spec.inside_recording is not None:
            return load_recording_slice(spec.inside_recording, self.tasks_dir)
        return ShedShelfToolTask(
            name=f"shed_shelf_0x{self.item_id:02X}",
            tool_id=self.item_id,
            stand_px=spec.inside_stand_px,
            face=spec.inside_face,
            settle_frames=spec.inside_settle_frames,
            timeout=spec.inside_timeout,
        )

    def _activate(self, phase: str, task: Task, world: WorldState) -> TaskResult:
        self._phase = phase
        self._task = task
        task.reset(world)
        return task.step(world)

    def _item_ready(self, ram: np.ndarray) -> bool:
        return tool_in_carry_pair(ram, self.item_id)

    def _needs_leave_shed_door(self, ram: np.ndarray) -> bool:
        """True on (26,30) 0xFF / unloaded farm after ExitToFarm."""
        from harvest.tasks.farm_clear_quota import farm_map_loaded

        pos = get_pos_from_ram(ram)
        tile = (pos.x // TILE_SIZE, pos.y // TILE_SIZE)
        if tile == SHED_DOOR_TILE:
            return True
        if int(get_tile_at(ram, *tile)) in STALE_TILE_IDS:
            return True
        return is_farm_tilemap(
            int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
        ) and not farm_map_loaded(ram)

    def _leave_door_task(self) -> NavTask:
        stand = LOADED_FARM_STAND
        return NavTask(
            name=f"leave_shed_door_0x{self.item_id:02X}",
            target_px=Point(stand[0] * TILE_SIZE + 8, stand[1] * TILE_SIZE + 8),
            radius=14,
            timeout=600,
        )

    def _finish_ready(self, world: WorldState, reason: str) -> TaskResult:
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if self.exit_when_done and tilemap == SHED_TILEMAP:
            return self._activate("exit_after_ready", ExitToFarmTask(tasks_dir=self.tasks_dir), world)
        if self.exit_when_done and self._needs_leave_shed_door(world.ram):
            return self._activate("leave_door", self._leave_door_task(), world)
        return TaskResult(status=TaskStatus.SUCCESS, reason=reason)

    def _start_next_phase(self, world: WorldState) -> TaskResult:
        if self._item_ready(world.ram):
            return self._finish_ready(world, f"tool 0x{self.item_id:02X} ready")

        spec = self._spec()
        if spec is None:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"no shed shelf plan for 0x{self.item_id:02X}",
            )

        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if tilemap == SHED_TILEMAP:
            return self._activate("inside", self._inside_task(spec), world)
        if is_farm_tilemap(tilemap):
            farm_route = shed_farm_route_name(world.ram, spec.farm_route)
            if farm_route:
                waypoints = ROUTES.get(farm_route, [])
                if not waypoints:
                    return TaskResult(
                        status=TaskStatus.FAILURE,
                        reason=f"missing shed farm route {farm_route}",
                    )
                return self._activate(
                    "route",
                    MultiMapNavTask(
                        name=f"route_shed_0x{self.item_id:02X}",
                        waypoints=list(waypoints),
                        timeout=self.nav_timeout,
                        initial_settle_frames=0,
                    ),
                    world,
                )
            return self._activate(
                "nav",
                NavTask(
                    name=f"nav_shed_0x{self.item_id:02X}",
                    target_px=Point(spec.nav_target_px[0], spec.nav_target_px[1]),
                    radius=spec.nav_radius,
                    timeout=self.nav_timeout,
                ),
                world,
            )

        return self._activate("exit_to_farm", ExitToFarmTask(tasks_dir=self.tasks_dir), world)

    def _control_lost_result(self, world: WorldState) -> TaskResult:
        gs = int(read_ram_value(world.ram, "game_state", raw=True))
        try:
            f68 = int(read_ram_value(world.ram, "event_flags_1f68", raw=True))
        except Exception:
            f68 = -1
        pos = get_pos_from_ram(world.ram)
        intro = outdoor_intro_flags_ready(world.ram)
        return TaskResult(
            status=TaskStatus.FAILURE,
            reason=(
                f"farm_control_lost gs=0x{gs:04X} f1f68=0x{f68:04X} "
                f"intro_ok={intro} pos=({pos.x},{pos.y}) "
                f"(post-truck D1 ExitToFarm: missing outdoor-intro flags "
                f"0x00A1 fires CODE_83CEAE → free-move clear → house-front "
                f"soft-lock → tilemap 0x5F; Y1 has 0x00B1)"
            ),
        )

    def step(self, world: WorldState) -> TaskResult:
        if self._task is None:
            return self._start_next_phase(world)

        # While routing on farm after house exit, abort if free-move is gone.
        if self._phase in {"route", "nav", "exit_to_farm"}:
            tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
            if is_farm_tilemap(tilemap):
                self._farm_frames += 1
                if farm_house_front_softlock(world.ram) or (
                    not farm_free_move_ready(world.ram) and self._farm_frames > 120
                ):
                    return self._control_lost_result(world)

        result = self._task.step(world)
        if result.status == TaskStatus.RUNNING:
            return result

        if self._phase == "leave_door":
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=f"tool 0x{self.item_id:02X} ready",
            )

        if result.status == TaskStatus.FAILURE:
            tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
            if self._phase in {"route", "nav", "enter"} and tilemap == SHED_TILEMAP:
                spec = self._spec()
                if spec is not None:
                    return self._activate("inside", self._inside_task(spec), world)
            reason = result.reason or "unknown"
            if self._phase == "exit_after_ready":
                # Item is equipped; leave farm even if exit path was messy.
                return TaskResult(
                    status=TaskStatus.SUCCESS,
                    reason=f"tool 0x{self.item_id:02X} ready; shed exit failed: {reason}",
                )
            if self._phase == "exit_after_failure":
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=self._failed_reason or reason,
                )
            if tilemap == SHED_TILEMAP and self._phase not in {
                "exit_after_failure",
                "exit_after_ready",
            }:
                self._failed_reason = f"{self._phase} failed: {reason}"
                return self._activate(
                    "exit_after_failure",
                    ExitToFarmTask(tasks_dir=self.tasks_dir),
                    world,
                )
            return TaskResult(status=TaskStatus.FAILURE, reason=f"{self._phase} failed: {reason}")

        # Sub-task succeeded.
        if self._phase == "exit_after_ready":
            return self._finish_ready(world, f"tool 0x{self.item_id:02X} ready")

        if self._phase == "exit_after_failure":
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=self._failed_reason or "shed fetch failed",
            )

        if self._phase == "exit_to_farm":
            if is_farm_tilemap(
                int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
            ) and not farm_free_move_ready(world.ram):
                # Do not start MultiMapNav without free-move — soft-locks to 0x5F.
                return self._control_lost_result(world)
            self._task = None
            return self._start_next_phase(world)

        if self._phase in {"route", "nav"}:
            return self._activate(
                "enter",
                shed_enter_transition(
                    name=f"enter_shed_0x{self.item_id:02X}",
                    timeout=self.enter_timeout,
                ),
                world,
            )

        if self._phase == "enter":
            spec = self._spec()
            if spec is None:
                return TaskResult(status=TaskStatus.FAILURE, reason="missing shed shelf spec")
            return self._activate("inside", self._inside_task(spec), world)

        if self._phase == "inside":
            if self._item_ready(world.ram):
                return self._finish_ready(world, f"tool 0x{self.item_id:02X} ready")
            self._failed_reason = f"tool 0x{self.item_id:02X} missing after shed task"
            tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
            if tilemap == SHED_TILEMAP:
                return self._activate(
                    "exit_after_failure",
                    ExitToFarmTask(tasks_dir=self.tasks_dir),
                    world,
                )
            return TaskResult(status=TaskStatus.FAILURE, reason=self._failed_reason)

        return TaskResult(status=TaskStatus.FAILURE, reason=f"unknown phase {self._phase}")


@dataclass
class EnsureCarryToolTask(Task):
    """Ensure a desired shed tool is present in the 2-slot carry pair."""

    name: str = "ensure_carry_tool"
    tool_id: int = int(Tool.WATERING_CAN)
    tasks_dir: str = TASKS_DIR
    nav_timeout: int = 8000
    enter_timeout: int = 1500
    exit_when_done: bool = True

    _fetch: Optional[ShedFetchItemTask] = field(default=None, init=False)

    def reset(self, world: WorldState) -> None:
        self._fetch = ShedFetchItemTask(
            name=f"shed_fetch_0x{self.tool_id:02X}",
            item_id=self.tool_id,
            shelf=SHED_TOOL_SPECS.get(self.tool_id),
            tasks_dir=self.tasks_dir,
            nav_timeout=self.nav_timeout,
            enter_timeout=self.enter_timeout,
            exit_when_done=self.exit_when_done,
        )
        self._fetch.reset(world)

    def can_start(self, world: WorldState) -> bool:
        return True

    # Tests inspect phase / nested task for route selection.
    @property
    def _phase(self) -> str:
        return self._fetch._phase if self._fetch is not None else "start"

    @property
    def _task(self) -> Optional[Task]:
        return self._fetch._task if self._fetch is not None else None

    def step(self, world: WorldState) -> TaskResult:
        if self._fetch is None:
            self.reset(world)
        assert self._fetch is not None
        return self._fetch.step(world)


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
        return tool_in_carry_pair(ram, self.first_tool_id) and tool_in_carry_pair(
            ram, self.second_tool_id
        )

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

        missing = (
            self.first_tool_id
            if not tool_in_carry_pair(world.ram, self.first_tool_id)
            else self.second_tool_id
        )
        # Nested ensure already exits the shed; no double-exit.
        return self._activate(
            f"ensure_0x{missing:02X}",
            EnsureCarryToolTask(
                name=f"ensure_animal_tool_0x{missing:02X}",
                tool_id=missing,
                tasks_dir=self.tasks_dir,
                exit_when_done=True,
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
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=self._failed_reason or "animal tool prep failed",
            )
        if result.status == TaskStatus.FAILURE:
            reason = result.reason or "unknown"
            tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
            if tilemap == SHED_TILEMAP:
                self._failed_reason = f"{self._phase} failed: {reason}"
                return self._activate(
                    "exit_after_failure",
                    ExitToFarmTask(tasks_dir=self.tasks_dir),
                    world,
                )
            return TaskResult(status=TaskStatus.FAILURE, reason=f"{self._phase} failed: {reason}")
        self._last_reason = result.reason or ""
        self._task = None
        return self._start_next_phase(world)


@dataclass
class EnsureCropSeedsTask(Task):
    """Equip plantable crop seeds (and hoe) into the carry pair.

    Shop-bought bags sit on the tool-shed shelf until picked up with A. X only
    swaps the two carried slots — it never pulls shelf bags. For virgin plant
    work we also grab the hoe so CropWaterTask can till before seeding.

    Shelf A replaces the **selected** slot. Grabbing seeds while the hoe is
    selected (or hoe while seeds are selected) thrash-loops shed multi_nav
    forever (rr-6byj). Before each shelf grab we X-swap so the keep-tool is in
    the backpack and a disposable tool is selected. Hoe fetch stays inside
    the shed so potato-bag A is the same visit; exit after the pair is ready.
    """

    name: str = "ensure_crop_seeds"
    seed_type: str = "potato"
    tasks_dir: str = TASKS_DIR
    nav_timeout: int = 4000
    enter_timeout: int = 1500
    ensure_hoe: bool = True
    # Cap hoe/seed shed trips so a residual swap bug cannot hang the day plan.
    max_shed_trips: int = 4

    _phase: str = field(default="start", init=False)
    _task: Optional[Task] = field(default=None, init=False)
    _failed_reason: str = field(default="", init=False)
    _shed_trips: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._phase = "start"
        self._task = None
        self._failed_reason = ""
        self._shed_trips = 0

    def can_start(self, world: WorldState) -> bool:
        return True

    def _seed_item(self) -> int:
        return seed_item_id(self.seed_type)

    def _shed_spec(self) -> Optional[ShedShelfSpec]:
        return SHED_SEED_SPECS.get(self.seed_type)

    def _activate(self, phase: str, task: Task, world: WorldState) -> TaskResult:
        self._phase = phase
        self._task = task
        task.reset(world)
        return task.step(world)

    def _seeds_ready(self, ram: np.ndarray) -> bool:
        return tool_in_carry_pair(ram, self._seed_item())

    def _plant_tools_ready(self, ram: np.ndarray) -> bool:
        if not self._seeds_ready(ram):
            return False
        if not self.ensure_hoe:
            return True
        return tool_in_carry_pair(ram, int(Tool.HOE))

    def _finish_ready(self, world: WorldState, reason: str) -> TaskResult:
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if tilemap == SHED_TILEMAP:
            return self._activate("exit_after_ready", ExitToFarmTask(tasks_dir=self.tasks_dir), world)
        return TaskResult(status=TaskStatus.SUCCESS, reason=reason)

    def _maybe_swap_selected_off_keep(
        self, world: WorldState, keep_id: int, phase: str
    ) -> Optional[TaskResult]:
        """If ``keep_id`` is selected and backpack holds a real disposable, swap."""
        if keep_selected_needs_swap(world.ram, keep_id):
            return self._activate(phase, SwapCarrySlotsTask(), world)
        return None

    def _begin_shed_trip_or_fail(self, label: str) -> Optional[TaskResult]:
        self._shed_trips += 1
        if self._shed_trips > self.max_shed_trips:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=(
                    f"carry thrash after {self._shed_trips - 1} shed trips "
                    f"({label}); cannot hold hoe+seeds in 2-slot pair"
                ),
            )
        return None

    def _start_next_phase(self, world: WorldState) -> TaskResult:
        seed_item = self._seed_item()
        hoe_id = int(Tool.HOE)
        stored = ram_seed_count(world.ram, self.seed_type)

        if self._plant_tools_ready(world.ram):
            return self._finish_ready(world, f"seed tool 0x{seed_item:02X} ready")

        if stored <= 0 and not self._seeds_ready(world.ram):
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=f"no stored {self.seed_type} seeds",
            )

        # Hoe first so the 2-slot pair ends as seeds+hoe after the seed grab.
        # Stay in the shed (exit_when_done=False) so the seed shelf is the
        # same visit; two outdoor trips were the rr-6byj / settle-timeout path.
        if self.ensure_hoe and not tool_in_carry_pair(world.ram, hoe_id):
            # If seeds are already selected, swap so the seed bag stays when
            # the hoe shelf replaces the disposable selected slot.
            if self._seeds_ready(world.ram):
                swapped = self._maybe_swap_selected_off_keep(
                    world, seed_item, "swap_preserve_seed"
                )
                if swapped is not None:
                    return swapped
            trip_fail = self._begin_shed_trip_or_fail("ensure_hoe")
            if trip_fail is not None:
                return trip_fail
            return self._activate(
                "ensure_hoe",
                EnsureCarryToolTask(
                    name=f"ensure_hoe_for_{self.seed_type}",
                    tool_id=hoe_id,
                    tasks_dir=self.tasks_dir,
                    nav_timeout=self.nav_timeout,
                    enter_timeout=self.enter_timeout,
                    exit_when_done=False,
                ),
                world,
            )

        if self._seeds_ready(world.ram):
            return self._finish_ready(world, f"seed tool 0x{seed_item:02X} ready")

        # Stock exists but bag is not carried: only the shelf can equip it.
        # Preserve hoe in backpack before seed shelf A (selected = disposable).
        if self.ensure_hoe and tool_in_carry_pair(world.ram, hoe_id):
            swapped = self._maybe_swap_selected_off_keep(
                world, hoe_id, "swap_preserve_hoe"
            )
            if swapped is not None:
                return swapped

        spec = self._shed_spec()
        if spec is None:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=(
                    f"seed stock={stored} but bag 0x{seed_item:02X} has no shed plan"
                ),
            )
        trip_fail = self._begin_shed_trip_or_fail("fetch_seed")
        if trip_fail is not None:
            return trip_fail
        return self._activate(
            "fetch_seed",
            ShedFetchItemTask(
                name=f"shed_fetch_seed_{self.seed_type}",
                item_id=seed_item,
                shelf=spec,
                tasks_dir=self.tasks_dir,
                nav_timeout=self.nav_timeout,
                enter_timeout=self.enter_timeout,
                exit_when_done=True,
            ),
            world,
        )

    def step(self, world: WorldState) -> TaskResult:
        if self._task is None:
            return self._start_next_phase(world)

        result = self._task.step(world)
        if result.status == TaskStatus.RUNNING:
            return result

        if result.status == TaskStatus.FAILURE:
            reason = result.reason or "unknown"
            if self._phase == "exit_after_ready":
                return TaskResult(
                    status=TaskStatus.SUCCESS,
                    reason=f"seed tool 0x{self._seed_item():02X} ready; shed exit failed: {reason}",
                )
            if self._phase == "exit_after_failure":
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=self._failed_reason or reason,
                )
            tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
            if tilemap == SHED_TILEMAP and self._phase not in {
                "exit_after_failure",
                "exit_after_ready",
                "swap_preserve_seed",
                "swap_preserve_hoe",
            }:
                self._failed_reason = f"{self._phase} failed: {reason}"
                return self._activate(
                    "exit_after_failure",
                    ExitToFarmTask(tasks_dir=self.tasks_dir),
                    world,
                )
            return TaskResult(status=TaskStatus.FAILURE, reason=f"{self._phase} failed: {reason}")

        if self._phase == "exit_after_ready":
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=f"seed tool 0x{self._seed_item():02X} ready",
            )

        if self._phase == "exit_after_failure":
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=self._failed_reason or "seed equip failed",
            )

        # Nested fetch / ensure_hoe / preserve-swap finished — re-evaluate.
        self._task = None
        return self._start_next_phase(world)

    # Test helpers that used to call internal shed-route selection directly.
    def _start_shed_seed_fetch(self, world: WorldState) -> TaskResult:
        """Activate seed shelf fetch (tests / debug)."""
        spec = self._shed_spec()
        if spec is None:
            stored = ram_seed_count(world.ram, self.seed_type)
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=(
                    f"seed stock={stored} but bag 0x{self._seed_item():02X} has no shed plan"
                ),
            )
        return self._activate(
            "fetch_seed",
            ShedFetchItemTask(
                name=f"shed_fetch_seed_{self.seed_type}",
                item_id=self._seed_item(),
                shelf=spec,
                tasks_dir=self.tasks_dir,
                nav_timeout=self.nav_timeout,
                enter_timeout=self.enter_timeout,
                exit_when_done=True,
            ),
            world,
        )

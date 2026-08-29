"""Fence post handling: detection + toss into pond via recorded task."""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.core.animal_status import read_held_item
from harvest.core.tile_catalog import ADDR_INPUT_LOCK, DebrisType
from harvest.maps.map_config import FARM_POND_ACCESS_FENCE_ROW
from harvest.tasks.carry_toss import CarryToPondStand
from harvest.tasks.farm_ops import Target, TileScanner
from harvest.tasks.fence_corridor import (
    ACTION_CARRYING_BIT,
    ACTION_DROPPING,
    ADDR_PLAYER_ACTION,
    ADDR_PLAYER_STATE,
    POND_NO_GO_TILES,
    POND_TILES,
    choose_corridor_stage,
    corridor_after_lift,
    debug_fence_map,
    lift_actions,
    local_drop_actions,
    local_drop_faces,
    nearest_pond,
    pick_fence_target,
    scan_fence_targets,
    sort_fence_targets,
    south_charge_actions,
    weed_tiles,
)
from harvest.tasks.nav import (
    VIEWPORT_HOP_TILES,
    Navigator,
    Pathfinder,
    get_tile_at,
    make_action,
)
from harvest.tasks.recorded_task import RecordedTask


@dataclass
class FencePostTossTask(Task):
    name: str = "toss_fence_pond"
    fallback_task: str = "toss_bush_pond"
    _task: Optional[RecordedTask] = None

    def reset(self, world: WorldState) -> None:
        if self._task is None:
            self._task = self._load_task()
        self._task.reset(world)

    def can_start(self, world: WorldState) -> bool:
        try:
            if self._task is None:
                self._task = self._load_task()
            return True
        except FileNotFoundError:
            return False

    def step(self, world: WorldState) -> TaskResult:
        if self._task is None:
            return TaskResult(status=TaskStatus.BLOCKED, reason="recorded toss task missing")
        return self._task.step(world)

    def _load_task(self) -> RecordedTask:
        try:
            return RecordedTask.load(self.name)
        except FileNotFoundError:
            return RecordedTask.load(self.fallback_task)


@dataclass
class FenceClearLoopTask(Task):
    """Find fences, navigate, pick up, and toss into pond repeatedly."""

    name: str = "clear_fences"
    toss_task_name: str = "toss_fence_pond"
    max_fences: Optional[int] = 3
    max_steps_per_fence: int = 2400
    stasis_repath: int = 180
    max_failures: int = 3
    debug: bool = False
    debug_interval: int = 300
    # When True (empty-can pond corridor): after one successful lift+local_drop
    # the y=31 gap is open — return SUCCESS without requiring pond toss. ROM
    # soft-blocks south transit while standing on the just-lifted gap tile with
    # a false BFS path through (x,32); thrashing navigate_pond burns the day.
    corridor_only: bool = False
    # Leftover smash dumps stones with the same pond toss. Default is posts.
    debris_types: tuple = (DebrisType.FENCE,)
    # Leftover smash: dump every post/stone in a pond. Do not treat a local
    # drop as a clear, skip a stuck target, and work the y=31 wall first.
    pond_dump: bool = False
    # D2 leftover chunks clip the scan so a last distant stone cannot stall
    # the whole farm. Inclusive (x0, y0, x1, y1); None is the full map.
    farm_bounds: Optional[tuple[int, int, int, int]] = None

    _scanner: TileScanner = field(default_factory=TileScanner, init=False)
    _pathfinder: Pathfinder = field(init=False)
    _navigator: Navigator = field(init=False)
    _toss_task: Optional[RecordedTask] = None
    _state: str = "scan"
    _current: Optional[Target] = None
    _approach_tile: Optional[tuple[int, int]] = None
    _action_queue: deque = field(default_factory=deque, init=False)
    _steps_on_fence: int = 0
    _total_steps: int = 0
    _failures: int = 0
    cleared_count: int = 0
    _corridor_staged: bool = field(default=False, init=False)
    _corridor_stage: Optional[tuple[int, int]] = field(default=None, init=False)
    _skip_tiles: set = field(default_factory=set, init=False)
    _pond_carry: CarryToPondStand = field(init=False)

    def __post_init__(self):
        self._pathfinder = Pathfinder(self._scanner)
        self._navigator = Navigator(self._pathfinder)
        self._pond_carry = CarryToPondStand(
            stasis_repath=self.stasis_repath,
            debug=self.debug,
        )
        self._pathfinder.no_go_tiles.update(POND_NO_GO_TILES)

    def reset(self, world: WorldState) -> None:
        if os.getenv("FENCE_DEBUG", "").lower() in ("1", "true", "yes"):
            self.debug = True
        self._state = "scan"
        self._current = None
        self._approach_tile = None
        self._action_queue.clear()
        self._steps_on_fence = 0
        self._total_steps = 0
        self._failures = 0
        self._pond_hop_steps = 0
        self._corridor_charge_done = False
        self._local_drop_cycles = 0
        self.cleared_count = 0
        self._corridor_staged = False
        self._corridor_stage = None
        self._skip_tiles = set()
        self._stasis_repaths = 0
        self._toss_face = "up"
        self._pond_carry.debug = self.debug
        self._pond_carry.reset(world)
        if self._toss_task is None:
            self._toss_task = RecordedTask.load(self.toss_task_name)
            non_zero = sum(1 for f in self._toss_task.frames if any(v != 0 for v in f))
            if non_zero < len(self._toss_task.frames) * 0.1:
                print(
                    f"[FENCE] Warning: {self.toss_task_name} appears dead or nearly empty "
                    f"({non_zero}/{len(self._toss_task.frames)} non-zero frames)"
                )
        self._navigator.update(world.ram)

    def can_start(self, world: WorldState) -> bool:
        try:
            if self._toss_task is None:
                self._toss_task = RecordedTask.load(self.toss_task_name)
            return True
        except FileNotFoundError:
            return False

    def _skip_current(self, reason: str) -> TaskResult:
        if self._current is not None:
            self._skip_tiles.add(
                (int(self._current.tile[0]), int(self._current.tile[1]))
            )
        self._failures += 1
        self._state = "scan"
        self._current = None
        self._approach_tile = None
        self._steps_on_fence = 0
        self._navigator.path = []
        self._corridor_charge_done = False
        self._action_queue.clear()
        self._pathfinder.temp_blocked.clear()
        if self._failures >= self.max_failures:
            return TaskResult(status=TaskStatus.FAILURE, reason="too many fence failures")
        return TaskResult(status=TaskStatus.RUNNING, reason=reason)

    def _arm_south_charge(
        self, current, reason: str = "pond south charge"
    ) -> TaskResult:
        self._corridor_charge_done = True
        self._action_queue.clear()
        self._action_queue.extend(south_charge_actions())
        if self.debug:
            if reason == "corridor_only south charge":
                print(f"[FENCE] corridor_only: carry-south charge at {current}")
            else:
                print(f"[FENCE] south charge at {current}")
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(self._action_queue.popleft()),
            reason=reason,
        )

    def _mark_pond_toss(self) -> None:
        self.cleared_count += 1
        self._failures = 0
        self._skip_tiles.clear()
        self._corridor_charge_done = False
        self._pond_hop_steps = 0

    def _finish_pond_carry(self, world: WorldState) -> TaskResult:
        self._mark_pond_toss()
        self._state = "scan"
        self._current = None
        self._approach_tile = None
        self._steps_on_fence = 0
        self._pond_carry.reset(world)
        return TaskResult(
            status=TaskStatus.RUNNING,
            reason=f"pond dump complete cleared={self.cleared_count}",
        )

    def _idle(self) -> TaskResult:
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(np.zeros(12, dtype=np.int32)),
        )

    def _begin_local_drop(self, reason: str) -> TaskResult:
        self._state = "local_drop"
        self._steps_on_fence = 0
        self._pond_hop_steps = 0
        return TaskResult(status=TaskStatus.RUNNING, reason=reason)

    def _step_stage_corridor(self, world: WorldState) -> TaskResult:
        stage = self._corridor_stage
        if stage is None:
            self._state = "scan"
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action()),
                reason="corridor stage unavailable",
            )
        if self._navigator.at_tile(stage):
            self._corridor_staged = True
            self._navigator.path = []
            self._state = "scan"
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action()),
                reason="corridor staged west",
            )
        if self._navigator.current_tile == stage:
            action = self._navigator.center_on_tile(stage, tolerance=2)
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(action if action is not None else make_action()),
                reason="centering corridor stage",
            )
        action = self._navigator.follow_path(world.ram)
        if action is None:
            self._navigator.path = self._pathfinder.find_path(
                world.ram, self._navigator.current_tile, stage
            ) or []
            action = self._navigator.follow_path(world.ram)
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(action if action is not None else make_action()),
            reason="staging west of corridor",
        )

    def _maybe_stage_corridor(self, world: WorldState) -> None:
        if not (
            self.corridor_only
            and not self._corridor_staged
            and self._state == "scan"
            and self._navigator.current_tile[1] < FARM_POND_ACCESS_FENCE_ROW
        ):
            return
        self._pathfinder.no_go_tiles.update(weed_tiles(self._scanner, world.ram))
        player = self._navigator.current_tile
        stage, path = choose_corridor_stage(
            self._pathfinder, world.ram, player, self._corridor_stage
        )
        self._corridor_stage = stage
        if stage is not None and self._navigator.at_tile(stage):
            self._corridor_staged = True
            self._navigator.path = []
        elif stage is not None and path:
            self._navigator.path = path
            self._state = "stage_corridor"
            if self.debug:
                print(
                    f"[FENCE] corridor stage {self._navigator.current_tile} → {stage}"
                )

    def _step_scan(self, world: WorldState) -> Optional[TaskResult]:
        if world.ram[ADDR_PLAYER_STATE] & ACTION_CARRYING_BIT:
            self._state = "navigate_pond"
            self._steps_on_fence = 0
            return TaskResult(status=TaskStatus.RUNNING)

        if self.corridor_only:
            self._pathfinder.no_go_tiles.update(weed_tiles(self._scanner, world.ram))
        targets = scan_fence_targets(
            world.ram,
            self.debris_types,
            self.farm_bounds,
            scanner=self._scanner,
            corridor_only=self.corridor_only,
        )
        if not targets:
            reason = "corridor already open" if self.corridor_only else "no fences found"
            return TaskResult(status=TaskStatus.SUCCESS, reason=reason)
        sort_fence_targets(
            targets,
            pond_dump=self.pond_dump,
            skip_tiles=self._skip_tiles,
            player_pos=self._navigator.current_pos,
        )
        pick = pick_fence_target(
            targets,
            ram=world.ram,
            pathfinder=self._pathfinder,
            player_tile=self._navigator.current_tile,
            player_pos=self._navigator.current_pos,
            skip_tiles=self._skip_tiles,
            corridor_only=self.corridor_only,
            pond_dump=self.pond_dump,
            debug=self.debug,
        )
        if pick is not None:
            target, approach, path = pick
            self._current = target
            self._approach_tile = approach
            self._navigator.path = path
            self._state = "navigate"
            self._steps_on_fence = 0
            self._pond_hop_steps = 0
            self._corridor_charge_done = False
            self._local_drop_cycles = 0
            self._stasis_repaths = 0
            self._toss_face = "up"
            return None
        if self._skip_tiles:
            self._skip_tiles.clear()
            self._pathfinder.temp_blocked.clear()
            return TaskResult(status=TaskStatus.RUNNING, reason="retry skipped fences")
        return TaskResult(status=TaskStatus.FAILURE, reason="no reachable fence")

    def _step_navigate(self, world: WorldState) -> TaskResult:
        if not self._current or not self._approach_tile:
            self._state = "scan"
            return self._idle()
        if world.ram[ADDR_PLAYER_STATE] & ACTION_CARRYING_BIT:
            self._state = "navigate_pond"
            return TaskResult(status=TaskStatus.RUNNING)
        if get_tile_at(world.ram, *self._current.tile) != self._current.tile_id:
            self._state = "scan"
            return self._idle()

        if self._navigator.current_tile == self._approach_tile or self._navigator.at_tile(
            self._approach_tile
        ):
            action = self._navigator.center_on_tile(self._approach_tile, tolerance=1)
            if action is None:
                self._state = "lift"
                return TaskResult(status=TaskStatus.RUNNING)
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

        if self._navigator.stasis > self.stasis_repath or not self._navigator.path:
            if self._navigator.path:
                self._pathfinder.temp_blocked.add(self._navigator.path[0])
            path = self._pathfinder.find_path(
                world.ram,
                self._navigator.current_tile,
                self._approach_tile,
                max_steps=VIEWPORT_HOP_TILES,
            )
            reached = bool(
                path is not None
                and (
                    not path
                    or path[-1] == self._approach_tile
                    or self._navigator.current_tile == self._approach_tile
                )
            )
            if path and reached:
                self._navigator.path = path
                if self._navigator.stasis > self.stasis_repath:
                    self._stasis_repaths = getattr(self, "_stasis_repaths", 0) + 1
                    if self._stasis_repaths >= 4:
                        return self._skip_current("stasis repath cap")
            elif path and self._navigator.stasis <= self.stasis_repath:
                self._navigator.path = path
            elif self._navigator.stasis > self.stasis_repath:
                return self._skip_current("repath failed, skipping")

        action = self._navigator.follow_path(world.ram)
        if action is None:
            if self._navigator.stasis > self.stasis_repath:
                return self._skip_current("idle at approach, skipping")
            action = np.zeros(12, dtype=np.int32)
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

    def _step_corridor_carry(self, current) -> Optional[TaskResult]:
        nxt = corridor_after_lift(
            current, charge_done=getattr(self, "_corridor_charge_done", False)
        )
        if nxt == "drop_south":
            if self.debug:
                print(f"[FENCE] corridor_only: crossed to {current}")
            return self._begin_local_drop("corridor_only drop south of wall")
        if nxt == "south_charge":
            return self._arm_south_charge(current, reason="corridor_only south charge")
        if self.debug:
            print(
                f"[FENCE] corridor_only: local drop at {current} "
                f"after failed carry-south (gap open)"
            )
        return self._begin_local_drop("corridor_only local drop")

    def _step_navigate_pond(self, world: WorldState) -> TaskResult:
        state_val = world.ram[ADDR_PLAYER_STATE]
        if not (state_val & ACTION_CARRYING_BIT):
            if self.debug:
                print(
                    f"[FENCE] navigate_pond called but not carrying! "
                    f"(state=0x{state_val:02x})"
                )
            self._state = "scan"
            return self._idle()

        if self.pond_dump:
            result = self._pond_carry.step(world)
            if result.status == TaskStatus.SUCCESS:
                return self._finish_pond_carry(world)
            return result

        current = self._navigator.current_tile
        best_pond = nearest_pond(current)

        if self.corridor_only:
            corridor = self._step_corridor_carry(current)
            if corridor is not None:
                return corridor

        # ROM trap: BFS invents a path through (x,32) after lift, but game
        # physics soft-blocks south transit while standing on the gap tile.
        if current[1] == 31 and self._navigator.stasis > 90:
            if self.debug:
                print(
                    f"[FENCE] gap-tile stasis at {current} "
                    f"(stasis={self._navigator.stasis}); local drop"
                )
            self._navigator.path = []
            return self._begin_local_drop("local drop after gap soft-block")

        if current == best_pond or self._navigator.at_tile(best_pond):
            action = self._navigator.center_on_tile(best_pond, tolerance=1)
            if action is None:
                self._toss_face = "up"
                self._state = "toss"
                self._steps_on_fence = 0
                if self.debug:
                    print(f"[FENCE] reached pond toss gap at {current}. TOSSING!")
                return TaskResult(status=TaskStatus.RUNNING)
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

        overrides = {best_pond}
        if self._current:
            gx, gy = self._current.tile
            overrides.add((gx, gy))
            overrides.add((gx, gy + 1))

        hop = self._pathfinder.find_path(
            world.ram,
            current,
            best_pond,
            walkable_override=overrides,
            max_steps=VIEWPORT_HOP_TILES,
        )
        path = None
        if hop:
            hop_end = hop[-1]
            cur_dist = abs(current[0] - best_pond[0]) + abs(current[1] - best_pond[1])
            hop_dist = abs(hop_end[0] - best_pond[0]) + abs(hop_end[1] - best_pond[1])
            if hop_dist < cur_dist:
                path = hop
                self._pond_hop_steps = getattr(self, "_pond_hop_steps", 0) + 1
        if path is not None and getattr(self, "_pond_hop_steps", 0) > 6:
            path = None

        if path:
            self._navigator.path = path
            if self.debug and self._total_steps % self.debug_interval == 0:
                print(
                    f"[FENCE] pond path len={len(path)} next={path[0]} "
                    f"target={best_pond} override="
                    f"{self._current.tile if self._current else None}"
                )
            action = self._navigator.follow_path(world.ram)
            if action is None:
                action = np.zeros(12, dtype=np.int32)
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

        if self.debug:
            print(
                f"[FENCE] pond {best_pond} unreachable from {current}; "
                f"local drop (gap open for refill)"
            )
        return self._begin_local_drop("local drop after pond unreachable")

    def _step_local_drop(self, world: WorldState) -> TaskResult:
        state_val = world.ram[ADDR_PLAYER_STATE]
        if (
            not (state_val & ACTION_CARRYING_BIT)
            and read_held_item(world.ram) == 0
            and not self._action_queue
        ):
            origin = (
                (int(self._current.tile[0]), int(self._current.tile[1]))
                if self._current is not None
                else None
            )
            self._state = "scan"
            self._current = None
            self._approach_tile = None
            self._steps_on_fence = 0
            self._corridor_charge_done = False
            if self.pond_dump:
                if origin is not None:
                    self._skip_tiles.add(origin)
                if self.debug:
                    print("[FENCE] pond_dump recovery drop (not a toss)")
                return TaskResult(
                    status=TaskStatus.RUNNING, reason="pond_dump recovery drop"
                )
            self.cleared_count += 1
            if self.debug:
                print(f"[FENCE] local drop complete cleared={self.cleared_count}")
            if self.corridor_only and (
                self.max_fences is None or self.cleared_count >= self.max_fences
            ):
                return TaskResult(
                    status=TaskStatus.SUCCESS,
                    reason=f"corridor open cleared={self.cleared_count}",
                )
            return TaskResult(status=TaskStatus.RUNNING, reason="local drop done")
        if not self._action_queue:
            attempts = getattr(self, "_local_drop_cycles", 0)
            current = self._navigator.current_tile
            if self.corridor_only and attempts >= 6:
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=(
                        "corridor open but hands not clear "
                        f"held=0x{read_held_item(world.ram):02X}"
                    ),
                )
            self._local_drop_cycles = attempts + 1
            faces = local_drop_faces(
                corridor_only=self.corridor_only,
                tile_y=current[1],
                attempts=attempts,
            )
            self._action_queue.extend(local_drop_actions(faces))
        action = self._action_queue.popleft()
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

    def _step_lift(self) -> TaskResult:
        if not self._current:
            self._state = "scan"
            return self._idle()
        player = self._navigator.current_tile
        target = self._current.tile
        dx, dy = target[0] - player[0], target[1] - player[1]
        direction = (
            "right"
            if abs(dx) >= abs(dy) and dx > 0
            else "left"
            if abs(dx) >= abs(dy)
            else "down"
            if dy > 0
            else "up"
        )
        self._action_queue.extend(lift_actions(direction))
        self._state = "verify_lift"
        if self.debug:
            print(f"[FENCE] lift at {target}, direction={direction}, queueing verification")
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(self._action_queue.popleft()),
        )

    def _step_verify_lift(self, world: WorldState) -> TaskResult:
        if self._action_queue:
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(self._action_queue.popleft()),
            )
        state_val = world.ram[ADDR_PLAYER_STATE]
        dir_val = world.ram[0xDA] if 0xDA < len(world.ram) else -1
        if state_val & ACTION_CARRYING_BIT:
            if self.debug:
                print(
                    f"[FENCE] lift verified! (state=0x{state_val:02x}, dir={dir_val}) "
                    f"Transitioning to navigate_pond"
                )
            self._state = "navigate_pond"
            self._steps_on_fence = 0
        else:
            if self.debug:
                pos = self._navigator.current_tile
                print(
                    f"[FENCE] lift FAILED at {pos} "
                    f"(state=0x{state_val:02x}, dir={dir_val}). Returning to scan."
                )
            self._state = "scan"
            self._current = None
            self._approach_tile = None
            self._steps_on_fence = 0
            self._action_queue.extend([make_action() for _ in range(10)])
        return self._idle()

    def _step_toss(self, world: WorldState) -> TaskResult:
        state_val = world.ram[ADDR_PLAYER_STATE]
        if not (state_val & ACTION_CARRYING_BIT) and not self._action_queue:
            self._mark_pond_toss()
            self._state = "scan"
            self._current = None
            self._approach_tile = None
            self._steps_on_fence = 0
            if self.debug:
                print("[FENCE] toss complete (ram check)")
            return TaskResult(status=TaskStatus.RUNNING)

        if not self._action_queue:
            face = getattr(self, "_toss_face", "up") or "up"
            self._action_queue.extend([make_action(**{face: True}) for _ in range(10)])
            self._action_queue.extend(
                [make_action(**{face: True, "a": True}) for _ in range(15)]
            )
            self._action_queue.extend([make_action() for _ in range(10)])

        action = self._action_queue.popleft()
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

    def step(self, world: WorldState) -> TaskResult:
        self._navigator.update(world.ram)
        self._total_steps += 1

        if self.debug and self._total_steps % self.debug_interval == 0:
            cur = self._navigator.current_tile
            if self._state == "navigate_pond":
                tgt = nearest_pond(cur)
                app = None
            else:
                tgt = tuple(map(int, self._current.tile)) if self._current else None
                app = tuple(map(int, self._approach_tile)) if self._approach_tile else None
            debug_fence_map(
                world.ram, self._total_steps, self._state, cur, tgt, app
            )

        if (
            self.pond_dump
            and self._state == "navigate_pond"
            and not (world.ram[ADDR_PLAYER_STATE] & ACTION_CARRYING_BIT)
            and read_held_item(world.ram) == 0
        ):
            carry_result = self._pond_carry.step(world)
            if carry_result.status == TaskStatus.SUCCESS:
                if carry_result.reason == "pond toss complete":
                    return self._finish_pond_carry(world)
                self._state = "scan"
                self._current = None
                self._approach_tile = None
                self._steps_on_fence = 0
                self._pond_carry.reset(world)
            elif carry_result.action is not None:
                return carry_result

        input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
        if input_lock != 1:
            if self.corridor_only and (
                world.ram[ADDR_PLAYER_STATE] & ACTION_CARRYING_BIT
            ):
                if self._state not in ("local_drop",):
                    self._state = "local_drop"
                    self._steps_on_fence = 0
            action = (
                make_action(a=True)
                if (self._steps_on_fence % 2 == 0)
                else make_action(b=True)
            )
            return TaskResult(
                status=TaskStatus.RUNNING, action=ActionResult(action), reason="input_lock"
            )

        if self.max_fences is not None and self.cleared_count >= self.max_fences:
            return TaskResult(status=TaskStatus.SUCCESS)

        if (
            self.corridor_only
            and (world.ram[ADDR_PLAYER_STATE] & ACTION_CARRYING_BIT)
            and self._state == "scan"
        ):
            self._state = "navigate_pond"
            self._steps_on_fence = 0

        self._steps_on_fence += 1
        if self._steps_on_fence > self.max_steps_per_fence:
            carrying = bool(world.ram[ADDR_PLAYER_STATE] & ACTION_CARRYING_BIT)
            if carrying and self.pond_dump:
                self._steps_on_fence = 0
                self._pond_hop_steps = 0
                self._pathfinder.temp_blocked.clear()
                self._state = "navigate_pond"
                return TaskResult(
                    status=TaskStatus.RUNNING, reason="pond_dump keep carrying"
                )
            if carrying:
                self._state = "local_drop"
                self._steps_on_fence = 0
                self._action_queue.clear()
                return TaskResult(status=TaskStatus.RUNNING, reason="timeout drop")
            return self._skip_current("fence timeout, skipping")

        if self._action_queue:
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(self._action_queue.popleft()),
            )

        self._maybe_stage_corridor(world)
        if self._state == "stage_corridor":
            return self._step_stage_corridor(world)
        if self._state == "scan":
            scanned = self._step_scan(world)
            if scanned is not None:
                return scanned
        if self._state == "navigate":
            return self._step_navigate(world)
        if self._state == "navigate_pond":
            return self._step_navigate_pond(world)
        if self._state == "local_drop":
            return self._step_local_drop(world)
        if self._state == "lift":
            return self._step_lift()
        if self._state == "verify_lift":
            return self._step_verify_lift(world)
        if self._state == "toss":
            return self._step_toss(world)
        return TaskResult(status=TaskStatus.FAILURE, reason="invalid state")

"""Fence post handling: detection + toss into pond via recorded task."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

import numpy as np

from harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from recorded_task import RecordedTask
from farm_clearer import (
    TileScanner,
    Pathfinder,
    Navigator,
    DebrisType,
    Target,
    make_action,
    TILE_SIZE,
    get_tile_at,
    manhattan,
    ADDR_INPUT_LOCK,
    WALKABLE_TILES,
    POND_CHARACTERISTIC_TILES,
)

POND_TILES = [(32, 34), (33, 34)] # X=32, 33 is the gap at Y=34
ADDR_PLAYER_STATE = 0xD2
ACTION_CARRYING_BIT = 0x02
ADDR_PLAYER_ACTION = 0xD4
ACTION_DROPPING = 0x05


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

    def __post_init__(self):
        self._pathfinder = Pathfinder(self._scanner)
        self._navigator = Navigator(self._pathfinder)
        # Add pond side/top fences as impassable barriers
        # Block ONLY verified side/top barriers (6-tile model)
        # Left: X=30-31, Right: X=34-35, Top: Y=29, Bottom: Y=34 (except gap 32-33)
        self._pathfinder.no_go_tiles.update({
            (30, 29), (31, 29), (32, 29), (33, 29), (34, 29), (35, 29), # Top
            (30, 30), (30, 31), (30, 32), (30, 33), (30, 34), # Far Left
            (31, 30), (31, 31), (31, 32), (31, 33), (31, 34), # Near Left
            (34, 30), (34, 31), (34, 32), (34, 33), (34, 34), # Near Right
            (35, 30), (35, 31), (35, 32), (35, 33), (35, 34), # Far Right
            # Also block water tiles for routing
            (32, 31), (32, 32), (32, 33),
            (33, 31), (33, 32), (33, 33),
        })
        # Note: (30, 34) and (31, 34) are left OPEN as the entrance

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
        self.cleared_count = 0
        if self._toss_task is None:
            self._toss_task = RecordedTask.load(self.toss_task_name)
            # Warn but don't fallback (User requested no fallback hacks)
            non_zero = sum(1 for f in self._toss_task.frames if any(v != 0 for v in f))
            if non_zero < len(self._toss_task.frames) * 0.1:
                print(f"[FENCE] Warning: {self.toss_task_name} appears dead or nearly empty ({non_zero}/{len(self._toss_task.frames)} non-zero frames)")
        self._navigator.update(world.ram)

    def can_start(self, world: WorldState) -> bool:
        try:
            if self._toss_task is None:
                self._toss_task = RecordedTask.load(self.toss_task_name)
            return True
        except FileNotFoundError:
            return False

    def step(self, world: WorldState) -> TaskResult:
        self._navigator.update(world.ram)
        self._total_steps += 1

        if self.debug and self._total_steps % self.debug_interval == 0:
            cur = self._navigator.current_tile
            best_pond = min(POND_TILES, key=lambda p: abs(p[0]-cur[0]) + abs(p[1]-cur[1])) if self._state == "navigate_pond" else None
            
            if self._state == "navigate_pond":
                tgt = best_pond
                app = None
            else:
                tgt = tuple(map(int, self._current.tile)) if self._current else None
                app = tuple(map(int, self._approach_tile)) if self._approach_tile else None
            print(f"[FENCE] step={self._total_steps} state={self._state} pos={cur} target={tgt} approach={app}")
            
            # Local Map Dump for Pond Area Debugging
            # Check a wider area [25, 40]
            print("--- Map Area Dump (X:25-39, Y:25-39) ---")
            for y in range(25, 40):
                row = []
                for x in range(25, 40):
                    tid = get_tile_at(world.ram, x, y)
                    row.append(f"{tid:02x}")
                print(f"Y={y:2d}: {' '.join(row)}")
            print("-----------------------------------------")

        input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
        if input_lock != 1:
            action = make_action(a=True) if (self._steps_on_fence % 2 == 0) else make_action(b=True)
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action), reason="input_lock")

        if self.max_fences is not None and self.cleared_count >= self.max_fences:
            return TaskResult(status=TaskStatus.SUCCESS)

        self._steps_on_fence += 1
        if self._steps_on_fence > self.max_steps_per_fence:
            self._failures += 1
            self._state = "scan"
            self._current = None
            self._approach_tile = None
            self._steps_on_fence = 0
            if self._failures >= self.max_failures:
                return TaskResult(status=TaskStatus.FAILURE, reason="too many fence failures")
            return TaskResult(status=TaskStatus.RUNNING, reason="fence timeout, skipping")

        if self._action_queue:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(self._action_queue.popleft()))

        if self._state == "scan":
            # Check if we are already carrying something
            if (world.ram[ADDR_PLAYER_STATE] & ACTION_CARRYING_BIT):
                self._state = "navigate_pond"
                self._steps_on_fence = 0
                return TaskResult(status=TaskStatus.RUNNING)

            targets = [t for t in self._scanner.scan(world.ram) if t.debris_type == DebrisType.FENCE]
            if not targets:
                return TaskResult(status=TaskStatus.SUCCESS, reason="no fences found")
            targets.sort(key=lambda t: manhattan(t.pos, self._navigator.current_pos))
            picked = False
            for target in targets:
                approach = self._pathfinder.find_approach(world.ram, target.tile, self._navigator.current_pos)
                if approach is None:
                    if self.debug:
                        print(f"[FENCE] skip target {target.tile}: no approach")
                    continue
                path = self._pathfinder.find_path(world.ram, self._navigator.current_tile, approach)
                if path is None:
                    if self.debug:
                        print(f"[FENCE] skip target {target.tile}: no path")
                    continue
                self._current = target
                self._approach_tile = approach
                self._navigator.path = path
                self._state = "navigate"
                self._steps_on_fence = 0
                picked = True
                break
            if not picked:
                return TaskResult(status=TaskStatus.FAILURE, reason="no reachable fence")

        if self._state == "navigate":
            if not self._current or not self._approach_tile:
                self._state = "scan"
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(np.zeros(12, dtype=np.int32)))
            
            # Check if we already picked it up somehow
            if (world.ram[ADDR_PLAYER_STATE] & ACTION_CARRYING_BIT):
                self._state = "navigate_pond"
                return TaskResult(status=TaskStatus.RUNNING)

            if get_tile_at(world.ram, *self._current.tile) != self._current.tile_id:
                self._state = "scan"
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(np.zeros(12, dtype=np.int32)))
            
            if self._navigator.current_tile == self._approach_tile or self._navigator.at_tile(self._approach_tile):
                # Before lifting, make sure we are DEAD CENTER (tolerance 1) 
                # to avoid clipping corners when we start moving to the pond
                action = self._navigator.center_on_tile(self._approach_tile, tolerance=1)
                if action is None:
                    # Already centered
                    self._state = "lift"
                else:
                    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            else:
                if self._navigator.stasis > self.stasis_repath and self._navigator.path:
                    self._pathfinder.temp_blocked.add(self._navigator.path[0])
                    path = self._pathfinder.find_path(world.ram, self._navigator.current_tile, self._approach_tile)
                    if path:
                        self._navigator.path = path
                    else:
                        self._failures += 1
                        self._state = "scan"
                        self._current = None
                        self._approach_tile = None
                        self._steps_on_fence = 0
                        if self._failures >= self.max_failures:
                            return TaskResult(status=TaskStatus.FAILURE, reason="too many fence failures")
                        return TaskResult(status=TaskStatus.RUNNING, reason="repath failed, skipping")
                
                action = self._navigator.follow_path(world.ram)
                if action is None:
                    action = np.zeros(12, dtype=np.int32)
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

        if self._state == "navigate_pond":
            # Safety check: if we are not carrying anything, we shouldn't be here
            state_val = world.ram[ADDR_PLAYER_STATE]
            if not (state_val & ACTION_CARRYING_BIT):
                if self.debug:
                    print(f"[FENCE] navigate_pond called but not carrying! (state=0x{state_val:02x})")
                self._state = "scan"
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(np.zeros(12, dtype=np.int32)))

            current = self._navigator.current_tile
            best_pond = min(POND_TILES, key=lambda p: abs(p[0]-current[0]) + abs(p[1]-current[1]))

            # We are at the pond if we are AT the gap tile
            if current == best_pond or self._navigator.at_tile(best_pond):
                # Center on the toss tile
                action = self._navigator.center_on_tile(best_pond, tolerance=1)
                if action is None:
                    self._state = "toss"
                    self._steps_on_fence = 0
                    if self.debug:
                        print(f"[FENCE] reached pond toss gap at {current}. TOSSING!")
                else:
                    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
                return TaskResult(status=TaskStatus.RUNNING)
            
            # Pathfind to pond. CRITICAL: Treat the tile we just lifted AND the pond as WALKABLE
            # for pathfinding purposes so we can get a path to the boundary.
            overrides = {best_pond}
            if self._current: overrides.add(self._current.tile)
            
            path = self._pathfinder.find_path(
                world.ram, 
                current, 
                best_pond,
                walkable_override=overrides
            )

            if path:
                self._navigator.path = path
                if self.debug and self._total_steps % self.debug_interval == 0:
                    print(f"[FENCE] pond path len={len(path)} next={path[0]} target={best_pond} override={self._current.tile if self._current else None}")
                action = self._navigator.follow_path(world.ram)
                if action is None: action = np.zeros(12, dtype=np.int32)
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            else:
                if self.debug:
                    print(f"[FENCE] pond {best_pond} unreachable from {current} (current_target={self._current.tile if self._current else None})")
                return TaskResult(status=TaskStatus.FAILURE, reason=f"pond {best_pond} unreachable")

        if self._state == "lift":
            if not self._current:
                self._state = "scan"
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(np.zeros(12, dtype=np.int32)))
            player = self._navigator.current_tile
            target = self._current.tile
            dx, dy = target[0] - player[0], target[1] - player[1]
            direction = 'right' if abs(dx) >= abs(dy) and dx > 0 else 'left' if abs(dx) >= abs(dy) else 'down' if dy > 0 else 'up'
            
            # Stationary lift: face the direction then press A.
            self._action_queue.extend([make_action(**{direction: True}) for _ in range(10)]) 
            self._action_queue.extend([make_action(**{direction: True, 'a': True}) for _ in range(25)]) # Hold A longer
            self._action_queue.extend([make_action() for _ in range(30)]) # Wait for lift animation to settle
            
            # Transition to a new intermediate state to VERIFY the lift
            self._state = "verify_lift"
            if self.debug:
                print(f"[FENCE] lift at {target}, direction={direction}, queueing verification")
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(self._action_queue.popleft()))

        if self._state == "verify_lift":
            if self._action_queue:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(self._action_queue.popleft()))
            
            state_val = world.ram[ADDR_PLAYER_STATE]
            dir_val = world.ram[0xDA] if 0xDA < len(world.ram) else -1
            if (state_val & ACTION_CARRYING_BIT):
                if self.debug:
                    print(f"[FENCE] lift verified! (state=0x{state_val:02x}, dir={dir_val}) Transitioning to navigate_pond")
                self._state = "navigate_pond"
                self._steps_on_fence = 0
            else:
                if self.debug:
                    pos = self._navigator.current_tile
                    print(f"[FENCE] lift FAILED at {pos} (state=0x{state_val:02x}, dir={dir_val}). Returning to scan.")
                # Maybe mark this tile as problematic if it fails repeatedly?
                # For now, just retry scanning.
                self._state = "scan"
                self._current = None
                self._approach_tile = None
                self._steps_on_fence = 0
                # Add a few empty frames to let things settle
                self._action_queue.extend([make_action() for _ in range(10)])
            
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(np.zeros(12, dtype=np.int32)))

        if self._state == "toss":
            # If we are no longer carrying, we are done
            state_val = world.ram[ADDR_PLAYER_STATE]
            if not (state_val & ACTION_CARRYING_BIT) and not self._action_queue:
                self.cleared_count += 1
                self._state = "scan"
                self._current = None
                self._approach_tile = None
                self._steps_on_fence = 0
                if self.debug:
                    print(f"[FENCE] toss complete (ram check)")
                return TaskResult(status=TaskStatus.RUNNING)

            if not self._action_queue:
                # Sequence: Face UP 10 frames, Press A 15 frames
                # Approaching from bottom (e.g. at (31, 32), pond at (31, 31))
                self._action_queue.extend([make_action(up=True) for _ in range(10)])
                self._action_queue.extend([make_action(up=True, a=True) for _ in range(15)])
                self._action_queue.extend([make_action() for _ in range(10)])
            
            action = self._action_queue.popleft()
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

        return TaskResult(status=TaskStatus.FAILURE, reason="invalid state")

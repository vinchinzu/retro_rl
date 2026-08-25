"""Fence post handling: detection + toss into pond via recorded task."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

# Add parent directory for retro_harness import

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.tasks.recorded_task import RecordedTask
from harvest.core.tile_catalog import (
    DebrisType,
    ADDR_INPUT_LOCK,
    POND_CHARACTERISTIC_TILES,
)
from harvest.core.animal_status import read_held_item
from harvest.tasks.nav import (
    Pathfinder,
    Navigator,
    make_action,
    TILE_SIZE,
    get_tile_at,
    manhattan,
    VIEWPORT_HOP_TILES,
    WALKABLE_TILES,
)
from harvest.tasks.farm_ops import (
    TileScanner,
    Target,
)
from harvest.maps.map_config import (
    FARM_POND_ACCESS_FENCE_ROW,
    FARM_POND_ACCESS_FENCE_X_RANGE,
    FARM_POND_ACCESS_STAGING_TILES,
)


# Main F0 pond south lip (same as map_config pond_edge / go_to_water_source).
POND_TILES = [(32, 34), (33, 34)]
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

    def __post_init__(self):
        self._pathfinder = Pathfinder(self._scanner)
        self._navigator = Navigator(self._pathfinder)
        # Pond-side barriers (6-tile model). Leave south lip approach open:
        # (30,34)/(31,34) are the west approach to POND_TILES (32,34)/(33,34).
        self._pathfinder.no_go_tiles.update({
            (30, 29), (31, 29), (32, 29), (33, 29), (34, 29), (35, 29),  # Top
            (30, 30), (30, 31), (30, 32), (30, 33),  # Far Left (not y=34)
            (31, 30), (31, 31), (31, 32), (31, 33),  # Near Left (not y=34)
            (34, 30), (34, 31), (34, 32), (34, 33), (34, 34),  # Near Right
            (35, 30), (35, 31), (35, 32), (35, 33), (35, 34),  # Far Right
            # Water body — stands (32–33,34) remain walkable for toss
            (32, 31), (32, 32), (32, 33),
            (33, 31), (33, 32), (33, 33),
        })

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

    def _pond_dump_key(self, target: Target):
        tile = (int(target.tile[0]), int(target.tile[1]))
        skip = 1 if tile in self._skip_tiles else 0
        x, y = tile
        x0, x1 = FARM_POND_ACCESS_FENCE_X_RANGE
        wall = 0 if y == FARM_POND_ACCESS_FENCE_ROW and x0 <= x <= x1 else 1
        pond = min(abs(x - p[0]) + abs(y - p[1]) for p in POND_TILES)
        return (skip, wall, pond)

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

    def _arm_south_charge(self, current) -> TaskResult:
        """B-run south through a just-lifted y=31 gap (ROM soft-blocks BFS)."""
        self._corridor_charge_done = True
        self._action_queue.clear()
        self._action_queue.extend([make_action(down=True) for _ in range(12)])
        self._action_queue.extend(
            [make_action(down=True, b=True) for _ in range(160)]
        )
        for _ in range(4):
            self._action_queue.extend(
                [make_action(down=True, b=True) for _ in range(36)]
            )
            self._action_queue.extend([make_action(left=True) for _ in range(5)])
            self._action_queue.extend(
                [make_action(down=True, b=True) for _ in range(36)]
            )
            self._action_queue.extend([make_action(right=True) for _ in range(5)])
        self._action_queue.extend([make_action() for _ in range(12)])
        if self.debug:
            print(f"[FENCE] south charge at {current}")
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(self._action_queue.popleft()),
            reason="pond south charge",
        )

    def _mark_pond_toss(self) -> None:
        self.cleared_count += 1
        self._failures = 0
        self._skip_tiles.clear()
        self._corridor_charge_done = False
        self._pond_hop_steps = 0

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
            # Still need A/B mash to clear lock. corridor_only: arm local_drop
            # so the moment lock clears we drop (not navigate_pond thrash).
            if self.corridor_only and (
                world.ram[ADDR_PLAYER_STATE] & ACTION_CARRYING_BIT
            ):
                if self._state not in ("local_drop",):
                    self._state = "local_drop"
                    self._steps_on_fence = 0
            action = make_action(a=True) if (self._steps_on_fence % 2 == 0) else make_action(b=True)
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action), reason="input_lock")

        if self.max_fences is not None and self.cleared_count >= self.max_fences:
            return TaskResult(status=TaskStatus.SUCCESS)

        # corridor_only: if stuck on scan while carrying, enter navigate_pond
        # (which arms south-charge then local_drop). Do NOT steal verify_lift.
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
                return TaskResult(
                    status=TaskStatus.RUNNING, reason="timeout drop"
                )
            return self._skip_current("fence timeout, skipping")

        if self._action_queue:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(self._action_queue.popleft()))

        # Natural house-front entry lands around (13,27). Pure south from that
        # tile is a ROM soft-block even though the metatile is walkable. Stage
        # west first, then approach the wall from its north-west corner.
        if (
            self.corridor_only
            and not self._corridor_staged
            and self._state == "scan"
            and self._navigator.current_tile[1] < FARM_POND_ACCESS_FENCE_ROW
        ):
            # Weeds are ROM-walkable but physically pin travel. Block them for
            # both staging and fence approaches.
            weed_tiles = {
                target.tile
                for target in self._scanner.scan(
                    world.ram,
                    types={DebrisType.WEED},
                )
            }
            self._pathfinder.no_go_tiles.update(weed_tiles)
            player = self._navigator.current_tile
            candidates = sorted(
                FARM_POND_ACCESS_STAGING_TILES,
                key=lambda tile: abs(tile[0] - player[0]) + abs(tile[1] - player[1]),
            )
            stage = self._corridor_stage
            path = None
            if stage is None:
                for candidate in candidates:
                    candidate_path = self._pathfinder.find_path(
                        world.ram,
                        player,
                        candidate,
                    )
                    if candidate_path is not None:
                        stage = candidate
                        path = candidate_path
                        self._corridor_stage = candidate
                        break
            else:
                path = self._pathfinder.find_path(world.ram, player, stage)

            if stage is not None and self._navigator.at_tile(stage):
                self._corridor_staged = True
                self._navigator.path = []
            elif stage is not None and path:
                self._navigator.path = path
                self._state = "stage_corridor"
                if self.debug:
                    print(
                        f"[FENCE] corridor stage {self._navigator.current_tile} "
                        f"→ {stage}"
                    )

        if self._state == "stage_corridor":
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
                    world.ram,
                    self._navigator.current_tile,
                    stage,
                ) or []
                action = self._navigator.follow_path(world.ram)
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(action if action is not None else make_action()),
                reason="staging west of corridor",
            )

        if self._state == "scan":
            # Check if we are already carrying something
            if (world.ram[ADDR_PLAYER_STATE] & ACTION_CARRYING_BIT):
                self._state = "navigate_pond"
                self._steps_on_fence = 0
                return TaskResult(status=TaskStatus.RUNNING)

            wanted = tuple(self.debris_types) or (DebrisType.FENCE,)
            targets = [
                t for t in self._scanner.scan(world.ram, types=set(wanted))
                if t.debris_type in wanted
            ]
            if self.corridor_only:
                self._pathfinder.no_go_tiles.update(
                    target.tile
                    for target in self._scanner.scan(
                        world.ram,
                        types={DebrisType.WEED},
                    )
                )
                row = FARM_POND_ACCESS_FENCE_ROW
                x_min, x_max = FARM_POND_ACCESS_FENCE_X_RANGE
                # The corridor contract is specific: lift the y=31 wall.
                # Without this filter the nearest-fence policy clears a post
                # beside the house, then repeatedly re-lifts its own south-field
                # drop while the actual wall remains sealed.
                targets = [
                    target
                    for target in targets
                    if target.tile[1] == row and x_min <= target.tile[0] <= x_max
                ]
            if not targets:
                reason = "corridor already open" if self.corridor_only else "no fences found"
                return TaskResult(status=TaskStatus.SUCCESS, reason=reason)
            if self.pond_dump:
                targets.sort(key=self._pond_dump_key)
            else:
                targets.sort(key=lambda t: manhattan(t.pos, self._navigator.current_pos))
            x0, x1 = FARM_POND_ACCESS_FENCE_X_RANGE
            player_tile = self._navigator.current_tile
            reached_wall = None
            reached_other = None
            hop_wall = None
            hop_wall_dist = None
            hop_other = None
            hop_other_dist = None
            for target in targets:
                tile = (int(target.tile[0]), int(target.tile[1]))
                if tile in self._skip_tiles:
                    continue
                if self.corridor_only:
                    # Always lift from north of the y=31 wall. A generic nearest
                    # approach can choose the sealed south side.
                    approach = (tile[0], tile[1] - 1)
                    if not self._pathfinder.is_walkable(world.ram, *approach):
                        approach = None
                else:
                    approach = self._pathfinder.find_approach(
                        world.ram,
                        target.tile,
                        self._navigator.current_pos,
                    )
                if approach is None:
                    if self.debug:
                        print(f"[FENCE] skip target {target.tile}: no approach")
                    continue
                path = self._pathfinder.find_path(
                    world.ram,
                    player_tile,
                    approach,
                    max_steps=VIEWPORT_HOP_TILES,
                )
                if path is None:
                    if self.debug:
                        print(f"[FENCE] skip target {target.tile}: no path")
                    continue
                wall = (
                    tile[1] == FARM_POND_ACCESS_FENCE_ROW and x0 <= tile[0] <= x1
                )
                reached = (
                    not path
                    or path[-1] == approach
                    or player_tile == approach
                )
                row = (target, approach, path)
                if reached:
                    if wall:
                        reached_wall = row
                        break
                    if reached_other is None:
                        reached_other = row
                    continue
                dist = abs(player_tile[0] - approach[0]) + abs(
                    player_tile[1] - approach[1]
                )
                if wall:
                    if hop_wall is None or dist < hop_wall_dist:
                        hop_wall = row
                        hop_wall_dist = dist
                elif hop_other is None or dist < hop_other_dist:
                    hop_other = row
                    hop_other_dist = dist
            if self.pond_dump:
                pick = reached_wall or hop_wall or reached_other or hop_other
            else:
                pick = reached_wall or reached_other or hop_wall or hop_other
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
            if pick is None:
                if self._skip_tiles:
                    self._skip_tiles.clear()
                    self._pathfinder.temp_blocked.clear()
                    return TaskResult(
                        status=TaskStatus.RUNNING, reason="retry skipped fences"
                    )
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
                if (
                    self._navigator.stasis > self.stasis_repath
                    or not self._navigator.path
                ):
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

            # Corridor-only (empty-can refill): skip pond toss thrash.
            # ROM (Y1_Test_Crops_Planted_Dry): empty-handed south through a
            # y=31 gap soft-blocks on (13,31) y≈505. Carry-south after lift
            # crosses: player often stands on y=30 approach after lift — must
            # charge from y<=31 (not only y==31). Drop only after y>=32 or
            # after the charge attempt fails.
            if self.corridor_only:
                if current[1] >= 32:
                    # Crossed while carrying — drop south of wall, gap stays open.
                    if self.debug:
                        print(f"[FENCE] corridor_only: crossed to {current}")
                    self._state = "local_drop"
                    self._steps_on_fence = 0
                    self._pond_hop_steps = 0
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        reason="corridor_only drop south of wall",
                    )
                if current[1] <= 31 and not getattr(self, "_corridor_charge_done", False):
                    self._corridor_charge_done = True
                    self._action_queue.clear()
                    # Face/align south then long B-run. ROM lift leaves player
                    # on approach (x,30); charge from there crosses to y>=32.
                    self._action_queue.extend(
                        [make_action(down=True) for _ in range(12)]
                    )
                    self._action_queue.extend(
                        [make_action(down=True, b=True) for _ in range(160)]
                    )
                    # A straight charge stops on the gap tile in this ROM.
                    # Brief lateral wiggles while continuing south break the
                    # soft collision without walking into neighboring posts.
                    for _ in range(4):
                        self._action_queue.extend(
                            [make_action(down=True, b=True) for _ in range(36)]
                        )
                        self._action_queue.extend(
                            [make_action(left=True) for _ in range(5)]
                        )
                        self._action_queue.extend(
                            [make_action(down=True, b=True) for _ in range(36)]
                        )
                        self._action_queue.extend(
                            [make_action(right=True) for _ in range(5)]
                        )
                    self._action_queue.extend([make_action() for _ in range(12)])
                    if self.debug:
                        print(
                            f"[FENCE] corridor_only: carry-south charge "
                            f"at {current}"
                        )
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        action=ActionResult(self._action_queue.popleft()),
                        reason="corridor_only south charge",
                    )
                # Charge already tried and still north of wall — local drop
                # (gap open). CropWaterTask will east-crawl y=30→x≥28 then south.
                if self.debug:
                    print(
                        f"[FENCE] corridor_only: local drop at {current} "
                        f"after failed carry-south (gap open)"
                    )
                self._state = "local_drop"
                self._steps_on_fence = 0
                self._pond_hop_steps = 0
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    reason="corridor_only local drop",
                )

            # Leftover pond dump: ROM south-gap trap only on the y=31 wall.
            # Charging south from a north-farm lift runs into rocks and drops.
            if (
                self.pond_dump
                and 30 <= current[1] <= 31
                and not getattr(self, "_corridor_charge_done", False)
            ):
                return self._arm_south_charge(current)

            # ROM trap: BFS invents a path through (x,32) after lift, but game
            # physics soft-blocks south transit while standing on the gap tile.
            # After short stasis on y=31, fall through to local_drop so the gap
            # stays open for crop multi-hop (re-approach from y≤29 then charge).
            if (
                current[1] == 31
                and self._navigator.stasis > 90
                and not self.pond_dump
            ):
                if self.debug:
                    print(
                        f"[FENCE] gap-tile stasis at {current} "
                        f"(stasis={self._navigator.stasis}); local drop"
                    )
                self._state = "local_drop"
                self._steps_on_fence = 0
                self._pond_hop_steps = 0
                self._navigator.path = []
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    reason="local drop after gap soft-block",
                )

            # We are at the pond if we are AT the gap tile
            if current == best_pond or self._navigator.at_tile(best_pond):
                # Center on the toss tile
                action = self._navigator.center_on_tile(best_pond, tolerance=1)
                if action is None:
                    self._toss_face = "up"
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
            if self._current:
                overrides.add(self._current.tile)
            # Also treat the cleared gap row tile as walkable if still stale in RAM.
            if self._current is not None:
                gx, gy = self._current.tile
                overrides.add((gx, gy))
                overrides.add((gx, gy + 1))  # one step south through the wall gap

            # Viewport hop only. Unbounded BFS walks 0x00 stale farm cells that
            # load as stumps/rocks when the camera catches up.
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
                cur_dist = abs(current[0] - best_pond[0]) + abs(
                    current[1] - best_pond[1]
                )
                hop_dist = abs(hop_end[0] - best_pond[0]) + abs(
                    hop_end[1] - best_pond[1]
                )
                if self.pond_dump or hop_dist < cur_dist:
                    path = hop
                    self._pond_hop_steps = getattr(self, "_pond_hop_steps", 0) + 1
            hop_limit = 16 if self.pond_dump else 6
            if path is not None and getattr(self, "_pond_hop_steps", 0) > hop_limit:
                if self.pond_dump:
                    self._pond_hop_steps = 0
                    self._pathfinder.temp_blocked.clear()
                else:
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

            # Pond still unreachable (wall residue / viewport). Corridor-open
            # only needs the gap; leftover dump keeps the post until F0.
            if self.pond_dump:
                self._pathfinder.temp_blocked.clear()
                self._navigator.path = []
                self._pond_hop_steps = 0
                return TaskResult(
                    status=TaskStatus.RUNNING, reason="pond_dump repath"
                )
            if self.debug:
                print(
                    f"[FENCE] pond {best_pond} unreachable from {current}; "
                    f"local drop (gap open for refill)"
                )
            self._state = "local_drop"
            self._steps_on_fence = 0
            self._pond_hop_steps = 0
            return TaskResult(
                status=TaskStatus.RUNNING,
                reason="local drop after pond unreachable",
            )

        if self._state == "local_drop":
            # Place/throw the fence post nearby (prefer south) without pond path.
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
                # corridor_only: one (or max_fences) local drops open the gap —
                # return SUCCESS without scanning more fences.
                if self.corridor_only and (
                    self.max_fences is None
                    or self.cleared_count >= self.max_fences
                ):
                    return TaskResult(
                        status=TaskStatus.SUCCESS,
                        reason=f"corridor open cleared={self.cleared_count}",
                    )
                return TaskResult(status=TaskStatus.RUNNING, reason="local drop done")
            if not self._action_queue:
                attempts = getattr(self, "_local_drop_cycles", 0)
                # Prefer south/sideways first. After one failed cycle include up
                # (crop drop uses up successfully). Never prefer up first —
                # that seals y=29 re-approach.
                current = self._navigator.current_tile
                if self.corridor_only and current[1] <= 31:
                    # At the gap, dropping south reseals the only exit cell.
                    faces = ("left", "right", "up")
                elif self.corridor_only and attempts == 0:
                    faces = ("down", "left", "right")
                else:
                    faces = ("down", "left", "right", "up")
                # A berry cannot be picked with debris still in hand. Never
                # report an open corridor as successful until both carry RAM
                # signals are clear.
                if self.corridor_only and attempts >= 6:
                    return TaskResult(
                        status=TaskStatus.FAILURE,
                        reason=(
                            "corridor open but hands not clear "
                            f"held=0x{read_held_item(world.ram):02X}"
                        ),
                    )
                self._local_drop_cycles = attempts + 1
                for face in faces:
                    self._action_queue.extend(
                        [make_action(**{face: True}) for _ in range(10)]
                    )
                    self._action_queue.extend([make_action() for _ in range(4)])
                    self._action_queue.extend(
                        [make_action(**{face: True, "a": True}) for _ in range(20)]
                    )
                    self._action_queue.extend([make_action(a=True) for _ in range(8)])
                    self._action_queue.extend([make_action() for _ in range(16)])
            action = self._action_queue.popleft()
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

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
                self._mark_pond_toss()
                self._state = "scan"
                self._current = None
                self._approach_tile = None
                self._steps_on_fence = 0
                if self.debug:
                    print(f"[FENCE] toss complete (ram check)")
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

        return TaskResult(status=TaskStatus.FAILURE, reason="invalid state")

"""Viewport-aware navigation tasks used by day-plan phases."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.tasks.nav import (
    Point,
    Pathfinder,
    Navigator,
    make_action,
    get_tile_at,
    TILE_SIZE,
)
from harvest.core.tile_catalog import ADDR_TILEMAP, ADDR_INPUT_LOCK
from harvest.tasks.farm_ops import TileScanner

from harvest.core.scene import classify_scene_from_ram
from harvest.tasks.primitives import dismiss_dialogue_result, drain_action_queue
from harvest.tasks.recorded_task import RecordedTask
from harvest.planner.day_plan_status import TASKS_DIR, tilemaps_match

_OPPOSITE_FACE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}

# Directions we refuse to B-charge into (solid / "bush thrash" tiles).
_DIR_DELTA = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


def _neighbor_tile(tx: int, ty: int, direction: str) -> Tuple[int, int]:
    dx, dy = _DIR_DELTA[direction]
    return tx + dx, ty + dy


def _nav_needs_menu_dismiss(ram: np.ndarray, step_count: int) -> Optional[TaskResult]:
    """Dismiss dialogue/menu/input-lock so navigation does not walk blind."""
    input_lock = int(ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(ram) else 1
    if input_lock != 1:
        return dismiss_dialogue_result(step_count, reason="input locked")
    scene = classify_scene_from_ram(ram)
    if scene.needs_input_dismiss:
        return dismiss_dialogue_result(
            step_count,
            reason=f"nav {scene.mode.value}",
        )
    return None

# ── NavTask ───────────────────────────────────────────────────────

MAX_HOP = 10  # Max BFS hop distance (SNES viewport loads ~10 tiles around player)
STALE_TILE_IDS = {0x72, 0x75, 0x76, 0xFF}


def find_frontier_path(
    pathfinder: Pathfinder,
    ram: np.ndarray,
    start: Tuple[int, int],
    final: Tuple[int, int],
    *,
    radius: int = 8,
    max_candidates: int = 24,
) -> Optional[List[Tuple[int, int]]]:
    """Route to the best loaded walkable frontier tile toward the final goal."""
    sx, sy = start
    fx, fy = final
    start_dist = abs(fx - sx) + abs(fy - sy)
    candidates: List[Tuple[Tuple[int, int, int], Tuple[int, int]]] = []

    for ty in range(max(0, sy - radius), min(63, sy + radius) + 1):
        for tx in range(max(0, sx - radius), min(63, sx + radius) + 1):
            if (tx, ty) == start:
                continue
            tid = get_tile_at(ram, tx, ty)
            if tid in STALE_TILE_IDS:
                continue
            if not pathfinder.is_walkable(ram, tx, ty, current_pos=start):
                continue
            dist = abs(fx - tx) + abs(fy - ty)
            progress = start_dist - dist
            if progress <= 0:
                continue
            axis_progress = abs(tx - sx) + abs(ty - sy)
            candidates.append(((progress, axis_progress, -dist), (tx, ty)))

    candidates.sort(reverse=True)
    for _, goal in candidates[:max_candidates]:
        path = pathfinder.find_path(ram, start, goal)
        if path:
            return path
    return None


def find_loaded_direction(
    ram: np.ndarray,
    start: Tuple[int, int],
    final: Tuple[int, int],
    *,
    max_scan: int = 6,
) -> Optional[str]:
    """Pick a cardinal direction that reaches loaded tiles while improving progress."""
    sx, sy = start
    start_dist = abs(final[0] - sx) + abs(final[1] - sy)
    best: Optional[Tuple[int, int, str]] = None
    for direction, dx, dy in (
        ("right", 1, 0),
        ("left", -1, 0),
        ("down", 0, 1),
        ("up", 0, -1),
    ):
        for steps in range(1, max_scan + 1):
            tx = sx + dx * steps
            ty = sy + dy * steps
            tid = get_tile_at(ram, tx, ty)
            if tid in STALE_TILE_IDS:
                continue
            dist = abs(final[0] - tx) + abs(final[1] - ty)
            progress = start_dist - dist
            score = (progress, -steps, direction)
            if best is None or score > best:
                best = score
            break
    return best[2] if best and best[0] >= 0 else None


@dataclass
class NavTask(Task):
    """BFS point-to-point navigation with viewport-aware short hops.

    SNES only loads tile data near the player into RAM, so distant tiles
    read as invalid (0x72).  To avoid routing through stale data, the BFS
    target is clamped to MAX_HOP tiles from the current position, creating
    a series of short hops toward the destination.
    """

    name: str = "nav"
    target_px: Point = field(default_factory=lambda: Point(0, 0))
    radius: int = 12
    soft_radius: Optional[int] = None
    soft_stasis: int = 90
    timeout: int = 3000
    stasis_repath: int = 180

    _scanner: TileScanner = field(default_factory=TileScanner, init=False)
    _pathfinder: Pathfinder = field(init=False)
    _navigator: Navigator = field(init=False)
    _step_count: int = field(default=0, init=False)
    _action_queue: deque = field(default_factory=deque, init=False)

    def __post_init__(self):
        self._pathfinder = Pathfinder(self._scanner)
        self._navigator = Navigator(self._pathfinder)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._action_queue.clear()
        self._navigator.update(world.ram)
        self._navigator.path = []
        self._navigator.stasis = 0
        self._pathfinder.temp_blocked.clear()

    def resume_after_hotswap(self, world: WorldState) -> None:
        self._action_queue.clear()
        self._navigator.update(world.ram)
        self._navigator.path = []
        self._navigator.stasis = 0
        self._pathfinder.temp_blocked.clear()

    def can_start(self, world: WorldState) -> bool:
        return True

    def _distance_to_target(self) -> int:
        pos = self._navigator.current_pos
        return max(abs(pos.x - self.target_px.x), abs(pos.y - self.target_px.y))

    def _soft_limit(self) -> int:
        if self.soft_radius is not None:
            return int(self.soft_radius)
        return max(self.radius * 2, self.radius + 16)

    def _at_target(self) -> bool:
        return self._distance_to_target() <= self.radius

    def _soft_arrived(self) -> bool:
        """Accept near-miss when stuck or nearly out of time."""
        dist = self._distance_to_target()
        if dist > self._soft_limit():
            return False
        if self._navigator.stasis >= self.soft_stasis:
            return True
        # Last 20% of the timeout budget: close enough is good enough.
        return self._step_count >= max(1, int(self.timeout * 0.8))

    def _hop_target(self) -> tuple:
        """BFS target clamped to MAX_HOP tiles from current position."""
        cur = self._navigator.current_tile
        final = (self.target_px.x // TILE_SIZE, self.target_px.y // TILE_SIZE)
        dx = final[0] - cur[0]
        dy = final[1] - cur[1]
        dist = max(abs(dx), abs(dy))
        if dist <= MAX_HOP:
            return final
        scale = MAX_HOP / dist
        return (cur[0] + int(dx * scale), cur[1] + int(dy * scale))

    def _fallback_action(self, ram: np.ndarray) -> np.ndarray:
        """Walk toward the target when BFS cannot build a path."""
        current_tile = self._navigator.current_tile
        final_tile = (self.target_px.x // TILE_SIZE, self.target_px.y // TILE_SIZE)
        if get_tile_at(ram, *current_tile) in STALE_TILE_IDS:
            loaded_direction = find_loaded_direction(ram, current_tile, final_tile)
            if loaded_direction is not None:
                return make_action(**{loaded_direction: True, "b": True})

        cur = self._navigator.current_pos
        dx = self.target_px.x - cur.x
        dy = self.target_px.y - cur.y
        if abs(dx) >= abs(dy):
            primary = "right" if dx > 0 else "left"
            secondary = "down" if dy > 0 else "up"
        else:
            primary = "down" if dy > 0 else "up"
            secondary = "right" if dx > 0 else "left"

        opposites = {"up": "down", "down": "up", "left": "right", "right": "left"}
        stasis = self._navigator.stasis
        if stasis < 30:
            direction = primary
        elif stasis < 60:
            direction = secondary
        elif stasis < 90:
            direction = opposites[primary]
        else:
            direction = opposites[secondary]
        return make_action(**{direction: True, "b": True})

    def step(self, world: WorldState) -> TaskResult:
        self._navigator.update(world.ram)
        self._step_count += 1

        # Dialog / menu dismissal (tool menus and shop prompts block BFS).
        dismissed = _nav_needs_menu_dismiss(world.ram, self._step_count)
        if dismissed is not None:
            return dismissed

        # Drain queued actions
        queued = drain_action_queue(self._action_queue)
        if queued is not None:
            return queued

        # Arrived?
        if self._at_target():
            return TaskResult(status=TaskStatus.SUCCESS, reason="arrived")
        if self._soft_arrived():
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=(
                    f"soft arrived dist={self._distance_to_target()} "
                    f"stasis={self._navigator.stasis}"
                ),
            )

        if self._step_count > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason="nav timeout")

        # Stuck recovery
        if self._navigator.stasis > self.stasis_repath and self._navigator.path:
            self._pathfinder.temp_blocked.add(self._navigator.path[0])
            self._navigator.path = []
            self._navigator.stasis = 0

        # Path if needed (use hop target within viewport range)
        if not self._navigator.path:
            hop = self._hop_target()
            goal = self._pathfinder.find_nearest_walkable(world.ram, hop, max_radius=4)
            if goal is None:
                goal = hop
            path = self._pathfinder.find_path(world.ram, self._navigator.current_tile, goal)
            if not path:
                final = (self.target_px.x // TILE_SIZE, self.target_px.y // TILE_SIZE)
                path = find_frontier_path(self._pathfinder, world.ram, self._navigator.current_tile, final)
            if path:
                self._navigator.path = path
                self._navigator.stasis = 0
            else:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(self._fallback_action(world.ram)))

        action = self._navigator.follow_path(world.ram)
        if action is not None:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
        if not self._navigator.path:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(self._fallback_action(world.ram)))
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))


# ── CrossMapRecordedTask ──────────────────────────────────────────

@dataclass
class CrossMapRecordedTask(Task):
    """Walk off current map, replay recording for off-map actions, return.

    Two phases:
      1. "exit" — walk in exit_direction until tilemap changes
      2. "replay" — replay recording frames; succeed when tilemap returns
         to origin or recording exhausted
    """

    name: str = "cross_map"
    exit_direction: str = "left"
    recording_name: str = ""
    recording_start: int = 0
    origin_tilemap: int = 0x00
    tasks_dir: str = TASKS_DIR
    timeout: int = 5000
    min_replay_before_return: int = 100
    continue_after_return: int = 0  # extra frames to play after return detection
    stock_field: str = ""
    require_purchase: bool = False

    _phase: str = field(default="exit", init=False)
    _step_count: int = field(default=0, init=False)
    _frames: list = field(default_factory=list, init=False)
    _frame_idx: int = field(default=0, init=False)
    _return_frame: int = field(default=0, init=False)
    _stock_before: int = field(default=0, init=False)
    _money_before: int = field(default=0, init=False)
    _seen_shop: bool = field(default=False, init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._phase = "exit"
        self._frame_idx = 0
        self._return_frame = 0
        self._seen_shop = False
        from harvest.core.ram_catalog import read_ram_value

        if self.stock_field:
            try:
                self._stock_before = int(read_ram_value(world.ram, self.stock_field) or 0)
            except Exception:
                self._stock_before = 0
        try:
            self._money_before = int(read_ram_value(world.ram, "money") or 0)
        except Exception:
            self._money_before = 0
        recording = RecordedTask.load(self.recording_name, self.tasks_dir)
        self._frames = recording.frames[self.recording_start:]

    def can_start(self, world: WorldState) -> bool:
        return True

    def _purchase_ok(self, world: WorldState) -> bool:
        if not self.require_purchase:
            return True
        from harvest.core.ram_catalog import read_ram_value

        stock = self._stock_before
        money = self._money_before
        if self.stock_field:
            try:
                stock = int(read_ram_value(world.ram, self.stock_field) or 0)
            except Exception:
                stock = self._stock_before
        try:
            money = int(read_ram_value(world.ram, "money") or 0)
        except Exception:
            money = self._money_before
        return self._seen_shop and stock > self._stock_before and money < self._money_before

    def _close_replay(self, world: WorldState, reason: str) -> TaskResult:
        if self.require_purchase and not self._purchase_ok(world):
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"shop miss ({reason}): no 0x1C/stock/wallet delta",
            )
        return TaskResult(status=TaskStatus.SUCCESS, reason=reason)

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        if self._step_count > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason="cross_map timeout")

        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if tilemap == 0x1C:
            self._seen_shop = True

        if self._phase == "exit":
            if not tilemaps_match(tilemap, self.origin_tilemap):
                self._phase = "replay"
                print(f"[CROSS_MAP] Exited to tilemap 0x{tilemap:02X}")
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            action = make_action(**{self.exit_direction: True, "b": True})
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

        # Replay phase: check if returned to origin tilemap
        if tilemaps_match(tilemap, self.origin_tilemap) and self._frame_idx > self.min_replay_before_return:
            if self._return_frame == 0:
                self._return_frame = self._frame_idx
            # Continue playing recording for extra frames to walk into map
            if self._frame_idx - self._return_frame >= self.continue_after_return:
                return self._close_replay(world, "returned to origin map")

        if self._frame_idx >= len(self._frames):
            return self._close_replay(world, "recording complete")

        action = np.array(self._frames[self._frame_idx], dtype=np.int32)
        self._frame_idx += 1
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))


@dataclass
class RecordedTransitionTask(Task):
    """Replay a recorded chunk and require a specific map transition."""

    name: str = "recorded_transition"
    task_name: str = ""
    target_tilemap: int = 0x00
    origin_tilemap: Optional[int] = None
    tasks_dir: str = TASKS_DIR
    timeout: int = 2000
    min_frames_before_success: int = 1

    _step_count: int = field(default=0, init=False)
    _frames: list = field(default_factory=list, init=False)
    _frame_idx: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._frame_idx = 0
        self._frames = RecordedTask.load(self.task_name, self.tasks_dir).frames

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        if self._step_count > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason="recorded_transition timeout")

        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if (
            self.origin_tilemap is not None
            and self._frame_idx == 0
            and not tilemaps_match(tilemap, self.origin_tilemap)
        ):
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"expected origin 0x{self.origin_tilemap:02X}, got 0x{tilemap:02X}",
            )

        if tilemaps_match(tilemap, self.target_tilemap) and self._frame_idx >= self.min_frames_before_success:
            return TaskResult(status=TaskStatus.SUCCESS, reason=f"tilemap=0x{tilemap:02X}")

        if self._frame_idx >= len(self._frames):
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"recording exhausted before tilemap 0x{self.target_tilemap:02X}",
            )

        action = np.array(self._frames[self._frame_idx], dtype=np.int32)
        self._frame_idx += 1
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

__all__ = [
    "MAX_HOP",
    "STALE_TILE_IDS",
    "find_frontier_path",
    "find_loaded_direction",
    "NavTask",
    "CrossMapRecordedTask",
    "RecordedTransitionTask",
    "MultiMapNavTask",
    "_DIR_DELTA",
    "_neighbor_tile",
    "_nav_needs_menu_dismiss",
    "_OPPOSITE_FACE",
]


def __getattr__(name: str):
    """Lazy re-export so callers can still import MultiMapNavTask from here."""
    if name == "MultiMapNavTask":
        from harvest.planner.tasks.multi_nav import MultiMapNavTask as _cls
        return _cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

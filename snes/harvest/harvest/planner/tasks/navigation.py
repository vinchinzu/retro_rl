"""Viewport-aware navigation tasks used by day-plan phases."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.tasks.farm_clearer import (
    Point,
    TileScanner,
    Pathfinder,
    Navigator,
    make_action,
    get_tile_at,
    ADDR_TILEMAP,
    ADDR_INPUT_LOCK,
    TILE_SIZE,
)
from harvest.maps.map_config import Waypoint, get_walkable_tiles
from harvest.core.scene import classify_scene_from_ram
from harvest.tasks.primitives import (
    dismiss_dialogue_result,
    drain_action_queue,
    press_button_sequence,
)
from harvest.tasks.recorded_task import RecordedTask
from harvest.planner.day_plan_status import TASKS_DIR, tilemaps_match


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

    _phase: str = field(default="exit", init=False)
    _step_count: int = field(default=0, init=False)
    _frames: list = field(default_factory=list, init=False)
    _frame_idx: int = field(default=0, init=False)
    _return_frame: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._phase = "exit"
        self._frame_idx = 0
        self._return_frame = 0
        recording = RecordedTask.load(self.recording_name, self.tasks_dir)
        self._frames = recording.frames[self.recording_start:]

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        if self._step_count > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason="cross_map timeout")

        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0

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
                return TaskResult(status=TaskStatus.SUCCESS, reason="returned to origin map")

        if self._frame_idx >= len(self._frames):
            return TaskResult(status=TaskStatus.SUCCESS, reason="recording complete")

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


# ── MultiMapNavTask ───────────────────────────────────────────────

@dataclass
class MultiMapNavTask(Task):
    """Navigate a sequence of waypoints across multiple maps using BFS.

    State machine per waypoint:
      nav        → BFS navigate to target_px using per-map walkable tiles
      action     → Face direction, press button, cooldown
      exit_walk  → Walk exit_direction + B until tilemap changes
      exit_settle → Idle frames for tile data to load, rebuild pathfinder
    """

    name: str = "multi_nav"
    waypoints: List[Waypoint] = field(default_factory=list)
    timeout: int = 8000
    initial_settle_frames: int = 60

    _scanner: TileScanner = field(default_factory=TileScanner, init=False)
    _pathfinder: Pathfinder = field(init=False)
    _navigator: Navigator = field(init=False)
    _step_count: int = field(default=0, init=False)
    _wp_index: int = field(default=0, init=False)
    _phase: str = field(default="nav", init=False)
    _action_queue: deque = field(default_factory=deque, init=False)
    _settle_frames: int = field(default=0, init=False)
    _exit_walk_frames: int = field(default=0, init=False)
    _no_path_frames: int = field(default=0, init=False)
    _initial_settle: int = field(default=0, init=False)

    def __post_init__(self):
        self._scanner = TileScanner()
        self._pathfinder = Pathfinder(self._scanner)
        self._navigator = Navigator(self._pathfinder)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._wp_index = 0
        self._phase = "nav"
        self._action_queue.clear()
        self._settle_frames = 0
        self._exit_walk_frames = 0
        self._no_path_frames = 0
        self._initial_settle = 0
        self._navigator.update(world.ram)
        self._navigator.path = []
        self._navigator.stasis = 0
        self._pathfinder.temp_blocked.clear()
        # Set initial walkable tiles based on current tilemap
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        self._rebuild_pathfinder(tilemap)
        if self.waypoints:
            print(f"[MULTI_NAV] Start: {len(self.waypoints)} waypoints, tilemap=0x{tilemap:02X}")

    def can_start(self, world: WorldState) -> bool:
        return len(self.waypoints) > 0

    def resume_after_hotswap(self, world: WorldState) -> None:
        self._action_queue.clear()
        if self._phase != "exit_settle":
            self._phase = "nav"
        self._settle_frames = 0
        self._exit_walk_frames = 0
        self._no_path_frames = 0
        self._navigator.update(world.ram)
        self._navigator.path = []
        self._navigator.stasis = 0
        self._pathfinder.temp_blocked.clear()

    def _rebuild_pathfinder(self, tilemap_id: int) -> None:
        """Rebuild pathfinder with walkable tiles for the given map."""
        walkable = get_walkable_tiles(tilemap_id)
        self._pathfinder = Pathfinder(self._scanner, walkable_tiles=set(walkable))
        self._navigator = Navigator(self._pathfinder)
        self._navigator.stasis = 0

    def _current_wp(self) -> Optional[Waypoint]:
        if self._wp_index < len(self.waypoints):
            return self.waypoints[self._wp_index]
        return None

    def _waypoint_tilemap_matches(self, tilemap: int, wp: Waypoint) -> bool:
        return tilemaps_match(tilemap, wp.tilemap)

    def _at_wp_target(self, wp: Waypoint) -> bool:
        pos = self._navigator.current_pos
        return (abs(pos.x - wp.target_px[0]) <= wp.radius and
                abs(pos.y - wp.target_px[1]) <= wp.radius)

    def _hop_target(self, wp: Waypoint) -> Tuple[int, int]:
        """BFS target clamped to stay within loaded viewport.

        SNES loads ~16x14 tiles around the player. Clamp each axis
        independently to 7 tiles so diagonal hops stay within the
        loaded region (unlike Chebyshev MAX_HOP which can overshoot).
        """
        cur = self._navigator.current_tile
        final = (wp.target_px[0] // TILE_SIZE, wp.target_px[1] // TILE_SIZE)
        dx = final[0] - cur[0]
        dy = final[1] - cur[1]
        if abs(dx) <= MAX_HOP and abs(dy) <= MAX_HOP:
            return final
        # Clamp each axis to MAX_HOP tiles
        cx = max(-MAX_HOP, min(MAX_HOP, dx))
        cy = max(-MAX_HOP, min(MAX_HOP, dy))
        # Scale down to keep within viewport (7 tiles = half viewport)
        limit = 7
        if abs(cx) > limit or abs(cy) > limit:
            scale = limit / max(abs(cx), abs(cy))
            cx = int(cx * scale)
            cy = int(cy * scale)
        return (cur[0] + cx, cur[1] + cy)

    def _advance_waypoint(self) -> None:
        """Move to next waypoint."""
        self._wp_index += 1
        self._phase = "nav"
        self._action_queue.clear()
        self._navigator.path = []
        self._navigator.stasis = 0
        self._pathfinder.temp_blocked.clear()
        wp = self._current_wp()
        if wp:
            print(f"[MULTI_NAV] Waypoint {self._wp_index + 1}/{len(self.waypoints)}"
                  f" tilemap=0x{wp.tilemap:02X} target={wp.target_px}")

    def step(self, world: WorldState) -> TaskResult:
        self._navigator.update(world.ram)
        self._step_count += 1

        if self._step_count > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason="multi_nav timeout")

        # Initial settle: walk toward first waypoint to trigger tile loading.
        # After map transition, SNES tile RAM reads stale 0x72 until the
        # player moves and the viewport scrolls to load new tile data.
        SETTLE_FRAMES = self.initial_settle_frames
        if self._initial_settle < SETTLE_FRAMES:
            self._initial_settle += 1
            dismissed = _nav_needs_menu_dismiss(world.ram, self._step_count)
            if dismissed is not None:
                return dismissed
            # Walk toward first waypoint during settle to trigger tile loading
            wp = self._current_wp()
            if wp:
                cur = self._navigator.current_pos
                dx = wp.target_px[0] - cur.x
                dy = wp.target_px[1] - cur.y
                if abs(dx) >= abs(dy):
                    direction = "right" if dx > 0 else "left"
                else:
                    direction = "down" if dy > 0 else "up"
                action = make_action(**{direction: True, "b": True})
            else:
                action = make_action()
            if self._initial_settle == SETTLE_FRAMES:
                tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
                self._rebuild_pathfinder(tilemap)
                self._navigator.update(world.ram)
                print(f"[MULTI_NAV] Settle done, pos=({self._navigator.current_pos.x},"
                      f"{self._navigator.current_pos.y}) tilemap=0x{tilemap:02X}")
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

        # All waypoints done
        wp = self._current_wp()
        if wp is None:
            return TaskResult(status=TaskStatus.SUCCESS, reason="all waypoints reached")

        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if (
            not self._waypoint_tilemap_matches(tilemap, wp)
            and self._phase not in {"exit_walk", "exit_settle"}
        ):
            for idx in range(self._wp_index + 1, len(self.waypoints)):
                if self._waypoint_tilemap_matches(tilemap, self.waypoints[idx]):
                    print(f"[MULTI_NAV] Relocalized from waypoint {self._wp_index + 1}"
                          f" to {idx + 1} on tilemap=0x{tilemap:02X}")
                    self._wp_index = idx
                    self._phase = "nav"
                    self._navigator.path = []
                    self._navigator.stasis = 0
                    self._pathfinder.temp_blocked.clear()
                    wp = self._current_wp()
                    break
            if wp is not None and not self._waypoint_tilemap_matches(tilemap, wp):
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=f"expected tilemap 0x{wp.tilemap:02X}, got 0x{tilemap:02X}",
                )

        # Dialog / menu dismissal
        dismissed = _nav_needs_menu_dismiss(world.ram, self._step_count)
        if dismissed is not None:
            return dismissed

        # Drain queued actions
        queued = drain_action_queue(self._action_queue)
        if queued is not None:
            return queued

        # ── Phase: exit_settle ──
        if self._phase == "exit_settle":
            self._settle_frames += 1
            if self._settle_frames >= 30:
                # Rebuild pathfinder for new map
                self._rebuild_pathfinder(tilemap)
                self._navigator.update(world.ram)
                print(f"[MULTI_NAV] Settled on tilemap 0x{tilemap:02X}"
                      f" pos=({self._navigator.current_pos.x},{self._navigator.current_pos.y})")
                self._advance_waypoint()
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        # ── Phase: exit_walk ──
        if self._phase == "exit_walk":
            self._exit_walk_frames += 1
            # Check if tilemap changed
            if not self._waypoint_tilemap_matches(tilemap, wp):
                print(f"[MULTI_NAV] Exited map 0x{wp.tilemap:02X} → 0x{tilemap:02X}"
                      f" after {self._exit_walk_frames} frames")
                self._phase = "exit_settle"
                self._settle_frames = 0
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            # Timeout: give up after 500 frames of walking toward exit
            if self._exit_walk_frames > 500:
                print(f"[MULTI_NAV] Exit walk timeout (500 frames)")
                self._advance_waypoint()
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            direction = wp.exit_direction or "left"
            action = make_action(**{direction: True, "b": True})
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

        # ── Phase: action ──
        if self._phase == "action":
            # Queue the action sequence
            if wp.action_on_arrive == "press_a":
                self._action_queue.extend(
                    press_button_sequence(
                        "a",
                        face=wp.action_face,
                        face_frames=1 if wp.action_face else 0,
                        pre_press_settle_frames=5 if wp.action_face else 0,
                        hold_frames=wp.action_frames,
                        settle_frames=wp.action_cooldown,
                    )
                )
            elif wp.action_on_arrive == "press_b":
                self._action_queue.extend(
                    press_button_sequence(
                        "b",
                        face=wp.action_face,
                        face_frames=1 if wp.action_face else 0,
                        pre_press_settle_frames=5 if wp.action_face else 0,
                        hold_frames=wp.action_frames,
                        settle_frames=wp.action_cooldown,
                    )
                )
            elif wp.action_on_arrive == "press_y":
                self._action_queue.extend(
                    press_button_sequence(
                        "y",
                        face=wp.action_face,
                        face_frames=1 if wp.action_face else 0,
                        pre_press_settle_frames=5 if wp.action_face else 0,
                        hold_frames=wp.action_frames,
                        settle_frames=wp.action_cooldown,
                    )
                )

            print(f"[MULTI_NAV] Action: {wp.action_on_arrive} face={wp.action_face}")
            self._phase = "action_drain"
            queued = drain_action_queue(self._action_queue)
            if queued is not None:
                return queued
            self._advance_waypoint()
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        if self._phase == "action_drain":
            if not self._action_queue:
                # Action sequence done, advance
                self._advance_waypoint()
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            queued = drain_action_queue(self._action_queue)
            if queued is not None:
                return queued

        # ── Phase: nav ──
        # Check arrival
        if self._at_wp_target(wp):
            if wp.is_exit:
                print(f"[MULTI_NAV] Reached exit waypoint, walking {wp.exit_direction}")
                self._phase = "exit_walk"
                self._exit_walk_frames = 0
                direction = wp.exit_direction or "left"
                action = make_action(**{direction: True, "b": True})
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            elif wp.action_on_arrive:
                self._phase = "action"
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            else:
                # Just a nav waypoint, advance
                self._advance_waypoint()
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        # Direct run: if waypoint specifies run_direction, just hold that
        # direction + B. Much faster than BFS for known clear paths.
        # Check the axis of travel to detect overshoot.
        if wp.run_direction:
            cur = self._navigator.current_pos
            d = wp.run_direction
            if d in {"left", "right"} and abs(cur.y - wp.target_px[1]) > wp.radius:
                align = "down" if wp.target_px[1] > cur.y else "up"
                return TaskResult(status=TaskStatus.RUNNING,
                                  action=ActionResult(make_action(**{align: True, "b": True})))
            if d in {"up", "down"} and abs(cur.x - wp.target_px[0]) > wp.radius:
                align = "right" if wp.target_px[0] > cur.x else "left"
                return TaskResult(status=TaskStatus.RUNNING,
                                  action=ActionResult(make_action(**{align: True, "b": True})))
            overshot = False
            if d == "down" and cur.y > wp.target_px[1] + wp.radius:
                overshot = True
            elif d == "up" and cur.y < wp.target_px[1] - wp.radius:
                overshot = True
            elif d == "right" and cur.x > wp.target_px[0] + wp.radius:
                overshot = True
            elif d == "left" and cur.x < wp.target_px[0] - wp.radius:
                overshot = True
            if overshot:
                self._advance_waypoint()
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            return TaskResult(status=TaskStatus.RUNNING,
                              action=ActionResult(make_action(**{d: True, "b": True})))

        # Close-range direct walk: when within ~5 tiles, walk directly toward
        # the target without BFS. Bypasses stale-tile issues in RAM.
        # Skip for exit waypoints (need BFS to approach the exit properly).
        # If stasis is high (>40), fall through to BFS which has better
        # stuck recovery (4-direction cycling, temp_blocked tiles).
        if not wp.is_exit:
            cur = self._navigator.current_pos
            dx_close = abs(wp.target_px[0] - cur.x)
            dy_close = abs(wp.target_px[1] - cur.y)
            stasis = self._navigator.stasis
            if dx_close <= 80 and dy_close <= 80 and stasis < 40:  # ~5 tiles
                dx = wp.target_px[0] - cur.x
                dy = wp.target_px[1] - cur.y
                if abs(dx) >= abs(dy):
                    primary = "right" if dx > 0 else "left"
                    secondary = "down" if dy > 0 else "up"
                else:
                    primary = "down" if dy > 0 else "up"
                    secondary = "right" if dx > 0 else "left"
                direction = primary if stasis < 20 else secondary
                return TaskResult(status=TaskStatus.RUNNING,
                                  action=ActionResult(make_action(**{direction: True, "b": True})))

        # Stuck recovery
        if self._navigator.stasis > 180 and self._navigator.path:
            self._pathfinder.temp_blocked.add(self._navigator.path[0])
            self._navigator.path = []
            self._navigator.stasis = 0

        # BFS path (viewport-aware hopping)
        if not self._navigator.path:
            hop = self._hop_target(wp)
            goal = self._pathfinder.find_nearest_walkable(world.ram, hop, max_radius=4)
            if goal is None:
                goal = hop
            path = self._pathfinder.find_path(world.ram, self._navigator.current_tile, goal)
            if not path:
                final = (wp.target_px[0] // TILE_SIZE, wp.target_px[1] // TILE_SIZE)
                path = find_frontier_path(self._pathfinder, world.ram, self._navigator.current_tile, final)
            if path:
                self._navigator.path = path
                self._navigator.stasis = 0
                self._no_path_frames = 0
            else:
                # BFS failed — walk toward waypoint as fallback.
                # Cycle between primary and perpendicular directions
                # when stuck (stasis high) to navigate around obstacles.
                self._no_path_frames += 1
                if self._no_path_frames == 1 or self._no_path_frames % 300 == 0:
                    cur = self._navigator.current_pos
                    print(f"[MULTI_NAV] No BFS path from ({cur.x},{cur.y}), "
                          f"fallback walk (frame {self._no_path_frames})")
                cur = self._navigator.current_pos
                dx = wp.target_px[0] - cur.x
                dy = wp.target_px[1] - cur.y
                final = (wp.target_px[0] // TILE_SIZE, wp.target_px[1] // TILE_SIZE)
                loaded_direction = find_loaded_direction(world.ram, self._navigator.current_tile, final)
                if get_tile_at(world.ram, *self._navigator.current_tile) in STALE_TILE_IDS and loaded_direction is not None:
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        action=ActionResult(make_action(**{loaded_direction: True, "b": True})),
                    )
                # Primary direction toward target
                if abs(dx) >= abs(dy):
                    primary = "right" if dx > 0 else "left"
                    secondary = "down" if dy > 0 else "up"
                else:
                    primary = "down" if dy > 0 else "up"
                    secondary = "right" if dx > 0 else "left"
                # If stuck, cycle through directions every 30 frames
                stasis = self._navigator.stasis
                if stasis < 30:
                    direction = primary
                elif stasis < 60:
                    direction = secondary
                elif stasis < 90:
                    # Opposite of primary
                    opposites = {"up": "down", "down": "up", "left": "right", "right": "left"}
                    direction = opposites[primary]
                else:
                    # Opposite of secondary
                    opposites = {"up": "down", "down": "up", "left": "right", "right": "left"}
                    direction = opposites[secondary]
                return TaskResult(status=TaskStatus.RUNNING,
                                  action=ActionResult(make_action(**{direction: True, "b": True})))

        action = self._navigator.follow_path(world.ram)
        if action is not None:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))


__all__ = [
    "MAX_HOP",
    "STALE_TILE_IDS",
    "find_frontier_path",
    "find_loaded_direction",
    "NavTask",
    "CrossMapRecordedTask",
    "RecordedTransitionTask",
    "MultiMapNavTask",
]

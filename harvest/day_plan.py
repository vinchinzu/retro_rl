"""Day plan orchestrator — chains scripted tasks + BFS navigation + crop work.

Phase sequence for Day 1:
  EXIT_HOUSE → NAV_FARM_EXIT → BUY_SEEDS (cross-map) → NAV_CROP → CROP_WATER
"""

from __future__ import annotations

import os
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from farm_clearer import (
    Point,
    TileScanner,
    Pathfinder,
    Navigator,
    make_action,
    get_pos_from_ram,
    ADDR_TILEMAP,
    ADDR_INPUT_LOCK,
    TILE_SIZE,
)
from recorded_task import RecordedTask
from crop_planter import CropWaterTask

TASKS_DIR = os.path.join(_SCRIPT_DIR, "tasks")


# ── ExitBuildingTask ──────────────────────────────────────────────

# House exit path: (direction, frames) pairs.
# Player starts at (8,7) in house tilemap 0x15.
# Bed at (7,7)/(8,7) blocks direct south.  Route: left around bed,
# south through corridor, right to door column, south to exit.
HOUSE_EXIT_PATH: List[Tuple[str, int]] = [
    ("left", 50),
    ("down", 30),
    ("right", 30),
    ("down", 80),
    ("right", 30),
    ("down", 200),
]


@dataclass
class ExitBuildingTask(Task):
    """Dismiss morning dialog then walk scripted path out of house."""

    name: str = "exit_building"
    target_tilemap: int = 0x00
    dialog_frames: int = 120
    timeout: int = 600

    _step_count: int = field(default=0, init=False)
    _path_index: int = field(default=0, init=False)
    _path_frame: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._path_index = 0
        self._path_frame = 0

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1

        if self._step_count > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason="exit timeout")

        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if tilemap == self.target_tilemap and self._step_count > 30:
            return TaskResult(status=TaskStatus.SUCCESS, reason=f"tilemap=0x{tilemap:02X}")

        # Dialog dismiss phase
        if self._step_count <= self.dialog_frames:
            if self._step_count % 2 == 0:
                action = make_action(a=True)
            else:
                action = make_action(b=True)
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

        # Scripted path through house
        if self._path_index < len(HOUSE_EXIT_PATH):
            direction, frames = HOUSE_EXIT_PATH[self._path_index]
            action = make_action(**{direction: True})
            self._path_frame += 1
            if self._path_frame >= frames:
                self._path_index += 1
                self._path_frame = 0
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

        # Fallback: keep walking down
        action = make_action(down=True)
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))


# ── NavTask ───────────────────────────────────────────────────────

MAX_HOP = 10  # Max BFS hop distance (SNES viewport loads ~10 tiles around player)


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

    def can_start(self, world: WorldState) -> bool:
        return True

    def _at_target(self) -> bool:
        pos = self._navigator.current_pos
        return (abs(pos.x - self.target_px.x) <= self.radius and
                abs(pos.y - self.target_px.y) <= self.radius)

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

    def step(self, world: WorldState) -> TaskResult:
        self._navigator.update(world.ram)
        self._step_count += 1

        if self._step_count > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason="nav timeout")

        # Dialog dismissal
        input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
        if input_lock != 1:
            action = make_action(a=True) if self._step_count % 2 == 0 else make_action()
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action), reason="dialog")

        # Drain queued actions
        if self._action_queue:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(self._action_queue.popleft()))

        # Arrived?
        if self._at_target():
            return TaskResult(status=TaskStatus.SUCCESS, reason="arrived")

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
            if path:
                self._navigator.path = path
                self._navigator.stasis = 0
            else:
                # Can't path — idle frame
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        action = self._navigator.follow_path(world.ram)
        if action is not None:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
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

    _phase: str = field(default="exit", init=False)
    _step_count: int = field(default=0, init=False)
    _frames: list = field(default_factory=list, init=False)
    _frame_idx: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._phase = "exit"
        self._frame_idx = 0
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
            if tilemap != self.origin_tilemap:
                self._phase = "replay"
                print(f"[CROSS_MAP] Exited to tilemap 0x{tilemap:02X}")
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            action = make_action(**{self.exit_direction: True, "b": True})
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

        # Replay phase: check if returned to origin tilemap
        if tilemap == self.origin_tilemap and self._frame_idx > self.min_replay_before_return:
            return TaskResult(status=TaskStatus.SUCCESS, reason="returned to origin map")

        if self._frame_idx >= len(self._frames):
            return TaskResult(status=TaskStatus.SUCCESS, reason="recording complete")

        action = np.array(self._frames[self._frame_idx], dtype=np.int32)
        self._frame_idx += 1
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))


# ── DayPlanTask ───────────────────────────────────────────────────

@dataclass
class PhaseSpec:
    phase: str
    kind: str  # "exit", "nav", "recorded", "cross_map", "crop"
    params: dict = field(default_factory=dict)


# Day 1 sequence
PHASE_SEQUENCE: List[PhaseSpec] = [
    PhaseSpec("EXIT_HOUSE", "exit", {"target_tilemap": 0x00, "dialog_frames": 120, "timeout": 900}),
    PhaseSpec("NAV_FARM_EXIT", "nav", {"target_px": (40, 424), "radius": 12, "timeout": 3000}),
    PhaseSpec("BUY_SEEDS", "cross_map", {
        "exit_direction": "left",
        "recording_name": "buy_potato_seeds",
        "recording_start": 483,
        "origin_tilemap": 0x00,
        "timeout": 5000,
    }),
    PhaseSpec("NAV_CROP", "nav", {"target_px": (88, 568), "radius": 16, "timeout": 5000}),
    PhaseSpec("CROP_WATER", "crop", {"refill_bounds": (3, 45, 62, 60)}),
]


@dataclass
class DayPlanTask(Task):
    """Orchestrator: steps through PHASE_SEQUENCE, creating sub-tasks on demand."""

    name: str = "day_plan"
    seed_type: str = "potato"
    tasks_dir: str = TASKS_DIR

    _phase_index: int = field(default=0, init=False)
    _current_task: Optional[Task] = field(default=None, init=False)
    _step_count: int = field(default=0, init=False)
    _skip_map_lock: bool = field(default=False, init=False)

    def reset(self, world: WorldState) -> None:
        self._phase_index = 0
        self._current_task = None
        self._step_count = 0
        self._skip_map_lock = False

    def can_start(self, world: WorldState) -> bool:
        return True

    @property
    def skip_map_lock(self) -> bool:
        """True when a recorded task is active (may change tilemap)."""
        return self._skip_map_lock

    @property
    def phase_text(self) -> str:
        if self._phase_index < len(PHASE_SEQUENCE):
            return PHASE_SEQUENCE[self._phase_index].phase
        return "DONE"

    @property
    def progress_text(self) -> str:
        return f"phase={self._phase_index + 1}/{len(PHASE_SEQUENCE)} step={self._step_count}"

    def _make_task(self, spec: PhaseSpec, world: WorldState) -> Optional[Task]:
        """Create the sub-task for a phase spec."""
        if spec.kind == "exit":
            return ExitBuildingTask(
                target_tilemap=spec.params.get("target_tilemap", 0x00),
                dialog_frames=spec.params.get("dialog_frames", 120),
                timeout=spec.params.get("timeout", 600),
            )
        elif spec.kind == "nav":
            px = spec.params.get("target_px", (0, 0))
            return NavTask(
                name=f"nav_{spec.phase}",
                target_px=Point(px[0], px[1]),
                radius=spec.params.get("radius", 16),
                timeout=spec.params.get("timeout", 3000),
            )
        elif spec.kind == "recorded":
            task_name = spec.params.get("task_name", "")
            try:
                return RecordedTask.load(task_name, self.tasks_dir)
            except FileNotFoundError:
                print(f"[DAY_PLAN] Recording not found: {task_name}")
                return None
        elif spec.kind == "cross_map":
            return CrossMapRecordedTask(
                name=f"cross_map_{spec.phase}",
                exit_direction=spec.params.get("exit_direction", "left"),
                recording_name=spec.params.get("recording_name", ""),
                recording_start=spec.params.get("recording_start", 0),
                origin_tilemap=spec.params.get("origin_tilemap", 0x00),
                tasks_dir=self.tasks_dir,
                timeout=spec.params.get("timeout", 5000),
            )
        elif spec.kind == "crop":
            refill_bounds = spec.params.get("refill_bounds")
            return CropWaterTask(
                seed_type=self.seed_type,
                refill_bounds=refill_bounds,
            )
        return None

    def _advance(self, world: WorldState, reason: str) -> None:
        """Move to next phase."""
        phase_name = PHASE_SEQUENCE[self._phase_index].phase if self._phase_index < len(PHASE_SEQUENCE) else "?"
        print(f"[DAY_PLAN] {phase_name} -> {reason}")
        self._phase_index += 1
        self._current_task = None
        self._skip_map_lock = False

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1

        # All phases complete
        if self._phase_index >= len(PHASE_SEQUENCE):
            return TaskResult(status=TaskStatus.SUCCESS, reason="day plan complete")

        spec = PHASE_SEQUENCE[self._phase_index]

        # Create sub-task if needed
        if self._current_task is None:
            task = self._make_task(spec, world)
            if task is None:
                self._advance(world, "skipped (no task)")
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            task.reset(world)
            self._current_task = task
            self._skip_map_lock = spec.kind in ("recorded", "cross_map")
            print(f"[DAY_PLAN] Starting phase {self._phase_index + 1}/{len(PHASE_SEQUENCE)}: {spec.phase} ({spec.kind})")

        # Step the sub-task
        result = self._current_task.step(world)

        if result.status == TaskStatus.SUCCESS:
            self._advance(world, "SUCCESS")
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
        elif result.status == TaskStatus.FAILURE:
            reason = result.reason or "unknown"
            print(f"[DAY_PLAN] Phase {spec.phase} FAILED: {reason}")
            self._advance(world, f"FAILED ({reason})")
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        # Pass through RUNNING action
        if result.action is not None:
            return TaskResult(status=TaskStatus.RUNNING, action=result.action)
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

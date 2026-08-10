"""Day-plan / autoplay Task wrapper around :class:`FarmClearer`."""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import numpy as np
from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from harvest.core.animal_status import read_held_item
from harvest.core.task_progress import ProgressSnapshot
from harvest.core.tile_catalog import ADDR_TILEMAP, CLEARABLE_DEBRIS_TYPES
from harvest.paths import TASKS_DIR as PROJECT_TASKS_DIR
from harvest.planner.day_plan_status import FARM_TILEMAP, is_farm_tilemap
from harvest.planner.tasks.transitions import (
    hands_are_clear,
    multi_face_toss_actions,
    toss_held_actions,
)
from harvest.tasks.farm_clearer import (
    DebrisType,
    FarmClearer,
    Point,
    TileScanner,
)

DEFAULT_TASKS_DIR = os.fspath(PROJECT_TASKS_DIR)


@dataclass
class FarmClearTask(Task):
    """Clear weeds, stones, rocks, and stumps on the farm map.

    Stops with SUCCESS when the field is clean or stamina is too low to
    continue safely (day plan can resume tomorrow). Always tries to drop a
    held weed/rock before finishing so return-home is not blocked.
    """

    name: str = "farm_clear"
    priority: Optional[List[DebrisType]] = None
    tasks_dir: str = DEFAULT_TASKS_DIR
    timeout: int = 120000
    fetch_tools: bool = True
    prefer_lift_for_weeds: bool = True
    prefer_lift_for_stones: bool = False

    _clearer: FarmClearer = field(init=False, repr=False)
    _step_count: int = field(default=0, init=False)
    _started: bool = field(default=False, init=False)
    _drop_queue: Deque[np.ndarray] = field(default_factory=deque, init=False)
    _drop_attempts: int = field(default=0, init=False)
    _pending_finish_reason: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self._clearer = FarmClearer(priority=self.priority)
        self._configure_clearer()

    def _configure_clearer(self) -> None:
        self._clearer.tasks_dir = self.tasks_dir
        self._clearer.configure(
            prefer_lift_for_weeds=self.prefer_lift_for_weeds,
            prefer_lift_for_stones=self.prefer_lift_for_stones,
        )
        if self.fetch_tools:
            self._register_default_tool_startup()
        else:
            # No shed/tool recordings — skip inventory cycle startup.
            self._clearer.startup_done = True
            self._clearer._tool_scan_done = True

    def _register_default_tool_startup(self) -> None:
        # Position-locked get_hammer/get_axe recordings only work from their
        # original start states. Prefer inventory scan + lift-only weeds/stones
        # unless explicitly re-enabled for tool-fetch experiments.
        use_recordings = os.getenv(
            "FETCH_CLEAR_TOOL_RECORDINGS", ""
        ).lower() in ("1", "true", "yes")
        if not use_recordings:
            return

        if not os.getenv("SKIP_HAMMER", "").lower() in ("1", "true", "yes"):
            get_hammer = os.path.join(self.tasks_dir, "get_hammer.json")
            shed_grab = os.path.join(
                self.tasks_dir, "shed_grab_hammer_smash_rock.json"
            )
            if os.path.exists(get_hammer):
                self._clearer.add_startup_task("task", name="get_hammer")
            elif os.path.exists(shed_grab):
                self._clearer.add_startup_task(
                    "nav",
                    name="go_shed",
                    target=Point(342, 489),
                    radius=12,
                    timeout=1800,
                )
                self._clearer.add_startup_task(
                    "task", name="shed_grab_hammer_smash_rock"
                )

        if not os.getenv("SKIP_AXE", "").lower() in ("1", "true", "yes"):
            get_axe = os.path.join(self.tasks_dir, "get_axe.json")
            if os.path.exists(get_axe):
                self._clearer.add_startup_task("task", name="get_axe")

    @property
    def clearer(self) -> FarmClearer:
        return self._clearer

    @property
    def progress_text(self) -> str:
        phase = self._clearer.current_phase
        phase_name = phase.name if phase else self._clearer.state
        return (
            f"{phase_name} cleared={self._clearer.cleared_count} "
            f"failed={len(self._clearer.failed_tiles)}"
        )

    def progress_snapshot(self) -> ProgressSnapshot:
        target = self._clearer.current_target
        details = (
            ("cleared", self._clearer.cleared_count),
            ("failed", len(self._clearer.failed_tiles)),
            ("state", self._clearer.state),
            (
                "target",
                target.tile if target is not None else None,
            ),
            ("stamina_exhausted", self._clearer.stamina_exhausted),
        )
        return ProgressSnapshot(
            task_name=self.name,
            phase_text=self._clearer.state,
            step_count=self._step_count,
            details=details,
        )

    def reset(self, world: WorldState) -> None:
        priority = list(self.priority) if self.priority else None
        self._clearer = FarmClearer(priority=priority)
        self._configure_clearer()
        self._step_count = 0
        self._started = True
        self._drop_queue.clear()
        self._drop_attempts = 0
        self._pending_finish_reason = ""

    def can_start(self, world: WorldState) -> bool:
        ram = world.ram
        if ram is None or ADDR_TILEMAP >= len(ram):
            return False
        tilemap = int(ram[ADDR_TILEMAP])
        if not is_farm_tilemap(tilemap) and tilemap != FARM_TILEMAP:
            # Allow house/shed startup tool fetches; clearer handles map travel
            # via recordings. Debris presence is checked from farm RAM when on
            # farm; off-farm we still allow start so tools can be fetched.
            return True
        return TileScanner().has_clearable_debris(ram)

    def _finish_or_drop(self, world: WorldState, reason: str) -> TaskResult:
        """Do not hand a carried weed/rock to the next day-plan phase.

        Prefer multi-face stationary A-drop (fence_flow proven). Still SUCCESS
        with held after budget so the day can sleep — ReturnHomeTask then
        relocates to open ground and retries (rr-6g7g).
        """
        if self._drop_queue:
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(self._drop_queue.popleft()),
                reason="drop carried before clear done",
            )
        if hands_are_clear(world.ram):
            return TaskResult(status=TaskStatus.SUCCESS, reason=reason)
        drop_limit = 6
        if self._drop_attempts >= drop_limit:
            held = read_held_item(world.ram)
            print(
                f"[CLEAR] Leaving clear with held=0x{held:02X} after drop attempts"
            )
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=f"{reason}; held=0x{held:02X}",
            )
        self._drop_attempts += 1
        self._pending_finish_reason = reason
        if self._drop_attempts == 1:
            self._drop_queue.extend(toss_held_actions(face="down", step_away=True))
            self._drop_queue.extend(multi_face_toss_actions(prefer_south=True))
        else:
            self._drop_queue.extend(multi_face_toss_actions(prefer_south=True))
        print(
            f"[CLEAR] Dropping held item before done "
            f"({self._drop_attempts}/{drop_limit} "
            f"held=0x{read_held_item(world.ram):02X})"
        )
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(self._drop_queue.popleft()),
            reason="drop carried before clear done",
        )

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        if self._drop_queue:
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(self._drop_queue.popleft()),
                reason="drop carried before clear done",
            )
        if self._pending_finish_reason and not hands_are_clear(world.ram):
            return self._finish_or_drop(world, self._pending_finish_reason)
        if self._pending_finish_reason and hands_are_clear(world.ram):
            reason = self._pending_finish_reason
            self._pending_finish_reason = ""
            return TaskResult(status=TaskStatus.SUCCESS, reason=reason)

        if self._step_count > self.timeout:
            remaining = TileScanner().scan(
                world.ram, types=set(CLEARABLE_DEBRIS_TYPES)
            )
            lift_note = " lift_only" if self._clearer.tools_missing else ""
            return self._finish_or_drop(
                world,
                (
                    f"clear_budget cleared={self._clearer.cleared_count} "
                    f"remaining={len(remaining)}{lift_note}"
                ),
            )

        action = self._clearer.tick(world.ram)
        if action is None:
            if self._clearer.stamina_exhausted:
                return self._finish_or_drop(
                    world,
                    f"stamina_low cleared={self._clearer.cleared_count}",
                )
            remaining = TileScanner().scan(
                world.ram, types=set(CLEARABLE_DEBRIS_TYPES)
            )
            lift_note = (
                " lift_only" if self._clearer.tools_missing else ""
            )
            if remaining:
                return self._finish_or_drop(
                    world,
                    (
                        f"partial_clear cleared={self._clearer.cleared_count} "
                        f"remaining={len(remaining)}{lift_note}"
                    ),
                )
            return self._finish_or_drop(
                world,
                f"field_clear cleared={self._clearer.cleared_count}{lift_note}",
            )

        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(action=action),
        )


__all__ = ["FarmClearTask"]

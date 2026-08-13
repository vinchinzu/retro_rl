"""Natural Spring D2 mountain-grape pickup, return, and shipping skill."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from harvest.core.animal_status import read_held_item
from harvest.core.ram_catalog import read_ram_value
from harvest.core.task_progress import ProgressSnapshot, task_progress_snapshot
from harvest.maps.map_config import ROUTES, slice_route_from_position
from harvest.planner.tasks.navigation import MultiMapNavTask
from harvest.tasks.harvest_task import read_shipping_money
from harvest.tasks.mountain_berry import MountainBerryTask, is_mountain_forage
from harvest.tasks.nav import get_pos_from_ram, make_action
from harvest.tasks.primitives import drain_action_queue, press_a_sequence

ROUTE_NAME = "first_mountain_berry_to_shipping_bin"
VERIFY_WAIT_FRAMES = 120
DROP_RETRY_LIMIT = 2


@dataclass
class MountainGrapeShipTask(Task):
    """Pick the first mountain grape, carry it home, and ship it.

    The postcondition is domain-visible: the held forage is gone and the
    same-day shipping accumulator increased.  Wallet money is intentionally
    not checked because Harvest Moon credits it during overnight settle.
    """

    name: str = "mountain_grape_ship"
    timeout: int = 20_000
    pick_timeout: int = 12_000
    nav_timeout: int = 12_000
    pick_attempts: int = 3

    _step_count: int = field(default=0, init=False)
    _phase: str = field(default="pick", init=False)
    _child: Optional[Task] = field(default=None, init=False, repr=False)
    _shipping_before: int = field(default=0, init=False)
    _shipping_after: int = field(default=0, init=False)
    _verify_frames: int = field(default=0, init=False)
    _drop_attempts: int = field(default=0, init=False)
    _drop_queue: deque[np.ndarray] = field(default_factory=deque, init=False, repr=False)

    @property
    def phase_text(self) -> str:
        return self._phase

    @property
    def shipped_count(self) -> int:
        return int(self._phase == "done" and self._shipping_after > self._shipping_before)

    def progress_snapshot(self) -> ProgressSnapshot:
        child = task_progress_snapshot(self._child) if self._child is not None else None
        return ProgressSnapshot(
            task_name=self.name,
            phase_text=self.phase_text,
            step_count=self._step_count,
            details=(
                ("shipping_before", self._shipping_before),
                ("shipping_after", self._shipping_after),
                ("drop_attempts", self._drop_attempts),
            ),
            child=child,
        )

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._shipping_before = int(read_shipping_money(world.ram))
        self._shipping_after = self._shipping_before
        self._verify_frames = 0
        self._drop_attempts = 0
        self._drop_queue.clear()
        if is_mountain_forage(int(read_held_item(world.ram))):
            self._start_return(world)
        else:
            self._start_pick(world)

    def _start_pick(self, world: WorldState) -> None:
        self._child = MountainBerryTask(
            name=f"{self.name}_pick",
            timeout=self.pick_timeout,
            nav_timeout=min(self.nav_timeout, 6_000),
            approach_only=False,
            pick_attempts=self.pick_attempts,
        )
        self._child.reset(world)
        self._phase = "pick"

    def can_start(self, world: WorldState) -> bool:
        return bool(ROUTES.get(ROUTE_NAME))

    def _start_return(self, world: WorldState) -> None:
        held = int(read_held_item(world.ram))
        if not is_mountain_forage(held):
            self._child = None
            self._phase = "missing_forage"
            return
        route = list(ROUTES.get(ROUTE_NAME, []))
        pos = get_pos_from_ram(world.ram)
        tilemap = int(read_ram_value(world.ram, "tilemap"))
        sliced = slice_route_from_position(route, pos.x, pos.y, tilemap=tilemap)
        self._child = MultiMapNavTask(
            name=f"{self.name}_return_to_bin",
            waypoints=sliced or route,
            timeout=self.nav_timeout,
            initial_settle_frames=12,
            # A generic lift/throw recovery would discard the grape.
            allow_opportunistic_clear=False,
        )
        self._child.reset(world)
        self._phase = "return_to_bin"

    def _success_or_verify(self, world: WorldState) -> Optional[TaskResult]:
        held = int(read_held_item(world.ram))
        self._shipping_after = int(read_shipping_money(world.ram))
        if held == 0 and self._shipping_after > self._shipping_before:
            self._phase = "done"
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=(
                    "mountain grape shipped: "
                    f"shipping_money={self._shipping_before}->{self._shipping_after}"
                ),
            )
        return None

    def _step_verify(self, world: WorldState) -> TaskResult:
        success = self._success_or_verify(world)
        if success is not None:
            return success

        queued = drain_action_queue(self._drop_queue, reason="retry mountain grape bin drop")
        if queued is not None:
            return queued

        self._verify_frames += 1
        held = int(read_held_item(world.ram))
        if held != 0 and self._verify_frames >= 24 and self._drop_attempts < DROP_RETRY_LIMIT:
            self._drop_attempts += 1
            self._verify_frames = 0
            self._drop_queue.extend(
                press_a_sequence(
                    "down",
                    face_frames=6,
                    pre_press_settle_frames=6,
                    hold_frames=28,
                    settle_frames=36,
                )
            )
            queued = drain_action_queue(
                self._drop_queue,
                reason=f"retry mountain grape bin drop {self._drop_attempts}",
            )
            if queued is not None:
                return queued

        if self._verify_frames > VERIFY_WAIT_FRAMES:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=(
                    "mountain grape ship unverified: "
                    f"held=0x{held:02X} "
                    f"shipping_money={self._shipping_before}->{self._shipping_after}"
                ),
            )
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(make_action()),
            reason="verify mountain grape bin drop",
        )

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        if self._step_count > self.timeout:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"{self.name} timeout phase={self.phase_text}",
            )
        if self._phase == "missing_forage":
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason="mountain return armed without held forage",
            )
        if self._phase in {"verify", "done"}:
            success = self._success_or_verify(world)
            if success is not None:
                return success
            return self._step_verify(world)
        if self._child is None:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"{self.name} missing child phase={self.phase_text}",
            )

        result = self._child.step(world)
        if result.status == TaskStatus.RUNNING:
            return result
        if result.status in {TaskStatus.FAILURE, TaskStatus.BLOCKED}:
            return TaskResult(
                status=result.status,
                action=result.action,
                reason=f"{self.phase_text}: {result.reason or result.status.value}",
            )
        if self._phase == "pick":
            if not is_mountain_forage(int(read_held_item(world.ram))):
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason="mountain pickup reported success without held forage",
                )
            self._start_return(world)
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action()),
                reason="mountain grape kept; return to farm bin",
            )
        if self._phase == "return_to_bin":
            self._child = None
            self._phase = "verify"
            self._verify_frames = 0
            return self._step_verify(world)
        return TaskResult(
            status=TaskStatus.FAILURE,
            reason=f"unexpected successful child phase={self.phase_text}",
        )


__all__ = ["MountainGrapeShipTask", "ROUTE_NAME"]

"""Verified wild-berry pick and shipping skill.

The route is delegated to :class:`MultiMapNavTask`; this wrapper owns the
domain postcondition. Reaching the bin is not success unless the live
``shipping_money`` accumulator increased. Hands must be clear of debris so
the bush interact and bin drop are not thrash no-ops.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from harvest.core.animal_status import read_held_item
from harvest.maps.map_config import Waypoint
from harvest.planner.tasks.navigation import MultiMapNavTask
from harvest.tasks.harvest_task import read_shipping_money
from harvest.tasks.nav import get_pos_from_ram, make_action
from harvest.tasks.pond_charges import build_east_south_corridor_charge
from harvest.tasks.primitives import press_a_sequence


@dataclass
class BerryShipTask(Task):
    """Compose berry multi-nav with a fail-closed shipping-money check."""

    name: str = "berry_ship"
    waypoints: List[Waypoint] = field(default_factory=list)
    timeout: int = 18_000
    initial_settle_frames: int = 20

    _nav: MultiMapNavTask = field(init=False, repr=False)
    _shipping_before: int = field(default=0, init=False)
    _escape_queue: deque[np.ndarray] = field(default_factory=deque, init=False, repr=False)
    _escape_pending: bool = field(default=False, init=False)
    _escape_attempts: int = field(default=0, init=False)
    _drop_queue: deque[np.ndarray] = field(default_factory=deque, init=False, repr=False)
    _drop_attempts: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._nav = MultiMapNavTask(
            name=f"{self.name}_multi_nav",
            waypoints=list(self.waypoints),
            timeout=self.timeout,
            initial_settle_frames=self.initial_settle_frames,
        )

    def reset(self, world: WorldState) -> None:
        self._shipping_before = int(read_shipping_money(world.ram))
        self._escape_queue.clear()
        self._drop_queue.clear()
        self._drop_attempts = 0
        pos = get_pos_from_ram(world.ram)
        self._escape_pending = bool(
            self.waypoints
            and pos.y // 16 <= 31
            and self.waypoints[0].target_px[1] // 16 >= 35
        )
        self._escape_attempts = 0
        self._nav.waypoints = list(self.waypoints)
        self._nav.timeout = self.timeout
        self._nav.initial_settle_frames = self.initial_settle_frames
        self._nav.reset(world)

    def can_start(self, world: WorldState) -> bool:
        return bool(self.waypoints)

    def resume_after_hotswap(self, world: WorldState) -> None:
        self._nav.resume_after_hotswap(world)

    def step(self, world: WorldState) -> TaskResult:
        if self._drop_queue:
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(self._drop_queue.popleft()),
                reason="dropping debris before berry route",
            )
        if self._escape_queue:
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(self._escape_queue.popleft()),
                reason="crossing opened fence gap",
            )
        if self._escape_pending:
            pos = get_pos_from_ram(world.ram)
            tile = (pos.x // 16, pos.y // 16)
            if tile[1] >= 32:
                self._escape_pending = False
                self._nav.reset(world)
            elif self._escape_attempts >= 4:
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=f"berry fence escape failed at {tile}",
                )
            else:
                actions = build_east_south_corridor_charge(
                    tile,
                    self._escape_attempts,
                )
                self._escape_attempts += 1
                self._escape_queue.extend(actions)
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(self._escape_queue.popleft()),
                    reason=f"berry east-south escape attempt={self._escape_attempts}",
                )

        # Debris in hand blocks bush pick and thrash-seals MultNav. Drop south
        # before committing to the forage path.
        held = int(read_held_item(world.ram))
        if held != 0 and self._nav._wp_index == 0 and not self._nav._action_queue:
            if self._drop_attempts >= 3:
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=f"berry start hands not clear held=0x{held:02X}",
                )
            self._drop_attempts += 1
            self._drop_queue.extend(
                press_a_sequence(
                    "down",
                    face_frames=6,
                    pre_press_settle_frames=4,
                    hold_frames=20,
                    settle_frames=24,
                )
            )
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(self._drop_queue.popleft()),
                reason=f"pre-route drop held=0x{held:02X}",
            )

        result = self._nav.step(world)
        if result.status != TaskStatus.SUCCESS:
            return result

        shipping_now = int(read_shipping_money(world.ram))
        if shipping_now <= self._shipping_before:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=(
                    "berry ship unverified: "
                    f"shipping_money={self._shipping_before}->{shipping_now}"
                ),
            )
        return TaskResult(
            status=TaskStatus.SUCCESS,
            reason=f"berry shipped: shipping_money={self._shipping_before}->{shipping_now}",
        )


__all__ = ["BerryShipTask"]

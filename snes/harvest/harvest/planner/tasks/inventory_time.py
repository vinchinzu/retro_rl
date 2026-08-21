"""Deadline and in-game clock wait tasks used by the day planner."""

from __future__ import annotations

from dataclasses import dataclass, field

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.tasks.nav import make_action
from harvest.core.tile_catalog import ADDR_INPUT_LOCK, ADDR_TILEMAP
from harvest.tasks.primitives import dismiss_dialogue_result
from harvest.core.shipping_credit import shipping_scene_needs_dismiss
from harvest.planner.day_plan_status import (
    read_world_day_time,
    is_farm_tilemap,
)


@dataclass
class DeadlineCheckTask(Task):
    """Require the current in-game time to still be before a cutoff."""

    name: str = "deadline_check"
    latest_hour: int = 17
    latest_minute: int = 0

    def reset(self, world: WorldState) -> None:
        return None

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        _day, hour, minute = read_world_day_time(world.ram)
        if (hour, minute) >= (self.latest_hour, self.latest_minute):
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"cutoff reached at {hour:02d}:{minute:02d}",
            )
        return TaskResult(
            status=TaskStatus.SUCCESS,
            reason=f"before cutoff at {hour:02d}:{minute:02d}",
        )


@dataclass
class WaitUntilTimeTask(Task):
    """Idle until the in-game clock reaches a target time."""

    name: str = "wait_until_time"
    target_hour: int = 12
    target_minute: int = 0
    timeout: int = 4000

    _step_count: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        _day, hour, minute = read_world_day_time(world.ram)
        if (hour, minute) >= (self.target_hour, self.target_minute):
            return TaskResult(status=TaskStatus.SUCCESS, reason=f"reached {hour:02d}:{minute:02d}")
        if self._step_count > self.timeout:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"wait timeout at {hour:02d}:{minute:02d}",
            )
        input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
        if input_lock != 1:
            return dismiss_dialogue_result(self._step_count, buttons=("a", "b"), pulse_every=1)
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))


@dataclass
class FarmShippingWaitTask(Task):
    """Stay on the farm through the 5pm ShippingScene (Day09 / rr-y8n path).

    ROM: ``ShippingScene`` only fires when hour==17 and tilemap < 4. Wallet
    ``AddMoney(shipping_money)`` still runs on NightReset even if this wait is
    skipped, but the calendar loop should stay on-farm so the shipping dialogue
    plays (parity with ``harvest_ship_money_probe``).
    """

    name: str = "farm_shipping_wait"
    target_hour: int = 17
    settle_minute: int = 5
    timeout: int = 25000

    _step_count: int = field(default=0, init=False)
    _saw_input_lock: bool = field(default=False, init=False)
    _post_scene_settle: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._saw_input_lock = False
        self._post_scene_settle = 0

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        ram = world.ram
        _day, hour, minute = read_world_day_time(ram)
        tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
        lock = int(ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(ram) else 1

        if not is_farm_tilemap(tilemap):
            # Not on farm — cannot trigger ShippingScene; succeed so multi-day
            # can still sleep (NightReset credits shipping_money regardless).
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=f"off-farm tilemap=0x{tilemap:02X} hour={hour:02d}",
            )

        if shipping_scene_needs_dismiss(ram):
            self._saw_input_lock = True
            return dismiss_dialogue_result(
                self._step_count,
                buttons=("a",),
                pulse_every=2,
                reason="shipping scene",
            )

        if hour > self.target_hour or (
            hour == self.target_hour and minute >= self.settle_minute
        ):
            if lock != 1:
                return dismiss_dialogue_result(
                    self._step_count, buttons=("a", "b"), pulse_every=1
                )
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=f"farm shipping window done {hour:02d}:{minute:02d}",
            )

        if hour >= self.target_hour:
            if lock != 1:
                self._saw_input_lock = True
                return dismiss_dialogue_result(
                    self._step_count, buttons=("a", "b"), pulse_every=1
                )
            # Unlocked at/after 17:00 — brief settle then success.
            if self._saw_input_lock or self._step_count > 500:
                self._post_scene_settle += 1
                if self._post_scene_settle >= 8:
                    return TaskResult(
                        status=TaskStatus.SUCCESS,
                        reason=f"shipping scene settled {hour:02d}:{minute:02d}",
                    )
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action()),
                reason=f"post-5pm settle {hour:02d}:{minute:02d}",
            )

        if self._step_count > self.timeout:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"farm shipping wait timeout at {hour:02d}:{minute:02d}",
            )

        if lock != 1:
            return dismiss_dialogue_result(
                self._step_count, buttons=("a", "b"), pulse_every=1
            )
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(make_action()),
            reason=f"waiting farm 5pm at {hour:02d}:{minute:02d}",
        )

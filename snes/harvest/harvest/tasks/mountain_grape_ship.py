"""Natural Spring D2 mountain-grape pickup, return, and shipping skill."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from harvest.core.animal_status import read_held_item
from harvest.core.game_clock import clock_from_ram
from harvest.core.ram_catalog import read_ram_value
from harvest.core.task_progress import ProgressSnapshot, task_progress_snapshot
from harvest.maps.map_config import (
    FARM_TILEMAP_IDS,
    ROUTES,
    mountain_downhill_escape,
    mountain_exit_then_farm,
    slice_route_from_position,
)
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
    # Grapes to ship per run. The second+ pickup is best-effort: once one
    # grape has reached the bin the run reports SUCCESS even if a later
    # pick/return fails (rr-20w.3 daily spring forage).
    target_count: int = 1
    # Do not start another mountain loop at/after this hour. A loop is ~4h
    # (measured run11: 3000 f from house to bin). D3+ restock days pass 12
    # so a ~10:00 first grape still starts loop 2 (2 grapes + shop is
    # ROM-proven 13:12 / 16:08). Harvest mornings pass 9. The 10 here is
    # only the conservative D2 default.
    shop_bail_hour: int = 10
    # One house->grape->bin loop, in in-game hours. Measured run11: ~3000 f
    # at ~15 f/in-game-minute, and grape 1 lands 10:00-12:00 from a 06:00
    # start. Used to refuse a loop that cannot get back before
    # ``hard_return_hour``.
    loop_hours: int = 4
    # The farmer must be on the farm for the 17:00 ShippingScene, and a grape
    # only counts once it is in the bin. Leave an hour of slack.
    hard_return_hour: int = 16
    # Carpenter-corridor pins (run6 D15) retry a downhill suffix instead of
    # failing while still on mountain 0x10.
    max_return_retries: int = 3
    # Any leg (outbound pick or return) may be re-planned from the live pose
    # after a nav failure. run13 lost 7 of 15 berry phases to single nav pins
    # that a fresh route slice walks straight past; each retry costs a few
    # hundred frames against a ~300 G day.
    max_leg_retries: int = 3

    _step_count: int = field(default=0, init=False)
    _phase: str = field(default="pick", init=False)
    _child: Optional[Task] = field(default=None, init=False, repr=False)
    _shipped: int = field(default=0, init=False)
    _shipping_before: int = field(default=0, init=False)
    _shipping_after: int = field(default=0, init=False)
    _verify_frames: int = field(default=0, init=False)
    _drop_attempts: int = field(default=0, init=False)
    _drop_queue: deque[np.ndarray] = field(default_factory=deque, init=False, repr=False)
    _return_retries: int = field(default=0, init=False)
    _leg_retries: int = field(default=0, init=False)
    _walk_back_retries: int = field(default=0, init=False)
    _bail_reason: str = field(default="", init=False)

    @property
    def phase_text(self) -> str:
        return self._phase

    @property
    def shipped_count(self) -> int:
        return self._shipped

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
                ("leg_retries", self._leg_retries),
            ),
            child=child,
        )

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._shipping_before = int(read_shipping_money(world.ram))
        self._shipping_after = self._shipping_before
        self._verify_frames = 0
        self._drop_attempts = 0
        self._shipped = 0
        self._return_retries = 0
        self._leg_retries = 0
        self._walk_back_retries = 0
        self._bail_reason = ""
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

    def _start_nav_home(
        self, world: WorldState, *, downhill: bool, phase: str, require_forage: bool
    ) -> None:
        if require_forage and not is_mountain_forage(int(read_held_item(world.ram))):
            self._child = None
            self._phase = "missing_forage"
            return
        route = list(ROUTES.get(ROUTE_NAME, []))
        pos = get_pos_from_ram(world.ram)
        tilemap = int(read_ram_value(world.ram, "tilemap"))
        if downhill and tilemap == 0x10:
            mountain = mountain_downhill_escape(int(pos.x), int(pos.y), tilemap=tilemap)
            farm = [wp for wp in route if wp.tilemap not in (0x10, 0x0C)]
            sliced = mountain_exit_then_farm(mountain) + farm
        else:
            sliced = slice_route_from_position(route, pos.x, pos.y, tilemap=tilemap)
        self._child = MultiMapNavTask(
            name=f"{self.name}_{phase}",
            waypoints=sliced or route,
            timeout=self.nav_timeout,
            initial_settle_frames=12,
            # A generic lift/throw recovery would discard the grape.
            allow_opportunistic_clear=False,
        )
        self._child.reset(world)
        self._phase = phase

    def _start_return(self, world: WorldState, *, downhill: bool = False) -> None:
        self._start_nav_home(
            world, downhill=downhill, phase="return_to_bin", require_forage=True
        )

    def _success_or_verify(self, world: WorldState) -> Optional[TaskResult]:
        held = int(read_held_item(world.ram))
        self._shipping_after = int(read_shipping_money(world.ram))
        if held == 0 and self._shipping_after > self._shipping_before:
            return self._grape_shipped(world)
        return None

    def _grape_shipped(self, world: WorldState) -> TaskResult:
        """Count one shipped grape; loop back for the next or finish."""
        self._shipped += 1
        shipped_reason = (
            f"mountain grape {self._shipped}/{self.target_count} shipped: "
            f"shipping_money={self._shipping_before}->{self._shipping_after}"
        )
        if self._shipped >= self.target_count:
            self._phase = "done"
            return TaskResult(status=TaskStatus.SUCCESS, reason=shipped_reason)
        # Pre-flight only. The farmer is standing at the bin right now, so
        # this is the one safe moment to decide; aborting later leaves them
        # stranded on mountain 0x10 and the rest of the day's phases all fail
        # their map lock (measured: grapefix_d3_d9 D3, seed buy + establish
        # both lost that way).
        hour = int(clock_from_ram(world.ram).hour)
        if hour >= int(self.shop_bail_hour):
            return self._best_effort_success("shop window")
        if hour + int(self.loop_hours) > int(self.hard_return_hour):
            return self._best_effort_success(
                f"next loop would land after {self.hard_return_hour}:00"
            )
        # More grapes wanted: rebase the shipping baseline and forage again.
        self._shipping_before = self._shipping_after
        self._verify_frames = 0
        self._drop_attempts = 0
        self._drop_queue.clear()
        self._start_pick(world)
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(make_action()),
            reason=f"{shipped_reason}; returning for next grape",
        )

    def _start_walk_back(self, world: WorldState, *, downhill: bool = False) -> None:
        self._start_nav_home(
            world, downhill=downhill, phase="walk_back", require_forage=False
        )

    def _retry_leg(self, world: WorldState, result: TaskResult) -> Optional[TaskResult]:
        """Re-plan a failed leg from the live pose before losing the day.

        A nav failure is nearly always one pinned cell, not an unreachable
        bin: run13's six identical ``return_to_bin`` pins and three identical
        ``farm_to_path`` pins each ended a whole berry phase. Re-arming picks
        a fresh route slice from wherever the farmer actually stands, so the
        retry is a different path, not the same one replayed. Bounded by
        ``max_leg_retries`` and, above it, by the task timeout and
        ``hard_return_hour``.
        """
        if self._phase not in {"pick", "return_to_bin"}:
            return None
        if self._leg_retries >= int(self.max_leg_retries):
            return None
        self._leg_retries += 1
        why = result.reason or result.status.value
        phase = self._phase
        if phase == "pick":
            self._start_pick(world)
        else:
            tilemap = int(read_ram_value(world.ram, "tilemap"))
            self._start_return(world, downhill=tilemap == 0x10)
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(make_action()),
            reason=(
                f"{phase} retry {self._leg_retries}/{self.max_leg_retries} "
                f"after {why}"
            ),
        )

    def _abort_empty(self, world: WorldState, why: str) -> TaskResult:
        """Give up the loop with nothing in hand, but end up on the farm."""
        if self._on_farm(world):
            return TaskResult(status=TaskStatus.FAILURE, reason=why)
        self._bail_reason = why
        tilemap = int(read_ram_value(world.ram, "tilemap"))
        self._start_walk_back(world, downhill=tilemap == 0x10)
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(make_action()),
            reason=f"walking back to the farm: {why}",
        )

    def _best_effort_success(self, why: str) -> TaskResult:
        self._phase = "done"
        return TaskResult(
            status=TaskStatus.SUCCESS,
            reason=(
                f"mountain grape {self._shipped}/{self.target_count} shipped; "
                f"stopped early ({why})"
            ),
        )

    def _on_farm(self, world: WorldState) -> bool:
        return int(read_ram_value(world.ram, "tilemap")) in FARM_TILEMAP_IDS

    def _finish_or_walk_home(self, world: WorldState, why: str) -> TaskResult:
        """SUCCESS only on the farm. Off-farm, walk back; never strand later phases.

        grapefix_d3_d9 D3 reported SUCCESS from mountain 0x10, so NAV_FARM_EXIT
        failed its map lock and the day lost the seed buy and CROP_ESTABLISH.
        """
        if self._on_farm(world):
            if self._shipped < 1:
                return TaskResult(
                    status=TaskStatus.FAILURE, reason=f"no grape shipped: {why}"
                )
            return self._best_effort_success(why)
        tilemap = int(read_ram_value(world.ram, "tilemap"))
        if self._walk_back_retries >= self.max_return_retries:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=(
                    f"stranded off-farm on tilemap 0x{tilemap:02X} after "
                    f"{self._shipped} grape(s): {why}"
                ),
            )
        self._walk_back_retries += 1
        self._bail_reason = why
        self._start_walk_back(world, downhill=tilemap == 0x10)
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(make_action()),
            reason=f"walking back to the farm after {why}",
        )

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
        # Was a hard-coded ``hour >= 12`` with no map test, which both
        # shadowed shop_bail_hour (making the field dead) and could fire
        # mid-loop on mountain 0x10 — reporting SUCCESS while leaving the
        # farmer off-farm, so every later phase failed its map lock and the
        # day lost its seed buy and establish (grapefix_d3_d9 D3).
        #
        # Only give up between loops, standing on the farm. A loop already in
        # flight is bounded by ``timeout`` and by the return-leg retries, both
        # of which end with the farmer walked home.
        hour = int(clock_from_ram(world.ram).hour)
        if (
            self._shipped >= 1
            and hour >= int(self.hard_return_hour)
            and self._on_farm(world)
        ):
            return self._best_effort_success("past return deadline")
        # The bin stops crediting around the 17:00 ShippingScene: a grape
        # dropped after it leaves the farmer's hands without raising
        # ``shipping_money`` and lands on the ground, unrecoverable (run13 and
        # run14, three occurrences, every one of them at 16:00-17:01). A loop
        # is ~4 h, so an outbound leg still walking at ``hard_return_hour`` can
        # only produce that — abandon it and walk home instead of spending
        # another 3 000 frames on a grape that cannot be banked.
        if (
            self._shipped < 1
            and self._phase == "pick"
            and hour >= int(self.hard_return_hour)
            and not is_mountain_forage(int(read_held_item(world.ram)))
        ):
            return self._abort_empty(
                world, f"no loop can bank a grape past {self.hard_return_hour}:00"
            )
        if self._step_count > self.timeout:
            if self._shipped >= 1:
                return self._finish_or_walk_home(world, "timeout")
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"{self.name} timeout phase={self.phase_text}",
            )
        if self._phase == "missing_forage":
            if self._shipped >= 1:
                return self._finish_or_walk_home(
                    world, "return armed without held forage"
                )
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
            tilemap = int(read_ram_value(world.ram, "tilemap"))
            if (
                self._phase == "return_to_bin"
                and tilemap == 0x10
                and self._return_retries < self.max_return_retries
            ):
                self._return_retries += 1
                self._start_return(world, downhill=True)
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action()),
                    reason=(
                        f"return downhill retry {self._return_retries}/"
                        f"{self.max_return_retries} "
                        f"({result.reason or result.status.value})"
                    ),
                )
            # Best-effort second+ grape: one already reached the bin, so a
            # later forage/return failure still ends SUCCESS — but only once
            # the farmer is back on the farm.
            if self._shipped >= 1:
                return self._finish_or_walk_home(
                    world,
                    f"{self.phase_text}: {result.reason or result.status.value}",
                )
            retry = self._retry_leg(world, result)
            if retry is not None:
                return retry
            return TaskResult(
                status=result.status,
                action=result.action,
                reason=f"{self.phase_text}: {result.reason or result.status.value}",
            )
        if self._phase == "pick":
            if not is_mountain_forage(int(read_held_item(world.ram))):
                if self._shipped >= 1:
                    return self._finish_or_walk_home(
                        world, "pickup reported success without held forage"
                    )
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
        if self._phase == "walk_back":
            self._child = None
            return self._finish_or_walk_home(
                world, self._bail_reason or "walked back"
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

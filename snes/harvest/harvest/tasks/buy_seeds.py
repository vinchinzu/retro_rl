"""RAM-closed seed-shop buy (door + wallet, not a house→town tape).

Spring D2: MultNav to ``shop_door`` face-up, enter 0x1C, A at the clerk
until potato stock rises and money falls, then nav back onto farm 0x00.
CrossMap "returned to origin" without those deltas is a miss.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from harvest.core.ram_catalog import read_ram_value
from harvest.core.task_progress import ProgressSnapshot, task_progress_snapshot
from harvest.core.tile_catalog import ADDR_INPUT_LOCK, ADDR_TILEMAP
from harvest.maps.map_config import SEGMENTS, find_landmark, segment_waypoints, slice_route_from_position
from harvest.planner.day_plan_status import is_farm_tilemap, is_house_tilemap
from harvest.planner.tasks.inventory import ExitToFarmTask
from harvest.planner.tasks.navigation import MultiMapNavTask
from harvest.tasks.nav import get_pos_from_ram, make_action
from harvest.tasks.primitives import drain_action_queue, press_a_sequence

PATH_TILEMAP = 0x0C
TOWN_TILEMAP = 0x04
SHOP_TILEMAP = 0x1C
SHED_TILEMAP = 0x26
BARN_TILEMAP = 0x27
COOP_TILEMAP = 0x28

SHOP_NAV_SEGMENTS: Tuple[str, ...] = (
    "farm_to_path",
    "path_to_town_shop",
    "town_to_shop_door",
)
SHOP_BUY_SEGMENTS: Tuple[str, ...] = ("shop_to_counter",)
SHOP_RETURN_SEGMENTS: Tuple[str, ...] = (
    "shop_to_town",
    "town_shop_to_path",
    "path_to_farm",
)

# buy_potato_seeds_d2: first A is at (182,342) tile (11,21) face up.
SEED_CLERK_PX = (182, 342)
SEED_CLERK_RADIUS = 10
SHOP_INTERIOR_SETTLE = 45
TOWN_SETTLE_FRAMES = 40
BUY_ATTEMPT_LIMIT = 8
ENTER_PULSE_LIMIT = 90
POTATO_BAG_PRICE = 200


def _tilemap(world: WorldState) -> int:
    ram = world.ram
    return int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0


def _needs_building_exit(tilemap: int) -> bool:
    return is_house_tilemap(tilemap) or tilemap in (SHED_TILEMAP, BARN_TILEMAP, COOP_TILEMAP)


def _stock(ram, field: str = "potato_seeds") -> int:
    try:
        return int(read_ram_value(ram, field) or 0)
    except Exception:
        return 0


def _money(ram) -> int:
    try:
        return int(read_ram_value(ram, "money") or 0)
    except Exception:
        return 0


def shop_door_px() -> Tuple[int, int]:
    found = find_landmark("shop_door", tilemap_id=TOWN_TILEMAP)
    if found is None:
        return (37 * 16 + 8, 13 * 16 + 8)
    return found[1].target_px


def at_seed_clerk(ram, *, radius: int = SEED_CLERK_RADIUS) -> bool:
    pos = get_pos_from_ram(ram)
    return (
        abs(int(pos.x) - SEED_CLERK_PX[0]) <= radius
        and abs(int(pos.y) - SEED_CLERK_PX[1]) <= radius
    )


def shop_interior_coords_settled(ram) -> bool:
    """Town door pixels linger on 0x1C for a few frames after the flip."""
    pos = get_pos_from_ram(ram)
    return int(pos.x) < 400 or int(pos.y) > 250


def town_coords_settled(ram) -> bool:
    """Path-left entry leaks ~(10,128) before the east-gate stand ~(756,422)."""
    pos = get_pos_from_ram(ram)
    return int(pos.x) > 200


def first_shop_nav_segment(
    tilemap: int,
    segments: Sequence[str] = SHOP_NAV_SEGMENTS,
) -> Optional[str]:
    """Skip hops the live map has already finished."""
    if _needs_building_exit(tilemap):
        return None
    if is_farm_tilemap(tilemap):
        return segments[0] if segments else None
    if tilemap == PATH_TILEMAP:
        for name in segments:
            hops = SEGMENTS.get(name, [])
            if hops and hops[-1].tilemap in (PATH_TILEMAP, TOWN_TILEMAP) and name != "farm_to_path":
                return name
        return None
    if tilemap == TOWN_TILEMAP:
        for name in segments:
            hops = SEGMENTS.get(name, [])
            if hops and hops[0].tilemap == TOWN_TILEMAP:
                return name
        return None
    if tilemap == SHOP_TILEMAP:
        return None
    return segments[0] if segments else None


def first_shop_return_segment(tilemap: int) -> Optional[str]:
    if tilemap == SHOP_TILEMAP:
        return "shop_to_town"
    if tilemap == TOWN_TILEMAP:
        return "town_shop_to_path"
    if tilemap == PATH_TILEMAP:
        return "path_to_farm"
    return None


def purchase_closed(*, stock_before: int, stock_after: int, money_before: int, money_after: int) -> bool:
    return stock_after > stock_before and money_after < money_before


@dataclass
class BuySeedsTask(Task):
    """Farm/path/town → shop_door → clerk buy → farm, RAM-closed."""

    name: str = "buy_seeds"
    timeout: int = 18_000
    nav_timeout: int = 6_000
    stock_field: str = "potato_seeds"
    bag_price: int = POTATO_BAG_PRICE

    _step_count: int = field(default=0, init=False)
    _child: Optional[Task] = field(default=None, init=False)
    _child_name: str = field(default="", init=False)
    _done_nav: set[str] = field(default_factory=set, init=False, repr=False)
    _phase: str = field(default="nav", init=False)
    _stock_before: int = field(default=0, init=False)
    _money_before: int = field(default=0, init=False)
    _seen_shop: bool = field(default=False, init=False)
    _bought: bool = field(default=False, init=False)
    _buy_attempts: int = field(default=0, init=False)
    _enter_pulses: int = field(default=0, init=False)
    _interior_wait: int = field(default=0, init=False)
    _town_wait: int = field(default=0, init=False)
    _buy_queue: deque = field(default_factory=deque, init=False, repr=False)
    _last_reason: str = field(default="start", init=False)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._child = None
        self._child_name = ""
        self._done_nav.clear()
        self._phase = "nav"
        self._stock_before = _stock(world.ram, self.stock_field)
        self._money_before = _money(world.ram)
        self._seen_shop = _tilemap(world) == SHOP_TILEMAP
        self._bought = False
        self._buy_attempts = 0
        self._enter_pulses = 0
        self._interior_wait = 0
        self._town_wait = 0
        self._buy_queue.clear()
        self._last_reason = "start"
        self._arm_nav(world)

    def can_start(self, world: WorldState) -> bool:
        return True

    @property
    def phase_text(self) -> str:
        return self._child_name or self._phase

    def progress_snapshot(self) -> ProgressSnapshot:
        child = task_progress_snapshot(self._child) if self._child is not None else None
        return ProgressSnapshot(
            task_name=self.name,
            phase_text=self.phase_text,
            step_count=self._step_count,
            details=(
                ("stock_before", self._stock_before),
                ("money_before", self._money_before),
                ("seen_shop", int(self._seen_shop)),
                ("bought", int(self._bought)),
            ),
            child=child,
        )

    def _purchase_ok(self, ram) -> bool:
        stock = _stock(ram, self.stock_field)
        money = _money(ram)
        if purchase_closed(
            stock_before=self._stock_before,
            stock_after=stock,
            money_before=self._money_before,
            money_after=money,
        ):
            return True
        # Wallet posts inside 0x1C before the bag count. Treat a full
        # bag-price debit as the buy so we can leave and re-read stock.
        return money <= self._money_before - max(1, int(self.bag_price))

    def _success(self, ram) -> TaskResult:
        stock = _stock(ram, self.stock_field)
        money = _money(ram)
        return TaskResult(
            status=TaskStatus.SUCCESS,
            reason=(
                f"bought {self.stock_field} {self._stock_before}->{stock} "
                f"money {self._money_before}->{money}"
            ),
        )

    def _nav_for(self, name: str, world: WorldState) -> MultiMapNavTask:
        hops = list(SEGMENTS.get(name, []))
        if not hops:
            hops = segment_waypoints(name)
        pos = get_pos_from_ram(world.ram)
        tilemap = _tilemap(world)
        # Path-leak town pixels would slice to the shop door and walk off
        # the east gate. Keep the full east-to-door list until coords settle.
        if name in {"town_to_shop_door", "town_shop_to_path"} and not town_coords_settled(world.ram):
            sliced = hops
        else:
            sliced = slice_route_from_position(hops, pos.x, pos.y, tilemap=tilemap)
        timeout = self.nav_timeout
        if name in {"town_to_shop_door", "town_shop_to_path"}:
            timeout = max(timeout, 9_000)
        task = MultiMapNavTask(
            name=f"seg_{name}",
            waypoints=sliced or hops,
            timeout=timeout,
            initial_settle_frames=12,
        )
        task.reset(world)
        return task

    def _arm_nav(self, world: WorldState) -> None:
        tilemap = _tilemap(world)
        if _needs_building_exit(tilemap):
            self._child = ExitToFarmTask()
            self._child.reset(world)
            self._child_name = "exit_to_farm"
            self._last_reason = "leaving building"
            return
        if self._bought:
            if tilemap == TOWN_TILEMAP and not town_coords_settled(world.ram):
                self._child = None
                self._child_name = "town_settle"
                self._last_reason = "town settle after shop"
                return
            pos = get_pos_from_ram(world.ram)
            if tilemap == TOWN_TILEMAP and int(pos.y) < 268:
                # Doorface ~(601,232). Walk south off (37,14) before MultNav.
                self._child = None
                self._child_name = "leave_door"
                self._last_reason = "walk south off shop door"
                return
            nxt = first_shop_return_segment(tilemap)
            if nxt is None or nxt in self._done_nav:
                self._child = None
                self._child_name = "done"
                return
            self._child = self._nav_for(nxt, world)
            self._child_name = nxt
            self._last_reason = f"nav {nxt}"
            return
        if tilemap == SHOP_TILEMAP:
            self._seen_shop = True
            if not shop_interior_coords_settled(world.ram) and self._interior_wait < SHOP_INTERIOR_SETTLE:
                self._child = None
                self._child_name = "shop_settle"
                self._last_reason = "shop interior settle"
                return
            if at_seed_clerk(world.ram):
                self._child = None
                self._child_name = "buy"
                self._phase = "buy"
                self._last_reason = "at clerk"
                return
            if "shop_to_counter" not in self._done_nav:
                self._child = self._nav_for("shop_to_counter", world)
                self._child_name = "shop_to_counter"
                self._last_reason = "nav shop_to_counter"
                return
            self._child = None
            self._child_name = "buy"
            self._phase = "buy"
            return
        if tilemap == TOWN_TILEMAP and not town_coords_settled(world.ram) and self._town_wait < TOWN_SETTLE_FRAMES:
            self._child = None
            self._child_name = "town_settle"
            self._last_reason = "town east-gate settle"
            return
        remaining = [name for name in SHOP_NAV_SEGMENTS if name not in self._done_nav]
        nxt = first_shop_nav_segment(tilemap, remaining)
        if nxt is None:
            self._child = None
            self._child_name = "enter"
            self._phase = "enter"
            self._last_reason = "at shop door"
            return
        self._child = self._nav_for(nxt, world)
        self._child_name = nxt
        self._last_reason = f"nav {nxt}"

    def _queue_buy(self, world: WorldState, *, lock: int) -> TaskResult:
        if self._buy_attempts >= BUY_ATTEMPT_LIMIT:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=(
                    f"shop buy unverified after {self._buy_attempts} A "
                    f"stock={self._stock_before} money={self._money_before}"
                ),
            )
        pos = get_pos_from_ram(world.ram)
        # Tape last step is Up onto (182,342), then A in place. Extra Up
        # walks off the clerk into the counter.
        if abs(int(pos.y) - SEED_CLERK_PX[1]) > 2:
            face = "up" if int(pos.y) > SEED_CLERK_PX[1] else "down"
            self._buy_queue.extend(
                press_a_sequence(
                    face,
                    face_frames=min(8, abs(int(pos.y) - SEED_CLERK_PX[1])),
                    pre_press_settle_frames=12,
                    hold_frames=10,
                    settle_frames=36,
                    hold_face_with_a=False,
                )
            )
        else:
            self._buy_queue.extend(
                press_a_sequence(
                    None,
                    face_frames=0,
                    pre_press_settle_frames=20 if lock == 1 else 8,
                    hold_frames=8,
                    settle_frames=40 if lock != 2 else 28,
                )
            )
        self._buy_attempts += 1
        self._last_reason = f"buy attempt={self._buy_attempts} lock={lock}"
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(self._buy_queue.popleft()),
            reason=self._last_reason,
        )

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        if self._step_count > self.timeout:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"{self.name} timeout after {self._step_count}f phase={self.phase_text}",
            )

        tilemap = _tilemap(world)
        if tilemap == SHOP_TILEMAP:
            if not self._seen_shop:
                self._town_wait = 0
            self._seen_shop = True
        elif self._bought and tilemap == TOWN_TILEMAP and self._child_name == "shop_to_town":
            # Fresh settle after the shop→town flip (inbound wait already spent).
            self._town_wait = 0
        if self._purchase_ok(world.ram):
            self._bought = True
        lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
        if self._bought and tilemap == SHOP_TILEMAP and lock in {0, 2, 4}:
            queued = drain_action_queue(self._buy_queue)
            if queued is not None:
                return queued
            self._buy_queue.extend(
                press_a_sequence(None, face_frames=0, pre_press_settle_frames=6, hold_frames=6, settle_frames=18)
            )
            self._last_reason = "dismiss shop menu"
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(self._buy_queue.popleft()),
                reason=self._last_reason,
            )
        if (
            self._bought
            and self._seen_shop
            and is_farm_tilemap(tilemap)
        ):
            return self._success(world.ram)

        if self._child_name == "leave_door":
            pos = get_pos_from_ram(world.ram)
            if int(pos.y) >= 268 or _tilemap(world) != TOWN_TILEMAP:
                self._arm_nav(world)
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action(down=True, b=True)),
                reason=self._last_reason,
            )

        if self._child_name == "town_settle":
            self._town_wait += 1
            # Inbound may proceed after a cap (full east-gate list). After
            # shop, leak (138,468) walking toward (602,274) re-enters 0x1C.
            ready = town_coords_settled(world.ram)
            inbound = not self._bought
            if ready or (inbound and self._town_wait >= TOWN_SETTLE_FRAMES):
                self._arm_nav(world)
            elif self._bought and self._town_wait > 180:
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason="town coords never settled after shop exit",
                )
            action = make_action(down=True) if self._bought else make_action()
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(action),
                reason=self._last_reason,
            )

        if self._child_name == "shop_settle":
            self._interior_wait += 1
            if shop_interior_coords_settled(world.ram) or self._interior_wait >= SHOP_INTERIOR_SETTLE:
                self._arm_nav(world)
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action()),
                reason=self._last_reason,
            )

        if self._phase == "enter" and tilemap != SHOP_TILEMAP:
            self._enter_pulses += 1
            if self._enter_pulses > ENTER_PULSE_LIMIT:
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason="shop_door enter failed; MultNav did not reach 0x1C",
                )
            # Face-up A then walk up. Door is the landmark open face.
            if self._enter_pulses % 20 < 8:
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action(a=True)),
                    reason="shop_door A",
                )
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action(up=True)),
                reason="shop_door walk up",
            )
        if self._phase == "enter" and tilemap == SHOP_TILEMAP:
            self._phase = "nav"
            self._enter_pulses = 0
            self._arm_nav(world)

        if self._phase == "buy" and not self._bought:
            lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
            queued = drain_action_queue(self._buy_queue)
            if queued is not None:
                return queued
            if lock in {0, 2, 4} and self._buy_attempts > 0:
                # Dialogue still up — keep A pulses. Buy is cursor 0.
                return self._queue_buy(world, lock=lock)
            if at_seed_clerk(world.ram) or self._buy_attempts == 0:
                return self._queue_buy(world, lock=lock)
            self._phase = "nav"
            self._arm_nav(world)

        if self._child is not None:
            result = self._child.step(world)
            if result.status == TaskStatus.RUNNING:
                return result
            name = self._child_name
            self._child = None
            if result.status == TaskStatus.FAILURE:
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=f"{name} failed: {result.reason}",
                )
            if name in SHOP_NAV_SEGMENTS or name in SHOP_BUY_SEGMENTS or name in SHOP_RETURN_SEGMENTS:
                self._done_nav.add(name)
            if name == "shop_to_counter":
                self._phase = "buy"
            self._arm_nav(world)
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action()),
                reason=self._last_reason,
            )

        if self._bought and not is_farm_tilemap(tilemap):
            self._arm_nav(world)
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action()),
                reason=self._last_reason,
            )

        if not self._bought and tilemap == TOWN_TILEMAP:
            self._phase = "enter"
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action(up=True)),
                reason="shop_door walk up",
            )

        return TaskResult(
            status=TaskStatus.FAILURE,
            reason=(
                f"shop unfinished tm=0x{tilemap:02X} seen={self._seen_shop} "
                f"bought={self._bought}"
            ),
        )


__all__ = [
    "BuySeedsTask",
    "PATH_TILEMAP",
    "POTATO_BAG_PRICE",
    "SEED_CLERK_PX",
    "SHOP_NAV_SEGMENTS",
    "SHOP_RETURN_SEGMENTS",
    "SHOP_TILEMAP",
    "TOWN_TILEMAP",
    "at_seed_clerk",
    "first_shop_nav_segment",
    "first_shop_return_segment",
    "purchase_closed",
    "shop_door_px",
    "town_coords_settled",
]

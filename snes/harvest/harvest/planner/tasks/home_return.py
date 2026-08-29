"""Return-home task: farm recovery, drop hands, enter house.

Geometry helpers live in :mod:`home_approach`; failure policy in
:mod:`home_recover`. Public imports should prefer
``harvest.planner.tasks.home``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.tasks.nav import (
    Point,
    get_pos_from_ram,
)
from harvest.core.tile_catalog import ADDR_TILEMAP
from harvest.maps.map_config import (
    ROUTES,
    Waypoint,
)
from harvest.core.ram_catalog import field_spec, read_ram_u16
from harvest.planner.day_plan_status import (
    TASKS_DIR,
    FARM_TILEMAP,
    is_farm_tilemap,
    is_house_tilemap,
)
from harvest.planner.tasks.home_approach import (
    EAST_AROUND_FENCE_X,
    build_house_approach_waypoints,
    deep_south_of_house,
    drop_spot_px,
    far_east_of_pond_lane,
    house_enter_task,
    south_of_fence_wall,
)
from harvest.planner.tasks.home_recover import (
    RecoverDecision,
    RecoverKind,
    decide_child_failure,
    drop_carried_actions,
    enter_fail_south_recovery_actions,
    exit_to_farm_recover_actions,
    short_east_north_actions,
    south_escape_actions,
)
from harvest.planner.tasks.inventory import ExitToFarmTask
from harvest.planner.tasks.navigation import MultiMapNavTask, NavTask
from harvest.core.animal_status import read_held_item
from harvest.planner.tasks.transitions import hands_are_clear

# Return-home routes live in map_config and are selected by upgrade state; these
# constants are fallbacks for any missing route data.
HOUSE_FRONT_PX = Point(136, 424)
HOUSE_DOOR_FRONT_PX = Point(136, 424)


@dataclass
class ReturnHomeTask(Task):
    """Recover to the farm if needed, then enter the farmhouse."""

    name: str = "return_home"
    tasks_dir: str = TASKS_DIR
    # Hard budget so multi-day soaks fail cleanly instead of hanging ~D5 when
    # enter_house / hands-full / nav soft-stalls without terminal status.
    # South-of-fence densified approach (east around wall) needs headroom.
    timeout: int = 11000

    _phase: str = field(default="start", init=False)
    _task: Optional[Task] = field(default=None, init=False)
    _action_queue: deque = field(default_factory=deque, init=False)
    _drop_attempts: int = field(default=0, init=False)
    _drop_spot_navs: int = field(default=0, init=False)
    _drop_same_held: int = field(default=0, init=False)
    _drop_last_held: int = field(default=-1, init=False)
    _drop_deep_relocated: bool = field(default=False, init=False)
    _enter_retries: int = field(default=0, init=False)
    _exit_to_farm_retries: int = field(default=0, init=False)
    _total_steps: int = field(default=0, init=False)
    # Soft-success off-stand re-navs can thrash forever without terminal
    # status if the door stand soft-radius keeps "succeeding" a few tiles
    # away. Cap corrections so the outer timeout is not the only exit.
    _offstand_corrections: int = field(default=0, init=False)
    _best_door_dist: int = field(default=99999, init=False)
    # Hard budget for in-place toss cycles after relocating to open ground.
    # Keep well under outer timeout: each multi-face cycle is ~250f and prior
    # nav/escape already burned frames (power-on D19 drop_carried@0x0F).
    drop_attempt_limit: int = 6
    # Same held id across this many cycles → hard-fail (stuck rock fragment).
    drop_stuck_held_limit: int = 4
    # rr-uru1: ExitToFarm dialogue/unknown-map thrash budget before hard fail.
    exit_to_farm_retry_limit: int = 3
    # SW debris softlock escape + densified re-nav budget (rr-5in D9).
    # Shared by pre-escape and post-nav recover — not a one-shot flag so a
    # later drop→south-of-fence approach can still pre-escape (D19 residual).
    _south_escape_attempts: int = field(default=0, init=False)
    south_escape_limit: int = 4

    @staticmethod
    def _house_route_name(ram: np.ndarray) -> str:
        upgrade_flags = read_ram_u16(ram, field_spec("upgrade_flags").address)
        if upgrade_flags & 0x80:
            return "farm_to_house_level2"
        if upgrade_flags & 0x40:
            return "farm_to_house_level1"
        return "farm_to_house"

    @classmethod
    def _house_front_px(cls, ram: np.ndarray) -> Point:
        route_name = cls._house_route_name(ram)
        route = ROUTES.get(route_name) or ROUTES.get("farm_to_house") or []
        if route:
            x, y = route[-1].target_px
            return Point(x, y)
        return HOUSE_FRONT_PX

    @classmethod
    def _house_enter_task(cls, world: WorldState):
        """Build the outdoor→house doorway push (geometry in home_approach)."""
        return house_enter_task(cls._house_front_px(world.ram))

    def reset(self, world: WorldState) -> None:
        self._phase = "start"
        self._task = None
        self._action_queue.clear()
        self._drop_attempts = 0
        self._drop_spot_navs = 0
        self._drop_same_held = 0
        self._drop_last_held = -1
        self._drop_deep_relocated = False
        self._enter_retries = 0
        self._exit_to_farm_retries = 0
        self._total_steps = 0
        self._offstand_corrections = 0
        self._best_door_dist = 99999
        self._south_escape_attempts = 0

    def can_start(self, world: WorldState) -> bool:
        return True

    def _at_drop_spot(self, pos: Point, front: Point) -> bool:
        drop = drop_spot_px(front, deep=self._drop_deep_relocated)
        return abs(pos.x - drop.x) <= 28 and abs(pos.y - drop.y) <= 28

    @classmethod
    def _house_approach_waypoints(
        cls, ram: np.ndarray, front: Point, pos: Point
    ) -> List[Waypoint]:
        """Densified farm→door waypoints (geometry in home_approach)."""
        route_name = cls._house_route_name(ram)
        base = list(ROUTES.get(route_name) or ROUTES.get("farm_to_house") or [])
        return build_house_approach_waypoints(base, front, pos, ram)

    def _read_tilemap(self, world: WorldState) -> int:
        return int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0

    def _house_arrival_success(
        self, tilemap: int, *, via: str
    ) -> Optional[TaskResult]:
        """SUCCESS when already on a house tilemap (any phase / timeout)."""
        if not is_house_tilemap(tilemap):
            return None
        return TaskResult(
            status=TaskStatus.SUCCESS,
            reason=(
                f"already in house tilemap=0x{tilemap:02X} "
                f"phase={self._phase} via={via}"
            ),
        )

    def _queue_drop_carried(self) -> None:
        self._action_queue.extend(drop_carried_actions(self._drop_attempts))

    def _queue_south_escape(
        self, *, long_east: bool = False, far_east: bool = False
    ) -> None:
        self._action_queue.extend(
            south_escape_actions(long_east=long_east, far_east=far_east)
        )

    def _activate(self, phase: str, task: Task, world: WorldState) -> None:
        self._phase = phase
        self._task = task
        task.reset(world)

    def _nav_to_house_front(self, world: WorldState, front: Point) -> TaskResult:
        # Cap child nav so outer timeout can fire; leave headroom for enter.
        child_timeout = min(4500, max(800, self.timeout - self._total_steps - 400))
        self._activate(
            "nav_house_front",
            NavTask(
                name="nav_house_front",
                target_px=front,
                radius=8,
                soft_radius=14,
                soft_stasis=45,
                timeout=child_timeout,
            ),
            world,
        )
        return self._task.step(world)

    def _nav_to_drop_spot(self, world: WorldState, front: Point) -> TaskResult:
        """Walk south of the door so throws are not blocked by the house wall.

        From deep south, use densified multi_nav (same as house approach) so
        we are not stuck with single-point NavTask in the SW pocket.
        """
        drop = drop_spot_px(front, deep=self._drop_deep_relocated)
        pos = get_pos_from_ram(world.ram)
        child_timeout = min(4000, max(800, self.timeout - self._total_steps - 400))
        if deep_south_of_house(pos, front):
            # Approach via corridor then finish at drop (not door stand).
            wps = self._house_approach_waypoints(world.ram, front, pos)
            # Replace final door stand with drop spot.
            if wps:
                wps = list(wps[:-1]) + [
                    Waypoint(tilemap=0x00, target_px=(drop.x, drop.y), radius=16)
                ]
            else:
                wps = [Waypoint(tilemap=0x00, target_px=(drop.x, drop.y), radius=16)]
            self._activate(
                "nav_drop_spot",
                MultiMapNavTask(
                    name="nav_drop_spot",
                    waypoints=wps,
                    timeout=child_timeout,
                    initial_settle_frames=0,
                ),
                world,
            )
            return self._task.step(world)
        self._activate(
            "nav_drop_spot",
            NavTask(
                name="nav_drop_spot",
                target_px=drop,
                radius=14,
                soft_radius=22,
                soft_stasis=40,
                timeout=child_timeout,
            ),
            world,
        )
        return self._task.step(world)

    def _queue_short_east_north(self) -> None:
        self._action_queue.extend(short_east_north_actions())

    def _start_house_approach(self, world: WorldState, front: Point) -> TaskResult:
        """Activate multi_nav (densified from south) or simple NavTask."""
        pos = get_pos_from_ram(world.ram)
        remaining = self.timeout - self._total_steps if self.timeout > 0 else 99999
        # Low budget south of fence: last-ditch short charge, else fail clean.
        # Power-on D19: drop thrash leaves ~(153,518) with ~1.1k frames left —
        # too little for multi_nav, but enough for a compact east→north.
        if (
            south_of_fence_wall(pos)
            and remaining < 2500
            and abs(pos.x - front.x) + abs(pos.y - front.y) > 80
        ):
            if (
                remaining >= 700
                and pos.x < EAST_AROUND_FENCE_X - 16
                and self._south_escape_attempts < self.south_escape_limit + 1
            ):
                self._south_escape_attempts += 1
                self._phase = "south_escape"
                self._queue_short_east_north()
                print(
                    f"[RETURN_HOME] Low-budget east→north pos=({pos.x},{pos.y}) "
                    f"remaining={remaining}f"
                )
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(self._action_queue.popleft()),
                    reason="low-budget east-north charge",
                )
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=(
                    f"return_home budget exhausted south of fence "
                    f"pos=({pos.x},{pos.y}) remaining={remaining}f"
                ),
            )
        # South of y=31 wall off the free northbound lane: scripted escape
        # *before* multi_nav when geometry is hostile.
        #
        # Always pre-escape SW pocket + far-east pond thrash zones.
        # Mid-south (x≈150–500) only after drop thrash or a prior escape this
        # return_home — unconditional mid-south pre-escape burned ~3× frames
        # per day and hit the end-of-spring planner budget at D12.
        far_east = far_east_of_pond_lane(pos)
        sw_pocket = pos.x < 200
        mid_south = (
            not far_east
            and not sw_pocket
            and pos.x < EAST_AROUND_FENCE_X - 32
        )
        mid_south_armed = (
            mid_south
            and (
                self._south_escape_attempts > 0
                or self._drop_attempts > 0
                or self._drop_deep_relocated
            )
        )
        need_pre_escape = (
            south_of_fence_wall(pos)
            and self._south_escape_attempts < self.south_escape_limit
            and remaining > 1200
            and (far_east or sw_pocket or mid_south_armed)
        )
        if need_pre_escape:
            self._south_escape_attempts += 1
            self._phase = "south_escape"
            if far_east:
                self._queue_south_escape(far_east=True)
                label = "far-east pond"
            elif sw_pocket:
                self._queue_south_escape(long_east=True)
                label = "SW pocket"
            else:
                # Mid-corridor after drop/fail: east then north.
                self._queue_south_escape(long_east=False)
                label = "mid-south fence"
            print(
                f"[RETURN_HOME] Pre-escape {label} pos=({pos.x},{pos.y}) "
                f"before house approach "
                f"({self._south_escape_attempts}/{self.south_escape_limit})"
            )
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(self._action_queue.popleft()),
                reason=f"pre-escape {label}",
            )
        waypoints = self._house_approach_waypoints(world.ram, front, pos)
        # Cap child nav so drop/escape recovery still has budget after a long
        # first approach (power-on D19 burned ~8k on the opening multi_nav).
        route_timeout = min(3500, max(800, remaining - 4000))
        if remaining < 5000:
            route_timeout = min(route_timeout, max(600, remaining - 1500))
        print(
            f"[RETURN_HOME] House approach from ({pos.x},{pos.y}) → "
            f"({front.x},{front.y}) wps={len(waypoints)} "
            f"deep_south={deep_south_of_house(pos, front)} "
            f"south_of_fence={south_of_fence_wall(pos)} "
            f"route_timeout={route_timeout}"
        )
        self._activate(
            "nav_house_front",
            MultiMapNavTask(
                name="nav_house_front",
                waypoints=waypoints,
                timeout=route_timeout,
                initial_settle_frames=0,
            ),
            world,
        )
        return self._task.step(world)

    def _start_next_phase(self, world: WorldState) -> TaskResult:
        tilemap = self._read_tilemap(world)
        arrived = self._house_arrival_success(tilemap, via="start_next_phase")
        if arrived is not None:
            return arrived
        if not is_farm_tilemap(tilemap):
            self._activate("exit_to_farm", ExitToFarmTask(tasks_dir=self.tasks_dir), world)
            return self._task.step(world)

        pos = get_pos_from_ram(world.ram)
        front = self._house_front_px(world.ram)

        if not hands_are_clear(world.ram):
            # Always relocate to open ground south of the house before thrashing
            # A-drops in a debris field (rr-6g7g: CLEAR leaves held=0x0D).
            held = int(read_held_item(world.ram))
            at_drop = self._at_drop_spot(pos, front)
            if not at_drop and self._drop_spot_navs < 3:
                self._drop_spot_navs += 1
                print(
                    f"[RETURN_HOME] Moving to open drop spot "
                    f"pos=({pos.x},{pos.y}) held=0x{held:02X} "
                    f"(nav {self._drop_spot_navs}/3)"
                )
                return self._nav_to_drop_spot(world, front)
            # Stuck same held id (power-on D19 rock fragment 0x0F): do not burn
            # the full outer timeout on multi-face thrash.
            if held == self._drop_last_held:
                self._drop_same_held += 1
            else:
                self._drop_same_held = 1
                self._drop_last_held = held
            # After a couple same-held thrash cycles at the primary drop, try a
            # deeper south stand once (debris re-pickup / soft-block at y≈480).
            if (
                self._drop_same_held >= 2
                and not self._drop_deep_relocated
                and self._drop_spot_navs < 5
            ):
                self._drop_deep_relocated = True
                self._drop_spot_navs += 1
                deep = drop_spot_px(front, deep=True)
                print(
                    f"[RETURN_HOME] Deep drop relocate held=0x{held:02X} "
                    f"→ ({deep.x},{deep.y}) after same_held="
                    f"{self._drop_same_held}"
                )
                child_timeout = min(
                    2500, max(600, self.timeout - self._total_steps - 400)
                )
                self._activate(
                    "nav_drop_spot",
                    MultiMapNavTask(
                        name="nav_drop_spot_deep",
                        waypoints=[
                            Waypoint(
                                tilemap=FARM_TILEMAP,
                                target_px=(deep.x, deep.y),
                                radius=16,
                            )
                        ],
                        timeout=child_timeout,
                    ),
                    world,
                )
                return self._task.step(world)
            if (
                self._drop_same_held >= self.drop_stuck_held_limit
                or self._drop_attempts >= self.drop_attempt_limit
            ):
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=(
                        "could not clear hands before house entry "
                        f"(held=0x{held:02X} attempts={self._drop_attempts} "
                        f"same_held={self._drop_same_held})"
                    ),
                )
            if self._drop_attempts < self.drop_attempt_limit:
                self._drop_attempts += 1
                self._phase = "drop_carried"
                self._queue_drop_carried()
                print(
                    f"[RETURN_HOME] Dropping carried item before house entry "
                    f"({self._drop_attempts}/{self.drop_attempt_limit} "
                    f"held=0x{held:02X} at_drop={at_drop} "
                    f"same_held={self._drop_same_held})"
                )
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(self._action_queue.popleft()),
                    reason="drop carried before house",
                )
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=(
                    "could not clear hands before house entry "
                    f"(held=0x{held:02X})"
                ),
            )

        # Too far north of the outdoor stand: walk south before pushing up.
        north_of_stand = abs(pos.x - front.x) <= 24 and pos.y < front.y - 10
        if north_of_stand:
            print(
                f"[RETURN_HOME] North of door stand pos=({pos.x},{pos.y}); "
                f"re-nav to house front ({front.x},{front.y})"
            )
            return self._nav_to_house_front(world, front)

        door_dist = abs(pos.x - front.x) + abs(pos.y - front.y)
        if door_dist < self._best_door_dist:
            self._best_door_dist = door_dist

        at_stand = abs(pos.x - front.x) <= 16 and abs(pos.y - front.y) <= 16
        if at_stand:
            self._activate("enter_house", self._house_enter_task(world), world)
            return self._task.step(world)

        # Near door but slightly off-stand after several corrections: push enter
        # rather than re-nav forever (D5 hang residual).
        near_door = abs(pos.x - front.x) <= 28 and abs(pos.y - front.y) <= 28
        if near_door and self._offstand_corrections >= 3:
            print(
                f"[RETURN_HOME] Near door after {self._offstand_corrections} "
                f"corrections pos=({pos.x},{pos.y}); forcing enter"
            )
            self._activate("enter_house", self._house_enter_task(world), world)
            return self._task.step(world)

        if self._phase not in {"nav_house_front", "nav_drop_spot"}:
            return self._start_house_approach(world, front)
        return self._nav_to_house_front(world, front)

    def _apply_recover(
        self,
        decision: RecoverDecision,
        world: WorldState,
        pos: Point,
        front: Point,
        reason: str,
    ) -> TaskResult:
        """Apply a home_recover decision (counters, queues, prints, re-entry)."""
        kind = decision.kind
        if kind == RecoverKind.QUEUE_EXIT_MASH:
            self._task = None
            self._exit_to_farm_retries += 1
            print(
                f"[RETURN_HOME] ExitToFarm recover "
                f"({self._exit_to_farm_retries}/"
                f"{self.exit_to_farm_retry_limit}): {reason}"
            )
            self._action_queue.extend(exit_to_farm_recover_actions())
            self._phase = "exit_to_farm_recover"
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(self._action_queue.popleft()),
                reason="exit_to_farm recover mash",
            )
        if kind == RecoverKind.FAIL_EXIT:
            self._task = None
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"exit_to_farm failed after retries: {reason}",
            )
        if kind in {RecoverKind.RETRY_ENTER_SOUTH, RecoverKind.RETRY_ENTER_RESTART}:
            self._enter_retries += 1
            self._task = None
            self._phase = "start"
            if decision.hands_not_clear or "hands not clear" in reason:
                self._drop_attempts = 0
                self._drop_spot_navs = 0
            print(
                f"[RETURN_HOME] Retry house enter "
                f"({self._enter_retries}/4): {reason}"
            )
            held = read_held_item(world.ram)
            print(
                f"[RETURN_HOME] Enter fail diagnostics "
                f"pos=({pos.x},{pos.y}) front=({front.x},{front.y}) "
                f"held=0x{held:02X} hands_clear={hands_are_clear(world.ram)}"
            )
            if kind == RecoverKind.RETRY_ENTER_SOUTH:
                actions = enter_fail_south_recovery_actions(pos, front)
                if actions:
                    self._action_queue.extend(actions)
                    self._phase = "drop_carried"
                    self._offstand_corrections = 0
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        action=ActionResult(self._action_queue.popleft()),
                        reason="south recovery after enter fail",
                    )
            return self._start_next_phase(world)
        if kind == RecoverKind.RETRY_DROP_THEN_NAV:
            self._task = None
            self._phase = "start"
            print(
                f"[RETURN_HOME] Nav failed with hands full; "
                f"drop then retry: {reason}"
            )
            return self._start_next_phase(world)
        if kind == RecoverKind.FORCE_ENTER:
            print(
                f"[RETURN_HOME] Nav failed near door "
                f"pos=({pos.x},{pos.y}): {reason}; forcing enter"
            )
            self._activate(
                "enter_house",
                self._house_enter_task(world),
                world,
            )
            return self._task.step(world)
        if kind == RecoverKind.MID_YARD_RENAV:
            self._offstand_corrections += 1
            self._task = None
            print(
                f"[RETURN_HOME] Mid-yard re-nav after fail "
                f"pos=({pos.x},{pos.y}) "
                f"({self._offstand_corrections}/6): {reason}"
            )
            return self._nav_to_house_front(world, front)
        if kind == RecoverKind.SOUTH_ESCAPE:
            self._south_escape_attempts += 1
            self._task = None
            self._phase = "south_escape"
            self._queue_south_escape(far_east=decision.far_east)
            if decision.escape_from_drop:
                print(
                    f"[RETURN_HOME] South softlock escape (drop) "
                    f"({self._south_escape_attempts}/"
                    f"{self.south_escape_limit}) pos=({pos.x},{pos.y}): "
                    f"{reason}"
                )
                escape_reason = "south escape after drop-spot nav fail"
            else:
                print(
                    f"[RETURN_HOME] South softlock escape "
                    f"({self._south_escape_attempts}/"
                    f"{self.south_escape_limit}) pos=({pos.x},{pos.y}): "
                    f"{reason}"
                )
                escape_reason = "south escape after house nav fail"
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(self._action_queue.popleft()),
                reason=escape_reason,
            )
        # HARD_FAIL
        if decision.clear_task:
            self._task = None
        if decision.set_phase is not None:
            self._phase = decision.set_phase
        return TaskResult(
            status=TaskStatus.FAILURE,
            reason=f"{self._phase} failed: {reason}",
        )

    def step(self, world: WorldState) -> TaskResult:
        self._total_steps += 1
        tilemap = self._read_tilemap(world)
        # House arrival short-circuit: already inside is SUCCESS in any phase
        # (rr-ws8h: exit_to_farm child could run until hard timeout while house).
        arrived = self._house_arrival_success(tilemap, via="step")
        if arrived is not None:
            return arrived
        if self.timeout > 0 and self._total_steps > self.timeout:
            # Defense in depth: re-check house on the timeout path as well.
            arrived = self._house_arrival_success(tilemap, via="timeout")
            if arrived is not None:
                return arrived
            held = int(read_held_item(world.ram))
            held_note = ""
            if self._phase == "drop_carried" or not hands_are_clear(world.ram):
                held_note = f" held=0x{held:02X}"
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=(
                    f"return_home timeout after {self._total_steps}f "
                    f"phase={self._phase}{held_note}"
                ),
            )
        if self._action_queue:
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(self._action_queue.popleft()),
            )
        if self._phase == "drop_carried":
            self._task = None
            return self._start_next_phase(world)
        if self._phase == "exit_to_farm_recover":
            # Mash queue drained; re-run exit_to_farm / farm approach.
            self._task = None
            self._phase = "start"
            return self._start_next_phase(world)
        if self._phase == "south_escape":
            # Escape B-run finished; densified re-nav from new position.
            self._task = None
            self._phase = "start"
            return self._start_next_phase(world)
        if self._task is None:
            return self._start_next_phase(world)

        result = self._task.step(world)
        if result.status == TaskStatus.RUNNING:
            return result
        if result.status in {TaskStatus.FAILURE, TaskStatus.BLOCKED}:
            reason = result.reason or "unknown"
            pos = get_pos_from_ram(world.ram)
            front = self._house_front_px(world.ram)
            decision = decide_child_failure(
                phase=self._phase,
                pos=pos,
                front=front,
                reason=reason,
                hands_clear=hands_are_clear(world.ram),
                exit_to_farm_retries=self._exit_to_farm_retries,
                exit_to_farm_retry_limit=self.exit_to_farm_retry_limit,
                enter_retries=self._enter_retries,
                drop_attempts=self._drop_attempts,
                drop_attempt_limit=self.drop_attempt_limit,
                offstand_corrections=self._offstand_corrections,
                south_escape_attempts=self._south_escape_attempts,
                south_escape_limit=self.south_escape_limit,
            )
            return self._apply_recover(decision, world, pos, front, reason)

        if self._phase == "exit_to_farm":
            self._task = None
            return self._start_next_phase(world)
        if self._phase == "nav_drop_spot":
            self._task = None
            return self._start_next_phase(world)
        if self._phase == "nav_house_front":
            pos = get_pos_from_ram(world.ram)
            front = self._house_front_px(world.ram)
            if abs(pos.x - front.x) > 20 or pos.y < front.y - 10:
                self._offstand_corrections += 1
                # After a few soft-successes off-stand, force enter if close
                # enough, else re-nav with a hard cap.
                # Gate B D5: (190,423) vs (136,424) is lateral-near (dx=54).
                dx = abs(pos.x - front.x)
                dy = abs(pos.y - front.y)
                near = (dx <= 40 and dy <= 40) or (dx <= 72 and dy <= 24)
                if near and self._offstand_corrections >= 3:
                    print(
                        f"[RETURN_HOME] Off-stand but near door "
                        f"pos=({pos.x},{pos.y}) corrections="
                        f"{self._offstand_corrections}; forcing enter"
                    )
                    self._activate(
                        "enter_house",
                        self._house_enter_task(world),
                        world,
                    )
                    return self._task.step(world)
                if self._offstand_corrections >= 6:
                    return TaskResult(
                        status=TaskStatus.FAILURE,
                        reason=(
                            f"nav_house_front off-stand loop "
                            f"pos=({pos.x},{pos.y}) front=({front.x},{front.y}) "
                            f"corrections={self._offstand_corrections}"
                        ),
                    )
                print(
                    f"[RETURN_HOME] Nav finished off-stand pos=({pos.x},{pos.y}); "
                    f"correcting to ({front.x},{front.y}) "
                    f"({self._offstand_corrections}/6)"
                )
                return self._nav_to_house_front(world, front)
            self._activate(
                "enter_house",
                self._house_enter_task(world),
                world,
            )
            return self._task.step(world)
        if self._phase == "enter_house":
            tilemap = self._read_tilemap(world)
            arrived = self._house_arrival_success(tilemap, via="enter_house")
            if arrived is not None:
                return arrived
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"expected house tilemap, got 0x{tilemap:02X}",
            )

        return TaskResult(status=TaskStatus.FAILURE, reason=f"unknown phase {self._phase}")



__all__ = [
    "HOUSE_FRONT_PX",
    "HOUSE_DOOR_FRONT_PX",
    "ReturnHomeTask",
]

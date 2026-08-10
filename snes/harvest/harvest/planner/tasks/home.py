"""Return-home and sleep tasks used by the day planner."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.tasks.farm_clearer import (
    Point,
    make_action,
    get_pos_from_ram,
    ADDR_TILEMAP,
    ADDR_INPUT_LOCK,
)
from harvest.maps.map_config import ROUTES
from harvest.core.ram_catalog import field_spec, read_ram_u16
from harvest.core.scene import (
    SceneMode,
    classify_scene_from_ram,
    scene_indicates_ending,
)
from harvest.tasks.primitives import dismiss_dialogue_result, drain_action_queue
from harvest.planner.day_plan_status import (
    TASKS_DIR,
    FARM_TILEMAP,
    HOUSE_TILEMAP,
    HOUSE_TILEMAPS,
    read_world_date,
    is_farm_tilemap,
    is_house_tilemap,
)
from harvest.planner.tasks.inventory import ExitToFarmTask
from harvest.planner.tasks.navigation import MultiMapNavTask, NavTask
from harvest.core.animal_status import read_held_item
from harvest.planner.tasks.transitions import (
    DirectionalTransitionTask,
    HOUSE_ENTER_DOOR_X,
    HOUSE_ENTER_OVERSHOOT_Y,
    HOUSE_ENTER_STAND_TILE,
    hands_are_clear,
    multi_face_toss_actions,
    toss_held_actions,
)

# ── DayPlanTask ───────────────────────────────────────────────────

# Return-home routes live in map_config and are selected by upgrade state; these
# constants are fallbacks for any missing route data.
HOUSE_FRONT_PX = Point(136, 424)
HOUSE_DOOR_FRONT_PX = Point(136, 424)
HOUSE_BED_STAND_PX = Point(70, 86)
HOUSE_L2_BED_STAND_PX = Point(294, 102)
HOUSE_SLEEP_TRANSITION_TILEMAP = 0x0F
HOUSE_BED_STAND_TOLERANCE = 1
HOUSE_L2_BED_STAND_TOLERANCE = 2
HOUSE_BED_ROUTE_LOWER: List[Tuple[Point, str, int, bool]] = [
    (Point(98, 200), "y", 4, True),
    (Point(98, 120), "y", 4, True),
    (Point(72, 120), "x", 4, True),
    (Point(72, 86), "y", 2, True),
    (HOUSE_BED_STAND_PX, "x", 0, False),
]
HOUSE_BED_ROUTE_UPPER: List[Tuple[Point, str, int, bool]] = [
    (Point(168, 120), "x", 4, True),
    (Point(72, 120), "x", 4, True),
    (Point(72, 86), "y", 2, True),
    (HOUSE_BED_STAND_PX, "x", 0, False),
]
HOUSE_BED_ROUTE_MID: List[Tuple[Point, str, int, bool]] = [
    (Point(72, 120), "x", 4, True),
    (Point(72, 86), "y", 2, True),
    (HOUSE_BED_STAND_PX, "x", 0, False),
]
HOUSE_BED_ROUTE_FINAL: List[Tuple[Point, str, int, bool]] = [
    (HOUSE_BED_STAND_PX, "x", 0, False),
]
HOUSE_L2_BED_ROUTE_COMMON: List[Tuple[Point, str, int, bool]] = [
    (Point(233, 201), "x", 6, True),
    (Point(393, 201), "x", 6, True),
    (Point(393, 174), "y", 6, True),
    (Point(377, 134), "diag", 6, True),
    (Point(393, 126), "diag", 6, True),
    (Point(361, 126), "x", 6, True),
    (Point(361, 102), "y", 4, True),
    (Point(347, 110), "diag", 4, True),
    (Point(347, 133), "y", 2, True),
    (Point(334, 133), "y", 0, False),
    (Point(294, 133), "x", 2, True),
    (HOUSE_L2_BED_STAND_PX, "y", 2, True),
    (HOUSE_L2_BED_STAND_PX, "x", 1, False),
]
HOUSE_L2_BED_ROUTE_LOWER: List[Tuple[Point, str, int, bool]] = [
    (Point(136, 201), "y", 6, True),
    *HOUSE_L2_BED_ROUTE_COMMON,
]
HOUSE_L2_BED_ROUTE_LEFT_MID: List[Tuple[Point, str, int, bool]] = [
    (Point(150, 94), "y", 4, True),
    (Point(233, 94), "x", 6, True),
    (Point(233, 201), "y", 6, True),
    *HOUSE_L2_BED_ROUTE_COMMON[1:],
]
HOUSE_L2_BED_ROUTE_UPPER_LEFT: List[Tuple[Point, str, int, bool]] = [
    (Point(70, 88), "y", 2, False),
    (Point(22, 88), "x", 4, True),
    (Point(22, 145), "y", 4, True),
    (Point(70, 145), "x", 2, True),
    (Point(86, 145), "x", 2, True),
    (Point(86, 165), "y", 4, True),
    (Point(72, 191), "diag", 4, True),
    (Point(22, 201), "diag", 4, True),
    (Point(102, 193), "diag", 6, True),
    (Point(121, 166), "diag", 6, True),
    (Point(105, 121), "diag", 6, True),
    (Point(166, 123), "diag", 6, True),
    (Point(166, 154), "y", 6, True),
    (Point(142, 201), "diag", 6, True),
    *HOUSE_L2_BED_ROUTE_COMMON,
]
HOUSE_L2_BED_ROUTE_RIGHT: List[Tuple[Point, str, int, bool]] = [
    (Point(334, 133), "y", 6, True),
    (Point(294, 133), "x", 2, True),
    (HOUSE_L2_BED_STAND_PX, "y", 2, True),
    (HOUSE_L2_BED_STAND_PX, "x", 1, False),
]
HOUSE_L2_BED_ROUTE_FINAL: List[Tuple[Point, str, int, bool]] = [
    (HOUSE_L2_BED_STAND_PX, "x", 1, False),
]


@dataclass
class ReadyToGoHomeTask(Task):
    """Marker task: town/day work is done; planner should end the day.

    Success is the go-home flag. The day planner records it and advances into
    ``RETURN_HOME`` / ``GO_TO_SLEEP`` (or appends them if missing).
    """

    name: str = "ready_to_go_home"

    def reset(self, world: WorldState) -> None:
        return None

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        return TaskResult(
            status=TaskStatus.SUCCESS,
            reason="ready_to_go_home",
            checkpoint="ready_to_go_home",
            meta={"ready_to_go_home": True},
        )


@dataclass
class ReturnHomeTask(Task):
    """Recover to the farm if needed, then enter the farmhouse."""

    name: str = "return_home"
    tasks_dir: str = TASKS_DIR
    # Hard budget so multi-day soaks fail cleanly instead of hanging ~D5 when
    # enter_house / hands-full / nav soft-stalls without terminal status.
    timeout: int = 5500

    _phase: str = field(default="start", init=False)
    _task: Optional[Task] = field(default=None, init=False)
    _action_queue: deque = field(default_factory=deque, init=False)
    _drop_attempts: int = field(default=0, init=False)
    _drop_spot_navs: int = field(default=0, init=False)
    _enter_retries: int = field(default=0, init=False)
    _total_steps: int = field(default=0, init=False)
    # Soft-success off-stand re-navs can thrash forever without terminal
    # status if the door stand soft-radius keeps "succeeding" a few tiles
    # away. Cap corrections so the outer timeout is not the only exit.
    _offstand_corrections: int = field(default=0, init=False)
    _best_door_dist: int = field(default=99999, init=False)
    # Hard budget for in-place toss cycles after relocating to open ground.
    drop_attempt_limit: int = 10

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
    def _house_enter_task(cls, world: WorldState) -> DirectionalTransitionTask:
        """Build the outdoor→house doorway push.

        Always stand on the catalog door tile (or the remodeled threshold
        waypoint). Never adopt a mid-wall overshoot tile as the stand — that
        is what stuck soaks at (8,24)/y≈389 pushing into the house sprite.
        """
        front = cls._house_front_px(world.ram)
        # Base farmhouse approach ends at (136,424) → tile (8,26).
        # Remodeled routes end on the door threshold (~y=344).
        if front.y <= 360:
            stand_tile = (front.x // 16, front.y // 16)
            overshoot_y = min(HOUSE_ENTER_OVERSHOOT_Y, front.y - 12)
        else:
            stand_tile = HOUSE_ENTER_STAND_TILE
            overshoot_y = HOUSE_ENTER_OVERSHOOT_Y
        return DirectionalTransitionTask(
            name="enter_house",
            direction="up",
            origin_tilemap=FARM_TILEMAP,
            target_tilemap=HOUSE_TILEMAP,
            target_tilemaps=tuple(sorted(HOUSE_TILEMAPS)),
            timeout=2500,
            min_frames_before_success=15,
            settle_frames=20,
            stand_tile=stand_tile,
            stand_tolerance=0,
            door_align_px=front.x if front.x else HOUSE_ENTER_DOOR_X,
            overshoot_limit_px=overshoot_y,
            require_empty_hands=True,
            clear_hands_limit=6,
        )

    def reset(self, world: WorldState) -> None:
        self._phase = "start"
        self._task = None
        self._action_queue.clear()
        self._drop_attempts = 0
        self._drop_spot_navs = 0
        self._enter_retries = 0
        self._total_steps = 0
        self._offstand_corrections = 0
        self._best_door_dist = 99999

    def can_start(self, world: WorldState) -> bool:
        return True

    @staticmethod
    def _drop_spot_px(front: Point) -> Point:
        """Open ground south of the house door — not mid-field debris."""
        return Point(front.x, min(520, front.y + 56))

    def _at_drop_spot(self, pos: Point, front: Point) -> bool:
        drop = self._drop_spot_px(front)
        return abs(pos.x - drop.x) <= 28 and abs(pos.y - drop.y) <= 28

    def _queue_drop_carried(self) -> None:
        """Toss held debris so building doors accept entry.

        After CLEAR_FIELD leaves a stone/weed (held 0x0D/0x09), in-place field
        tosses often fail or re-pickup. Prefer multi-face stationary at the
        open drop spot; step-away only on the first cycle.
        """
        if self._drop_attempts <= 1:
            self._action_queue.extend(toss_held_actions(face="down", step_away=True))
            self._action_queue.extend(multi_face_toss_actions(prefer_south=True))
        else:
            self._action_queue.extend(multi_face_toss_actions(prefer_south=True))

    def _activate(self, phase: str, task: Task, world: WorldState) -> None:
        self._phase = phase
        self._task = task
        task.reset(world)

    def _nav_to_house_front(self, world: WorldState, front: Point) -> TaskResult:
        # Cap child nav so outer timeout can fire; leave headroom for enter.
        child_timeout = min(4000, max(800, self.timeout - self._total_steps - 400))
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
        """Walk south of the door so throws are not blocked by the house wall."""
        drop = self._drop_spot_px(front)
        child_timeout = min(3500, max(800, self.timeout - self._total_steps - 400))
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

    def _start_next_phase(self, world: WorldState) -> TaskResult:
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if is_house_tilemap(tilemap):
            return TaskResult(status=TaskStatus.SUCCESS, reason=f"tilemap=0x{tilemap:02X}")
        if not is_farm_tilemap(tilemap):
            self._activate("exit_to_farm", ExitToFarmTask(tasks_dir=self.tasks_dir), world)
            return self._task.step(world)

        pos = get_pos_from_ram(world.ram)
        front = self._house_front_px(world.ram)

        if not hands_are_clear(world.ram):
            # Always relocate to open ground south of the house before thrashing
            # A-drops in a debris field (rr-6g7g: CLEAR leaves held=0x0D).
            held = read_held_item(world.ram)
            at_drop = self._at_drop_spot(pos, front)
            if not at_drop and self._drop_spot_navs < 3:
                self._drop_spot_navs += 1
                print(
                    f"[RETURN_HOME] Moving to open drop spot "
                    f"pos=({pos.x},{pos.y}) held=0x{held:02X} "
                    f"(nav {self._drop_spot_navs}/3)"
                )
                return self._nav_to_drop_spot(world, front)
            if self._drop_attempts < self.drop_attempt_limit:
                self._drop_attempts += 1
                self._phase = "drop_carried"
                self._queue_drop_carried()
                print(
                    f"[RETURN_HOME] Dropping carried item before house entry "
                    f"({self._drop_attempts}/{self.drop_attempt_limit} "
                    f"held=0x{held:02X} at_drop={at_drop})"
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

        route_name = self._house_route_name(world.ram)
        route = ROUTES.get(route_name) or ROUTES.get("farm_to_house") or []
        if route and self._phase not in {"nav_house_front", "nav_drop_spot"}:
            route_timeout = min(5000, max(1200, self.timeout - self._total_steps - 500))
            self._activate(
                "nav_house_front",
                MultiMapNavTask(
                    name="nav_house_front",
                    waypoints=list(route),
                    timeout=route_timeout,
                    initial_settle_frames=0,
                ),
                world,
            )
            return self._task.step(world)
        return self._nav_to_house_front(world, front)

    def step(self, world: WorldState) -> TaskResult:
        self._total_steps += 1
        if self.timeout > 0 and self._total_steps > self.timeout:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=(
                    f"return_home timeout after {self._total_steps}f "
                    f"phase={self._phase}"
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
        if self._task is None:
            return self._start_next_phase(world)

        result = self._task.step(world)
        if result.status == TaskStatus.RUNNING:
            return result
        if result.status == TaskStatus.FAILURE:
            reason = result.reason or "unknown"
            if self._phase == "enter_house" and self._enter_retries < 4:
                self._enter_retries += 1
                self._task = None
                self._phase = "start"
                # Hands still full — reset drop budget and try again.
                if "hands not clear" in reason:
                    self._drop_attempts = 0
                    self._drop_spot_navs = 0
                print(
                    f"[RETURN_HOME] Retry house enter "
                    f"({self._enter_retries}/4): {reason}"
                )
                # Mid-wall pin (y≈372) after a timeout: strafe + walk south
                # into open ground before re-approaching.
                pos = get_pos_from_ram(world.ram)
                front = self._house_front_px(world.ram)
                held = read_held_item(world.ram)
                print(
                    f"[RETURN_HOME] Enter fail diagnostics "
                    f"pos=({pos.x},{pos.y}) front=({front.x},{front.y}) "
                    f"held=0x{held:02X} hands_clear={hands_are_clear(world.ram)}"
                )
                if pos.y < front.y - 16:
                    self._action_queue.extend(make_action(left=True) for _ in range(10))
                    self._action_queue.extend(
                        make_action(down=True, b=True) for _ in range(40)
                    )
                    self._action_queue.extend(make_action(right=True) for _ in range(12))
                    self._action_queue.extend(
                        make_action(down=True, b=True) for _ in range(24)
                    )
                    self._action_queue.extend(make_action() for _ in range(8))
                    self._phase = "drop_carried"
                    self._offstand_corrections = 0
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        action=ActionResult(self._action_queue.popleft()),
                        reason="south recovery after enter fail",
                    )
                return self._start_next_phase(world)
            # Nav failure with hands still full: drop then re-nav instead of die.
            if self._phase in {"nav_house_front", "nav_drop_spot"} and not hands_are_clear(
                world.ram
            ):
                self._task = None
                self._phase = "start"
                if self._drop_attempts < self.drop_attempt_limit:
                    print(
                        f"[RETURN_HOME] Nav failed with hands full; "
                        f"drop then retry: {reason}"
                    )
                    return self._start_next_phase(world)
            # Nav timed out but we are close to the door — attempt enter rather
            # than hard-fail the multi-day (return_home hang residual ~D5).
            if self._phase == "nav_house_front":
                pos = get_pos_from_ram(world.ram)
                front = self._house_front_px(world.ram)
                if abs(pos.x - front.x) <= 48 and abs(pos.y - front.y) <= 48:
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
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"{self._phase} failed: {reason}",
            )

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
                near = abs(pos.x - front.x) <= 40 and abs(pos.y - front.y) <= 40
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
            tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
            if is_house_tilemap(tilemap):
                return TaskResult(status=TaskStatus.SUCCESS, reason="entered house")
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"expected house tilemap, got 0x{tilemap:02X}",
            )

        return TaskResult(status=TaskStatus.FAILURE, reason=f"unknown phase {self._phase}")


@dataclass
class GoToSleepTask(Task):
    """Find the house if needed, walk to the bed, and sleep until the next day.

    Older sleep macros assumed the player was already inside. This task always
    recovers to the farmhouse first (``ReturnHomeTask``) so multi-day and
    late-day recovery can call sleep from anywhere.
    """

    name: str = "go_to_sleep"
    tasks_dir: str = TASKS_DIR
    timeout: int = 12000
    sleep_attempt_limit: int = 12
    # Wait long enough for the overnight fade before assuming A missed.
    # Overnight confirm + fade can exceed ~10s; do not re-mash mid-fade.
    sleep_verify_frames: int = 720
    # Budget for the outdoor/return-home recovery before bed navigation.
    return_home_timeout: int = 5500

    _phase: str = field(default="ensure_house", init=False)
    _step_count: int = field(default=0, init=False)
    _verify_count: int = field(default=0, init=False)
    _sleep_attempts: int = field(default=0, init=False)
    _action_queue: deque = field(default_factory=deque, init=False)
    _route: List[Tuple[Point, str, int, bool]] = field(default_factory=list, init=False)
    _route_index: int = field(default=0, init=False)
    _start_season: int = field(default=0, init=False)
    _start_day: int = field(default=0, init=False)
    _return_home: Optional[ReturnHomeTask] = field(default=None, init=False)
    _return_home_steps: int = field(default=0, init=False)
    # After day advances, require a settled morning house before SUCCESS so
    # callers (Gate B shed, multi-day planner) do not ExitToFarm mid-wake.
    _morning_settle_frames: int = field(default=0, init=False)
    morning_ready_frames: int = 45
    # Hands-full at bed: toss once before the first sleep A (doors already toss).
    _tossed_before_sleep: bool = field(default=False, init=False)

    def reset(self, world: WorldState) -> None:
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        self._phase = "nav_bed" if is_house_tilemap(tilemap) else "ensure_house"
        self._step_count = 0
        self._verify_count = 0
        self._sleep_attempts = 0
        self._action_queue.clear()
        self._route = []
        self._route_index = 0
        self._return_home = None
        self._return_home_steps = 0
        self._morning_settle_frames = 0
        self._tossed_before_sleep = False
        self._start_season, self._start_day = read_world_date(world.ram)
        if self._phase == "ensure_house":
            self._return_home = ReturnHomeTask(tasks_dir=self.tasks_dir)
            self._return_home.reset(world)

    def can_start(self, world: WorldState) -> bool:
        return True

    @staticmethod
    def _date_advanced(ram: np.ndarray, start_season: int, start_day: int) -> bool:
        season, day = read_world_date(ram)
        return (season, day) != (start_season, start_day)

    @staticmethod
    def _sleep_face_for_tilemap(tilemap: int) -> str:
        # Base + remodel L1: stand right of mattress, face up (go_to_sleep.json).
        # Level-2 wife bed: stand south of bed, face up (map landmark).
        # Never face left for A — left+A walks into the mattress and misses the
        # sleep prompt (recording + rr-m0wq D7 miss).
        return "up"

    def _queue_sleep_attempt(self, tilemap: int) -> None:
        """Match go_to_sleep.json: face up, one B settle, then plain A only.

        Human recording at (70,86):
          arrive facing left → hold Up → long idle → B (~10f) → A bursts.
          Sleep pull-in later walks left into the bed; left is not the face.

        rr-m0wq (power-on continuous D7): prior harden tried left-face A and
        mid-attempt B after A. Left+A walks into the mattress; B cancels the
        Yes/No sleep confirm. Keep face-up only, B only *before* the first A
        of each attempt, then A-only until verify timeout.

        Gate B carry (grass seeds + can): brief X settle once early so A is
        not eaten by a tool swing, then re-face up.
        """
        self._sleep_attempts += 1
        face = self._sleep_face_for_tilemap(tilemap)
        n = self._sleep_attempts

        # Tool settle only on early attempts — X mid-confirm is noise.
        if n <= 3:
            self._action_queue.extend(make_action(x=True) for _ in range(3))
            self._action_queue.extend(make_action() for _ in range(10))

        # Late attempts: micro re-seat on the bed stand (pixel slip / shove).
        if n >= 4:
            self._action_queue.extend(make_action(down=True) for _ in range(3))
            self._action_queue.extend(make_action() for _ in range(6))
            # Alternate x column slightly then return (still face-up for A).
            if n % 2 == 0:
                self._action_queue.extend(make_action(right=True) for _ in range(3))
            else:
                self._action_queue.extend(make_action(left=True) for _ in range(3))
            self._action_queue.extend(make_action() for _ in range(4))
            self._action_queue.extend(make_action(up=True) for _ in range(12))
            self._action_queue.extend(make_action() for _ in range(8))

        # Long face-up hold against the mattress (recording ~20–30f continuous).
        self._action_queue.extend(make_action(**{face: True}) for _ in range(36))
        self._action_queue.extend(make_action() for _ in range(48))
        # Single B settle *before* any A (closes tool residue / menus).
        self._action_queue.extend(make_action(b=True) for _ in range(12))
        self._action_queue.extend(make_action() for _ in range(24))
        # Re-assert face up after B (B can leave facing stale).
        self._action_queue.extend(make_action(**{face: True}) for _ in range(16))
        self._action_queue.extend(make_action() for _ in range(20))
        # A-only bursts — never B here (B selects No on the sleep confirm).
        bursts = 8 if n < 6 else 10
        for _ in range(bursts):
            self._action_queue.extend(make_action(a=True) for _ in range(16))
            self._action_queue.extend(make_action() for _ in range(14))
        # Wait for overnight fade / dialogue without canceling.
        self._action_queue.extend(make_action() for _ in range(220))

    @staticmethod
    def _bed_stand_for_tilemap(tilemap: int) -> Point:
        if tilemap == 0x17:
            return HOUSE_L2_BED_STAND_PX
        return HOUSE_BED_STAND_PX

    @staticmethod
    def _bed_tolerance_for_tilemap(tilemap: int) -> int:
        if tilemap == 0x17:
            return HOUSE_L2_BED_STAND_TOLERANCE
        return HOUSE_BED_STAND_TOLERANCE

    @classmethod
    def _at_bed(cls, pos: Point, tilemap: int) -> bool:
        bed = cls._bed_stand_for_tilemap(tilemap)
        tolerance = cls._bed_tolerance_for_tilemap(tilemap)
        return (
            abs(pos.x - bed.x) <= tolerance
            and abs(pos.y - bed.y) <= tolerance
        )

    @staticmethod
    def _route_for_position(pos: Point, tilemap: int) -> List[Tuple[Point, str, int, bool]]:
        if tilemap == 0x17:
            if abs(pos.x - HOUSE_L2_BED_STAND_PX.x) <= 8 and abs(pos.y - HOUSE_L2_BED_STAND_PX.y) <= 8:
                return list(HOUSE_L2_BED_ROUTE_FINAL)
            if pos.x >= 260 and pos.y <= 155:
                return list(HOUSE_L2_BED_ROUTE_RIGHT)
            if pos.x <= 100 and pos.y <= 100:
                return list(HOUSE_L2_BED_ROUTE_UPPER_LEFT)
            if pos.y >= 170:
                return list(HOUSE_L2_BED_ROUTE_LOWER)
            if pos.x < 220:
                return list(HOUSE_L2_BED_ROUTE_LEFT_MID)
            return list(HOUSE_L2_BED_ROUTE_COMMON)

        if pos.x <= 90 and pos.y <= 96:
            return list(HOUSE_BED_ROUTE_FINAL)
        if pos.y >= 150:
            return list(HOUSE_BED_ROUTE_LOWER)
        if pos.x >= 150:
            return list(HOUSE_BED_ROUTE_UPPER)
        return list(HOUSE_BED_ROUTE_MID)

    @staticmethod
    def _move_toward_point(
        pos: Point,
        target: Point,
        *,
        primary_axis: str,
        tolerance: int,
        run: bool,
    ) -> Optional[np.ndarray]:
        dx = target.x - pos.x
        dy = target.y - pos.y

        def axis_action(direction: str, delta: int) -> np.ndarray:
            return make_action(**{direction: True, "b": run and abs(delta) > 4})

        if primary_axis == "diag":
            buttons = {"b": run and (abs(dx) > 4 or abs(dy) > 4)}
            if abs(dx) > tolerance:
                buttons["left" if dx < 0 else "right"] = True
            if abs(dy) > tolerance:
                buttons["up" if dy < 0 else "down"] = True
            if len(buttons) > 1:
                return make_action(**buttons)
            return None

        if primary_axis == "y":
            if abs(dy) > tolerance:
                return axis_action("up" if dy < 0 else "down", dy)
            if abs(dx) > tolerance:
                return axis_action("left" if dx < 0 else "right", dx)
        else:
            if abs(dx) > tolerance:
                return axis_action("left" if dx < 0 else "right", dx)
            if abs(dy) > tolerance:
                return axis_action("up" if dy < 0 else "down", dy)
        return None

    def _move_toward_bed(self, pos: Point, tilemap: int) -> np.ndarray:
        if not self._route:
            self._route = self._route_for_position(pos, tilemap)
            self._route_index = 0

        while self._route_index < len(self._route):
            target, primary_axis, tolerance, run = self._route[self._route_index]
            action = self._move_toward_point(
                pos,
                target,
                primary_axis=primary_axis,
                tolerance=tolerance,
                run=run,
            )
            if action is not None:
                return action
            self._route_index += 1

        return make_action()

    def _step_ensure_house(self, world: WorldState) -> TaskResult:
        """Recover into the farmhouse before bed navigation."""
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if is_house_tilemap(tilemap):
            self._phase = "nav_bed"
            self._return_home = None
            self._route = []
            self._route_index = 0
            print(f"[SLEEP] Inside house tilemap=0x{tilemap:02X}; navigating to bed")
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        if self._return_home is None:
            self._return_home = ReturnHomeTask(tasks_dir=self.tasks_dir)
            self._return_home.reset(world)

        self._return_home_steps += 1
        if self._return_home_steps > self.return_home_timeout:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"could not find house (tilemap=0x{tilemap:02X})",
            )

        result = self._return_home.step(world)
        if result.status == TaskStatus.SUCCESS:
            self._phase = "nav_bed"
            self._return_home = None
            self._route = []
            self._route_index = 0
            print("[SLEEP] ReturnHome succeeded; navigating to bed")
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
        if result.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
            reason = result.reason or "unknown"
            # One hard retry — house entry can fail mid-throw/debris.
            if self._return_home_steps < self.return_home_timeout // 2:
                print(f"[SLEEP] ReturnHome retry after: {reason}")
                self._return_home = ReturnHomeTask(tasks_dir=self.tasks_dir)
                self._return_home.reset(world)
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"return home before sleep failed: {reason}",
            )
        return result

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        if self._step_count > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason="sleep timeout")

        scene = classify_scene_from_ram(world.ram)
        if scene_indicates_ending(scene):
            return TaskResult(status=TaskStatus.SUCCESS, reason="ending reached")

        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        # Only the real overnight transition (tilemap 0x0F / time_running), not
        # the morning-wake coordinate heuristic used by the scene classifier.
        if tilemap == HOUSE_SLEEP_TRANSITION_TILEMAP or scene.reason == "sleep/wake transition":
            self._morning_settle_frames = 0
            return dismiss_dialogue_result(
                self._step_count,
                buttons=("a", "b"),
                pulse_every=2,
                reason="sleep transition",
            )
        if scene.mode == SceneMode.CUTSCENE_EVENT:
            return dismiss_dialogue_result(
                self._step_count,
                buttons=("a", "b"),
                pulse_every=1,
                reason="bedtime cutscene",
            )

        # Day rolled: wait for controllable morning house (not mid-wake 0x0F).
        # Early SUCCESS here caused Gate B ExitToFarm to hit door glitch 0x5F.
        if self._date_advanced(world.ram, self._start_season, self._start_day):
            input_lock = (
                int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
            )
            pos = get_pos_from_ram(world.ram)
            morning_house = is_house_tilemap(tilemap) and input_lock == 1 and not scene.needs_input_dismiss
            # Wake coords sit near bed ~(70,86) then settle ~(136,120).
            settled_xy = pos.y >= 100 or (
                abs(pos.x - HOUSE_BED_STAND_PX.x) <= 24
                and abs(pos.y - HOUSE_BED_STAND_PX.y) <= 24
            )
            if morning_house and settled_xy:
                self._morning_settle_frames += 1
                if self._morning_settle_frames >= self.morning_ready_frames:
                    return TaskResult(
                        status=TaskStatus.SUCCESS,
                        reason="day advanced; morning house ready",
                    )
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action()),
                    reason=f"morning settle {self._morning_settle_frames}/{self.morning_ready_frames}",
                )
            self._morning_settle_frames = 0
            if scene.needs_input_dismiss or input_lock != 1:
                return dismiss_dialogue_result(self._step_count, reason="morning dismiss")
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action()),
                reason=f"morning wait tm=0x{tilemap:02X} pos=({pos.x},{pos.y})",
            )

        if self._phase == "ensure_house":
            return self._step_ensure_house(world)

        if not is_house_tilemap(tilemap):
            # Left the house mid-sleep attempt (cutscene, failed bed push).
            print(
                f"[SLEEP] Left house (tilemap=0x{tilemap:02X}); re-finding house"
            )
            self._phase = "ensure_house"
            self._return_home = ReturnHomeTask(tasks_dir=self.tasks_dir)
            self._return_home.reset(world)
            self._action_queue.clear()
            self._route = []
            self._route_index = 0
            return self._step_ensure_house(world)

        input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
        # A-only dismiss at the bed: B selects No on the sleep Yes/No confirm
        # (rr-m0wq). Outside the house, default recovery can still use A.
        if input_lock != 1 or scene.needs_input_dismiss:
            return dismiss_dialogue_result(
                self._step_count,
                buttons=("a",),
                pulse_every=2,
                reason="pre-sleep dismiss",
            )

        queued = drain_action_queue(self._action_queue)
        if queued is not None:
            return queued

        pos = get_pos_from_ram(world.ram)
        at_bed = self._at_bed(pos, tilemap)

        if self._phase == "nav_bed":
            if at_bed:
                # Hands-full blocks some house interactions; toss once away
                # from the mattress before the first A (south into open floor).
                if not self._tossed_before_sleep and not hands_are_clear(world.ram):
                    self._tossed_before_sleep = True
                    self._action_queue.extend(toss_held_actions(face="down"))
                    self._route = []
                    self._route_index = 0
                    print("[SLEEP] Tossing held item before bed interaction")
                    queued = drain_action_queue(self._action_queue)
                    if queued is not None:
                        return queued
                self._phase = "sleep_attempt"
            else:
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(self._move_toward_bed(pos, tilemap)),
                )

        if self._phase == "sleep_attempt":
            if self._sleep_attempts >= self.sleep_attempt_limit:
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=(
                        "sleep interaction did not advance day "
                        f"pos=({pos.x},{pos.y}) attempts={self._sleep_attempts}"
                    ),
                )
            print(
                f"[SLEEP] Attempt {self._sleep_attempts + 1}/"
                f"{self.sleep_attempt_limit} at bed pos=({pos.x},{pos.y}) "
                f"tm=0x{tilemap:02X}"
            )
            self._queue_sleep_attempt(tilemap)
            self._phase = "sleep_verify"
            self._verify_count = 0
            queued = drain_action_queue(self._action_queue)
            if queued is not None:
                return queued

        if self._phase == "sleep_verify":
            self._verify_count += 1
            # Drifted off the stand mid-attempt (tool shove / bad nudge):
            # re-nav immediately instead of mashing A into empty air.
            if (
                not self._at_bed(pos, tilemap)
                and self._verify_count > 40
                and scene.mode == SceneMode.NORMAL
                and input_lock == 1
            ):
                self._action_queue.clear()
                self._verify_count = 0
                self._route = []
                self._route_index = 0
                self._phase = "nav_bed"
                print(
                    f"[SLEEP] Off bed during verify pos=({pos.x},{pos.y}); "
                    f"re-nav (attempt {self._sleep_attempts}/"
                    f"{self.sleep_attempt_limit})"
                )
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(self._move_toward_bed(pos, tilemap)),
                )
            # Overnight fade / dialogue can take several seconds; do not mash
            # a retry that cancels the sleep confirmation.
            if self._verify_count > self.sleep_verify_frames:
                self._verify_count = 0
                self._route = []
                self._route_index = 0
                self._phase = "nav_bed" if not self._at_bed(pos, tilemap) else "sleep_attempt"
                print(
                    f"[SLEEP] Retry sleep interaction "
                    f"({self._sleep_attempts}/{self.sleep_attempt_limit}) "
                    f"pos=({pos.x},{pos.y})"
                )
            # Keep dismissing with A only (never B — cancels Yes).
            if scene.mode in {SceneMode.DIALOGUE, SceneMode.MENU}:
                return dismiss_dialogue_result(
                    self._step_count,
                    buttons=("a",),
                    pulse_every=2,
                    reason="sleep confirm/dialog",
                )
            # Pulse A while waiting — the Yes/No sleep prompt can open late
            # and is not always classified as dialogue/menu.
            if self._verify_count % 28 < 12:
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action(a=True)),
                )
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        return TaskResult(status=TaskStatus.FAILURE, reason=f"unknown phase {self._phase}")



__all__ = [
    "HOUSE_FRONT_PX",
    "HOUSE_DOOR_FRONT_PX",
    "HOUSE_BED_STAND_PX",
    "HOUSE_SLEEP_TRANSITION_TILEMAP",
    "HOUSE_BED_STAND_TOLERANCE",
    "ReadyToGoHomeTask",
    "ReturnHomeTask",
    "GoToSleepTask",
]

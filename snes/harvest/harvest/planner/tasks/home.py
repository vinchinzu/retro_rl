"""Return-home and sleep tasks used by the day planner."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.tasks.nav import (
    Point,
    make_action,
    get_pos_from_ram,
)
from harvest.core.tile_catalog import (
    ADDR_TILEMAP,
    ADDR_INPUT_LOCK,
)

from harvest.maps.map_config import (
    ROUTES,
    Waypoint,
)
from harvest.core.ram_catalog import field_spec, read_ram_u16, read_ram_value
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
from harvest.planner.tasks.home_approach import (
    EAST_AROUND_FENCE_X,
    build_house_approach_waypoints,
    deep_south_of_house,
    far_east_of_pond_lane,
    south_of_fence_wall,
)
from harvest.planner.tasks.home_recover import (
    RecoverDecision,
    RecoverKind,
    decide_child_failure,
    enter_fail_south_recovery_actions,
    exit_to_farm_recover_actions,
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
# Tight column for bed A (go_to_sleep recording stand x=70). Tolerance 4
# accepted x=74 and burned 12 evening attempts with no day advance (Gate B).
# Allow ±2 for face-up micro-slip without accepting the loose column.
HOUSE_BED_STAND_TOLERANCE = 2
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

    @staticmethod
    def _drop_spot_px(front: Point, *, deep: bool = False) -> Point:
        """Open ground south of the house door — not mid-field debris.

        ``deep`` is a second tier further south used when primary drop thrash
        leaves the same held rock fragment (power-on held=0x0F residual).
        """
        if deep:
            return Point(front.x, min(560, front.y + 112))
        return Point(front.x, min(520, front.y + 56))

    def _at_drop_spot(self, pos: Point, front: Point) -> bool:
        drop = self._drop_spot_px(front, deep=self._drop_deep_relocated)
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
        """Toss held debris so building doors accept entry.

        After CLEAR_FIELD leaves a stone/weed (held 0x0D/0x09/0x0F rock
        fragment), in-place field tosses often fail or re-pickup. Prefer
        multi-face stationary at the open drop spot; later cycles use shorter
        step-away tosses so the outer timeout can still hard-fail cleanly.
        """
        n = self._drop_attempts
        if n <= 1:
            self._action_queue.extend(toss_held_actions(face="down", step_away=True))
            self._action_queue.extend(multi_face_toss_actions(prefer_south=True))
        elif n <= 3:
            # Full multi-face without the expensive first-cycle step-away.
            self._action_queue.extend(multi_face_toss_actions(prefer_south=True))
        else:
            # Stuck debris (power-on held=0x0F): short step-away per face,
            # skip pure-up which re-seals toward the house wall.
            for face in ("down", "left", "right"):
                self._action_queue.extend(
                    toss_held_actions(face=face, step_away=True)
                )

    def _queue_south_escape(
        self, *, long_east: bool = False, far_east: bool = False
    ) -> None:
        """Leave south-of-fence softlock for re-nav.

        West of fence (x<176) or east-of-pond (x≥576): charge north. Mid-wall:
        east first. Far-east pond latitude: west toward free lane then north
        (right-first makes the D12 thrash worse). Mix A thrash so a blocking
        weed can be lifted.
        """
        # long_east used when pre-escape starts deep SW (unknown x at queue time).
        def _push(direction: str, frames: int) -> None:
            for i in range(frames):
                kwargs = {direction: True, "b": True}
                if i % 20 == 0:
                    kwargs = {direction: True, "a": True}
                self._action_queue.append(make_action(**kwargs))

        if far_east:
            # East of pond / shipping scrub: west onto free lane, then north.
            _push("left", 90)
            _push("up", 80)
            _push("left", 70)
            _push("up", 90)
            _push("left", 40)
            _push("up", 60)
        elif long_east:
            # Prefer north-first then east: SW pocket is often already west of
            # the fence wall (tile x≤10) where north is the free path.
            _push("up", 80)
            _push("right", 70)
            _push("up", 90)
            _push("right", 60)
            _push("up", 80)
            _push("left", 20)
        else:
            _push("right", 70)
            _push("up", 70)
            _push("right", 50)
            _push("up", 80)
            _push("left", 16)
        self._action_queue.extend(make_action() for _ in range(8))

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
        drop = self._drop_spot_px(front, deep=self._drop_deep_relocated)
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
        """Compact east→north charge when outer timeout is almost gone.

        Shorter than full south_escape (~370f vs ~500f+) so a late D19
        residual after drop can still clear the y=31 wall.
        """
        for i in range(50):
            kwargs = {"right": True, "b": True}
            if i % 16 == 0:
                kwargs = {"right": True, "a": True}
            self._action_queue.append(make_action(**kwargs))
        for i in range(70):
            kwargs = {"up": True, "b": True}
            if i % 16 == 0:
                kwargs = {"up": True, "a": True}
            self._action_queue.append(make_action(**kwargs))
        self._action_queue.extend(make_action() for _ in range(6))

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
                deep = self._drop_spot_px(front, deep=True)
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


@dataclass
class GoToSleepTask(Task):
    """Find the house if needed, walk to the bed, and sleep until the next day.

    Older sleep macros assumed the player was already inside. This task always
    recovers to the farmhouse first (``ReturnHomeTask``) so multi-day and
    late-day recovery can call sleep from anywhere.
    """

    name: str = "go_to_sleep"
    tasks_dir: str = TASKS_DIR
    # Headroom for early return_home (midday) → idle until evening + bed A.
    timeout: int = 24000
    sleep_attempt_limit: int = 12
    # Wait long enough for the overnight fade before assuming A missed.
    # Overnight confirm + fade can exceed ~10s; do not re-mash mid-fade.
    sleep_verify_frames: int = 720
    # Budget for the outdoor/return-home recovery before bed navigation.
    # Match ReturnHomeTask default (south-of-fence densified approach).
    return_home_timeout: int = 11000
    # ROM rejects bed A until evening. Outdoor wait advances the clock
    # (house idle freezes time). Prefer 18 — hour 17 still no-op'd on D4 soak.
    earliest_sleep_hour: int = 18

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
    # Midday end-day: house clock freezes; exit and idle outdoors until evening.
    _exit_for_evening: Optional[ExitToFarmTask] = field(default=None, init=False)

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
        self._exit_for_evening = None
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

        Gate B D5 residual: long queues + left/right re-seat shoved to x=73
        (off-bed thrash) and burned the outer 12k timeout before day advanced.
        Keep attempts short; only vertical re-seat; no lateral walk.
        """
        self._sleep_attempts += 1
        face = self._sleep_face_for_tilemap(tilemap)
        n = self._sleep_attempts

        # Tool settle only on early attempts — X mid-confirm is noise.
        if n <= 2:
            self._action_queue.extend(make_action(x=True) for _ in range(2))
            self._action_queue.extend(make_action() for _ in range(6))

        # Late attempts: vertical re-seat only (never left/right — slides off
        # the 1–4px bed stand into empty air).
        if n >= 3:
            self._action_queue.extend(make_action(down=True) for _ in range(2))
            self._action_queue.extend(make_action() for _ in range(4))
            self._action_queue.extend(make_action(up=True) for _ in range(10))
            self._action_queue.extend(make_action() for _ in range(6))

        # Face-up hold against the mattress.
        self._action_queue.extend(make_action(**{face: True}) for _ in range(24))
        self._action_queue.extend(make_action() for _ in range(20))
        # Single B settle *before* any A (closes tool residue / menus).
        self._action_queue.extend(make_action(b=True) for _ in range(8))
        self._action_queue.extend(make_action() for _ in range(12))
        # Re-assert face up after B (B can leave facing stale).
        self._action_queue.extend(make_action(**{face: True}) for _ in range(12))
        self._action_queue.extend(make_action() for _ in range(10))
        # A-only bursts — never B here (B selects No on the sleep confirm).
        bursts = 6 if n < 5 else 8
        for _ in range(bursts):
            self._action_queue.extend(make_action(a=True) for _ in range(12))
            self._action_queue.extend(make_action() for _ in range(10))
        # Wait for overnight fade / dialogue without canceling.
        self._action_queue.extend(make_action() for _ in range(120))

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

    def _step_wait_evening(
        self,
        world: WorldState,
        *,
        tilemap: int,
        hour: int,
        scene,
    ) -> TaskResult:
        """Advance the day clock outdoors until bed A can work.

        Indoor idle freezes time (Gate B D2 soak stuck at 11:05). Exit house,
        idle on farm until ``earliest_sleep_hour``, then re-enter via ensure_house.
        """
        self._action_queue.clear()
        self._verify_count = 0
        self._sleep_attempts = 0
        if scene.needs_input_dismiss:
            return dismiss_dialogue_result(
                self._step_count,
                buttons=("a", "b"),
                pulse_every=1,
                reason=f"evening-wait dismiss hour={hour}",
            )
        if is_house_tilemap(tilemap):
            if self._exit_for_evening is None:
                print(
                    f"[SLEEP] Pre-evening: exit house to advance clock "
                    f"(hour={hour} need>={self.earliest_sleep_hour})"
                )
                self._exit_for_evening = ExitToFarmTask(tasks_dir=self.tasks_dir)
                self._exit_for_evening.reset(world)
            result = self._exit_for_evening.step(world)
            if result.status == TaskStatus.SUCCESS:
                self._exit_for_evening = None
                print(f"[SLEEP] Pre-evening: on farm; idling until hour>={self.earliest_sleep_hour}")
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action()),
                    reason=f"evening outdoor wait hour={hour}",
                )
            if result.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
                # Retry exit; do not hard-fail the whole sleep yet.
                self._exit_for_evening = ExitToFarmTask(tasks_dir=self.tasks_dir)
                self._exit_for_evening.reset(world)
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action()),
                    reason=f"evening exit retry: {result.reason or 'fail'}",
                )
            return result
        # Outdoor: neutral frames advance the clock. Do NOT B-run into fences —
        # live soak stuck mashing B+down against the house south wall for 10k+
        # frames with zero progress (power-on watch feedback).
        self._exit_for_evening = None
        if self._step_count % 300 == 1:
            pos = get_pos_from_ram(world.ram)
            print(
                f"[SLEEP] Outdoor evening wait hour={hour}/"
                f"{self.earliest_sleep_hour} pos=({pos.x},{pos.y})"
            )
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(make_action()),
            reason=f"evening outdoor wait hour={hour}",
        )

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

        hour = int(read_ram_value(world.ram, "hour"))
        # Midday end-day: bed A is a no-op and house time freezes (Gate B soak).
        # Exit outdoors and idle until evening — farm clock advances.
        if hour < self.earliest_sleep_hour:
            return self._step_wait_evening(world, tilemap=tilemap, hour=hour, scene=scene)

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
                # Hands-full blocks bed A; re-toss every approach (not only once)
                # so CLEAR debris re-pickup cannot stick across retries.
                if not hands_are_clear(world.ram):
                    self._tossed_before_sleep = True
                    self._action_queue.extend(toss_held_actions(face="down"))
                    self._route = []
                    self._route_index = 0
                    held = int(read_held_item(world.ram))
                    print(
                        f"[SLEEP] Tossing held item before bed interaction "
                        f"held=0x{held:02X}"
                    )
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
            held = int(read_held_item(world.ram))
            print(
                f"[SLEEP] Attempt {self._sleep_attempts + 1}/"
                f"{self.sleep_attempt_limit} at bed pos=({pos.x},{pos.y}) "
                f"tm=0x{tilemap:02X} held=0x{held:02X} hour={hour}"
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

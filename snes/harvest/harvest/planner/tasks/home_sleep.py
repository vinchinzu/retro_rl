"""Go-to-sleep task: ensure house, walk to bed, overnight advance.

Uses :class:`ReturnHomeTask` for outdoor recovery. Public imports should
prefer ``harvest.planner.tasks.home``.
"""

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
from harvest.core.ram_catalog import read_ram_value
from harvest.core.scene import (
    SceneMode,
    classify_scene_from_ram,
    scene_indicates_ending,
)
from harvest.tasks.primitives import dismiss_dialogue_result, drain_action_queue
from harvest.planner.day_plan_status import (
    TASKS_DIR,
    read_world_date,
    is_house_tilemap,
)
from harvest.planner.tasks.home_return import ReturnHomeTask
from harvest.planner.tasks.inventory import ExitToFarmTask
from harvest.core.animal_status import read_held_item
from harvest.planner.tasks.transitions import (
    hands_are_clear,
    toss_held_actions,
)

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
    "HOUSE_BED_STAND_PX",
    "HOUSE_L2_BED_STAND_PX",
    "HOUSE_SLEEP_TRANSITION_TILEMAP",
    "HOUSE_BED_STAND_TOLERANCE",
    "HOUSE_L2_BED_STAND_TOLERANCE",
    "GoToSleepTask",
]

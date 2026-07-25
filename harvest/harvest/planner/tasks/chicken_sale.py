"""Chicken sale and animal-shop tasks used by the day planner."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.tasks.farm_clearer import (
    Point,
    TileScanner,
    Pathfinder,
    Navigator,
    make_action,
    get_pos_from_ram,
    ADDR_TILEMAP,
    ADDR_INPUT_LOCK,
    MAP_WIDTH,
)
from harvest.tasks.harvest_task import read_shipping_money
from harvest.maps.map_config import ROUTES, get_walkable_tiles
from harvest.core.animal_probe import chicken_slot_snapshots
from harvest.core.ram_catalog import field_spec, read_ram_u8, read_ram_u16, read_ram_value
from harvest.tasks.primitives import (
    dismiss_dialogue_result,
    drain_action_queue,
    press_a_sequence,
    press_button_sequence,
)
from harvest.tasks.recorded_task import RecordedTask
from harvest.planner.day_plan_status import (
    TASKS_DIR,
    COOP_TILEMAP,
    ADDR_CHICKEN_COUNT,
    ADDR_ITEM_ON_HAND,
    ADDR_COW_COUNT,
    read_world_day_time,
    is_farm_tilemap,
)
from harvest.planner.tasks.navigation import MultiMapNavTask, NavTask

ITEM_CHICKEN = 0x25

@dataclass
class CoopPickupChickenTask(Task):
    """Pick up one reachable adult chicken inside the coop."""

    name: str = "coop_pickup_chicken"
    timeout: int = 1800

    _scanner: TileScanner = field(default_factory=TileScanner, init=False)
    _pathfinder: Pathfinder = field(init=False)
    _navigator: Navigator = field(init=False)
    _step_count: int = field(default=0, init=False)
    _phase: str = field(default="nav", init=False)
    _action_queue: deque = field(default_factory=deque, init=False)
    _target: Optional[Tuple[Tuple[int, int], str, Tuple[int, int]]] = field(default=None, init=False)
    _attempts: int = field(default=0, init=False)
    _verify_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._pathfinder = Pathfinder(self._scanner, walkable_tiles=set(get_walkable_tiles(COOP_TILEMAP)))
        self._navigator = Navigator(self._pathfinder)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._phase = "nav"
        self._action_queue.clear()
        self._target = None
        self._attempts = 0
        self._verify_count = 0
        self._navigator.update(world.ram)
        self._navigator.path = []
        self._navigator.stasis = 0
        self._pathfinder.temp_blocked.clear()

    def can_start(self, world: WorldState) -> bool:
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        return tilemap == COOP_TILEMAP

    def _held_item(self, ram: np.ndarray) -> int:
        return read_ram_u8(ram, ADDR_ITEM_ON_HAND)

    def _adult_chicken_tiles(self, ram: np.ndarray) -> list[Tuple[int, int]]:
        tiles: list[Tuple[int, int]] = []
        seen: set[Tuple[int, int]] = set()
        for row in chicken_slot_snapshots(ram, require_coop=True):
            if row.get("stage") != "adult":
                continue
            tile = row.get("tile")
            if not (isinstance(tile, list) and len(tile) == 2):
                continue
            chicken_tile = (int(tile[0]), int(tile[1]))
            if chicken_tile in seen:
                continue
            seen.add(chicken_tile)
            tiles.append(chicken_tile)
        return tiles

    def _blocker_tiles(self, ram: np.ndarray) -> set[Tuple[int, int]]:
        blockers: set[Tuple[int, int]] = set()
        for row in chicken_slot_snapshots(ram, require_coop=True):
            if row.get("stage") not in {"adult", "egg"}:
                continue
            tile = row.get("tile")
            if isinstance(tile, list) and len(tile) == 2:
                blockers.add((int(tile[0]), int(tile[1])))
        return blockers

    def _candidate_stands(self, chicken_tile: Tuple[int, int]) -> tuple[Tuple[Tuple[int, int], str], ...]:
        x, y = chicken_tile
        return (
            ((x + 1, y), "left"),
            ((x - 1, y), "right"),
            ((x, y + 1), "up"),
            ((x, y - 1), "down"),
        )

    def _find_path_around_chickens(
        self,
        ram: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[list[Tuple[int, int]]]:
        blockers = self._blocker_tiles(ram)
        blockers.discard(start)
        if goal in blockers:
            return None
        if start == goal:
            return []

        queue = deque([start])
        came_from: dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) == goal:
                break
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                nxt = (nx, ny)
                if not (0 <= nx < MAP_WIDTH and 0 <= ny < MAP_WIDTH):
                    continue
                if nxt in came_from or nxt in blockers:
                    continue
                if not self._pathfinder.is_walkable(ram, nx, ny, current_pos=start):
                    continue
                came_from[nxt] = (cx, cy)
                queue.append(nxt)

        if goal not in came_from:
            return None

        path: list[Tuple[int, int]] = []
        cur = goal
        while cur != start:
            path.append(cur)
            parent = came_from[cur]
            if parent is None:
                break
            cur = parent
        path.reverse()
        return path

    def _select_target(self, ram: np.ndarray) -> Optional[Tuple[Tuple[int, int], str, Tuple[int, int]]]:
        current = self._navigator.current_tile
        blockers = self._blocker_tiles(ram)
        best: Optional[Tuple[int, Tuple[int, int], str, Tuple[int, int]]] = None
        for chicken_tile in self._adult_chicken_tiles(ram):
            for stand, face in self._candidate_stands(chicken_tile):
                sx, sy = stand
                if not (0 <= sx < MAP_WIDTH and 0 <= sy < MAP_WIDTH):
                    continue
                if stand in blockers and stand != current:
                    continue
                if not self._pathfinder.is_walkable(ram, sx, sy, current_pos=current):
                    continue
                path = self._find_path_around_chickens(ram, current, stand)
                if path is None:
                    continue
                score = len(path)
                if best is None or score < best[0]:
                    best = (score, stand, face, chicken_tile)
        if best is None:
            return None
        return best[1], best[2], best[3]

    def _fallback_action(self, goal: Tuple[int, int]) -> np.ndarray:
        current = self._navigator.current_tile
        dx = goal[0] - current[0]
        dy = goal[1] - current[1]
        if abs(dx) >= abs(dy):
            direction = "right" if dx > 0 else "left"
        else:
            direction = "down" if dy > 0 else "up"
        return make_action(**{direction: True, "b": True})

    def _navigate_to_tile(self, ram: np.ndarray, goal: Tuple[int, int]) -> Optional[np.ndarray]:
        if self._navigator.current_tile == goal or self._navigator.at_tile(goal, tolerance=1):
            return self._navigator.center_on_tile(goal, tolerance=1)

        blockers = self._blocker_tiles(ram)
        blockers.discard(self._navigator.current_tile)
        if goal in blockers:
            self._navigator.path = []
            return make_action()
        if self._navigator.path and self._navigator.path[0] in blockers:
            self._navigator.path = []
            return make_action()
        if self._navigator.stasis > 90 and self._navigator.path:
            self._pathfinder.temp_blocked.add(self._navigator.path[0])
            self._navigator.path = []

        if not self._navigator.path:
            path = self._find_path_around_chickens(ram, self._navigator.current_tile, goal)
            if path is None:
                return self._fallback_action(goal)
            self._navigator.path = path

        action = self._navigator.follow_path(ram)
        if action is None:
            return self._fallback_action(goal)
        return action

    def _queue_pickup(self, face: str) -> None:
        self._action_queue.extend(
            press_a_sequence(
                face,
                face_frames=4,
                pre_press_settle_frames=0,
                hold_frames=16,
                settle_frames=24,
                hold_face_with_a=True,
            )
        )

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        self._navigator.update(world.ram)

        if self._step_count > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason="chicken pickup timeout")

        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if tilemap != COOP_TILEMAP:
            return TaskResult(status=TaskStatus.BLOCKED, reason=f"not in coop tilemap=0x{tilemap:02X}")

        input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
        if input_lock != 1:
            return dismiss_dialogue_result(self._step_count, pulse_every=1)

        if self._held_item(world.ram) == ITEM_CHICKEN:
            return TaskResult(status=TaskStatus.SUCCESS, reason="holding chicken")

        queued = drain_action_queue(self._action_queue)
        if queued is not None:
            return queued

        if self._phase == "nav":
            target = self._target
            blockers = self._blocker_tiles(world.ram)
            if (
                target is None
                or (target[0] in blockers and target[0] != self._navigator.current_tile)
                or target[2] not in self._adult_chicken_tiles(world.ram)
            ):
                target = self._select_target(world.ram)
                self._target = target
                self._navigator.path = []
            if target is None:
                return TaskResult(status=TaskStatus.FAILURE, reason="no reachable adult chicken")

            stand, face, _chicken_tile = target
            action = self._navigate_to_tile(world.ram, stand)
            if action is not None:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

            self._queue_pickup(face)
            self._attempts += 1
            self._verify_count = 0
            self._phase = "verify"
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        if self._phase == "verify":
            if self._held_item(world.ram) == ITEM_CHICKEN:
                return TaskResult(status=TaskStatus.SUCCESS, reason="holding chicken")
            self._verify_count += 1
            if self._verify_count > 24:
                if self._attempts >= 5:
                    return TaskResult(status=TaskStatus.FAILURE, reason="pickup did not register")
                self._phase = "nav"
                self._target = None
                self._navigator.path = []
                self._verify_count = 0
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        return TaskResult(status=TaskStatus.FAILURE, reason=f"unknown pickup phase {self._phase}")


@dataclass
class DropCarriedChickenTask(Task):
    """Carry the chicken to the farm drop point used by the sale event."""

    name: str = "drop_carried_chicken"
    target_px: Tuple[int, int] = (60, 480)
    radius: int = 2
    timeout: int = 3000

    _phase: str = field(default="nav", init=False)
    _step_count: int = field(default=0, init=False)
    _task: Optional[Task] = field(default=None, init=False)
    _action_queue: deque = field(default_factory=deque, init=False)
    _verify_count: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        self._phase = "nav"
        self._step_count = 0
        self._task = NavTask(
            name="nav_chicken_sale_drop",
            target_px=Point(self.target_px[0], self.target_px[1]),
            radius=self.radius,
            timeout=self.timeout,
            stasis_repath=90,
        )
        self._task.reset(world)
        self._action_queue.clear()
        self._verify_count = 0

    def can_start(self, world: WorldState) -> bool:
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        return is_farm_tilemap(tilemap)

    def _held_item(self, ram: np.ndarray) -> int:
        return read_ram_u8(ram, ADDR_ITEM_ON_HAND)

    def _queue_drop(self) -> None:
        self._action_queue.extend(
            press_a_sequence(
                None,
                face_frames=0,
                pre_press_settle_frames=4,
                hold_frames=10,
                settle_frames=30,
            )
        )

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        if self._step_count > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason="chicken drop timeout")

        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if not is_farm_tilemap(tilemap) and self._phase != "reenter_farm":
            return TaskResult(status=TaskStatus.BLOCKED, reason=f"not on farm tilemap=0x{tilemap:02X}")

        input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
        if input_lock != 1:
            return dismiss_dialogue_result(self._step_count, pulse_every=1)

        queued = drain_action_queue(self._action_queue)
        if queued is not None:
            return queued

        if self._phase == "nav":
            if self._task is None:
                self.reset(world)
            assert self._task is not None
            result = self._task.step(world)
            if result.status == TaskStatus.RUNNING:
                return result
            if result.status != TaskStatus.SUCCESS:
                return TaskResult(status=result.status, reason=f"drop nav failed: {result.reason or result.status.value}")
            if self._held_item(world.ram) != ITEM_CHICKEN:
                return TaskResult(status=TaskStatus.SUCCESS, reason="not holding chicken at drop point")
            self._queue_drop()
            self._phase = "verify"
            self._verify_count = 0
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        if self._phase == "verify":
            if self._held_item(world.ram) != ITEM_CHICKEN:
                return TaskResult(status=TaskStatus.SUCCESS, reason="chicken dropped")
            self._verify_count += 1
            if self._verify_count > 45:
                self._queue_drop()
                self._verify_count = 0
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        return TaskResult(status=TaskStatus.FAILURE, reason=f"unknown drop phase {self._phase}")


@dataclass
class ChickenSaleFollowupTask(Task):
    """Replay the verified sale follow-up after the chicken is dropped."""

    name: str = "chicken_sale_followup"
    task_name: str = "sell_chicken"
    start_frame: int = 1295
    end_frame: Optional[int] = None
    tasks_dir: str = TASKS_DIR
    require_start_px: Optional[Tuple[int, int]] = (60, 480)
    start_tolerance: int = 12
    success_settle_frames: int = 30

    _frames: List[List[int]] = field(default_factory=list, init=False)
    _idx: int = field(default=0, init=False)
    _start_chickens: int = field(default=0, init=False)
    _start_shipping_money: int = field(default=0, init=False)
    _sale_seen: bool = field(default=False, init=False)
    _money_seen: bool = field(default=False, init=False)
    _success_frames: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        recording = RecordedTask.load(self.task_name, self.tasks_dir)
        self._frames = recording.frames[self.start_frame:self.end_frame]
        self._idx = 0
        self._start_chickens = read_ram_u8(world.ram, ADDR_CHICKEN_COUNT)
        self._start_shipping_money = read_shipping_money(world.ram)
        self._sale_seen = False
        self._money_seen = False
        self._success_frames = 0

    def can_start(self, world: WorldState) -> bool:
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        return is_farm_tilemap(tilemap)

    def _start_position_ok(self, ram: np.ndarray) -> bool:
        if self.require_start_px is None:
            return True
        pos = get_pos_from_ram(ram)
        return (
            abs(pos.x - self.require_start_px[0]) <= self.start_tolerance
            and abs(pos.y - self.require_start_px[1]) <= self.start_tolerance
        )

    def _update_observed_sale(self, ram: np.ndarray) -> None:
        current_chickens = read_ram_u8(ram, ADDR_CHICKEN_COUNT)
        current_shipping_money = read_shipping_money(ram)
        if current_chickens < self._start_chickens:
            self._sale_seen = True
        if current_shipping_money > self._start_shipping_money:
            self._money_seen = True

    def _success_ready(self, ram: np.ndarray) -> bool:
        self._update_observed_sale(ram)
        tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
        input_lock = int(ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(ram) else 1
        if self._sale_seen and self._money_seen and is_farm_tilemap(tilemap) and input_lock == 1:
            self._success_frames += 1
        else:
            self._success_frames = 0
        return self._success_frames >= self.success_settle_frames

    def _completion_reason(self, ram: np.ndarray) -> str:
        return (
            f"chickens {self._start_chickens}->{read_ram_u8(ram, ADDR_CHICKEN_COUNT)} "
            f"shipping {self._start_shipping_money}->{read_shipping_money(ram)}"
        )

    def step(self, world: WorldState) -> TaskResult:
        if not self._frames:
            return TaskResult(status=TaskStatus.FAILURE, reason="sell_chicken recording slice empty")

        if self._idx == 0:
            tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
            if not is_farm_tilemap(tilemap):
                return TaskResult(status=TaskStatus.FAILURE, reason=f"expected farm start, got tilemap=0x{tilemap:02X}")
            if not self._start_position_ok(world.ram):
                pos = get_pos_from_ram(world.ram)
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=f"expected drop start near {self.require_start_px}, got ({pos.x},{pos.y})",
                )

        if self._success_ready(world.ram):
            return TaskResult(status=TaskStatus.SUCCESS, reason=self._completion_reason(world.ram))

        if self._idx >= len(self._frames):
            self._update_observed_sale(world.ram)
            if self._sale_seen and self._money_seen:
                return TaskResult(status=TaskStatus.SUCCESS, reason=self._completion_reason(world.ram))
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=(
                    "chicken sale did not register "
                    f"({self._completion_reason(world.ram)}, frames={len(self._frames)})"
                ),
            )

        action = np.asarray(self._frames[self._idx], dtype=np.int32)
        self._idx += 1
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))


@dataclass
class ChickenSaleRequestTask(Task):
    """Request a chicken sale at the animal-shop counter."""

    name: str = "chicken_sale_request"
    task_name: str = "sell_chicken"
    start_frame: int = 2863
    end_frame: Optional[int] = 3297
    tasks_dir: str = TASKS_DIR
    require_start_px: Tuple[int, int] = (201, 158)
    start_tolerance: int = 6
    timeout: int = 1200
    request_text_id: int = 0x030B

    _frames: List[List[int]] = field(default_factory=list, init=False)
    _idx: int = field(default=0, init=False)
    _step_count: int = field(default=0, init=False)
    _request_seen: bool = field(default=False, init=False)
    _saw_shop_menu: bool = field(default=False, init=False)
    _settle_frames: int = field(default=0, init=False)
    _phase: str = field(default="align", init=False)
    _task: Optional[Task] = field(default=None, init=False)
    _action_queue: deque = field(default_factory=deque, init=False)

    def reset(self, world: WorldState) -> None:
        self._frames = []
        self._idx = 0
        self._step_count = 0
        self._request_seen = False
        self._saw_shop_menu = False
        self._settle_frames = 0
        self._phase = "align"
        self._task = None
        self._action_queue.clear()

    def can_start(self, world: WorldState) -> bool:
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        return tilemap == 0x24

    def _start_position_ok(self, ram: np.ndarray) -> bool:
        pos = get_pos_from_ram(ram)
        return (
            abs(pos.x - self.require_start_px[0]) <= self.start_tolerance
            and abs(pos.y - self.require_start_px[1]) <= self.start_tolerance
        )

    def _aligned_for_replay(self, ram: np.ndarray) -> bool:
        pos = get_pos_from_ram(ram)
        return abs(pos.x - self.require_start_px[0]) <= 1 and abs(pos.y - self.require_start_px[1]) <= 1

    def _update_request_seen(self, ram: np.ndarray) -> None:
        if self._saw_shop_menu and self._dialog_text_id(ram) == self.request_text_id:
            self._request_seen = True

    def _dialog_text_id(self, ram: np.ndarray) -> int:
        return read_ram_u16(ram, field_spec("dialog_text_id").address, live_offset=False)

    def _queue_open_menu(self) -> None:
        self._action_queue.extend(
            press_a_sequence(
                "right",
                face_frames=4,
                pre_press_settle_frames=0,
                hold_frames=12,
                settle_frames=12,
                hold_face_with_a=True,
            )
        )

    def _queue_sell_chicken_choice(self) -> None:
        # Match the successful counter-menu cadence from sell_chicken.json.
        # The shop menu ignores early A pulses, then two Down taps select
        # "sell chicken" (dialog text 0x030B) from the four-option menu.
        self._action_queue.extend(make_action(right=True, a=True) for _ in range(6))
        self._action_queue.extend(make_action(right=True) for _ in range(7))
        self._action_queue.extend(make_action(right=True, a=True) for _ in range(3))
        self._action_queue.extend(make_action(a=True) for _ in range(4))
        self._action_queue.extend(make_action() for _ in range(12))
        self._action_queue.extend(press_button_sequence("a", hold_frames=8, settle_frames=19))
        self._action_queue.extend(press_button_sequence("a", hold_frames=8, settle_frames=141))
        self._action_queue.extend(press_button_sequence("down", hold_frames=11, settle_frames=7))
        self._action_queue.extend(press_button_sequence("down", hold_frames=7, settle_frames=16))
        self._action_queue.extend(press_button_sequence("a", hold_frames=10, settle_frames=30))

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        if self._step_count > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason="chicken sale request timeout")

        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if tilemap != 0x24:
            return TaskResult(status=TaskStatus.BLOCKED, reason=f"not in animal shop tilemap=0x{tilemap:02X}")
        if self._idx == 0 and not self._start_position_ok(world.ram):
            pos = get_pos_from_ram(world.ram)
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"expected animal counter near {self.require_start_px}, got ({pos.x},{pos.y})",
            )

        if self._phase == "align":
            if not self._aligned_for_replay(world.ram):
                if self._task is None:
                    self._task = NavTask(
                        name="nav_chicken_sale_request_counter",
                        target_px=Point(self.require_start_px[0], self.require_start_px[1]),
                        radius=1,
                        timeout=240,
                        stasis_repath=30,
                    )
                    self._task.reset(world)
                result = self._task.step(world)
                if result.status == TaskStatus.RUNNING:
                    return result
                if result.status != TaskStatus.SUCCESS:
                    return TaskResult(status=result.status, reason=f"counter align failed: {result.reason or result.status.value}")
            self._phase = "replay"
            self._task = None

        self._update_request_seen(world.ram)
        input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
        if self._request_seen and input_lock == 1:
            self._settle_frames += 1
            if self._settle_frames >= 12:
                return TaskResult(status=TaskStatus.SUCCESS, reason="chicken sale requested")
        else:
            self._settle_frames = 0

        queued = drain_action_queue(self._action_queue)
        if queued is not None:
            return queued

        text_id = self._dialog_text_id(world.ram)
        if self._request_seen:
            if input_lock != 1:
                return dismiss_dialogue_result(self._step_count, buttons=("a", "b"), pulse_every=1)
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        if text_id == 0x0305:
            self._saw_shop_menu = True
            if self._phase != "choose_sale":
                self._phase = "choose_sale"
                self._queue_sell_chicken_choice()
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        if input_lock != 1:
            return dismiss_dialogue_result(self._step_count, buttons=("a",), pulse_every=2)

        self._queue_open_menu()
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))


@dataclass
class ChickenSaleEventTask(Task):
    """Finish the delayed ranch pickup/payout after requesting a chicken sale."""

    name: str = "chicken_sale_event"
    standby_px: Tuple[int, int] = (62, 448)
    payout_px: Tuple[int, int] = (146, 457)
    event_hour: int = 15
    target_sales: int = 1
    timeout: int = 18000
    success_settle_frames: int = 30

    _phase: str = field(default="nav_standby", init=False)
    _step_count: int = field(default=0, init=False)
    _task: Optional[Task] = field(default=None, init=False)
    _action_queue: deque = field(default_factory=deque, init=False)
    _start_chickens: int = field(default=0, init=False)
    _start_money: int = field(default=0, init=False)
    _sale_seen: bool = field(default=False, init=False)
    _money_seen: bool = field(default=False, init=False)
    _success_frames: int = field(default=0, init=False)
    _post_event_wait: int = field(default=0, init=False)
    _sale_settle_frames: int = field(default=0, init=False)
    _entry_settle_frames: int = field(default=0, init=False)

    def reset(self, world: WorldState) -> None:
        pos = get_pos_from_ram(world.ram)
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if is_farm_tilemap(tilemap) and pos.x > 200 and pos.y < 200:
            self._phase = "entry_settle"
        else:
            self._phase = "nav_standby"
        self._step_count = 0
        self._task = None
        self._action_queue.clear()
        self._start_chickens = read_ram_u8(world.ram, ADDR_CHICKEN_COUNT)
        self._start_money = int(read_ram_value(world.ram, "money"))
        self._sale_seen = False
        self._money_seen = False
        self._success_frames = 0
        self._post_event_wait = 0
        self._sale_settle_frames = 0
        self._entry_settle_frames = 0

    def can_start(self, world: WorldState) -> bool:
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        return is_farm_tilemap(tilemap)

    def _update_observed_sale(self, ram: np.ndarray) -> None:
        sold = self._start_chickens - read_ram_u8(ram, ADDR_CHICKEN_COUNT)
        expected_money = max(0, int(self.target_sales)) * 500
        if sold >= max(1, int(self.target_sales)):
            self._sale_seen = True
        if int(read_ram_value(ram, "money")) - self._start_money >= max(1, expected_money):
            self._money_seen = True

    def _completion_reason(self, ram: np.ndarray) -> str:
        return (
            f"chickens {self._start_chickens}->{read_ram_u8(ram, ADDR_CHICKEN_COUNT)} "
            f"money {self._start_money}->{int(read_ram_value(ram, 'money'))} "
            f"target_sales={self.target_sales}"
        )

    def _make_nav(self, name: str, px: Tuple[int, int], *, radius: int = 10, timeout: int = 3000) -> NavTask:
        return NavTask(
            name=name,
            target_px=Point(int(px[0]), int(px[1])),
            radius=radius,
            timeout=timeout,
            stasis_repath=90,
        )

    def _queue_payout_action(self) -> None:
        self._action_queue.extend(
            press_button_sequence(
                "a",
                face="down",
                face_frames=8,
                pre_press_settle_frames=0,
                hold_frames=10,
                settle_frames=45,
                hold_face_with_button=True,
            )
        )

    def _near_payout(self, ram: np.ndarray, *, radius: int = 12) -> bool:
        pos = get_pos_from_ram(ram)
        return abs(pos.x - self.payout_px[0]) <= radius and abs(pos.y - self.payout_px[1]) <= radius

    def _payout_alignment_action(self, ram: np.ndarray) -> Optional[np.ndarray]:
        pos = get_pos_from_ram(ram)
        target_x, target_y = self.payout_px
        if pos.x < target_x:
            return make_action(right=True)
        if pos.x > target_x:
            return make_action(left=True)
        if pos.y < target_y:
            return make_action(down=True)
        if pos.y > target_y:
            return make_action(up=True)
        return None

    def _start_reenter_route(self, world: WorldState) -> None:
        self._task = MultiMapNavTask(
            name="chicken_sale_reenter_farm",
            waypoints=list(ROUTES["farm_to_town"]) + list(ROUTES["town_to_farm"]),
            timeout=9000,
            initial_settle_frames=0,
        )
        self._task.reset(world)
        self._phase = "reenter_farm"

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        if self._step_count > self.timeout:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"chicken sale event timeout ({self._completion_reason(world.ram)})",
            )

        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if not is_farm_tilemap(tilemap) and self._phase != "reenter_farm":
            return TaskResult(status=TaskStatus.BLOCKED, reason=f"not on farm tilemap=0x{tilemap:02X}")

        self._update_observed_sale(world.ram)
        input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
        if input_lock != 1:
            return dismiss_dialogue_result(self._step_count, buttons=("a", "b"), pulse_every=1)

        if self._sale_seen and self._money_seen:
            self._success_frames += 1
            if self._success_frames >= self.success_settle_frames:
                return TaskResult(status=TaskStatus.SUCCESS, reason=self._completion_reason(world.ram))
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
        self._success_frames = 0

        queued = drain_action_queue(self._action_queue)
        if queued is not None:
            return queued

        if self._sale_seen and not self._money_seen:
            if self._phase not in {"nav_payout", "align_payout", "press_payout", "wait_money"}:
                self._sale_settle_frames += 1
                if self._sale_settle_frames < 60:
                    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
                self._task = self._make_nav("nav_chicken_sale_payout", self.payout_px, radius=8, timeout=2400)
                self._task.reset(world)
                self._phase = "nav_payout"

        if self._phase == "entry_settle":
            _day, hour, _minute = read_world_day_time(world.ram)
            if hour < self.event_hour:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            pos = get_pos_from_ram(world.ram)
            if (pos.x < 80 and pos.y >= 440) or self._entry_settle_frames >= 240:
                self._phase = "nav_standby"
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            self._entry_settle_frames += 1
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(b=True)))

        if self._phase == "nav_standby":
            pos = get_pos_from_ram(world.ram)
            _day, hour, _minute = read_world_day_time(world.ram)
            if hour < self.event_hour:
                if pos.x <= 48 and abs(pos.y - self.standby_px[1]) <= 8:
                    self._task = None
                    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
                if self._task is None:
                    self._task = self._make_nav(
                        "nav_chicken_sale_wait_gate",
                        (24, self.standby_px[1]),
                        radius=8,
                        timeout=2000,
                    )
                    self._task.reset(world)
                result = self._task.step(world)
                if result.status == TaskStatus.RUNNING:
                    return result
                self._task = None
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            if pos.x <= self.standby_px[0] and abs(pos.y - self.standby_px[1]) <= 4:
                if pos.x < self.standby_px[0]:
                    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(right=True, b=True)))
                self._task = None
                self._phase = "wait_event"
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            if self._task is None:
                self._task = self._make_nav("nav_chicken_sale_standby", self.standby_px, radius=2, timeout=4000)
                self._task.reset(world)
            result = self._task.step(world)
            if result.status == TaskStatus.RUNNING:
                return result
            self._task = None
            self._phase = "wait_event"
            if result.status not in {TaskStatus.SUCCESS, TaskStatus.FAILURE}:
                return result

        if self._phase == "wait_event":
            _day, hour, _minute = read_world_day_time(world.ram)
            if hour < self.event_hour:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            player_state = read_ram_u8(world.ram, field_spec("player_state").address)
            held_item = read_ram_u8(world.ram, ADDR_ITEM_ON_HAND)
            if player_state in {0x43, 0x83} or held_item == 0x03:
                self._post_event_wait = 0
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            self._post_event_wait += 1
            if self._post_event_wait > 300:
                self._start_reenter_route(world)
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        if self._phase == "reenter_farm":
            if self._task is None:
                self._start_reenter_route(world)
            assert self._task is not None
            result = self._task.step(world)
            if result.status == TaskStatus.RUNNING:
                return result
            self._task = None
            self._phase = "wait_event"
            self._post_event_wait = 0
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        if self._phase == "nav_payout":
            assert self._task is not None
            result = self._task.step(world)
            if result.status == TaskStatus.RUNNING:
                return result
            if result.status != TaskStatus.SUCCESS and not self._near_payout(world.ram):
                return TaskResult(status=result.status, reason=f"payout nav failed: {result.reason or result.status.value}")
            self._task = None
            self._phase = "align_payout"
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        if self._phase == "align_payout":
            align_action = self._payout_alignment_action(world.ram)
            if align_action is not None:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(align_action))
            self._queue_payout_action()
            self._phase = "wait_money"
            self._post_event_wait = 0
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        if self._phase == "wait_money":
            self._post_event_wait += 1
            if self._post_event_wait > 240 and not self._money_seen:
                self._queue_payout_action()
                self._post_event_wait = 0
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        return TaskResult(status=TaskStatus.FAILURE, reason=f"unknown chicken sale event phase {self._phase}")


@dataclass
class CowPurchaseTask(Task):
    """Replay the precise shop interaction and require the cow count to change."""

    name: str = "cow_purchase"
    task_name: str = "buy_cow"
    start_frame: int = 1631
    end_frame: Optional[int] = 2328
    tasks_dir: str = TASKS_DIR

    _frames: List[List[int]] = field(default_factory=list, init=False)
    _idx: int = field(default=0, init=False)
    _start_cows: int = field(default=0, init=False)
    _saw_purchase: bool = field(default=False, init=False)

    def reset(self, world: WorldState) -> None:
        recording = RecordedTask.load(self.task_name, self.tasks_dir)
        self._frames = recording.frames[self.start_frame:self.end_frame]
        self._idx = 0
        self._start_cows = read_ram_u8(world.ram, ADDR_COW_COUNT)
        self._saw_purchase = False

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        current_cows = read_ram_u8(world.ram, ADDR_COW_COUNT)
        if current_cows > self._start_cows:
            self._saw_purchase = True

        if self._idx >= len(self._frames):
            if self._saw_purchase:
                return TaskResult(
                    status=TaskStatus.SUCCESS,
                    reason=f"cows {self._start_cows}->{current_cows}",
                )
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"cow purchase did not register (cows={current_cows})",
            )

        action = np.asarray(self._frames[self._idx], dtype=np.int32)
        self._idx += 1
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))



__all__ = [
    "ITEM_CHICKEN",
    "CoopPickupChickenTask",
    "DropCarriedChickenTask",
    "ChickenSaleFollowupTask",
    "ChickenSaleRequestTask",
    "ChickenSaleEventTask",
    "CowPurchaseTask",
]

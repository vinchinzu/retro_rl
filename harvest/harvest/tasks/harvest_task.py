"""Dynamic crop harvest + ship task driven by the save state's ripe tiles."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import os
import sys
from typing import Deque, List, Optional, Tuple

import numpy as np


from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from harvest.tasks.crop_planter import DEFAULT_CROP_BOUNDS, is_crop_tile, is_mature_crop_tile
from harvest.tasks.farm_clearer import (
    ADDR_INPUT_LOCK,
    MAP_WIDTH,
    TILE_SIZE,
    WALKABLE_TILES,
    Navigator,
    Pathfinder,
    Point,
    TileScanner,
    get_pos_from_ram,
    get_tile_at,
    make_action,
)
from harvest.core.harvest_state import HarvestStateDocument
from harvest.core.ram_catalog import field_spec, live_wram_base, read_ram_u24

ADDR_PLAYER_STATE = field_spec("player_state").address
ACTION_CARRYING_BIT = 0x02
ADDR_SHIPPING_MONEY = field_spec("shipping_money").address
READY_VISIBLE_MIN = 0xA0
SHIP_STAND_TILE = (11, 30)
SHIP_FACE = "left"
SHIP_FALLBACKS: Tuple[Tuple[Tuple[int, int], str], ...] = (
    ((11, 30), "left"),
    ((8, 32), "up"),
)


def is_carrying(ram: np.ndarray) -> bool:
    idx = ADDR_PLAYER_STATE + live_wram_base(ram)
    return bool(int(ram[idx]) & ACTION_CARRYING_BIT) if idx < len(ram) else False


def read_shipping_money(ram: np.ndarray) -> int:
    return read_ram_u24(ram, ADDR_SHIPPING_MONEY) * 10


def is_ripe_crop_tile(tile_id: int) -> bool:
    return is_mature_crop_tile(tile_id)


def _live_harvestable_crop_tiles_from_ram(
    ram: np.ndarray,
    *,
    bounds: Tuple[int, int, int, int] = DEFAULT_CROP_BOUNDS,
) -> List[Tuple[int, int]]:
    left, top, right, bottom = bounds
    candidates: List[Tuple[int, int]] = []
    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            tile_id = get_tile_at(ram, x, y)
            if not is_ripe_crop_tile(tile_id):
                continue
            candidates.append((x, y))
    return candidates


def state_harvestable_crop_tiles(
    state_name: Optional[str],
    *,
    bounds: Tuple[int, int, int, int] = DEFAULT_CROP_BOUNDS,
) -> List[Tuple[int, int]]:
    """Return ripe crop tiles using explicit pickable crop tile IDs."""
    if not state_name:
        return []
    try:
        document = HarvestStateDocument.load(state_name)
    except FileNotFoundError:
        return []

    left, top, right, bottom = bounds
    candidates: List[Tuple[int, int]] = []
    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            tile = document.farm_tile(x, y)
            if not is_ripe_crop_tile(tile.persistent_value):
                continue
            if tile.visible_value < READY_VISIBLE_MIN:
                continue
            candidates.append((x, y))

    if candidates:
        return candidates

    ram_array = getattr(document, "ram_array", None)
    if callable(ram_array):
        return _live_harvestable_crop_tiles_from_ram(np.asarray(ram_array(), dtype=np.uint8), bounds=bounds)
    return []


def live_harvestable_crop_tiles(
    ram: np.ndarray,
    state_name: Optional[str],
    *,
    bounds: Tuple[int, int, int, int] = DEFAULT_CROP_BOUNDS,
) -> List[Tuple[int, int]]:
    """Return ripe crop tiles by combining persistent crop positions with live map tiles."""
    if not state_name:
        return _live_harvestable_crop_tiles_from_ram(ram, bounds=bounds)
    try:
        document = HarvestStateDocument.load(state_name)
    except FileNotFoundError:
        return _live_harvestable_crop_tiles_from_ram(ram, bounds=bounds)

    left, top, right, bottom = bounds
    document_candidates: List[Tuple[int, int]] = []
    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            tile = document.farm_tile(x, y)
            if not is_ripe_crop_tile(tile.persistent_value):
                continue
            if tile.visible_value < READY_VISIBLE_MIN:
                continue
            document_candidates.append((x, y))

    if not document_candidates:
        return _live_harvestable_crop_tiles_from_ram(ram, bounds=bounds)

    candidates: List[Tuple[int, int]] = []
    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            tile = document.farm_tile(x, y)
            if not is_ripe_crop_tile(tile.persistent_value):
                continue
            if tile.visible_value < READY_VISIBLE_MIN:
                continue
            live_tile = get_tile_at(ram, x, y)
            if not is_ripe_crop_tile(live_tile):
                continue
            candidates.append((x, y))

    return candidates


@dataclass(frozen=True)
class HarvestStep:
    target: Tuple[int, int]
    stand: Tuple[int, int]
    face: str
    group: int = 0


def _target_groups(target_tiles: List[Tuple[int, int]]) -> dict[Tuple[int, int], int]:
    """Group adjacent crop cells so harvesting clears one plot before moving on."""
    targets = set(target_tiles)
    groups: dict[Tuple[int, int], int] = {}
    group_id = 0
    for start in target_tiles:
        if start in groups:
            continue
        stack = [start]
        groups[start] = group_id
        while stack:
            tx, ty = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nxt = (tx + dx, ty + dy)
                    if nxt in targets and nxt not in groups:
                        groups[nxt] = group_id
                        stack.append(nxt)
        group_id += 1
    return groups


def build_harvest_steps(
    ram: np.ndarray,
    target_tiles: List[Tuple[int, int]],
    *,
    ship_stand: Tuple[int, int] = SHIP_STAND_TILE,
) -> List[HarvestStep]:
    """Build adjacent stand tiles for each ripe crop tile."""
    face_to_delta = {
        "up": (0, 1),
        "down": (0, -1),
        "left": (1, 0),
        "right": (-1, 0),
    }
    groups = _target_groups(target_tiles)
    steps: List[HarvestStep] = []

    for target in target_tiles:
        choices: List[Tuple[Tuple[int, int, int, int], HarvestStep]] = []
        for face, (dx, dy) in face_to_delta.items():
            stand = (target[0] + dx, target[1] + dy)
            sx, sy = stand
            if not (0 <= sx < MAP_WIDTH and 0 <= sy < MAP_WIDTH):
                continue
            stand_tid = get_tile_at(ram, sx, sy)
            if stand_tid not in WALKABLE_TILES:
                continue
            score = (
                0,
                abs(ship_stand[0] - sx) + abs(ship_stand[1] - sy),
                abs(target[1] - ship_stand[1]),
                abs(target[0] - ship_stand[0]),
            )
            choices.append((score, HarvestStep(target=target, stand=stand, face=face, group=groups.get(target, 0))))

        if choices:
            choices.sort(key=lambda item: item[0])
            steps.append(choices[0][1])

    steps.sort(
        key=lambda step: (
            abs(step.stand[0] - ship_stand[0]) + abs(step.stand[1] - ship_stand[1]),
            step.group,
            step.stand[1],
            step.stand[0],
        )
    )
    return steps


# Virgin plant / water days: ship-area fallback (136,520) made the planner
# pick unreachable south plots. Prefer the open early-spring field anchor
# used by crop_planner.DEFAULT_START_TILE (15, 29).
PREFERRED_PLANT_TILE: Tuple[int, int] = (15, 29)
PREFERRED_PLANT_PX: Tuple[int, int] = (
    PREFERRED_PLANT_TILE[0] * TILE_SIZE + 8,
    PREFERRED_PLANT_TILE[1] * TILE_SIZE + 8,
)


def crop_nav_target_px(
    ram: np.ndarray,
    state_name: Optional[str],
    *,
    bounds: Tuple[int, int, int, int] = DEFAULT_CROP_BOUNDS,
    fallback_px: Tuple[int, int] = PREFERRED_PLANT_PX,
) -> Tuple[int, int]:
    """Choose the nearest crop-adjacent stand for the pre-crop NAV_CROP phase."""
    target_tiles = state_harvestable_crop_tiles(state_name, bounds=bounds)
    if not target_tiles:
        target_tiles = _live_harvestable_crop_tiles_from_ram(ram, bounds=bounds)
    if target_tiles:
        steps = build_harvest_steps(ram, target_tiles)
        if steps:
            pos = get_pos_from_ram(ram)
            current_tile = (pos.x // TILE_SIZE, pos.y // TILE_SIZE)
            step = min(
                steps,
                key=lambda item: (
                    abs(item.stand[0] - current_tile[0])
                    + abs(item.stand[1] - current_tile[1]),
                    item.group,
                    item.stand[1],
                    item.stand[0],
                ),
            )
            return (step.stand[0] * TILE_SIZE + 8, step.stand[1] * TILE_SIZE + 8)

    # Planting / watering days: aim at the nearest detected plot center.
    from harvest.tasks.crop_planter import detect_plots

    plots = detect_plots(ram, bounds)
    if plots:
        pos = get_pos_from_ram(ram)
        current_tile = (pos.x // TILE_SIZE, pos.y // TILE_SIZE)
        center = min(
            plots,
            key=lambda tile: (
                abs(tile[0] - current_tile[0]) + abs(tile[1] - current_tile[1]),
                tile[1],
                tile[0],
            ),
        )
        return (center[0] * TILE_SIZE + 8, center[1] * TILE_SIZE + 8)

    return fallback_px


@dataclass
class HarvestTask(Task):
    """Harvest ripe crops from the current save layout and ship them one by one."""

    name: str = "harvest"
    state_name: Optional[str] = None
    bounds: Tuple[int, int, int, int] = DEFAULT_CROP_BOUNDS
    ship_stand: Tuple[int, int] = SHIP_STAND_TILE
    ship_face: str = SHIP_FACE
    ship_fallbacks: Tuple[Tuple[Tuple[int, int], str], ...] = SHIP_FALLBACKS
    timeout: int = 20000

    _scanner: TileScanner = field(default_factory=TileScanner, init=False)
    _pathfinder: Pathfinder = field(init=False)
    _navigator: Navigator = field(init=False)
    _phase: str = field(default="select", init=False)
    _steps: List[HarvestStep] = field(default_factory=list, init=False)
    _current: Optional[HarvestStep] = field(default=None, init=False)
    _action_queue: Deque[np.ndarray] = field(default_factory=deque, init=False)
    _step_count: int = field(default=0, init=False)
    _verify_count: int = field(default=0, init=False)
    _ship_money_before: int = field(default=0, init=False)
    _ship_options: List[Tuple[Tuple[int, int], str]] = field(default_factory=list, init=False)
    _ship_option_index: int = field(default=0, init=False)
    _target_live_before: int = field(default=-1, init=False)
    _initial_target_count: int = field(default=0, init=False)
    _unreachable_count: int = field(default=0, init=False)
    _active_group: Optional[int] = field(default=None, init=False)
    harvested_count: int = field(default=0, init=False)
    shipped_count: int = field(default=0, init=False)
    skipped_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._pathfinder = Pathfinder(self._scanner)
        self._navigator = Navigator(self._pathfinder)

    def reset(self, world: WorldState) -> None:
        self._phase = "select"
        self._current = None
        self._action_queue.clear()
        self._step_count = 0
        self._verify_count = 0
        self._target_live_before = -1
        self._ship_options = []
        self._ship_option_index = 0
        self._initial_target_count = 0
        self._unreachable_count = 0
        self._active_group = None
        self.harvested_count = 0
        self.shipped_count = 0
        self.skipped_count = 0
        self._navigator.path = []
        self._navigator.stasis = 0
        self._pathfinder.temp_blocked.clear()

        target_tiles = live_harvestable_crop_tiles(world.ram, self.state_name, bounds=self.bounds)
        self._steps = build_harvest_steps(world.ram, target_tiles, ship_stand=self.ship_stand)
        self._initial_target_count = len(target_tiles)
        self._unreachable_count = max(0, len(target_tiles) - len(self._steps))

        self._pathfinder.extra_walkable = {step.stand for step in self._steps}
        print(f"[HARVEST] Detected {len(self._steps)} ripe crop targets from live map")

    def can_start(self, world: WorldState) -> bool:
        return True

    @property
    def progress_text(self) -> str:
        total = self._initial_target_count
        return (
            f"harvested={self.harvested_count} shipped={self.shipped_count} "
            f"skipped={self.skipped_count} unreachable={self._unreachable_count} "
            f"remaining={len(self._steps)} total={total}"
        )

    def _completion_result(self) -> TaskResult:
        reason = (
            f"harvested={self.harvested_count} shipped={self.shipped_count} "
            f"skipped={self.skipped_count} unreachable={self._unreachable_count}"
        )
        if self._initial_target_count == 0:
            return TaskResult(status=TaskStatus.SUCCESS, reason=reason)
        if (
            self.skipped_count > 0
            or self._unreachable_count > 0
            or self.harvested_count < self._initial_target_count
            or self.shipped_count < self.harvested_count
        ):
            return TaskResult(status=TaskStatus.FAILURE, reason=f"incomplete harvest: {reason}")
        return TaskResult(status=TaskStatus.SUCCESS, reason=reason)

    def _queue_press_a(
        self,
        face: str,
        *,
        face_frames: int = 2,
        hold_frames: int = 14,
        settle_frames: int = 10,
    ) -> None:
        self._action_queue.extend(make_action(**{face: True}) for _ in range(face_frames))
        self._action_queue.extend(make_action() for _ in range(2))
        self._action_queue.extend(make_action(a=True) for _ in range(hold_frames))
        self._action_queue.extend(make_action() for _ in range(settle_frames))

    def _dialog_pulse_action(self) -> np.ndarray:
        return make_action(a=True) if self._step_count % 2 == 0 else make_action()

    def _pop_action(self) -> Optional[np.ndarray]:
        if self._action_queue:
            return self._action_queue.popleft()
        return None

    def _clear_navigation_state(self) -> None:
        self._navigator.path = []
        self._navigator.stasis = 0
        self._pathfinder.temp_blocked.clear()

    def _choose_next_step(self) -> Optional[HarvestStep]:
        if not self._steps:
            return None
        if self._active_group is not None and not any(step.group == self._active_group for step in self._steps):
            self._active_group = None
        current = self._navigator.current_tile
        if self._active_group is None:
            self._active_group = min(
                {step.group for step in self._steps},
                key=lambda group: min(
                    abs(step.stand[0] - current[0]) + abs(step.stand[1] - current[1])
                    for step in self._steps
                    if step.group == group
                ),
            )
        group_indices = [idx for idx, step in enumerate(self._steps) if step.group == self._active_group]
        best_idx = min(
            group_indices,
            key=lambda idx: abs(self._steps[idx].stand[0] - current[0]) + abs(self._steps[idx].stand[1] - current[1]),
        )
        return self._steps.pop(best_idx)

    def _reset_ship_options(self, ram: np.ndarray) -> None:
        options: List[Tuple[Tuple[int, int], str]] = []
        for stand, face in self.ship_fallbacks:
            if stand not in [item[0] for item in options]:
                options.append((stand, face))
        if self.ship_stand not in [item[0] for item in options]:
            options.append((self.ship_stand, self.ship_face))

        current = self._navigator.current_tile
        scored: List[Tuple[Tuple[int, int, int, int], Tuple[Tuple[int, int], str]]] = []
        for index, (stand, face) in enumerate(options):
            path = self._pathfinder.find_path(ram, current, stand)
            if path is not None:
                score = (0, len(path), abs(stand[0] - current[0]) + abs(stand[1] - current[1]), index)
            else:
                score = (1, abs(stand[0] - current[0]) + abs(stand[1] - current[1]), 0, index)
            scored.append((score, (stand, face)))

        self._ship_options = [option for _score, option in sorted(scored, key=lambda item: item[0])]
        self._ship_option_index = 0

    def _current_ship_option(self, ram: np.ndarray) -> Tuple[Tuple[int, int], str]:
        if not self._ship_options:
            self._reset_ship_options(ram)
        if not self._ship_options:
            return self.ship_stand, self.ship_face
        self._ship_option_index = min(self._ship_option_index, len(self._ship_options) - 1)
        return self._ship_options[self._ship_option_index]

    def _try_next_ship_option(self, ram: np.ndarray) -> bool:
        if not self._ship_options:
            self._reset_ship_options(ram)
        self._ship_option_index += 1
        if self._ship_option_index >= len(self._ship_options):
            return False
        self._clear_navigation_state()
        self._verify_count = 0
        self._phase = "ship_nav"
        stand, face = self._ship_options[self._ship_option_index]
        print(f"[HARVEST] Retrying ship from stand={stand} face={face}")
        return True

    def _fallback_action(self, goal: Tuple[int, int]) -> np.ndarray:
        current = self._navigator.current_tile
        dx = goal[0] - current[0]
        dy = goal[1] - current[1]
        if abs(dx) >= abs(dy):
            primary = "right" if dx > 0 else "left"
            secondary = "down" if dy > 0 else "up"
        else:
            primary = "down" if dy > 0 else "up"
            secondary = "right" if dx > 0 else "left"

        opposites = {"up": "down", "down": "up", "left": "right", "right": "left"}
        stasis = self._navigator.stasis
        if stasis < 45:
            direction = primary
        elif stasis < 90:
            direction = secondary
        elif stasis < 135:
            direction = opposites[primary]
        else:
            direction = opposites[secondary]
        return make_action(**{direction: True, "b": True})

    def _navigate_to_tile(self, ram: np.ndarray, goal: Tuple[int, int]) -> Optional[np.ndarray]:
        if self._navigator.current_tile == goal or self._navigator.at_tile(goal):
            return self._navigator.center_on_tile(goal, tolerance=1)

        if self._navigator.stasis > 120 and self._navigator.path:
            self._pathfinder.temp_blocked.add(self._navigator.path[0])
            self._navigator.path = []

        if not self._navigator.path:
            path = self._pathfinder.find_path(
                ram,
                self._navigator.current_tile,
                goal,
                max_steps=7,
            )
            if path is None:
                return self._fallback_action(goal)
            self._navigator.path = path

        action = self._navigator.follow_path(ram)
        if action is None:
            return self._fallback_action(goal)
        return action

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        self._navigator.update(world.ram)
        if self._step_count > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason="harvest timeout")

        carrying = is_carrying(world.ram)

        action = self._pop_action()
        if action is not None:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

        if self._phase == "select":
            if carrying:
                self._clear_navigation_state()
                self._phase = "ship_nav"
            else:
                self._current = self._choose_next_step()
                if self._current is None:
                    return self._completion_result()
                self._clear_navigation_state()
                self._phase = "target_nav"
                self._ship_options = []
                self._ship_option_index = 0
                print(f"[HARVEST] Target {self._current.target} stand={self._current.stand} face={self._current.face}")

        if self._phase == "target_nav":
            action = self._navigate_to_tile(world.ram, self._current.stand)
            if action is not None:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            self._target_live_before = get_tile_at(world.ram, *self._current.target)
            self._queue_press_a(self._current.face)
            self._verify_count = 0
            self._phase = "target_verify"
            return TaskResult(status=TaskStatus.RUNNING)

        if self._phase == "target_verify":
            if carrying:
                self.harvested_count += 1
                self._clear_navigation_state()
                self._phase = "ship_nav"
                print(f"[HARVEST] PICK OK target={self._current.target} carrying=1")
                return TaskResult(status=TaskStatus.RUNNING)
            target_tile_id = get_tile_at(world.ram, *self._current.target)
            tile_changed_after_pick = (
                self._target_live_before >= 0
                and target_tile_id != self._target_live_before
                and (not is_crop_tile(target_tile_id) or target_tile_id < self._target_live_before)
            )
            if (
                (self.state_name is not None and tile_changed_after_pick)
                or (self.state_name is None and not is_crop_tile(target_tile_id))
            ):
                self.harvested_count += 1
                self._clear_navigation_state()
                self._phase = "ship_nav" if is_carrying(world.ram) else "select"
                print(f"[HARVEST] PICK OK target={self._current.target} tile changed")
                return TaskResult(status=TaskStatus.RUNNING)
            self._verify_count += 1
            if self._verify_count > 12:
                self.skipped_count += 1
                print(f"[HARVEST] SKIP target={self._current.target} (not ready or no pickup)")
                self._current = None
                self._clear_navigation_state()
                self._phase = "select"
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        if self._phase == "ship_nav":
            if not carrying:
                self._current = None
                self._phase = "select"
                return TaskResult(status=TaskStatus.RUNNING)
            ship_stand, ship_face = self._current_ship_option(world.ram)
            action = self._navigate_to_tile(world.ram, ship_stand)
            if action is not None:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            self._ship_money_before = read_shipping_money(world.ram)
            self._queue_press_a(ship_face, hold_frames=14, settle_frames=12)
            self._verify_count = 0
            self._phase = "ship_verify"
            print(f"[HARVEST] SHIP from stand={ship_stand} face={ship_face} money={self._ship_money_before}")
            return TaskResult(status=TaskStatus.RUNNING)

        if self._phase == "ship_verify":
            carrying = is_carrying(world.ram)
            ship_money = read_shipping_money(world.ram)
            input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
            if input_lock != 1:
                self._verify_count += 1
                if self._verify_count > 180:
                    return TaskResult(status=TaskStatus.FAILURE, reason="ship dialog timeout")
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(self._dialog_pulse_action()),
                    reason="dialog",
                )
            if not carrying:
                if ship_money > self._ship_money_before:
                    self.shipped_count += max(1, (ship_money - self._ship_money_before) // 80)
                else:
                    return TaskResult(status=TaskStatus.FAILURE, reason="crop cleared without shipping money")
                print(f"[HARVEST] SHIP OK money={ship_money} shipped={self.shipped_count}")
                self._current = None
                self._clear_navigation_state()
                self._phase = "select"
                return TaskResult(status=TaskStatus.RUNNING)
            self._verify_count += 1
            if self._verify_count > 20:
                if self._try_next_ship_option(world.ram):
                    return TaskResult(status=TaskStatus.RUNNING)
                return TaskResult(status=TaskStatus.FAILURE, reason="ship verify timeout")
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        return TaskResult(status=TaskStatus.FAILURE, reason=f"unknown phase {self._phase}")

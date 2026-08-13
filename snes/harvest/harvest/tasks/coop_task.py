"""Autonomous chicken coop chores — scales to 12 chickens.

Phases:
  feed → collect_egg → decide → incubate / ship / gift → exit_prep → done

Feed places N hay (one per adult chicken). Egg collection picks up the
daily egg if available. The decide phase routes the egg to:
  - Incubator (if empty and flock < max)
  - Shipping bin (sell for 5G)
  - Gift carry-out (exit coop holding egg)

All terminal branches regroup at a common exit staging tile so the outer
planner can use one reliable coop-exit transition instead of depending on
whatever tile the last interaction happened to leave us on.

Extracted arms:
  - ``coop_layout`` — stands, flags, route constants
  - ``coop_feed_ops`` — feed bin / trough phase mixin
  - ``coop_egg_ops`` — egg / incubate / ship / exit-prep mixin
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import numpy as np

from harvest.core.animal_probe import COOP_TILEMAP, chicken_slot_snapshots
from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from harvest.core.animal_status import (  # noqa: F401 — re-export for tests
    ADDR_EGG_AVAILABLE,
    ADDR_FED_CHICKENS_FLAGS,
    ADDR_FED_CHICKENS_N,
    ADDR_HAY_COUNT,
    ADDR_INCUBATOR_FLAGS,
    ADDR_ITEM_ON_HAND,
    CHICKEN_SLOT_BASE,
    CHICKEN_SLOT_SIZE,
    INCUBATOR_BIT,
    INCUBATOR_EGG_TILES,
    ITEM_CHICKEN_FEED,
    ITEM_EGG,
    chicken_slot_eggs_available,
    count_chicken_slots,
    egg_available_today,
    is_incubating,
    read_fed_chickens_flags,
    read_hay_count,
    read_item_on_hand,
)
from harvest.core.npc_catalog import game_objects
from harvest.core.task_progress import ProgressSnapshot, task_progress_snapshot
from harvest.core.tile_catalog import ADDR_TILEMAP
from harvest.tasks.animal_navigation import fallback_action, find_path_around_blockers
from harvest.tasks.coop_egg_ops import CoopEggMixin
from harvest.tasks.coop_feed_ops import CoopFeedMixin
from harvest.tasks.coop_layout import (  # noqa: F401 — re-export for tests/skills
    CHICKEN_FEED_SPOTS,
    COOP_ENTRY_STAND,
    COOP_FALSE_OPEN_COLUMN_X,
    COOP_FALSE_OPEN_MIN_Y,
    COOP_LEFT_TOP_APPROACH,
    COOP_MAIN_AISLE_TOP,
    EGG_PICKUP_STAND,
    EXIT_PREP_ESCAPE_ROUTE,
    EXIT_PREP_STAND,
    FEED_BIN_STAND,
    FEED_CLEAR_STAND,
    INCUBATOR_STAND,
    MAX_EGG_DEFERRALS,
    MAX_EGG_NAV_FRAMES,
    MAX_EXIT_PREP_FRAMES,
    MAX_FLOCK_SIZE,
    SHIP_BIN_INTERACT_STAND,
    SHIP_BIN_STAND,
    VISIBLE_EGG_SPRITE,
    ChickenFeedSpot,
)
from harvest.tasks.farm_clearer import TileScanner
from harvest.tasks.nav import MAP_WIDTH, Navigator, Pathfinder, TILE_SIZE, make_action
from harvest.tasks.primitives import press_a_sequence


@dataclass
class CoopChoresTask(CoopFeedMixin, CoopEggMixin, Task):
    """Dynamic coop chores that scale to up to 12 chickens.

    ``egg_mode`` controls what happens after egg pickup:
      - ``"auto"`` — incubate if empty and flock < max, else ship
      - ``"incubate"`` — always try incubator
      - ``"ship"`` — always ship
      - ``"gift"`` — exit coop holding the egg (caller handles delivery)
    """

    name: str = "coop_chores"
    egg_mode: str = "auto"
    max_feed_adults: Optional[int] = None
    timeout: int = 8000

    _scanner: TileScanner = field(default_factory=TileScanner, init=False)
    _pathfinder: Pathfinder = field(init=False)
    _navigator: Navigator = field(init=False)
    _phase: str = field(default="feed_nav", init=False)
    _action_queue: Deque[np.ndarray] = field(default_factory=deque, init=False)
    _step_count: int = field(default=0, init=False)
    _verify_count: int = field(default=0, init=False)
    # Active skill for feed_nav / ship_nav far approach (skills.py factories).
    _active_skill: Optional[Task] = field(default=None, init=False)

    # Counters tracked during the task
    _adult_count: int = field(default=0, init=False)
    _feed_remaining: int = field(default=0, init=False)
    _hay_before: int = field(default=0, init=False)
    _fed_before: int = field(default=0, init=False)
    _fed_flags_before: int = field(default=0, init=False)
    _ship_money_before: int = field(default=0, init=False)
    _egg_attempts: int = field(default=0, init=False)
    _incubator_wp_index: int = field(default=0, init=False)
    _feed_registered: bool = field(default=False, init=False)
    _current_feed_spot: Optional[ChickenFeedSpot] = field(default=None, init=False)
    _blocked_feed_flags: set[int] = field(default_factory=set, init=False)
    _left_top_route_goal: Optional[Tuple[int, int]] = field(default=None, init=False)
    _left_top_route_points: Tuple[Tuple[int, int], ...] = field(default_factory=tuple, init=False)
    _left_top_route_index: int = field(default=0, init=False)
    _exit_prep_started_step: int = field(default=0, init=False)
    _exit_prep_route_index: int = field(default=0, init=False)
    _egg_nav_started_step: int = field(default=0, init=False)
    _current_egg_flag: int = field(default=0, init=False)
    _skipped_egg_flags: set[int] = field(default_factory=set, init=False)
    _deferred_egg_counts: dict[int, int] = field(default_factory=dict, init=False)
    _feed_place_started_step: int = field(default=0, init=False)
    _deferred_feed_counts: dict[int, int] = field(default_factory=dict, init=False)
    fed_count: int = field(default=0, init=False)
    egg_collected: bool = field(default=False, init=False)
    egg_shipped: bool = field(default=False, init=False)
    egg_incubated: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._pathfinder = Pathfinder(self._scanner)
        self._navigator = Navigator(self._pathfinder)

    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._verify_count = 0
        self._action_queue.clear()
        self._active_skill = None
        self._navigator.path = []
        self._navigator.stasis = 0
        self._pathfinder.temp_blocked.clear()
        self._egg_attempts = 0
        self._incubator_wp_index = 0
        self._feed_registered = False
        self._current_feed_spot = None
        self._blocked_feed_flags.clear()
        self._left_top_route_goal = None
        self._left_top_route_points = ()
        self._left_top_route_index = 0
        self._exit_prep_started_step = 0
        self._exit_prep_route_index = 0
        self._egg_nav_started_step = 0
        self._current_egg_flag = 0
        self._skipped_egg_flags.clear()
        self._deferred_egg_counts.clear()
        self._feed_place_started_step = 0
        self._deferred_feed_counts.clear()
        self.egg_collected = False
        self.egg_shipped = False
        self.egg_incubated = False

        adults, _chicks, _eggs = count_chicken_slots(world.ram)
        feed_goal = adults
        if self.max_feed_adults is not None:
            feed_goal = min(feed_goal, max(0, int(self.max_feed_adults)))
        fed_now = min(self._fed_count_now(world.ram), feed_goal)
        self._adult_count = feed_goal
        self.fed_count = fed_now
        self._feed_remaining = max(0, self._adult_count - fed_now)
        self._hay_before = read_hay_count(world.ram)
        self._fed_before = fed_now
        self._fed_flags_before = read_fed_chickens_flags(world.ram)

        if self._feed_remaining > 0 and read_item_on_hand(world.ram) == ITEM_CHICKEN_FEED:
            self._phase = "feed_place_nav"
        elif self._feed_remaining > 0 and self._hay_before > 0:
            self._phase = "feed_nav"
        elif self._egg_present(world.ram):
            self._phase = "egg_nav"
        else:
            self._phase = "exit_prep_nav"

        print(
            f"[COOP] adults={adults} feed_goal={self._adult_count} hay={self._hay_before} "
            f"egg_avail={egg_available_today(world.ram)} slot_egg={chicken_slot_eggs_available(world.ram)}"
        )

    def can_start(self, world: WorldState) -> bool:
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        return tilemap == COOP_TILEMAP

    def resume_after_hotswap(self, world: WorldState) -> None:
        self._action_queue.clear()
        self._active_skill = None
        self._navigator.update(world.ram)
        self._navigator.path = []
        self._navigator.stasis = 0
        self._pathfinder.temp_blocked.clear()
        self._sync_incubator_waypoint()

    @property
    def progress_text(self) -> str:
        return (
            f"fed={self.fed_count}/{self._adult_count} "
            f"egg={'Y' if self.egg_collected else 'N'} "
            f"ship={'Y' if self.egg_shipped else 'N'} "
            f"incub={'Y' if self.egg_incubated else 'N'}"
        )

    def progress_snapshot(self) -> ProgressSnapshot:
        child = task_progress_snapshot(self._active_skill)
        return ProgressSnapshot(
            task_name=self.name,
            phase_text=self._phase,
            step_count=self._step_count,
            details=(
                ("fed", self.fed_count),
                ("adults", self._adult_count),
                ("egg", self.egg_collected),
                ("ship", self.egg_shipped),
                ("incub", self.egg_incubated),
            ),
            child=child,
        )

    def _step_nav_skill(
        self,
        world: WorldState,
        *,
        skill_name: str,
        make_skill,
    ) -> Optional[TaskResult]:
        """Step a host-backed nav skill.

        Returns RUNNING/FAILURE results to bubble up, or None when arrived
        (skill SUCCESS) so the caller can advance the phase.
        """
        if self._active_skill is None or self._active_skill.name != skill_name:
            skill = make_skill()
            skill.reset(world)
            self._active_skill = skill
        result = self._active_skill.step(world)
        if result.status == TaskStatus.SUCCESS:
            self._active_skill = None
            return None
        if result.status == TaskStatus.FAILURE:
            self._active_skill = None
        return result

    # ── Action helpers ───────────────────────────────────────────

    def _queue_press_a(
        self,
        face: str,
        *,
        face_frames: int = 8,
        hold_frames: int = 25,
        settle_frames: int = 18,
        hold_face_with_a: bool = True,
    ) -> None:
        self._action_queue.extend(
            press_a_sequence(
                face,
                face_frames=face_frames,
                pre_press_settle_frames=0,
                hold_frames=hold_frames,
                settle_frames=settle_frames,
                hold_face_with_a=hold_face_with_a,
            )
        )

    def _clear_left_top_route(self) -> None:
        self._left_top_route_goal = None
        self._left_top_route_points = ()
        self._left_top_route_index = 0
        self._navigator.path = []
        self._navigator.stasis = 0

    def _coop_false_open_tiles(self) -> set[Tuple[int, int]]:
        """Tiles the tilemap marks walkable but that trap the player."""
        return {
            (COOP_FALSE_OPEN_COLUMN_X, y)
            for y in range(COOP_FALSE_OPEN_MIN_Y, 14)
        }

    # ── Navigation ───────────────────────────────────────────────

    def _chicken_tiles(self, ram: np.ndarray) -> set[Tuple[int, int]]:
        tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
        if tilemap != COOP_TILEMAP:
            return set()

        tiles: set[Tuple[int, int]] = set()
        tiles.update(self._flagged_egg_tiles(ram))
        incubating = is_incubating(ram)
        saw_positioned_slots = False
        for row in chicken_slot_snapshots(ram, require_coop=True):
            tile = row.get("tile")
            if not (isinstance(tile, list) and len(tile) == 2):
                continue
            saw_positioned_slots = True
            if row.get("stage") not in ("adult", "egg"):
                continue
            tx, ty = int(tile[0]), int(tile[1])
            if incubating and row.get("stage") == "egg" and (tx, ty) in INCUBATOR_EGG_TILES:
                continue
            if 0 <= tx < MAP_WIDTH and 0 <= ty < MAP_WIDTH:
                tiles.add((tx, ty))

        for obj in game_objects(ram):
            if obj.sprite_table_idx == VISIBLE_EGG_SPRITE:
                tx, ty = obj.tile
                if incubating and (tx, ty) in INCUBATOR_EGG_TILES:
                    continue
                if 0 <= tx < MAP_WIDTH and 0 <= ty < MAP_WIDTH:
                    tiles.add((tx, ty))
                continue
            if obj.label != "chicken" or saw_positioned_slots:
                continue
            tx, ty = obj.tile
            if 0 <= tx < MAP_WIDTH and 0 <= ty < MAP_WIDTH:
                tiles.add((tx, ty))
        return tiles

    def _find_path_around_chickens(
        self,
        ram: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[list[Tuple[int, int]]]:
        blocked = self._chicken_tiles(ram)
        blocked.update(self._coop_false_open_tiles())
        return find_path_around_blockers(
            ram,
            self._pathfinder,
            start,
            goal,
            blocked,
        )

    def _strict_center_for_next_step(self) -> Optional[np.ndarray]:
        """Center tightly in the coop's narrow lanes before changing rows."""
        if not self._navigator.path or self._navigator.stasis >= 45:
            return None

        curr_tile = self._navigator.current_tile
        next_tile = self._navigator.path[0]
        center_x = curr_tile[0] * TILE_SIZE + 8
        center_y = curr_tile[1] * TILE_SIZE + 8

        if next_tile[0] == curr_tile[0] and next_tile[1] != curr_tile[1]:
            dx = center_x - self._navigator.current_pos.x
            if abs(dx) > 1:
                return make_action(right=dx > 0, left=dx < 0)
        if next_tile[1] == curr_tile[1] and next_tile[0] != curr_tile[0]:
            dy = center_y - self._navigator.current_pos.y
            if abs(dy) > 1:
                return make_action(down=dy > 0, up=dy < 0)
        return None

    def _left_top_route(self, goal: Tuple[int, int]) -> Tuple[Tuple[int, int], ...]:
        current = self._navigator.current_tile
        route: list[Tuple[int, int]] = []
        # Lower egg stands should not detour through the top aisle: the coop
        # tilemap reports a false-open vertical edge around x=5 that blocks.
        if goal[1] >= 8:
            if current[0] <= 3 and current[1] >= 9 and goal[0] >= 4:
                route.extend(((2, 10), (2, 9), (3, 9)))
            route.append(goal)
            return tuple(dict.fromkeys(route))
        # Far-left upper egg stands (e.g. (0,4) for flag 0x01): climb that
        # column from the ship pocket instead of cutting through the (2,5) wall.
        if (
            current[0] <= 3
            and current[1] >= 9
            and goal[0] <= 1
            and goal[1] <= 5
        ):
            route.append((goal[0], min(current[1], 6)))
            route.append(goal)
            return tuple(dict.fromkeys(route))
        if current[0] <= 4 and current[1] >= 11:
            route.append(COOP_ENTRY_STAND)
            route.append(COOP_MAIN_AISLE_TOP)
        elif current[0] >= 5 and current[1] >= 8:
            route.append(COOP_MAIN_AISLE_TOP)
        if (route or current[0] >= 5) and goal[0] <= 4:
            route.append(COOP_LEFT_TOP_APPROACH)
        route.append(goal)
        return tuple(route)

    def _navigate_to_left_top_goal(self, ram: np.ndarray, goal: Tuple[int, int]) -> Optional[np.ndarray]:
        """Use the recorded center-aisle route for feed/egg targets."""
        if self._navigator.current_tile == goal or self._navigator.at_tile(goal):
            return self._navigator.center_on_tile(goal, tolerance=1)
        if (
            self._left_top_route_goal != goal
            or not self._left_top_route_points
            or self._left_top_route_index >= len(self._left_top_route_points)
        ):
            self._left_top_route_goal = goal
            self._left_top_route_points = self._left_top_route(goal)
            self._left_top_route_index = 0
        route = self._left_top_route_points

        while self._left_top_route_index < len(route):
            waypoint = route[self._left_top_route_index]
            if self._navigator.current_tile == waypoint or self._navigator.at_tile(waypoint):
                self._left_top_route_index += 1
                self._navigator.path = []
                continue
            action = self._navigate_to_tile(ram, waypoint)
            if action is not None:
                return action
            self._left_top_route_index += 1
            self._navigator.path = []
        return None

    def _navigate_to_tile(self, ram: np.ndarray, goal: Tuple[int, int]) -> Optional[np.ndarray]:
        if self._navigator.current_tile == goal or self._navigator.at_tile(goal):
            return self._navigator.center_on_tile(goal, tolerance=1)

        chicken_tiles = self._chicken_tiles(ram)
        chicken_tiles.discard(self._navigator.current_tile)
        if goal in chicken_tiles:
            self._navigator.path = []
            return make_action()

        if self._navigator.path and self._navigator.path[0] in chicken_tiles:
            self._navigator.path = []
            return make_action()

        if self._navigator.stasis > 90 and self._navigator.path:
            self._pathfinder.temp_blocked.add(self._navigator.path[0])
            self._navigator.path = []

        if not self._navigator.path:
            path = self._find_path_around_chickens(ram, self._navigator.current_tile, goal)
            if path is None:
                return fallback_action(self._navigator.current_tile, goal)
            self._navigator.path = path

        action = self._strict_center_for_next_step()
        if action is not None:
            return action

        action = self._navigator.follow_path(ram)
        if action is None:
            return fallback_action(self._navigator.current_tile, goal)
        return action

    # ── Main step ────────────────────────────────────────────────

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        self._navigator.update(world.ram)

        if self._step_count > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason="coop timeout")

        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if tilemap != COOP_TILEMAP:
            return TaskResult(status=TaskStatus.BLOCKED, reason=f"not in coop tilemap=0x{tilemap:02X}")

        action = self._action_queue.popleft() if self._action_queue else None
        if action is not None:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

        if self._phase == "done":
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=f"fed={self.fed_count} egg={self.egg_collected} ship={self.egg_shipped} incub={self.egg_incubated}",
            )

        handler = {
            "feed_nav": self._step_feed_nav,
            "feed_act": self._step_feed_act,
            "feed_verify": self._step_feed_verify,
            "feed_place_nav": self._step_feed_place_nav,
            "feed_place_verify": self._step_feed_place_verify,
            "feed_clear_nav": self._step_feed_clear_nav,
            "feed_clear_verify": self._step_feed_clear_verify,
            "egg_nav": self._step_egg_nav,
            "egg_verify": self._step_egg_verify,
            "decide": self._step_decide,
            "incubate_nav": self._step_incubate_nav,
            "incubate_verify": self._step_incubate_verify,
            "ship_nav": self._step_ship_nav,
            "ship_verify": self._step_ship_verify,
            "exit_prep_nav": self._step_exit_prep_nav,
        }.get(self._phase)
        if handler is not None:
            return handler(world)

        return TaskResult(status=TaskStatus.FAILURE, reason=f"unknown phase {self._phase}")

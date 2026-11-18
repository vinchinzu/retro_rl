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

Phase arms are plain functions, not mixins:
  - ``coop_layout`` — stands, flags, route constants
  - ``coop_feed_ops`` — feed bin / trough phases
  - ``coop_egg_ops`` — egg / incubate / ship / exit-prep phases
  Feed-bin and shipping-bin nav/press call ``skills.py`` factories.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import numpy as np

from harvest.core.animal_probe import COOP_TILEMAP, chicken_slot_snapshots, stray_hen_px
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
from harvest.core.ram_catalog import read_ram_value
from harvest.core.tile_catalog import ADDR_INPUT_LOCK, ADDR_TILEMAP
from harvest.tasks.animal_navigation import fallback_action, find_path_around_blockers
from harvest.tasks.coop_layout import (  # noqa: F401 — re-export for tests/skills
    CHICKEN_COOP_MAP,
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
    flock_drop_alarm,
    flock_must_stay_in,
    SHIP_BIN_INTERACT_STAND,
    SHIP_BIN_STAND,
    VISIBLE_EGG_SPRITE,
    ChickenFeedSpot,
)
from harvest.tasks.farm_clearer import TileScanner
from harvest.tasks.nav import MAP_WIDTH, Navigator, Pathfinder, TILE_SIZE, make_action
from harvest.tasks.primitives import (
    dismiss_dialogue_result,
    drain_action_queue,
    press_a_sequence,
)


@dataclass
class CoopChoresTask(Task):
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
    _hen_drops: int = field(default=0, init=False)
    _exit_prep_route_index: int = field(default=0, init=False)
    _egg_nav_started_step: int = field(default=0, init=False)
    _current_egg_flag: int = field(default=0, init=False)
    _skipped_egg_flags: set[int] = field(default_factory=set, init=False)
    _deferred_egg_counts: dict[int, int] = field(default_factory=dict, init=False)
    _current_egg_tile: Optional[Tuple[int, int]] = field(default=None, init=False)
    _skipped_egg_tiles: set[Tuple[int, int]] = field(default_factory=set, init=False)
    _deferred_egg_tile_counts: dict[Tuple[int, int], int] = field(default_factory=dict, init=False)
    _feed_place_started_step: int = field(default=0, init=False)
    _deferred_feed_counts: dict[int, int] = field(default_factory=dict, init=False)
    fed_count: int = field(default=0, init=False)
    egg_collected: bool = field(default=False, init=False)
    egg_shipped: bool = field(default=False, init=False)
    egg_incubated: bool = field(default=False, init=False)
    weather_hold: bool = field(default=False, init=False)
    flock_alarms: list = field(default_factory=list, init=False)
    _flock_before: int = field(default=0, init=False)

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
        self._hen_drops = 0
        self._exit_prep_route_index = 0
        self._egg_nav_started_step = 0
        self._current_egg_flag = 0
        self._skipped_egg_flags.clear()
        self._deferred_egg_counts.clear()
        self._current_egg_tile = None
        self._skipped_egg_tiles.clear()
        self._deferred_egg_tile_counts.clear()
        self._feed_place_started_step = 0
        self._deferred_feed_counts.clear()
        self.egg_collected = False
        self.egg_shipped = False
        self.egg_incubated = False
        self.flock_alarms = []

        adults, _chicks, _eggs = count_chicken_slots(world.ram)
        self._flock_before = adults + _chicks
        weather = 0
        if 0x0197 < len(world.ram):
            weather = int(world.ram[0x0196]) | (int(world.ram[0x0197]) << 8)
        self.weather_hold = flock_must_stay_in(weather)
        feed_goal = adults
        if self.max_feed_adults is not None:
            feed_goal = min(feed_goal, max(0, int(self.max_feed_adults)))
        from harvest.planner.livestock_econ import hens_to_feed

        feed_goal = min(feed_goal, hens_to_feed(world.ram))
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
            f"weather_hold={int(self.weather_hold)} "
            f"egg_avail={egg_available_today(world.ram)} slot_egg={chicken_slot_eggs_available(world.ram)}"
        )

    def can_start(self, world: WorldState) -> bool:
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        return tilemap == COOP_TILEMAP

    def note_flock_count(self, count: int) -> Optional[str]:
        """Ledger alarm when the counted flock falls. The cap stays 12."""
        alarm = flock_drop_alarm(self._flock_before, count)
        if alarm is not None:
            self.flock_alarms.append(alarm)
        self._flock_before = int(count)
        return alarm

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

    def _enqueue_skill_actions(self, world: WorldState, skill: Task) -> None:
        """Step a skills.py press skill and adopt its frames.

        Phase gates stay on this task. The skill owns the button sequence,
        which then drains through the same action queue as other presses.
        """
        skill.reset(world)
        for _ in range(getattr(skill, "timeout", 180) + 2):
            result = skill.step(world)
            if result.action is not None:
                self._action_queue.append(np.array(result.action.action, dtype=np.int32, copy=True))
            if result.status != TaskStatus.RUNNING:
                return

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
            if waypoint != goal and waypoint in self._chicken_tiles(ram):
                # A hen parked on a via-point never moves: waiting for her idled the
                # walk to the feed bin for its whole timeout, twice (Y2 Spring D22).
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

        # An A pulse at the feed bin or an egg with a hen in front lifts her. Left in
        # the hands she blocks every egg ("held_item_block") and EXIT_COOP carries her
        # onto the farm, where `stray_hen_px` reads her as a bought hen in the shop
        # yard and HEN_DELIVERY burns the next day there (Y2 Spring D28-D29).
        if self._phase != "done" and read_item_on_hand(world.ram) == _ITEM_CHICKEN and self._hen_drops < len(_HEN_DROP_WALK):
            # A drop needs open floor in front (probe: down and right of the left aisle
            # take her; left, up and the spot she was lifted from do not), so each retry
            # first steps toward open ground.
            walk = _HEN_DROP_WALK[self._hen_drops]
            self._hen_drops += 1
            print(f"[COOP] Lifted a hen in phase {self._phase}; putting her down ({self._hen_drops}/{len(_HEN_DROP_WALK)})")
            if walk:
                self._action_queue.extend(make_action(**{walk: True}) for _ in range(24))
            self._action_queue.extend(press_a_sequence(None, face_frames=0, pre_press_settle_frames=4, hold_frames=12, settle_frames=30))
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(self._action_queue.popleft()))

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


from harvest.tasks.coop_egg_ops import bind_task_methods as _bind_coop_egg
from harvest.tasks.coop_feed_ops import bind_task_methods as _bind_coop_feed

_bind_coop_feed(CoopChoresTask)
_bind_coop_egg(CoopChoresTask)

_ITEM_CHICKEN = 0x25
_HEN_DROP_WALK = ("", "down", "right", "down")
_ANIMAL_SHOP_TILEMAP = 0x24
_TOWN_TILEMAP = 0x04
# Fenced yard behind the animal shop (town map 0x04). Free ground is the strip
# above the roof (rows 48-50, cols 32-40) and the column left of the building
# (cols 32-34 down to row 53); the right edge is a one-tile corridor.
_YARD_STRIP = (518, 770, 655, 815)
_YARD_LEFT = (518, 770, 559, 862)
# Everything inside the yard fence (cols 31-41, rows 47-54), the right strip
# beside the building and the fence-hugging x=510 included. The only way out
# is the shop's back door at (36,51): a farmer anywhere in here must leave by
# ``animal_yard_to_farm``; a town route pixel-sticks on the fence (Summer D7,
# pin LN_hen_whatif_yard_strand_FAIL at (649,845)) and RETURN_HOME wedged.
ANIMAL_YARD_BOUNDS = (500, 760, 664, 868)
_YARD_LANE_Y = 792
_COOP_DROP_MAX_Y = 184
# The shop shuts at 17:00 and its back door shuts with it, which seals the yard.
# Do not walk in when the pickup cannot finish, and walk out before the bell.
YARD_ENTRY_DEADLINE_MIN = 16 * 60 + 20
YARD_LEAVE_DEADLINE_MIN = 16 * 60 + 45


def _in_rect(rect: tuple[int, int, int, int], x: int, y: int) -> bool:
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def yard_free(x: int, y: int) -> bool:
    return _in_rect(_YARD_STRIP, x, y) or _in_rect(_YARD_LEFT, x, y)


def in_animal_yard(tilemap: int, x: int, y: int) -> bool:
    """True inside the fenced yard behind the animal shop (town 0x04/0x05)."""
    return tilemap in (_TOWN_TILEMAP, _TOWN_TILEMAP + 1) and _in_rect(ANIMAL_YARD_BOUNDS, x, y)


def hen_stand(bird: tuple[int, int], player: tuple[int, int]) -> tuple[tuple[int, int], str]:
    """Free yard pixel beside the hen nearest the player, and the face toward her.

    A hen against the fence has no free side on that edge; the stand then
    clamps onto the nearest free ground and the press faces her from there.
    """
    bx, by = bird
    options = [
        ((bx + 16, by), "left"),
        ((bx - 16, by), "right"),
        ((bx, by + 16), "up"),
        ((bx, by - 16), "down"),
    ]
    free = [(pt, face) for pt, face in options if yard_free(*pt)]
    return min(free or options, key=lambda o: abs(o[0][0] - player[0]) + abs(o[0][1] - player[1]))


def yard_step(player: tuple[int, int], stand: tuple[int, int], tol: int = 3) -> str | None:
    """Next held direction toward a yard stand, or None on arrival.

    Every leg runs along the lane above the roof (y=792): up to it, across to
    the stand column, then down into the left strip or the right corridor.
    """
    px, py = player
    sx, sy = stand
    if abs(px - sx) <= tol and abs(py - sy) <= tol:
        return None
    if abs(px - sx) > tol:
        if abs(py - _YARD_LANE_Y) > tol and not (_in_rect(_YARD_STRIP, px, py) and _in_rect(_YARD_STRIP, sx, sy)):
            return "up" if py > _YARD_LANE_Y else "down"
        return "right" if sx > px else "left"
    return "down" if sy > py else "up"


@dataclass
class HenDeliveryTask(Task):
    """From the animal-shop counter, carry the new hen into the coop and drop her.

    The shop does not hand the hen over. She spawns in the fenced yard behind
    the shop (about (568, 792) on town map 0x04). The yard has no gate; the
    way in is the shop's back door, and the shop's closing time seals it. Pick
    her up beside the sprite, walk back through the shop and the town and farm
    gates and press A one tile inside the coop door. Only a hen set down in the
    coop (map byte 0x28) lays. Ends inside the coop (the planner then feeds her
    and leaves). Too late to finish before 17:00, the task walks home
    empty-handed, ``deferred`` is set, and the hen waits in the yard; it ends
    on the farm.
    """

    name: str = "hen_delivery"
    pickup_tries_max: int = 6
    drop_tries_max: int = 8
    chase_budget: int = 1200

    deferred: bool = field(default=False, init=False)
    _phase: str = field(default="boot", init=False)
    _child: Optional[Task] = field(default=None, init=False)
    _step_count: int = field(default=0, init=False)
    _chase_frames: int = field(default=0, init=False)
    _pickup_tries: int = field(default=0, init=False)
    _drop_tries: int = field(default=0, init=False)
    _stand: Optional[tuple[int, int]] = field(default=None, init=False)
    _face: str = field(default="left", init=False)
    _action_queue: Deque[np.ndarray] = field(default_factory=deque, init=False)
    _to_farm_retried: bool = field(default=False, init=False)

    def reset(self, world: WorldState) -> None:
        self._phase = "boot"
        self._to_farm_retried = False
        self._child = None
        self.deferred = False
        self._step_count = 0
        self._chase_frames = 0
        self._pickup_tries = 0
        self._drop_tries = 0
        self._stand = None
        self._face = "left"
        self._action_queue.clear()

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        if self._phase == "done":
            return TaskResult(status=TaskStatus.SUCCESS, reason="deferred" if self.deferred else "hen in coop")
        if self._phase in ("chase", "pickup_press", "drop"):
            return self._hands(world)
        if self._phase == "boot":
            return self._boot(world)
        if self._child is None:
            return self._fail(f"no child in {self._phase}")
        result = self._child.step(world)
        if result.status == TaskStatus.RUNNING:
            return result
        if result.status != TaskStatus.SUCCESS:
            if self._phase == "to_farm" and not self._to_farm_retried:
                # A walk out of the shop door can step back in (Summer D6: the
                # route was past its shop hops). Re-arm once from where we are.
                self._to_farm_retried = True
                self._log(world, f"to_farm retry after: {result.reason}")
                return self._carry_on(world, self._tilemap(world.ram))
            return self._fail(f"{self._phase} failed: {result.reason}")
        return self._advance(world)

    def _boot(self, world: WorldState) -> TaskResult:
        ram = world.ram
        tilemap = self._tilemap(ram)
        if read_item_on_hand(ram) == _ITEM_CHICKEN:
            return self._carry_on(world, tilemap)
        if self._stray_bird(ram) is None:
            if tilemap in (_ANIMAL_SHOP_TILEMAP, _TOWN_TILEMAP):
                self.deferred = True  # nothing to carry (the buy failed): just walk home
                return self._carry_on(world, tilemap)
            return self._finish("no hen outside the coop")
        if self._is_farm(tilemap):
            return self._begin_chase(world)
        if tilemap == _TOWN_TILEMAP and yard_free(*self._player_px(ram)):
            return self._begin_chase(world)
        if tilemap != _ANIMAL_SHOP_TILEMAP:
            return self._fail(f"hen delivery starts in the shop, not tilemap 0x{tilemap:02X}")
        if self._clock_min(ram) > YARD_ENTRY_DEADLINE_MIN:
            return self._defer(world, "too late for the yard")
        return self._arm("to_yard", self._nav("animal_shop_to_yard", clear=False, timeout=4000), world)

    def _defer(self, world: WorldState, why: str) -> TaskResult:
        self.deferred = True
        self._log(world, f"defer: {why}")
        return self._carry_on(world, self._tilemap(world.ram))

    def _carry_on(self, world: WorldState, tilemap: int) -> TaskResult:
        """Head for the farm (then the coop, if the hen is in hand) from any map on the way."""
        if tilemap == CHICKEN_COOP_MAP:
            if read_item_on_hand(world.ram) == _ITEM_CHICKEN:
                return self._begin_drop(world)
            return self._finish("hen in coop")
        if tilemap == _ANIMAL_SHOP_TILEMAP:
            route = ("animal_shop_to_town", "animal_shop_town_to_farm")
        elif tilemap in (_TOWN_TILEMAP, _TOWN_TILEMAP + 1):
            in_yard = in_animal_yard(tilemap, *self._player_px(world.ram))
            route = ("animal_yard_to_farm",) if in_yard else ("animal_shop_town_to_farm",)
        elif self._is_farm(tilemap):
            return self._after_farm(world)
        else:
            return self._fail(f"carrying a hen on tilemap 0x{tilemap:02X}")
        from harvest.maps.map_config import ROUTES

        waypoints = [wp for name in route for wp in ROUTES[name]]
        return self._arm("to_farm", self._nav_wps("hen_home", waypoints, clear=False, timeout=20000), world)

    def _after_farm(self, world: WorldState) -> TaskResult:
        if self.deferred or read_item_on_hand(world.ram) != _ITEM_CHICKEN:
            return self._finish("deferred" if self.deferred else "back on the farm")
        return self._arm("to_coop_door", self._nav("farm_to_coop", clear=False, timeout=8000), world)

    def _advance(self, world: WorldState) -> TaskResult:
        phase = self._phase
        if phase == "to_yard":
            self._phase, self._child, self._stand = "chase", None, None
            # The map fade-in swallows presses for about half a second.
            self._action_queue.clear()
            self._action_queue.extend(make_action() for _ in range(30))
            return self._hands(world)
        if phase == "to_farm":
            return self._after_farm(world)
        if phase == "to_coop_door":
            return self._arm("enter_coop", self._enter_coop(), world)
        if phase == "enter_coop":
            if read_item_on_hand(world.ram) != _ITEM_CHICKEN:
                return self._fail("coop door cleared the hen")
            return self._begin_drop(world)
        return self._fail(f"no advance from {phase}")

    def _begin_chase(self, world: WorldState) -> TaskResult:
        self._phase = "chase"
        self._child = None
        self._stand = None
        self._action_queue.clear()
        self._log(world, "chase")
        return self._hands(world)

    def _begin_press(self, world: WorldState) -> TaskResult:
        self._phase = "pickup_press"
        self._pickup_tries += 1
        self._action_queue.clear()
        self._action_queue.extend(
            press_a_sequence(
                self._face,
                face_frames=4,
                pre_press_settle_frames=2,
                hold_frames=16,
                settle_frames=24,
                hold_face_with_a=True,
            )
        )
        self._log(world, f"pickup try={self._pickup_tries}")
        return self._hands(world)

    def _begin_drop(self, world: WorldState) -> TaskResult:
        self._phase = "drop"
        self._child = None
        self._drop_tries = 0
        self._action_queue.clear()
        self._log(world, "drop")
        return self._hands(world)

    def _hands(self, world: WorldState) -> TaskResult:
        ram = world.ram
        held = read_item_on_hand(ram)
        tilemap = self._tilemap(ram)
        if self._phase in ("chase", "pickup_press"):
            if held == _ITEM_CHICKEN:
                return self._carry_on(world, tilemap)
            if tilemap != _TOWN_TILEMAP and not self._is_farm(tilemap):
                return self._fail(f"left the yard map=0x{tilemap:02X} held=0x{held:02X}")
        if self._phase == "drop":
            if held != _ITEM_CHICKEN and self._stray_bird(ram) is None:
                return self._finish("hen in coop")
            if tilemap != CHICKEN_COOP_MAP:
                return self._fail(f"left coop during drop map=0x{tilemap:02X} held=0x{held:02X}")
        lock = int(ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(ram) else 1
        if lock != 1:
            return dismiss_dialogue_result(self._step_count, pulse_every=1)
        queued = drain_action_queue(self._action_queue)
        if queued is not None:
            return queued
        if self._phase == "drop":
            return self._drop_press(world, held)
        self._chase_frames += 1
        if ((not self._is_farm(tilemap) and self._clock_min(ram) > YARD_LEAVE_DEADLINE_MIN)
                or self._chase_frames > (8000 if self._is_farm(tilemap) else self.chase_budget)):
            return self._defer(world, "pickup ran past the closing deadline")
        if self._phase == "pickup_press":
            if self._pickup_tries >= self.pickup_tries_max:
                return self._defer(world, f"pickup missed slots={self._slots(ram)}")
            self._phase = "chase"
        return self._chase(world)

    def _chase(self, world: WorldState) -> TaskResult:
        """Greedy walk to the stand beside the hen, then press A at her."""
        ram = world.ram
        bird = self._stray_bird(ram)
        if bird is None:
            return self._defer(world, "hen vanished from the yard")
        player = self._player_px(ram)
        if self._is_farm(self._tilemap(ram)):
            from harvest.tasks.skills import NavSkill
            dx, dy = bird[0] - player[0], bird[1] - player[1]
            if abs(dx) + abs(dy) > 28:
                if self._child is None or self._stand is None or self._step_count % 60 == 0:
                    self._stand = bird
                    self._child = NavSkill(target_px=bird, radius=20, timeout=8000)
                    self._child.reset(world)
                result = self._child.step(world)
                if result.status == TaskStatus.SUCCESS:
                    self._child = None
                return result if result.status == TaskStatus.RUNNING else TaskResult(
                    TaskStatus.RUNNING, action=ActionResult(make_action()))
            self._face = ("right" if dx > 0 else "left") if abs(dx) > abs(dy) else ("down" if dy > 0 else "up")
            self._child = None
            return self._begin_press(world)
        if self._stand is None or self._step_count % 30 == 0:
            self._stand, self._face = hen_stand(bird, player)
        direction = yard_step(player, self._stand)
        if direction is None:
            self._stand = None
            return self._begin_press(world)
        far = abs(player[0] - self._stand[0]) + abs(player[1] - self._stand[1]) > 24
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(**{direction: True, "b": far})))

    def _drop_press(self, world: WorldState, held: int) -> TaskResult:
        # The door mat (tile 8,12) refuses the drop. One tile up takes it.
        if self._player_px(world.ram)[1] > _COOP_DROP_MAX_Y:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(up=True)))
        if self._drop_tries >= self.drop_tries_max:
            return self._fail(f"drop missed held=0x{held:02X} slots={self._slots(world.ram)}")
        self._drop_tries += 1
        self._action_queue.extend(
            press_a_sequence(None, face_frames=0, pre_press_settle_frames=4, hold_frames=12, settle_frames=30)
        )
        queued = drain_action_queue(self._action_queue)
        return queued or TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

    def _arm(self, phase: str, child: Task, world: WorldState) -> TaskResult:
        self._phase = phase
        self._child = child
        child.reset(world)
        self._log(world, "arm")
        result = child.step(world)
        if result.status == TaskStatus.RUNNING:
            return result
        if result.status != TaskStatus.SUCCESS:
            return self._fail(f"{phase} failed: {result.reason}")
        return self._advance(world)

    def _nav(self, route: str, *, clear: bool, timeout: int) -> Task:
        from harvest.maps.map_config import ROUTES

        return self._nav_wps(route, list(ROUTES[route]), clear=clear, timeout=timeout)

    def _nav_wps(self, name: str, waypoints: list, *, clear: bool, timeout: int) -> Task:
        from harvest.planner.tasks.multi_nav import MultiMapNavTask

        return MultiMapNavTask(
            name=name,
            waypoints=list(waypoints),
            timeout=timeout,
            allow_opportunistic_clear=clear,
            initial_settle_frames=20,
        )

    def _enter_coop(self) -> Task:
        from harvest.planner.tasks.transitions import DirectionalTransitionTask

        return DirectionalTransitionTask(
            name="hen_enter_coop",
            direction="up",
            origin_tilemap=0x00,
            target_tilemap=CHICKEN_COOP_MAP,
            timeout=1200,
            stand_tile=(28, 22),
            stand_tolerance=0,
            target_stand_tile=(8, 12),
            target_stand_tolerance=1,
            settle_frames=60,
            door_align_px=28 * 16 + 8,
            overshoot_limit_px=330,
            require_empty_hands=False,
        )

    def _finish(self, reason: str) -> TaskResult:
        self._phase = "done"
        self._child = None
        print(f"[HEN] DONE {reason}", flush=True)
        return TaskResult(status=TaskStatus.SUCCESS, reason=reason)

    def _fail(self, reason: str) -> TaskResult:
        self._phase = "failed"
        self._child = None
        print(f"[HEN] FAIL {reason}", flush=True)
        return TaskResult(status=TaskStatus.FAILURE, reason=reason)

    def _log(self, world: WorldState, note: str) -> None:
        ram = world.ram
        print(
            f"[HEN] f={getattr(world, "frame", 0)} {note} phase={self._phase} map=0x{self._tilemap(ram):02X} "
            f"clock={self._clock_min(ram) // 60}:{self._clock_min(ram) % 60:02d} "
            f"bird={self._stray_bird(ram)} player={self._player_px(ram)}",
            flush=True,
        )

    def _clock_min(self, ram: np.ndarray) -> int:
        return int(read_ram_value(ram, "hour")) * 60 + int(read_ram_value(ram, "minute"))

    def _tilemap(self, ram: np.ndarray) -> int:
        return int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0

    def _is_farm(self, tilemap: int) -> bool:
        from harvest.planner.day_plan_status import is_farm_tilemap

        return bool(is_farm_tilemap(tilemap))

    def _player_px(self, ram: np.ndarray) -> tuple[int, int]:
        from harvest.tasks.nav import get_pos_from_ram

        pos = get_pos_from_ram(ram)
        return int(pos.x), int(pos.y)

    def _stray_bird(self, ram: np.ndarray) -> tuple[int, int] | None:
        return stray_hen_px(ram)

    def _slots(self, ram: np.ndarray) -> str:
        return ",".join(
            f"{row.get('slot')}:{row.get('status_hex')}@{row.get('map_hex')}:{row.get('pixel')}"
            for row in chicken_slot_snapshots(ram, require_coop=False)
        )

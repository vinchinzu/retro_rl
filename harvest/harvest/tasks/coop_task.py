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
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import numpy as np

from harvest.core.animal_probe import COOP_TILEMAP, chicken_slot_snapshots
from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from harvest.core.animal_status import (
    ADDR_EGG_AVAILABLE,
    ADDR_HAY_COUNT,
    ADDR_INCUBATOR_FLAGS,
    ADDR_ITEM_ON_HAND,
    ADDR_FED_CHICKENS_FLAGS,
    ADDR_FED_CHICKENS_N,
    CHICKEN_SLOT_BASE,
    CHICKEN_SLOT_SIZE,
    INCUBATOR_EGG_TILES,
    INCUBATOR_BIT,
    ITEM_CHICKEN_FEED,
    ITEM_EGG,
    chicken_slot_eggs_available,
    count_chicken_slots,
    egg_available_today,
    is_holding_egg,
    is_incubating,
    read_egg_available_flags,
    read_fed_chickens_flags,
    read_fed_chickens_n,
    read_hay_count,
    read_item_on_hand,
)
from harvest.maps.map_config import find_landmark
from harvest.core.npc_catalog import game_objects
from harvest.tasks.farm_clearer import (
    ADDR_TILEMAP,
    MAP_WIDTH,
    Navigator,
    Pathfinder,
    TileScanner,
    make_action,
    get_pos_from_ram,
    get_tile_at,
    TILE_SIZE,
)
from harvest.tasks.harvest_task import read_shipping_money
from harvest.tasks.animal_navigation import align_to_pixel, fallback_action, find_path_around_blockers
from harvest.tasks.primitives import press_a_sequence

# ── Coop interior layout (tilemap 0x28) ─────────────────────────
# Positions discovered via coop_chores / coop_sell_egg recording traces.

FEED_BIN_STAND: Tuple[int, int] = (2, 6)
FEED_BIN_FACE: str = "left"
FEED_CLEAR_STAND: Tuple[int, int] = (2, 3)
VISIBLE_EGG_SPRITE = 0x00F3


@dataclass(frozen=True)
class ChickenFeedSpot:
    stand: Tuple[int, int]
    face: str
    interact_px: Tuple[int, int]
    flag: int


CHICKEN_FEED_FLAGS: Tuple[int, ...] = (
    0x0001,
    0x0002,
    0x0004,
    0x0008,
    0x0010,
    0x0020,
    0x0040,
    0x0080,
    0x0100,
    0x0200,
    0x0400,
    0x0800,
)

# Coop feed trough tiles are the top row tile properties 0xE2..0xED.
# The reliable interaction point, from recordings, is near the lower-left
# of the stand tile while holding Up+A into the trough.
CHICKEN_FEED_SPOTS: Tuple[ChickenFeedSpot, ...] = tuple(
    ChickenFeedSpot(
        stand=(x, 3),
        face="up",
        interact_px=(x * TILE_SIZE + (6 if x == 2 else 10), 3 * TILE_SIZE + 14),
        flag=flag,
    )
    for x, flag in zip(range(2, 14), CHICKEN_FEED_FLAGS)
)

EGG_PICKUP_STAND: Tuple[int, int] = (2, 4)
EGG_PICKUP_FACE: str = "left"
CHICKEN_EGG_SPAWN_PIXELS: Tuple[Tuple[int, int], ...] = (
    (0x18, 0x48),
    (0x38, 0x58),
    (0x48, 0x98),
    (0x58, 0x78),
    (0x68, 0xA8),
    (0x78, 0x88),
    (0x88, 0x58),
    (0x98, 0x98),
    (0xA8, 0x78),
    (0xB8, 0xA8),
    (0xC8, 0x68),
    (0xD8, 0x88),
    (0x28, 0xA8),
)
CHICKEN_EGG_FLAGS: Tuple[int, ...] = tuple(1 << slot for slot in range(len(CHICKEN_EGG_SPAWN_PIXELS)))


def _egg_recording_stand(px: int, py: int) -> Tuple[Tuple[int, int], str]:
    """Stand tile/face for a spawn pixel, avoiding the false-open x=5 column."""
    tx, ty = px // TILE_SIZE, py // TILE_SIZE
    stand = (tx + 1, ty)
    if stand[0] == 5:
        return (tx, ty + 1), "up"
    return stand, "left"


EGG_PICKUP_SPOTS: Tuple[Tuple[int, Tuple[int, int], str], ...] = (
    (0x02, (4, 5), "left"),
    (0x01, (2, 4), "left"),
    *(
        (flag, *_egg_recording_stand(px, py))
        for flag, (px, py) in zip(CHICKEN_EGG_FLAGS[2:], CHICKEN_EGG_SPAWN_PIXELS[2:])
    ),
)

COOP_ENTRY_STAND: Tuple[int, int] = (8, 12)
COOP_MAIN_AISLE_TOP: Tuple[int, int] = (8, 6)
COOP_LEFT_TOP_APPROACH: Tuple[int, int] = (4, 5)

INCUBATOR_STAND: Tuple[int, int] = (13, 11)
INCUBATOR_FACE: str = "right"
INCUBATOR_APPROACH: Tuple[Tuple[int, int], ...] = ((8, 10), (10, 11), INCUBATOR_STAND)

_EGG_SHIPPING_BIN_LANDMARK = find_landmark("egg_shipping_bin", tilemap_id=0x28)

# Bottom-left coop shipping bin is interacted with from the aisle tile just
# above the bin frontage, facing down into the bin.
SHIP_BIN_STAND: Tuple[int, int] = (2, 10)
SHIP_BIN_INTERACT_STAND: Tuple[int, int] = (
    _EGG_SHIPPING_BIN_LANDMARK[1].tile
    if _EGG_SHIPPING_BIN_LANDMARK is not None
    else (1, 10)
)
SHIP_BIN_FACE: str = (
    _EGG_SHIPPING_BIN_LANDMARK[1].face
    if _EGG_SHIPPING_BIN_LANDMARK is not None and _EGG_SHIPPING_BIN_LANDMARK[1].face
    else "down"
)
SHIP_LANE_X = 38
SHIP_APPROACH_Y = 165
SHIP_INTERACT_PX = (22, 169)
SHIP_RIGHT_LANE_CORNER: Tuple[int, int] = (3, 10)

# Stage at the same door tile EXIT_COOP expects. Handing off at the left
# service lane (3,11) left EXIT_COOP to cross the false-open x=5 edge and time
# out (seen at pos=(57,184) tile=(3,11)).
EXIT_PREP_STAND: Tuple[int, int] = COOP_ENTRY_STAND
# The coop tilemap reports a walkable vertical strip around x=5 that is not
# actually passable. Long-run stalls consistently pin at (5, 11) / (86, 183).
COOP_FALSE_OPEN_COLUMN_X = 5
COOP_FALSE_OPEN_MIN_Y = 8
EXIT_PREP_ESCAPE_ROUTE: Tuple[Tuple[int, int], ...] = (
    (5, 12),
    (2, 12),
    EXIT_PREP_STAND,
)
EXIT_PREP_LEFT_ROUTE: Tuple[Tuple[int, int], ...] = (
    (2, 12),
    EXIT_PREP_STAND,
)
MAX_EXIT_PREP_FRAMES = 360
MAX_EGG_NAV_FRAMES = 480
MAX_EGG_ATTEMPTS = 4
MAX_EGG_DEFERRALS = 1
MAX_FEED_PLACE_FRAMES = 480
MAX_FEED_SLOT_DEFERRALS = 1
MAX_FLOCK_SIZE = 12


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

    def _queue_place_feed(self, face: str) -> None:
        self._action_queue.extend(make_action(**{face: True}) for _ in range(4))
        self._action_queue.extend(make_action(**{face: True, "a": True}) for _ in range(8))
        self._action_queue.extend(make_action(a=True) for _ in range(4))
        self._action_queue.extend(make_action(down=True) for _ in range(12))
        self._action_queue.extend(make_action() for _ in range(8))

    def _fed_count_now(self, ram: np.ndarray) -> int:
        flags = read_fed_chickens_flags(ram)
        flag_count = sum(1 for spot in CHICKEN_FEED_SPOTS if flags & spot.flag)
        return max(read_fed_chickens_n(ram), flag_count)

    def _next_feed_spot(self, ram: np.ndarray) -> Optional[ChickenFeedSpot]:
        flags = read_fed_chickens_flags(ram)
        blocked = self._chicken_tiles(ram)
        blocked.discard(self._navigator.current_tile)

        for spot in CHICKEN_FEED_SPOTS:
            if flags & spot.flag:
                continue
            if spot.flag in self._blocked_feed_flags:
                continue
            if spot.stand in blocked:
                continue
            return spot

        for spot in CHICKEN_FEED_SPOTS:
            if not (flags & spot.flag) and spot.flag not in self._blocked_feed_flags:
                return spot
        return None

    def _clear_left_top_route(self) -> None:
        self._left_top_route_goal = None
        self._left_top_route_points = ()
        self._left_top_route_index = 0
        self._navigator.path = []
        self._navigator.stasis = 0

    def _begin_egg_nav(self) -> TaskResult:
        self._egg_attempts = 0
        self._verify_count = 0
        self._egg_nav_started_step = self._step_count
        self._current_egg_flag = 0
        self._clear_left_top_route()
        self._pathfinder.temp_blocked.clear()
        self._phase = "egg_nav"
        return TaskResult(status=TaskStatus.RUNNING)

    def _advance_after_feed(self, ram: np.ndarray) -> TaskResult:
        self._feed_registered = False
        fed_now = min(self._fed_count_now(ram), self._adult_count)
        self.fed_count = max(self.fed_count, fed_now)
        self._feed_remaining = max(0, self._adult_count - fed_now)
        self._current_feed_spot = None
        self._feed_place_started_step = 0
        if self._feed_remaining > 0:
            self._clear_left_top_route()
            if read_item_on_hand(ram) == ITEM_CHICKEN_FEED:
                self._phase = "feed_place_nav"
            else:
                self._phase = "feed_nav"
        elif self._collectable_egg_present(ram):
            return self._begin_egg_nav()
        else:
            return self._begin_exit_prep()
        return TaskResult(status=TaskStatus.RUNNING)

    def _after_egg_handled(self, ram: np.ndarray) -> TaskResult:
        self._egg_attempts = 0
        self._verify_count = 0
        self._egg_nav_started_step = 0
        self._current_egg_flag = 0
        if self._collectable_egg_present(ram):
            return self._begin_egg_nav()
        return self._begin_exit_prep()

    def _begin_exit_prep(self) -> TaskResult:
        self._verify_count = 0
        self._exit_prep_started_step = self._step_count
        self._exit_prep_route_index = 0
        self._clear_left_top_route()
        self._pathfinder.temp_blocked.clear()
        self._phase = "exit_prep_nav"
        return TaskResult(status=TaskStatus.RUNNING)

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

    def _flagged_egg_tiles(self, ram: np.ndarray) -> set[Tuple[int, int]]:
        flags = read_egg_available_flags(ram)
        tiles: set[Tuple[int, int]] = set()
        for flag, (px, py) in zip(CHICKEN_EGG_FLAGS, CHICKEN_EGG_SPAWN_PIXELS):
            if not (flags & flag):
                continue
            tile = (px // TILE_SIZE, py // TILE_SIZE)
            if is_incubating(ram) and tile in INCUBATOR_EGG_TILES:
                continue
            tiles.add(tile)
        return tiles

    def _egg_present(self, ram: np.ndarray) -> bool:
        if egg_available_today(ram) or chicken_slot_eggs_available(ram):
            return True
        incubating = is_incubating(ram)
        for obj in game_objects(ram):
            if obj.sprite_table_idx != VISIBLE_EGG_SPRITE:
                continue
            if incubating and obj.tile in INCUBATOR_EGG_TILES:
                continue
            return True
        return False

    def _collectable_egg_present(self, ram: np.ndarray) -> bool:
        """Like `_egg_present`, but ignores egg flags already skipped this run."""
        flags = read_egg_available_flags(ram)
        for mask, _stand, _face in EGG_PICKUP_SPOTS:
            if (flags & mask) and mask not in self._skipped_egg_flags:
                return True
        if chicken_slot_eggs_available(ram) or self._egg_tiles(ram):
            return True
        incubating = is_incubating(ram)
        for obj in game_objects(ram):
            if obj.sprite_table_idx != VISIBLE_EGG_SPRITE:
                continue
            if incubating and obj.tile in INCUBATOR_EGG_TILES:
                continue
            return True
        return False

    def _egg_tiles(self, ram: np.ndarray) -> list[Tuple[int, int]]:
        tiles: list[Tuple[int, int]] = []
        seen: set[Tuple[int, int]] = set()
        for row in chicken_slot_snapshots(ram, require_coop=True):
            if row.get("stage") != "egg":
                continue
            tile = row.get("tile")
            if not (isinstance(tile, list) and len(tile) == 2):
                continue
            egg_tile = (int(tile[0]), int(tile[1]))
            if is_incubating(ram) and egg_tile in INCUBATOR_EGG_TILES:
                continue
            if egg_tile not in seen:
                seen.add(egg_tile)
                tiles.append(egg_tile)
        for obj in game_objects(ram):
            if obj.sprite_table_idx != VISIBLE_EGG_SPRITE:
                continue
            egg_tile = obj.tile
            if is_incubating(ram) and egg_tile in INCUBATOR_EGG_TILES:
                continue
            if egg_tile not in seen:
                seen.add(egg_tile)
                tiles.append(egg_tile)
        return tiles

    def _egg_tile_for_flag(self, flag: int) -> Optional[Tuple[int, int]]:
        for egg_flag, (px, py) in zip(CHICKEN_EGG_FLAGS, CHICKEN_EGG_SPAWN_PIXELS):
            if egg_flag == flag:
                return px // TILE_SIZE, py // TILE_SIZE
        return None

    def _stand_candidates_for_egg(
        self,
        egg_tile: Tuple[int, int],
        *,
        preferred: Optional[Tuple[Tuple[int, int], str]] = None,
    ) -> list[Tuple[Tuple[int, int], str]]:
        x, y = egg_tile
        # Prefer body-side / below stands before same-column traps above the egg.
        geometric = (
            ((x - 1, y), "right"),
            ((x + 1, y), "left"),
            ((x, y + 1), "up"),
            ((x, y - 1), "down"),
        )
        candidates: list[Tuple[Tuple[int, int], str]] = []
        seen: set[Tuple[int, int]] = set()
        if preferred is not None:
            candidates.append(preferred)
            seen.add(preferred[0])
        for stand, face in geometric:
            if stand in seen:
                continue
            seen.add(stand)
            candidates.append((stand, face))
        return candidates

    def _stand_for_egg_tile(
        self,
        ram: np.ndarray,
        egg_tile: Tuple[int, int],
        *,
        preferred: Optional[Tuple[Tuple[int, int], str]] = None,
        require_path: bool = True,
    ) -> Optional[Tuple[Tuple[int, int], str]]:
        blocked = self._chicken_tiles(ram)
        blocked.discard(self._navigator.current_tile)
        current = self._navigator.current_tile
        scored: list[Tuple[Tuple[int, int], Tuple[Tuple[int, int], str]]] = []
        loose: list[Tuple[Tuple[int, int], Tuple[Tuple[int, int], str]]] = []
        for index, (stand, face) in enumerate(
            self._stand_candidates_for_egg(egg_tile, preferred=preferred)
        ):
            sx, sy = stand
            if not (0 <= sx < MAP_WIDTH and 0 <= sy < MAP_WIDTH):
                continue
            if stand in blocked:
                continue
            if not self._pathfinder.is_walkable(
                ram, sx, sy, current_pos=current
            ):
                continue
            distance = abs(sx - current[0]) + abs(sy - current[1])
            if current == stand or self._navigator.at_tile(stand):
                return stand, face
            path = self._find_path_around_chickens(ram, current, stand)
            # Prefer the recording stand when reachable; otherwise shortest path.
            preferred_penalty = (
                0
                if preferred is not None and stand == preferred[0]
                else 1
            )
            if path is not None:
                scored.append(
                    ((preferred_penalty, len(path), index, distance), (stand, face))
                )
            else:
                loose.append(
                    ((preferred_penalty, index, distance), (stand, face))
                )
        if scored:
            scored.sort(key=lambda row: row[0])
            return scored[0][1]
        if not require_path and loose:
            loose.sort(key=lambda row: row[0])
            return loose[0][1]
        return None

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

    def _sync_incubator_waypoint(self) -> None:
        """Pick a safe incubator waypoint after human/bot handoff."""
        current = self._navigator.current_tile
        if self._navigator.at_tile(INCUBATOR_STAND):
            self._incubator_wp_index = len(INCUBATOR_APPROACH) - 1
        elif current[1] < 10 or (current[0] >= 12 and current[1] < 11) or current[0] > INCUBATOR_STAND[0]:
            self._incubator_wp_index = 0
        elif current[0] < 10:
            self._incubator_wp_index = 1
        elif current[1] < 11:
            self._incubator_wp_index = 1
        else:
            self._incubator_wp_index = 2

    def _navigate_to_incubator_stand(self, ram: np.ndarray) -> Optional[np.ndarray]:
        """Approach the incubator from the left, matching the working recording."""
        self._sync_incubator_waypoint()
        while self._incubator_wp_index < len(INCUBATOR_APPROACH):
            goal = INCUBATOR_APPROACH[self._incubator_wp_index]
            action = self._navigate_to_tile(ram, goal)
            if action is not None:
                return action
            if goal == INCUBATOR_STAND:
                return None
            self._incubator_wp_index += 1
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

    def _egg_pickup_spot(
        self, ram: np.ndarray, *, require_path: bool = True
    ) -> Optional[Tuple[Tuple[int, int], str]]:
        """Pick a reachable stand for the next floor egg.

        Hardcoded recording stands can be islands once the egg tile itself is
        treated as collision (Spring 22 flag 0x01 → stand (2,4) with walls at
        (2,3)/(2,5)/(3,4)). Prefer geometric side stands with a real path.
        """
        available = read_egg_available_flags(ram)
        preferred_by_flag = {
            mask: (stand, face) for mask, stand, face in EGG_PICKUP_SPOTS
        }
        for mask, _stand, _face in EGG_PICKUP_SPOTS:
            if not (available & mask) or mask in self._skipped_egg_flags:
                continue
            egg_tile = self._egg_tile_for_flag(mask)
            if egg_tile is None:
                continue
            spot = self._stand_for_egg_tile(
                ram,
                egg_tile,
                preferred=preferred_by_flag.get(mask),
                require_path=require_path,
            )
            if spot is not None:
                self._current_egg_flag = mask
                return spot
        for egg_tile in self._egg_tiles(ram):
            dynamic_spot = self._stand_for_egg_tile(
                ram, egg_tile, require_path=require_path
            )
            if dynamic_spot is not None:
                self._current_egg_flag = 0
                return dynamic_spot
        if require_path:
            return self._egg_pickup_spot(ram, require_path=False)
        self._current_egg_flag = 0
        return None

    def _defer_or_skip_egg(self, reason: str) -> TaskResult:
        flag = self._current_egg_flag
        if flag:
            deferred = self._deferred_egg_counts.get(flag, 0)
            if deferred < MAX_EGG_DEFERRALS:
                self._deferred_egg_counts[flag] = deferred + 1
                print(
                    f"[COOP] Egg deferred flag=0x{flag:04X} reason={reason} "
                    f"count={deferred + 1}"
                )
                self._egg_nav_started_step = self._step_count
                self._egg_attempts = 0
                self._clear_left_top_route()
                self._phase = "egg_nav"
                return TaskResult(status=TaskStatus.RUNNING)
            self._skipped_egg_flags.add(flag)
            print(
                f"[COOP] Egg skipped flag=0x{flag:04X} reason={reason}"
            )
        else:
            print(f"[COOP] Egg pickup failed, skipping ({reason})")
        self._egg_attempts = 0
        self._egg_nav_started_step = 0
        self._current_egg_flag = 0
        self._clear_left_top_route()
        return TaskResult(status=TaskStatus.RUNNING)

    def _after_egg_nav_budget(self, ram: np.ndarray, reason: str) -> TaskResult:
        result = self._defer_or_skip_egg(reason)
        if self._collectable_egg_present(ram):
            self._phase = "egg_nav"
            self._egg_nav_started_step = self._step_count
            return result
        return self._begin_exit_prep()

    def _ship_pixel_action(self) -> Optional[np.ndarray]:
        x = self._navigator.current_pos.x
        y = self._navigator.current_pos.y
        if y < SHIP_APPROACH_Y:
            if abs(x - SHIP_LANE_X) > 2:
                return make_action(right=x < SHIP_LANE_X, left=x > SHIP_LANE_X, b=True)
            return make_action(down=True, b=True)
        if abs(x - SHIP_INTERACT_PX[0]) > 1:
            return make_action(right=x < SHIP_INTERACT_PX[0], left=x > SHIP_INTERACT_PX[0], b=True)
        if abs(y - SHIP_INTERACT_PX[1]) > 1:
            return make_action(down=y < SHIP_INTERACT_PX[1], up=y > SHIP_INTERACT_PX[1])
        return None

    def _near_ship_lane(self) -> bool:
        current = self._navigator.current_tile
        return (
            current in {SHIP_BIN_STAND, SHIP_BIN_INTERACT_STAND, SHIP_RIGHT_LANE_CORNER}
            or current[0] <= SHIP_BIN_STAND[0]
        )

    def _navigate_to_ship_stand(self, ram: np.ndarray) -> Optional[np.ndarray]:
        goal = (
            SHIP_RIGHT_LANE_CORNER
            if self._navigator.current_tile[0] >= 3 and self._navigator.current_tile[1] >= 10
            else SHIP_BIN_STAND
        )
        if goal == SHIP_RIGHT_LANE_CORNER and (
            self._navigator.current_tile == goal or self._navigator.at_tile(goal, tolerance=3)
        ):
            self._navigator.path = []
            return None
        if self._navigator.path and self._navigator.path[-1] != goal:
            self._navigator.path = []
        action = self._navigate_to_tile(ram, goal)
        if action is not None:
            return action
        return None

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

        # ── Feed phases ──

        if self._phase == "feed_nav":
            if self._fed_count_now(world.ram) >= self._adult_count:
                return self._advance_after_feed(world.ram)
            if read_item_on_hand(world.ram) == ITEM_CHICKEN_FEED:
                self._phase = "feed_place_nav"
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            action = self._navigate_to_left_top_goal(world.ram, FEED_BIN_STAND)
            if action is not None:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            self._hay_before = read_hay_count(world.ram)
            self._phase = "feed_act"

        if self._phase == "feed_act":
            held_item = read_item_on_hand(world.ram)
            if held_item == ITEM_CHICKEN_FEED:
                self._phase = "feed_place_nav"
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            if held_item != 0:
                self._phase = "feed_verify"
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            if self._fed_count_now(world.ram) >= self._adult_count:
                return self._advance_after_feed(world.ram)
            # Cap feeds at available hay
            if read_hay_count(world.ram) <= 0:
                print(f"[COOP] Out of hay after feeding {self.fed_count}")
                if self._egg_present(world.ram):
                    self._phase = "egg_nav"
                else:
                    self._phase = "done"
                return TaskResult(status=TaskStatus.RUNNING)
            self._queue_press_a(FEED_BIN_FACE, hold_frames=20, settle_frames=30)
            self._feed_registered = False
            self._verify_count = 0
            self._phase = "feed_verify"
            return TaskResult(status=TaskStatus.RUNNING)

        if self._phase == "feed_verify":
            if read_item_on_hand(world.ram) == ITEM_CHICKEN_FEED:
                self._verify_count = 0
                self._phase = "feed_place_nav"
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            self._verify_count += 1
            if self._verify_count > 40:
                # Feed pickup did not register; retry the bin interaction.
                self._phase = "feed_act"
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        if self._phase == "feed_place_nav":
            if self._fed_count_now(world.ram) >= self._adult_count:
                return self._advance_after_feed(world.ram)
            if read_item_on_hand(world.ram) != ITEM_CHICKEN_FEED:
                self._phase = "feed_nav"
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

            if self._feed_place_started_step <= 0:
                self._feed_place_started_step = self._step_count

            spot = self._current_feed_spot
            flags_now = read_fed_chickens_flags(world.ram)
            chicken_tiles = self._chicken_tiles(world.ram)
            if spot is None or (flags_now & spot.flag) or (
                spot.stand in chicken_tiles and spot.stand != self._navigator.current_tile
            ):
                spot = self._next_feed_spot(world.ram)
                self._current_feed_spot = spot
                self._feed_place_started_step = self._step_count
            if spot is None:
                print("[COOP] No open feed slot; continuing to eggs/exit")
                self._feed_remaining = 0
                return self._advance_after_feed(world.ram)

            timed_out = (
                self._step_count - self._feed_place_started_step > MAX_FEED_PLACE_FRAMES
            )
            if self._navigator.stasis > 120 or timed_out:
                deferred = self._deferred_feed_counts.get(spot.flag, 0)
                if deferred < MAX_FEED_SLOT_DEFERRALS and not timed_out:
                    self._deferred_feed_counts[spot.flag] = deferred + 1
                    print(
                        f"[COOP] Feed deferred flag=0x{spot.flag:04X} "
                        f"reason=stasis count={deferred + 1}"
                    )
                else:
                    reason = "slot_timeout" if timed_out else "stasis"
                    print(
                        f"[COOP] Feed skipped flag=0x{spot.flag:04X} reason={reason}"
                    )
                    self._blocked_feed_flags.add(spot.flag)
                self._current_feed_spot = None
                self._feed_place_started_step = self._step_count
                self._navigator.path = []
                self._navigator.stasis = 0
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

            if (
                self._navigator.current_tile[0] == spot.stand[0]
                and self._navigator.current_tile[1] > spot.stand[1]
            ):
                dx = spot.interact_px[0] - self._navigator.current_pos.x
                if abs(dx) > 1:
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        action=ActionResult(make_action(right=dx > 0, left=dx < 0)),
                    )

            if self._navigator.current_tile != spot.stand:
                action = self._navigate_to_tile(world.ram, spot.stand)
                if action is not None:
                    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

            action = align_to_pixel(
                (self._navigator.current_pos.x, self._navigator.current_pos.y),
                spot.interact_px,
                tolerance=1,
            )
            if action is not None:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

            self._fed_before = self._fed_count_now(world.ram)
            self._fed_flags_before = read_fed_chickens_flags(world.ram)
            self._queue_place_feed(spot.face)
            self._verify_count = 0
            self._feed_place_started_step = 0
            self._phase = "feed_place_verify"
            return TaskResult(status=TaskStatus.RUNNING)

        if self._phase == "feed_place_verify":
            fed_now = self._fed_count_now(world.ram)
            flags_now = read_fed_chickens_flags(world.ram)
            if fed_now > self._fed_before or flags_now != self._fed_flags_before:
                self.fed_count = max(self.fed_count, min(fed_now, self._adult_count))
                self._feed_remaining = max(0, self._adult_count - self.fed_count)
                print(
                    f"[COOP] Feed OK count={self.fed_count} "
                    f"remaining={self._feed_remaining} flags=0x{flags_now:04X}"
                )
                return self._advance_after_feed(world.ram)

            self._verify_count += 1
            if self._verify_count > 30:
                if read_item_on_hand(world.ram) == ITEM_CHICKEN_FEED:
                    self._phase = "feed_place_nav"
                else:
                    self._current_feed_spot = None
                    self._phase = "feed_nav"
                self._verify_count = 0
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        if self._phase == "feed_clear_nav":
            self._phase = "feed_place_nav"
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        if self._phase == "feed_clear_verify":
            self._phase = "feed_place_verify"
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        # ── Egg collection phases ──

        if self._phase == "egg_nav":
            if not self._collectable_egg_present(world.ram):
                return self._begin_exit_prep()
            if self._egg_nav_started_step <= 0:
                self._egg_nav_started_step = self._step_count
            if self._step_count - self._egg_nav_started_step > MAX_EGG_NAV_FRAMES:
                return self._after_egg_nav_budget(world.ram, "slot_timeout")
            pickup = self._egg_pickup_spot(world.ram)
            if pickup is None:
                return self._after_egg_nav_budget(world.ram, "no_reachable_stand")
            egg_stand, egg_face = pickup
            if self._navigator.stasis > 150:
                return self._after_egg_nav_budget(world.ram, "nav_stasis")
            action = self._navigate_to_left_top_goal(world.ram, egg_stand)
            if action is not None:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            self._egg_attempts += 1
            self._queue_press_a(
                egg_face,
                face_frames=4,
                hold_frames=28,
                settle_frames=30,
                hold_face_with_a=False,
            )
            self._verify_count = 0
            self._phase = "egg_verify"
            return TaskResult(status=TaskStatus.RUNNING)

        if self._phase == "egg_verify":
            if not self._egg_present(world.ram) or is_holding_egg(world.ram):
                self.egg_collected = True
                print(f"[COOP] Egg collected")
                if self._current_egg_flag:
                    self._skipped_egg_flags.discard(self._current_egg_flag)
                self._phase = "decide"
                return TaskResult(status=TaskStatus.RUNNING)
            held_item = read_item_on_hand(world.ram)
            if held_item not in (0, ITEM_EGG):
                self._verify_count += 1
                if self._verify_count > 60 and self._egg_attempts < MAX_EGG_ATTEMPTS:
                    self._phase = "egg_nav"
                    self._verify_count = 0
                elif self._verify_count > 60:
                    return self._after_egg_nav_budget(world.ram, "held_item_block")
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            self._verify_count += 1
            if self._verify_count > 15:
                if self._egg_attempts < MAX_EGG_ATTEMPTS:
                    self._phase = "egg_nav"
                    self._verify_count = 0
                else:
                    return self._after_egg_nav_budget(world.ram, "pickup_failed")
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        # ── Decision phase ──

        if self._phase == "decide":
            mode = self.egg_mode
            if mode == "auto":
                adults, chicks, eggs = count_chicken_slots(world.ram)
                total = adults + chicks + eggs
                if not is_incubating(world.ram) and total < MAX_FLOCK_SIZE:
                    mode = "incubate"
                else:
                    mode = "ship"
            if mode == "incubate":
                self._incubator_wp_index = 0
                self._phase = "incubate_nav"
            elif mode == "gift":
                print("[COOP] Gift mode — exiting with egg")
                return self._begin_exit_prep()
            else:
                self._phase = "ship_nav"
            return TaskResult(status=TaskStatus.RUNNING)

        # ── Incubate phases ──

        if self._phase == "incubate_nav":
            action = self._navigate_to_incubator_stand(world.ram)
            if action is not None:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            self._queue_press_a(INCUBATOR_FACE, hold_frames=20, settle_frames=24, hold_face_with_a=False)
            self._verify_count = 0
            self._phase = "incubate_verify"
            return TaskResult(status=TaskStatus.RUNNING)

        if self._phase == "incubate_verify":
            if is_incubating(world.ram):
                self.egg_incubated = True
                print("[COOP] Egg incubated")
                return self._after_egg_handled(world.ram)
            self._verify_count += 1
            if self._verify_count > 15:
                print("[COOP] Incubation failed, shipping instead")
                self._phase = "ship_nav"
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        # ── Ship phases ──

        if self._phase == "ship_nav":
            if not self._near_ship_lane():
                action = self._navigate_to_ship_stand(world.ram)
                if action is not None:
                    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            action = self._ship_pixel_action()
            if action is not None:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            current = self._navigator.current_tile
            if current == SHIP_BIN_INTERACT_STAND:
                self._ship_money_before = read_shipping_money(world.ram)
                self._queue_press_a(
                    SHIP_BIN_FACE,
                    face_frames=1,
                    hold_frames=20,
                    settle_frames=24,
                    hold_face_with_a=False,
                )
                self._verify_count = 0
                self._phase = "ship_verify"
                return TaskResult(status=TaskStatus.RUNNING)
            if current == SHIP_BIN_STAND:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(left=True, b=True)))
            action = self._navigate_to_tile(world.ram, SHIP_BIN_STAND)
            if action is not None:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            self._ship_money_before = read_shipping_money(world.ram)
            self._queue_press_a(
                SHIP_BIN_FACE,
                face_frames=1,
                hold_frames=20,
                settle_frames=24,
                hold_face_with_a=False,
            )
            self._verify_count = 0
            self._phase = "ship_verify"
            return TaskResult(status=TaskStatus.RUNNING)

        if self._phase == "ship_verify":
            money_now = read_shipping_money(world.ram)
            if money_now > self._ship_money_before:
                self.egg_shipped = True
                print(f"[COOP] Egg shipped, money={money_now}")
                return self._after_egg_handled(world.ram)
            if not is_holding_egg(world.ram):
                return TaskResult(status=TaskStatus.FAILURE, reason="egg cleared without shipping money")
            self._verify_count += 1
            if self._verify_count > 20:
                print("[COOP] Ship verify timeout, retrying")
                self._phase = "ship_nav"
                self._verify_count = 0
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        if self._phase == "exit_prep_nav":
            if self._exit_prep_started_step <= 0:
                self._exit_prep_started_step = self._step_count
            if self._step_count - self._exit_prep_started_step > MAX_EXIT_PREP_FRAMES:
                # Hand off wherever we are; EXIT_COOP owns the door route.
                print(
                    f"[COOP] Exit prep timeout at {self._navigator.current_tile}; "
                    "handing off to EXIT_COOP"
                )
                self._phase = "done"
                return TaskResult(status=TaskStatus.RUNNING)
            current = self._navigator.current_tile
            if current == EXIT_PREP_STAND or self._navigator.at_tile(EXIT_PREP_STAND):
                self._phase = "done"
                return TaskResult(status=TaskStatus.RUNNING)
            x = self._navigator.current_pos.x
            y = self._navigator.current_pos.y
            door_x = EXIT_PREP_STAND[0] * TILE_SIZE + 8
            safe_cross_max_y = 7 * TILE_SIZE + 8
            # Match EXIT_COOP: leave bin pocket, climb above false-open, cross,
            # then drop to the door. Hand off early once on the door column.
            if x < door_x - 4 or y > safe_cross_max_y or current != EXIT_PREP_STAND:
                if x <= 50 and y >= 165:
                    if abs(x - SHIP_LANE_X) > 2:
                        return TaskResult(
                            status=TaskStatus.RUNNING,
                            action=ActionResult(
                                make_action(
                                    right=x < SHIP_LANE_X,
                                    left=x > SHIP_LANE_X,
                                    b=True,
                                )
                            ),
                        )
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        action=ActionResult(make_action(up=True, b=True)),
                    )
                if x < door_x - 4 and y > safe_cross_max_y:
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        action=ActionResult(make_action(up=True, b=True)),
                    )
                if abs(x - door_x) > 3:
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        action=ActionResult(
                            make_action(right=x < door_x, left=x > door_x, b=True)
                        ),
                    )
                if current[1] < EXIT_PREP_STAND[1]:
                    return TaskResult(
                        status=TaskStatus.RUNNING,
                        action=ActionResult(make_action(down=True, b=True)),
                    )
            self._phase = "done"
            return TaskResult(status=TaskStatus.RUNNING)

        return TaskResult(status=TaskStatus.FAILURE, reason=f"unknown phase {self._phase}")

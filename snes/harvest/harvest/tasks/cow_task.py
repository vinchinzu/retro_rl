"""Autonomous barn cow chores.

Current scope follows recorded barn chores: milk ready cows and ship milk in
the barn bin, talk to and brush each cow when tools are available, then place
fodder in the trough.

Extracted arms (rr-y80y):
  - ``cow_geometry`` — pure barn geometry
  - ``cow_care`` — pixel-lane action builders
  - ``cow_fsm`` — CowPhase enum + shared constants
  - ``cow_talk_ops`` / ``cow_brush_ops`` / ``cow_milk_ops`` /
    ``cow_feed_ops`` / ``cow_exit_ops`` — phase step mixins
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import numpy as np

from harvest.core.animal_probe import BARN_TILEMAP, cow_slot_snapshots, cow_tiles_from_slots
from harvest.core.animal_status import (
    COW_DAILY_BRUSHED_FLAG,
    COW_DAILY_TALKED_FLAG,
    ITEM_FODDER,
    count_cow_slots,
    cow_needs_milking,
    existing_cow_slots,
    read_cow_daily_flags,
    read_cow_happiness,
    read_fed_cows_flags,
    read_fed_cows_n,
    read_held_item,
    read_num_cows,
    read_stored_grass,
)
from harvest.core.npc_catalog import game_objects
from harvest.core.ram_catalog import read_ram_u8
from harvest.core.tile_catalog import ADDR_INPUT_LOCK, ADDR_TILEMAP
from harvest.tasks.animal_navigation import align_to_pixel, fallback_action, find_path_around_blockers
from harvest.tasks.cow_brush_ops import CowBrushMixin
from harvest.tasks.cow_care import (
    left_lower_lane_from_right_action,
    left_side_vertical_nav_action,
    recorded_interact_lane_action,
    run_to_pixel_axis,
)
from harvest.tasks.cow_exit_ops import CowExitMixin
from harvest.tasks.cow_feed_ops import CowFeedMixin
from harvest.tasks.cow_fsm import (
    ADDR_PLAYER_ACTION,
    ADDR_TOOL_BACKPACK,
    ADDR_TOOL_SELECTED,
    BRUSH_TOOL_ID,
    CARE_PHASES,
    MAX_CARE_DEFERRALS,
    MAX_COW_NAV_FAILURES,
    MAX_COW_SLOT_CARE_FRAMES,
    MAX_COW_SLOT_MILK_FRAMES,
    MAX_NAV_FALLBACK_FRAMES,
    MAX_PIXEL_NAV_STALLS,
    MILK_CARE_PHASES,
    MILKER_TOOL_ID,
    PIXEL_NAV_STALL_FRAMES,
    TOOL_CARE_PHASES,
    CowPhase,
)
from harvest.tasks.cow_geometry import (
    CARE_TROUGH_EXIT_ANCHOR_X,
    CARE_TROUGH_EXIT_BOTTOM_Y,
    CARE_TROUGH_EXIT_MIN_Y,
    CARE_TROUGH_EXIT_X,
    COW_FEED_SPOTS,
    COW_TALK_FACE,
    COW_TALK_STAND,
    LEFT_TROUGH_RETURN_X,
    CowFeedSpot,
    body_side_stand_candidates,
    count_fed_trough_flags,
    cow_body_tile,
    cow_interact_pixel,
    cow_push_escape_tile,
    face_for_cow_at_stand,
    facing_tile,
    feed_route_for_spot,
    geometric_fallback_stands,
    is_adjacent_to_cow_tile,
    left_cow_lane_x,
    next_unfed_spot,
    preferred_cow_stands,
    stand_blocked,
    stand_in_bounds,
    talk_route_to,
)
from harvest.tasks.cow_milk_ops import CowMilkMixin
from harvest.tasks.cow_talk_ops import CowTalkMixin
from harvest.tasks.farm_clearer import TileScanner
from harvest.tasks.nav import MAP_WIDTH, Navigator, Pathfinder, make_action
from harvest.tasks.primitives import press_a_sequence
from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

# Test / caller re-exports (historical import path: harvest.tasks.cow_task).
from harvest.core.animal_status import (  # noqa: F401
    ADDR_FED_COWS_N,
    ADDR_HELD_ITEM,
    ADDR_NUM_COWS,
    ADDR_STORED_GRASS,
    COW_DAILY_MILKED_FLAG,
)
from harvest.tasks.cow_fsm import (  # noqa: F401
    MAX_BRUSH_ATTEMPTS,
    MAX_EXIT_PREP_FRAMES,
    MAX_MILK_ATTEMPTS,
    MAX_MILK_DEFERRALS,
    MAX_TALK_ATTEMPTS,
)
from harvest.tasks.cow_geometry import (  # noqa: F401
    BARN_SHIP_BIN_INTERACT_STAND,
    BARN_SHIP_BIN_STAND,
    COW_EXIT_PREP_STAND,
    COW_INTERACT_X_OFFSET,
    COW_LEFT_INTERACT_X,
    COW_TALK_ROUTE,
    FEED_TROUGH_INTERACT_PX,
    FEED_TROUGH_ROUTE,
    FEED_TROUGH_STAND,
    FODDER_ROUTE,
    FODDER_STAND,
    FODDER_TROUGH_ROUTE,
    MILK_SHIP_ROUTE,
)

@dataclass
class CowChoresTask(
    CowTalkMixin,
    CowBrushMixin,
    CowMilkMixin,
    CowFeedMixin,
    CowExitMixin,
    Task,
):
    """Talk to and feed cows inside the barn."""

    name: str = "cow_chores"
    talk: bool = True
    brush: bool = True
    milk: bool = True
    feed: bool = True
    timeout: int = 30000

    _scanner: TileScanner = field(default_factory=TileScanner, init=False)
    _pathfinder: Pathfinder = field(init=False)
    _navigator: Navigator = field(init=False)
    _phase: str = field(default=CowPhase.TALK_NAV, init=False)
    _action_queue: Deque[np.ndarray] = field(default_factory=deque, init=False)
    _step_count: int = field(default=0, init=False)
    _verify_count: int = field(default=0, init=False)
    _interaction_started: bool = field(default=False, init=False)
    _cow_count: int = field(default=0, init=False)
    _feed_remaining: int = field(default=0, init=False)
    _feed_goal_count: int = field(default=0, init=False)
    _grass_before: int = field(default=0, init=False)
    _fed_before: int = field(default=0, init=False)
    _fed_flags_before: int = field(default=0, init=False)
    _target_cow_slot: Optional[int] = field(default=None, init=False)
    _care_slots: list[int] = field(default_factory=list, init=False)
    _skipped_talk_slots: set[int] = field(default_factory=set, init=False)
    _skipped_brush_slots: set[int] = field(default_factory=set, init=False)
    _skipped_milk_slots: set[int] = field(default_factory=set, init=False)
    _deferred_care_counts: dict[int, int] = field(default_factory=dict, init=False)
    _deferred_milk_counts: dict[int, int] = field(default_factory=dict, init=False)
    _talk_flags_before: int = field(default=0, init=False)
    _talk_happiness_before: int = field(default=0, init=False)
    _brush_flags_before: int = field(default=0, init=False)
    _brush_happiness_before: int = field(default=0, init=False)
    _talk_attempts: int = field(default=0, init=False)
    _talk_route_index: int = field(default=0, init=False)
    _brush_route_index: int = field(default=0, init=False)
    _brush_select_frames: int = field(default=0, init=False)
    _brush_attempts: int = field(default=0, init=False)
    _milk_slots: list[int] = field(default_factory=list, init=False)
    _milked_slots: set[int] = field(default_factory=set, init=False)
    _milk_select_frames: int = field(default=0, init=False)
    _milk_attempts: int = field(default=0, init=False)
    _milk_flags_before: int = field(default=0, init=False)
    _milk_held_before: int = field(default=0, init=False)
    _ship_money_before: int = field(default=0, init=False)
    _ship_route_index: int = field(default=0, init=False)
    _care_slot_started_step: int = field(default=0, init=False)
    _fodder_route_index: int = field(default=0, init=False)
    _feed_route_index: int = field(default=0, init=False)
    _talk_stand: Tuple[int, int] = field(default=COW_TALK_STAND, init=False)
    _talk_face: str = field(default=COW_TALK_FACE, init=False)
    _nav_failures: int = field(default=0, init=False)
    _recent_pin_slot: Optional[int] = field(default=None, init=False)
    _recent_pin_stand: Optional[Tuple[int, int]] = field(default=None, init=False)
    _recent_pin_face: str = field(default=COW_TALK_FACE, init=False)
    _care_trough_exit_logged: bool = field(default=False, init=False)
    _pixel_nav_target: Optional[Tuple[int, int]] = field(default=None, init=False)
    _pixel_nav_best_dist: int = field(default=10**9, init=False)
    _pixel_nav_stale_frames: int = field(default=0, init=False)
    _pixel_nav_stall_count: int = field(default=0, init=False)
    _exit_prep_started_step: int = field(default=0, init=False)
    talked: bool = field(default=False, init=False)
    brushed: bool = field(default=False, init=False)
    milked_count: int = field(default=0, init=False)
    milk_shipped_count: int = field(default=0, init=False)
    fed_count: int = field(default=0, init=False)


    def __post_init__(self) -> None:
        self._pathfinder = Pathfinder(self._scanner)
        self._navigator = Navigator(self._pathfinder)


    def reset(self, world: WorldState) -> None:
        self._step_count = 0
        self._verify_count = 0
        self._interaction_started = False
        self._action_queue.clear()
        self._navigator.update(world.ram)
        self._navigator.path = []
        self._navigator.stasis = 0
        self._pathfinder.temp_blocked.clear()
        self._talk_route_index = 0
        self._brush_route_index = 0
        self._talk_attempts = 0
        self._brush_select_frames = 0
        self._brush_attempts = 0
        self._milk_select_frames = 0
        self._milk_attempts = 0
        self._milk_flags_before = 0
        self._milk_held_before = 0
        self._ship_money_before = 0
        self._ship_route_index = 0
        self._care_slot_started_step = 0
        self._milked_slots.clear()
        self._care_slots.clear()
        self._skipped_talk_slots.clear()
        self._skipped_brush_slots.clear()
        self._skipped_milk_slots.clear()
        self._deferred_care_counts.clear()
        self._deferred_milk_counts.clear()
        self._recent_pin_slot = None
        self._recent_pin_stand = None
        self._recent_pin_face = COW_TALK_FACE
        self._nav_failures = 0
        self._fodder_route_index = 0
        self._feed_route_index = 0
        self._pixel_nav_target = None
        self._pixel_nav_best_dist = 10**9
        self._pixel_nav_stale_frames = 0
        self._pixel_nav_stall_count = 0
        self._exit_prep_started_step = 0
        self.talked = False
        self.brushed = False
        self.milked_count = 0
        self.milk_shipped_count = 0
        self.fed_count = 0

        self._cow_count = max(read_num_cows(world.ram), count_cow_slots(world.ram))
        self._milk_slots = self._milkable_cow_slots(world.ram)
        self._care_slots = self._care_needed_cow_slots(world.ram)
        self._target_cow_slot = self._care_slots[0] if self._care_slots else self._select_target_cow_slot(world.ram)
        self._feed_goal_count = self._feed_goal(world.ram)
        self._fed_before = self._fed_count_now(world.ram)
        self._fed_flags_before = read_fed_cows_flags(world.ram)
        self._grass_before = read_stored_grass(world.ram)
        self._feed_remaining = max(0, self._feed_goal_count - self._fed_before)
        self._refresh_talk_approach(world.ram)

        if self._cow_count <= 0:
            self._phase = CowPhase.DONE
        elif self.milk and self._milker_in_carry_pair(world.ram) and self._begin_next_milk(world.ram):
            pass
        elif self.feed and self._feed_remaining > 0 and self._grass_before > 0:
            self._phase = CowPhase.FODDER_NAV if read_held_item(world.ram) != ITEM_FODDER else "feed_place_nav"
        elif self._begin_next_cow_care(world.ram):
            pass
        else:
            self._begin_exit_prep()

        print(
            f"[COW] cows={self._cow_count} fed={self._fed_before} "
            f"hay={self._grass_before} target_slot={self._target_cow_slot} phase={self._phase}"
        )


    def can_start(self, world: WorldState) -> bool:
        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        return tilemap == BARN_TILEMAP


    def resume_after_hotswap(self, world: WorldState) -> None:
        self._action_queue.clear()
        self._navigator.update(world.ram)
        self._navigator.path = []
        self._navigator.stasis = 0
        self._pathfinder.temp_blocked.clear()
        self._talk_route_index = 0
        self._brush_route_index = 0
        self._talk_attempts = 0
        self._brush_select_frames = 0
        self._brush_attempts = 0
        self._milk_select_frames = 0
        self._milk_attempts = 0
        self._ship_route_index = 0
        self._care_slot_started_step = self._step_count
        self._nav_failures = 0
        self._fodder_route_index = 0
        self._feed_route_index = 0
        self._care_trough_exit_logged = False


    def progress_text(self) -> str:
        return (
            f"talk={'Y' if self.talked else 'N'} "
            f"brush={'Y' if self.brushed else 'N'} "
            f"milk={self.milked_count}/{len(self._milk_slots)} "
            f"fed={self.fed_count}/{self._feed_goal_count or self._cow_count}"
        )


    def _facing_tile(self, stand: Tuple[int, int], face: str) -> Tuple[int, int]:
        return facing_tile(stand, face)


    def _select_target_cow_slot(self, ram: np.ndarray) -> Optional[int]:
        rows = cow_slot_snapshots(ram, require_barn=True)
        if not rows:
            slots = existing_cow_slots(ram)
            return slots[0] if slots else None

        target_tile = facing_tile(COW_TALK_STAND, COW_TALK_FACE)
        for row in rows:
            tile = row.get("tile")
            if isinstance(tile, list) and tuple(tile) == target_tile:
                return int(row["slot"])

        def score(row: dict[str, object]) -> int:
            tile = row.get("tile")
            if not isinstance(tile, list) or len(tile) != 2:
                return 999
            return abs(int(tile[0]) - target_tile[0]) + abs(int(tile[1]) - target_tile[1])

        return int(min(rows, key=score)["slot"])


    def _cow_flag_set(self, ram: np.ndarray, flag: int) -> bool:
        if self._target_cow_slot is None:
            return False
        return bool(read_cow_daily_flags(ram, self._target_cow_slot) & flag)


    def _cow_flag_set_for_slot(self, ram: np.ndarray, slot: int, flag: int) -> bool:
        return bool(read_cow_daily_flags(ram, slot) & flag)


    def _milkable_cow_slots(self, ram: np.ndarray) -> list[int]:
        return [
            slot
            for slot in existing_cow_slots(ram)
            if cow_needs_milking(ram, slot) and slot not in self._skipped_milk_slots
        ]


    def _barn_cow_slots(self, ram: np.ndarray) -> list[int]:
        rows = cow_slot_snapshots(ram, require_barn=True)
        slots = [int(row["slot"]) for row in rows if "slot" in row]
        return slots or existing_cow_slots(ram)


    def _slot_needs_talk(self, ram: np.ndarray, slot: int) -> bool:
        return (
            self.talk
            and slot not in self._skipped_talk_slots
            and not self._cow_flag_set_for_slot(ram, slot, COW_DAILY_TALKED_FLAG)
        )


    def _slot_needs_brush(self, ram: np.ndarray, slot: int) -> bool:
        return (
            self.brush
            and self._brush_in_carry_pair(ram)
            and slot not in self._skipped_brush_slots
            and not self._cow_flag_set_for_slot(ram, slot, COW_DAILY_BRUSHED_FLAG)
        )


    def _slot_ready_for_milk(self, ram: np.ndarray, slot: int) -> bool:
        return True


    def _slot_needs_milk(self, ram: np.ndarray, slot: int) -> bool:
        return (
            self.milk
            and self._milker_in_carry_pair(ram)
            and slot not in self._skipped_milk_slots
            and cow_needs_milking(ram, slot)
            and self._slot_ready_for_milk(ram, slot)
        )


    def _slot_needs_care(self, ram: np.ndarray, slot: int) -> bool:
        return (
            self._slot_needs_talk(ram, slot)
            or self._slot_needs_brush(ram, slot)
            or self._slot_needs_milk(ram, slot)
        )


    def _care_needed_cow_slots(self, ram: np.ndarray) -> list[int]:
        return [slot for slot in self._barn_cow_slots(ram) if self._slot_needs_care(ram, slot)]


    def _feedable_cow_slots(self, ram: np.ndarray) -> list[int]:
        return existing_cow_slots(ram)


    def _feed_goal(self, ram: np.ndarray) -> int:
        return min(len(self._feedable_cow_slots(ram)), len(COW_FEED_SPOTS))


    def _current_feed_goal(self, ram: np.ndarray) -> int:
        return self._feed_goal_count or self._feed_goal(ram)


    def _fed_trough_count(self, ram: np.ndarray) -> int:
        return count_fed_trough_flags(read_fed_cows_flags(ram), self._current_feed_goal(ram))


    def _fed_count_now(self, ram: np.ndarray) -> int:
        return max(read_fed_cows_n(ram), self._fed_trough_count(ram))


    def _next_feed_spot(self, ram: np.ndarray) -> CowFeedSpot:
        return next_unfed_spot(read_fed_cows_flags(ram), self._current_feed_goal(ram))


    def _feed_route(self, spot: CowFeedSpot) -> Tuple[Tuple[int, int], ...]:
        return feed_route_for_spot(spot)


    def _target_cow_tile(self, ram: np.ndarray) -> Optional[Tuple[int, int]]:
        if self._target_cow_slot is None:
            return None
        for row in cow_slot_snapshots(ram, require_barn=True):
            if int(row.get("slot", -1)) != self._target_cow_slot:
                continue
            tile = row.get("tile")
            if not isinstance(tile, list) or len(tile) != 2:
                return None
            return int(tile[0]), int(tile[1])
        return None


    def _target_cow_pixel(self, ram: np.ndarray) -> Optional[Tuple[int, int]]:
        if self._target_cow_slot is None:
            return None
        for row in cow_slot_snapshots(ram, require_barn=True):
            if int(row.get("slot", -1)) != self._target_cow_slot:
                continue
            pixel = row.get("pixel")
            if not isinstance(pixel, list) or len(pixel) != 2:
                return None
            return int(pixel[0]), int(pixel[1])
        return None


    def _target_cow_body_tile(self, ram: np.ndarray) -> Optional[Tuple[int, int]]:
        tile = self._target_cow_tile(ram)
        if tile is None:
            return None
        return cow_body_tile(tile)


    def _is_adjacent_to_target_cow(self, ram: np.ndarray, stand: Tuple[int, int], face: str) -> bool:
        tile = self._target_cow_tile(ram)
        if tile is None:
            return False
        return is_adjacent_to_cow_tile(stand, face, tile)


    def _remember_current_pin(self) -> None:
        if self._target_cow_slot is None:
            return
        self._recent_pin_slot = self._target_cow_slot
        self._recent_pin_stand = self._navigator.current_tile
        self._recent_pin_face = self._talk_face


    def _recent_pin_milk_face(self, ram: np.ndarray, stand: Optional[Tuple[int, int]] = None) -> Optional[str]:
        if self._target_cow_slot is None or self._recent_pin_slot != self._target_cow_slot:
            return None
        stand = stand or self._navigator.current_tile
        if self._recent_pin_stand != stand:
            return None
        if self._recent_pin_face not in ("left", "right"):
            return None
        if self._is_adjacent_to_target_cow(ram, stand, self._recent_pin_face):
            return self._recent_pin_face
        tile = self._target_cow_tile(ram)
        if tile is None:
            return None
        facing = self._facing_tile(stand, self._recent_pin_face)
        # After a successful brush/talk, the cow can idle one horizontal body
        # tile away before the milker is selected. Reuse only that proven pin;
        # do not make this a general brush/talk adjacency rule.
        if facing[1] == tile[1] and abs(facing[0] - tile[0]) == 1:
            flags = read_cow_daily_flags(ram, self._target_cow_slot)
            if flags & (COW_DAILY_BRUSHED_FLAG | COW_DAILY_TALKED_FLAG):
                return self._recent_pin_face
        return None


    def _face_for_target_cow(self, ram: np.ndarray, stand: Optional[Tuple[int, int]] = None) -> str:
        stand = stand or self._talk_stand
        return face_for_cow_at_stand(
            stand,
            self._target_cow_tile(ram),
            default_face=COW_TALK_FACE,
            talk_stand=self._talk_stand,
            talk_face=self._talk_face,
        )


    def _cow_interact_pixel(self, ram: np.ndarray, *, tool: bool) -> Optional[Tuple[int, int]]:
        pixel = self._target_cow_pixel(ram)
        if pixel is None:
            return None
        return cow_interact_pixel(
            pixel,
            self._talk_face,
            tool=tool,
            cow_tile=self._target_cow_tile(ram),
        )


    def _at_cow_interact_pixel(self, ram: np.ndarray, *, tool: bool, tolerance: int = 1) -> bool:
        target = self._cow_interact_pixel(ram, tool=tool)
        if target is None:
            return False
        if tool and self._talk_face in ("left", "right"):
            return (
                target[0] == self._navigator.current_pos.x
                and target[1] == self._navigator.current_pos.y
            )
        return (
            abs(target[0] - self._navigator.current_pos.x) <= tolerance
            and abs(target[1] - self._navigator.current_pos.y) <= tolerance
        )


    def _align_to_cow_interact_pixel(self, ram: np.ndarray, *, tool: bool) -> Optional[np.ndarray]:
        target = self._cow_interact_pixel(ram, tool=tool)
        if target is None:
            return None
        if tool and self._talk_face in ("left", "right"):
            dx = target[0] - self._navigator.current_pos.x
            dy = target[1] - self._navigator.current_pos.y
            if dx != 0:
                return make_action(right=dx > 0, left=dx < 0)
            if dy != 0:
                return make_action(down=dy > 0, up=dy < 0)
            return None
        return align_to_pixel(
            (self._navigator.current_pos.x, self._navigator.current_pos.y),
            target,
            tolerance=1,
        )


    def _candidate_cow_stands(self, ram: np.ndarray) -> list[Tuple[Tuple[int, int], str]]:
        tile = self._target_cow_tile(ram)
        if tile is None:
            return [(COW_TALK_STAND, COW_TALK_FACE)]

        cx, cy = tile
        preferred: list[Tuple[Tuple[int, int], str]] = []
        current = self._navigator.current_tile
        current_face = self._face_for_target_cow(ram, current)
        if self._is_adjacent_to_target_cow(ram, current, current_face):
            preferred.append((current, current_face))
        preferred.extend(preferred_cow_stands(cx, cy))

        candidates: list[Tuple[Tuple[int, int], str]] = []
        scored: list[Tuple[Tuple[int, int, int], Tuple[Tuple[int, int], str]]] = []
        seen: set[Tuple[int, int]] = set()
        cow_tiles = self._cow_tiles(ram)
        for index, (stand, face) in enumerate(preferred):
            sx, sy = stand
            if stand in seen:
                continue
            seen.add(stand)
            if not stand_in_bounds(stand):
                continue
            if stand_blocked(stand, cow_tiles):
                continue
            if not self._pathfinder.is_walkable(ram, sx, sy, current_pos=self._navigator.current_tile):
                continue
            if self._find_path_around_cows(ram, self._navigator.current_tile, stand) is None:
                continue
            candidates.append((stand, face))
            # Wall-side cows already prefer body-right stands in `preferred`;
            # escape-pin scoring would re-rank head-on (1, cy) first and that
            # stand often fails to start talk/brush dialog.
            if cx <= 4:
                pin_penalty = 0
            else:
                pin_penalty = 0 if self._cow_escape_blocked(ram, tile, stand, face, cow_tiles) else 1
            current = self._navigator.current_tile
            distance = abs(sx - current[0]) + abs(sy - current[1])
            scored.append(((pin_penalty, index, distance), (stand, face)))
        if scored:
            return [item for _score, item in sorted(scored, key=lambda row: row[0])]
        if candidates:
            return candidates
        # Path checks can fail while cows shuffle; still aim at a geometric
        # side stand instead of snapping to the default talk tile across barn.
        loose: list[Tuple[Tuple[int, int], Tuple[Tuple[int, int], str]]] = []
        for index, (stand, face) in enumerate(preferred):
            sx, sy = stand
            if not stand_in_bounds(stand):
                continue
            if stand_blocked(stand, cow_tiles):
                continue
            if not self._pathfinder.is_walkable(
                ram, sx, sy, current_pos=self._navigator.current_tile
            ):
                continue
            current = self._navigator.current_tile
            distance = abs(sx - current[0]) + abs(sy - current[1])
            loose.append(((index, distance), (stand, face)))
        if loose:
            return [item for _score, item in sorted(loose, key=lambda row: row[0])]
        # Absolute geometric fallback — never snap to the default talk stand
        # when we still know where the target cow is.
        current = self._navigator.current_tile
        return geometric_fallback_stands(
            cx,
            cy,
            cow_tiles,
            current=current,
            current_face=self._face_for_target_cow(ram, current),
        )


    def _cow_escape_blocked(
        self,
        ram: np.ndarray,
        cow_tile: Tuple[int, int],
        stand: Tuple[int, int],
        face: str,
        cow_tiles: set[Tuple[int, int]],
    ) -> bool:
        escape = cow_push_escape_tile(cow_tile, stand, face)
        if escape is None:
            return False
        if not stand_in_bounds(escape):
            return True
        other_cow_tiles = set(cow_tiles)
        other_cow_tiles.discard(cow_tile)
        other_cow_tiles.discard(cow_body_tile(cow_tile))
        if escape in other_cow_tiles:
            return True
        return not self._pathfinder.is_walkable(
            ram, escape[0], escape[1], current_pos=self._navigator.current_tile
        )


    def _refresh_talk_approach(self, ram: np.ndarray) -> None:
        stand, face = self._candidate_cow_stands(ram)[0]
        if stand != self._talk_stand:
            self._clear_navigation()
        self._talk_stand = stand
        self._talk_face = face


    def _talk_route(self) -> Tuple[Tuple[int, int], ...]:
        return talk_route_to(self._talk_stand)


    def _refresh_stale_cow_approach(self, ram: np.ndarray, index_attr: str) -> None:
        if self._target_cow_slot is None:
            return
        if self._is_adjacent_to_target_cow(ram, self._talk_stand, self._talk_face):
            return
        self._refresh_talk_approach(ram)
        setattr(self, index_attr, max(0, len(self._talk_route()) - 1))


    def _cow_ram_changed(self, ram: np.ndarray, flag: int, before_flags: int, before_happiness: int) -> bool:
        if self._target_cow_slot is None:
            return False
        flags_now = read_cow_daily_flags(ram, self._target_cow_slot)
        happiness_now = read_cow_happiness(ram, self._target_cow_slot)
        if before_flags & flag:
            return True
        return bool((flags_now & flag) and (flags_now != before_flags or happiness_now > before_happiness))


    def _selected_tool(self, ram: np.ndarray) -> int:
        return read_ram_u8(ram, ADDR_TOOL_SELECTED)


    def _backpack_tool(self, ram: np.ndarray) -> int:
        return read_ram_u8(ram, ADDR_TOOL_BACKPACK)


    def _player_action(self, ram: np.ndarray) -> int:
        return read_ram_u8(ram, ADDR_PLAYER_ACTION)


    def _brush_selected(self, ram: np.ndarray) -> bool:
        return self._selected_tool(ram) == BRUSH_TOOL_ID


    def _brush_in_carry_pair(self, ram: np.ndarray) -> bool:
        return self._selected_tool(ram) == BRUSH_TOOL_ID or self._backpack_tool(ram) == BRUSH_TOOL_ID


    def _milker_selected(self, ram: np.ndarray) -> bool:
        return self._selected_tool(ram) == MILKER_TOOL_ID


    def _milker_in_carry_pair(self, ram: np.ndarray) -> bool:
        return self._selected_tool(ram) == MILKER_TOOL_ID or self._backpack_tool(ram) == MILKER_TOOL_ID


    def _queue_press_a(
        self,
        face: str,
        *,
        face_frames: int = 8,
        hold_frames: int = 20,
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


    def _queue_use_tool(
        self,
        face: str,
        *,
        face_frames: int = 0,
        hold_frames: int = 22,
        y_only_frames: int = 0,
        settle_frames: int = 20,
        hold_face_with_y: bool = True,
    ) -> None:
        self._action_queue.extend(make_action(**{face: True}) for _ in range(face_frames))
        if hold_face_with_y:
            self._action_queue.extend(make_action(**{face: True, "y": True}) for _ in range(hold_frames))
        else:
            self._action_queue.extend(make_action(y=True) for _ in range(hold_frames))
        self._action_queue.extend(make_action(y=True) for _ in range(y_only_frames))
        self._action_queue.extend(make_action() for _ in range(settle_frames))


    def _clear_navigation(self) -> None:
        self._navigator.path = []
        self._navigator.stasis = 0
        self._pathfinder.temp_blocked.clear()
        self._nav_failures = 0


    def _reset_pixel_nav_progress(self) -> None:
        self._pixel_nav_target = None
        self._pixel_nav_best_dist = 10**9
        self._pixel_nav_stale_frames = 0


    def _pixel_nav_stalled(self, target: Tuple[int, int]) -> bool:
        """Detect sub-tile oscillation that keeps Navigator.stasis at 0."""
        x = self._navigator.current_pos.x
        y = self._navigator.current_pos.y
        dist = abs(x - target[0]) + abs(y - target[1])
        if self._pixel_nav_target != target:
            self._pixel_nav_target = target
            self._pixel_nav_best_dist = dist
            self._pixel_nav_stale_frames = 0
            return False
        if dist + 1 < self._pixel_nav_best_dist:
            self._pixel_nav_best_dist = dist
            self._pixel_nav_stale_frames = 0
            return False
        self._pixel_nav_stale_frames += 1
        return self._pixel_nav_stale_frames >= PIXEL_NAV_STALL_FRAMES


    def _handle_pixel_nav_action(
        self,
        ram: np.ndarray,
        action: Optional[np.ndarray],
        *,
        tool: bool,
    ) -> Optional[TaskResult]:
        """Apply recorded pixel-lane action, or escalate when it stops closing."""
        if action is None:
            return None
        target = self._cow_interact_pixel(ram, tool=tool)
        if target is not None and self._pixel_nav_stalled(target):
            self._pixel_nav_stall_count += 1
            print(
                f"[COW] Pixel nav stall slot={self._target_cow_slot} "
                f"count={self._pixel_nav_stall_count} target={target} "
                f"{self._care_debug_context(ram)}"
            )
            self._reset_pixel_nav_progress()
            self._clear_navigation()
            if self._pixel_nav_stall_count >= MAX_PIXEL_NAV_STALLS:
                self._pixel_nav_stall_count = 0
                return self._skip_current_cow_care(ram, "pixel_nav_stall")
            self._refresh_talk_approach(ram)
            self._talk_route_index = max(0, len(self._talk_route()) - 1)
            self._brush_route_index = self._talk_route_index
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
        self._clear_navigation()
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))


    def _care_debug_context(self, ram: np.ndarray) -> str:
        tool = self._phase in TOOL_CARE_PHASES
        return (
            f"phase={self._phase} pos=({self._navigator.current_pos.x},{self._navigator.current_pos.y}) "
            f"tile={self._navigator.current_tile} cow_tile={self._target_cow_tile(ram)} "
            f"cow_px={self._target_cow_pixel(ram)} stand={self._talk_stand} face={self._talk_face} "
            f"interact_px={self._cow_interact_pixel(ram, tool=tool)} "
            f"route_idx=t{self._talk_route_index}/b{self._brush_route_index} "
            f"path_next={self._navigator.path[0] if self._navigator.path else None} "
            f"stasis={self._navigator.stasis} nav_failures={self._nav_failures}"
        )


    def _dialog_pulse_action(self) -> np.ndarray:
        """Tap A with gaps so modal text advances instead of treating A as held."""
        cycle = self._verify_count % 22
        return make_action(a=6 <= cycle < 12)


    def _run_to_pixel_axis(
        self,
        target: Tuple[int, int],
        *,
        tolerance: int = 2,
        x_first: bool = False,
        y_first: bool = False,
    ) -> Optional[np.ndarray]:
        return run_to_pixel_axis(
            (self._navigator.current_pos.x, self._navigator.current_pos.y),
            target,
            tolerance=tolerance,
            x_first=x_first,
            y_first=y_first,
        )


    def _left_cow_lane_x(self, current_y: int) -> int:
        return left_cow_lane_x(current_y)


    def _left_lower_lane_from_right_action(self) -> Optional[np.ndarray]:
        return left_lower_lane_from_right_action(
            self._navigator.current_pos.x,
            self._navigator.current_pos.y,
        )


    def _left_side_vertical_nav_action(
        self,
        x: int,
        y: int,
        tx: int,
        ty: int,
        *,
        going_down: bool,
    ) -> Optional[np.ndarray]:
        """Reach wall-side interact pixels via the recorded left vertical lane."""
        return left_side_vertical_nav_action(x, y, tx, ty, going_down=going_down)


    def _recorded_interact_nav_action(self, ram: np.ndarray, *, tool: bool) -> Optional[np.ndarray]:
        if self._talk_face not in ("left", "right"):
            return None
        target = self._cow_interact_pixel(ram, tool=tool)
        if target is None:
            return None

        tx, ty = target
        x = self._navigator.current_pos.x
        y = self._navigator.current_pos.y
        if abs(x - tx) <= 1 and abs(y - ty) <= 1:
            return None
        # Talk only: already beside the cow, let fine align / A-press finish.
        # Tool use still needs recorded nav to the exact interact pixel.
        if (
            not tool
            and self._is_adjacent_to_target_cow(
                ram, self._navigator.current_tile, self._talk_face
            )
            and abs(x - tx) <= 16
            and abs(y - ty) <= 16
        ):
            return None

        return recorded_interact_lane_action(x, y, tx, ty, face=self._talk_face)


    def _care_trough_exit_action(self, ram: np.ndarray) -> Optional[np.ndarray]:
        x = self._navigator.current_pos.x
        y = self._navigator.current_pos.y
        if x < CARE_TROUGH_EXIT_X - 18 or x > LEFT_TROUGH_RETURN_X:
            return None
        if y < CARE_TROUGH_EXIT_MIN_Y:
            return None
        # Lower corridor + left-wall care targets: do not yank back to the
        # right aisle anchor (that fought pixel nav at ~x=129,y=345).
        target = self._cow_interact_pixel(ram, tool=False)
        if (
            target is not None
            and target[0] < LEFT_TROUGH_RETURN_X
            and y >= CARE_TROUGH_EXIT_BOTTOM_Y - 16
        ):
            return None
        if y < CARE_TROUGH_EXIT_BOTTOM_Y - 2:
            if abs(x - CARE_TROUGH_EXIT_X) > 2:
                action = make_action(right=x < CARE_TROUGH_EXIT_X, left=x > CARE_TROUGH_EXIT_X, b=True)
            else:
                action = make_action(down=True, b=True)
        elif x < CARE_TROUGH_EXIT_ANCHOR_X - 2:
            action = make_action(right=True, b=True)
        elif y > CARE_TROUGH_EXIT_BOTTOM_Y + 8:
            action = make_action(up=True, b=True)
        else:
            return None
        if not self._care_trough_exit_logged:
            print(
                f"[COW] Care trough exit slot={self._target_cow_slot} "
                f"anchor=({CARE_TROUGH_EXIT_ANCHOR_X},{CARE_TROUGH_EXIT_BOTTOM_Y}) "
                f"{self._care_debug_context(ram)}"
            )
            self._care_trough_exit_logged = True
        self._clear_navigation()
        return action


    def _recorded_left_tool_nav_action(self, ram: np.ndarray) -> Optional[np.ndarray]:
        return self._recorded_interact_nav_action(ram, tool=True)


    def _navigate_route(
        self,
        ram: np.ndarray,
        route: Tuple[Tuple[int, int], ...],
        index_attr: str,
        *,
        center_final: bool = True,
    ) -> Optional[np.ndarray]:
        index = int(getattr(self, index_attr))
        target = route[min(index, len(route) - 1)]
        if index < len(route) - 1 and self._navigator.current_tile == target:
            setattr(self, index_attr, index + 1)
            self._clear_navigation()
            return make_action()
        if index == len(route) - 1 and self._navigator.current_tile == target and not center_final:
            self._clear_navigation()
            return None

        action = self._navigate_to_tile(ram, target)
        if action is not None:
            return action

        if index < len(route) - 1:
            setattr(self, index_attr, index + 1)
            self._clear_navigation()
            return make_action()
        return None


    def _can_reach_talk_stand_directly(self, ram: np.ndarray) -> bool:
        return self._find_path_around_cows(
            ram,
            self._navigator.current_tile,
            self._talk_stand,
        ) is not None


    def _pin_care_route_to_direct_stand(self, ram: np.ndarray) -> None:
        if self._can_reach_talk_stand_directly(ram):
            direct_index = max(0, len(self._talk_route()) - 1)
            self._talk_route_index = direct_index
            self._brush_route_index = direct_index


    def _prefer_body_side_stand(self, ram: np.ndarray) -> bool:
        tile = self._target_cow_tile(ram)
        if tile is None:
            return False
        cx, cy = tile
        cow_tiles = self._cow_tiles(ram)
        for stand, face in body_side_stand_candidates(cx, cy):
            sx, sy = stand
            if not stand_in_bounds(stand):
                continue
            if stand_blocked(stand, cow_tiles):
                continue
            if not self._is_adjacent_to_target_cow(ram, stand, face):
                continue
            if not self._pathfinder.is_walkable(ram, sx, sy, current_pos=self._navigator.current_tile):
                continue
            if self._find_path_around_cows(ram, self._navigator.current_tile, stand) is None:
                continue
            self._talk_stand = stand
            self._talk_face = face
            self._talk_route_index = max(0, len(self._talk_route()) - 1)
            self._brush_route_index = self._talk_route_index
            return True
        return False


    def _base_cow_tiles(self, ram: np.ndarray) -> set[Tuple[int, int]]:
        tiles = cow_tiles_from_slots(ram, require_barn=True)
        if tiles:
            return tiles
        fallback: set[Tuple[int, int]] = set()
        for obj in game_objects(ram):
            if obj.label != "cow" and obj.kind != "animal":
                continue
            tx, ty = obj.tile
            if 0 <= tx < MAP_WIDTH and 0 <= ty < MAP_WIDTH:
                fallback.add((tx, ty))
        return fallback


    def _cow_tiles(self, ram: np.ndarray) -> set[Tuple[int, int]]:
        tiles = self._base_cow_tiles(ram)
        expanded = set(tiles)
        for tx, ty in tiles:
            if 0 <= ty + 1 < MAP_WIDTH:
                expanded.add((tx, ty + 1))
        return expanded


    def _find_path_around_cows(
        self,
        ram: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[list[Tuple[int, int]]]:
        blocked = self._cow_tiles(ram)
        blocked.update(self._pathfinder.temp_blocked)
        blocked.discard(goal)
        return find_path_around_blockers(
            ram,
            self._pathfinder,
            start,
            goal,
            blocked,
        )


    def _navigate_to_tile(self, ram: np.ndarray, goal: Tuple[int, int]) -> Optional[np.ndarray]:
        if self._navigator.current_tile == goal or self._navigator.at_tile(goal):
            self._nav_failures = 0
            return self._navigator.center_on_tile(goal, tolerance=1)

        cow_tiles = self._cow_tiles(ram)
        cow_tiles.discard(self._navigator.current_tile)
        cow_tiles.discard(goal)
        if self._navigator.path and self._navigator.path[0] in cow_tiles:
            self._navigator.path = []
            return make_action()

        if self._navigator.stasis > 90 and self._navigator.path:
            self._pathfinder.temp_blocked.add(self._navigator.path[0])
            self._navigator.path = []

        if not self._navigator.path:
            path = self._find_path_around_cows(ram, self._navigator.current_tile, goal)
            if path is None:
                self._nav_failures += 1
                if self._nav_failures > MAX_NAV_FALLBACK_FRAMES:
                    return make_action()
                return fallback_action(self._navigator.current_tile, goal)
            self._nav_failures = 0
            self._navigator.path = path

        action = self._navigator.follow_path(ram)
        if action is None:
            self._nav_failures += 1
            if self._nav_failures > MAX_NAV_FALLBACK_FRAMES:
                return make_action()
            return fallback_action(self._navigator.current_tile, goal)
        self._nav_failures = 0
        return action


    def _defer_pending_slot(
        self,
        slots: list[int],
        counts: dict[int, int],
        slot: int,
        *,
        max_deferrals: int,
    ) -> bool:
        count = counts.get(slot, 0)
        if count >= max_deferrals:
            return False
        counts[slot] = count + 1
        if slot in slots:
            slots.remove(slot)
        slots.append(slot)
        return True


    def _defer_current_care(self, ram: np.ndarray, reason: str) -> bool:
        slot = self._target_cow_slot
        if slot is None or not self._slot_needs_care(ram, slot):
            return False
        if not self._defer_pending_slot(
            self._care_slots,
            self._deferred_care_counts,
            slot,
            max_deferrals=MAX_CARE_DEFERRALS,
        ):
            return False
        print(
            f"[COW] Care deferred slot={slot} reason={reason} "
            f"count={self._deferred_care_counts[slot]}"
        )
        return True


    def _skip_current_cow_care(self, ram: np.ndarray, reason: str) -> TaskResult:
        slot = self._target_cow_slot
        retryable = reason in {"slot_timeout", "nav_unreachable", "pixel_nav_stall"}
        self._pixel_nav_stall_count = 0
        self._reset_pixel_nav_progress()
        if retryable and self._phase in MILK_CARE_PHASES:
            if self._defer_current_milk(ram, reason):
                self._verify_count = 0
                self._interaction_started = False
                self._clear_navigation()
                return self._after_milk(ram)
        if retryable and self._phase not in MILK_CARE_PHASES:
            if self._defer_current_care(ram, reason):
                self._verify_count = 0
                self._interaction_started = False
                self._clear_navigation()
                if self._begin_next_cow_care(ram):
                    return TaskResult(status=TaskStatus.RUNNING)
                return self._after_milk(ram)
        if slot is not None:
            if self._slot_needs_talk(ram, slot):
                self._skipped_talk_slots.add(slot)
            if self._slot_needs_brush(ram, slot):
                self._skipped_brush_slots.add(slot)
            if self._slot_needs_milk(ram, slot):
                self._skipped_milk_slots.add(slot)
            print(f"[COW] Care skipped slot={slot} reason={reason} {self._care_debug_context(ram)}")
        self._verify_count = 0
        self._interaction_started = False
        self._clear_navigation()
        if self._begin_next_cow_care(ram):
            return TaskResult(status=TaskStatus.RUNNING)
        return self._after_milk(ram)



    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        self._navigator.update(world.ram)

        if self._step_count > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason="cow chores timeout")

        tilemap = int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0
        if tilemap != BARN_TILEMAP:
            return TaskResult(status=TaskStatus.BLOCKED, reason=f"not in barn tilemap=0x{tilemap:02X}")

        if (
            self._phase in CARE_PHASES
            and self._target_cow_slot is not None
            and self._care_slot_started_step > 0
        ):
            slot_limit = (
                MAX_COW_SLOT_MILK_FRAMES
                if self._phase in MILK_CARE_PHASES
                else MAX_COW_SLOT_CARE_FRAMES
            )
            if self._step_count - self._care_slot_started_step > slot_limit:
                return self._skip_current_cow_care(world.ram, "slot_timeout")
        if (
            self._phase in CARE_PHASES
            and self._target_cow_slot is not None
            and self._nav_failures > MAX_COW_NAV_FAILURES
        ):
            return self._skip_current_cow_care(world.ram, "nav_unreachable")

        if self._phase == CowPhase.BRUSH_VERIFY:
            # The cow interaction flag can be visible during the tool animation
            # before the queued Y/cooldown frames drain, so sample it first.
            self._mark_brushed_if_changed(world.ram)
        if self._phase == CowPhase.MILK_VERIFY:
            self._mark_milked_if_changed(world.ram)
        if self._action_queue:
            input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
            if self._phase == CowPhase.BRUSH_VERIFY and self.brushed and input_lock == 1:
                self._action_queue.clear()
            elif self._phase == CowPhase.MILK_VERIFY:
                milk_done = self._target_cow_slot is None or not cow_needs_milking(
                    world.ram,
                    self._target_cow_slot,
                )
                if milk_done and read_held_item(world.ram) and input_lock == 1:
                    self._action_queue.clear()

        action = self._action_queue.popleft() if self._action_queue else None
        if action is not None:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

        if self._phase == CowPhase.DONE:
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=(
                    f"talk={self.talked} brush={self.brushed} "
                    f"milk={self.milked_count} ship={self.milk_shipped_count} fed={self.fed_count}"
                ),
            )

        handler = {
            CowPhase.TALK_NAV: self._step_talk_nav,
            CowPhase.TALK_VERIFY: self._step_talk_verify,
            CowPhase.BRUSH_SELECT: self._step_brush_select,
            CowPhase.BRUSH_NAV: self._step_brush_nav,
            CowPhase.BRUSH_VERIFY: self._step_brush_verify,
            CowPhase.MILK_SELECT: self._step_milk_select,
            CowPhase.MILK_NAV: self._step_milk_nav,
            CowPhase.MILK_VERIFY: self._step_milk_verify,
            CowPhase.MILK_SHIP_NAV: self._step_milk_ship_nav,
            CowPhase.MILK_SHIP_VERIFY: self._step_milk_ship_verify,
            CowPhase.FODDER_NAV: self._step_fodder_nav,
            CowPhase.FODDER_VERIFY: self._step_fodder_verify,
            CowPhase.FEED_PLACE_NAV: self._step_feed_place_nav,
            CowPhase.FEED_VERIFY: self._step_feed_verify,
            CowPhase.EXIT_PREP_NAV: self._step_exit_prep_nav,
        }.get(self._phase if isinstance(self._phase, CowPhase) else CowPhase(self._phase))
        if handler is not None:
            return handler(world)
        return TaskResult(status=TaskStatus.FAILURE, reason=f"unknown phase {self._phase}")

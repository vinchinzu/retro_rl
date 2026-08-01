"""Autonomous barn cow chores.

Current scope follows recorded barn chores: milk ready cows and ship milk in
the barn bin, talk to and brush each cow when tools are available, then place
fodder in the trough.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import numpy as np

from harvest.core.animal_probe import BARN_TILEMAP, cow_slot_snapshots, cow_tiles_from_slots
from harvest.core.animal_status import (
    ADDR_FED_COWS_N,
    ADDR_HELD_ITEM,
    ADDR_NUM_COWS,
    ADDR_STORED_GRASS,
    COW_DAILY_BRUSHED_FLAG,
    COW_DAILY_MILKED_FLAG,
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
from harvest.tasks.farm_clearer import (
    ADDR_INPUT_LOCK,
    ADDR_TILEMAP,
    MAP_WIDTH,
    Navigator,
    Pathfinder,
    TileScanner,
    Tool,
    make_action,
)
from harvest.maps.map_config import find_landmark
from harvest.core.npc_catalog import game_objects
from harvest.core.ram_catalog import field_spec, read_ram_u8
from harvest.tasks.animal_navigation import align_to_pixel, fallback_action, find_path_around_blockers
from harvest.tasks.harvest_task import read_shipping_money
from harvest.tasks.primitives import press_a_sequence
from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState


_COW_TALK = find_landmark("cow_talk_stand", tilemap_id=BARN_TILEMAP)
_FODDER = find_landmark("fodder_dispenser", tilemap_id=BARN_TILEMAP)
_TROUGH = find_landmark("cow_feed_trough", tilemap_id=BARN_TILEMAP)
_BARN_BIN = find_landmark("barn_shipping_bin", tilemap_id=BARN_TILEMAP)

COW_TALK_STAND: Tuple[int, int] = _COW_TALK[1].tile if _COW_TALK else (10, 17)
COW_TALK_FACE: str = _COW_TALK[1].face if _COW_TALK and _COW_TALK[1].face else "left"
COW_TALK_ROUTE: Tuple[Tuple[int, int], ...] = ((11, 21), COW_TALK_STAND)
COW_TALK_ANCHOR: Tuple[int, int] = COW_TALK_ROUTE[0]
COW_BAD_INTERACT_STANDS: set[Tuple[int, int]] = {(9, 17), (10, 16), (10, 18), (13, 18)}
_FACE_VECTORS: dict[str, Tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}
FODDER_STAND: Tuple[int, int] = _FODDER[1].tile if _FODDER else (13, 11)
FODDER_FACE: str = _FODDER[1].face if _FODDER and _FODDER[1].face else "right"
FODDER_ROUTE: Tuple[Tuple[int, int], ...] = ((11, 11), FODDER_STAND)
FODDER_TROUGH_ROUTE: Tuple[Tuple[int, int], ...] = ((9, 11), (11, 11), FODDER_STAND)
LEFT_TROUGH_RETURN_X = 130
LEFT_TROUGH_LANE_Y = FODDER_STAND[1] * 16 + 8
ADDR_TOOL_SELECTED = field_spec("tool_selected").address
ADDR_TOOL_BACKPACK = field_spec("tool_backpack").address
ADDR_PLAYER_ACTION = field_spec("player_action").address
BRUSH_TOOL_ID = int(Tool.BRUSH)
MILKER_TOOL_ID = int(Tool.MILKER)
MAX_BRUSH_ATTEMPTS = 3
MAX_TALK_ATTEMPTS = 3
MAX_MILK_ATTEMPTS = 3
MAX_COW_SLOT_CARE_FRAMES = 480
# Keep milk attempts well under the external 3600-frame stall watchdog.
# Pixel-lane nav can reset tile stasis while making zero net progress.
MAX_COW_SLOT_MILK_FRAMES = 720
MAX_NAV_FALLBACK_FRAMES = 12
MAX_COW_NAV_FAILURES = 45
MAX_CARE_DEFERRALS = 1
MAX_MILK_DEFERRALS = 2
PIXEL_NAV_STALL_FRAMES = 120
MAX_PIXEL_NAV_STALLS = 2
MAX_EXIT_PREP_FRAMES = 480
# Hand EXIT_BARN a known-good lower-aisle staging tile instead of finishing
# among dynamic cow stalls (long-run EXIT_BARN timeouts at ~tile (13,17)).
COW_EXIT_PREP_STAND: Tuple[int, int] = COW_TALK_ANCHOR
COW_EXIT_PREP_PX: Tuple[int, int] = (
    COW_EXIT_PREP_STAND[0] * 16 + 8,
    COW_EXIT_PREP_STAND[1] * 16 + 8,
)
COW_INTERACT_X_OFFSET = 13
COW_INTERACT_Y_OFFSET = 3
COW_LEFT_INTERACT_X = 163
LEFT_COW_VERTICAL_LANE_X = 38
LEFT_COW_LOWER_LANE_X = 55
LEFT_COW_LANE_SWITCH_Y = 315
COW_UPPER_RIGHT_ROUTE_MAX_Y = 100
COW_RIGHT_AISLE_X = 204
UPPER_BARN_SHIP_ESCAPE_Y = 184
UPPER_BARN_RIGHT_AISLE_X = 216
UPPER_BARN_SHIP_CROSS_Y = 217
UPPER_BARN_SHIP_AISLE_X = 205
UPPER_BARN_SHIP_LOWER_LANE_Y = 315
LEFT_BARN_SHIP_LANE_X = 55
LEFT_BARN_SHIP_LOWER_Y = 346
BARN_SHIP_BIN_STAND: Tuple[int, int] = _BARN_BIN[1].tile if _BARN_BIN else (2, 22)
BARN_SHIP_BIN_INTERACT_STAND: Tuple[int, int] = BARN_SHIP_BIN_STAND
BARN_SHIP_BIN_FACE = _BARN_BIN[1].face if _BARN_BIN and _BARN_BIN[1].face else "left"
MILK_SHIP_ROUTE: Tuple[Tuple[int, int], ...] = ((11, 21), (5, 22), BARN_SHIP_BIN_STAND)
MILK_SHIP_PIXEL_ROUTE: Tuple[Tuple[int, int], ...] = (
    (183, 328),
    (139, 328),
    (139, 346),
    (55, 346),
    (55, 358),
    (38, 361),
)

@dataclass(frozen=True)
class CowFeedSpot:
    stand: Tuple[int, int]
    face: str
    interact_px: Tuple[int, int]
    flag: int


# Feed trough coordinates decoded from the barn replacement table
# (DATA16_81B0ED) and Cow_Feed_Flags in the decomp.
COW_FEED_SPOTS: Tuple[CowFeedSpot, ...] = (
    CowFeedSpot((7, 9), "right", (113, 149), 0x0008),
    CowFeedSpot((7, 13), "right", (113, 213), 0x0004),
    CowFeedSpot((7, 15), "right", (113, 245), 0x0002),
    CowFeedSpot((7, 17), "right", (113, 277), 0x0001),
    CowFeedSpot((7, 7), "right", (113, 117), 0x0010),
    CowFeedSpot((7, 5), "right", (113, 85), 0x0020),
    CowFeedSpot((6, 17), "left", (111, 277), 0x0040),
    CowFeedSpot((6, 15), "left", (111, 245), 0x0080),
    CowFeedSpot((6, 13), "left", (111, 213), 0x0100),
    CowFeedSpot((6, 9), "left", (111, 149), 0x0200),
    CowFeedSpot((6, 7), "left", (111, 117), 0x0400),
    CowFeedSpot((6, 5), "left", (111, 85), 0x0800),
)
CARE_TROUGH_EXIT_X = COW_FEED_SPOTS[0].interact_px[0]
CARE_TROUGH_EXIT_MIN_Y = COW_FEED_SPOTS[0].interact_px[1] - 8
CARE_TROUGH_EXIT_ANCHOR_X = COW_TALK_ANCHOR[0] * 16 + 8
CARE_TROUGH_EXIT_BOTTOM_Y = COW_TALK_ANCHOR[1] * 16 + 8
FEED_TROUGH_STAND: Tuple[int, int] = COW_FEED_SPOTS[0].stand if not _TROUGH else _TROUGH[1].tile
FEED_TROUGH_FACE: str = COW_FEED_SPOTS[0].face if not _TROUGH else (_TROUGH[1].face or COW_FEED_SPOTS[0].face)
FEED_TROUGH_ROUTE: Tuple[Tuple[int, int], ...] = ((9, 11), FEED_TROUGH_STAND)
FEED_TROUGH_INTERACT_PX: Tuple[int, int] = COW_FEED_SPOTS[0].interact_px

@dataclass
class CowChoresTask(Task):
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
    _phase: str = field(default="talk_nav", init=False)
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
            self._phase = "done"
        elif self.milk and self._milker_in_carry_pair(world.ram) and self._begin_next_milk(world.ram):
            pass
        elif self.feed and self._feed_remaining > 0 and self._grass_before > 0:
            self._phase = "fodder_nav" if read_held_item(world.ram) != ITEM_FODDER else "feed_place_nav"
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

    @property
    def progress_text(self) -> str:
        return (
            f"talk={'Y' if self.talked else 'N'} "
            f"brush={'Y' if self.brushed else 'N'} "
            f"milk={self.milked_count}/{len(self._milk_slots)} "
            f"fed={self.fed_count}/{self._feed_goal_count or self._cow_count}"
        )

    def _facing_tile(self, stand: Tuple[int, int], face: str) -> Tuple[int, int]:
        dx, dy = _FACE_VECTORS.get(face, (-1, 0))
        return stand[0] + dx, stand[1] + dy

    def _select_target_cow_slot(self, ram: np.ndarray) -> Optional[int]:
        rows = cow_slot_snapshots(ram, require_barn=True)
        if not rows:
            slots = existing_cow_slots(ram)
            return slots[0] if slots else None

        target_tile = self._facing_tile(COW_TALK_STAND, COW_TALK_FACE)
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
        flags = read_fed_cows_flags(ram)
        return min(
            self._current_feed_goal(ram),
            sum(1 for spot in COW_FEED_SPOTS if flags & spot.flag),
        )

    def _fed_count_now(self, ram: np.ndarray) -> int:
        return max(read_fed_cows_n(ram), self._fed_trough_count(ram))

    def _next_feed_spot(self, ram: np.ndarray) -> CowFeedSpot:
        flags = read_fed_cows_flags(ram)
        goal = self._current_feed_goal(ram)
        for spot in COW_FEED_SPOTS[:goal]:
            if not (flags & spot.flag):
                return spot
        return COW_FEED_SPOTS[max(0, goal - 1)]

    def _feed_route(self, spot: CowFeedSpot) -> Tuple[Tuple[int, int], ...]:
        if spot.stand[0] <= 7:
            return ((9, 11), spot.stand)
        return ((11, 11), spot.stand)

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
        return tile[0], tile[1] + 1

    def _is_adjacent_to_target_cow(self, ram: np.ndarray, stand: Tuple[int, int], face: str) -> bool:
        tile = self._target_cow_tile(ram)
        if tile is None:
            return False
        facing = self._facing_tile(stand, face)
        if facing == tile:
            return True
        body = (tile[0], tile[1] + 1)
        # Horizontal sides and below-body (face up) are valid talk/brush pins.
        return facing == body and face in ("left", "right", "up")

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
        tile = self._target_cow_tile(ram)
        if tile is not None:
            dx = tile[0] - stand[0]
            dy = tile[1] - stand[1]
            body = self._target_cow_body_tile(ram)
            body_dx = body[0] - stand[0] if body is not None else 0
            body_dy = body[1] - stand[1] if body is not None else 0
            if abs(dx) + abs(dy) != 1 and not (body is not None and body_dy == 0 and abs(body_dx) == 1):
                return self._talk_face if stand == self._talk_stand else COW_TALK_FACE
            if dx > 0:
                return "right"
            if dx < 0:
                return "left"
            if body is not None and body_dy == 0:
                if body_dx > 0:
                    return "right"
                if body_dx < 0:
                    return "left"
            if dy > 0:
                return "down"
            return "up"
        return COW_TALK_FACE

    def _cow_interact_pixel(self, ram: np.ndarray, *, tool: bool) -> Optional[Tuple[int, int]]:
        pixel = self._target_cow_pixel(ram)
        if pixel is None:
            return None
        px, py = pixel
        if self._talk_face == "left":
            # Recorded left-aisle clamp (x=163) for left/center stall cows.
            # Right-side cows (tile x >= 12) must keep px+offset or pixel nav
            # aims through the cow toward the wrong aisle.
            target_x = px + COW_INTERACT_X_OFFSET
            tile = self._target_cow_tile(ram)
            if tile is None or tile[0] <= 10:
                target_x = min(target_x, COW_LEFT_INTERACT_X)
            if tile is not None:
                if tile[1] == 17 and tool:
                    return target_x, 278
                if tile[1] == 15:
                    return target_x, 249
            return target_x, py
        if self._talk_face == "right":
            return px - COW_INTERACT_X_OFFSET, py + COW_INTERACT_Y_OFFSET
        if self._talk_face == "up":
            return px, py + COW_INTERACT_X_OFFSET
        if self._talk_face == "down":
            return px, py - COW_INTERACT_X_OFFSET
        return None

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
        if cx <= 4:
            # Wall-side cows: stay on the right/body column. Same-x stands
            # (face up/down) trap the player on the cow's tile column.
            preferred.extend(
                [
                    ((cx + 1, cy + 1), "left"),
                    ((cx + 1, cy), "left"),
                    ((cx - 1, cy), "right"),
                    ((cx - 1, cy + 1), "right"),
                ]
            )
        elif cx <= 10:
            preferred.append(((cx + 1, cy), "left"))
            preferred.append(((cx + 1, cy + 1), "left"))
        elif cx >= 12:
            preferred.append(((cx - 1, cy), "right"))
            preferred.append(((cx - 1, cy + 1), "right"))
            # Prefer staying on the right aisle for right-side cows instead of
            # flipping to a left-face stand that pixel-nav used to mis-aim.
            preferred.append(((cx + 1, cy), "left"))
            preferred.append(((cx + 1, cy + 1), "left"))
        else:
            preferred.extend(
                [
                    ((cx + 1, cy), "left"),
                    ((cx - 1, cy), "right"),
                    ((cx + 1, cy + 1), "left"),
                    ((cx - 1, cy + 1), "right"),
                ]
            )

        preferred.extend(
            [
                ((cx, cy + 1), "up"),
                ((cx, cy - 1), "down"),
                ((cx + 1, cy + 1), "left"),
                ((cx - 1, cy + 1), "right"),
                ((cx + 1, cy), "left"),
                ((cx - 1, cy), "right"),
            ]
        )

        candidates: list[Tuple[Tuple[int, int], str]] = []
        scored: list[Tuple[Tuple[int, int, int], Tuple[Tuple[int, int], str]]] = []
        seen: set[Tuple[int, int]] = set()
        cow_tiles = self._cow_tiles(ram)
        for index, (stand, face) in enumerate(preferred):
            sx, sy = stand
            if stand in seen:
                continue
            seen.add(stand)
            if not (0 <= sx < MAP_WIDTH and 0 <= sy < MAP_WIDTH):
                continue
            if stand in cow_tiles or stand in COW_BAD_INTERACT_STANDS:
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
            if not (0 <= sx < MAP_WIDTH and 0 <= sy < MAP_WIDTH):
                continue
            if stand in cow_tiles or stand in COW_BAD_INTERACT_STANDS:
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
        if cx >= 12:
            return [((cx - 1, cy), "right"), ((cx - 1, cy + 1), "right")]
        if cx <= 10:
            return [((cx + 1, cy), "left"), ((cx + 1, cy + 1), "left")]
        if COW_TALK_STAND not in cow_tiles and COW_TALK_STAND not in COW_BAD_INTERACT_STANDS:
            return [(COW_TALK_STAND, COW_TALK_FACE)]
        current = self._navigator.current_tile
        return [(current, self._face_for_target_cow(ram, current))]

    def _cow_escape_blocked(
        self,
        ram: np.ndarray,
        cow_tile: Tuple[int, int],
        stand: Tuple[int, int],
        face: str,
        cow_tiles: set[Tuple[int, int]],
    ) -> bool:
        dx, dy = _FACE_VECTORS.get(face, (0, 0))
        if (stand[0] + dx, stand[1] + dy) != cow_tile:
            return False
        escape = cow_tile[0] + dx, cow_tile[1] + dy
        if not (0 <= escape[0] < MAP_WIDTH and 0 <= escape[1] < MAP_WIDTH):
            return True
        other_cow_tiles = set(cow_tiles)
        other_cow_tiles.discard(cow_tile)
        other_cow_tiles.discard((cow_tile[0], cow_tile[1] + 1))
        if escape in other_cow_tiles:
            return True
        return not self._pathfinder.is_walkable(ram, escape[0], escape[1], current_pos=self._navigator.current_tile)

    def _refresh_talk_approach(self, ram: np.ndarray) -> None:
        stand, face = self._candidate_cow_stands(ram)[0]
        if stand != self._talk_stand:
            self._clear_navigation()
        self._talk_stand = stand
        self._talk_face = face

    def _talk_route(self) -> Tuple[Tuple[int, int], ...]:
        return (COW_TALK_ANCHOR, self._talk_stand)

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

    def _mark_brushed_if_changed(self, ram: np.ndarray) -> None:
        if self.brushed:
            return
        if self._cow_ram_changed(
            ram,
            COW_DAILY_BRUSHED_FLAG,
            self._brush_flags_before,
            self._brush_happiness_before,
        ):
            print(f"[COW] Brush OK slot={self._target_cow_slot} attempts={self._brush_attempts}")
            self.brushed = True
            self._remember_current_pin()

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

    def _begin_exit_prep(self) -> None:
        self._exit_prep_started_step = self._step_count
        self._verify_count = 0
        self._clear_navigation()
        self._reset_pixel_nav_progress()
        self._phase = "exit_prep_nav"

    def _exit_prep_escape_action(self) -> Optional[np.ndarray]:
        """Pixel route out of left/upper dead-ends toward the lower aisle."""
        x = self._navigator.current_pos.x
        y = self._navigator.current_pos.y
        tx, ty = COW_EXIT_PREP_PX
        if abs(x - tx) <= 3 and abs(y - ty) <= 3:
            return None
        # Upper-left stalls are a dead-end: go south before trying to cross east.
        if x < 120:
            if y < UPPER_BARN_SHIP_LOWER_LANE_Y - 2:
                return make_action(down=True, b=True)
            lane_x = self._left_cow_lane_x(y)
            if abs(x - lane_x) > 2:
                return make_action(right=x < lane_x, left=x > lane_x, b=True)
            if abs(y - ty) > 3:
                return make_action(down=y < ty, up=y > ty, b=True)
            return make_action(right=True, b=True)
        if y < UPPER_BARN_SHIP_LOWER_LANE_Y - 2:
            if abs(x - COW_RIGHT_AISLE_X) > 3:
                return make_action(
                    right=x < COW_RIGHT_AISLE_X,
                    left=x > COW_RIGHT_AISLE_X,
                    b=True,
                )
            return make_action(down=True, b=True)
        if abs(y - ty) > 3:
            return make_action(down=y < ty, up=y > ty, b=True)
        if abs(x - tx) > 3:
            return make_action(right=x < tx, left=x > tx, b=True)
        return None

    def _care_debug_context(self, ram: np.ndarray) -> str:
        tool = self._phase in {"brush_nav", "brush_verify", "milk_nav", "milk_verify"}
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
        dx = target[0] - self._navigator.current_pos.x
        dy = target[1] - self._navigator.current_pos.y
        if abs(dx) <= tolerance and abs(dy) <= tolerance:
            return None
        if x_first and abs(dx) > tolerance:
            return make_action(right=dx > 0, left=dx < 0, b=True)
        if y_first and abs(dy) > tolerance:
            return make_action(down=dy > 0, up=dy < 0, b=True)
        if abs(dx) >= abs(dy) and abs(dx) > tolerance:
            return make_action(right=dx > 0, left=dx < 0, b=True)
        return make_action(down=dy > 0, up=dy < 0, b=True)

    def _left_cow_lane_x(self, current_y: int) -> int:
        if current_y > LEFT_COW_LANE_SWITCH_Y:
            return LEFT_COW_LOWER_LANE_X
        return LEFT_COW_VERTICAL_LANE_X

    def _left_lower_lane_from_right_action(self) -> Optional[np.ndarray]:
        x = self._navigator.current_pos.x
        y = self._navigator.current_pos.y
        if x <= LEFT_COW_LOWER_LANE_X + 2 and y >= LEFT_BARN_SHIP_LOWER_Y - 2:
            return None
        if y < UPPER_BARN_SHIP_ESCAPE_Y - 2:
            if x >= 120 and x < COW_RIGHT_AISLE_X - 2:
                return make_action(right=True, b=True)
            return make_action(down=True, b=True)
        if y <= UPPER_BARN_SHIP_ESCAPE_Y + 2 and x < UPPER_BARN_RIGHT_AISLE_X - 2:
            return make_action(right=True, b=True)
        if x >= UPPER_BARN_RIGHT_AISLE_X - 3 and y < UPPER_BARN_SHIP_CROSS_Y - 2:
            return make_action(down=True, b=True)
        if abs(y - UPPER_BARN_SHIP_CROSS_Y) <= 2 and x > UPPER_BARN_SHIP_AISLE_X:
            return make_action(left=True, b=True)
        if abs(y - UPPER_BARN_SHIP_LOWER_LANE_Y) <= 2 and x > MILK_SHIP_PIXEL_ROUTE[0][0]:
            return make_action(left=True, b=True)
        if x >= MILK_SHIP_PIXEL_ROUTE[0][0] + 6 and y < UPPER_BARN_SHIP_LOWER_LANE_Y - 2:
            return make_action(down=True, b=True)

        route = MILK_SHIP_PIXEL_ROUTE[:4]
        start_index = 0
        if y >= route[0][1] - 1 and x <= route[0][0] + 1:
            start_index = 1
        if y >= route[1][1] - 1 and x <= route[1][0] + 1:
            start_index = 2
        if y >= route[2][1] - 1 and x <= route[2][0] + 1:
            start_index = 3
        for index, target in enumerate(route[start_index:], start=start_index):
            if abs(x - target[0]) <= 1 and abs(y - target[1]) <= 1:
                continue
            if index in (0, 1, 3):
                return self._run_to_pixel_axis(target, x_first=True)
            return self._run_to_pixel_axis(target, y_first=True)
        return None

    def _left_side_vertical_nav_action(
        self,
        x: int,
        y: int,
        tx: int,
        ty: int,
        *,
        going_down: bool,
    ) -> Optional[np.ndarray]:
        """Reach wall-side interact pixels via the recorded left vertical lane.

        Climb/descend on lane x first while far from the target row. Only settle
        onto the interact column (~27) near the target Y — cutting left early at
        the lower corridor (x=54,y=345) dead-ends against the bottom wall.
        """
        if abs(x - tx) <= 2:
            if abs(y - ty) <= 1:
                return None
            return make_action(down=going_down, up=not going_down, b=abs(y - ty) > 8)
        if x > 120:
            action = self._left_lower_lane_from_right_action()
            if action is not None:
                return action
        if abs(y - ty) > 16:
            lane_x = self._left_cow_lane_x(y)
            if abs(x - lane_x) > 1:
                return make_action(
                    right=x < lane_x,
                    left=x > lane_x,
                    b=abs(x - lane_x) > 8,
                )
            return make_action(down=going_down, up=not going_down, b=True)
        if abs(x - tx) > 1:
            return make_action(right=x < tx, left=x > tx, b=abs(x - tx) > 8)
        return make_action(down=going_down, up=not going_down, b=True)

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

        # The barn stall rows are much faster and more reliable when entered
        # through the recorded right-side lane instead of BFS-chasing moving cows.
        if self._talk_face == "left" and y < ty - 10:
            if tx >= 120:
                if ty <= COW_UPPER_RIGHT_ROUTE_MAX_Y and abs(x - COW_RIGHT_AISLE_X) > 2:
                    return make_action(right=x < COW_RIGHT_AISLE_X, left=x > COW_RIGHT_AISLE_X, b=True)
                if x < 192:
                    return make_action(right=True, b=True)
                if y < ty:
                    return make_action(down=True, b=True)
            else:
                return self._left_side_vertical_nav_action(x, y, tx, ty, going_down=True)

        if self._talk_face == "right" and y < ty - 10:
            if tx < 100:
                return self._left_side_vertical_nav_action(x, y, tx, ty, going_down=True)
            if x > 96:
                return make_action(left=True, b=True)
            if y < ty:
                return make_action(down=True, b=True)

        if self._talk_face == "left" and y > ty + 10:
            if tx < 100:
                return self._left_side_vertical_nav_action(x, y, tx, ty, going_down=False)
            if ty <= COW_UPPER_RIGHT_ROUTE_MAX_Y:
                if abs(x - COW_RIGHT_AISLE_X) > 2:
                    return make_action(right=x < COW_RIGHT_AISLE_X, left=x > COW_RIGHT_AISLE_X, b=True)
                return make_action(up=True, b=True)
            if ty <= 255:
                if x < 102:
                    return make_action(right=True, b=True)
                if y > 342:
                    return make_action(up=True, b=True)
                if x < 174:
                    return make_action(right=True, b=True)
                if y > 315:
                    return make_action(up=True, b=True)
                if x < 192:
                    return make_action(right=True, b=True)
            else:
                if x < 160 and y > 330:
                    return make_action(right=True, b=True)
                if y > 326:
                    return make_action(up=True, b=True)
                if x < 192:
                    return make_action(right=True, b=True)
            return make_action(up=True, b=True)

        if self._talk_face == "right" and y > ty + 10:
            if tx < 100:
                return self._left_side_vertical_nav_action(x, y, tx, ty, going_down=False)
            if x < 102:
                return make_action(right=True, b=True)
            if y > 342:
                return make_action(up=True, b=True)
            if x < min(tx, 174):
                return make_action(right=True, b=True)
            return make_action(up=True, b=True)

        if abs(x - tx) > 1:
            return make_action(right=x < tx, left=x > tx, b=abs(x - tx) > 8)
        if abs(y - ty) > 1:
            return make_action(down=y < ty, up=y > ty, b=abs(y - ty) > 8)
        return None

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

    def _milk_ship_pixel_action(self) -> Optional[np.ndarray]:
        x = self._navigator.current_pos.x
        y = self._navigator.current_pos.y
        if x < 120 and y < LEFT_BARN_SHIP_LOWER_Y - 2:
            if y >= UPPER_BARN_SHIP_LOWER_LANE_Y and x < LEFT_BARN_SHIP_LANE_X - 2:
                return make_action(right=True, b=True)
            return make_action(down=True, b=True)
        if x < 120 and x > LEFT_BARN_SHIP_LANE_X + 2:
            return make_action(left=True, b=True)
        if x < 120 and y < MILK_SHIP_PIXEL_ROUTE[-1][1] - 2:
            return make_action(down=True, b=True)
        if x < 120 and abs(x - MILK_SHIP_PIXEL_ROUTE[-1][0]) > 2:
            return make_action(right=x < MILK_SHIP_PIXEL_ROUTE[-1][0], left=x > MILK_SHIP_PIXEL_ROUTE[-1][0], b=True)
        if y < UPPER_BARN_SHIP_ESCAPE_Y - 2:
            if x >= 120 and x < COW_RIGHT_AISLE_X - 2:
                return make_action(right=True, b=True)
            return make_action(down=True, b=True)
        if y <= UPPER_BARN_SHIP_ESCAPE_Y + 2 and x < UPPER_BARN_RIGHT_AISLE_X - 2:
            return make_action(right=True, b=True)
        if self._ship_route_index == 0 and x >= UPPER_BARN_RIGHT_AISLE_X - 3 and y < UPPER_BARN_SHIP_CROSS_Y - 2:
            return make_action(down=True, b=True)
        if (
            self._ship_route_index == 0
            and abs(y - UPPER_BARN_SHIP_CROSS_Y) <= 2
            and x > UPPER_BARN_SHIP_AISLE_X
        ):
            return make_action(left=True, b=True)
        if (
            self._ship_route_index == 0
            and abs(y - UPPER_BARN_SHIP_LOWER_LANE_Y) <= 2
            and x > MILK_SHIP_PIXEL_ROUTE[0][0]
        ):
            return make_action(left=True, b=True)
        if self._ship_route_index == 0 and x >= MILK_SHIP_PIXEL_ROUTE[0][0] + 6 and y < UPPER_BARN_SHIP_LOWER_LANE_Y - 2:
            return make_action(down=True, b=True)

        index = min(self._ship_route_index, len(MILK_SHIP_PIXEL_ROUTE) - 1)
        target = MILK_SHIP_PIXEL_ROUTE[index]
        if abs(x - target[0]) <= 2 and abs(y - target[1]) <= 2:
            if self._ship_route_index < len(MILK_SHIP_PIXEL_ROUTE) - 1:
                self._ship_route_index += 1
                return make_action()
            return None

        if index == 0:
            return self._run_to_pixel_axis(target, x_first=True)
        if index in (1, 3, 5):
            return self._run_to_pixel_axis(target, x_first=True)
        return self._run_to_pixel_axis(target, y_first=True)

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

    def _fodder_route(self) -> Tuple[Tuple[int, int], ...]:
        tx, ty = self._navigator.current_tile
        if tx == 10 and 13 <= ty <= 18:
            return ((11, ty),) + FODDER_ROUTE
        if (ty >= 18 and tx <= 9) or (ty >= 13 and tx <= 8):
            return FODDER_TROUGH_ROUTE
        return FODDER_ROUTE

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
        candidates: list[Tuple[Tuple[int, int], str]] = []
        if cx <= 4:
            candidates.extend(
                [
                    ((cx + 1, cy + 1), "left"),
                    ((cx + 1, cy), "left"),
                ]
            )
        elif cx <= 10:
            candidates.extend([((cx + 1, cy + 1), "left"), ((cx + 1, cy), "left")])
        elif cx >= 12:
            candidates.extend([((cx - 1, cy + 1), "right"), ((cx - 1, cy), "right")])
        else:
            candidates.extend(
                [
                    ((cx + 1, cy + 1), "left"),
                    ((cx - 1, cy + 1), "right"),
                    ((cx + 1, cy), "left"),
                    ((cx - 1, cy), "right"),
                ]
            )

        cow_tiles = self._cow_tiles(ram)
        for stand, face in candidates:
            sx, sy = stand
            if not (0 <= sx < MAP_WIDTH and 0 <= sy < MAP_WIDTH):
                continue
            if stand in cow_tiles or stand in COW_BAD_INTERACT_STANDS:
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

    def _left_feed_spot_action(self, spot: CowFeedSpot) -> Optional[np.ndarray]:
        if not (spot.stand[0] <= 7):
            return None
        target_x, target_y = spot.interact_px
        current_x = self._navigator.current_pos.x
        current_y = self._navigator.current_pos.y
        if spot.face == "left":
            if current_x > target_x + 2:
                if abs(current_y - LEFT_TROUGH_LANE_Y) > 2:
                    return make_action(
                        up=current_y > LEFT_TROUGH_LANE_Y,
                        down=current_y < LEFT_TROUGH_LANE_Y,
                        b=True,
                    )
                return make_action(left=True, b=True)
            if abs(current_y - target_y) > 2:
                return make_action(up=current_y > target_y, down=current_y < target_y, b=True)
            if abs(current_x - target_x) > 2:
                return make_action(left=current_x > target_x, right=current_x < target_x, b=True)
            return None
        if abs(current_x - target_x) > 2:
            return make_action(left=current_x > target_x, right=current_x < target_x, b=True)
        if abs(current_y - target_y) > 2:
            return make_action(up=current_y > target_y, down=current_y < target_y, b=True)
        return None

    def _left_cow_to_fodder_action(self) -> Optional[np.ndarray]:
        current_x = self._navigator.current_pos.x
        current_y = self._navigator.current_pos.y
        fodder_x = FODDER_STAND[0] * 16 + 8
        if current_x <= 90 and current_y >= 240:
            if current_y < 327:
                if current_y < 300 and current_x > 22:
                    return make_action(left=True, b=True)
                if current_x < 22:
                    return make_action(right=True, b=True)
                return make_action(down=True, b=True)
            return make_action(right=True, b=True)
        if current_y >= 326 and current_x < 239:
            if current_x > 90 and current_y > 335:
                return None
            return make_action(right=True, b=True)
        if current_x >= 239 and current_y > 312:
            return make_action(up=True, b=True)
        if current_y <= 312 and current_y > LEFT_TROUGH_LANE_Y + 2 and current_x > 203:
            return make_action(left=True, b=True)
        if 196 <= current_x <= 205 and current_y > LEFT_TROUGH_LANE_Y + 2:
            return make_action(up=True, b=True)
        if 196 <= current_x <= fodder_x and abs(current_y - LEFT_TROUGH_LANE_Y) <= 2:
            if current_x < fodder_x - 2:
                return make_action(right=True, b=True)
        return None

    def _left_trough_return_action(self) -> Optional[np.ndarray]:
        current_x = self._navigator.current_pos.x
        current_y = self._navigator.current_pos.y
        fodder_x = FODDER_STAND[0] * 16 + 8
        if current_x > LEFT_TROUGH_RETURN_X:
            return None
        if current_y > LEFT_TROUGH_LANE_Y + 2:
            return make_action(up=True, b=True)
        if current_y < LEFT_TROUGH_LANE_Y - 2:
            return make_action(down=True, b=True)
        if current_x < fodder_x - 2:
            return make_action(right=True, b=True)
        return None

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

    def _retry_talk_nav(self, ram: np.ndarray, reason: str) -> Optional[TaskResult]:
        if self._talk_attempts >= MAX_TALK_ATTEMPTS:
            return None
        print(f"[COW] Talk retry slot={self._target_cow_slot} attempts={self._talk_attempts} reason={reason}")
        self._refresh_talk_approach(ram)
        self._talk_route_index = max(0, len(self._talk_route()) - 1)
        self._verify_count = 0
        self._interaction_started = False
        self._clear_navigation()
        self._phase = "talk_nav"
        return TaskResult(status=TaskStatus.RUNNING)

    def _after_talk(self, ram: np.ndarray) -> TaskResult:
        if (
            self._target_cow_slot is not None
            and self._slot_needs_talk(ram, self._target_cow_slot)
            and not self._cow_flag_set_for_slot(ram, self._target_cow_slot, COW_DAILY_TALKED_FLAG)
        ):
            self._skipped_talk_slots.add(self._target_cow_slot)
        if self._target_cow_slot is not None and self._slot_needs_brush(ram, self._target_cow_slot):
            self._refresh_talk_approach(ram)
            self._brush_route_index = max(0, len(self._talk_route()) - 1)
            self._brush_select_frames = 0
            self._brush_attempts = 0
            self._verify_count = 0
            self._interaction_started = False
            self._clear_navigation()
            self._talk_face = self._face_for_target_cow(ram)
            if self._brush_selected(ram) and self._is_adjacent_to_target_cow(ram, self._navigator.current_tile, self._talk_face):
                return self._begin_brush_verify(ram)
            self._phase = "brush_nav" if self._brush_selected(ram) else "brush_select"
            return TaskResult(status=TaskStatus.RUNNING)
        if self._target_cow_slot is not None and self._slot_needs_milk(ram, self._target_cow_slot):
            self._refresh_talk_approach(ram)
            self._brush_route_index = max(0, len(self._talk_route()) - 1)
            self._milk_select_frames = 0
            self._milk_attempts = 0
            self._verify_count = 0
            self._interaction_started = False
            self._clear_navigation()
            self._phase = "milk_nav" if self._milker_selected(ram) else "milk_select"
            return TaskResult(status=TaskStatus.RUNNING)
        if self._begin_next_cow_care(ram):
            return TaskResult(status=TaskStatus.RUNNING)
        return self._after_brush(ram)

    def _begin_brush_verify(self, ram: np.ndarray) -> TaskResult:
        if self._target_cow_slot is None:
            return TaskResult(status=TaskStatus.FAILURE, reason="no target cow slot for brush")
        self._brush_flags_before = read_cow_daily_flags(ram, self._target_cow_slot)
        self._brush_happiness_before = read_cow_happiness(ram, self._target_cow_slot)
        self.brushed = bool(self._brush_flags_before & COW_DAILY_BRUSHED_FLAG)
        self._clear_navigation()
        self._queue_use_tool(
            self._talk_face,
            face_frames=10,
            hold_frames=18,
            y_only_frames=2,
            settle_frames=75,
        )
        self._brush_attempts += 1
        self._verify_count = 0
        self._interaction_started = False
        self._phase = "brush_verify"
        action = self._action_queue.popleft() if self._action_queue else None
        if action is not None:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
        return TaskResult(status=TaskStatus.RUNNING)

    def _begin_next_cow_care(self, ram: np.ndarray) -> bool:
        self._care_slots = [slot for slot in self._care_slots if self._slot_needs_care(ram, slot)]
        if not self._care_slots:
            self._target_cow_slot = self._select_target_cow_slot(ram)
            return False

        self._target_cow_slot = self._care_slots[0]
        self._refresh_talk_approach(ram)
        self._talk_route_index = 0
        self._brush_route_index = 0
        self._talk_attempts = 0
        self._brush_select_frames = 0
        self._brush_attempts = 0
        self._milk_select_frames = 0
        self._milk_attempts = 0
        self._verify_count = 0
        self._interaction_started = False
        self._care_slot_started_step = self._step_count
        self._care_trough_exit_logged = False
        self._pixel_nav_stall_count = 0
        self._reset_pixel_nav_progress()
        self._clear_navigation()
        self._pin_care_route_to_direct_stand(ram)
        current_face = self._face_for_target_cow(ram, self._navigator.current_tile)
        if current_face in ("left", "right") and self._is_adjacent_to_target_cow(
            ram,
            self._navigator.current_tile,
            current_face,
        ):
            self._talk_stand = self._navigator.current_tile
            self._talk_face = current_face
            self._talk_route_index = max(0, len(self._talk_route()) - 1)
            self._brush_route_index = self._talk_route_index

        if self._slot_needs_talk(ram, self._target_cow_slot):
            self.talked = False
            self._phase = "talk_nav"
        elif self._slot_needs_brush(ram, self._target_cow_slot):
            self.brushed = False
            self._phase = "brush_nav" if self._brush_selected(ram) else "brush_select"
        elif self._slot_needs_milk(ram, self._target_cow_slot):
            self._phase = "milk_nav" if self._milker_selected(ram) else "milk_select"
        else:
            self._care_slots.pop(0)
            return self._begin_next_cow_care(ram)
        needs = []
        if self._slot_needs_talk(ram, self._target_cow_slot):
            needs.append("talk")
        if self._slot_needs_brush(ram, self._target_cow_slot):
            needs.append("brush")
        if self._slot_needs_milk(ram, self._target_cow_slot):
            needs.append("milk")
        print(
            f"[COW] Care start slot={self._target_cow_slot} needs={','.join(needs)} "
            f"{self._care_debug_context(ram)}"
        )
        return True

    def _after_brush(self, ram: np.ndarray) -> TaskResult:
        if self._target_cow_slot is not None and self._slot_needs_milk(ram, self._target_cow_slot):
            if self._target_cow_slot in self._skipped_brush_slots:
                self._prefer_body_side_stand(ram)
            self._milk_select_frames = 0
            self._milk_attempts = 0
            self._verify_count = 0
            self._interaction_started = False
            self._phase = "milk_nav" if self._milker_selected(ram) else "milk_select"
            return TaskResult(status=TaskStatus.RUNNING)
        if self._begin_next_cow_care(ram):
            return TaskResult(status=TaskStatus.RUNNING)
        return self._after_milk(ram)

    def _begin_next_milk(self, ram: np.ndarray) -> bool:
        self._milk_slots = [
            slot
            for slot in self._milk_slots
            if slot not in self._skipped_milk_slots and cow_needs_milking(ram, slot)
        ]
        if not self._milk_slots:
            return False
        self._target_cow_slot = self._milk_slots[0]
        self._refresh_talk_approach(ram)
        self._brush_route_index = max(0, len(self._talk_route()) - 1)
        self._pin_care_route_to_direct_stand(ram)
        self._milk_select_frames = 0
        self._milk_attempts = 0
        self._verify_count = 0
        self._interaction_started = False
        self._care_slot_started_step = self._step_count
        self._pixel_nav_stall_count = 0
        self._reset_pixel_nav_progress()
        self._clear_navigation()
        self._phase = "milk_nav" if self._milker_selected(ram) else "milk_select"
        return True

    def _begin_milk_verify(self, ram: np.ndarray) -> TaskResult:
        if self._target_cow_slot is None:
            return TaskResult(status=TaskStatus.FAILURE, reason="no target cow slot for milk")
        self._milk_flags_before = read_cow_daily_flags(ram, self._target_cow_slot)
        self._milk_held_before = read_held_item(ram)
        self._clear_navigation()
        self._queue_use_tool(self._talk_face, face_frames=8, hold_frames=9, y_only_frames=1, settle_frames=85)
        self._milk_attempts += 1
        self._verify_count = 0
        self._interaction_started = False
        self._phase = "milk_verify"
        action = self._action_queue.popleft() if self._action_queue else None
        if action is not None:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
        return TaskResult(status=TaskStatus.RUNNING)

    def _mark_milked_if_changed(self, ram: np.ndarray) -> None:
        if self._target_cow_slot is None:
            return
        flags_now = read_cow_daily_flags(ram, self._target_cow_slot)
        if not (flags_now & COW_DAILY_MILKED_FLAG):
            return
        if self._target_cow_slot in self._milk_slots:
            self._milk_slots.remove(self._target_cow_slot)
        if self._target_cow_slot not in self._milked_slots:
            self._milked_slots.add(self._target_cow_slot)
            self.milked_count += 1
            print(f"[COW] Milk OK slot={self._target_cow_slot} attempts={self._milk_attempts}")

    def _after_milk(self, ram: np.ndarray) -> TaskResult:
        if self.milk and self._milker_in_carry_pair(ram) and self._begin_next_milk(ram):
            return TaskResult(status=TaskStatus.RUNNING)
        if self.feed and self._feed_remaining > 0 and read_stored_grass(ram) > 0:
            self._phase = "feed_place_nav" if read_held_item(ram) == ITEM_FODDER else "fodder_nav"
            self._fodder_route_index = 0
            self._feed_route_index = 0
        elif self._begin_next_cow_care(ram):
            return TaskResult(status=TaskStatus.RUNNING)
        else:
            self._begin_exit_prep()
            return TaskResult(status=TaskStatus.RUNNING)
        self._verify_count = 0
        self._interaction_started = False
        self._clear_navigation()
        return TaskResult(status=TaskStatus.RUNNING)

    def _after_feed(self, ram: np.ndarray) -> TaskResult:
        goal = self._current_feed_goal(ram)
        fed_now = self._fed_count_now(ram)
        if fed_now > self._fed_before:
            self._feed_remaining = max(0, goal - fed_now)
            self._fed_before = fed_now
            self.fed_count += 1
            print(
                f"[COW] Feed OK count={fed_now} remaining={self._feed_remaining} "
                f"flags=0x{read_fed_cows_flags(ram):04X}"
            )
        else:
            self._feed_remaining = max(0, goal - fed_now)
            print(
                f"[COW] Feed no flag change count={fed_now} remaining={self._feed_remaining} "
                f"flags=0x{read_fed_cows_flags(ram):04X}"
            )
        if self._feed_remaining > 0 and read_stored_grass(ram) > 0:
            self._phase = "feed_place_nav" if read_held_item(ram) == ITEM_FODDER else "fodder_nav"
            self._fodder_route_index = 0
            self._feed_route_index = 0
        elif self._begin_next_cow_care(ram):
            return TaskResult(status=TaskStatus.RUNNING)
        else:
            self._begin_exit_prep()
            return TaskResult(status=TaskStatus.RUNNING)
        self._verify_count = 0
        self._clear_navigation()
        return TaskResult(status=TaskStatus.RUNNING)

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

    def _defer_current_milk(self, ram: np.ndarray, reason: str) -> bool:
        slot = self._target_cow_slot
        if slot is None or not self._slot_needs_milk(ram, slot):
            return False
        if not self._defer_pending_slot(
            self._milk_slots,
            self._deferred_milk_counts,
            slot,
            max_deferrals=MAX_MILK_DEFERRALS,
        ):
            return False
        print(
            f"[COW] Milk deferred slot={slot} reason={reason} "
            f"count={self._deferred_milk_counts[slot]}"
        )
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
        if retryable and self._phase in {"milk_select", "milk_nav", "milk_verify"}:
            if self._defer_current_milk(ram, reason):
                self._verify_count = 0
                self._interaction_started = False
                self._clear_navigation()
                return self._after_milk(ram)
        if retryable and self._phase not in {"milk_select", "milk_nav", "milk_verify"}:
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

        care_phases = {
            "talk_nav",
            "talk_verify",
            "brush_select",
            "brush_nav",
            "brush_verify",
            "milk_select",
            "milk_nav",
            "milk_verify",
        }
        if (
            self._phase in care_phases
            and self._target_cow_slot is not None
            and self._care_slot_started_step > 0
        ):
            slot_limit = (
                MAX_COW_SLOT_MILK_FRAMES
                if self._phase in {"milk_select", "milk_nav", "milk_verify"}
                else MAX_COW_SLOT_CARE_FRAMES
            )
            if self._step_count - self._care_slot_started_step > slot_limit:
                return self._skip_current_cow_care(world.ram, "slot_timeout")
        if (
            self._phase in care_phases
            and self._target_cow_slot is not None
            and self._nav_failures > MAX_COW_NAV_FAILURES
        ):
            return self._skip_current_cow_care(world.ram, "nav_unreachable")

        if self._phase == "brush_verify":
            # The cow interaction flag can be visible during the tool animation
            # before the queued Y/cooldown frames drain, so sample it first.
            self._mark_brushed_if_changed(world.ram)
        if self._phase == "milk_verify":
            self._mark_milked_if_changed(world.ram)
        if self._action_queue:
            input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
            if self._phase == "brush_verify" and self.brushed and input_lock == 1:
                self._action_queue.clear()
            elif self._phase == "milk_verify":
                milk_done = self._target_cow_slot is None or not cow_needs_milking(
                    world.ram,
                    self._target_cow_slot,
                )
                if milk_done and read_held_item(world.ram) and input_lock == 1:
                    self._action_queue.clear()

        action = self._action_queue.popleft() if self._action_queue else None
        if action is not None:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

        if self._phase == "done":
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=(
                    f"talk={self.talked} brush={self.brushed} "
                    f"milk={self.milked_count} ship={self.milk_shipped_count} fed={self.fed_count}"
                ),
            )

        if self._phase == "talk_nav":
            if self._talk_route_index >= 1 and (not self._navigator.path or self._navigator.stasis > 90):
                self._refresh_talk_approach(world.ram)
            if self._talk_route_index >= 1:
                self._refresh_stale_cow_approach(world.ram, "_talk_route_index")
            if self._target_cow_slot is None:
                return TaskResult(status=TaskStatus.FAILURE, reason="no target cow slot for talk")
            self._talk_face = self._face_for_target_cow(world.ram)
            action = self._care_trough_exit_action(world.ram)
            if action is not None:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            action = self._recorded_interact_nav_action(world.ram, tool=False)
            handled = self._handle_pixel_nav_action(world.ram, action, tool=False)
            if handled is not None:
                if action is not None:
                    self._talk_route_index = max(0, len(self._talk_route()) - 1)
                return handled
            route = self._talk_route()
            target = route[min(self._talk_route_index, len(route) - 1)]
            if self._talk_route_index < len(route) - 1 and self._navigator.current_tile == target:
                self._talk_route_index += 1
                self._refresh_talk_approach(world.ram)
                self._navigator.path = []
                self._navigator.stasis = 0
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            if self._talk_route_index < len(route) - 1:
                action = self._navigate_to_tile(world.ram, target)
            else:
                action = None
            if action is not None:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            if self._talk_route_index < len(route) - 1:
                self._talk_route_index += 1
                self._refresh_talk_approach(world.ram)
                self._navigator.path = []
                self._navigator.stasis = 0
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            self._clear_navigation()
            if self._navigator.current_tile != self._talk_stand and not self._at_cow_interact_pixel(world.ram, tool=False):
                action = self._navigate_route(
                    world.ram,
                    self._talk_route(),
                    "_talk_route_index",
                    center_final=False,
                )
                if action is not None:
                    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            self._talk_face = self._face_for_target_cow(world.ram)
            action = self._align_to_cow_interact_pixel(world.ram, tool=False)
            if action is not None:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            if (
                not self._at_cow_interact_pixel(world.ram, tool=False)
                and not self._is_adjacent_to_target_cow(world.ram, self._navigator.current_tile, self._talk_face)
            ):
                self._refresh_talk_approach(world.ram)
                self._talk_route_index = max(0, len(self._talk_route()) - 1)
                self._clear_navigation()
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            self._talk_flags_before = read_cow_daily_flags(world.ram, self._target_cow_slot)
            self._talk_happiness_before = read_cow_happiness(world.ram, self._target_cow_slot)
            self.talked = bool(self._talk_flags_before & COW_DAILY_TALKED_FLAG)
            self._talk_attempts += 1
            self._queue_press_a(
                self._talk_face,
                face_frames=8,
                hold_frames=16,
                settle_frames=28,
            )
            self._verify_count = 0
            self._interaction_started = False
            self._phase = "talk_verify"
            return TaskResult(status=TaskStatus.RUNNING)

        if self._phase == "talk_verify":
            input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
            if self._cow_ram_changed(
                world.ram,
                COW_DAILY_TALKED_FLAG,
                self._talk_flags_before,
                self._talk_happiness_before,
            ):
                self.talked = True
                self._remember_current_pin()
            if input_lock != 1:
                self._interaction_started = True
            if self.talked and (not self._interaction_started or input_lock == 1):
                return self._after_talk(world.ram)
            if self._interaction_started and input_lock == 1:
                retry = self._retry_talk_nav(world.ram, "dialog_closed_without_flag")
                if retry is not None:
                    return retry
                return self._after_talk(world.ram)
            self._verify_count += 1
            if self._verify_count > 90 and not self._interaction_started:
                retry = self._retry_talk_nav(world.ram, "no_dialog")
                if retry is not None:
                    return retry
                return self._after_talk(world.ram)
            action = self._dialog_pulse_action() if self._interaction_started else make_action()
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

        if self._phase == "brush_select":
            if not self._brush_in_carry_pair(world.ram):
                return self._after_brush(world.ram)
            if self._brush_selected(world.ram):
                self._phase = "brush_nav"
                self._brush_select_frames = 0
                face = self._face_for_target_cow(world.ram, self._navigator.current_tile)
                if self._is_adjacent_to_target_cow(world.ram, self._navigator.current_tile, face):
                    self._talk_face = face
                    self._brush_route_index = max(0, len(self._talk_route()) - 1)
                else:
                    self._brush_route_index = 0
                    self._pin_care_route_to_direct_stand(world.ram)
                self._clear_navigation()
                return TaskResult(status=TaskStatus.RUNNING)
            if self._player_action(world.ram) != 0:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            self._brush_select_frames += 1
            if self._brush_select_frames > 60:
                if self._target_cow_slot is not None:
                    self._skipped_brush_slots.add(self._target_cow_slot)
                    print(f"[COW] Brush skipped slot={self._target_cow_slot} attempts=select_timeout")
                return self._after_brush(world.ram)
            action = make_action(x=True) if self._brush_select_frames % 6 == 1 else make_action()
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

        if self._phase == "brush_nav":
            if not self._brush_in_carry_pair(world.ram):
                return self._after_brush(world.ram)
            if not self._brush_selected(world.ram):
                self._phase = "brush_select"
                self._brush_select_frames = 0
                return TaskResult(status=TaskStatus.RUNNING)
            self._talk_face = self._face_for_target_cow(world.ram)
            action = self._care_trough_exit_action(world.ram)
            if action is not None:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            action = self._recorded_left_tool_nav_action(world.ram)
            handled = self._handle_pixel_nav_action(world.ram, action, tool=True)
            if handled is not None:
                return handled
            if self._brush_route_index >= 1:
                self._refresh_stale_cow_approach(world.ram, "_brush_route_index")
            if (
                self._brush_route_index >= 1
                and self._navigator.current_tile != self._talk_stand
                and self._navigator.path
                and self._navigator.stasis > 90
            ):
                self._refresh_talk_approach(world.ram)
            if self._brush_route_index < len(self._talk_route()) - 1:
                action = self._navigate_route(
                    world.ram,
                    self._talk_route(),
                    "_brush_route_index",
                    center_final=False,
                )
            else:
                action = None
            if action is not None:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            if self._target_cow_slot is None:
                return TaskResult(status=TaskStatus.FAILURE, reason="no target cow slot for brush")
            self._clear_navigation()
            if self._navigator.current_tile != self._talk_stand and not self._at_cow_interact_pixel(world.ram, tool=True):
                action = self._navigate_route(
                    world.ram,
                    self._talk_route(),
                    "_brush_route_index",
                    center_final=False,
                )
                if action is not None:
                    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            self._talk_face = self._face_for_target_cow(world.ram)
            action = self._align_to_cow_interact_pixel(world.ram, tool=True)
            if action is not None:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            if (
                not self._at_cow_interact_pixel(world.ram, tool=True)
                and not self._is_adjacent_to_target_cow(world.ram, self._navigator.current_tile, self._talk_face)
            ):
                self._refresh_talk_approach(world.ram)
                self._brush_route_index = max(0, len(self._talk_route()) - 1)
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            return self._begin_brush_verify(world.ram)

        if self._phase == "brush_verify":
            input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
            self._mark_brushed_if_changed(world.ram)
            if input_lock != 1 or self._player_action(world.ram) != 0:
                self._interaction_started = True
            if self.brushed and (not self._interaction_started or input_lock == 1):
                return self._after_brush(world.ram)
            self._verify_count += 1
            if (self._interaction_started and input_lock == 1 and self._verify_count > 20) or self._verify_count > 90:
                if self._brush_attempts < MAX_BRUSH_ATTEMPTS and self._brush_in_carry_pair(world.ram):
                    print(f"[COW] Brush retry slot={self._target_cow_slot} attempts={self._brush_attempts}")
                    if self._brush_attempts < 2 or not self._prefer_body_side_stand(world.ram):
                        self._refresh_talk_approach(world.ram)
                    self._phase = "brush_nav" if self._brush_selected(world.ram) else "brush_select"
                    self._brush_route_index = max(0, len(self._talk_route()) - 1)
                    self._brush_select_frames = 0
                    self._verify_count = 0
                    self._interaction_started = False
                    self._clear_navigation()
                    return TaskResult(status=TaskStatus.RUNNING)
                if self._target_cow_slot is not None:
                    self._skipped_brush_slots.add(self._target_cow_slot)
                    print(f"[COW] Brush skipped slot={self._target_cow_slot} attempts={self._brush_attempts}")
                return self._after_brush(world.ram)
            action = self._dialog_pulse_action() if self._interaction_started else make_action()
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

        if self._phase == "milk_select":
            if not self._milker_in_carry_pair(world.ram):
                return self._after_milk(world.ram)
            if self._milker_selected(world.ram):
                self._phase = "milk_nav"
                self._milk_select_frames = 0
                face = self._face_for_target_cow(world.ram, self._navigator.current_tile)
                avoid_current = self._target_cow_slot in self._skipped_brush_slots
                if (
                    not avoid_current
                    and self._is_adjacent_to_target_cow(world.ram, self._navigator.current_tile, face)
                ):
                    self._talk_face = face
                    self._brush_route_index = max(0, len(self._talk_route()) - 1)
                elif not avoid_current and (
                    pin_face := self._recent_pin_milk_face(world.ram, self._navigator.current_tile)
                ):
                    self._talk_face = pin_face
                    self._talk_stand = self._navigator.current_tile
                    self._brush_route_index = max(0, len(self._talk_route()) - 1)
                else:
                    self._brush_route_index = 0
                    self._pin_care_route_to_direct_stand(world.ram)
                self._clear_navigation()
                return TaskResult(status=TaskStatus.RUNNING)
            if self._player_action(world.ram) != 0:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            self._milk_select_frames += 1
            if self._milk_select_frames > 60:
                if self._target_cow_slot is not None:
                    self._skipped_milk_slots.add(self._target_cow_slot)
                    if self._target_cow_slot in self._milk_slots:
                        self._milk_slots.remove(self._target_cow_slot)
                return self._after_milk(world.ram)
            action = make_action(x=True) if self._milk_select_frames % 6 == 1 else make_action()
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

        if self._phase == "milk_nav":
            if self._target_cow_slot is None or not cow_needs_milking(world.ram, self._target_cow_slot):
                return self._after_milk(world.ram)
            if not self._milker_in_carry_pair(world.ram):
                return self._after_milk(world.ram)
            if not self._milker_selected(world.ram):
                self._phase = "milk_select"
                self._milk_select_frames = 0
                return TaskResult(status=TaskStatus.RUNNING)
            self._talk_face = self._face_for_target_cow(world.ram)
            action = self._recorded_left_tool_nav_action(world.ram)
            handled = self._handle_pixel_nav_action(world.ram, action, tool=True)
            if handled is not None:
                return handled
            if self._brush_route_index >= 1:
                self._refresh_stale_cow_approach(world.ram, "_brush_route_index")
            if (
                self._navigator.current_tile != self._talk_stand
                and self._navigator.path
                and self._navigator.stasis > 90
            ):
                self._refresh_talk_approach(world.ram)
            if self._brush_route_index < len(self._talk_route()) - 1:
                action = self._navigate_route(
                    world.ram,
                    self._talk_route(),
                    "_brush_route_index",
                    center_final=False,
                )
            else:
                action = None
            if action is not None:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            self._clear_navigation()
            if self._navigator.current_tile != self._talk_stand and not self._at_cow_interact_pixel(world.ram, tool=True):
                action = self._navigate_route(
                    world.ram,
                    self._talk_route(),
                    "_brush_route_index",
                    center_final=False,
                )
                if action is not None:
                    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            self._talk_face = self._face_for_target_cow(world.ram)
            action = self._align_to_cow_interact_pixel(world.ram, tool=True)
            if action is not None:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            if (
                not self._at_cow_interact_pixel(world.ram, tool=True)
                and not self._is_adjacent_to_target_cow(world.ram, self._navigator.current_tile, self._talk_face)
            ):
                if pin_face := self._recent_pin_milk_face(world.ram):
                    self._talk_face = pin_face
                    return self._begin_milk_verify(world.ram)
                self._refresh_talk_approach(world.ram)
                self._brush_route_index = max(0, len(self._talk_route()) - 1)
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            self._pixel_nav_stall_count = 0
            self._reset_pixel_nav_progress()
            return self._begin_milk_verify(world.ram)

        if self._phase == "milk_verify":
            input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
            self._mark_milked_if_changed(world.ram)
            held_now = read_held_item(world.ram)
            if input_lock != 1 or self._player_action(world.ram) != 0:
                self._interaction_started = True
            milk_done = self._target_cow_slot is None or not cow_needs_milking(world.ram, self._target_cow_slot)
            if milk_done and held_now:
                if input_lock == 1:
                    self._phase = "milk_ship_nav"
                    self._ship_route_index = 0
                    self._verify_count = 0
                    self._clear_navigation()
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
            self._verify_count += 1
            if (self._interaction_started and input_lock == 1 and self._verify_count > 20) or self._verify_count > 110:
                if self._target_cow_slot is not None and not cow_needs_milking(world.ram, self._target_cow_slot):
                    if read_held_item(world.ram):
                        self._phase = "milk_ship_nav"
                        self._ship_route_index = 0
                        self._verify_count = 0
                        self._clear_navigation()
                        return TaskResult(status=TaskStatus.RUNNING)
                    return self._after_milk(world.ram)
                if self._milk_attempts < MAX_MILK_ATTEMPTS and self._milker_in_carry_pair(world.ram):
                    print(f"[COW] Milk retry slot={self._target_cow_slot} attempts={self._milk_attempts}")
                    self._refresh_talk_approach(world.ram)
                    self._phase = "milk_nav" if self._milker_selected(world.ram) else "milk_select"
                    self._brush_route_index = max(0, len(self._talk_route()) - 1)
                    self._milk_select_frames = 0
                    self._verify_count = 0
                    self._interaction_started = False
                    # Keep the original slot timer so retries cannot outrun the
                    # external stall watchdog by resetting every attempt.
                    self._clear_navigation()
                    self._reset_pixel_nav_progress()
                    return TaskResult(status=TaskStatus.RUNNING)
                if self._defer_current_milk(world.ram, "attempts"):
                    self._verify_count = 0
                    self._interaction_started = False
                    self._clear_navigation()
                    return self._after_milk(world.ram)
                if self._target_cow_slot in self._milk_slots:
                    print(f"[COW] Milk skipped slot={self._target_cow_slot} attempts={self._milk_attempts}")
                    self._milk_slots.remove(self._target_cow_slot)
                if self._target_cow_slot is not None:
                    self._skipped_milk_slots.add(self._target_cow_slot)
                return self._after_milk(world.ram)
            action = self._dialog_pulse_action() if self._interaction_started else make_action()
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

        if self._phase == "milk_ship_nav":
            if read_held_item(world.ram) == 0:
                return self._after_milk(world.ram)
            if (
                self._navigator.current_tile == BARN_SHIP_BIN_INTERACT_STAND
                and abs(self._navigator.current_pos.x - MILK_SHIP_PIXEL_ROUTE[-1][0]) <= 3
                and abs(self._navigator.current_pos.y - MILK_SHIP_PIXEL_ROUTE[-1][1]) <= 3
            ):
                self._ship_money_before = read_shipping_money(world.ram)
                self._queue_press_a(
                    BARN_SHIP_BIN_FACE,
                    face_frames=8,
                    hold_frames=16,
                    settle_frames=24,
                )
                self._verify_count = 0
                self._phase = "milk_ship_verify"
                return TaskResult(status=TaskStatus.RUNNING)
            action = self._milk_ship_pixel_action()
            if action is not None:
                self._clear_navigation()
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(left=True, b=True)))

        if self._phase == "milk_ship_verify":
            money_now = read_shipping_money(world.ram)
            if money_now > self._ship_money_before:
                self.milk_shipped_count += 1
                print(f"[COW] Milk shipped money={money_now}")
                return self._after_milk(world.ram)
            if read_held_item(world.ram) == 0:
                self.milk_shipped_count += 1
                print(f"[COW] Milk shipped")
                return self._after_milk(world.ram)
            self._verify_count += 1
            if self._verify_count > 30:
                self._phase = "milk_ship_nav"
                self._verify_count = 0
                self._clear_navigation()
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        if self._phase == "fodder_nav":
            if read_held_item(world.ram) == ITEM_FODDER:
                self._phase = "feed_place_nav"
                self._feed_route_index = 0
                self._clear_navigation()
                return TaskResult(status=TaskStatus.RUNNING)
            if read_stored_grass(world.ram) <= 0:
                self._begin_exit_prep()
                return TaskResult(status=TaskStatus.RUNNING)
            action = self._left_cow_to_fodder_action()
            if action is not None:
                self._clear_navigation()
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            action = self._left_trough_return_action()
            if action is not None:
                self._clear_navigation()
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            fodder_x = FODDER_STAND[0] * 16 + 8
            at_fodder = (
                abs(self._navigator.current_pos.x - fodder_x) <= 2
                and abs(self._navigator.current_pos.y - LEFT_TROUGH_LANE_Y) <= 2
            )
            if not at_fodder:
                action = self._navigate_route(world.ram, self._fodder_route(), "_fodder_route_index")
                if action is not None:
                    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            self._grass_before = read_stored_grass(world.ram)
            self._clear_navigation()
            self._queue_press_a(FODDER_FACE, face_frames=4, hold_frames=10, settle_frames=4)
            self._verify_count = 0
            self._phase = "fodder_verify"
            return TaskResult(status=TaskStatus.RUNNING)

        if self._phase == "fodder_verify":
            has_fodder = read_held_item(world.ram) == ITEM_FODDER
            grass_now = read_stored_grass(world.ram)
            if has_fodder and (grass_now < self._grass_before or self._verify_count > 8):
                self._grass_before = grass_now
                self._phase = "feed_place_nav"
                self._feed_route_index = 0
                self._verify_count = 0
                self._clear_navigation()
                return TaskResult(status=TaskStatus.RUNNING)
            self._verify_count += 1
            if self._verify_count > 30:
                self._phase = "fodder_nav"
                self._clear_navigation()
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        if self._phase == "feed_place_nav":
            if read_held_item(world.ram) != ITEM_FODDER:
                self._phase = "fodder_nav"
                self._fodder_route_index = 0
                self._clear_navigation()
                return TaskResult(status=TaskStatus.RUNNING)
            feed_spot = self._next_feed_spot(world.ram)
            action = self._left_feed_spot_action(feed_spot)
            if action is not None:
                self._clear_navigation()
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            if feed_spot.stand[0] <= 7:
                self._fed_before = self._fed_count_now(world.ram)
                self._fed_flags_before = read_fed_cows_flags(world.ram)
                self._clear_navigation()
                self._queue_press_a(
                    feed_spot.face,
                    face_frames=4,
                    hold_frames=8,
                    settle_frames=4,
                    hold_face_with_a=False,
                )
                self._verify_count = 0
                self._phase = "feed_verify"
                return TaskResult(status=TaskStatus.RUNNING)
            action = self._navigate_route(
                world.ram,
                self._feed_route(feed_spot),
                "_feed_route_index",
                center_final=False,
            )
            if action is not None:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            action = align_to_pixel(
                (self._navigator.current_pos.x, self._navigator.current_pos.y),
                feed_spot.interact_px,
                tolerance=0,
            )
            if action is not None:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            self._fed_before = self._fed_count_now(world.ram)
            self._fed_flags_before = read_fed_cows_flags(world.ram)
            self._clear_navigation()
            self._queue_press_a(
                feed_spot.face,
                face_frames=4,
                hold_frames=8,
                settle_frames=4,
                hold_face_with_a=False,
            )
            self._verify_count = 0
            self._phase = "feed_verify"
            return TaskResult(status=TaskStatus.RUNNING)

        if self._phase == "feed_verify":
            flags_now = read_fed_cows_flags(world.ram)
            held_now = read_held_item(world.ram)
            if held_now != ITEM_FODDER:
                if flags_now != self._fed_flags_before:
                    self._fed_flags_before = flags_now
                return self._after_feed(world.ram)
            self._verify_count += 1
            if self._verify_count > 30:
                self._phase = "feed_place_nav"
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

        if self._phase == "exit_prep_nav":
            if self._exit_prep_started_step <= 0:
                self._exit_prep_started_step = self._step_count
            if self._step_count - self._exit_prep_started_step > MAX_EXIT_PREP_FRAMES:
                print(
                    f"[COW] Exit prep timeout at {self._navigator.current_tile}; "
                    "handing off to EXIT_BARN"
                )
                self._phase = "done"
                return TaskResult(status=TaskStatus.RUNNING)
            if (
                self._navigator.current_tile == COW_EXIT_PREP_STAND
                or self._navigator.at_tile(COW_EXIT_PREP_STAND)
            ):
                self._phase = "done"
                return TaskResult(status=TaskStatus.RUNNING)
            action = self._exit_prep_escape_action()
            if action is not None:
                self._clear_navigation()
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            action = self._navigate_to_tile(world.ram, COW_EXIT_PREP_STAND)
            if action is not None:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
            self._phase = "done"
            return TaskResult(status=TaskStatus.RUNNING)

        return TaskResult(status=TaskStatus.FAILURE, reason=f"unknown phase {self._phase}")

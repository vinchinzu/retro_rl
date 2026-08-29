"""Autonomous barn cow chores.

Current scope follows recorded barn chores: milk ready cows and ship milk in
the barn bin, talk to and brush each cow when tools are available, then place
fodder in the trough.

Extracted arms (rr-y80y):
  - ``cow_geometry`` — pure barn geometry
  - ``cow_care`` — pixel-lane action builders
  - ``cow_target`` / ``cow_nav_ops`` — target/nav/brush mixins
  - ``cow_milk_ops`` / ``cow_feed_ops`` — milk/feed phase mixins
  Talk/care-slot handoff lives here with the composer.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Optional, Tuple

import numpy as np

from harvest.core.animal_probe import BARN_TILEMAP
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
    read_cow_daily_flags,
    read_cow_happiness,
    read_fed_cows_flags,
    read_held_item,
    read_num_cows,
    read_stored_grass,
)
from harvest.core.ram_catalog import field_spec
from harvest.core.tile_catalog import ADDR_INPUT_LOCK, ADDR_TILEMAP, Tool
from harvest.tasks.cow_geometry import (
    BARN_SHIP_BIN_INTERACT_STAND,
    BARN_SHIP_BIN_STAND,
    COW_EXIT_PREP_STAND,
    COW_FEED_SPOTS,
    COW_INTERACT_X_OFFSET,
    COW_LEFT_INTERACT_X,
    COW_TALK_FACE,
    COW_TALK_ROUTE,
    COW_TALK_STAND,
    FEED_TROUGH_INTERACT_PX,
    FEED_TROUGH_ROUTE,
    FEED_TROUGH_STAND,
    FODDER_ROUTE,
    FODDER_STAND,
    FODDER_TROUGH_ROUTE,
    MILK_SHIP_ROUTE,
)
from harvest.tasks.farm_clearer import TileScanner
from harvest.tasks.nav import Navigator, Pathfinder, make_action
from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

# CowPhase is a str Enum so legacy comparisons and test assignments like
# task._phase = "talk_nav" keep working. Remaining mixins import these names
# from this module, so they are defined before those imports.


class CowPhase(str, Enum):
    TALK_NAV = "talk_nav"
    TALK_VERIFY = "talk_verify"
    BRUSH_SELECT = "brush_select"
    BRUSH_NAV = "brush_nav"
    BRUSH_VERIFY = "brush_verify"
    MILK_SELECT = "milk_select"
    MILK_NAV = "milk_nav"
    MILK_VERIFY = "milk_verify"
    MILK_SHIP_NAV = "milk_ship_nav"
    MILK_SHIP_VERIFY = "milk_ship_verify"
    FODDER_NAV = "fodder_nav"
    FODDER_VERIFY = "fodder_verify"
    FEED_PLACE_NAV = "feed_place_nav"
    FEED_VERIFY = "feed_verify"
    EXIT_PREP_NAV = "exit_prep_nav"
    DONE = "done"


CARE_PHASES = frozenset(
    {
        CowPhase.TALK_NAV,
        CowPhase.TALK_VERIFY,
        CowPhase.BRUSH_SELECT,
        CowPhase.BRUSH_NAV,
        CowPhase.BRUSH_VERIFY,
        CowPhase.MILK_SELECT,
        CowPhase.MILK_NAV,
        CowPhase.MILK_VERIFY,
    }
)

MILK_CARE_PHASES = frozenset(
    {
        CowPhase.MILK_SELECT,
        CowPhase.MILK_NAV,
        CowPhase.MILK_VERIFY,
    }
)

TOOL_CARE_PHASES = frozenset(
    {
        CowPhase.BRUSH_NAV,
        CowPhase.BRUSH_VERIFY,
        CowPhase.MILK_NAV,
        CowPhase.MILK_VERIFY,
    }
)

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

from harvest.tasks.cow_feed_ops import CowFeedMixin
from harvest.tasks.cow_milk_ops import CowMilkMixin
from harvest.tasks.cow_nav_ops import CowNavMixin
from harvest.tasks.cow_target import CowTargetMixin


@dataclass
class CowChoresTask(
    CowTargetMixin,
    CowNavMixin,
    CowMilkMixin,
    CowFeedMixin,
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

    def _retry_talk_nav(self, ram: np.ndarray, reason: str) -> Optional[TaskResult]:
        if self._talk_attempts >= MAX_TALK_ATTEMPTS:
            return None
        print(f"[COW] Talk retry slot={self._target_cow_slot} attempts={self._talk_attempts} reason={reason}")
        self._refresh_talk_approach(ram)
        self._talk_route_index = max(0, len(self._talk_route()) - 1)
        self._verify_count = 0
        self._interaction_started = False
        self._clear_navigation()
        self._phase = CowPhase.TALK_NAV
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
            self._phase = CowPhase.BRUSH_NAV if self._brush_selected(ram) else "brush_select"
            return TaskResult(status=TaskStatus.RUNNING)
        if self._target_cow_slot is not None and self._slot_needs_milk(ram, self._target_cow_slot):
            self._refresh_talk_approach(ram)
            self._brush_route_index = max(0, len(self._talk_route()) - 1)
            self._milk_select_frames = 0
            self._milk_attempts = 0
            self._verify_count = 0
            self._interaction_started = False
            self._clear_navigation()
            self._phase = CowPhase.MILK_NAV if self._milker_selected(ram) else "milk_select"
            return TaskResult(status=TaskStatus.RUNNING)
        if self._begin_next_cow_care(ram):
            return TaskResult(status=TaskStatus.RUNNING)
        return self._after_brush(ram)

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
            self._phase = CowPhase.TALK_NAV
        elif self._slot_needs_brush(ram, self._target_cow_slot):
            self.brushed = False
            self._phase = CowPhase.BRUSH_NAV if self._brush_selected(ram) else "brush_select"
        elif self._slot_needs_milk(ram, self._target_cow_slot):
            self._phase = CowPhase.MILK_NAV if self._milker_selected(ram) else "milk_select"
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

    def _step_talk_nav(self, world: WorldState) -> TaskResult:
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
        self._phase = CowPhase.TALK_VERIFY
        return TaskResult(status=TaskStatus.RUNNING)

    def _step_talk_verify(self, world: WorldState) -> TaskResult:
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

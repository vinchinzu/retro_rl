"""Fodder pickup + trough feed phase arms for CowChoresTask (rr-y80y)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from harvest.core.animal_status import (
    ITEM_FODDER,
    existing_cow_slots,
    read_fed_cows_flags,
    read_fed_cows_n,
    read_held_item,
    read_stored_grass,
)
from harvest.tasks.animal_navigation import align_to_pixel
from harvest.tasks.cow_care import (
    left_cow_to_fodder_action,
    left_feed_spot_action,
    left_trough_return_action,
)
from harvest.tasks.cow_task import CowPhase
from harvest.tasks.cow_geometry import (
    COW_FEED_SPOTS,
    FODDER_FACE,
    FODDER_STAND,
    LEFT_TROUGH_LANE_Y,
    CowFeedSpot,
    count_fed_trough_flags,
    feed_route_for_spot,
    fodder_route_from,
    next_unfed_spot,
)
from harvest.tasks.nav import make_action
from retro_harness import ActionResult, TaskResult, TaskStatus, WorldState


class CowFeedMixin:
    """Fodder bin nav and trough placement phases."""

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

    def _fodder_route(self) -> Tuple[Tuple[int, int], ...]:
        return fodder_route_from(self._navigator.current_tile)

    def _left_feed_spot_action(self, spot: CowFeedSpot) -> Optional[np.ndarray]:
        return left_feed_spot_action(
            spot,
            self._navigator.current_pos.x,
            self._navigator.current_pos.y,
        )

    def _left_cow_to_fodder_action(self) -> Optional[np.ndarray]:
        return left_cow_to_fodder_action(
            self._navigator.current_pos.x,
            self._navigator.current_pos.y,
        )

    def _left_trough_return_action(self) -> Optional[np.ndarray]:
        return left_trough_return_action(
            self._navigator.current_pos.x,
            self._navigator.current_pos.y,
        )

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
            self._phase = CowPhase.FEED_PLACE_NAV if read_held_item(ram) == ITEM_FODDER else "fodder_nav"
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

    def _step_fodder_nav(self, world: WorldState) -> TaskResult:
        if read_held_item(world.ram) == ITEM_FODDER:
            self._phase = CowPhase.FEED_PLACE_NAV
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
        self._phase = CowPhase.FODDER_VERIFY
        return TaskResult(status=TaskStatus.RUNNING)

    def _step_fodder_verify(self, world: WorldState) -> TaskResult:
        has_fodder = read_held_item(world.ram) == ITEM_FODDER
        grass_now = read_stored_grass(world.ram)
        if has_fodder and (grass_now < self._grass_before or self._verify_count > 8):
            self._grass_before = grass_now
            self._phase = CowPhase.FEED_PLACE_NAV
            self._feed_route_index = 0
            self._verify_count = 0
            self._clear_navigation()
            return TaskResult(status=TaskStatus.RUNNING)
        self._verify_count += 1
        if self._verify_count > 30:
            self._phase = CowPhase.FODDER_NAV
            self._clear_navigation()
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

    def _step_feed_place_nav(self, world: WorldState) -> TaskResult:
        if read_held_item(world.ram) != ITEM_FODDER:
            self._phase = CowPhase.FODDER_NAV
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
            self._phase = CowPhase.FEED_VERIFY
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
        self._phase = CowPhase.FEED_VERIFY
        return TaskResult(status=TaskStatus.RUNNING)

    def _step_feed_verify(self, world: WorldState) -> TaskResult:
        flags_now = read_fed_cows_flags(world.ram)
        held_now = read_held_item(world.ram)
        if held_now != ITEM_FODDER:
            if flags_now != self._fed_flags_before:
                self._fed_flags_before = flags_now
            return self._after_feed(world.ram)
        self._verify_count += 1
        if self._verify_count > 30:
            self._phase = CowPhase.FEED_PLACE_NAV
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

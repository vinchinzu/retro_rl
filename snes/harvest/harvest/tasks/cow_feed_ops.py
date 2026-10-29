"""Fodder pickup + trough feed phase functions for CowChoresTask (rr-y80y)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from harvest.core.animal_status import (
    ITEM_FODDER,
    existing_cow_slots,
    read_fed_cows_flags,
    read_held_item,
    read_stored_grass,
    read_cow_status,
    COW_STATUS_PREGNANT_FLAG,
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
    PREGNANT_COW_FEED_SPOT,
    feed_route_for_spot,
    fodder_route_from,
    next_unfed_spot,
)
from harvest.tasks.nav import make_action
from retro_harness import ActionResult, TaskResult, TaskStatus, WorldState

def _feedable_cow_slots(task, ram: np.ndarray) -> list[int]:
    return existing_cow_slots(ram)

def _required_feed_spots(task, ram: np.ndarray) -> tuple[CowFeedSpot, ...]:
    """The ROM checks each extant slot's bit, not a prefix of filled troughs."""
    flags = {
        0x1000 if read_cow_status(ram, slot) & COW_STATUS_PREGNANT_FLAG else 1 << slot
        for slot in task._feedable_cow_slots(ram)
    }
    return tuple(spot for spot in (*COW_FEED_SPOTS, PREGNANT_COW_FEED_SPOT) if spot.flag in flags)

def _feed_goal(task, ram: np.ndarray) -> int:
    return len(task._required_feed_spots(ram))

def _current_feed_goal(task, ram: np.ndarray) -> int:
    return task._feed_goal_count or task._feed_goal(ram)

def _fed_trough_count(task, ram: np.ndarray) -> int:
    return sum(bool(read_fed_cows_flags(ram) & spot.flag) for spot in task._required_feed_spots(ram))

def _fed_count_now(task, ram: np.ndarray) -> int:
    return task._fed_trough_count(ram)

def _next_feed_spot(task, ram: np.ndarray) -> CowFeedSpot:
    spots = task._required_feed_spots(ram)
    return next_unfed_spot(read_fed_cows_flags(ram), len(spots), spots)

def _feed_route(task, spot: CowFeedSpot) -> Tuple[Tuple[int, int], ...]:
    return feed_route_for_spot(spot)

def _fodder_route(task) -> Tuple[Tuple[int, int], ...]:
    return fodder_route_from(task._navigator.current_tile)

def _left_feed_spot_action(task, spot: CowFeedSpot) -> Optional[np.ndarray]:
    return left_feed_spot_action(
        spot,
        task._navigator.current_pos.x,
        task._navigator.current_pos.y,
    )

def _left_cow_to_fodder_action(task) -> Optional[np.ndarray]:
    return left_cow_to_fodder_action(
        task._navigator.current_pos.x,
        task._navigator.current_pos.y,
    )

def _left_trough_return_action(task) -> Optional[np.ndarray]:
    return left_trough_return_action(
        task._navigator.current_pos.x,
        task._navigator.current_pos.y,
    )

def _after_feed(task, ram: np.ndarray) -> TaskResult:
    goal = task._current_feed_goal(ram)
    fed_now = task._fed_count_now(ram)
    if fed_now > task._fed_before:
        task._feed_remaining = max(0, goal - fed_now)
        task._fed_before = fed_now
        task.fed_count += 1
        print(
            f"[COW] Feed OK count={fed_now} remaining={task._feed_remaining} "
            f"flags=0x{read_fed_cows_flags(ram):04X}"
        )
    else:
        task._feed_remaining = max(0, goal - fed_now)
        print(
            f"[COW] Feed no flag change count={fed_now} remaining={task._feed_remaining} "
            f"flags=0x{read_fed_cows_flags(ram):04X}"
        )
    if task._feed_remaining > 0 and read_stored_grass(ram) > 0:
        task._phase = CowPhase.FEED_PLACE_NAV if read_held_item(ram) == ITEM_FODDER else "fodder_nav"
        task._fodder_route_index = 0
        task._feed_route_index = 0
    elif task._begin_next_cow_care(ram):
        return TaskResult(status=TaskStatus.RUNNING)
    else:
        task._begin_exit_prep()
        return TaskResult(status=TaskStatus.RUNNING)
    task._verify_count = 0
    task._clear_navigation()
    return TaskResult(status=TaskStatus.RUNNING)

def _step_fodder_nav(task, world: WorldState) -> TaskResult:
    if read_held_item(world.ram) == ITEM_FODDER:
        task._phase = CowPhase.FEED_PLACE_NAV
        task._feed_route_index = 0
        task._clear_navigation()
        return TaskResult(status=TaskStatus.RUNNING)
    if read_stored_grass(world.ram) <= 0:
        task._begin_exit_prep()
        return TaskResult(status=TaskStatus.RUNNING)
    x, y = task._navigator.current_pos.x, task._navigator.current_pos.y
    if x >= 120 and y >= 335:
        action = make_action(left=x > 137, right=x < 135, b=True) if abs(x - 136) > 1 else make_action(up=True, b=True)
        task._clear_navigation()
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
    if x >= 120 and y > LEFT_TROUGH_LANE_Y + 2:
        # Cross the fodder shelf's row on the open aisle, not at x=204.
        action = make_action(left=x > 193, right=x < 191, b=True) if abs(x - 192) > 1 else make_action(up=True, b=True)
        task._clear_navigation()
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
    if x >= 120 and abs(y - LEFT_TROUGH_LANE_Y) <= 2 and x < 215:
        task._clear_navigation()
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(right=True, b=True)))
    action = task._left_cow_to_fodder_action()
    if action is not None:
        task._clear_navigation()
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
    action = task._left_trough_return_action()
    if action is not None:
        task._clear_navigation()
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
    fodder_x = FODDER_STAND[0] * 16 + 8
    at_fodder = (
        abs(task._navigator.current_pos.x - fodder_x) <= 2
        and abs(task._navigator.current_pos.y - LEFT_TROUGH_LANE_Y) <= 2
    )
    if not at_fodder:
        action = task._navigate_route(world.ram, task._fodder_route(), "_fodder_route_index")
        if action is not None:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
    task._grass_before = read_stored_grass(world.ram)
    task._clear_navigation()
    task._queue_press_a(FODDER_FACE, face_frames=4, hold_frames=10, settle_frames=4)
    task._verify_count = 0
    task._phase = CowPhase.FODDER_VERIFY
    return TaskResult(status=TaskStatus.RUNNING)

def _step_fodder_verify(task, world: WorldState) -> TaskResult:
    has_fodder = read_held_item(world.ram) == ITEM_FODDER
    grass_now = read_stored_grass(world.ram)
    if has_fodder and (grass_now < task._grass_before or task._verify_count > 8):
        task._grass_before = grass_now
        task._phase = CowPhase.FEED_PLACE_NAV
        task._feed_route_index = 0
        task._verify_count = 0
        task._clear_navigation()
        return TaskResult(status=TaskStatus.RUNNING)
    task._verify_count += 1
    if task._verify_count > 30:
        task._phase = CowPhase.FODDER_NAV
        task._clear_navigation()
    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

def _step_feed_place_nav(task, world: WorldState) -> TaskResult:
    if read_held_item(world.ram) != ITEM_FODDER:
        task._phase = CowPhase.FODDER_NAV
        task._fodder_route_index = 0
        task._clear_navigation()
        return TaskResult(status=TaskStatus.RUNNING)
    feed_spot = task._next_feed_spot(world.ram)
    action = task._left_feed_spot_action(feed_spot)
    if action is not None:
        task._clear_navigation()
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
    if feed_spot.stand[0] <= 7:
        task._fed_before = task._fed_count_now(world.ram)
        task._fed_flags_before = read_fed_cows_flags(world.ram)
        task._clear_navigation()
        task._queue_press_a(
            feed_spot.face,
            face_frames=4,
            hold_frames=8,
            settle_frames=4,
            hold_face_with_a=False,
        )
        task._verify_count = 0
        task._phase = CowPhase.FEED_VERIFY
        return TaskResult(status=TaskStatus.RUNNING)
    action = task._navigate_route(
        world.ram,
        task._feed_route(feed_spot),
        "_feed_route_index",
        center_final=False,
    )
    if action is not None:
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
    action = align_to_pixel(
        (task._navigator.current_pos.x, task._navigator.current_pos.y),
        feed_spot.interact_px,
        tolerance=0,
    )
    if action is not None:
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
    task._fed_before = task._fed_count_now(world.ram)
    task._fed_flags_before = read_fed_cows_flags(world.ram)
    task._clear_navigation()
    task._queue_press_a(
        feed_spot.face,
        face_frames=4,
        hold_frames=8,
        settle_frames=4,
        hold_face_with_a=False,
    )
    task._verify_count = 0
    task._phase = CowPhase.FEED_VERIFY
    return TaskResult(status=TaskStatus.RUNNING)

def _step_feed_verify(task, world: WorldState) -> TaskResult:
    flags_now = read_fed_cows_flags(world.ram)
    held_now = read_held_item(world.ram)
    if held_now != ITEM_FODDER:
        if flags_now != task._fed_flags_before:
            task._fed_flags_before = flags_now
        return task._after_feed(world.ram)
    task._verify_count += 1
    if task._verify_count > 30:
        task._phase = CowPhase.FEED_PLACE_NAV
    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))


def bind_task_methods(cls: type) -> None:
    """Attach these phase functions to the task class. Not a mixin."""
    for name, fn in list(globals().items()):
        if not name.startswith("_") or not callable(fn):
            continue
        if getattr(fn, "__module__", None) != __name__:
            continue
        setattr(cls, name, fn)

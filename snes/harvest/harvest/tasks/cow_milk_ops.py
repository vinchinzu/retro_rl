"""Milk + ship phase functions for CowChoresTask (rr-y80y)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from harvest.core.animal_status import (
    COW_DAILY_MILKED_FLAG,
    ITEM_FODDER,
    cow_needs_milking,
    read_cow_daily_flags,
    read_held_item,
    read_stored_grass,
)
from harvest.core.tile_catalog import ADDR_INPUT_LOCK
from harvest.tasks.cow_task import MAX_MILK_ATTEMPTS, MAX_MILK_DEFERRALS, CowPhase
from harvest.tasks.cow_geometry import BARN_SHIP_BIN_FACE, BARN_SHIP_BIN_INTERACT_STAND, MILK_SHIP_PIXEL_ROUTE
from harvest.tasks.cow_care import milk_ship_escape_prefix_action, milk_ship_route_step_action
from harvest.tasks.harvest_task import read_shipping_money
from harvest.tasks.nav import make_action
from retro_harness import ActionResult, TaskResult, TaskStatus, WorldState

def _milk_ship_pixel_action(task) -> Optional[np.ndarray]:
    x = task._navigator.current_pos.x
    y = task._navigator.current_pos.y
    prefix = milk_ship_escape_prefix_action(x, y, ship_route_index=task._ship_route_index)
    if prefix is not None:
        return prefix

    index = min(task._ship_route_index, len(MILK_SHIP_PIXEL_ROUTE) - 1)
    target = MILK_SHIP_PIXEL_ROUTE[index]
    if abs(x - target[0]) <= 2 and abs(y - target[1]) <= 2:
        if task._ship_route_index < len(MILK_SHIP_PIXEL_ROUTE) - 1:
            task._ship_route_index += 1
            return make_action()
        return None

    return milk_ship_route_step_action(x, y, index)

def _begin_next_milk(task, ram: np.ndarray) -> bool:
    task._milk_slots = [
        slot
        for slot in task._milk_slots
        if slot not in task._skipped_milk_slots and cow_needs_milking(ram, slot)
    ]
    if not task._milk_slots:
        return False
    task._target_cow_slot = task._milk_slots[0]
    task._refresh_talk_approach(ram)
    task._brush_route_index = max(0, len(task._talk_route()) - 1)
    task._pin_care_route_to_direct_stand(ram)
    task._milk_select_frames = 0
    task._milk_attempts = 0
    task._verify_count = 0
    task._interaction_started = False
    task._care_slot_started_step = task._step_count
    task._pixel_nav_stall_count = 0
    task._reset_pixel_nav_progress()
    task._clear_navigation()
    task._phase = CowPhase.MILK_NAV if task._milker_selected(ram) else "milk_select"
    return True

def _begin_milk_verify(task, ram: np.ndarray) -> TaskResult:
    if task._target_cow_slot is None:
        return TaskResult(status=TaskStatus.FAILURE, reason="no target cow slot for milk")
    task._milk_flags_before = read_cow_daily_flags(ram, task._target_cow_slot)
    task._milk_held_before = read_held_item(ram)
    task._clear_navigation()
    task._queue_use_tool(task._talk_face, face_frames=8, hold_frames=9, y_only_frames=1, settle_frames=85)
    task._milk_attempts += 1
    task._verify_count = 0
    task._interaction_started = False
    task._phase = CowPhase.MILK_VERIFY
    action = task._action_queue.popleft() if task._action_queue else None
    if action is not None:
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
    return TaskResult(status=TaskStatus.RUNNING)

def _mark_milked_if_changed(task, ram: np.ndarray) -> None:
    if task._target_cow_slot is None:
        return
    flags_now = read_cow_daily_flags(ram, task._target_cow_slot)
    if not (flags_now & COW_DAILY_MILKED_FLAG):
        return
    if task._target_cow_slot in task._milk_slots:
        task._milk_slots.remove(task._target_cow_slot)
    if task._target_cow_slot not in task._milked_slots:
        task._milked_slots.add(task._target_cow_slot)
        task.milked_count += 1
        print(f"[COW] Milk OK slot={task._target_cow_slot} attempts={task._milk_attempts}")

def _after_milk(task, ram: np.ndarray) -> TaskResult:
    if task.milk and task._milker_in_carry_pair(ram) and task._begin_next_milk(ram):
        return TaskResult(status=TaskStatus.RUNNING)
    if task.feed and task._feed_remaining > 0 and read_stored_grass(ram) > 0:
        task._phase = CowPhase.FEED_PLACE_NAV if read_held_item(ram) == ITEM_FODDER else "fodder_nav"
        task._fodder_route_index = 0
        task._feed_route_index = 0
    elif task._begin_next_cow_care(ram):
        return TaskResult(status=TaskStatus.RUNNING)
    else:
        task._begin_exit_prep()
        return TaskResult(status=TaskStatus.RUNNING)
    task._verify_count = 0
    task._interaction_started = False
    task._clear_navigation()
    return TaskResult(status=TaskStatus.RUNNING)

def _defer_current_milk(task, ram: np.ndarray, reason: str) -> bool:
    slot = task._target_cow_slot
    if slot is None or not task._slot_needs_milk(ram, slot):
        return False
    if not task._defer_pending_slot(
        task._milk_slots,
        task._deferred_milk_counts,
        slot,
        max_deferrals=MAX_MILK_DEFERRALS,
    ):
        return False
    print(
        f"[COW] Milk deferred slot={slot} reason={reason} "
        f"count={task._deferred_milk_counts[slot]}"
    )
    return True

def _step_milk_select(task, world: WorldState) -> TaskResult:
    if not task._milker_in_carry_pair(world.ram):
        return task._after_milk(world.ram)
    if task._milker_selected(world.ram):
        task._phase = CowPhase.MILK_NAV
        task._milk_select_frames = 0
        face = task._face_for_target_cow(world.ram, task._navigator.current_tile)
        avoid_current = task._target_cow_slot in task._skipped_brush_slots
        if (
            not avoid_current
            and task._is_adjacent_to_target_cow(world.ram, task._navigator.current_tile, face)
        ):
            task._talk_face = face
            task._brush_route_index = max(0, len(task._talk_route()) - 1)
        elif not avoid_current and (
            pin_face := task._recent_pin_milk_face(world.ram, task._navigator.current_tile)
        ):
            task._talk_face = pin_face
            task._talk_stand = task._navigator.current_tile
            task._brush_route_index = max(0, len(task._talk_route()) - 1)
        else:
            task._brush_route_index = 0
            task._pin_care_route_to_direct_stand(world.ram)
        task._clear_navigation()
        return TaskResult(status=TaskStatus.RUNNING)
    if task._player_action(world.ram) != 0:
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
    task._milk_select_frames += 1
    if task._milk_select_frames > 60:
        if task._target_cow_slot is not None:
            task._skipped_milk_slots.add(task._target_cow_slot)
            if task._target_cow_slot in task._milk_slots:
                task._milk_slots.remove(task._target_cow_slot)
        return task._after_milk(world.ram)
    action = make_action(x=True) if task._milk_select_frames % 6 == 1 else make_action()
    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

def _step_milk_nav(task, world: WorldState) -> TaskResult:
    if task._target_cow_slot is None or not cow_needs_milking(world.ram, task._target_cow_slot):
        return task._after_milk(world.ram)
    if not task._milker_in_carry_pair(world.ram):
        return task._after_milk(world.ram)
    if not task._milker_selected(world.ram):
        task._phase = CowPhase.MILK_SELECT
        task._milk_select_frames = 0
        return TaskResult(status=TaskStatus.RUNNING)
    task._talk_face = task._face_for_target_cow(world.ram)
    action = task._recorded_left_tool_nav_action(world.ram)
    handled = task._handle_pixel_nav_action(world.ram, action, tool=True)
    if handled is not None:
        return handled
    if task._brush_route_index >= 1:
        task._refresh_stale_cow_approach(world.ram, "_brush_route_index")
    if (
        task._navigator.current_tile != task._talk_stand
        and task._navigator.path
        and task._navigator.stasis > 90
    ):
        task._refresh_talk_approach(world.ram)
    if task._brush_route_index < len(task._talk_route()) - 1:
        action = task._navigate_route(
            world.ram,
            task._talk_route(),
            "_brush_route_index",
            center_final=False,
        )
    else:
        action = None
    if action is not None:
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
    task._clear_navigation()
    if task._navigator.current_tile != task._talk_stand and not task._at_cow_interact_pixel(world.ram, tool=True):
        action = task._navigate_route(
            world.ram,
            task._talk_route(),
            "_brush_route_index",
            center_final=False,
        )
        if action is not None:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
    task._talk_face = task._face_for_target_cow(world.ram)
    action = task._align_to_cow_interact_pixel(world.ram, tool=True)
    if action is not None:
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
    if (
        not task._at_cow_interact_pixel(world.ram, tool=True)
        and not task._is_adjacent_to_target_cow(world.ram, task._navigator.current_tile, task._talk_face)
    ):
        if pin_face := task._recent_pin_milk_face(world.ram):
            task._talk_face = pin_face
            return task._begin_milk_verify(world.ram)
        task._refresh_talk_approach(world.ram)
        task._brush_route_index = max(0, len(task._talk_route()) - 1)
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
    task._pixel_nav_stall_count = 0
    task._reset_pixel_nav_progress()
    return task._begin_milk_verify(world.ram)

def _step_milk_verify(task, world: WorldState) -> TaskResult:
    input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
    task._mark_milked_if_changed(world.ram)
    held_now = read_held_item(world.ram)
    if input_lock != 1 or task._player_action(world.ram) != 0:
        task._interaction_started = True
    milk_done = task._target_cow_slot is None or not cow_needs_milking(world.ram, task._target_cow_slot)
    if milk_done and held_now:
        if input_lock == 1:
            task._phase = CowPhase.MILK_SHIP_NAV
            task._ship_route_index = 0
            task._verify_count = 0
            task._clear_navigation()
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
    task._verify_count += 1
    if (task._interaction_started and input_lock == 1 and task._verify_count > 20) or task._verify_count > 110:
        if task._target_cow_slot is not None and not cow_needs_milking(world.ram, task._target_cow_slot):
            if read_held_item(world.ram):
                task._phase = CowPhase.MILK_SHIP_NAV
                task._ship_route_index = 0
                task._verify_count = 0
                task._clear_navigation()
                return TaskResult(status=TaskStatus.RUNNING)
            return task._after_milk(world.ram)
        if task._milk_attempts < MAX_MILK_ATTEMPTS and task._milker_in_carry_pair(world.ram):
            print(f"[COW] Milk retry slot={task._target_cow_slot} attempts={task._milk_attempts}")
            task._refresh_talk_approach(world.ram)
            task._phase = CowPhase.MILK_NAV if task._milker_selected(world.ram) else "milk_select"
            task._brush_route_index = max(0, len(task._talk_route()) - 1)
            task._milk_select_frames = 0
            task._verify_count = 0
            task._interaction_started = False
            # Keep the original slot timer so retries cannot outrun the
            # external stall watchdog by resetting every attempt.
            task._clear_navigation()
            task._reset_pixel_nav_progress()
            return TaskResult(status=TaskStatus.RUNNING)
        if task._defer_current_milk(world.ram, "attempts"):
            task._verify_count = 0
            task._interaction_started = False
            task._clear_navigation()
            return task._after_milk(world.ram)
        if task._target_cow_slot in task._milk_slots:
            print(f"[COW] Milk skipped slot={task._target_cow_slot} attempts={task._milk_attempts}")
            task._milk_slots.remove(task._target_cow_slot)
        if task._target_cow_slot is not None:
            task._skipped_milk_slots.add(task._target_cow_slot)
        return task._after_milk(world.ram)
    action = task._dialog_pulse_action() if task._interaction_started else make_action()
    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

def _step_milk_ship_nav(task, world: WorldState) -> TaskResult:
    if read_held_item(world.ram) == 0:
        return task._after_milk(world.ram)
    if (
        task._navigator.current_tile == BARN_SHIP_BIN_INTERACT_STAND
        and abs(task._navigator.current_pos.x - MILK_SHIP_PIXEL_ROUTE[-1][0]) <= 3
        and abs(task._navigator.current_pos.y - MILK_SHIP_PIXEL_ROUTE[-1][1]) <= 3
    ):
        task._ship_money_before = read_shipping_money(world.ram)
        task._queue_press_a(
            BARN_SHIP_BIN_FACE,
            face_frames=8,
            hold_frames=16,
            settle_frames=24,
        )
        task._verify_count = 0
        task._phase = CowPhase.MILK_SHIP_VERIFY
        return TaskResult(status=TaskStatus.RUNNING)
    action = task._milk_ship_pixel_action()
    if action is not None:
        task._clear_navigation()
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(left=True, b=True)))

def _step_milk_ship_verify(task, world: WorldState) -> TaskResult:
    money_now = read_shipping_money(world.ram)
    if money_now > task._ship_money_before:
        task.milk_shipped_count += 1
        print(f"[COW] Milk shipped money={money_now}")
        return task._after_milk(world.ram)
    if read_held_item(world.ram) == 0:
        task.milk_shipped_count += 1
        print("[COW] Milk shipped")
        return task._after_milk(world.ram)
    task._verify_count += 1
    if task._verify_count > 30:
        task._phase = CowPhase.MILK_SHIP_NAV
        task._verify_count = 0
        task._clear_navigation()
    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))


def bind_task_methods(cls: type) -> None:
    """Attach these phase functions to the task class. Not a mixin."""
    for name, fn in list(globals().items()):
        if not name.startswith("_") or not callable(fn):
            continue
        if getattr(fn, "__module__", None) != __name__:
            continue
        setattr(cls, name, fn)

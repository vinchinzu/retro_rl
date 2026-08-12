"""Milk + ship phase arms for CowChoresTask (rr-y80y)."""

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
from harvest.tasks.cow_fsm import MAX_MILK_ATTEMPTS, MAX_MILK_DEFERRALS, CowPhase
from harvest.tasks.cow_geometry import BARN_SHIP_BIN_FACE, BARN_SHIP_BIN_INTERACT_STAND, MILK_SHIP_PIXEL_ROUTE
from harvest.tasks.cow_care import milk_ship_escape_prefix_action, milk_ship_route_step_action
from harvest.tasks.harvest_task import read_shipping_money
from harvest.tasks.nav import make_action
from retro_harness import ActionResult, TaskResult, TaskStatus, WorldState


class CowMilkMixin:
    """Milk select/nav/verify and barn-bin shipping."""

    def _milk_ship_pixel_action(self) -> Optional[np.ndarray]:
        x = self._navigator.current_pos.x
        y = self._navigator.current_pos.y
        prefix = milk_ship_escape_prefix_action(x, y, ship_route_index=self._ship_route_index)
        if prefix is not None:
            return prefix

        index = min(self._ship_route_index, len(MILK_SHIP_PIXEL_ROUTE) - 1)
        target = MILK_SHIP_PIXEL_ROUTE[index]
        if abs(x - target[0]) <= 2 and abs(y - target[1]) <= 2:
            if self._ship_route_index < len(MILK_SHIP_PIXEL_ROUTE) - 1:
                self._ship_route_index += 1
                return make_action()
            return None

        return milk_ship_route_step_action(x, y, index)

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
        self._phase = CowPhase.MILK_NAV if self._milker_selected(ram) else "milk_select"
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
        self._phase = CowPhase.MILK_VERIFY
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
            self._phase = CowPhase.FEED_PLACE_NAV if read_held_item(ram) == ITEM_FODDER else "fodder_nav"
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

    def _step_milk_select(self, world: WorldState) -> TaskResult:
        if not self._milker_in_carry_pair(world.ram):
            return self._after_milk(world.ram)
        if self._milker_selected(world.ram):
            self._phase = CowPhase.MILK_NAV
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

    def _step_milk_nav(self, world: WorldState) -> TaskResult:
        if self._target_cow_slot is None or not cow_needs_milking(world.ram, self._target_cow_slot):
            return self._after_milk(world.ram)
        if not self._milker_in_carry_pair(world.ram):
            return self._after_milk(world.ram)
        if not self._milker_selected(world.ram):
            self._phase = CowPhase.MILK_SELECT
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

    def _step_milk_verify(self, world: WorldState) -> TaskResult:
        input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
        self._mark_milked_if_changed(world.ram)
        held_now = read_held_item(world.ram)
        if input_lock != 1 or self._player_action(world.ram) != 0:
            self._interaction_started = True
        milk_done = self._target_cow_slot is None or not cow_needs_milking(world.ram, self._target_cow_slot)
        if milk_done and held_now:
            if input_lock == 1:
                self._phase = CowPhase.MILK_SHIP_NAV
                self._ship_route_index = 0
                self._verify_count = 0
                self._clear_navigation()
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
        self._verify_count += 1
        if (self._interaction_started and input_lock == 1 and self._verify_count > 20) or self._verify_count > 110:
            if self._target_cow_slot is not None and not cow_needs_milking(world.ram, self._target_cow_slot):
                if read_held_item(world.ram):
                    self._phase = CowPhase.MILK_SHIP_NAV
                    self._ship_route_index = 0
                    self._verify_count = 0
                    self._clear_navigation()
                    return TaskResult(status=TaskStatus.RUNNING)
                return self._after_milk(world.ram)
            if self._milk_attempts < MAX_MILK_ATTEMPTS and self._milker_in_carry_pair(world.ram):
                print(f"[COW] Milk retry slot={self._target_cow_slot} attempts={self._milk_attempts}")
                self._refresh_talk_approach(world.ram)
                self._phase = CowPhase.MILK_NAV if self._milker_selected(world.ram) else "milk_select"
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

    def _step_milk_ship_nav(self, world: WorldState) -> TaskResult:
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
            self._phase = CowPhase.MILK_SHIP_VERIFY
            return TaskResult(status=TaskStatus.RUNNING)
        action = self._milk_ship_pixel_action()
        if action is not None:
            self._clear_navigation()
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(left=True, b=True)))

    def _step_milk_ship_verify(self, world: WorldState) -> TaskResult:
        money_now = read_shipping_money(world.ram)
        if money_now > self._ship_money_before:
            self.milk_shipped_count += 1
            print(f"[COW] Milk shipped money={money_now}")
            return self._after_milk(world.ram)
        if read_held_item(world.ram) == 0:
            self.milk_shipped_count += 1
            print("[COW] Milk shipped")
            return self._after_milk(world.ram)
        self._verify_count += 1
        if self._verify_count > 30:
            self._phase = CowPhase.MILK_SHIP_NAV
            self._verify_count = 0
            self._clear_navigation()
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

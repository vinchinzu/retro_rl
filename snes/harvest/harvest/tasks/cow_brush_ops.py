"""Brush phase arms for CowChoresTask (rr-y80y)."""

from __future__ import annotations

import numpy as np

from harvest.core.animal_status import COW_DAILY_BRUSHED_FLAG, read_cow_daily_flags, read_cow_happiness
from harvest.core.tile_catalog import ADDR_INPUT_LOCK
from harvest.tasks.cow_fsm import MAX_BRUSH_ATTEMPTS, CowPhase
from harvest.tasks.nav import make_action
from retro_harness import ActionResult, TaskResult, TaskStatus, WorldState


class CowBrushMixin:
    """Brush select / nav / verify phase methods."""

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
        self._phase = CowPhase.BRUSH_VERIFY
        action = self._action_queue.popleft() if self._action_queue else None
        if action is not None:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
        return TaskResult(status=TaskStatus.RUNNING)

    def _after_brush(self, ram: np.ndarray) -> TaskResult:
        if self._target_cow_slot is not None and self._slot_needs_milk(ram, self._target_cow_slot):
            if self._target_cow_slot in self._skipped_brush_slots:
                self._prefer_body_side_stand(ram)
            self._milk_select_frames = 0
            self._milk_attempts = 0
            self._verify_count = 0
            self._interaction_started = False
            self._phase = CowPhase.MILK_NAV if self._milker_selected(ram) else "milk_select"
            return TaskResult(status=TaskStatus.RUNNING)
        if self._begin_next_cow_care(ram):
            return TaskResult(status=TaskStatus.RUNNING)
        return self._after_milk(ram)

    def _step_brush_select(self, world: WorldState) -> TaskResult:
        if not self._brush_in_carry_pair(world.ram):
            return self._after_brush(world.ram)
        if self._brush_selected(world.ram):
            self._phase = CowPhase.BRUSH_NAV
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

    def _step_brush_nav(self, world: WorldState) -> TaskResult:
        if not self._brush_in_carry_pair(world.ram):
            return self._after_brush(world.ram)
        if not self._brush_selected(world.ram):
            self._phase = CowPhase.BRUSH_SELECT
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

    def _step_brush_verify(self, world: WorldState) -> TaskResult:
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
                self._phase = CowPhase.BRUSH_NAV if self._brush_selected(world.ram) else "brush_select"
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

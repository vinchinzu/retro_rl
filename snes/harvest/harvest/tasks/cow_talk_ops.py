"""Talk phase arms + care-slot handoff for CowChoresTask (rr-y80y)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from harvest.core.animal_status import (
    COW_DAILY_TALKED_FLAG,
    read_cow_daily_flags,
    read_cow_happiness,
)
from harvest.core.tile_catalog import ADDR_INPUT_LOCK
from harvest.tasks.cow_fsm import MAX_TALK_ATTEMPTS, CowPhase
from harvest.tasks.nav import make_action
from retro_harness import ActionResult, TaskResult, TaskStatus, WorldState


class CowTalkMixin:
    """Talk-nav / talk-verify and begin-next-care orchestration."""

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

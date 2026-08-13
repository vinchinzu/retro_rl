"""Navigation, tool queue, and care defer helpers for CowChoresTask."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from harvest.core.animal_probe import cow_tiles_from_slots
from harvest.core.animal_status import read_cow_daily_flags, read_cow_happiness
from harvest.core.npc_catalog import game_objects
from harvest.core.ram_catalog import read_ram_u8
from harvest.tasks.animal_navigation import fallback_action, find_path_around_blockers
from harvest.tasks.cow_care import (
    left_lower_lane_from_right_action,
    left_side_vertical_nav_action,
    recorded_interact_lane_action,
    run_to_pixel_axis,
)
from harvest.tasks.cow_fsm import (
    ADDR_PLAYER_ACTION,
    ADDR_TOOL_BACKPACK,
    ADDR_TOOL_SELECTED,
    BRUSH_TOOL_ID,
    MAX_CARE_DEFERRALS,
    MAX_NAV_FALLBACK_FRAMES,
    MAX_PIXEL_NAV_STALLS,
    MILK_CARE_PHASES,
    MILKER_TOOL_ID,
    PIXEL_NAV_STALL_FRAMES,
    TOOL_CARE_PHASES,
)
from harvest.tasks.cow_geometry import (
    CARE_TROUGH_EXIT_ANCHOR_X,
    CARE_TROUGH_EXIT_BOTTOM_Y,
    CARE_TROUGH_EXIT_MIN_Y,
    CARE_TROUGH_EXIT_X,
    LEFT_TROUGH_RETURN_X,
    body_side_stand_candidates,
    left_cow_lane_x,
    stand_blocked,
    stand_in_bounds,
    talk_route_to,
)
from harvest.tasks.nav import MAP_WIDTH, make_action
from harvest.tasks.primitives import press_a_sequence
from retro_harness import ActionResult, TaskResult, TaskStatus


class CowNavMixin:
    """Pixel/route navigation, tool presses, and care skip/defer."""

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

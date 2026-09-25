"""Navigation, tool queue, brush phases, and care defer for CowChoresTask."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from harvest.core.animal_probe import cow_tiles_from_slots
from harvest.core.animal_status import (
    COW_DAILY_BRUSHED_FLAG,
    read_cow_daily_flags,
    read_cow_happiness,
)
from harvest.core.tile_catalog import ADDR_INPUT_LOCK
from harvest.core.npc_catalog import game_objects
from harvest.core.ram_catalog import read_ram_u8
from harvest.tasks.animal_navigation import fallback_action, find_path_around_blockers
from harvest.tasks.cow_care import (
    exit_prep_escape_action,
    left_lower_lane_from_right_action,
    left_side_vertical_nav_action,
    recorded_interact_lane_action,
    run_to_pixel_axis,
)
from harvest.tasks.cow_task import (
    ADDR_PLAYER_ACTION,
    ADDR_TOOL_BACKPACK,
    ADDR_TOOL_SELECTED,
    BRUSH_TOOL_ID,
    MAX_BRUSH_ATTEMPTS,
    MAX_CARE_DEFERRALS,
    MAX_EXIT_PREP_FRAMES,
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
    COW_EXIT_PREP_STAND,
    LEFT_TROUGH_RETURN_X,
    body_side_stand_candidates,
    left_cow_lane_x,
    stand_blocked,
    stand_in_bounds,
    talk_route_to,
)
from harvest.tasks.nav import MAP_WIDTH, make_action
from harvest.tasks.primitives import press_a_sequence
from retro_harness import ActionResult, TaskResult, TaskStatus, WorldState

def _refresh_talk_approach(task, ram: np.ndarray) -> None:
    stand, face = task._candidate_cow_stands(ram)[0]
    if stand != task._talk_stand:
        task._clear_navigation()
    task._talk_stand = stand
    task._talk_face = face

def _talk_route(task) -> Tuple[Tuple[int, int], ...]:
    return talk_route_to(task._talk_stand)

def _refresh_stale_cow_approach(task, ram: np.ndarray, index_attr: str) -> None:
    if task._target_cow_slot is None:
        return
    if task._is_adjacent_to_target_cow(ram, task._talk_stand, task._talk_face):
        return
    task._refresh_talk_approach(ram)
    setattr(task, index_attr, max(0, len(task._talk_route()) - 1))

def _cow_ram_changed(task, ram: np.ndarray, flag: int, before_flags: int, before_happiness: int) -> bool:
    if task._target_cow_slot is None:
        return False
    flags_now = read_cow_daily_flags(ram, task._target_cow_slot)
    happiness_now = read_cow_happiness(ram, task._target_cow_slot)
    if before_flags & flag:
        return True
    return bool((flags_now & flag) and (flags_now != before_flags or happiness_now > before_happiness))

def _selected_tool(task, ram: np.ndarray) -> int:
    return read_ram_u8(ram, ADDR_TOOL_SELECTED)

def _backpack_tool(task, ram: np.ndarray) -> int:
    return read_ram_u8(ram, ADDR_TOOL_BACKPACK)

def _player_action(task, ram: np.ndarray) -> int:
    return read_ram_u8(ram, ADDR_PLAYER_ACTION)

def _brush_selected(task, ram: np.ndarray) -> bool:
    return task._selected_tool(ram) == BRUSH_TOOL_ID

def _brush_in_carry_pair(task, ram: np.ndarray) -> bool:
    return task._selected_tool(ram) == BRUSH_TOOL_ID or task._backpack_tool(ram) == BRUSH_TOOL_ID

def _milker_selected(task, ram: np.ndarray) -> bool:
    return task._selected_tool(ram) == MILKER_TOOL_ID

def _milker_in_carry_pair(task, ram: np.ndarray) -> bool:
    return task._selected_tool(ram) == MILKER_TOOL_ID or task._backpack_tool(ram) == MILKER_TOOL_ID

def _queue_press_a(
    task,
    face: str,
    *,
    face_frames: int = 8,
    hold_frames: int = 20,
    settle_frames: int = 18,
    hold_face_with_a: bool = True,
) -> None:
    task._action_queue.extend(
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
    task,
    face: str,
    *,
    face_frames: int = 0,
    hold_frames: int = 22,
    y_only_frames: int = 0,
    settle_frames: int = 20,
    hold_face_with_y: bool = True,
) -> None:
    task._action_queue.extend(make_action(**{face: True}) for _ in range(face_frames))
    if hold_face_with_y:
        task._action_queue.extend(make_action(**{face: True, "y": True}) for _ in range(hold_frames))
    else:
        task._action_queue.extend(make_action(y=True) for _ in range(hold_frames))
    task._action_queue.extend(make_action(y=True) for _ in range(y_only_frames))
    task._action_queue.extend(make_action() for _ in range(settle_frames))

def _clear_navigation(task) -> None:
    task._navigator.path = []
    task._navigator.stasis = 0
    task._pathfinder.temp_blocked.clear()
    task._nav_failures = 0

def _reset_pixel_nav_progress(task) -> None:
    task._pixel_nav_target = None
    task._pixel_nav_best_dist = 10**9
    task._pixel_nav_stale_frames = 0

def _pixel_nav_stalled(task, target: Tuple[int, int]) -> bool:
    """Detect sub-tile oscillation that keeps Navigator.stasis at 0."""
    x = task._navigator.current_pos.x
    y = task._navigator.current_pos.y
    dist = abs(x - target[0]) + abs(y - target[1])
    if task._pixel_nav_target != target:
        task._pixel_nav_target = target
        task._pixel_nav_best_dist = dist
        task._pixel_nav_stale_frames = 0
        return False
    if dist + 1 < task._pixel_nav_best_dist:
        task._pixel_nav_best_dist = dist
        task._pixel_nav_stale_frames = 0
        return False
    task._pixel_nav_stale_frames += 1
    return task._pixel_nav_stale_frames >= PIXEL_NAV_STALL_FRAMES

def _handle_pixel_nav_action(
    task,
    ram: np.ndarray,
    action: Optional[np.ndarray],
    *,
    tool: bool,
) -> Optional[TaskResult]:
    """Apply recorded pixel-lane action, or escalate when it stops closing."""
    if action is None:
        return None
    target = task._cow_interact_pixel(ram, tool=tool)
    if target is not None and task._pixel_nav_stalled(target):
        task._pixel_nav_stall_count += 1
        print(
            f"[COW] Pixel nav stall slot={task._target_cow_slot} "
            f"count={task._pixel_nav_stall_count} target={target} "
            f"{task._care_debug_context(ram)}"
        )
        task._reset_pixel_nav_progress()
        task._clear_navigation()
        if task._pixel_nav_stall_count >= MAX_PIXEL_NAV_STALLS:
            task._pixel_nav_stall_count = 0
            return task._skip_current_cow_care(ram, "pixel_nav_stall")
        task._refresh_talk_approach(ram)
        task._talk_route_index = max(0, len(task._talk_route()) - 1)
        task._brush_route_index = task._talk_route_index
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
    task._clear_navigation()
    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

def _care_debug_context(task, ram: np.ndarray) -> str:
    tool = task._phase in TOOL_CARE_PHASES
    return (
        f"phase={task._phase} pos=({task._navigator.current_pos.x},{task._navigator.current_pos.y}) "
        f"tile={task._navigator.current_tile} cow_tile={task._target_cow_tile(ram)} "
        f"cow_px={task._target_cow_pixel(ram)} stand={task._talk_stand} face={task._talk_face} "
        f"interact_px={task._cow_interact_pixel(ram, tool=tool)} "
        f"route_idx=t{task._talk_route_index}/b{task._brush_route_index} "
        f"path_next={task._navigator.path[0] if task._navigator.path else None} "
        f"stasis={task._navigator.stasis} nav_failures={task._nav_failures}"
    )

def _dialog_pulse_action(task) -> np.ndarray:
    """Tap A with gaps so modal text advances instead of treating A as held."""
    cycle = task._verify_count % 22
    return make_action(a=6 <= cycle < 12)

def _run_to_pixel_axis(
    task,
    target: Tuple[int, int],
    *,
    tolerance: int = 2,
    x_first: bool = False,
    y_first: bool = False,
) -> Optional[np.ndarray]:
    return run_to_pixel_axis(
        (task._navigator.current_pos.x, task._navigator.current_pos.y),
        target,
        tolerance=tolerance,
        x_first=x_first,
        y_first=y_first,
    )

def _left_cow_lane_x(task, current_y: int) -> int:
    return left_cow_lane_x(current_y)

def _left_lower_lane_from_right_action(task) -> Optional[np.ndarray]:
    return left_lower_lane_from_right_action(
        task._navigator.current_pos.x,
        task._navigator.current_pos.y,
    )

def _left_side_vertical_nav_action(
    task,
    x: int,
    y: int,
    tx: int,
    ty: int,
    *,
    going_down: bool,
) -> Optional[np.ndarray]:
    """Reach wall-side interact pixels via the recorded left vertical lane."""
    return left_side_vertical_nav_action(x, y, tx, ty, going_down=going_down)

def _recorded_interact_nav_action(task, ram: np.ndarray, *, tool: bool) -> Optional[np.ndarray]:
    if task._talk_face not in ("left", "right"):
        return None
    target = task._cow_interact_pixel(ram, tool=tool)
    if target is None:
        return None

    tx, ty = target
    x = task._navigator.current_pos.x
    y = task._navigator.current_pos.y
    if abs(x - tx) <= 1 and abs(y - ty) <= 1:
        return None
    # Talk only: already beside the cow, let fine align / A-press finish.
    # Tool use still needs recorded nav to the exact interact pixel.
    if (
        not tool
        and task._is_adjacent_to_target_cow(
            ram, task._navigator.current_tile, task._talk_face
        )
        and abs(x - tx) <= 16
        and abs(y - ty) <= 16
    ):
        return None

    return recorded_interact_lane_action(x, y, tx, ty, face=task._talk_face)

def _care_trough_exit_action(task, ram: np.ndarray) -> Optional[np.ndarray]:
    x = task._navigator.current_pos.x
    y = task._navigator.current_pos.y
    if x < CARE_TROUGH_EXIT_X - 18 or x > LEFT_TROUGH_RETURN_X:
        return None
    if y < CARE_TROUGH_EXIT_MIN_Y:
        return None
    # Lower corridor + left-wall care targets: do not yank back to the
    # right aisle anchor (that fought pixel nav at ~x=129,y=345).
    target = task._cow_interact_pixel(ram, tool=False)
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
    if not task._care_trough_exit_logged:
        print(
            f"[COW] Care trough exit slot={task._target_cow_slot} "
            f"anchor=({CARE_TROUGH_EXIT_ANCHOR_X},{CARE_TROUGH_EXIT_BOTTOM_Y}) "
            f"{task._care_debug_context(ram)}"
        )
        task._care_trough_exit_logged = True
    task._clear_navigation()
    return action

def _recorded_left_tool_nav_action(task, ram: np.ndarray) -> Optional[np.ndarray]:
    return task._recorded_interact_nav_action(ram, tool=True)

def _navigate_route(
    task,
    ram: np.ndarray,
    route: Tuple[Tuple[int, int], ...],
    index_attr: str,
    *,
    center_final: bool = True,
) -> Optional[np.ndarray]:
    index = int(getattr(task, index_attr))
    target = route[min(index, len(route) - 1)]
    if index < len(route) - 1 and task._navigator.current_tile == target:
        setattr(task, index_attr, index + 1)
        task._clear_navigation()
        return make_action()
    if index == len(route) - 1 and task._navigator.current_tile == target and not center_final:
        task._clear_navigation()
        return None

    action = task._navigate_to_tile(ram, target)
    if action is not None:
        return action

    if index < len(route) - 1:
        setattr(task, index_attr, index + 1)
        task._clear_navigation()
        return make_action()
    return None

def _can_reach_talk_stand_directly(task, ram: np.ndarray) -> bool:
    return task._find_path_around_cows(
        ram,
        task._navigator.current_tile,
        task._talk_stand,
    ) is not None

def _pin_care_route_to_direct_stand(task, ram: np.ndarray) -> None:
    if task._can_reach_talk_stand_directly(ram):
        direct_index = max(0, len(task._talk_route()) - 1)
        task._talk_route_index = direct_index
        task._brush_route_index = direct_index

def _prefer_body_side_stand(task, ram: np.ndarray) -> bool:
    tile = task._target_cow_tile(ram)
    if tile is None:
        return False
    cx, cy = tile
    cow_tiles = task._cow_tiles(ram)
    for stand, face in body_side_stand_candidates(cx, cy):
        sx, sy = stand
        if not stand_in_bounds(stand):
            continue
        if stand_blocked(stand, cow_tiles):
            continue
        if not task._is_adjacent_to_target_cow(ram, stand, face):
            continue
        if not task._pathfinder.is_walkable(ram, sx, sy, current_pos=task._navigator.current_tile):
            continue
        if task._find_path_around_cows(ram, task._navigator.current_tile, stand) is None:
            continue
        task._talk_stand = stand
        task._talk_face = face
        task._talk_route_index = max(0, len(task._talk_route()) - 1)
        task._brush_route_index = task._talk_route_index
        return True
    return False

def _base_cow_tiles(task, ram: np.ndarray) -> set[Tuple[int, int]]:
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

def _cow_tiles(task, ram: np.ndarray) -> set[Tuple[int, int]]:
    tiles = task._base_cow_tiles(ram)
    expanded = set(tiles)
    for tx, ty in tiles:
        if 0 <= ty + 1 < MAP_WIDTH:
            expanded.add((tx, ty + 1))
    return expanded

def _find_path_around_cows(
    task,
    ram: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[list[Tuple[int, int]]]:
    blocked = task._cow_tiles(ram)
    blocked.update(task._pathfinder.temp_blocked)
    blocked.discard(goal)
    return find_path_around_blockers(
        ram,
        task._pathfinder,
        start,
        goal,
        blocked,
    )

def _navigate_to_tile(task, ram: np.ndarray, goal: Tuple[int, int]) -> Optional[np.ndarray]:
    if task._navigator.current_tile == goal or task._navigator.at_tile(goal):
        task._nav_failures = 0
        return task._navigator.center_on_tile(goal, tolerance=1)

    cow_tiles = task._cow_tiles(ram)
    cow_tiles.discard(task._navigator.current_tile)
    cow_tiles.discard(goal)
    if task._navigator.path and task._navigator.path[0] in cow_tiles:
        task._navigator.path = []
        return make_action()

    if task._navigator.stasis > 90 and task._navigator.path:
        task._pathfinder.temp_blocked.add(task._navigator.path[0])
        task._navigator.path = []

    if not task._navigator.path:
        path = task._find_path_around_cows(ram, task._navigator.current_tile, goal)
        if path is None:
            task._nav_failures += 1
            if task._nav_failures > MAX_NAV_FALLBACK_FRAMES:
                return make_action()
            return fallback_action(task._navigator.current_tile, goal)
        task._nav_failures = 0
        task._navigator.path = path

    action = task._navigator.follow_path(ram)
    if action is None:
        task._nav_failures += 1
        if task._nav_failures > MAX_NAV_FALLBACK_FRAMES:
            return make_action()
        return fallback_action(task._navigator.current_tile, goal)
    task._nav_failures = 0
    return action

def _defer_pending_slot(
    task,
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

def _defer_current_care(task, ram: np.ndarray, reason: str) -> bool:
    slot = task._target_cow_slot
    if slot is None or not task._slot_needs_care(ram, slot):
        return False
    if not task._defer_pending_slot(
        task._care_slots,
        task._deferred_care_counts,
        slot,
        max_deferrals=MAX_CARE_DEFERRALS,
    ):
        return False
    print(
        f"[COW] Care deferred slot={slot} reason={reason} "
        f"count={task._deferred_care_counts[slot]}"
    )
    return True

def _skip_current_cow_care(task, ram: np.ndarray, reason: str) -> TaskResult:
    slot = task._target_cow_slot
    retryable = reason in {"slot_timeout", "nav_unreachable", "pixel_nav_stall"}
    task._pixel_nav_stall_count = 0
    task._reset_pixel_nav_progress()
    if retryable and task._phase in MILK_CARE_PHASES:
        if task._defer_current_milk(ram, reason):
            task._verify_count = 0
            task._interaction_started = False
            task._clear_navigation()
            return task._after_milk(ram)
    if retryable and task._phase not in MILK_CARE_PHASES:
        if task._defer_current_care(ram, reason):
            task._verify_count = 0
            task._interaction_started = False
            task._clear_navigation()
            if task._begin_next_cow_care(ram):
                return TaskResult(status=TaskStatus.RUNNING)
            return task._after_milk(ram)
    if slot is not None:
        if task._slot_needs_talk(ram, slot):
            task._skipped_talk_slots.add(slot)
        if task._slot_needs_brush(ram, slot):
            task._skipped_brush_slots.add(slot)
        if task._slot_needs_milk(ram, slot):
            task._skipped_milk_slots.add(slot)
        print(f"[COW] Care skipped slot={slot} reason={reason} {task._care_debug_context(ram)}")
    task._verify_count = 0
    task._interaction_started = False
    task._clear_navigation()
    if task._begin_next_cow_care(ram):
        return TaskResult(status=TaskStatus.RUNNING)
    return task._after_milk(ram)

def _mark_brushed_if_changed(task, ram: np.ndarray) -> None:
    if task.brushed:
        return
    if task._cow_ram_changed(
        ram,
        COW_DAILY_BRUSHED_FLAG,
        task._brush_flags_before,
        task._brush_happiness_before,
    ):
        print(f"[COW] Brush OK slot={task._target_cow_slot} attempts={task._brush_attempts}")
        task.brushed = True
        task._remember_current_pin()

def _begin_brush_verify(task, ram: np.ndarray) -> TaskResult:
    if task._target_cow_slot is None:
        return TaskResult(status=TaskStatus.FAILURE, reason="no target cow slot for brush")
    task._brush_flags_before = read_cow_daily_flags(ram, task._target_cow_slot)
    task._brush_happiness_before = read_cow_happiness(ram, task._target_cow_slot)
    task.brushed = bool(task._brush_flags_before & COW_DAILY_BRUSHED_FLAG)
    task._clear_navigation()
    task._queue_use_tool(
        task._talk_face,
        face_frames=10,
        hold_frames=18,
        y_only_frames=2,
        settle_frames=75,
    )
    task._brush_attempts += 1
    task._verify_count = 0
    task._interaction_started = False
    task._phase = CowPhase.BRUSH_VERIFY
    action = task._action_queue.popleft() if task._action_queue else None
    if action is not None:
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
    return TaskResult(status=TaskStatus.RUNNING)

def _after_brush(task, ram: np.ndarray) -> TaskResult:
    if task._target_cow_slot is not None and task._slot_needs_milk(ram, task._target_cow_slot):
        if task._target_cow_slot in task._skipped_brush_slots:
            task._prefer_body_side_stand(ram)
        task._milk_select_frames = 0
        task._milk_attempts = 0
        task._verify_count = 0
        task._interaction_started = False
        task._phase = CowPhase.MILK_NAV if task._milker_selected(ram) else "milk_select"
        return TaskResult(status=TaskStatus.RUNNING)
    if task._begin_next_cow_care(ram):
        return TaskResult(status=TaskStatus.RUNNING)
    return task._after_milk(ram)

def _step_brush_select(task, world: WorldState) -> TaskResult:
    if not task._brush_in_carry_pair(world.ram):
        return task._after_brush(world.ram)
    if task._brush_selected(world.ram):
        task._phase = CowPhase.BRUSH_NAV
        task._brush_select_frames = 0
        face = task._face_for_target_cow(world.ram, task._navigator.current_tile)
        if task._is_adjacent_to_target_cow(world.ram, task._navigator.current_tile, face):
            task._talk_face = face
            task._brush_route_index = max(0, len(task._talk_route()) - 1)
        else:
            task._brush_route_index = 0
            task._pin_care_route_to_direct_stand(world.ram)
        task._clear_navigation()
        return TaskResult(status=TaskStatus.RUNNING)
    if task._player_action(world.ram) != 0:
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
    task._brush_select_frames += 1
    if task._brush_select_frames > 60:
        if task._target_cow_slot is not None:
            task._skipped_brush_slots.add(task._target_cow_slot)
            print(f"[COW] Brush skipped slot={task._target_cow_slot} attempts=select_timeout")
        return task._after_brush(world.ram)
    action = make_action(x=True) if task._brush_select_frames % 6 == 1 else make_action()
    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

def _step_brush_nav(task, world: WorldState) -> TaskResult:
    if not task._brush_in_carry_pair(world.ram):
        return task._after_brush(world.ram)
    if not task._brush_selected(world.ram):
        task._phase = CowPhase.BRUSH_SELECT
        task._brush_select_frames = 0
        return TaskResult(status=TaskStatus.RUNNING)
    task._talk_face = task._face_for_target_cow(world.ram)
    action = task._care_trough_exit_action(world.ram)
    if action is not None:
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
    action = task._recorded_left_tool_nav_action(world.ram)
    handled = task._handle_pixel_nav_action(world.ram, action, tool=True)
    if handled is not None:
        return handled
    if task._brush_route_index >= 1:
        task._refresh_stale_cow_approach(world.ram, "_brush_route_index")
    if (
        task._brush_route_index >= 1
        and task._navigator.current_tile != task._talk_stand
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
    if task._target_cow_slot is None:
        return TaskResult(status=TaskStatus.FAILURE, reason="no target cow slot for brush")
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
        task._refresh_talk_approach(world.ram)
        task._brush_route_index = max(0, len(task._talk_route()) - 1)
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
    return task._begin_brush_verify(world.ram)

def _step_brush_verify(task, world: WorldState) -> TaskResult:
    input_lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
    task._mark_brushed_if_changed(world.ram)
    if input_lock != 1 or task._player_action(world.ram) != 0:
        task._interaction_started = True
    if task.brushed and (not task._interaction_started or input_lock == 1):
        return task._after_brush(world.ram)
    task._verify_count += 1
    if (task._interaction_started and input_lock == 1 and task._verify_count > 20) or task._verify_count > 90:
        if task._brush_attempts < MAX_BRUSH_ATTEMPTS and task._brush_in_carry_pair(world.ram):
            print(f"[COW] Brush retry slot={task._target_cow_slot} attempts={task._brush_attempts}")
            if task._brush_attempts < 2 or not task._prefer_body_side_stand(world.ram):
                task._refresh_talk_approach(world.ram)
            task._phase = CowPhase.BRUSH_NAV if task._brush_selected(world.ram) else "brush_select"
            task._brush_route_index = max(0, len(task._talk_route()) - 1)
            task._brush_select_frames = 0
            task._verify_count = 0
            task._interaction_started = False
            task._clear_navigation()
            return TaskResult(status=TaskStatus.RUNNING)
        if task._target_cow_slot is not None:
            task._skipped_brush_slots.add(task._target_cow_slot)
            print(f"[COW] Brush skipped slot={task._target_cow_slot} attempts={task._brush_attempts}")
        return task._after_brush(world.ram)
    action = task._dialog_pulse_action() if task._interaction_started else make_action()
    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

def _begin_exit_prep(task) -> None:
    task._exit_prep_started_step = task._step_count
    task._verify_count = 0
    task._clear_navigation()
    task._reset_pixel_nav_progress()
    task._phase = CowPhase.EXIT_PREP_NAV

def _step_exit_prep_nav(task, world: WorldState) -> TaskResult:
    if task._exit_prep_started_step <= 0:
        task._exit_prep_started_step = task._step_count
    if task._step_count - task._exit_prep_started_step > MAX_EXIT_PREP_FRAMES:
        print(
            f"[COW] Exit prep timeout at {task._navigator.current_tile}; "
            "handing off to EXIT_BARN"
        )
        task._phase = CowPhase.DONE
        return TaskResult(status=TaskStatus.RUNNING)
    if (
        task._navigator.current_tile == COW_EXIT_PREP_STAND
        or task._navigator.at_tile(COW_EXIT_PREP_STAND)
    ):
        task._phase = CowPhase.DONE
        return TaskResult(status=TaskStatus.RUNNING)
    action = exit_prep_escape_action(
        task._navigator.current_pos.x,
        task._navigator.current_pos.y,
    )
    if action is not None:
        task._clear_navigation()
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
    action = task._navigate_to_tile(world.ram, COW_EXIT_PREP_STAND)
    if action is not None:
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
    task._phase = CowPhase.DONE
    return TaskResult(status=TaskStatus.RUNNING)


def bind_task_methods(cls: type) -> None:
    """Attach these phase functions to the task class. Not a mixin."""
    for name, fn in list(globals().items()):
        if not name.startswith("_") or not callable(fn):
            continue
        if getattr(fn, "__module__", None) != __name__:
            continue
        setattr(cls, name, fn)

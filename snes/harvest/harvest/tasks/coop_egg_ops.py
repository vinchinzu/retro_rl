"""Egg collect / incubate / ship / exit-prep phase functions for CoopChoresTask."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from harvest.core.animal_probe import chicken_slot_snapshots
from harvest.core.animal_status import (
    INCUBATOR_EGG_TILES,
    ITEM_EGG,
    chicken_slot_eggs_available,
    count_chicken_slots,
    egg_available_today,
    is_holding_egg,
    is_incubating,
    read_egg_available_flags,
    read_item_on_hand,
)
from harvest.core.npc_catalog import game_objects
from harvest.tasks.coop_layout import (
    CHICKEN_EGG_FLAGS,
    CHICKEN_EGG_SPAWN_PIXELS,
    EGG_PICKUP_SPOTS,
    EXIT_PREP_STAND,
    INCUBATOR_APPROACH,
    INCUBATOR_FACE,
    INCUBATOR_STAND,
    MAX_EGG_ATTEMPTS,
    MAX_EGG_DEFERRALS,
    MAX_EGG_NAV_FRAMES,
    MAX_EXIT_PREP_FRAMES,
    MAX_FLOCK_SIZE,
    SHIP_APPROACH_Y,
    SHIP_BIN_FACE,
    SHIP_BIN_INTERACT_STAND,
    SHIP_BIN_STAND,
    SHIP_INTERACT_PX,
    SHIP_LANE_X,
    SHIP_RIGHT_LANE_CORNER,
    VISIBLE_EGG_SPRITE,
)
from harvest.tasks.harvest_task import read_shipping_money
from harvest.tasks.nav import MAP_WIDTH, TILE_SIZE, make_action
from harvest.tasks.skills import (
    coop_nav_to_shipping_bin_skill,
    coop_press_egg_pickup_skill,
    coop_press_incubator_skill,
    coop_press_ship_skill,
)
from retro_harness import ActionResult, TaskResult, TaskStatus, WorldState

def _begin_egg_nav(task) -> TaskResult:
    task._egg_attempts = 0
    task._verify_count = 0
    task._egg_nav_started_step = task._step_count
    task._current_egg_flag = 0
    task._current_egg_tile = None
    task._clear_left_top_route()
    task._pathfinder.temp_blocked.clear()
    task._phase = "egg_nav"
    return TaskResult(status=TaskStatus.RUNNING)

def _after_egg_handled(task, ram: np.ndarray) -> TaskResult:
    task._egg_attempts = 0
    task._verify_count = 0
    task._egg_nav_started_step = 0
    task._current_egg_flag = 0
    task._current_egg_tile = None
    if task._collectable_egg_present(ram):
        return task._begin_egg_nav()
    return task._begin_exit_prep()

def _begin_exit_prep(task) -> TaskResult:
    task._verify_count = 0
    task._exit_prep_started_step = task._step_count
    task._exit_prep_route_index = 0
    task._clear_left_top_route()
    task._pathfinder.temp_blocked.clear()
    task._phase = "exit_prep_nav"
    return TaskResult(status=TaskStatus.RUNNING)

def _flagged_egg_tiles(task, ram: np.ndarray) -> set[Tuple[int, int]]:
    flags = read_egg_available_flags(ram)
    tiles: set[Tuple[int, int]] = set()
    for flag, (px, py) in zip(CHICKEN_EGG_FLAGS, CHICKEN_EGG_SPAWN_PIXELS):
        if not (flags & flag):
            continue
        tile = (px // TILE_SIZE, py // TILE_SIZE)
        if is_incubating(ram) and tile in INCUBATOR_EGG_TILES:
            continue
        tiles.add(tile)
    return tiles

def _egg_present(task, ram: np.ndarray) -> bool:
    if egg_available_today(ram) or chicken_slot_eggs_available(ram):
        return True
    incubating = is_incubating(ram)
    for obj in game_objects(ram):
        if obj.sprite_table_idx != VISIBLE_EGG_SPRITE:
            continue
        if incubating and obj.tile in INCUBATOR_EGG_TILES:
            continue
        return True
    return False

def _collectable_egg_present(task, ram: np.ndarray) -> bool:
    """Like `_egg_present`, but ignores egg flags and tiles already skipped this run."""
    flags = read_egg_available_flags(ram)
    for mask, _stand, _face in EGG_PICKUP_SPOTS:
        if (flags & mask) and mask not in task._skipped_egg_flags:
            egg_tile = task._egg_tile_for_flag(mask)
            if egg_tile is None or egg_tile not in task._skipped_egg_tiles:
                return True
    for egg_tile in task._egg_tiles(ram):
        if egg_tile not in task._skipped_egg_tiles:
            return True
    incubating = is_incubating(ram)
    for obj in game_objects(ram):
        if obj.sprite_table_idx != VISIBLE_EGG_SPRITE:
            continue
        if incubating and obj.tile in INCUBATOR_EGG_TILES:
            continue
        if obj.tile not in task._skipped_egg_tiles:
            return True
    return False

def _egg_tiles(task, ram: np.ndarray) -> list[Tuple[int, int]]:
    tiles: list[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    for row in chicken_slot_snapshots(ram, require_coop=True):
        if row.get("stage") != "egg":
            continue
        tile = row.get("tile")
        if not (isinstance(tile, list) and len(tile) == 2):
            continue
        egg_tile = (int(tile[0]), int(tile[1]))
        if is_incubating(ram) and egg_tile in INCUBATOR_EGG_TILES:
            continue
        if egg_tile not in seen:
            seen.add(egg_tile)
            tiles.append(egg_tile)
    for obj in game_objects(ram):
        if obj.sprite_table_idx != VISIBLE_EGG_SPRITE:
            continue
        egg_tile = obj.tile
        if is_incubating(ram) and egg_tile in INCUBATOR_EGG_TILES:
            continue
        if egg_tile not in seen:
            seen.add(egg_tile)
            tiles.append(egg_tile)
    return tiles

def _egg_tile_for_flag(task, flag: int) -> Optional[Tuple[int, int]]:
    for egg_flag, (px, py) in zip(CHICKEN_EGG_FLAGS, CHICKEN_EGG_SPAWN_PIXELS):
        if egg_flag == flag:
            return px // TILE_SIZE, py // TILE_SIZE
    return None

def _stand_candidates_for_egg(
    task,
    egg_tile: Tuple[int, int],
    *,
    preferred: Optional[Tuple[Tuple[int, int], str]] = None,
) -> list[Tuple[Tuple[int, int], str]]:
    x, y = egg_tile
    # Prefer body-side / below stands before same-column traps above the egg.
    geometric = (
        ((x - 1, y), "right"),
        ((x + 1, y), "left"),
        ((x, y + 1), "up"),
        ((x, y - 1), "down"),
    )
    candidates: list[Tuple[Tuple[int, int], str]] = []
    seen: set[Tuple[int, int]] = set()
    if preferred is not None:
        candidates.append(preferred)
        seen.add(preferred[0])
    for stand, face in geometric:
        if stand in seen:
            continue
        seen.add(stand)
        candidates.append((stand, face))
    return candidates

def _stand_for_egg_tile(
    task,
    ram: np.ndarray,
    egg_tile: Tuple[int, int],
    *,
    preferred: Optional[Tuple[Tuple[int, int], str]] = None,
    require_path: bool = True,
) -> Optional[Tuple[Tuple[int, int], str]]:
    blocked = task._chicken_tiles(ram)
    blocked.discard(task._navigator.current_tile)
    current = task._navigator.current_tile
    scored: list[Tuple[Tuple[int, int], Tuple[Tuple[int, int], str]]] = []
    loose: list[Tuple[Tuple[int, int], Tuple[Tuple[int, int], str]]] = []
    for index, (stand, face) in enumerate(
        task._stand_candidates_for_egg(egg_tile, preferred=preferred)
    ):
        sx, sy = stand
        if not (0 <= sx < MAP_WIDTH and 0 <= sy < MAP_WIDTH):
            continue
        if stand in blocked:
            continue
        if not task._pathfinder.is_walkable(
            ram, sx, sy, current_pos=current
        ):
            continue
        distance = abs(sx - current[0]) + abs(sy - current[1])
        if current == stand or task._navigator.at_tile(stand):
            return stand, face
        path = task._find_path_around_chickens(ram, current, stand)
        # Prefer the recording stand when reachable; otherwise shortest path.
        preferred_penalty = (
            0
            if preferred is not None and stand == preferred[0]
            else 1
        )
        if path is not None:
            scored.append(
                ((preferred_penalty, len(path), index, distance), (stand, face))
            )
        else:
            loose.append(
                ((preferred_penalty, index, distance), (stand, face))
            )
    if scored:
        scored.sort(key=lambda row: row[0])
        return scored[0][1]
    if not require_path and loose:
        loose.sort(key=lambda row: row[0])
        return loose[0][1]
    return None

def _egg_pickup_spot(
    task, ram: np.ndarray, *, require_path: bool = True
) -> Optional[Tuple[Tuple[int, int], str]]:
    """Pick a reachable stand for the next floor egg.

    Hardcoded recording stands can be islands once the egg tile itself is
    treated as collision (Spring 22 flag 0x01 → stand (2,4) with walls at
    (2,3)/(2,5)/(3,4)). Prefer geometric side stands with a real path.
    """
    available = read_egg_available_flags(ram)
    preferred_by_flag = {
        mask: (stand, face) for mask, stand, face in EGG_PICKUP_SPOTS
    }
    for mask, _stand, _face in EGG_PICKUP_SPOTS:
        if not (available & mask) or mask in task._skipped_egg_flags:
            continue
        egg_tile = task._egg_tile_for_flag(mask)
        if egg_tile is None or egg_tile in task._skipped_egg_tiles:
            continue
        spot = task._stand_for_egg_tile(
            ram,
            egg_tile,
            preferred=preferred_by_flag.get(mask),
            require_path=require_path,
        )
        if spot is not None:
            task._current_egg_flag = mask
            task._current_egg_tile = egg_tile
            return spot
    for egg_tile in task._egg_tiles(ram):
        if egg_tile in task._skipped_egg_tiles:
            continue
        dynamic_spot = task._stand_for_egg_tile(
            ram, egg_tile, require_path=require_path
        )
        if dynamic_spot is not None:
            task._current_egg_flag = 0
            task._current_egg_tile = egg_tile
            return dynamic_spot
    if require_path:
        return task._egg_pickup_spot(ram, require_path=False)
    task._current_egg_flag = 0
    task._current_egg_tile = None
    return None

def _defer_or_skip_egg(task, reason: str) -> TaskResult:
    flag = task._current_egg_flag
    tile = task._current_egg_tile
    if flag:
        deferred = task._deferred_egg_counts.get(flag, 0)
        if deferred < MAX_EGG_DEFERRALS:
            task._deferred_egg_counts[flag] = deferred + 1
            print(
                f"[COOP] Egg deferred flag=0x{flag:04X} reason={reason} "
                f"count={deferred + 1}"
            )
            task._egg_nav_started_step = task._step_count
            task._egg_attempts = 0
            task._clear_left_top_route()
            task._phase = "egg_nav"
            return TaskResult(status=TaskStatus.RUNNING)
        task._skipped_egg_flags.add(flag)
        if tile:
            task._skipped_egg_tiles.add(tile)
        print(
            f"[COOP] Egg skipped flag=0x{flag:04X} reason={reason}"
        )
    elif tile:
        deferred = task._deferred_egg_tile_counts.get(tile, 0)
        if deferred < MAX_EGG_DEFERRALS:
            task._deferred_egg_tile_counts[tile] = deferred + 1
            print(
                f"[COOP] Egg deferred tile={tile} reason={reason} "
                f"count={deferred + 1}"
            )
            task._egg_nav_started_step = task._step_count
            task._egg_attempts = 0
            task._clear_left_top_route()
            task._phase = "egg_nav"
            return TaskResult(status=TaskStatus.RUNNING)
        task._skipped_egg_tiles.add(tile)
        print(
            f"[COOP] Egg skipped tile={tile} reason={reason}"
        )
    else:
        print(f"[COOP] Egg pickup failed, skipping ({reason})")
    task._egg_attempts = 0
    task._egg_nav_started_step = 0
    task._current_egg_flag = 0
    task._current_egg_tile = None
    task._clear_left_top_route()
    return TaskResult(status=TaskStatus.RUNNING)

def _after_egg_nav_budget(task, ram: np.ndarray, reason: str) -> TaskResult:
    result = task._defer_or_skip_egg(reason)
    if task._collectable_egg_present(ram):
        task._phase = "egg_nav"
        task._egg_nav_started_step = task._step_count
        return result
    return task._begin_exit_prep()

def _ship_pixel_action(task) -> Optional[np.ndarray]:
    x = task._navigator.current_pos.x
    y = task._navigator.current_pos.y
    if y < SHIP_APPROACH_Y:
        if abs(x - SHIP_LANE_X) > 2:
            return make_action(right=x < SHIP_LANE_X, left=x > SHIP_LANE_X, b=True)
        return make_action(down=True, b=True)
    if abs(x - SHIP_INTERACT_PX[0]) > 1:
        return make_action(right=x < SHIP_INTERACT_PX[0], left=x > SHIP_INTERACT_PX[0], b=True)
    if abs(y - SHIP_INTERACT_PX[1]) > 1:
        return make_action(down=y < SHIP_INTERACT_PX[1], up=y > SHIP_INTERACT_PX[1])
    return None

def _near_ship_lane(task) -> bool:
    current = task._navigator.current_tile
    return (
        current in {SHIP_BIN_STAND, SHIP_BIN_INTERACT_STAND, SHIP_RIGHT_LANE_CORNER}
        or current[0] <= SHIP_BIN_STAND[0]
    )

def _navigate_to_ship_stand(task, ram: np.ndarray) -> Optional[np.ndarray]:
    goal = (
        SHIP_RIGHT_LANE_CORNER
        if task._navigator.current_tile[0] >= 3 and task._navigator.current_tile[1] >= 10
        else SHIP_BIN_STAND
    )
    if goal == SHIP_RIGHT_LANE_CORNER and (
        task._navigator.current_tile == goal or task._navigator.at_tile(goal, tolerance=3)
    ):
        task._navigator.path = []
        return None
    if task._navigator.path and task._navigator.path[-1] != goal:
        task._navigator.path = []
    action = task._navigate_to_tile(ram, goal)
    if action is not None:
        return action
    return None

def _sync_incubator_waypoint(task) -> None:
    """Pick a safe incubator waypoint after human/bot handoff."""
    current = task._navigator.current_tile
    if task._navigator.at_tile(INCUBATOR_STAND):
        task._incubator_wp_index = len(INCUBATOR_APPROACH) - 1
    elif current[1] < 10 or (current[0] >= 12 and current[1] < 11) or current[0] > INCUBATOR_STAND[0]:
        task._incubator_wp_index = 0
    elif current[0] < 10:
        task._incubator_wp_index = 1
    elif current[1] < 11:
        task._incubator_wp_index = 1
    else:
        task._incubator_wp_index = 2

def _navigate_to_incubator_stand(task, ram: np.ndarray) -> Optional[np.ndarray]:
    """Approach the incubator from the left, matching the working recording."""
    task._sync_incubator_waypoint()
    while task._incubator_wp_index < len(INCUBATOR_APPROACH):
        goal = INCUBATOR_APPROACH[task._incubator_wp_index]
        action = task._navigate_to_tile(ram, goal)
        if action is not None:
            return action
        if goal == INCUBATOR_STAND:
            return None
        task._incubator_wp_index += 1
        task._navigator.path = []
    return None

def _step_egg_nav(task, world: WorldState) -> TaskResult:
    if not task._collectable_egg_present(world.ram):
        return task._begin_exit_prep()
    if task._egg_nav_started_step <= 0:
        task._egg_nav_started_step = task._step_count
    if task._step_count - task._egg_nav_started_step > MAX_EGG_NAV_FRAMES:
        return task._after_egg_nav_budget(world.ram, "slot_timeout")
    pickup = task._egg_pickup_spot(world.ram)
    if pickup is None:
        return task._after_egg_nav_budget(world.ram, "no_reachable_stand")
    egg_stand, egg_face = pickup
    if task._navigator.stasis > 150:
        return task._after_egg_nav_budget(world.ram, "nav_stasis")
    action = task._navigate_to_left_top_goal(world.ram, egg_stand)
    if action is not None:
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
    task._egg_attempts += 1
    task._enqueue_skill_actions(world, coop_press_egg_pickup_skill(face=egg_face))
    task._verify_count = 0
    task._phase = "egg_verify"
    return TaskResult(status=TaskStatus.RUNNING)

def _step_egg_verify(task, world: WorldState) -> TaskResult:
    if not task._egg_present(world.ram) or is_holding_egg(world.ram):
        task.egg_collected = True
        print(f"[COOP] Egg collected")
        if task._current_egg_flag:
            task._skipped_egg_flags.discard(task._current_egg_flag)
        if task._current_egg_tile:
            task._skipped_egg_tiles.discard(task._current_egg_tile)
        task._phase = "decide"
        return TaskResult(status=TaskStatus.RUNNING)
    held_item = read_item_on_hand(world.ram)
    if held_item not in (0, ITEM_EGG):
        task._verify_count += 1
        if task._verify_count > 60 and task._egg_attempts < MAX_EGG_ATTEMPTS:
            task._phase = "egg_nav"
            task._verify_count = 0
        elif task._verify_count > 60:
            return task._after_egg_nav_budget(world.ram, "held_item_block")
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
    task._verify_count += 1
    if task._verify_count > 15:
        if task._egg_attempts < MAX_EGG_ATTEMPTS:
            task._phase = "egg_nav"
            task._verify_count = 0
        else:
            return task._after_egg_nav_budget(world.ram, "pickup_failed")
    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

def _step_decide(task, world: WorldState) -> TaskResult:
    mode = task.egg_mode
    if mode == "auto":
        adults, chicks, eggs = count_chicken_slots(world.ram)
        total = adults + chicks + eggs
        if not is_incubating(world.ram) and total < MAX_FLOCK_SIZE:
            mode = "incubate"
        else:
            mode = "ship"
    if mode == "incubate":
        task._incubator_wp_index = 0
        task._phase = "incubate_nav"
    elif mode == "gift":
        print("[COOP] Gift mode — exiting with egg")
        return task._begin_exit_prep()
    else:
        task._phase = "ship_nav"
    return TaskResult(status=TaskStatus.RUNNING)

def _step_incubate_nav(task, world: WorldState) -> TaskResult:
    action = task._navigate_to_incubator_stand(world.ram)
    if action is not None:
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
    task._enqueue_skill_actions(world, coop_press_incubator_skill(face=INCUBATOR_FACE))
    task._verify_count = 0
    task._phase = "incubate_verify"
    return TaskResult(status=TaskStatus.RUNNING)

def _step_incubate_verify(task, world: WorldState) -> TaskResult:
    if is_incubating(world.ram):
        task.egg_incubated = True
        print("[COOP] Egg incubated")
        return task._after_egg_handled(world.ram)
    task._verify_count += 1
    if task._verify_count > 15:
        print("[COOP] Incubation failed, shipping instead")
        task._phase = "ship_nav"
    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

def _step_ship_nav(task, world: WorldState) -> TaskResult:
    if not task._near_ship_lane():
        # Far approach via skills.py factory; pixel lane stays host-owned.
        skill_result = task._step_nav_skill(
            world,
            skill_name="coop_nav_ship_bin",
            make_skill=lambda: coop_nav_to_shipping_bin_skill(
                navigate=lambda w: task._navigate_to_ship_stand(w.ram),
            ),
        )
        if skill_result is not None:
            return skill_result
    elif task._active_skill is not None and task._active_skill.name == "coop_nav_ship_bin":
        task._active_skill = None
    action = task._ship_pixel_action()
    if action is not None:
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
    current = task._navigator.current_tile
    if current == SHIP_BIN_INTERACT_STAND:
        task._active_skill = None
        task._ship_money_before = read_shipping_money(world.ram)
        task._enqueue_skill_actions(world, coop_press_ship_skill(face=SHIP_BIN_FACE))
        task._verify_count = 0
        task._phase = "ship_verify"
        return TaskResult(status=TaskStatus.RUNNING)
    if current == SHIP_BIN_STAND:
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(left=True, b=True)))
    action = task._navigate_to_tile(world.ram, SHIP_BIN_STAND)
    if action is not None:
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))
    task._active_skill = None
    task._ship_money_before = read_shipping_money(world.ram)
    task._enqueue_skill_actions(world, coop_press_ship_skill(face=SHIP_BIN_FACE))
    task._verify_count = 0
    task._phase = "ship_verify"
    return TaskResult(status=TaskStatus.RUNNING)

def _step_ship_verify(task, world: WorldState) -> TaskResult:
    money_now = read_shipping_money(world.ram)
    if money_now > task._ship_money_before:
        task.egg_shipped = True
        print(f"[COOP] Egg shipped, money={money_now}")
        return task._after_egg_handled(world.ram)
    if not is_holding_egg(world.ram):
        return TaskResult(status=TaskStatus.FAILURE, reason="egg cleared without shipping money")
    task._verify_count += 1
    if task._verify_count > 20:
        print("[COOP] Ship verify timeout, retrying")
        task._phase = "ship_nav"
        task._verify_count = 0
    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

def _step_exit_prep_nav(task, world: WorldState) -> TaskResult:
    if task._exit_prep_started_step <= 0:
        task._exit_prep_started_step = task._step_count
    if task._step_count - task._exit_prep_started_step > MAX_EXIT_PREP_FRAMES:
        # Hand off wherever we are; EXIT_COOP owns the door route.
        print(
            f"[COOP] Exit prep timeout at {task._navigator.current_tile}; "
            "handing off to EXIT_COOP"
        )
        task._phase = "done"
        return TaskResult(status=TaskStatus.RUNNING)
    current = task._navigator.current_tile
    if current == EXIT_PREP_STAND or task._navigator.at_tile(EXIT_PREP_STAND):
        task._phase = "done"
        return TaskResult(status=TaskStatus.RUNNING)
    x = task._navigator.current_pos.x
    y = task._navigator.current_pos.y
    door_x = EXIT_PREP_STAND[0] * TILE_SIZE + 8
    safe_cross_max_y = 7 * TILE_SIZE + 8
    # Match EXIT_COOP: leave bin pocket, climb above false-open, cross,
    # then drop to the door. Hand off early once on the door column.
    if x < door_x - 4 or y > safe_cross_max_y or current != EXIT_PREP_STAND:
        if x <= 50 and y >= 165:
            if abs(x - SHIP_LANE_X) > 2:
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(
                        make_action(
                            right=x < SHIP_LANE_X,
                            left=x > SHIP_LANE_X,
                            b=True,
                        )
                    ),
                )
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action(up=True, b=True)),
            )
        if x < door_x - 4 and y > safe_cross_max_y:
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action(up=True, b=True)),
            )
        if abs(x - door_x) > 3:
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(
                    make_action(right=x < door_x, left=x > door_x, b=True)
                ),
            )
        if current[1] < EXIT_PREP_STAND[1]:
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action(down=True, b=True)),
            )
    task._phase = "done"
    return TaskResult(status=TaskStatus.RUNNING)


def bind_task_methods(cls: type) -> None:
    """Attach these phase functions to the task class. Not a mixin."""
    for name, fn in list(globals().items()):
        if not name.startswith("_") or not callable(fn):
            continue
        if getattr(fn, "__module__", None) != __name__:
            continue
        setattr(cls, name, fn)

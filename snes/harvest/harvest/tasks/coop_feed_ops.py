"""Chicken-feed phase functions for CoopChoresTask."""

from __future__ import annotations

from typing import Optional

import numpy as np

from harvest.core.animal_status import (
    ITEM_CHICKEN_FEED,
    read_fed_chickens_flags,
    read_fed_chickens_n,
    read_hay_count,
    read_item_on_hand,
)
from harvest.tasks.animal_navigation import align_to_pixel
from harvest.tasks.coop_layout import (
    CHICKEN_FEED_SPOTS,
    FEED_BIN_FACE,
    FEED_BIN_STAND,
    MAX_FEED_PLACE_FRAMES,
    MAX_FEED_SLOT_DEFERRALS,
    ChickenFeedSpot,
)
from harvest.tasks.nav import make_action
from harvest.tasks.skills import (
    coop_nav_to_feed_bin_skill,
    coop_press_feed_place_skill,
    coop_press_feed_skill,
)
from retro_harness import ActionResult, TaskResult, TaskStatus, WorldState

def _queue_place_feed(task, face: str) -> None:
    task._action_queue.extend(make_action(**{face: True}) for _ in range(4))
    task._action_queue.extend(make_action(**{face: True, "a": True}) for _ in range(8))
    task._action_queue.extend(make_action(a=True) for _ in range(4))
    task._action_queue.extend(make_action(down=True) for _ in range(12))
    task._action_queue.extend(make_action() for _ in range(8))

def _fed_count_now(task, ram: np.ndarray) -> int:
    flags = read_fed_chickens_flags(ram)
    flag_count = sum(1 for spot in CHICKEN_FEED_SPOTS if flags & spot.flag)
    return max(read_fed_chickens_n(ram), flag_count)

def _next_feed_spot(task, ram: np.ndarray) -> Optional[ChickenFeedSpot]:
    flags = read_fed_chickens_flags(ram)
    blocked = task._chicken_tiles(ram)
    blocked.discard(task._navigator.current_tile)

    for spot in CHICKEN_FEED_SPOTS:
        if flags & spot.flag:
            continue
        if spot.flag in task._blocked_feed_flags:
            continue
        if spot.stand in blocked:
            continue
        return spot

    for spot in CHICKEN_FEED_SPOTS:
        if not (flags & spot.flag) and spot.flag not in task._blocked_feed_flags:
            return spot

    # Chickens wander dynamically. If all remaining unfed spots were marked
    # blocked, clear blocked flags so other trough spots can be retried.
    unfed = [s for s in CHICKEN_FEED_SPOTS if not (flags & s.flag)]
    if unfed and all(s.flag in task._blocked_feed_flags for s in unfed):
        for s in unfed:
            task._blocked_feed_flags.discard(s.flag)
        for s in unfed:
            if s.stand not in blocked:
                return s
        return unfed[0]

    return None

def _advance_after_feed(task, ram: np.ndarray) -> TaskResult:
    task._feed_registered = False
    task._pathfinder.temp_blocked.clear()
    fed_now = min(task._fed_count_now(ram), task._adult_count)
    task.fed_count = max(task.fed_count, fed_now)
    task._feed_remaining = max(0, task._adult_count - fed_now)
    task._current_feed_spot = None
    task._feed_place_started_step = 0
    if task._feed_remaining > 0:
        task._clear_left_top_route()
        if read_item_on_hand(ram) == ITEM_CHICKEN_FEED:
            task._phase = "feed_place_nav"
        elif read_hay_count(ram) <= 0:
            print(f"[COOP] Out of hay after feeding {task.fed_count}")
            task._feed_remaining = 0
            if task._collectable_egg_present(ram):
                return task._begin_egg_nav()
            return task._begin_exit_prep()
        else:
            task._phase = "feed_nav"
    elif task._collectable_egg_present(ram):
        return task._begin_egg_nav()
    else:
        return task._begin_exit_prep()
    return TaskResult(status=TaskStatus.RUNNING)

def _step_feed_nav(task, world: WorldState) -> TaskResult:
    if task._fed_count_now(world.ram) >= task._adult_count:
        task._active_skill = None
        return task._advance_after_feed(world.ram)
    if read_item_on_hand(world.ram) == ITEM_CHICKEN_FEED:
        task._active_skill = None
        task._phase = "feed_place_nav"
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
    # Production path: skills.py factory + host left-top aisle routing.
    skill_result = task._step_nav_skill(
        world,
        skill_name="coop_nav_feed_bin",
        make_skill=lambda: coop_nav_to_feed_bin_skill(
            navigate=lambda w: task._navigate_to_left_top_goal(
                w.ram, FEED_BIN_STAND
            ),
        ),
    )
    if skill_result is not None:
        return skill_result
    task._hay_before = read_hay_count(world.ram)
    task._phase = "feed_act"
    return task._step_feed_act(world)

def _step_feed_act(task, world: WorldState) -> TaskResult:
    held_item = read_item_on_hand(world.ram)
    if held_item == ITEM_CHICKEN_FEED:
        task._phase = "feed_place_nav"
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
    if held_item != 0:
        task._phase = "feed_verify"
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
    if task._fed_count_now(world.ram) >= task._adult_count:
        return task._advance_after_feed(world.ram)
    # Cap feeds at available hay
    if read_hay_count(world.ram) <= 0:
        print(f"[COOP] Out of hay after feeding {task.fed_count}")
        if task._egg_present(world.ram):
            task._phase = "egg_nav"
        else:
            task._phase = "done"
        return TaskResult(status=TaskStatus.RUNNING)
    task._enqueue_skill_actions(world, coop_press_feed_skill(face=FEED_BIN_FACE))
    task._feed_registered = False
    task._verify_count = 0
    task._phase = "feed_verify"
    return TaskResult(status=TaskStatus.RUNNING)

def _step_feed_verify(task, world: WorldState) -> TaskResult:
    if read_item_on_hand(world.ram) == ITEM_CHICKEN_FEED:
        task._verify_count = 0
        task._phase = "feed_place_nav"
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
    task._verify_count += 1
    if task._verify_count > 40:
        # Feed pickup did not register; retry the bin interaction.
        task._phase = "feed_act"
    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

def _step_feed_place_nav(task, world: WorldState) -> TaskResult:
    if task._fed_count_now(world.ram) >= task._adult_count:
        return task._advance_after_feed(world.ram)
    if read_item_on_hand(world.ram) != ITEM_CHICKEN_FEED:
        task._phase = "feed_nav"
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

    if task._feed_place_started_step <= 0:
        task._feed_place_started_step = task._step_count

    spot = task._current_feed_spot
    flags_now = read_fed_chickens_flags(world.ram)
    chicken_tiles = task._chicken_tiles(world.ram)
    if spot is None or (flags_now & spot.flag) or (
        spot.stand in chicken_tiles and spot.stand != task._navigator.current_tile
    ):
        spot = task._next_feed_spot(world.ram)
        task._current_feed_spot = spot
        task._feed_place_started_step = task._step_count
    if spot is None:
        if task._blocked_feed_flags:
            task._blocked_feed_flags.clear()
            spot = task._next_feed_spot(world.ram)
            task._current_feed_spot = spot
        if spot is None:
            spot = CHICKEN_FEED_SPOTS[0]
            task._current_feed_spot = spot

    timed_out = (
        task._step_count - task._feed_place_started_step > MAX_FEED_PLACE_FRAMES
    )
    if task._navigator.stasis > 120 or timed_out:
        deferred = task._deferred_feed_counts.get(spot.flag, 0)
        if deferred < MAX_FEED_SLOT_DEFERRALS and not timed_out:
            task._deferred_feed_counts[spot.flag] = deferred + 1
            print(
                f"[COOP] Feed deferred flag=0x{spot.flag:04X} "
                f"reason=stasis count={deferred + 1}"
            )
        else:
            reason = "slot_timeout" if timed_out else "stasis"
            print(
                f"[COOP] Feed skipped flag=0x{spot.flag:04X} reason={reason}"
            )
            task._blocked_feed_flags.add(spot.flag)
        task._current_feed_spot = None
        task._feed_place_started_step = task._step_count
        task._navigator.path = []
        task._navigator.stasis = 0
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

    if (
        task._navigator.current_tile[0] == spot.stand[0]
        and task._navigator.current_tile[1] > spot.stand[1]
    ):
        dx = spot.interact_px[0] - task._navigator.current_pos.x
        if abs(dx) > 1:
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action(right=dx > 0, left=dx < 0)),
            )

    if task._navigator.current_tile != spot.stand:
        action = task._navigate_to_tile(world.ram, spot.stand)
        if action is not None:
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

    action = align_to_pixel(
        (task._navigator.current_pos.x, task._navigator.current_pos.y),
        spot.interact_px,
        tolerance=1,
    )
    if action is not None:
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))

    task._fed_before = task._fed_count_now(world.ram)
    task._fed_flags_before = read_fed_chickens_flags(world.ram)
    task._enqueue_skill_actions(world, coop_press_feed_place_skill(face=spot.face))
    task._verify_count = 0
    task._feed_place_started_step = 0
    task._phase = "feed_place_verify"
    return TaskResult(status=TaskStatus.RUNNING)

def _step_feed_place_verify(task, world: WorldState) -> TaskResult:
    fed_now = task._fed_count_now(world.ram)
    flags_now = read_fed_chickens_flags(world.ram)
    if fed_now > task._fed_before or flags_now != task._fed_flags_before:
        task.fed_count = max(task.fed_count, min(fed_now, task._adult_count))
        task._feed_remaining = max(0, task._adult_count - task.fed_count)
        print(
            f"[COOP] Feed OK count={task.fed_count} "
            f"remaining={task._feed_remaining} flags=0x{flags_now:04X}"
        )
        return task._advance_after_feed(world.ram)

    task._verify_count += 1
    if task._verify_count > 30:
        if read_item_on_hand(world.ram) == ITEM_CHICKEN_FEED:
            task._phase = "feed_place_nav"
        else:
            task._current_feed_spot = None
            task._phase = "feed_nav"
        task._verify_count = 0
    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

def _step_feed_clear_nav(task, world: WorldState) -> TaskResult:
    task._phase = "feed_place_nav"
    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))

def _step_feed_clear_verify(task, world: WorldState) -> TaskResult:
    task._phase = "feed_place_verify"
    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))


def bind_task_methods(cls: type) -> None:
    """Attach these phase functions to the task class. Not a mixin."""
    for name, fn in list(globals().items()):
        if not name.startswith("_") or not callable(fn):
            continue
        if getattr(fn, "__module__", None) != __name__:
            continue
        setattr(cls, name, fn)

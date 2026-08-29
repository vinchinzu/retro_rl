"""Shared navigation primitives for animal-interior chores.

Animal task modules supply their own dynamic blockers (cows, chickens, eggs,
or known bad tiles).  This module owns only the common grid search and the
small pixel/tile steering decisions used once those blockers are known.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Collection
from typing import Optional, Tuple

import numpy as np

from harvest.core.animal_probe import chicken_slot_snapshots
from harvest.tasks.nav import (
    MAP_WIDTH,
    Navigator,
    Pathfinder,
    make_action,
)


Tile = Tuple[int, int]


def find_path_around_blockers(
    ram: np.ndarray,
    pathfinder: Pathfinder,
    start: Tile,
    goal: Tile,
    blockers: Collection[Tile],
) -> Optional[list[Tile]]:
    """Return a four-direction path while preserving dynamic blockers.

    ``blockers`` is copied so callers can reuse their animal/object snapshot
    without the path search changing it.  The start tile remains traversable
    because it can contain the player or an object that overlaps the player.
    """
    blocked = set(blockers)
    blocked.discard(start)
    if goal in blocked:
        return None
    if start == goal:
        return []

    queue = deque([start])
    came_from: dict[Tile, Optional[Tile]] = {start: None}
    while queue:
        current_x, current_y = queue.popleft()
        if (current_x, current_y) == goal:
            break
        for delta_x, delta_y in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_x, next_y = current_x + delta_x, current_y + delta_y
            next_tile = (next_x, next_y)
            if not (0 <= next_x < MAP_WIDTH and 0 <= next_y < MAP_WIDTH):
                continue
            if next_tile in came_from or next_tile in blocked:
                continue
            if not pathfinder.is_walkable(ram, next_x, next_y, current_pos=start):
                continue
            came_from[next_tile] = (current_x, current_y)
            queue.append(next_tile)

    if goal not in came_from:
        return None

    path: list[Tile] = []
    current = goal
    while current != start:
        path.append(current)
        parent = came_from[current]
        if parent is None:
            break
        current = parent
    path.reverse()
    return path


def align_to_pixel(
    current: Tuple[int, int], target: Tuple[int, int], *, tolerance: int = 1
) -> Optional[np.ndarray]:
    """Move along the dominant axis until ``current`` reaches ``target``."""
    delta_x = target[0] - current[0]
    delta_y = target[1] - current[1]
    if abs(delta_x) <= tolerance and abs(delta_y) <= tolerance:
        return None
    if abs(delta_x) >= abs(delta_y) and abs(delta_x) > tolerance:
        return make_action(right=delta_x > 0, left=delta_x < 0)
    return make_action(down=delta_y > 0, up=delta_y < 0)


def fallback_action(current: Tile, goal: Tile) -> np.ndarray:
    """Run toward the dominant-axis tile delta when pathfinding has no route."""
    delta_x = goal[0] - current[0]
    delta_y = goal[1] - current[1]
    if abs(delta_x) >= abs(delta_y):
        direction = "right" if delta_x > 0 else "left"
    else:
        direction = "down" if delta_y > 0 else "up"
    return make_action(**{direction: True, "b": True})


def chicken_stage_tiles(
    ram: np.ndarray,
    stages: Collection[str],
    *,
    require_coop: bool = True,
) -> list[Tile]:
    """Unique chicken tiles at the given life stages, in slot order."""
    tiles: list[Tile] = []
    seen: set[Tile] = set()
    wanted = set(stages)
    for row in chicken_slot_snapshots(ram, require_coop=require_coop):
        if row.get("stage") not in wanted:
            continue
        tile = row.get("tile")
        if not (isinstance(tile, list) and len(tile) == 2):
            continue
        chicken_tile = (int(tile[0]), int(tile[1]))
        if chicken_tile in seen:
            continue
        seen.add(chicken_tile)
        tiles.append(chicken_tile)
    return tiles


def adjacent_face_stands(tile: Tile) -> tuple[tuple[Tile, str], ...]:
    """Cardinal neighbor stands and the face looking back at ``tile``."""
    x, y = tile
    return (
        ((x + 1, y), "left"),
        ((x - 1, y), "right"),
        ((x, y + 1), "up"),
        ((x, y - 1), "down"),
    )


def select_adjacent_pickup_target(
    ram: np.ndarray,
    pathfinder: Pathfinder,
    current: Tile,
    targets: Collection[Tile],
    blockers: Collection[Tile],
) -> Optional[tuple[Tile, str, Tile]]:
    """Nearest walkable neighbor stand facing one of ``targets``."""
    blocked = set(blockers)
    best: Optional[tuple[int, Tile, str, Tile]] = None
    for animal_tile in targets:
        for stand, face in adjacent_face_stands(animal_tile):
            sx, sy = stand
            if not (0 <= sx < MAP_WIDTH and 0 <= sy < MAP_WIDTH):
                continue
            if stand in blocked and stand != current:
                continue
            if not pathfinder.is_walkable(ram, sx, sy, current_pos=current):
                continue
            path = find_path_around_blockers(ram, pathfinder, current, stand, blocked)
            if path is None:
                continue
            score = len(path)
            if best is None or score < best[0]:
                best = (score, stand, face, animal_tile)
    if best is None:
        return None
    return best[1], best[2], best[3]


def navigate_to_tile_around_blockers(
    ram: np.ndarray,
    pathfinder: Pathfinder,
    navigator: Navigator,
    goal: Tile,
    blockers: Collection[Tile],
) -> Optional[np.ndarray]:
    """Walk to ``goal`` while treating live animals as dynamic no-go tiles."""
    if navigator.current_tile == goal or navigator.at_tile(goal, tolerance=1):
        return navigator.center_on_tile(goal, tolerance=1)

    blocked = set(blockers)
    blocked.discard(navigator.current_tile)
    if goal in blocked:
        navigator.path = []
        return make_action()
    if navigator.path and navigator.path[0] in blocked:
        navigator.path = []
        return make_action()
    if navigator.stasis > 90 and navigator.path:
        pathfinder.temp_blocked.add(navigator.path[0])
        navigator.path = []

    if not navigator.path:
        path = find_path_around_blockers(
            ram, pathfinder, navigator.current_tile, goal, blocked
        )
        if path is None:
            return fallback_action(navigator.current_tile, goal)
        navigator.path = path

    action = navigator.follow_path(ram)
    if action is None:
        return fallback_action(navigator.current_tile, goal)
    return action

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

from harvest.tasks.farm_clearer import MAP_WIDTH, Pathfinder, make_action


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

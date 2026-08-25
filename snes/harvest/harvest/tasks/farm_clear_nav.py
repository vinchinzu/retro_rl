"""Target-approach selection and no-progress recovery for farm clearing."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import numpy as np

from harvest.core.tile_catalog import MAP_WIDTH, TILE_TO_DEBRIS, DebrisType
from harvest.tasks.farm_ops import Target
from harvest.tasks.nav import (
    TILE_SIZE,
    VIEWPORT_HOP_TILES,
    Point,
    get_tile_at,
    manhattan,
)

if TYPE_CHECKING:
    from harvest.tasks.farm_clearer import FarmClearer

_CARDINALS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_OPENER_TYPES = (DebrisType.STONE, DebrisType.FENCE)

ApproachChoice = Tuple[Target, Tuple[int, int], List[Tuple[int, int]]]


def _stands_for(target: Target) -> List[Tuple[int, int]]:
    occupied = set(target.footprint)
    stands: List[Tuple[int, int]] = []
    for tx, ty in target.footprint:
        for dx, dy in _CARDINALS:
            stand = (tx + dx, ty + dy)
            if stand in occupied:
                continue
            if not (0 <= stand[0] < MAP_WIDTH and 0 <= stand[1] < MAP_WIDTH):
                continue
            stands.append(stand)
    return stands


def find_unfailed_approach(
    clearer: "FarmClearer", ram: np.ndarray, target: Target
) -> Optional[Tuple[int, int]]:
    """Choose the nearest viable stand not disproved for this target."""
    candidates: list[Tuple[int, Tuple[int, int]]] = []
    for stand in _stands_for(target):
        if (target.tile, stand) in clearer.failed_approaches:
            continue
        if not clearer.pathfinder.is_walkable(ram, *stand):
            continue
        distance = manhattan(
            Point(stand[0] * TILE_SIZE + 8, stand[1] * TILE_SIZE + 8),
            clearer.navigator.current_pos,
        )
        candidates.append((distance, stand))
    return min(candidates, default=(0, None))[1]


def _path_reaches(
    path: Optional[List[Tuple[int, int]]],
    start: Tuple[int, int],
    stand: Tuple[int, int],
) -> bool:
    if path is None:
        return False
    if start == stand:
        return True
    return bool(path) and path[-1] == stand


def find_pathable_approach(
    clearer: "FarmClearer", ram: np.ndarray, target: Target
) -> Tuple[Optional[Tuple[int, int]], Optional[List[Tuple[int, int]]]]:
    """Prefer a stand BFS can actually reach over a nearer blocked neighbor."""
    start = clearer.navigator.current_tile
    full: list[Tuple[int, Tuple[int, int], List[Tuple[int, int]]]] = []
    hops: list[Tuple[int, Tuple[int, int], List[Tuple[int, int]]]] = []
    for stand in _stands_for(target):
        if (target.tile, stand) in clearer.failed_approaches:
            continue
        if not clearer.pathfinder.is_walkable(ram, *stand):
            continue
        path = clearer.pathfinder.find_path(
            ram, start, stand, max_steps=VIEWPORT_HOP_TILES
        )
        if path is None:
            continue
        distance = manhattan(
            Point(stand[0] * TILE_SIZE + 8, stand[1] * TILE_SIZE + 8),
            clearer.navigator.current_pos,
        )
        row = (distance, stand, path)
        if _path_reaches(path, start, stand):
            full.append(row)
        else:
            hops.append(row)
    pool = full or hops
    if not pool:
        return None, None
    _dist, stand, path = min(pool, key=lambda item: item[0])
    return stand, path


def _blocker_target(ram: np.ndarray, tile: Tuple[int, int]) -> Optional[Target]:
    tid = int(get_tile_at(ram, *tile))
    debris = TILE_TO_DEBRIS.get(tid)
    if debris not in _OPENER_TYPES:
        return None
    return Target(
        tile=tile,
        pos=Point(tile[0] * TILE_SIZE + 8, tile[1] * TILE_SIZE + 8),
        debris_type=debris,
        tile_id=tid,
    )


def find_blocker_opener(
    clearer: "FarmClearer", ram: np.ndarray, weeds: Sequence[Target]
) -> Optional[ApproachChoice]:
    """Lift an adjacent stone/fence so a boxed weed gains a stand."""
    for weed in weeds:
        if find_unfailed_approach(clearer, ram, weed) is not None:
            continue
        blockers: list[Target] = []
        for dx, dy in _CARDINALS:
            blocker = _blocker_target(
                ram, (weed.tile[0] + dx, weed.tile[1] + dy)
            )
            if blocker is not None:
                blockers.append(blocker)
        blockers.sort(key=lambda t: 0 if t.debris_type == DebrisType.STONE else 1)
        for blocker in blockers:
            approach, path = find_pathable_approach(clearer, ram, blocker)
            if approach is not None and path is not None:
                return blocker, approach, path
    return None


def choose_clear_target(
    clearer: "FarmClearer", ram: np.ndarray, phase_targets: Sequence[Target]
) -> Optional[ApproachChoice]:
    """Pick the next debris stand, opening boxed weeds when needed."""
    for target in phase_targets:
        approach, path = find_pathable_approach(clearer, ram, target)
        if approach is not None and path is not None:
            return target, approach, path
    return find_blocker_opener(clearer, ram, phase_targets)


def start_progress_watch(clearer: "FarmClearer", approach: Tuple[int, int]) -> None:
    clearer._nav_best_distance = manhattan(
        clearer.navigator.current_pos,
        Point(approach[0] * TILE_SIZE + 8, approach[1] * TILE_SIZE + 8),
    )
    clearer._nav_last_progress_frame = clearer.frame_count


def handle_navigating(clearer: "FarmClearer", ram: np.ndarray) -> Optional[str]:
    """Navigate to a debris stand, rejecting cross-tile pixel oscillations."""
    if not clearer.current_target or not clearer.approach_tile:
        return "scanning"

    live_id = get_tile_at(ram, *clearer.current_target.tile)
    live_debris = TILE_TO_DEBRIS.get(live_id)
    if live_debris is None or live_debris != clearer.current_target.debris_type:
        clearer.current_target = None
        return "scanning"
    if live_id != clearer.current_target.tile_id:
        clearer.current_target = Target(
            tile=clearer.current_target.tile,
            pos=clearer.current_target.pos,
            debris_type=live_debris,
            tile_id=live_id,
        )

    if clearer.navigator.current_tile == clearer.approach_tile:
        return "clearing"

    approach_pos = Point(
        clearer.approach_tile[0] * TILE_SIZE + 8,
        clearer.approach_tile[1] * TILE_SIZE + 8,
    )
    distance = manhattan(clearer.navigator.current_pos, approach_pos)
    if clearer._nav_best_distance is None or distance < clearer._nav_best_distance:
        clearer._nav_best_distance = distance
        clearer._nav_last_progress_frame = clearer.frame_count
    elif clearer.frame_count - clearer._nav_last_progress_frame > clearer.max_nav_no_progress:
        failed = (clearer.current_target.tile, clearer.approach_tile)
        clearer.failed_approaches.add(failed)
        print(
            f"[NAV] No target progress at {clearer.navigator.current_pos}; "
            f"reject approach target={failed[0]} stand={failed[1]}"
        )
        clearer.navigator.path = []
        clearer.navigator.stasis = 0
        clearer.current_target = None
        clearer.approach_tile = None
        clearer._nav_best_distance = None
        return "scanning"

    if clearer.navigator.stasis > clearer.max_stasis:
        print(f"[NAV] Stuck at {clearer.navigator.current_tile}, trying alternate path")
        if clearer.navigator.path:
            clearer.pathfinder.temp_blocked.add(clearer.navigator.path[0])
        clearer.navigator.path = []
        clearer.navigator.stasis = 0
        return clearer._replan_nav_hop(ram)

    action = clearer.navigator.follow_path(ram)
    if action is not None:
        clearer.action_queue.append(action)
        return None
    if clearer.navigator.current_tile != clearer.approach_tile:
        return clearer._replan_nav_hop(ram)
    return "clearing"

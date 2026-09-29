"""find_frontier_path (one BFS tree) must equal the per-candidate find_path loop."""

from __future__ import annotations

import random

import numpy as np

from harvest.core.tile_catalog import ADDR_TILEMAP, get_tile_at
from harvest.planner.tasks.navigation import STALE_TILE_IDS, find_frontier_path
from harvest.tasks.nav import Pathfinder
from tests.test_pathfinder_fast_walk import _random_ram


def _reference(pathfinder, ram, start, final, radius=8, max_candidates=24):
    sx, sy = start
    fx, fy = final
    start_dist = abs(fx - sx) + abs(fy - sy)
    candidates = []
    for ty in range(max(0, sy - radius), min(63, sy + radius) + 1):
        for tx in range(max(0, sx - radius), min(63, sx + radius) + 1):
            if (tx, ty) == start:
                continue
            if get_tile_at(ram, tx, ty) in STALE_TILE_IDS:
                continue
            if not pathfinder.is_walkable(ram, tx, ty, current_pos=start):
                continue
            dist = abs(fx - tx) + abs(fy - ty)
            progress = start_dist - dist
            if progress <= 0:
                continue
            candidates.append(((progress, abs(tx - sx) + abs(ty - sy), -dist), (tx, ty)))
    candidates.sort(reverse=True)
    for _, goal in candidates[:max_candidates]:
        path = pathfinder.find_path(ram, start, goal)
        if path:
            return path
    return None


def test_frontier_path_matches_reference() -> None:
    rng = random.Random(3)
    found = 0
    for seed, tilemap in ((1, 0x00), (2, 0x04), (3, 0x0C), (4, 0x15), (5, 0x00), (6, 0x0C)):
        ram = _random_ram(seed, tilemap)
        pf = Pathfinder()
        pf.temp_blocked = {(rng.randrange(64), rng.randrange(64)) for _ in range(60)}
        for _ in range(12):
            start = (rng.randrange(8, 56), rng.randrange(8, 56))
            final = (rng.randrange(64), rng.randrange(64))
            want = _reference(pf, ram, start, final)
            assert find_frontier_path(pf, ram, start, final) == want
            found += want is not None
    assert found > 0

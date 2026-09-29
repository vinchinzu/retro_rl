"""The hoisted BFS walk test must agree with Pathfinder.is_walkable everywhere."""

import random
import unittest

import numpy as np

from harvest.core.tile_catalog import (
    ADDR_TILEMAP,
    LARGE_ROCK_TL,
    MAP_HEIGHT,
    MAP_WIDTH,
    STUMP_TL,
    TRAVEL_SOLID_TILES,
    set_tile_at,
)
from harvest.tasks.nav import Pathfinder

TILE_POOL = [0x01, 0x02, 0x03, 0xA0, 0xA1, 0xA8, 0xFF, 0x58, 0x5A, 0xF0, STUMP_TL, LARGE_ROCK_TL] + sorted(
    TRAVEL_SOLID_TILES
)[:4]


def _random_ram(seed: int, tilemap: int) -> np.ndarray:
    rng = random.Random(seed)
    ram = np.zeros(0x20000, dtype=np.uint8)
    ram[ADDR_TILEMAP] = tilemap
    for ty in range(MAP_HEIGHT):
        for tx in range(MAP_WIDTH):
            set_tile_at(ram, tx, ty, rng.choice(TILE_POOL))
    return ram


class FastWalkTests(unittest.TestCase):
    def test_walk_test_matches_is_walkable(self) -> None:
        rng = random.Random(7)
        for seed, tilemap in ((1, 0x00), (2, 0x04), (3, 0x0C), (4, 0x15)):
            ram = _random_ram(seed, tilemap)
            pf = Pathfinder()
            pf.temp_blocked = {(rng.randrange(64), rng.randrange(64)) for _ in range(40)}
            pf.extra_walkable = {(rng.randrange(64), rng.randrange(64)) for _ in range(40)}
            override = {0x58, 0xFF}
            start = (rng.randrange(64), rng.randrange(64))
            for kwargs in ({}, {"walkable_override": override}, {"current_pos": start}):
                fast = pf._walk_test(ram, kwargs.get("walkable_override"), kwargs.get("current_pos"))
                for ty in range(MAP_HEIGHT):
                    for tx in range(MAP_WIDTH):
                        self.assertEqual(
                            fast(tx, ty),
                            pf.is_walkable(ram, tx, ty, **kwargs),
                            (seed, tilemap, tx, ty, sorted(kwargs)),
                        )

    def test_find_path_matches_a_reference_bfs(self) -> None:
        from collections import deque

        def reference(pf, ram, start, goal):
            if start == goal:
                return []
            queue = deque([start])
            came = {start: None}
            while queue:
                cx, cy = queue.popleft()
                if (cx, cy) == goal:
                    break
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < MAP_WIDTH and 0 <= ny < MAP_WIDTH and (nx, ny) not in came:
                        if pf.is_walkable(ram, nx, ny, current_pos=start):
                            came[(nx, ny)] = (cx, cy)
                            queue.append((nx, ny))
            if goal not in came:
                return None
            path, cur = [], goal
            while cur != start:
                path.append(cur)
                cur = came[cur]
            return path[::-1]

        rng = random.Random(3)
        checked = 0
        for seed in (21, 22):
            ram = _random_ram(seed, 0x00)
            # Open most cells so paths exist.
            for ty in range(MAP_HEIGHT):
                for tx in range(MAP_WIDTH):
                    if rng.random() < 0.8:
                        set_tile_at(ram, tx, ty, 0x01)
            pf = Pathfinder()
            for _ in range(12):
                start = (rng.randrange(8, 56), rng.randrange(8, 56))
                goal = (rng.randrange(8, 56), rng.randrange(8, 56))
                got = pf.find_path(ram, start, goal)
                self.assertEqual(got, reference(pf, ram, start, goal), (seed, start, goal))
                checked += got is not None
        self.assertGreater(checked, 3)


if __name__ == "__main__":
    unittest.main()

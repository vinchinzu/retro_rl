"""The flat-index occupancy flood returns the same paths as the dict flood.

``OccupancyGrid._shortest_path`` was rewritten for speed (87% of an L6 0x19
fight frame). Every walker decision downstream reads its path, and the
chain is frame-perfect, so the rewrite must be exact: this keeps the old
flood as the reference and compares them on random rooms.
"""

from __future__ import annotations

import random
from collections import deque

import pytest

from zelda_i.walk.physics import WALK_DELTA, OccupancyGrid, _lattice_move


def _reference(grid, start, goal, extra_blocked, lattice):
    """The pre-2026-09-23 flood, verbatim apart from ``self``."""
    sx, sy = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])
    if (sx, sy) == (gx, gy):
        return [(sx, sy)]
    extra = set(extra_blocked) if extra_blocked else None
    if extra:
        extra.discard((sx, sy))
        extra.discard((gx, gy))

    def ok(x, y):
        if (x, y) == (sx, sy):
            return True
        if extra and (x, y) in extra:
            return False
        return grid.passable(x, y)

    if not ok(gx, gy) or not grid.in_bounds(gx, gy):
        return None
    dist = {(gx, gy): 0}
    queue = deque([(gx, gy)])
    while queue:
        x, y = queue.popleft()
        if (x, y) == (sx, sy):
            break
        step = dist[(x, y)] + 1
        for dx, dy in WALK_DELTA.values():
            cell = (x + dx, y + dy)
            if cell in dist or not ok(*cell):
                continue
            if lattice and not _lattice_move(x, y, dx):
                continue
            dist[cell] = step
            queue.append(cell)
    if (sx, sy) not in dist:
        return None
    path = [(sx, sy)]
    x, y = sx, sy
    heading = None
    while (x, y) != (gx, gy):
        nearer = [
            (delta, cell)
            for delta in WALK_DELTA.values()
            for cell in ((x + delta[0], y + delta[1]),)
            if dist.get(cell, -1) == dist[(x, y)] - 1
        ]
        if not nearer:
            return None
        delta, cell = next((pair for pair in nearer if pair[0] == heading), nearer[0])
        path.append(cell)
        heading = delta
        x, y = cell
    return path


def _room(rng: random.Random) -> OccupancyGrid:
    grid = OccupancyGrid(xmin=40, xmax=120, ymin=77, ymax=141)
    for _ in range(rng.randint(0, 14)):
        x0, y0 = rng.randint(30, 125), rng.randint(70, 145)
        w, h = rng.randint(1, 24), rng.randint(1, 24)
        grid.blocked |= {(x, y) for x in range(x0, x0 + w) for y in range(y0, y0 + h)}
    return grid


def _point(rng: random.Random, grid: OccupancyGrid, *, on_lattice: bool) -> tuple[int, int]:
    x = rng.randint(grid.xmin - 3, grid.xmax + 3)
    y = rng.randint(grid.ymin - 3, grid.ymax + 3)
    if on_lattice and rng.random() < 0.7:
        if rng.random() < 0.5:
            x -= x % 8
        else:
            y -= (y - 5) % 8
    return x, y


@pytest.mark.parametrize("seed", range(40))
def test_flat_flood_matches_the_dict_flood(seed: int) -> None:
    rng = random.Random(seed)
    grid = _room(rng)
    for _ in range(25):
        start = _point(rng, grid, on_lattice=True)
        goal = _point(rng, grid, on_lattice=True)
        if rng.random() < 0.2:
            goal = start
        bodies = [_point(rng, grid, on_lattice=False) for _ in range(rng.randint(0, 6))]
        bodies = {(bx + dx, by + dy) for bx, by in bodies for dx in range(-4, 5) for dy in range(-4, 5)}
        extra = bodies if rng.random() < 0.6 else None
        for lattice in (True, False):
            assert grid._shortest_path(start, goal, extra, lattice) == _reference(
                grid, start, goal, extra, lattice
            ), (seed, start, goal, lattice)

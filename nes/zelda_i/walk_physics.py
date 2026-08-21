"""Link walk model + occupancy BFS (no emulator).

Cardinal 1px/frame. Cells default passable; OccupancyWalker grades a predicted
step and blocks the cell ahead on a stuck miss, then replans. No path → stand.
Door clips (LEFT+UP residual) are not modeled here — those stay in ``level*_path``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from retro_harness.predict import grade_claims

__all__ = [
    "DEFAULT_BOUNDS",
    "WALK_DELTA",
    "WALK_SPEED",
    "OccupancyGrid",
    "OccupancyWalker",
    "follow_path",
    "predicted_xy",
]

WALK_SPEED = 1
WALK_DELTA: dict[str, tuple[int, int]] = {
    "UP": (0, -WALK_SPEED),
    "DOWN": (0, WALK_SPEED),
    "LEFT": (-WALK_SPEED, 0),
    "RIGHT": (WALK_SPEED, 0),
}
# Dungeon playfield (Link collision). North door approach y≈93 sits inside.
DEFAULT_BOUNDS: tuple[int, int, int, int] = (40, 216, 77, 205)
_DRIFT_REPLAN = 8


def predicted_xy(x: int, y: int, direction: str) -> tuple[int, int]:
    """Pixel Link occupies after one successful cardinal step."""
    dx, dy = WALK_DELTA[direction]
    return x + dx, y + dy


@dataclass
class OccupancyGrid:
    """In-room passability. Unknown cells are free until a miss blocks them."""

    blocked: set[tuple[int, int]] = field(default_factory=set)
    xmin: int = DEFAULT_BOUNDS[0]
    xmax: int = DEFAULT_BOUNDS[1]
    ymin: int = DEFAULT_BOUNDS[2]
    ymax: int = DEFAULT_BOUNDS[3]

    def in_bounds(self, x: int, y: int) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def passable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and (x, y) not in self.blocked

    def mark_blocked_ahead(self, x: int, y: int, direction: str) -> tuple[int, int]:
        """Record the cell the last predicted step failed to enter."""
        cell = predicted_xy(x, y, direction)
        self.blocked.add(cell)
        return cell

    def shortest_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]] | None:
        """4-connected BFS. ``start`` is always allowed so a pocket can escape."""
        sx, sy = int(start[0]), int(start[1])
        gx, gy = int(goal[0]), int(goal[1])
        if (sx, sy) == (gx, gy):
            return [(sx, sy)]

        def ok(x: int, y: int) -> bool:
            if (x, y) == (sx, sy):
                return True
            return self.passable(x, y)

        if not ok(gx, gy) and not self.in_bounds(gx, gy):
            return None

        queue: deque[tuple[int, int]] = deque([(sx, sy)])
        parent: dict[tuple[int, int], tuple[int, int] | None] = {(sx, sy): None}
        while queue:
            x, y = queue.popleft()
            if (x, y) == (gx, gy):
                break
            for dx, dy in WALK_DELTA.values():
                nx, ny = x + dx, y + dy
                if (nx, ny) in parent or not ok(nx, ny):
                    continue
                parent[(nx, ny)] = (x, y)
                queue.append((nx, ny))
        if (gx, gy) not in parent:
            return None
        path: list[tuple[int, int]] = []
        node: tuple[int, int] | None = (gx, gy)
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path


def follow_path(
    path: list[tuple[int, int]] | None,
    xy: tuple[int, int],
) -> str | None:
    """Cardinal toward the next BFS node, or None when the path is stale."""
    if not path or len(path) < 2:
        return None
    x, y = xy
    idx = min(
        range(len(path)),
        key=lambda i: abs(path[i][0] - x) + abs(path[i][1] - y),
    )
    if abs(path[idx][0] - x) + abs(path[idx][1] - y) > _DRIFT_REPLAN:
        return None
    if idx >= len(path) - 1:
        return None
    nx, ny = path[idx + 1]
    dx, dy = nx - x, ny - y
    if dx == 0 and dy == 0:
        return None
    if abs(dx) >= abs(dy) and dx != 0:
        return "RIGHT" if dx > 0 else "LEFT"
    return "DOWN" if dy > 0 else "UP"


@dataclass
class OccupancyWalker:
    """Predict → grade → replan. No path → stand (no hunt).

    Grades the same ``move DX,DY`` grammar as ``zelda_i.predict.walk_claim``
    via ``retro_harness.predict.grade_claims``.
    """

    grid: OccupancyGrid = field(default_factory=OccupancyGrid)
    path: list[tuple[int, int]] | None = None
    last_xy: tuple[int, int] | None = None
    last_dir: str | None = None
    misses: int = 0
    goal: tuple[int, int] | None = None

    def observe(self, xy: tuple[int, int]) -> None:
        xy = (int(xy[0]), int(xy[1]))
        if self.last_dir in WALK_DELTA and self.last_xy is not None:
            dx, dy = WALK_DELTA[self.last_dir]
            grade = grade_claims(
                f"move {dx},{dy}",
                {"x": self.last_xy[0], "y": self.last_xy[1]},
                {"x": xy[0], "y": xy[1]},
            )
            if not grade.ok:
                if xy == self.last_xy:
                    self.grid.mark_blocked_ahead(*self.last_xy, self.last_dir)
                    self.misses += 1
                self.path = None
        self.last_xy = xy

    def next_dir(
        self,
        xy: tuple[int, int],
        goal: tuple[int, int] | None = None,
    ) -> str | None:
        dest = self.goal if goal is None else goal
        xy = (int(xy[0]), int(xy[1]))
        if dest is None:
            self.last_dir = None
            return None
        dest = (int(dest[0]), int(dest[1]))
        if self.path is None:
            self.path = self.grid.shortest_path(xy, dest)
        direction = follow_path(self.path, xy)
        if direction is None:
            self.path = self.grid.shortest_path(xy, dest)
            direction = follow_path(self.path, xy)
        self.last_dir = direction
        return direction

"""Link walk model + occupancy BFS (no emulator).

Cardinal 1px/frame. Cells default passable; OccupancyWalker grades a predicted
step and blocks the cell ahead on a stuck miss, then replans. No path with
inferred blocks → forget those and replan once; spec-declared blocks stay and
a genuinely walled goal still stands.
Door clips (LEFT+UP residual) are not modeled here — those stay in ``level*_path``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from typing import Iterable

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
    # Cells blocked by a failed prediction rather than by the room spec.
    # Spec blocks are measured geometry and are never forgotten; these are
    # inferences from one 1px miss and can be wrong (a wall hug, a slide).
    inferred: set[tuple[int, int]] = field(default_factory=set)
    xmin: int = DEFAULT_BOUNDS[0]
    xmax: int = DEFAULT_BOUNDS[1]
    ymin: int = DEFAULT_BOUNDS[2]
    ymax: int = DEFAULT_BOUNDS[3]

    def in_bounds(self, x: int, y: int) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def passable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and (x, y) not in self.blocked

    def mark_blocked_ahead(
        self, x: int, y: int, direction: str, *, inferred: bool = True
    ) -> tuple[int, int]:
        """Record the cell the last predicted step failed to enter.

        Spec-declared cells already in ``blocked`` stay spec: a later miss
        must not tag them inferred, or forget drops measured geometry.
        ``inferred=False`` marks the cell permanent (sticky).
        """
        cell = predicted_xy(x, y, direction)
        if inferred and cell not in self.blocked:
            self.inferred.add(cell)
        self.blocked.add(cell)
        return cell

    def shortest_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        *,
        extra_blocked: Iterable[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int]] | None:
        """4-connected BFS. ``start`` is always allowed so a pocket can escape.

        ``extra_blocked`` specifies temporary obstacles (e.g. live enemy disks)
        for this search without mutating ``blocked``.
        """
        sx, sy = int(start[0]), int(start[1])
        gx, gy = int(goal[0]), int(goal[1])
        if (sx, sy) == (gx, gy):
            return [(sx, sy)]

        extra = set(extra_blocked) if extra_blocked else None
        if extra:
            extra.discard((sx, sy))
            extra.discard((gx, gy))

        def ok(x: int, y: int) -> bool:
            if (x, y) == (sx, sy):
                return True
            if extra and (x, y) in extra:
                return False
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
    """Predict → grade → replan; forget the inferred blocks before standing.

    Grades the same ``move DX,DY`` grammar as ``zelda_i.walk.predict.walk_claim``
    via ``retro_harness.predict.grade_claims``.

    ``sticky=True`` treats misses as spec-declared so ``next_dir`` will not
    forget them and yo-yo (0x31 water pocket / 0x5E Darknut flank).
    ``slide=True`` permits overworld 2px steps; only true no-move is a miss.
    """

    grid: OccupancyGrid = field(default_factory=OccupancyGrid)
    path: list[tuple[int, int]] | None = None
    last_xy: tuple[int, int] | None = None
    misses: int = 0
    forgets: int = 0
    goal: tuple[int, int] | None = None
    sticky: bool = False
    slide: bool = False
    _last_dir: str | None = field(default=None, repr=False)
    _graded: bool = field(default=False, repr=False)

    @property
    def last_dir(self) -> str | None:
        return self._last_dir

    @last_dir.setter
    def last_dir(self, value: str | None) -> None:
        self._last_dir = value
        self._graded = False

    def observe(
        self,
        xy: tuple[int, int],
        *,
        sticky: bool | None = None,
        slide: bool | None = None,
    ) -> None:
        xy = (int(xy[0]), int(xy[1]))
        is_sticky = self.sticky if sticky is None else sticky
        is_slide = self.slide if slide is None else slide

        if self._last_dir in WALK_DELTA and self.last_xy is not None and not self._graded:
            self._graded = True
            miss = False
            if is_slide:
                # Overworld slide: Link may step 2px or slide.
                # Only a true no-move is an obstacle miss.
                miss = (xy == self.last_xy)
            else:
                dx, dy = WALK_DELTA[self._last_dir]
                grade = grade_claims(
                    f"move {dx},{dy}",
                    {"x": self.last_xy[0], "y": self.last_xy[1]},
                    {"x": xy[0], "y": xy[1]},
                )
                miss = not grade.ok
            if miss:
                # Block the predicted cell even on a 1px slide (live 0x6e
                # south pocket: UP along diamonds oscillated 72↔73 and never
                # counted as stuck-in-place).
                self.grid.mark_blocked_ahead(
                    *self.last_xy, self._last_dir, inferred=not is_sticky
                )
                self.misses += 1
                self.path = None
        self.last_xy = xy

    def next_dir(
        self,
        xy: tuple[int, int],
        goal: tuple[int, int] | None = None,
        *,
        extra_blocked: Iterable[tuple[int, int]] | None = None,
        sticky: bool | None = None,
        slide: bool | None = None,
    ) -> str | None:
        self.observe(xy, sticky=sticky, slide=slide)
        is_sticky = self.sticky if sticky is None else sticky
        dest = self.goal if goal is None else goal
        xy = (int(xy[0]), int(xy[1]))
        if dest is None:
            self.last_dir = None
            return None
        dest = (
            min(max(int(dest[0]), self.grid.xmin), self.grid.xmax),
            min(max(int(dest[1]), self.grid.ymin), self.grid.ymax),
        )
        if self.goal != dest:
            self.goal = dest
            self.path = None
        if extra_blocked is not None:
            self.path = None

        if self.path is None:
            self.path = self.grid.shortest_path(xy, dest, extra_blocked=extra_blocked)
        direction = follow_path(self.path, xy)
        if direction is None:
            self.path = self.grid.shortest_path(xy, dest, extra_blocked=extra_blocked)
            direction = follow_path(self.path, xy)
        if direction is None and not is_sticky and self.grid.inferred:
            # Inferred blocks come from one failed 1px prediction, not ground
            # truth — a wall hug or a slide fences off a free cell. A walker
            # that has fenced itself in forgets them and replans; standing
            # there is how enter_6f_key burned its whole 4,000f budget in
            # "band_wait". A real wall is re-blocked by the next observe(),
            # so this self-corrects instead of looping. Spec-declared blocks
            # (measured geometry) are kept.
            self.grid.blocked -= self.grid.inferred
            self.grid.inferred.clear()
            self.forgets += 1
            self.path = self.grid.shortest_path(xy, dest, extra_blocked=extra_blocked)
            direction = follow_path(self.path, xy)
        self.last_dir = direction
        return direction

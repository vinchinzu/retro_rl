"""Link walk model + occupancy BFS (no emulator).

Cardinal 1px/frame. Cells default passable; OccupancyWalker grades a predicted
step and blocks the cell ahead on a stuck miss, then replans. No path with
inferred blocks → forget those and replan once; spec-declared blocks stay and
a genuinely walled goal still stands. A miss on a cell flagged
``transient_occupants`` is not wall geometry — a live body moves on next
frame, so the grid never learns it (rr-8t4.4 0x23).
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
    "measured_walker",
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

    def nearest_open(
        self, x: int, y: int, *, radius: int = 24
    ) -> tuple[int, int] | None:
        """Closest passable cell to ``(x, y)``, or ``None`` within ``radius``.

        Hand-written goals land inside geometry more often than anyone
        expects: L2 ``0x6e``'s key-door band target ``(120, 113)`` sits in a
        diamond, so ``shortest_path`` correctly reported *no path* and the
        walk stood in ``band_wait`` for 3999 of its 4000 frames without ever
        moving a pixel. Standing on an impossible goal is never the useful
        answer — walking to the nearest cell that exists is.

        Ring search by Chebyshev distance, so the first hit is the nearest.
        """
        if self.passable(x, y):
            return (x, y)
        for r in range(1, int(radius) + 1):
            best: tuple[int, int] | None = None
            best_d = None
            for dx in range(-r, r + 1):
                for dy in (-r, r) if abs(dx) != r else range(-r, r + 1):
                    cx, cy = x + dx, y + dy
                    if not self.passable(cx, cy):
                        continue
                    d = abs(dx) + abs(dy)
                    if best_d is None or d < best_d:
                        best, best_d = (cx, cy), d
            if best is not None:
                return best
        return None

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
        """Shortest 4-connected path, holding each heading to the last cell.

        Turn count matters as much as length. Link snaps his off-axis position
        when he changes direction, so a per-pixel staircase — LEFT, UP, LEFT,
        UP — never advances: every turn frame grades as a prediction miss, the
        walker marks a wall that is not there, fences itself in, forgets the
        inferred blocks and lays the same staircase again. Measured on L1
        ``0x23`` chasing a heart 16px away: 285 misses, 49 forgets, zero
        progress. Straight runs walk; staircases do not.

        Flooding from the goal and then descending that gradient — keeping the
        current heading while it still shortens the path — gives a path of the
        same length with turns only at real corners.

        ``start`` is always allowed so a pocket can escape. ``extra_blocked``
        specifies temporary obstacles (e.g. live enemy disks) for this search
        without mutating ``blocked``.
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

        # A goal that is out of bounds *or* inside geometry has no path. The
        # flood seeds ``dist`` at the goal, so an unwalkable goal that slips
        # past this guard is a BFS rooted inside the wall: it hands back a
        # path whose last step walks into the wall, and the caller presses
        # that button until its budget runs out. That is what made
        # ``collect_skip_unreachable`` dead code for a walled waypoint (48
        # frames of pressing into a statue instead of skipping) and what
        # latched ``_route_replanning`` forever in ``dungeon/route_entry.py``.
        if not ok(gx, gy) or not self.in_bounds(gx, gy):
            return None

        # Distance-to-goal for every cell the start can reach.
        dist: dict[tuple[int, int], int] = {(gx, gy): 0}
        queue: deque[tuple[int, int]] = deque([(gx, gy)])
        while queue:
            x, y = queue.popleft()
            if (x, y) == (sx, sy):
                break
            step = dist[(x, y)] + 1
            for dx, dy in WALK_DELTA.values():
                cell = (x + dx, y + dy)
                if cell in dist or not ok(*cell):
                    continue
                dist[cell] = step
                queue.append(cell)
        if (sx, sy) not in dist:
            return None

        path: list[tuple[int, int]] = [(sx, sy)]
        x, y = sx, sy
        heading: tuple[int, int] | None = None
        while (x, y) != (gx, gy):
            nearer = [
                (delta, cell)
                for delta in WALK_DELTA.values()
                for cell in ((x + delta[0], y + delta[1]),)
                if dist.get(cell, -1) == dist[(x, y)] - 1
            ]
            if not nearer:
                return None
            delta, cell = next(
                (pair for pair in nearer if pair[0] == heading), nearer[0]
            )
            path.append(cell)
            heading = delta
            x, y = cell
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
    # Goals that were inside geometry and were moved to the nearest open cell.
    retargets: int = 0
    # Opt in to that move. Off by default: it changes arrival frames, and a
    # frame-perfect chain downstream of the walk will notice.
    retarget_blocked_goal: bool = False
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
        transient_occupants: Iterable[tuple[int, int]] | None = None,
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
                cell = predicted_xy(*self.last_xy, self._last_dir)
                # A live body on the predicted cell explains the miss without
                # it being wall geometry — the body moves on, so a permanent
                # block here would scar real floor for good (rr-8t4.4 0x23).
                # Still count the miss; let next_dir's own extra_blocked
                # route around the body instead of the grid remembering it.
                transient = (
                    transient_occupants is not None
                    and cell in transient_occupants
                )
                if not transient:
                    # Block the predicted cell even on a 1px slide (live 0x6e
                    # south pocket: UP along diamonds oscillated 72↔73 and
                    # never counted as stuck-in-place).
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
        transient_occupants: Iterable[tuple[int, int]] | None = None,
        sticky: bool | None = None,
        slide: bool | None = None,
    ) -> str | None:
        """``transient_occupants`` is independent of ``extra_blocked``.

        ``extra_blocked`` steers *this frame's* BFS around live bodies and
        commonly carves out the fight target's own cell (it is the goal,
        not an obstacle). ``transient_occupants`` answers a different
        question — "was a live body physically here" — for miss-grading
        only, so pass the full body set (target included). Callers that
        omit it get today's behavior: a miss always blocks, per ``sticky``.
        """
        self.observe(
            xy,
            sticky=sticky,
            slide=slide,
            transient_occupants=transient_occupants,
        )
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

        # BFS neighbors must be in-bounds, so a start west of xmin (L1 0x45
        # door column x=32 vs DEFAULT_BOUNDS xmin=40) has no path at all —
        # collect then idled `collect_skip_unreachable` for thousands of
        # frames. Walk toward the box; the next call BFSs from inside.
        if not self.grid.in_bounds(*xy):
            if xy[0] < self.grid.xmin:
                direction = "RIGHT"
            elif xy[0] > self.grid.xmax:
                direction = "LEFT"
            elif xy[1] < self.grid.ymin:
                direction = "DOWN"
            else:
                direction = "UP"
            self.last_dir = direction
            return direction

        # A goal inside geometry has no path by definition, and standing is
        # the one answer that is never useful — but retargeting *here*
        # retargets route, collect and chase goals alike, which shifts the
        # frame at which a walk arrives, and the L1 chain is frame-perfect
        # (turning this on globally took Clean M5 to a red aquamentus_heart
        # at 18830f). Opt in per walker. The two goals that genuinely need it
        # retarget themselves, at the call site: ``engine._chase_goal`` and
        # the collect-waypoint branch of ``engine._collect_policy``.
        if self.retarget_blocked_goal and not self.grid.passable(*dest):
            open_dest = self.grid.nearest_open(*dest)
            if open_dest is not None and open_dest != dest:
                self.retargets += 1
                dest = open_dest
                self.goal = dest
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


def measured_walker(
    ram,
    bounds: tuple[int, int, int, int] = DEFAULT_BOUNDS,
    *,
    sticky: bool = True,
    retarget_blocked_goal: bool = True,
) -> "OccupancyWalker | None":
    """A walker that already knows the room, from the live ``$6530`` map.

    ``None`` when ``ram`` carries no cart-WRAM window, so a caller can fall
    back rather than silently take an empty grid. An empty grid is the worst
    of the options: the walker learns each wall by bumping it and, because a
    non-sticky walker forgets its inferred blocks whenever it fences itself
    in, it can re-learn the same wall forever. ``enter_6f_key`` spent its
    whole 4,000f budget that way, and the entry-route replan hit the same
    loop until it was made sticky.

    Defaults to ``sticky=True``: room geometry does not change mid-walk, so a
    measured miss is worth keeping. Pass ``sticky=False`` only where live
    bodies, not walls, are the expected cause of a miss.
    """
    from zelda_i.dungeon.tilemap import blocked_link_cells, has_room_tile_map

    if ram is None or not has_room_tile_map(ram):
        return None
    xmin, xmax, ymin, ymax = bounds
    return OccupancyWalker(
        grid=OccupancyGrid(
            blocked=set(blocked_link_cells(ram, bounds)),
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
        ),
        sticky=sticky,
        # A walker that knows the real walls is exactly the one that will
        # report "no path" for a hand-written goal inside geometry, so it is
        # also the one that needs the retarget. Callers that want the raw
        # verdict can turn it back off.
        retarget_blocked_goal=retarget_blocked_goal,
    )

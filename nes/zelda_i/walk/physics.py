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

from typing import Callable, Iterable

from retro_harness.predict import grade_claims

__all__ = [
    "DEFAULT_BOUNDS",
    "OPPOSITE",
    "WALK_DELTA",
    "WALK_SPEED",
    "OccupancyGrid",
    "OccupancyWalker",
    "follow_path",
    "measured_walker",
    "predicted_xy",
    "lattice_route",
    "lattice_starts",
    "lattice_step",
    "lattice_toward",
]

WALK_SPEED = 1
WALK_DELTA: dict[str, tuple[int, int]] = {
    "UP": (0, -WALK_SPEED),
    "DOWN": (0, WALK_SPEED),
    "LEFT": (-WALK_SPEED, 0),
    "RIGHT": (WALK_SPEED, 0),
}
OPPOSITE: dict[str, str] = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
# Dungeon playfield (Link collision). North door approach y≈93 sits inside.
DEFAULT_BOUNDS: tuple[int, int, int, int] = (40, 216, 77, 205)
_DRIFT_REPLAN = 8
LATTICE_STEP = 8


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
    # Plan on the ROM turn lattice: horizontal only on rows (y % 8 == 5),
    # vertical only on columns (x % 8 == 0). A pixel path off it is one the
    # game will not walk — pressing RIGHT on y=137 slides Link to 139, the
    # plan says UP, and he flips every frame (live 0x7A: 574 reversals).
    lattice: bool = True

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
        if self.lattice:
            path = self._shortest_path(start, self.lattice_goal(goal), extra_blocked, True)
            if path is not None:
                return path
        return self._shortest_path(start, goal, extra_blocked, False)

    def lattice_goal(self, goal: tuple[int, int]) -> tuple[int, int]:
        """Nearest open pixel to ``goal`` that lies on a lattice row or column."""
        gx, gy = int(goal[0]), int(goal[1])
        if on_lattice(gx, gy) or not self.in_bounds(gx, gy):
            return (gx, gy)
        row = gy - (gy - 5) % LATTICE_STEP
        col = gx - gx % LATTICE_STEP
        options = [
            (gx, row), (gx, row + LATTICE_STEP), (col, gy), (col + LATTICE_STEP, gy)
        ]
        options = [c for c in options if self.passable(*c)]
        if not options:
            return (gx, gy)
        return min(options, key=lambda c: abs(c[0] - gx) + abs(c[1] - gy))

    def _shortest_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        extra_blocked: Iterable[tuple[int, int]] | None,
        lattice: bool,
    ) -> list[tuple[int, int]] | None:
        sx, sy = int(start[0]), int(start[1])
        gx, gy = int(goal[0]), int(goal[1])
        if (sx, sy) == (gx, gy):
            return [(sx, sy)]

        extra = set(extra_blocked) if extra_blocked else None
        if extra:
            extra.discard((sx, sy))
            extra.discard((gx, gy))

        # A goal that is out of bounds *or* inside geometry has no path. The
        # flood seeds ``dist`` at the goal, so an unwalkable goal that slips
        # past this guard is a BFS rooted inside the wall: it hands back a
        # path whose last step walks into the wall, and the caller presses
        # that button until its budget runs out. That is what made
        # ``collect_skip_unreachable`` dead code for a walled waypoint (48
        # frames of pressing into a statue instead of skipping) and what
        # latched ``_route_replanning`` forever in ``dungeon/route_entry.py``.
        if not self.in_bounds(gx, gy) or (
            (gx, gy) != (sx, sy)
            and ((extra and (gx, gy) in extra) or (gx, gy) in self.blocked)
        ):
            return None

        # Distance-to-goal for every cell the start can reach. Cells are
        # flat indices into the bounds box; the start alone may sit outside
        # it or in a block. Every call runs this flood (0x19: 87% of a fight
        # frame), so the neighbour tests are inlined; the distances, and so
        # the path, are the same as the tuple-dict flood this replaced.
        xmin, xmax, ymin, ymax = self.xmin, self.xmax, self.ymin, self.ymax
        width = xmax - xmin + 1
        size = width * (ymax - ymin + 1)
        blocked = self.blocked
        dist = [-1] * size
        start_dist = -1
        goal_i = (gy - ymin) * width + (gx - xmin)
        dist[goal_i] = 0
        queue: deque[int] = deque([goal_i])
        while queue and start_dist < 0:
            i = queue.popleft()
            y, xo = divmod(i, width)
            y += ymin
            x = xo + xmin
            step = dist[i] + 1
            horizontal = not lattice or (y - 5) % LATTICE_STEP == 0
            vertical = not lattice or x % LATTICE_STEP == 0
            for nx, ny, ok_axis in (
                (x, y - 1, vertical),
                (x, y + 1, vertical),
                (x - 1, y, horizontal),
                (x + 1, y, horizontal),
            ):
                if not ok_axis:
                    continue
                if nx == sx and ny == sy:
                    start_dist = step
                    break
                if not (xmin <= nx <= xmax and ymin <= ny <= ymax):
                    continue
                j = (ny - ymin) * width + (nx - xmin)
                if dist[j] >= 0:
                    continue
                cell = (nx, ny)
                if cell in blocked or (extra and cell in extra):
                    continue
                dist[j] = step
                queue.append(j)
        if start_dist < 0:
            return None

        def dist_at(x: int, y: int) -> int:
            if x == sx and y == sy:
                return start_dist
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                return -1
            return dist[(y - ymin) * width + (x - xmin)]

        path: list[tuple[int, int]] = [(sx, sy)]
        x, y = sx, sy
        heading: tuple[int, int] | None = None
        while (x, y) != (gx, gy):
            here = dist_at(x, y)
            nearer = [
                (delta, cell)
                for delta in WALK_DELTA.values()
                for cell in ((x + delta[0], y + delta[1]),)
                if dist_at(*cell) == here - 1
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


def on_lattice(x: int, y: int) -> bool:
    return int(x) % LATTICE_STEP == 0 or (int(y) - 5) % LATTICE_STEP == 0


def _lattice_move(x: int, y: int, dx: int) -> bool:
    """A 1 px step from ``(x, y)`` the ROM lets Link take (``dx`` 0 = vertical)."""
    return (int(y) - 5) % LATTICE_STEP == 0 if dx else int(x) % LATTICE_STEP == 0


def follow_path(
    path: list[tuple[int, int]] | None,
    xy: tuple[int, int],
    passable: Callable[[int, int], bool] | None = None,
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
    first = _cardinal(dx, dy)
    # A short sidestep before a long leg is a pixel the game will not stop on:
    # Link steps 1-2 px and turns only on the ROM's 8 px grid, so "DOWN 1,
    # then RIGHT" overshoots and flips every frame (live 0x79: y 121<->123
    # for ~1300 frames). Press the long leg; the ROM snap absorbs the offset.
    leg = _leg(path, idx, first)
    if leg <= SIDESTEP_PX and idx + leg + 1 < len(path):
        ax, ay = path[idx + leg]
        bx, by = path[idx + leg + 1]
        second = _cardinal(bx - ax, by - ay)
        run = _leg(path, idx + leg, second)
        dx2, dy2 = WALK_DELTA[second]
        if (
            second != first
            and run > leg
            and passable is not None
            and all(passable(x + dx2 * k, y + dy2 * k) for k in range(1, leg + 2))
        ):
            return second
    return first


# Legs this short are absorbed by the ROM's turn snap (see ``follow_path``).
SIDESTEP_PX = 3


def _cardinal(dx: int, dy: int) -> str:
    if abs(dx) >= abs(dy) and dx != 0:
        return "RIGHT" if dx > 0 else "LEFT"
    return "DOWN" if dy > 0 else "UP"


def _leg(path: list[tuple[int, int]], start: int, direction: str) -> int:
    """Pixels ``path`` holds ``direction`` from node ``start``."""
    n = 0
    for i in range(start, len(path) - 1):
        (ax, ay), (bx, by) = path[i], path[i + 1]
        if _cardinal(bx - ax, by - ay) != direction:
            break
        n += 1
    return n


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
        extra = frozenset(extra_blocked) if extra_blocked is not None else frozenset()
        extra_blocked = extra if extra_blocked is not None else None

        def open_cell(x: int, y: int) -> bool:
            return self.grid.passable(x, y) and (x, y) not in extra

        if self.path is None:
            self.path = self.grid.shortest_path(xy, dest, extra_blocked=extra_blocked)
        direction = follow_path(self.path, xy, open_cell)
        if direction is None:
            self.path = self.grid.shortest_path(xy, dest, extra_blocked=extra_blocked)
            direction = follow_path(self.path, xy, open_cell)
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
            direction = follow_path(self.path, xy, open_cell)
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


# ---------------------------------------------- overworld lattice walk ---
# ``dungeon.tilemap.ow_walkable_nodes`` is the ROM's own collision test on
# the 8 px turn grid. A route over it is a list of corners Link can actually
# turn on, so a walk never has to learn a rock by bumping it.
# One turn costs about as much as this many 8 px steps. Link loses ~3 frames
# snapping at each corner, and a staircase is how walkers here fail.
LATTICE_TURN_COST = 2
_LATTICE_DIRS: dict[str, tuple[int, int]] = {
    "UP": (0, -LATTICE_STEP),
    "DOWN": (0, LATTICE_STEP),
    "LEFT": (-LATTICE_STEP, 0),
    "RIGHT": (LATTICE_STEP, 0),
}


# Pixels from a corner that count as on it (Link can step 2 px a frame).
LATTICE_SNAP_PX = 3


def lattice_starts(x: int, y: int) -> tuple[tuple[int, int], ...]:
    """The lattice nodes Link at ``(x, y)`` can reach without turning.

    On a row (y % 8 == 5) that is the two nodes either side on that row; on a
    column (x % 8 == 0) the two above and below; both when he is on a node.
    Knocked fully off the grid he is between four, and all four are offered.
    """
    x, y = int(x), int(y)
    xs = sorted({x - x % LATTICE_STEP, x - x % LATTICE_STEP + (LATTICE_STEP if x % LATTICE_STEP else 0)})
    oy = (y - 5) % LATTICE_STEP
    ys = sorted({y - oy, y - oy + (LATTICE_STEP if oy else 0)})
    return tuple((nx, ny) for nx in xs for ny in ys)


def lattice_component(
    nodes: frozenset[tuple[int, int]] | set[tuple[int, int]],
    start: tuple[int, int],
) -> set[tuple[int, int]]:
    """Every node Link can walk to from ``start`` (no ladder crossings)."""
    seen = {n for n in lattice_starts(*start) if n in nodes}
    queue = deque(seen)
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((0, -LATTICE_STEP), (0, LATTICE_STEP), (-LATTICE_STEP, 0), (LATTICE_STEP, 0)):
            cell = (x + dx, y + dy)
            if cell in nodes and cell not in seen:
                seen.add(cell)
                queue.append(cell)
    return seen


def lattice_route(
    nodes: frozenset[tuple[int, int]] | set[tuple[int, int]],
    start: tuple[int, int],
    goals: set[tuple[int, int]] | frozenset[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    """Fewest-turns-then-shortest route from ``start`` to any goal node.

    Returns the corners after ``start`` (the last is the goal), ``[]`` when
    Link already stands on a goal, ``None`` when no goal is reachable.
    ``start`` may be off the lattice; it joins at :func:`lattice_starts`.
    """
    import heapq

    goals = set(goals) & set(nodes)
    if not goals:
        return None
    sx, sy = int(start[0]), int(start[1])
    if (sx, sy) in goals:
        return []
    heap: list[tuple[int, int, tuple[int, int], str | None]] = []
    prev: dict[tuple[tuple[int, int], str | None], tuple[tuple[int, int], str | None] | None] = {}
    best: dict[tuple[tuple[int, int], str | None], int] = {}
    tick = 0
    starts = [n for n in lattice_starts(sx, sy) if n in nodes]
    if not starts:
        # Link stands where the tile model calls solid: the ROM tests only
        # the leading foot, so a LEFT walk parks him on a node whose other
        # foot is rock (OW 0x15 (96,181), Blue Ring power-on 4: no start, no
        # route, and the hand walk pressed LEFT into the rock). Join at the
        # walkable nodes one step off instead.
        starts = [
            (n[0] + dx, n[1] + dy)
            for n in lattice_starts(sx, sy)
            for dx, dy in _LATTICE_DIRS.values()
            if (n[0] + dx, n[1] + dy) in nodes
        ]
    for node in starts:
        d = abs(node[0] - sx) + abs(node[1] - sy)
        key = (node, None)
        cost = (d + LATTICE_STEP - 1) // LATTICE_STEP
        if cost < best.get(key, 10**9):
            best[key] = cost
            prev[key] = None
            heapq.heappush(heap, (cost, tick, node, None))
            tick += 1
    end: tuple[tuple[int, int], str | None] | None = None
    while heap:
        cost, _, node, heading = heapq.heappop(heap)
        key = (node, heading)
        if cost > best.get(key, 10**9):
            continue
        if node in goals:
            end = key
            break
        for direction, (dx, dy) in _LATTICE_DIRS.items():
            nxt = (node[0] + dx, node[1] + dy)
            if nxt not in nodes:
                continue
            step = 1 + (LATTICE_TURN_COST if heading not in (None, direction) else 0)
            nkey = (nxt, direction)
            if cost + step < best.get(nkey, 10**9):
                best[nkey] = cost + step
                prev[nkey] = key
                heapq.heappush(heap, (cost + step, tick, nxt, direction))
                tick += 1
    if end is None:
        return None
    chain: list[tuple[tuple[int, int], str | None]] = []
    cur: tuple[tuple[int, int], str | None] | None = end
    while cur is not None:
        chain.append(cur)
        cur = prev[cur]
    chain.reverse()
    corners: list[tuple[int, int]] = [chain[0][0]]
    for (node, heading), (_nxt, nheading) in zip(chain[1:], chain[2:]):
        if nheading != heading:
            corners.append(node)
    corners.append(chain[-1][0])
    # Drop a repeated goal (a one-node route) and a start Link is already on.
    out = [c for i, c in enumerate(corners) if i == 0 or c != corners[i - 1]]
    if out and out[0] == (sx, sy):
        out = out[1:]
    # A corner a step or two away is already reached: Link moves up to 2 px
    # a frame and overshoots it, and routing back to it flips direction
    # every frame (L6 0x28, 143<->145 around the 144 corner, 6000 frames).
    # The perpendicular press that follows slides him onto the line.
    if len(out) > 1 and abs(out[0][0] - sx) + abs(out[0][1] - sy) <= LATTICE_SNAP_PX:
        out = out[1:]
    return out


def lattice_step(x: int, y: int, corner: tuple[int, int]) -> str | None:
    """Direction toward ``corner`` that Link can take from ``(x, y)`` now.

    Off the grid on one axis he can only move along the other, which is
    also the axis the corner is reached on for any route this module plans.
    """
    x, y = int(x), int(y)
    dx, dy = int(corner[0]) - x, int(corner[1]) - y
    on_col = x % LATTICE_STEP == 0
    on_row = (y - 5) % LATTICE_STEP == 0
    # Within snap of the corner's line: press the leg's own axis and let the
    # ROM's turn-grid slide put Link on the line.
    if dy and dx and abs(dx) <= LATTICE_SNAP_PX:
        return "DOWN" if dy > 0 else "UP"
    if dx and dy and abs(dy) <= LATTICE_SNAP_PX:
        return "RIGHT" if dx > 0 else "LEFT"
    if dx and (on_row or not on_col):
        return "RIGHT" if dx > 0 else "LEFT"
    if dy:
        return "DOWN" if dy > 0 else "UP"
    if dx:
        return "RIGHT" if dx > 0 else "LEFT"
    return None


def lattice_node(x: int, y: int) -> tuple[int, int]:
    """The lattice node nearest ``(x, y)`` (column x % 8 == 0, row y % 8 == 5)."""
    x, y = int(x), int(y)
    nx = (x + LATTICE_STEP // 2) // LATTICE_STEP * LATTICE_STEP
    ny = (y - 5 + LATTICE_STEP // 2) // LATTICE_STEP * LATTICE_STEP + 5
    return nx, ny


def lattice_toward(
    x: int, y: int, goal: tuple[int, int], *, tol: int = LATTICE_SNAP_PX
) -> str | None:
    """Open-floor step toward ``goal``'s lattice node; ``None`` once within ``tol``.

    The replacement for the hand-rolled "bigger axis first" step. That step
    presses UP off a column, the ROM slides Link sideways onto one, the
    bigger axis flips, and he twitches a pixel each frame (L1 0x63 patrol:
    992 reversals). No walls: callers on real geometry use a lattice route.
    """
    node = lattice_node(*goal)
    if abs(int(x) - node[0]) <= tol and abs(int(y) - node[1]) <= tol:
        return None
    return lattice_step(int(x), int(y), node)

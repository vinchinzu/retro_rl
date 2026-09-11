"""Occupancy BFS + 1px walk model (no emulator)."""

from __future__ import annotations

from zelda_i.walk.physics import (
    OccupancyGrid,
    OccupancyWalker,
    follow_path,
    predicted_xy,
)


def test_predicted_cardinal() -> None:
    assert predicted_xy(120, 141, "UP") == (120, 140)
    assert predicted_xy(120, 141, "LEFT") == (119, 141)


def test_bfs_around_blocked_wall() -> None:
    grid = OccupancyGrid()
    for y in range(100, 180):
        grid.blocked.add((120, y))
    path = grid.shortest_path((100, 181), (120, 93))
    assert path is not None
    assert path[0] == (100, 181)
    assert path[-1] == (120, 93)
    assert (120, 140) not in path
    direction = follow_path(path, (100, 181))
    assert direction in {"UP", "RIGHT", "LEFT"}


def test_miss_blocks_ahead_and_replans() -> None:
    grid = OccupancyGrid()
    start = (120, 141)
    grid.mark_blocked_ahead(*start, "UP")
    assert (120, 140) in grid.blocked
    path = grid.shortest_path(start, (120, 93))
    assert path is not None
    assert (120, 140) not in path
    assert follow_path(path, start) in {"LEFT", "RIGHT"}


def test_no_path_when_goal_walled_off() -> None:
    grid = OccupancyGrid(xmin=100, xmax=140, ymin=100, ymax=140)
    for x in range(100, 141):
        grid.blocked.add((x, 120))
    assert grid.shortest_path((120, 130), (120, 110)) is None


def test_walker_miss_blocks_ahead_and_sidesteps() -> None:
    walker = OccupancyWalker(goal=(120, 93))
    start = (120, 141)
    walker.observe(start)
    assert walker.next_dir(start) == "UP"
    walker.observe(start)
    assert (120, 140) in walker.grid.blocked
    assert walker.misses == 1
    assert walker.next_dir(start) in {"LEFT", "RIGHT"}
    path = walker.grid.shortest_path(start, (120, 93))
    if path is None:
        path = walker.grid.shortest_path(start, (120, 109))
    assert path is not None
    assert path[-1] in {(120, 93), (120, 109)}
    assert (120, 140) not in path


def test_walker_slide_still_blocks_predicted_cell() -> None:
    """UP into a diamond that slides Link 1px sideways must still block UP."""
    walker = OccupancyWalker(goal=(120, 113))
    start = (72, 181)
    walker.observe(start)
    assert walker.next_dir(start) == "UP"
    walker.observe((73, 181))
    assert walker.misses == 1
    assert (72, 180) in walker.grid.blocked
    assert walker.next_dir(start) != "UP"


def test_walker_stands_when_no_path() -> None:
    grid = OccupancyGrid(xmin=100, xmax=140, ymin=100, ymax=140)
    for x in range(100, 141):
        grid.blocked.add((x, 120))
    walker = OccupancyWalker(grid=grid, goal=(120, 110))
    start = (120, 130)
    walker.observe(start)
    assert walker.next_dir(start) is None
    assert walker.last_dir is None


def test_walker_forgets_inferred_blocks_before_standing() -> None:
    """A walker fenced in by slide-inferred blocks replans, it does not stand."""
    from zelda_i.walk.physics import OccupancyGrid, OccupancyWalker

    grid = OccupancyGrid(xmin=0, xmax=10, ymin=0, ymax=10)
    walker = OccupancyWalker(grid=grid)
    # Fence (5,5) in with cells a failed prediction inferred, not spec walls.
    for direction in ("RIGHT", "LEFT", "DOWN", "UP"):
        grid.mark_blocked_ahead(5, 5, direction)
    step = walker.next_dir((5, 5), (9, 5))
    assert step is not None
    assert walker.forgets == 1
    assert not grid.blocked


def test_spec_blocks_are_never_forgotten() -> None:
    """Forgetting is not hunting: measured geometry still stands the walker."""
    from zelda_i.walk.physics import OccupancyGrid, OccupancyWalker

    grid = OccupancyGrid(xmin=0, xmax=10, ymin=0, ymax=10)
    for cell in ((6, 5), (4, 5), (5, 6), (5, 4)):
        grid.blocked.add(cell)
    walker = OccupancyWalker(grid=grid)
    assert walker.next_dir((5, 5), (9, 5)) is None
    assert walker.forgets == 0
    assert len(grid.blocked) == 4


def test_miss_on_spec_cell_does_not_make_it_inferred() -> None:
    """A miss into measured geometry must not let forget drop the spec cell."""
    grid = OccupancyGrid(xmin=0, xmax=10, ymin=0, ymax=10)
    grid.blocked.add((6, 5))
    grid.mark_blocked_ahead(5, 5, "RIGHT")
    assert (6, 5) in grid.blocked
    assert (6, 5) not in grid.inferred
    walker = OccupancyWalker(grid=grid)
    for direction in ("LEFT", "DOWN", "UP"):
        grid.mark_blocked_ahead(5, 5, direction)
    step = walker.next_dir((5, 5), (9, 5))
    assert (6, 5) in grid.blocked
    assert walker.forgets == 1
    assert step is not None


def test_walker_sticky_does_not_forget_inferred_blocks() -> None:
    """Sticky walker never drops inferred blocks: stands instead of yo-yoing."""
    grid = OccupancyGrid(xmin=0, xmax=10, ymin=0, ymax=10)
    walker = OccupancyWalker(grid=grid, sticky=True)
    for direction in ("RIGHT", "LEFT", "DOWN", "UP"):
        grid.mark_blocked_ahead(5, 5, direction)
    step = walker.next_dir((5, 5), (9, 5))
    assert step is None
    assert walker.forgets == 0
    assert len(grid.blocked) == 4


def test_walker_slide_allows_overworld_2px_step() -> None:
    """Slide walker accepts 2px step as valid progress; only true no-move misses."""
    grid = OccupancyGrid(xmin=0, xmax=255, ymin=0, ymax=255)
    walker = OccupancyWalker(grid=grid, slide=True, goal=(100, 50))
    # Frame 1: start
    dir1 = walker.next_dir((100, 100))
    assert dir1 == "UP"
    # Frame 2: 2px step in overworld (100, 98)
    dir2 = walker.next_dir((100, 98))
    assert dir2 == "UP"
    assert walker.misses == 0
    # Frame 3: no move (blocked by obstacle)
    dir3 = walker.next_dir((100, 98))
    assert walker.misses == 1
    assert (100, 97) in walker.grid.blocked


def test_walker_extra_blocked_routes_around_live_bodies() -> None:
    """extra_blocked avoids dynamic bodies without modifying persistent grid.blocked."""
    grid = OccupancyGrid(xmin=0, xmax=20, ymin=0, ymax=20)
    walker = OccupancyWalker(grid=grid, goal=(10, 5))
    # Direct path north from (10, 10) would step UP to (10, 9)
    extra = {(10, 9), (10, 8)}
    step = walker.next_dir((10, 10), extra_blocked=extra)
    assert step in {"LEFT", "RIGHT"}
    # grid.blocked must remain clean
    assert not grid.blocked


def test_walker_goal_clamping_and_replan_on_change() -> None:
    """Out-of-bounds goal is clamped; changing goal invalidates stale path."""
    grid = OccupancyGrid(xmin=40, xmax=200, ymin=70, ymax=190)
    walker = OccupancyWalker(grid=grid)
    step = walker.next_dir((100, 100), goal=(300, 100))
    assert walker.goal == (200, 100)
    assert step == "RIGHT"

    # Change goal to the left
    step2 = walker.next_dir((100, 100), goal=(50, 100))
    assert walker.goal == (50, 100)
    assert step2 == "LEFT"


def test_walker_auto_observes_on_next_dir() -> None:
    """next_dir automatically observes prior step without explicit observe() call."""
    walker = OccupancyWalker(goal=(120, 93))
    start = (120, 141)
    step1 = walker.next_dir(start)
    assert step1 == "UP"
    # No movement
    step2 = walker.next_dir(start)
    assert walker.misses == 1
    assert (120, 140) in walker.grid.blocked
    assert step2 in {"LEFT", "RIGHT"}


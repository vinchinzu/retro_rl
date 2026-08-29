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

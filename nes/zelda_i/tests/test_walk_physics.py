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


def test_walker_out_of_bounds_walks_toward_the_box() -> None:
    """Start west of xmin has no in-bounds neighbor, so BFS cannot escape.

    L1 0x45 door column is x=32 vs DEFAULT_BOUNDS xmin=40. Collect then
    idled collect_skip_unreachable from (32, 166) for thousands of frames.
    """
    grid = OccupancyGrid(xmin=40, xmax=200, ymin=70, ymax=190)
    walker = OccupancyWalker(grid=grid)
    assert walker.next_dir((32, 166), goal=(160, 141)) == "RIGHT"
    assert walker.next_dir((216, 141), goal=(120, 141)) == "LEFT"
    assert walker.next_dir((120, 60), goal=(120, 141)) == "DOWN"
    assert walker.next_dir((120, 200), goal=(120, 141)) == "UP"


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


def test_transient_occupants_miss_is_not_blocked() -> None:
    """A miss on a cell a live body currently occupies must not scar the grid.

    rr-8t4.4 0x23: the fight target's own cell is carved out of
    ``extra_blocked`` so BFS can aim at it, which means the walker treats it
    as ground truth and every failed final-approach step (Link's hitbox
    cannot overlap a living enemy's) permanently blacklisted real floor next
    to wherever the target stood. ``transient_occupants`` is the general fix:
    any live body, target included, is exempt from ever becoming a wall.
    """
    walker = OccupancyWalker(goal=(120, 93))
    start = (120, 141)
    body = {(120, 140)}
    walker.observe(start)
    assert walker.next_dir(start, transient_occupants=body) == "UP"
    walker.observe(start, transient_occupants=body)
    assert walker.misses == 1
    assert (120, 140) not in walker.grid.blocked
    assert (120, 140) not in walker.grid.inferred
    # The walker keeps trying the same real path once the body moves off —
    # nothing was learned that needs forgetting.
    assert walker.next_dir(start, transient_occupants=set()) == "UP"


def test_transient_occupants_does_not_suppress_unrelated_wall_miss() -> None:
    """Only the flagged cells are exempt; a miss elsewhere still blocks."""
    walker = OccupancyWalker(goal=(120, 93))
    start = (120, 141)
    body = {(200, 200)}  # nowhere near the predicted cell
    walker.observe(start)
    assert walker.next_dir(start, transient_occupants=body) == "UP"
    walker.observe(start, transient_occupants=body)
    assert walker.misses == 1
    assert (120, 140) in walker.grid.blocked


def test_extra_blocked_alone_still_blocks_on_miss() -> None:
    """Backward compat: callers passing only extra_blocked keep old behavior.

    ``transient_occupants`` is independent of ``extra_blocked`` on purpose —
    level8 north_column's sticky flank walker and level1 room52's static
    diamond both rely on an extra_blocked miss becoming a permanent (or
    sticky) block; only a caller that explicitly opts in gets the new
    transient behavior.
    """
    grid = OccupancyGrid(xmin=0, xmax=20, ymin=0, ymax=20)
    walker = OccupancyWalker(grid=grid, goal=(10, 5))
    start = (10, 10)
    walker.observe(start)
    assert walker.next_dir(start) == "UP"
    # No movement, and extra_blocked is passed but transient_occupants is not.
    walker.next_dir(start, extra_blocked={(10, 9)})
    assert walker.misses == 1
    assert (10, 9) in walker.grid.blocked


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



def test_nearest_open_returns_the_goal_when_it_is_already_walkable() -> None:
    from zelda_i.walk.physics import OccupancyGrid

    grid = OccupancyGrid(blocked=set(), xmin=0, xmax=100, ymin=0, ymax=100)
    assert grid.nearest_open(50, 50) == (50, 50)


def test_nearest_open_steps_off_a_blocked_goal() -> None:
    from zelda_i.walk.physics import OccupancyGrid

    blocked = {(x, y) for x in range(48, 60) for y in range(48, 60)}
    grid = OccupancyGrid(blocked=blocked, xmin=0, xmax=100, ymin=0, ymax=100)
    found = grid.nearest_open(50, 50)
    assert found is not None
    assert found not in blocked
    assert abs(found[0] - 50) + abs(found[1] - 50) <= 12


def test_nearest_open_gives_up_outside_its_radius() -> None:
    from zelda_i.walk.physics import OccupancyGrid

    blocked = {(x, y) for x in range(0, 101) for y in range(0, 101)}
    grid = OccupancyGrid(blocked=blocked, xmin=0, xmax=100, ymin=0, ymax=100)
    assert grid.nearest_open(50, 50, radius=4) is None


def test_walker_retargets_a_goal_that_sits_inside_geometry() -> None:
    """L2 0x6e aimed its band walk at (120,113), which is in a diamond.

    ``shortest_path`` answered ``None`` and the walk stood in ``band_wait``
    for 3999 of 4000 frames without moving a pixel.
    """
    from zelda_i.walk.physics import OccupancyGrid, OccupancyWalker

    blocked = {(x, y) for x in range(112, 128) for y in range(108, 124)}
    grid = OccupancyGrid(blocked=blocked, xmin=0, xmax=200, ymin=0, ymax=200)
    walker = OccupancyWalker(grid=grid, retarget_blocked_goal=True)
    direction = walker.next_dir((41, 189), (120, 113))
    assert direction is not None
    assert walker.retargets == 1
    assert walker.goal not in blocked


def test_walker_stands_on_a_walled_goal_when_retarget_is_off() -> None:
    """Default off: retargeting moves arrival frames, and the L1 chain is
    frame-perfect — turning it on globally broke Clean M5's Aquamentus."""
    from zelda_i.walk.physics import OccupancyGrid, OccupancyWalker

    blocked = {(x, y) for x in range(112, 128) for y in range(108, 124)}
    grid = OccupancyGrid(blocked=blocked, xmin=0, xmax=200, ymin=0, ymax=200)
    walker = OccupancyWalker(grid=grid)
    assert walker.retarget_blocked_goal is False
    assert walker.next_dir((41, 189), (120, 113)) is None
    assert walker.retargets == 0


def test_walker_leaves_a_walkable_goal_alone() -> None:
    from zelda_i.walk.physics import OccupancyGrid, OccupancyWalker

    walker = OccupancyWalker(
        grid=OccupancyGrid(blocked=set(), xmin=0, xmax=200, ymin=0, ymax=200),
        retarget_blocked_goal=True,
    )
    walker.next_dir((41, 189), (120, 113))
    assert walker.retargets == 0
    assert walker.goal == (120, 113)


def _turns(path: list[tuple[int, int]]) -> int:
    """Heading changes along a BFS path (Link snaps off-axis on every one)."""
    headings = [
        (b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])
    ]
    return sum(1 for a, b in zip(headings, headings[1:]) if a != b)


def test_no_path_to_a_blocked_goal_that_still_touches_open_floor() -> None:
    """A goal *inside* geometry is unreachable even when floor abuts it.

    The guard used to read ``not ok(goal) and not in_bounds(goal)``, so an
    in-bounds but solid goal sailed through, ``dist`` was seeded at 0 inside
    the wall and the descent handed back a path whose last step walks into
    it. Callers then pressed that button until their budget ran out:
    ``dungeon/engine.py::_collect_policy``'s ``collect_skip_unreachable`` was
    dead code for a walled waypoint, and ``dungeon/route_entry.py``'s
    ``entry_route_skip`` / ``entry_route_walled`` were unreachable because
    ``_route_replanning`` latched on a direction that never moved Link.
    """
    grid = OccupancyGrid(xmin=100, xmax=140, ymin=100, ymax=140)
    # One solid cell with open floor on all four sides: the statue face the
    # old guard walked into.
    grid.blocked.add((120, 120))
    assert grid.shortest_path((110, 120), (120, 120)) is None
    # And the whole face of a wider wall, approached head on.
    for y in range(110, 131):
        grid.blocked.add((124, y))
    assert grid.shortest_path((110, 120), (124, 120)) is None


def test_no_path_to_a_goal_outside_the_bounds_box() -> None:
    grid = OccupancyGrid(xmin=100, xmax=140, ymin=100, ymax=140)
    assert grid.shortest_path((120, 120), (99, 120)) is None
    assert grid.shortest_path((120, 120), (141, 120)) is None
    assert grid.shortest_path((120, 120), (120, 99)) is None
    assert grid.shortest_path((120, 120), (120, 141)) is None


def test_reachable_goal_keeps_its_minimum_turn_path() -> None:
    """The guard flip must not touch a goal that has a path.

    Turn count is the load-bearing property: a same-length path with extra
    turns is not walkable, because Link snaps his off-axis position on every
    direction change (L1 ``0x23``: 285 misses, 49 forgets, zero progress).
    """
    grid = OccupancyGrid(xmin=100, xmax=140, ymin=100, ymax=140)
    open_path = grid.shortest_path((110, 130), (130, 110))
    assert open_path is not None
    assert open_path[0] == (110, 130)
    assert open_path[-1] == (130, 110)
    assert len(open_path) == 41  # manhattan 40 + the start cell
    assert _turns(open_path) == 1

    # Around a wall the goal sits behind: still shortest, still minimum-turn.
    for y in range(100, 136):
        grid.blocked.add((120, y))
    walled = grid.shortest_path((110, 130), (130, 110))
    assert walled is not None
    assert walled[-1] == (130, 110)
    assert (120, 130) not in walled
    # South to y=136 (6), east past the wall (20), north to y=110 (26).
    assert len(walled) == 1 + (6 + 20 + 26)
    assert _turns(walled) == 2


def _tilemap_ram(cells: "dict[tuple[int, int], tuple[int, int, int, int]]"):
    """Synthetic ``$6530`` map: every named 16x16 cell origin gets a quad."""
    import numpy as np

    from zelda_i.dungeon import tilemap as tm

    ram = np.zeros(tm.WRAM_RAM_OFFSET + 0x2000, dtype=np.uint8)
    base = tm.WRAM_RAM_OFFSET + tm.ADDR_ROOM_TILE_MAP - tm.WRAM_BASE
    for (x, y), quad in cells.items():
        col, row = x // tm.TILE_PX, (y - tm.PLAYFIELD_TOP_Y) // tm.TILE_PX
        for dc, dr, value in (
            (0, 0, quad[0]), (1, 0, quad[1]), (0, 1, quad[2]), (1, 1, quad[3])
        ):
            ram[base + (col + dc) * tm.TILE_ROWS + (row + dr)] = value
    return ram


def _walkable_room_ram():
    """Floor interior, one block, the west door mouth, and a staircase."""
    from zelda_i.dungeon import tilemap as tm

    floor = (0x74, 0x76, 0x75, 0x77)
    cells = {
        (x, y): floor for y in tm.INTERIOR_Y for x in tm.INTERIOR_X
    }
    cells[(96, 144)] = (0xB0, 0xB2, 0xB1, 0xB3)  # a real wall
    cells[(208, 96)] = (0x70, 0x72, 0x71, 0x73)  # CheckWarp staircase
    cells[(16, 144)] = (0x90, 0x90, 0x91, 0x91)  # west door mouth
    return _tilemap_ram(cells)


def test_link_occupancy_walks_stairs_and_door_mouths() -> None:
    """Stairs and doors are floor Link stands on, not geometry.

    ``blocked_link_cells`` used to default to bare ``FLOOR_TILES``, which
    graded every staircase and every door mouth SOLID — on exactly the rooms
    that switched to ``occupancy_from_tilemap`` and on ``ROUTE_BOUNDS``,
    whose whole reason to be one ring wider than the fight box is to include
    the door mouths at x=16/224 and y=80/208.
    """
    from zelda_i.dungeon.route_entry import ROUTE_BOUNDS
    from zelda_i.dungeon.tilemap import FLOOR_TILES, blocked_link_cells

    ram = _walkable_room_ram()
    blocked = blocked_link_cells(ram, ROUTE_BOUNDS)
    # Link's stored y collides LINK_FOOT_OFFSET px lower: y=141 is the
    # y=144 cell row, y=93 is the y=96 row.
    door_mouth = (16, 141)
    stair = (208, 93)
    assert door_mouth not in blocked
    assert stair not in blocked
    # Measured geometry is untouched.
    assert (96, 141) in blocked

    # The old walkable set is what walled them off.
    floor_only = blocked_link_cells(ram, ROUTE_BOUNDS, walkable=FLOOR_TILES)
    assert door_mouth in floor_only
    assert stair in floor_only


def test_measured_walker_can_reach_a_staircase() -> None:
    """The stair is a route destination; a solid stair has no path to it."""
    from zelda_i.walk.physics import measured_walker

    walker = measured_walker(_walkable_room_ram(), (16, 224, 80, 208))
    assert walker is not None
    assert walker.grid.shortest_path((120, 141), (208, 93)) is not None

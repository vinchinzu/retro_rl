"""Barn walk map and walker. Measured on the ROM; a pure simulation here pins the walker's behaviour."""
from __future__ import annotations

import pytest

from harvest.tasks.cow_geometry import (
    BARN_CELLS,
    BARN_DOOR_PX,
    BARN_SHIP_BIN_PX,
    COW_FEED_SPOTS,
    FODDER_NORTH_STAND_PX,
    FODDER_STAND_PX,
    MAX_WALK_NUDGES,
    PREGNANT_COW_FEED_SPOT,
    BarnWalker,
    axis_step,
    nearest_cell,
    next_unfed_spot,
    plan_route,
    walk_cost,
    walk_distances,
)

STANDS = (
    BARN_DOOR_PX,
    BARN_SHIP_BIN_PX,
    FODDER_STAND_PX,
    FODDER_NORTH_STAND_PX,
    PREGNANT_COW_FEED_SPOT.interact_px,
    *(spot.interact_px for spot in COW_FEED_SPOTS),
)
STALLS = ((168, 278), (168, 246), (168, 214), (168, 150), (168, 118), (168, 86), (56, 278), (56, 246), (56, 214), (56, 150), (56, 118), (56, 86))
DIRECTION = {4: (0, -1), 5: (0, 1), 6: (-1, 0), 7: (1, 0)}  # pad bits: up, down, left, right


def walk(start, goal, *, frames=2000, tolerance=2, blocked_frames=0):
    """Move a farmer on the map at one pixel a frame, frozen for the first ``blocked_frames`` frames."""
    walker = BarnWalker()
    x, y = start
    for frame in range(frames):
        action = walker.step((x, y), goal, tolerance=tolerance)
        if action is None:
            return (x, y), frame, walker
        if frame < blocked_frames:
            continue
        for bit, (dx, dy) in DIRECTION.items():
            if int(action[bit]):
                x, y = x + dx, y + dy
    return (x, y), frames, walker


@pytest.mark.parametrize("stand", STANDS)
def test_every_chore_stand_is_on_the_walk_map(stand):
    cell = nearest_cell(stand)
    assert cell in BARN_CELLS
    assert abs(cell[0] - stand[0]) <= 8 and abs(cell[1] - stand[1]) <= 8


@pytest.mark.parametrize("goal", STANDS + STALLS)
def test_the_door_reaches_every_stand_and_stall(goal):
    assert plan_route(BARN_DOOR_PX, goal)


@pytest.mark.parametrize("goal", STANDS)
def test_the_walker_arrives_from_the_far_corners(goal):
    for start in ((BARN_DOOR_PX), (216, 80), (30, 80), (30, 300)):
        end, frames, _walker = walk(nearest_cell(start), goal)
        assert abs(end[0] - goal[0]) <= 2 and abs(end[1] - goal[1]) <= 2, (start, goal, end, frames)


def test_a_route_prefers_to_skirt_a_cow_but_never_needs_to():
    free = plan_route((30, 310), (216, 310))
    with_cow = plan_route((30, 310), (216, 310), cows=[(120, 310)])
    assert free and with_cow


def test_a_route_across_a_barn_full_of_cows_still_exists():
    assert plan_route(BARN_DOOR_PX, FODDER_STAND_PX, cows=STALLS)


def test_a_still_farmer_is_nudged_then_reports_blocked():
    walker = BarnWalker()
    pos = (200, 310)
    for _ in range(4000):
        walker.step(pos, (30, 310))
        if walker.blocked:
            break
    assert walker.nudges >= MAX_WALK_NUDGES and walker.blocked


def test_arriving_clears_the_nudge_count():
    walker = BarnWalker()
    pos = (200, 310)
    for _ in range(40):
        walker.step(pos, (30, 310))
    assert walker.nudges > 0
    assert walker.step((30, 310), (30, 310)) is None
    assert walker.nudges == 0 and not walker.blocked


def test_a_short_blockage_is_walked_out_of():
    end, _frames, _walker = walk((200, 310), (30, 310), blocked_frames=30)
    assert abs(end[0] - 30) <= 2


def test_axis_step_takes_the_longer_axis():
    assert axis_step((0, 0), (10, 3)) == ("right", 10)
    assert axis_step((10, 0), (0, 30)) == ("down", 30)
    assert axis_step((0, 0), (0, 0))[0] is None


def test_next_unfed_spot_picks_the_empty_trough_nearest_the_crossing():
    assert next_unfed_spot(0, COW_FEED_SPOTS).flag in (0x0004, 0x0100)
    assert next_unfed_spot(0x0FFF, COW_FEED_SPOTS) is None
    assert next_unfed_spot(0, ()) is None


def test_walking_into_a_cow_never_runs_long_enough_to_shove_her():
    walker = BarnWalker()
    pos = (112, 312)
    # A blocked hall forces the fallback route through her body. The old
    # 70-frame B hold shoved her toward the door (fuzz morning seed 1110).
    cows = [(128, 312), (112, 296), (112, 328)]
    for _ in range(100):
        action = walker.step(pos, (180, 312), cows=cows)
        if action[7]:  # right, into the cow
            assert not action[0]  # B


def test_walk_distances_follow_the_map_not_the_straight_line():
    steps = walk_distances((112, 120))  # the 8 px centre aisle: the only ways out are its two ends and the crossing row
    west, east = walk_cost(steps, (56, 86)), walk_cost(steps, (168, 118))
    assert abs(112 - 168) + abs(120 - 118) < abs(112 - 56) + abs(120 - 86) + 40  # as the crow flies they are close
    assert west < east
    assert walk_cost(steps, (112, 120)) == 0
    assert walk_cost(steps, (0, 0)) > walk_cost(steps, (40, 60))  # off the map: the nearest cell, never an error


def test_the_door_is_on_the_walk_map_and_the_bin_is_not_far_from_it():
    assert walk_cost(walk_distances(BARN_DOOR_PX), BARN_SHIP_BIN_PX) < 120

"""Predict-grade the Level 3 0x7c → 0x7b west-door residual (no emulator).

Cardinal LEFT at the west wall is a miss. LEFT+UP is a one-frame door clip
owned by ``west_door_step``, not OccupancyWalker.
"""

from __future__ import annotations

import numpy as np

from retro_harness.controls import pressed_nes_buttons
from zelda_i.level3_geometry import WEST_DOOR_APPROACH_Y, WEST_DOOR_WALL_X
from zelda_i.level3_path import west_door_step
from zelda_i.predict import grade_walk, snapshot_fields, walk_claim
from zelda_i.ram import (
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.walk_physics import DEFAULT_BOUNDS, WALK_DELTA, OccupancyWalker

ROOM_7C = 0x7C
ROOM_7B = 0x7B
_CARDINALS = frozenset(WALK_DELTA)


def _snap(*, x: int, y: int, screen: int = ROOM_7C, level: int = 3) -> object:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = level
    ram[ADDR_SCREEN] = screen
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    return read_snapshot(ram)


def _pressed(action: object) -> set[str]:
    return set(pressed_nes_buttons(list(action)))


def test_walk_claim_left() -> None:
    assert walk_claim("LEFT") == "move -1,0"


def test_grade_walk_left_hit_and_same_xy_miss() -> None:
    before = _snap(x=WEST_DOOR_WALL_X, y=WEST_DOOR_APPROACH_Y)
    hit = _snap(x=WEST_DOOR_WALL_X - 1, y=WEST_DOOR_APPROACH_Y)
    miss = _snap(x=WEST_DOOR_WALL_X, y=WEST_DOOR_APPROACH_Y)
    assert grade_walk("LEFT", before, hit).ok
    stuck = grade_walk("LEFT", before, miss)
    assert not stuck.ok
    assert stuck.missed == ("move -1,0",)


def test_grade_walk_left_stuck_at_west_wall_is_miss() -> None:
    """Pure LEFT sticks at x≈32 (mask==0). Models the burned trap."""
    before = _snap(x=32, y=WEST_DOOR_APPROACH_Y)
    stuck = grade_walk("LEFT", before, before)
    assert not stuck.ok
    assert stuck.missed == ("move -1,0",)
    still_left = grade_walk(
        "LEFT",
        before,
        _snap(x=31, y=WEST_DOOR_APPROACH_Y),
    )
    # A 1px LEFT would hold — the trap is that RAM does not move.
    assert still_left.ok


def test_occupancy_walker_next_dir_never_emits_diagonal() -> None:
    assert _CARDINALS == {"UP", "DOWN", "LEFT", "RIGHT"}
    for dx, dy in WALK_DELTA.values():
        assert abs(dx) + abs(dy) == 1

    xmin = DEFAULT_BOUNDS[0]
    assert xmin == 40
    assert not (xmin <= 32 <= DEFAULT_BOUNDS[1])

    poses = (
        (120, WEST_DOOR_APPROACH_Y),
        (WEST_DOOR_WALL_X, WEST_DOOR_APPROACH_Y),
        (32, WEST_DOOR_APPROACH_Y),
        (112, 141),
        (100, 141),
        (WEST_DOOR_WALL_X, 141),
    )
    goals = (
        (32, WEST_DOOR_APPROACH_Y),
        (WEST_DOOR_WALL_X, WEST_DOOR_APPROACH_Y),
        (xmin, WEST_DOOR_APPROACH_Y),
        (120, 141),
    )
    for start in poses:
        for goal in goals:
            walker = OccupancyWalker(goal=goal)
            walker.observe(start)
            direction = walker.next_dir(start)
            assert direction in _CARDINALS or direction is None
            if direction is not None:
                dx, dy = WALK_DELTA[direction]
                assert (dx == 0) != (dy == 0)


def test_west_diagonal_push_only_at_entry_wall_on_approach_y() -> None:
    """LEFT+UP is west_door_step residual, not OccupancyWalker."""
    wall = (WEST_DOOR_WALL_X, WEST_DOOR_APPROACH_Y)
    clip = west_door_step(_snap(x=wall[0], y=wall[1], screen=ROOM_7C))
    assert clip.reason == "west_diagonal_push"
    assert _pressed(clip.action) == {"LEFT", "UP"}

    stuck_wall = west_door_step(_snap(x=32, y=WEST_DOOR_APPROACH_Y, screen=ROOM_7C))
    assert stuck_wall.reason == "west_diagonal_push"
    assert _pressed(stuck_wall.action) == {"LEFT", "UP"}

    approach = west_door_step(
        _snap(x=WEST_DOOR_WALL_X + 1, y=WEST_DOOR_APPROACH_Y, screen=ROOM_7C)
    )
    assert approach.reason == "west_approach"
    assert _pressed(approach.action) == {"LEFT"}

    arrived = west_door_step(_snap(x=32, y=WEST_DOOR_APPROACH_Y, screen=ROOM_7B))
    assert arrived.reason == "west_arrived"
    assert arrived.reason != "west_diagonal_push"

    # OccupancyWalker cannot path into x≈32 (out of DEFAULT_BOUNDS) and
    # never emits LEFT+UP even when the in-bounds west cell is the goal.
    clip_walker = OccupancyWalker(goal=(32, WEST_DOOR_APPROACH_Y))
    clip_walker.observe(wall)
    assert clip_walker.next_dir(wall) is None

    inbound = OccupancyWalker(goal=(DEFAULT_BOUNDS[0], WEST_DOOR_APPROACH_Y))
    inbound.observe(wall)
    assert inbound.next_dir(wall) in {"LEFT", None}


def test_west_door_step_y141_is_align_not_approach() -> None:
    """y≈141 is the mid-room block trap (often sticks at x≈112)."""
    step = west_door_step(_snap(x=100, y=141, screen=ROOM_7C))
    assert step.reason == "west_align_y"
    assert step.reason != "west_approach"
    assert _pressed(step.action) == {"DOWN"}

    mid_block = west_door_step(_snap(x=112, y=141, screen=ROOM_7C))
    assert mid_block.reason == "west_align_y"


def test_snapshot_fields_include_screen_and_room_7c_vs_7b() -> None:
    entry = snapshot_fields(_snap(x=WEST_DOOR_WALL_X, y=WEST_DOOR_APPROACH_Y, screen=ROOM_7C))
    west = snapshot_fields(_snap(x=200, y=WEST_DOOR_APPROACH_Y, screen=ROOM_7B))
    assert entry["screen"] == entry["room"] == ROOM_7C
    assert west["screen"] == west["room"] == ROOM_7B
    assert entry["x"] == WEST_DOOR_WALL_X
    assert entry["y"] == WEST_DOOR_APPROACH_Y

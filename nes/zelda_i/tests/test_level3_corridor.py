"""Compass-west Raft corridor geometry (no emulator).

Locks KEY-LEFT y=141, 0x69 stairs y≈141, 0x5b west wall x≈26, bomb stands,
and OccupancyGrid vs path-policy push. OccupancyWalker is 1px cardinal;
key-door long push is not a walk_claim.
"""

from __future__ import annotations

import numpy as np

from zelda_i.level3_geometry import (
    BOMB_STAND_59_RIGHT,
    BOMB_STAND_5B_RIGHT,
    KEY_DOOR_Y,
    KEY_DOOR_Y_TOL,
    STAIRS_69_RIGHT_Y,
    WEST_DOOR_APPROACH_Y,
    WEST_DOOR_WALL_X,
    WEST_WALL_5B_X,
)
from zelda_i.level3_raft_path import _align_then_push
from zelda_i.ram import (
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.walk_physics import (
    DEFAULT_BOUNDS,
    WALK_SPEED,
    OccupancyGrid,
    predicted_xy,
)


def _snap(*, x: int, y: int, room: int = 0x5B):
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 3
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    return read_snapshot(ram)


def test_corridor_y_and_west_wall_constants() -> None:
    assert KEY_DOOR_Y == 141
    assert KEY_DOOR_Y_TOL == 3
    assert STAIRS_69_RIGHT_Y == 141
    assert WEST_WALL_5B_X == 26
    assert WEST_WALL_5B_X != 32
    assert WEST_DOOR_WALL_X == 48  # push plane; wall residual is x≈26
    assert abs(WEST_DOOR_APPROACH_Y - KEY_DOOR_Y) > KEY_DOOR_Y_TOL


def test_bomb_stands_remain_192_141() -> None:
    assert BOMB_STAND_5B_RIGHT == (192, 141)
    assert BOMB_STAND_59_RIGHT == (192, 141)


def test_occupancy_wrong_y_does_not_block_key_door_band() -> None:
    """A miss at y=133 can block that cell without sealing y=141."""
    grid = OccupancyGrid()
    plane_x = WEST_DOOR_WALL_X
    wrong = (plane_x, 133)
    band = (plane_x, KEY_DOOR_Y)
    assert grid.in_bounds(*wrong)
    assert grid.in_bounds(*band)
    grid.blocked.add(wrong)
    assert not grid.passable(*wrong)
    assert grid.passable(*band)
    path = grid.shortest_path((120, KEY_DOOR_Y), band)
    assert path is not None
    assert path[-1] == band
    assert wrong not in path
    assert all(y == KEY_DOOR_Y for _, y in path)


def test_occupancy_does_not_model_5b_west_wall_clip() -> None:
    """xmin=40: wall x≈26 is out of grid; 1px walker is not a long-push policy."""
    assert DEFAULT_BOUNDS[0] == 40
    assert WEST_WALL_5B_X < DEFAULT_BOUNDS[0]
    grid = OccupancyGrid()
    assert not grid.in_bounds(WEST_WALL_5B_X, KEY_DOOR_Y)
    assert WALK_SPEED == 1
    assert predicted_xy(WEST_DOOR_WALL_X, KEY_DOOR_Y, "LEFT") == (
        WEST_DOOR_WALL_X - 1,
        KEY_DOOR_Y,
    )


def test_align_then_push_y141_vs_y149() -> None:
    """One-frame helper: align y first; at y=141 hold LEFT, do not snap to x=32."""
    # Wrong y (entry west band 149): vertical align, not a key push.
    align = _align_then_push(
        _snap(x=WEST_WALL_5B_X, y=WEST_DOOR_APPROACH_Y),
        target_x=32,
        target_y=KEY_DOOR_Y,
        push_dir="LEFT",
        reason_prefix="left_5a",
        door_plane=WEST_DOOR_WALL_X,
    )
    assert align.reason == "left_5a_align_y"

    # Mid-room on the door band: approach the plane.
    approach = _align_then_push(
        _snap(x=120, y=KEY_DOOR_Y),
        target_x=32,
        target_y=KEY_DOOR_Y,
        push_dir="LEFT",
        reason_prefix="left_5a",
        door_plane=WEST_DOOR_WALL_X,
    )
    assert approach.reason == "left_5a_approach"

    # At the 0x5b west wall (x≈26): push, do not snap back to target_x=32.
    push = _align_then_push(
        _snap(x=WEST_WALL_5B_X, y=KEY_DOOR_Y),
        target_x=32,
        target_y=KEY_DOOR_Y,
        push_dir="LEFT",
        reason_prefix="left_5a",
        door_plane=WEST_DOOR_WALL_X,
    )
    assert push.reason == "left_5a_push_LEFT"

    # Stairs 0x69 RIGHT @ y≈141 vs y=149.
    stairs_align = _align_then_push(
        _snap(x=200, y=WEST_DOOR_APPROACH_Y, room=0x69),
        target_x=208,
        target_y=STAIRS_69_RIGHT_Y,
        push_dir="RIGHT",
        reason_prefix="stairs",
        door_plane=192,
    )
    assert stairs_align.reason == "stairs_align_y"
    stairs_push = _align_then_push(
        _snap(x=200, y=STAIRS_69_RIGHT_Y, room=0x69),
        target_x=208,
        target_y=STAIRS_69_RIGHT_Y,
        push_dir="RIGHT",
        reason_prefix="stairs",
        door_plane=192,
    )
    assert stairs_push.reason == "stairs_push_RIGHT"
    # Without door_plane, x=26 would snap back toward target_x=32.
    snap_back = _align_then_push(
        _snap(x=WEST_WALL_5B_X, y=KEY_DOOR_Y),
        target_x=32,
        target_y=KEY_DOOR_Y,
        push_dir="LEFT",
        reason_prefix="left_5a",
    )
    assert snap_back.reason == "left_5a_align_x"

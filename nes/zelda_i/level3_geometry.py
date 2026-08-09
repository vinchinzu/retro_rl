"""Level 3 (Manji) room / path geometry — integers/tuples only.

Single source for bomb stands, door bands, and raft-passage geometry.
Imported by ``level3_dungeon`` (re-exports public names for DoorRoutes),
``level3_path`` / ``level3_raft_path`` (controllers), and
``door_graph.level3_exits`` (must not import ``level3_dungeon`` — cycle).
"""

from __future__ import annotations

# Bomb stands (LIVE recon)
BOMB_STAND_5B_RIGHT: tuple[int, int] = (192, 141)  # 0x5b → 0x5c boss shortcut
BOMB_STAND_59_RIGHT: tuple[int, int] = (192, 141)  # post-Raft: walk-RIGHT sealed

# Raft passage EXIT (reverse of pickup; Level3Raft → 0x69)
PASSAGE_EXIT_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (176, 141),
    (176, 189),
    (48, 189),
    (48, 77),
)

DOOR_5C_RIGHT_Y: int = 141  # only y≈141 opens 0x5c → 0x5d after Darknut clear
STAIRS_69_RIGHT_Y: int = 141  # only y≈141 opens 0x69 → 0x0f
KEY_DOOR_Y: int = 141  # 0x5a LEFT KEY: long push @ y≈141
KEY_DOOR_Y_TOL: int = 3

# West door residual: pure LEFT sticks at x≈32 (mask==0). LEFT+UP at the west
# wall corner-clips into the scroll (mode 6/7 → room 0x7b). Approach band y≈149
# reaches the wall; y≈141 alone often blocks mid-room at x≈112.
WEST_DOOR_APPROACH_Y: int = 149
WEST_DOOR_WALL_X: int = 48

# North door residual from 0x7b: UP only works with |x-120|≤4. Threshold 8
# leaves Link at x≈112 and sticks on the north wall (live probe 2026-08-06).
NORTH_DOOR_X: int = 120
NORTH_DOOR_X_TOL: int = 4

# Raft passage geometry (mode 9): enter from 0x69 RIGHT @ y≈141 → spawn ~(48,77).
# Path: DOWN to y≈189 → RIGHT to x≈176 → UP channel to y≈141 → LEFT to x≈136 touch Raft.
RAFT_PASSAGE_MODE: int = 9
RAFT_CHANNEL_X: int = 176
RAFT_CHANNEL_X_TOL: int = 4
RAFT_PICKUP_X: int = 136
RAFT_PICKUP_Y: int = 141
RAFT_SOUTH_Y: int = 189
RAFT_SOUTH_Y_TOL: int = 6

__all__ = [
    "BOMB_STAND_59_RIGHT",
    "BOMB_STAND_5B_RIGHT",
    "DOOR_5C_RIGHT_Y",
    "KEY_DOOR_Y",
    "KEY_DOOR_Y_TOL",
    "NORTH_DOOR_X",
    "NORTH_DOOR_X_TOL",
    "PASSAGE_EXIT_WAYPOINTS",
    "RAFT_CHANNEL_X",
    "RAFT_CHANNEL_X_TOL",
    "RAFT_PASSAGE_MODE",
    "RAFT_PICKUP_X",
    "RAFT_PICKUP_Y",
    "RAFT_SOUTH_Y",
    "RAFT_SOUTH_Y_TOL",
    "STAIRS_69_RIGHT_Y",
    "WEST_DOOR_APPROACH_Y",
    "WEST_DOOR_WALL_X",
]

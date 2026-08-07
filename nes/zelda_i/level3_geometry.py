"""Level 3 (Manji) room geometry — integers/tuples only.

Single source for bomb stands, door Y bands, and raft-passage exit
waypoints. Imported by ``level3_dungeon`` (re-exports public names) and
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

__all__ = [
    "BOMB_STAND_59_RIGHT",
    "BOMB_STAND_5B_RIGHT",
    "DOOR_5C_RIGHT_Y",
    "KEY_DOOR_Y",
    "KEY_DOOR_Y_TOL",
    "PASSAGE_EXIT_WAYPOINTS",
    "STAIRS_69_RIGHT_Y",
]

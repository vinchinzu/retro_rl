"""Unit tests for Level 5 east-key path policy (no emulator)."""

from __future__ import annotations

import numpy as np

from zelda_i.level5_dungeon import (
    LEVEL_5,
    ROOM_L5_ENTRY,
    ROOM_L5_GIBDO_66,
)
from zelda_i.level5_path import (
    EAST_DOOR_APPROACH_Y,
    EAST_DOOR_CHANNEL_Y,
    EAST_DOOR_WALL_X,
    level5_east_key_step,
    should_force_keys_zero,
)
from zelda_i.ram import (
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)


def _ram(
    *,
    level: int = LEVEL_5,
    room: int = ROOM_L5_ENTRY,
    x: int = 120,
    y: int = 205,
    mode: int = PLAY_MODE,
    keys: int = 0,
) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_LEVEL] = level
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_KEYS] = keys
    return ram


def test_east_door_geometry() -> None:
    assert EAST_DOOR_APPROACH_Y == 157
    assert EAST_DOOR_CHANNEL_Y == 141
    assert EAST_DOOR_WALL_X == 200


def test_east_key_route_returns_south_from_cleared_66() -> None:
    snap = read_snapshot(_ram(room=ROOM_L5_GIBDO_66, x=56, y=117, keys=1))
    action = level5_east_key_step(snap)
    assert action.reason == "east_key_finish_ladder"
    off_ladder = level5_east_key_step(
        read_snapshot(_ram(room=ROOM_L5_GIBDO_66, x=56, y=149, keys=1))
    )
    assert off_ladder.reason == "east_key_align_south_x"


def test_east_key_route_uses_wall_before_door_channel() -> None:
    approach = level5_east_key_step(
        read_snapshot(_ram(room=ROOM_L5_ENTRY, x=180, y=157, keys=1))
    )
    channel = level5_east_key_step(
        read_snapshot(_ram(room=ROOM_L5_ENTRY, x=200, y=157, keys=1))
    )
    assert approach.reason == "east_key_approach_wall"
    assert channel.reason == "east_key_align_channel_y"


def test_should_force_keys_zero() -> None:
    assert should_force_keys_zero("L5_Room_77") is True
    assert should_force_keys_zero("Level5Cleared66") is False
    assert should_force_keys_zero("Level5EntranceFromL4") is False
    assert should_force_keys_zero("L5_Room_77", keep_keys=True) is False
    assert should_force_keys_zero("Level5Cleared66", force_keys_zero=True) is True

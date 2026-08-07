"""Unit tests for Level 3 overworld hop tables and stop predicates."""

from __future__ import annotations

import numpy as np

from zelda_i.level3_overworld import (
    LEVEL3,
    LEVEL3_DOOR_HOPS_FROM_66,
    LEVEL3_PATH_HOPS,
    LEVEL3_PATH_SCREENS,
    LEVEL3_SOURCE_PATH_SCREENS,
    SCREEN_LEVEL3_ENTRANCE,
    SCREEN_LEVEL3_ENTRY_ROOM,
    OverworldToLevel3Controller,
    level3_entrance_success,
    level3_path_success,
)
from zelda_i.overworld import neighbor_screens
from zelda_i.ram import (
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_SWORD,
    PLAY_MODE,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 0)
    ram[ADDR_SCREEN] = fields.get("screen", SCREEN_LEVEL3_ENTRANCE)
    ram[ADDR_LINK_X] = fields.get("x", 128)
    ram[ADDR_LINK_Y] = fields.get("y", 140)
    ram[ADDR_SWORD] = fields.get("sword", 1)
    return ram


def test_level3_path_screens_chain() -> None:
    assert LEVEL3_PATH_SCREENS[0] == 0x77
    assert LEVEL3_PATH_SCREENS[-1] == SCREEN_LEVEL3_ENTRANCE == 0x74
    assert len(LEVEL3_PATH_HOPS) == len(LEVEL3_PATH_SCREENS) - 1
    for a, b in zip(LEVEL3_PATH_SCREENS, LEVEL3_PATH_SCREENS[1:]):
        assert b in neighbor_screens(a).values(), f"{a:02x}->{b:02x}"


def test_level3_door_hops_from_66_neighbors() -> None:
    screens = (0x66,) + tuple(h.target for h in LEVEL3_DOOR_HOPS_FROM_66)
    assert screens[-1] == 0x74
    for a, b in zip(screens, screens[1:]):
        assert b in neighbor_screens(a).values()


def test_source_path_documented_but_not_default() -> None:
    """Source arithmetic ends on 0x74 but is not the controller default hops."""
    assert LEVEL3_SOURCE_PATH_SCREENS[-1] == 0x74
    assert LEVEL3_SOURCE_PATH_SCREENS[1] == 0x67
    assert LEVEL3_PATH_SCREENS[1] != 0x67


def test_level3_path_success() -> None:
    assert level3_path_success(_ram(screen=0x74, sword=1))
    assert not level3_path_success(_ram(screen=0x74, sword=0))
    assert not level3_path_success(_ram(screen=0x73, sword=1))


def test_level3_entrance_success() -> None:
    assert level3_entrance_success(
        _ram(level=LEVEL3, screen=SCREEN_LEVEL3_ENTRY_ROOM, mode=PLAY_MODE)
    )
    assert not level3_entrance_success(_ram(level=0, screen=0x74))
    assert not level3_entrance_success(_ram(level=LEVEL3, screen=0x7d))


def test_controller_defaults() -> None:
    nav = OverworldToLevel3Controller()
    assert nav.hops[-1].target == 0x74
    assert nav.entry_room == 0x7C
    assert nav.door_x == 128

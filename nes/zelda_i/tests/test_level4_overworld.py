"""Unit tests for Level 4 overworld hop tables and stop predicates."""

from __future__ import annotations

import numpy as np

from zelda_i.level4_overworld import (
    LEVEL4,
    LEVEL4_DOCK_SCREEN,
    LEVEL4_ENTRY_ROOM,
    LEVEL4_HOPS_FROM_POST_L3,
    LEVEL4_ISLAND_SCREEN,
    LEVEL4_POST_L3_SCREENS,
    SCREEN_POST_L3_RETURN,
    SOURCE_HYPOTHESIS,
    OverworldToLevel4Controller,
    level4_entrance_success,
    level4_entry_stop,
    level4_overworld_stop,
    on_level4_dock,
    planning_report,
    post_l3_overworld_ready,
)
from zelda_i.overworld import neighbor_screens
from zelda_i.ram import (
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_RAFT,
    ADDR_SCREEN,
    ADDR_SWORD,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 0)
    ram[ADDR_SCREEN] = fields.get("screen", LEVEL4_ISLAND_SCREEN)
    ram[ADDR_LINK_X] = fields.get("x", 128)
    ram[ADDR_LINK_Y] = fields.get("y", 140)
    ram[ADDR_SWORD] = fields.get("sword", 1)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x07)
    ram[ADDR_RAFT] = fields.get("raft", 1)
    return ram


def test_post_l3_path_screens_chain() -> None:
    assert LEVEL4_POST_L3_SCREENS[0] == SCREEN_POST_L3_RETURN == 0x74
    assert LEVEL4_POST_L3_SCREENS[-1] == LEVEL4_ISLAND_SCREEN == 0x45
    assert LEVEL4_DOCK_SCREEN == 0x55
    assert LEVEL4_DOCK_SCREEN in LEVEL4_POST_L3_SCREENS
    assert len(LEVEL4_HOPS_FROM_POST_L3) == len(LEVEL4_POST_L3_SCREENS) - 1
    for a, b in zip(LEVEL4_POST_L3_SCREENS, LEVEL4_POST_L3_SCREENS[1:]):
        assert b in neighbor_screens(a).values(), f"{a:02x}->{b:02x}"


def test_live_geometry_not_hypothesis() -> None:
    assert SOURCE_HYPOTHESIS is False
    report = planning_report()
    assert report["live"]["verified"] is True
    assert report["live"]["entry_room"] == hex(LEVEL4_ENTRY_ROOM)
    assert LEVEL4_ENTRY_ROOM == 0x71


def test_east_exit_band_on_63() -> None:
    hop = next(h for h in LEVEL4_HOPS_FROM_POST_L3 if h.target == 0x64)
    assert hop.direction == "RIGHT"
    assert hop.y_band_lo == 145
    assert hop.y_band_hi == 155


def test_level4_entry_stop() -> None:
    snap = read_snapshot(
        _ram(level=LEVEL4, screen=LEVEL4_ENTRY_ROOM, mode=PLAY_MODE)
    )
    assert level4_entry_stop(snap)
    assert level4_entrance_success(
        _ram(level=LEVEL4, screen=LEVEL4_ENTRY_ROOM, mode=PLAY_MODE)
    )
    assert not level4_entrance_success(_ram(level=0, screen=0x45))
    assert not level4_entrance_success(
        _ram(level=LEVEL4, screen=0x70, mode=PLAY_MODE)
    )


def test_dock_and_island_stops() -> None:
    dock = read_snapshot(_ram(screen=0x55, y=140))
    assert on_level4_dock(dock)
    island = read_snapshot(_ram(screen=0x45, y=140))
    assert level4_overworld_stop(island)
    assert not level4_overworld_stop(dock)


def test_post_l3_ready() -> None:
    assert post_l3_overworld_ready(
        _ram(screen=0x74, triforce=0x07, raft=1, sword=1)
    )
    assert not post_l3_overworld_ready(
        _ram(screen=0x74, triforce=0x07, raft=0, sword=1)
    )
    assert not post_l3_overworld_ready(
        _ram(screen=0x73, triforce=0x07, raft=1, sword=1)
    )


def test_controller_defaults() -> None:
    nav = OverworldToLevel4Controller()
    assert nav.hops[-1].target == 0x45
    assert nav.entry_room == 0x71
    assert nav.door_x == 128
    assert nav.entry_level == LEVEL4
    assert nav.require_dungeon is True

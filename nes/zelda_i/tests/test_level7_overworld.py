"""Unit tests for the Level 7 pond approach and capability gates."""

from __future__ import annotations

import numpy as np

from zelda_i.level7_overworld import (
    LEVEL7_POND_HOPS,
    LEVEL7_POND_APPROACH_HOPS,
    LEVEL7_POND_SCREENS,
    SCREEN_LEVEL7_BAIT_SHOP_HYP,
    SCREEN_LEVEL7_POND_HYP,
    OverworldToLevel7PondController,
    has_food,
    has_whistle,
    level7_overworld_stop,
    missing_entry_caps,
)
from zelda_i.overworld import neighbor_screens
from zelda_i.ram import (
    ADDR_FOOD,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_SWORD,
    ADDR_WHISTLE,
    PLAY_MODE,
    SCREEN_START,
    read_snapshot,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 0)
    ram[ADDR_SCREEN] = fields.get("screen", SCREEN_LEVEL7_POND_HYP)
    ram[ADDR_LINK_X] = fields.get("x", 112)
    ram[ADDR_LINK_Y] = fields.get("y", 141)
    ram[ADDR_SWORD] = fields.get("sword", 1)
    ram[ADDR_WHISTLE] = fields.get("whistle", 0)
    ram[ADDR_FOOD] = fields.get("food", 0)
    return ram


def test_level7_pond_path_is_contiguous_and_skips_shop_detour() -> None:
    assert LEVEL7_POND_HOPS[: len(LEVEL7_POND_APPROACH_HOPS)] == (
        LEVEL7_POND_APPROACH_HOPS
    )
    assert LEVEL7_POND_SCREENS[0] == SCREEN_START == 0x77
    assert LEVEL7_POND_SCREENS[-1] == SCREEN_LEVEL7_POND_HYP == 0x42
    assert SCREEN_LEVEL7_BAIT_SHOP_HYP == 0x34
    assert LEVEL7_POND_SCREENS[1] == 0x78  # 0x67 north pocket is sealed live.
    assert 0x67 not in LEVEL7_POND_SCREENS
    assert 0x34 not in LEVEL7_POND_SCREENS
    assert 0x44 not in LEVEL7_POND_SCREENS
    assert [h.target for h in LEVEL7_POND_HOPS[-3:]] == [0x53, 0x52, 0x42]
    hop_52 = next(h for h in LEVEL7_POND_HOPS if h.target == 0x52)
    assert hop_52.direction == "LEFT"
    assert hop_52.align_y == 189
    assert len(LEVEL7_POND_HOPS) == len(LEVEL7_POND_SCREENS) - 1
    for before, after in zip(LEVEL7_POND_SCREENS, LEVEL7_POND_SCREENS[1:]):
        assert after in neighbor_screens(before).values(), f"{before:02x}->{after:02x}"


def test_level7_controller_stops_at_pond_without_claiming_entry() -> None:
    nav = OverworldToLevel7PondController()
    assert nav.end_screen() == 0x42
    assert nav.require_dungeon is False
    assert nav.require_sword is True


def test_level7_64_north_escape_uses_middle_band() -> None:
    nav = OverworldToLevel7PondController()
    hop = next(h for h in nav.hops if h.target == 0x54)

    east_ledge = nav._extra_hop_action(
        read_snapshot(_ram(screen=0x64, x=232, y=109)), hop
    )
    assert east_ledge is not None
    assert east_ledge.reason.startswith("64_east_ledge_down")

    cross = nav._extra_hop_action(read_snapshot(_ram(screen=0x64, x=232, y=141)), hop)
    assert cross is not None
    assert cross.reason.startswith("64_cross_to_north")

    north = nav._extra_hop_action(read_snapshot(_ram(screen=0x64, x=48, y=141)), hop)
    assert north is not None
    assert north.reason.startswith("64_north")


def test_level7_pond_stop_is_exact() -> None:
    assert level7_overworld_stop(read_snapshot(_ram()))
    assert not level7_overworld_stop(read_snapshot(_ram(screen=0x52)))
    assert not level7_overworld_stop(read_snapshot(_ram(level=7)))
    assert not level7_overworld_stop(read_snapshot(_ram(mode=16)))


def test_level7_capability_gates_are_independent() -> None:
    empty = _ram()
    assert not has_whistle(empty)
    assert not has_food(empty)
    assert missing_entry_caps(empty) == ["whistle"]

    whistle_only = _ram(whistle=1)
    assert has_whistle(whistle_only)
    assert not has_food(whistle_only)
    assert missing_entry_caps(whistle_only) == []

"""Unit tests for Level 4 overworld hop tables and stop predicates."""

from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from zelda_i.level4.spine import l4_hops, rupees_51_stages, rupees_71_stages
from zelda_i.level4.overworld import (
    LEVEL4,
    LEVEL4_DOCK_SCREEN,
    LEVEL4_ENTRY_ROOM,
    LEVEL4_HOPS_FROM_POST_L3,
    LEVEL4_ISLAND_SCREEN,
    LEVEL4_POST_L3_SCREENS,
    RUPEES_51_BACK_HOPS,
    RUPEES_51_HOPS,
    SCREEN_POST_L3_RETURN,
    level4_entrance_success,
    level4_entry_stop,
)
from zelda_i.overworld.graph import neighbor_screens
from zelda_i.ram import ADDR_WORLD_FLAGS, PLAY_MODE, WORLD_FLAG_ITEM, read_snapshot
from zelda_i.tests.ram_helpers import make_ram

_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 0,
    "screen": LEVEL4_ISLAND_SCREEN,
    "x": 128,
    "y": 140,
    "sword": 1,
    "triforce": 0x07,
    "raft": 1,
}


def _ram(**fields: int) -> np.ndarray:
    return make_ram(_DEFAULTS, **fields)


def test_post_l3_path_screens_chain() -> None:
    assert LEVEL4_POST_L3_SCREENS[0] == SCREEN_POST_L3_RETURN == 0x74
    assert LEVEL4_POST_L3_SCREENS[-1] == LEVEL4_ISLAND_SCREEN == 0x45
    assert LEVEL4_DOCK_SCREEN == 0x55
    assert LEVEL4_DOCK_SCREEN in LEVEL4_POST_L3_SCREENS
    assert len(LEVEL4_HOPS_FROM_POST_L3) == len(LEVEL4_POST_L3_SCREENS) - 1
    for a, b in zip(LEVEL4_POST_L3_SCREENS, LEVEL4_POST_L3_SCREENS[1:]):
        assert b in neighbor_screens(a).values(), f"{a:02x}->{b:02x}"


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


@pytest.mark.parametrize(
    ("bombs", "rupees", "taken"),
    [(0, 30, False), (1, 30, True), (1, 230, False)],
)
def test_rupees_71_stages_skip_without_cave_pay(
    bombs: int, rupees: int, taken: bool
) -> None:
    """A skipped detour lets all five spine stages finish on the first frame."""
    ram = _ram(screen=0x74, bombs=bombs, rupees=rupees)
    if taken:
        ram[ADDR_WORLD_FLAGS + 0x71] = WORLD_FLAG_ITEM
    env = SimpleNamespace(get_ram=lambda: ram)
    snap = read_snapshot(ram)
    stages = rupees_71_stages()
    assert [name for name, _, _ in stages] == [
        "walk_71", "select_bombs_71", "rupees_71", "exit_cave_71", "return_73"
    ]
    for _, ctl, _ in stages:
        if hasattr(ctl, "bind_env"):
            ctl.bind_env(env)
        ctl.step(snap)
        assert ctl.success
        assert ctl.frames == 1


def test_rupees_51_detour_rejoins_shop_walk_and_precedes_purchases() -> None:
    route = (0x73, *(hop.target for hop in RUPEES_51_HOPS))
    assert route == (0x73, 0x63, 0x53, 0x52, 0x51)
    assert tuple(hop.target for hop in RUPEES_51_BACK_HOPS) == (0x52, 0x53, 0x63)
    for a, b in zip(route, route[1:]):
        assert b in neighbor_screens(a).values()
    entry = l4_hops(spine_fields=lambda snap: {})[0]
    names = [name for name, _, _ in entry.stages]
    assert names.index("return_63") < names.index("potion_restock_l3")
    assert names.index("return_63") < names.index("bomb_restock_l3")


@pytest.mark.parametrize(
    ("candle", "rupees", "taken"),
    [(0, 50, False), (1, 50, True), (1, 250, False)],
)
def test_rupees_51_stages_skip_without_cave_pay(
    candle: int, rupees: int, taken: bool
) -> None:
    ram = _ram(screen=0x73, candle=candle, rupees=rupees)
    if taken:
        ram[ADDR_WORLD_FLAGS + 0x51] = WORLD_FLAG_ITEM
    env = SimpleNamespace(get_ram=lambda: ram)
    snap = read_snapshot(ram)
    stages = rupees_51_stages()
    assert [name for name, _, _ in stages] == [
        "walk_51", "select_candle_51", "rupees_51", "exit_cave_51", "return_63"
    ]
    for _, ctl, _ in stages:
        if hasattr(ctl, "bind_env"):
            ctl.bind_env(env)
        ctl.step(snap)
        assert ctl.success
        assert ctl.frames == 1

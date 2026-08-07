"""Unit tests for Level 8 overworld hop tables (bush + Blue Candle shop)."""

from __future__ import annotations

import numpy as np

from zelda_i.level8_overworld import (
    CANDLE_BUY_X,
    CANDLE_BUY_Y,
    CANDLE_SHOP_CAVE_X,
    CANDLE_SHOP_HOPS,
    CANDLE_SHOP_PRICE,
    CANDLE_SHOP_SCREENS,
    LEVEL8_BUSH_HOPS,
    LEVEL8_BUSH_SCREENS,
    SCREEN_CANDLE_SHOP,
    SCREEN_LEVEL8_BUSH,
    CandleShopNavPhase,
    OverworldToCandleShopController,
    OverworldToLevel8Controller,
    candle_shop_cave_entered,
    candle_shop_screen_reached,
    level8_bush_screen_reached,
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
    SCREEN_START,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 0)
    ram[ADDR_SCREEN] = fields.get("screen", SCREEN_LEVEL8_BUSH)
    ram[ADDR_LINK_X] = fields.get("x", 48)
    ram[ADDR_LINK_Y] = fields.get("y", 93)
    ram[ADDR_SWORD] = fields.get("sword", 1)
    return ram


def test_level8_bush_path_screens_chain() -> None:
    assert LEVEL8_BUSH_SCREENS[0] == SCREEN_START == 0x77
    assert LEVEL8_BUSH_SCREENS[-1] == SCREEN_LEVEL8_BUSH == 0x6D
    assert len(LEVEL8_BUSH_HOPS) == len(LEVEL8_BUSH_SCREENS) - 1
    for a, b in zip(LEVEL8_BUSH_SCREENS, LEVEL8_BUSH_SCREENS[1:]):
        assert b in neighbor_screens(a).values(), f"{a:02x}->{b:02x}"


def test_candle_shop_path_screens_chain() -> None:
    assert CANDLE_SHOP_SCREENS[0] == SCREEN_START == 0x77
    assert CANDLE_SHOP_SCREENS[-1] == SCREEN_CANDLE_SHOP == 0x5E
    assert len(CANDLE_SHOP_HOPS) == len(CANDLE_SHOP_SCREENS) - 1
    for a, b in zip(CANDLE_SHOP_SCREENS, CANDLE_SHOP_SCREENS[1:]):
        assert b in neighbor_screens(a).values(), f"{a:02x}->{b:02x}"


def test_candle_shop_shares_bush_prefix_then_east() -> None:
    """Shop reuses bush corridor through 0x5D; final hop is east not south."""
    assert CANDLE_SHOP_HOPS[:-1] == LEVEL8_BUSH_HOPS[:-1]
    last = CANDLE_SHOP_HOPS[-1]
    assert last.target == 0x5E
    assert last.direction == "RIGHT"
    assert last.y_band == (130, 150)
    bush_last = LEVEL8_BUSH_HOPS[-1]
    assert bush_last.target == 0x6D
    assert bush_last.direction == "DOWN"


def test_candle_shop_price_and_geometry_constants() -> None:
    assert CANDLE_SHOP_PRICE == 60
    assert CANDLE_SHOP_CAVE_X == 112
    assert CANDLE_BUY_X == 152
    assert CANDLE_BUY_Y == 149


def test_shop_controller_defaults() -> None:
    nav = OverworldToCandleShopController()
    assert nav.hops is CANDLE_SHOP_HOPS or nav.hops == CANDLE_SHOP_HOPS
    assert nav.end_screen() == 0x5E
    assert nav.door_x == CANDLE_SHOP_CAVE_X
    assert nav.enter_cave is True
    assert nav.buy_candle is False
    assert nav.phase is CandleShopNavPhase.HOP


def test_bush_controller_defaults() -> None:
    nav = OverworldToLevel8Controller()
    assert nav.hops[-1].target == 0x6D
    assert nav.door_screen == SCREEN_LEVEL8_BUSH


def test_level8_bush_screen_reached() -> None:
    assert level8_bush_screen_reached(_ram(screen=0x6D, sword=1))
    assert not level8_bush_screen_reached(_ram(screen=0x6D, sword=0))
    assert not level8_bush_screen_reached(_ram(screen=0x5E, sword=1))


def test_candle_shop_predicates() -> None:
    assert candle_shop_screen_reached(_ram(screen=0x5E, sword=1, mode=PLAY_MODE))
    assert not candle_shop_screen_reached(_ram(screen=0x5E, sword=0))
    assert candle_shop_cave_entered(_ram(screen=0x5E, mode=11, level=0))
    assert not candle_shop_cave_entered(_ram(screen=0x5E, mode=PLAY_MODE, level=0))

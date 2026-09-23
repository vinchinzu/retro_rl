"""Unit tests for the 0x4A wooden-arrow shop hop. No emulator."""

from __future__ import annotations

import numpy as np
from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.level1.arrow_shop import (
    ARROW_BUY_X,
    ARROW_BUY_Y,
    ARROW_SHOP_CAVE_X,
    ARROW_SHOP_PRICE,
    ARROW_SHOP_SCREEN,
    ArrowShopNavPhase,
    level1_arrows_success,
    make_arrow_shop_controller,
)
from zelda_i.ram import CAVE_MODE, PLAY_MODE, read_snapshot
from zelda_i.tests.ram_helpers import make_ram

_DEFAULTS = {
    "mode": PLAY_MODE,
    "screen": ARROW_SHOP_SCREEN,
    "x": 0,
    "y": 149,
    "sword": 1,
    "rupees": 0,
    "arrows": 0,
}


def _ram(**fields: int) -> np.ndarray:
    return make_ram(_DEFAULTS, **fields)


def test_arrows_success_is_inventory_only() -> None:
    cave = read_snapshot(_ram(mode=CAVE_MODE, x=152, y=157, arrows=1, rupees=120))
    assert level1_arrows_success(cave)
    ow = read_snapshot(_ram(mode=PLAY_MODE, x=176, y=77, arrows=0))
    assert not level1_arrows_success(ow)


def test_arrow_shop_scoops_drops_on_the_walk() -> None:
    ctl = make_arrow_shop_controller()
    assert ctl.need_rupees == ARROW_SHOP_PRICE
    assert ctl.price == ARROW_SHOP_PRICE


def test_buy_climbs_then_right_at_y165_not_bomb_row() -> None:
    ctl = make_arrow_shop_controller()
    ctl.phase = ArrowShopNavPhase.BUY
    ctl._rupees_at_buy = 80
    ctl.buy_frames = 200
    stairs = read_snapshot(
        _ram(mode=CAVE_MODE, x=112, y=213, rupees=80, arrows=0)
    )
    act = ctl._buy_step(stairs)
    assert list(act.action) == list(nes_action("UP"))
    band = read_snapshot(
        _ram(mode=CAVE_MODE, x=112, y=ARROW_BUY_Y, rupees=80, arrows=0)
    )
    act = ctl._buy_step(band)
    assert list(act.action) == list(nes_action("RIGHT"))
    assert ARROW_BUY_Y > 149
    touch = read_snapshot(
        _ram(mode=CAVE_MODE, x=ARROW_BUY_X, y=ARROW_BUY_Y, rupees=80, arrows=0)
    )
    act = ctl._buy_step(touch)
    assert list(act.action) == list(nes_action("UP"))
    unpaid = read_snapshot(
        _ram(mode=CAVE_MODE, x=ARROW_BUY_X, y=157, rupees=80, arrows=1)
    )
    act = ctl._buy_step(unpaid)
    assert not ctl.success
    assert list(act.action) == list(nes_action("UP"))
    done = read_snapshot(
        _ram(mode=CAVE_MODE, x=ARROW_BUY_X, y=157, rupees=0, arrows=1)
    )
    act = ctl._buy_step(done)
    assert ctl.success
    assert list(act.action) == list(nes_idle_action())


def test_farm_skips_when_rupees_already_80() -> None:
    ctl = make_arrow_shop_controller()
    ctl.hop_index = len(ctl.hops)
    ctl.phase = ArrowShopNavPhase.HOP
    snap = read_snapshot(_ram(x=16, y=149, rupees=ARROW_SHOP_PRICE, arrows=0))
    ctl._after_hops(snap)
    assert ctl.phase is ArrowShopNavPhase.DOOR


def test_farm_starts_when_rupees_short() -> None:
    ctl = make_arrow_shop_controller()
    ctl.hop_index = len(ctl.hops)
    ctl.phase = ArrowShopNavPhase.HOP
    snap = read_snapshot(_ram(x=16, y=149, rupees=7, arrows=0))
    act = ctl._after_hops(snap)
    assert ctl.phase is ArrowShopNavPhase.FARM
    assert any(
        token in act.reason
        for token in ("farm_chase", "farm_leave", "farm_south", "farm_wait")
    )


def test_farm_leaves_after_empty_to_respawn() -> None:
    ctl = make_arrow_shop_controller()
    ctl.phase = ArrowShopNavPhase.FARM
    snap = read_snapshot(_ram(x=160, y=149, rupees=15, arrows=0))
    for _ in range(89):
        act = ctl._farm_step(snap)
        assert "farm_wait" in act.reason
    act = ctl._farm_step(snap)
    assert "farm_leave" in act.reason


def test_cave_door_aligns_176_not_north_gap() -> None:
    ctl = make_arrow_shop_controller()
    ctl.phase = ArrowShopNavPhase.DOOR
    west = read_snapshot(_ram(x=0, y=149, rupees=80, arrows=0))
    act = ctl._simple_door_hunt(west)
    assert act.reason.startswith("door_ax")
    assert ARROW_SHOP_CAVE_X == 176

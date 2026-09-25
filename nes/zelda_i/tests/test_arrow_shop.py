"""Unit tests for the 0x4A/0x44 wooden arrow shop hop and restock controller. No emulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.overworld.arrow_shop import (
    ARROW_BUY_X,
    ARROW_BUY_Y,
    ARROW_SHOP_CAVE_X,
    ARROW_SHOP_PRICE,
    ARROW_SHOP_SCREEN,
    ArrowRestockController,
    SHOP_E5_APPROACH_Y,
    SHOP_E5_CAVE_X,
    SHOP_E5_SCREEN,
    arrow_restock_stages,
    arrow_shop_restock,
    arrow_shop_success,
    make_arrow_restock_controller,
    make_arrow_shop_controller,
)
from zelda_i.overworld.cave_shop import CaveShopBuyPhase
from zelda_i.overworld.graph import ScreenHop
from zelda_i.ram import ADDR_ARROWS, CAVE_MODE, PLAY_MODE, read_snapshot
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
    cave = read_snapshot(_ram(mode=CAVE_MODE, x=152, y=165, arrows=1, rupees=100))
    assert arrow_shop_success(cave)
    ow = read_snapshot(_ram(mode=PLAY_MODE, x=176, y=77, arrows=0))
    assert not arrow_shop_success(ow)


def test_arrow_shop_wants_80_not_20() -> None:
    ctl = make_arrow_shop_controller()
    assert ctl.need_rupees == ARROW_SHOP_PRICE
    assert ctl.price == ARROW_SHOP_PRICE
    assert ARROW_SHOP_PRICE == 80


def test_buy_climbs_then_right_at_south_pedestal_y165() -> None:
    """Touches y=165 (south, arrows) not y=149 (mid, bombs) — same corridor,
    different lateral stop."""
    ctl = make_arrow_shop_controller()
    ctl.phase = CaveShopBuyPhase.BUY
    ctl._rupees_at_buy = 80
    ctl.buy_frames = 200
    stairs = read_snapshot(_ram(mode=CAVE_MODE, x=112, y=213, rupees=80, arrows=0))
    act = ctl._buy_step(stairs)
    assert list(act.action) == list(nes_action("UP"))
    band = read_snapshot(_ram(mode=CAVE_MODE, x=112, y=ARROW_BUY_Y, rupees=80, arrows=0))
    act = ctl._buy_step(band)
    assert list(act.action) == list(nes_action("RIGHT"))
    assert ARROW_BUY_Y == 165
    touch = read_snapshot(
        _ram(mode=CAVE_MODE, x=ARROW_BUY_X, y=ARROW_BUY_Y, rupees=80, arrows=0)
    )
    act = ctl._buy_step(touch)
    assert list(act.action) == list(nes_action("UP"))
    unpaid = read_snapshot(
        _ram(mode=CAVE_MODE, x=ARROW_BUY_X, y=ARROW_BUY_Y, rupees=80, arrows=1)
    )
    act = ctl._buy_step(unpaid)
    assert not ctl.success
    assert list(act.action) == list(nes_action("UP"))
    done = read_snapshot(
        _ram(mode=CAVE_MODE, x=ARROW_BUY_X, y=ARROW_BUY_Y, rupees=0, arrows=1)
    )
    act = ctl._buy_step(done)
    assert ctl.success
    assert list(act.action) == list(nes_idle_action())


def test_farm_skips_when_rupees_already_80() -> None:
    ctl = make_arrow_shop_controller()
    ctl.hop_index = len(ctl.hops)
    ctl.phase = CaveShopBuyPhase.HOP
    snap = read_snapshot(_ram(x=16, y=149, rupees=ARROW_SHOP_PRICE, arrows=0))
    ctl._after_hops(snap)
    assert ctl.phase is CaveShopBuyPhase.DOOR


def test_farm_starts_when_rupees_short() -> None:
    ctl = make_arrow_shop_controller()
    ctl.hop_index = len(ctl.hops)
    ctl.phase = CaveShopBuyPhase.HOP
    snap = read_snapshot(_ram(x=16, y=149, rupees=3, arrows=0))
    act = ctl._after_hops(snap)
    assert ctl.phase is CaveShopBuyPhase.FARM
    assert any(
        token in act.reason
        for token in (
            "farm_chase",
            "farm_leave",
            "farm_south",
            "farm_wait",
            "farm_inland",
        )
    )


def test_cave_door_aligns_176_not_north_gap() -> None:
    ctl = make_arrow_shop_controller()
    ctl.phase = CaveShopBuyPhase.DOOR
    west = read_snapshot(_ram(x=0, y=149, rupees=80, arrows=0))
    act = ctl._simple_door_hunt(west)
    assert act.reason.startswith("door_ax")
    assert ARROW_SHOP_CAVE_X == 176


def test_controller_never_touches_arrow_addresses_directly() -> None:
    """Source guard: the module must not import write_u8 or write to ADDR_ARROWS directly."""
    from zelda_i.overworld import arrow_shop as mod

    assert not hasattr(mod, "write_u8")
    assert ADDR_ARROWS == 0x0659


def test_importing_the_module_is_safe() -> None:
    """The 0x4A restock check must not run at import time."""
    import importlib
    import sys

    sys.modules.pop("zelda_i.overworld.arrow_shop", None)
    mod = importlib.import_module("zelda_i.overworld.arrow_shop")
    assert mod.ARROW_SHOP_SCREEN == 0x4A
    src = (Path(mod.__file__).read_text()).split("def arrow_shop_restock")[0]
    assert "raise" not in src


def test_restock_pair_is_the_catalog_0x49_west_leave() -> None:
    assert arrow_shop_restock() == (0x49, "LEFT")
    ctl = make_arrow_shop_controller()
    assert ctl.farm.restock_neighbor_screen == 0x49
    assert ctl.farm.restock_direction == "LEFT"


def test_missing_restock_pair_raises_at_construction_not_import(monkeypatch) -> None:
    from zelda_i.overworld import arrow_shop as mod

    monkeypatch.setattr(mod, "restock_for", lambda screen: None)
    with pytest.raises(RuntimeError, match="restock pair"):
        mod.make_arrow_shop_controller()


def test_restock_skips_on_first_frame_when_arrows_cover_want() -> None:
    ctl = ArrowRestockController(want=1)
    ctl.reset()
    ctl.step(read_snapshot(_ram(screen=0x74, arrows=1, rupees=80)))
    assert ctl.success
    assert ctl.frames == 1
    assert "arrow_restock_enough" in ctl.notes


def test_restock_with_want_2_does_not_skip_on_wooden() -> None:
    ctl = ArrowRestockController(want=2)
    ctl.reset()
    ctl.step(read_snapshot(_ram(screen=0x74, arrows=1, rupees=80)))
    assert not ctl.success


def test_restock_short_of_want_walks_to_0x44() -> None:
    hops = (
        ScreenHop(0x73, "LEFT"),
        ScreenHop(0x63, "UP"),
        ScreenHop(0x64, "RIGHT"),
        ScreenHop(0x54, "UP"),
        ScreenHop(SHOP_E5_SCREEN, "UP"),
    )
    ctl = make_arrow_restock_controller(hops=hops, want=1, screen=SHOP_E5_SCREEN)
    ctl.reset()
    ctl.step(read_snapshot(_ram(screen=0x74, arrows=0, rupees=85)))
    assert not ctl.success
    assert [hop.target for hop in ctl.hops] == [0x73, 0x63, 0x64, 0x54, SHOP_E5_SCREEN]
    assert ctl.shop_screen == SHOP_E5_SCREEN
    assert (ctl.door_x, ctl.mouth_approach_y) == (SHOP_E5_CAVE_X, SHOP_E5_APPROACH_Y)
    assert (ctl.buy_x, ctl.buy_y, ctl.price) == (ARROW_BUY_X, ARROW_BUY_Y, ARROW_SHOP_PRICE)
    assert ctl.farm is None  # a short wallet fails closed, never a poke


def test_restock_short_of_want_walks_to_0x4A() -> None:
    hops = (
        ScreenHop(0x37, "UP"),
        ScreenHop(0x4A, "RIGHT"),
    )
    ctl = make_arrow_restock_controller(hops=hops, want=1, screen=ARROW_SHOP_SCREEN)
    ctl.reset()
    ctl.step(read_snapshot(_ram(screen=0x37, arrows=0, rupees=90)))
    assert not ctl.success
    assert ctl.shop_screen == ARROW_SHOP_SCREEN
    assert (ctl.door_x, ctl.cave_x) == (ARROW_SHOP_CAVE_X, ARROW_SHOP_CAVE_X)
    assert (ctl.buy_x, ctl.buy_y, ctl.price) == (ARROW_BUY_X, ARROW_BUY_Y, ARROW_SHOP_PRICE)
    assert ctl.farm is None


def test_restock_fails_closed_when_rupees_short_without_farm() -> None:
    hops = (ScreenHop(SHOP_E5_SCREEN, "UP"),)
    ctl = make_arrow_restock_controller(hops=hops, want=1, screen=SHOP_E5_SCREEN)
    ctl.reset()
    # At the shop screen, hops are complete
    ctl.hop_index = len(ctl.hops)
    ctl.phase = CaveShopBuyPhase.HOP
    snap = read_snapshot(_ram(screen=SHOP_E5_SCREEN, arrows=0, rupees=40))
    act = ctl.step(snap)
    assert ctl.phase is CaveShopBuyPhase.FAILED
    assert not ctl.success
    assert "shop_need_80_have_40" in act.reason
    assert "shop_need_80_have_40" in ctl.notes[-1]


def test_arrow_restock_stages() -> None:
    hops = (
        ScreenHop(0x73, "LEFT"),
        ScreenHop(0x63, "UP"),
        ScreenHop(0x64, "RIGHT"),
        ScreenHop(0x54, "UP"),
        ScreenHop(SHOP_E5_SCREEN, "UP"),
        ScreenHop(0x35, "UP"),
    )
    stages = arrow_restock_stages(hops, "l3_l4", want=1, screen=SHOP_E5_SCREEN)
    assert len(stages) == 2
    (name1, ctl1, frames1), (name2, ctl2, frames2) = stages
    assert name1 == "arrow_restock_l3_l4"
    assert isinstance(ctl1, ArrowRestockController)
    assert [h.target for h in ctl1.hops] == [0x73, 0x63, 0x64, 0x54, SHOP_E5_SCREEN]
    assert name2 == "exit_arrow_restock_l3_l4"
    assert frames2 == 600


def test_post_l4_arrow_walk_steps_off_raft_before_turning_east() -> None:
    hops = (ScreenHop(0x55, "DOWN"), ScreenHop(0x56, "RIGHT", align_y=141))
    ctl = make_arrow_restock_controller(hops=hops, screen=ARROW_SHOP_SCREEN)
    ctl.frames = 1
    ctl.hop_index = 1
    dock = read_snapshot(_ram(screen=0x55, x=128, y=125, sword=2, rupees=75))
    act = ctl.step(dock)
    assert act.reason == "raft_dismount"
    assert list(act.action) == list(nes_action("DOWN"))
    landed = read_snapshot(_ram(screen=0x55, x=128, y=141, sword=2, rupees=75))
    act = ctl.step(landed)
    assert act.reason != "raft_dismount"

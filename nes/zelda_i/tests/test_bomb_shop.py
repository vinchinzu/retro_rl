"""Unit tests for the 0x4A bomb mid-pedestal shop hop. No emulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.overworld.bomb_shop import (
    BOMB_BUY_X,
    BOMB_BUY_Y,
    BOMB_SHOP_CAVE_X,
    BOMB_SHOP_PRICE,
    BOMB_SHOP_SCREEN,
    bomb_shop_restock,
    bomb_shop_success,
    make_bomb_shop_controller,
)
from zelda_i.overworld.cave_shop import CaveShopBuyPhase
from zelda_i.ram import ADDR_BOMBS, ADDR_MAX_BOMBS, CAVE_MODE, PLAY_MODE, read_snapshot
from zelda_i.tests.ram_helpers import make_ram

_DEFAULTS = {
    "mode": PLAY_MODE,
    "screen": BOMB_SHOP_SCREEN,
    "x": 0,
    "y": 149,
    "sword": 1,
    "rupees": 0,
    "bombs": 0,
}


def _ram(**fields: int) -> np.ndarray:
    return make_ram(_DEFAULTS, **fields)


def test_bombs_success_is_inventory_only() -> None:
    cave = read_snapshot(_ram(mode=CAVE_MODE, x=152, y=149, bombs=1, rupees=100))
    assert bomb_shop_success(cave)
    ow = read_snapshot(_ram(mode=PLAY_MODE, x=176, y=77, bombs=0))
    assert not bomb_shop_success(ow)


def test_bomb_shop_wants_20_not_80() -> None:
    ctl = make_bomb_shop_controller()
    assert ctl.need_rupees == BOMB_SHOP_PRICE
    assert ctl.price == BOMB_SHOP_PRICE
    assert BOMB_SHOP_PRICE == 20


def test_buy_climbs_then_right_at_mid_pedestal_y149() -> None:
    """Touches y=149 (mid, bombs) not y=165 (south, arrows) — same corridor,
    different lateral stop than ``OverworldToArrowShopController``."""
    ctl = make_bomb_shop_controller()
    ctl.phase = CaveShopBuyPhase.BUY
    ctl._rupees_at_buy = 20
    ctl.buy_frames = 200
    stairs = read_snapshot(_ram(mode=CAVE_MODE, x=112, y=213, rupees=20, bombs=0))
    act = ctl._buy_step(stairs)
    assert list(act.action) == list(nes_action("UP"))
    band = read_snapshot(_ram(mode=CAVE_MODE, x=112, y=BOMB_BUY_Y, rupees=20, bombs=0))
    act = ctl._buy_step(band)
    assert list(act.action) == list(nes_action("RIGHT"))
    assert BOMB_BUY_Y == 149
    touch = read_snapshot(
        _ram(mode=CAVE_MODE, x=BOMB_BUY_X, y=BOMB_BUY_Y, rupees=20, bombs=0)
    )
    act = ctl._buy_step(touch)
    assert list(act.action) == list(nes_action("UP"))
    done = read_snapshot(
        _ram(mode=CAVE_MODE, x=BOMB_BUY_X, y=BOMB_BUY_Y, rupees=20, bombs=1)
    )
    act = ctl._buy_step(done)
    assert ctl.success
    assert list(act.action) == list(nes_idle_action())


def test_farm_skips_when_rupees_already_20() -> None:
    ctl = make_bomb_shop_controller()
    ctl.hop_index = len(ctl.hops)
    ctl.phase = CaveShopBuyPhase.HOP
    snap = read_snapshot(_ram(x=16, y=149, rupees=BOMB_SHOP_PRICE, bombs=0))
    ctl._after_hops(snap)
    assert ctl.phase is CaveShopBuyPhase.DOOR


def test_farm_starts_when_rupees_short() -> None:
    ctl = make_bomb_shop_controller()
    ctl.hop_index = len(ctl.hops)
    ctl.phase = CaveShopBuyPhase.HOP
    snap = read_snapshot(_ram(x=16, y=149, rupees=3, bombs=0))
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
    ctl = make_bomb_shop_controller()
    ctl.phase = CaveShopBuyPhase.DOOR
    west = read_snapshot(_ram(x=0, y=149, rupees=20, bombs=0))
    act = ctl._simple_door_hunt(west)
    assert act.reason.startswith("door_ax")
    assert BOMB_SHOP_CAVE_X == 176


def test_controller_never_touches_bomb_addresses_directly() -> None:
    """Source guard: the module must not import ``ADDR_MAX_BOMBS``, and its
    only reference to a RAM-write helper must be absent (natural purchase
    only; ``ADDR_BOMBS`` is passed as ``success_addr`` for reporting only)."""
    from zelda_i.overworld import bomb_shop as mod

    assert not hasattr(mod, "ADDR_MAX_BOMBS")
    assert "ADDR_MAX_BOMBS" not in mod.__dict__
    assert not hasattr(mod, "write_u8")
    assert ADDR_BOMBS == 0x0658
    assert ADDR_MAX_BOMBS == 0x067C


def test_importing_the_module_is_safe() -> None:
    """The 0x4A restock check must not run at import time.

    ``spine.survival`` imports this module unconditionally, so a module-level
    ``raise`` over a *catalog* fact takes the whole package down on import —
    including every unrelated test and CLI. The check belongs where the pair
    is actually needed: constructing the controller.
    """
    import importlib
    import sys

    sys.modules.pop("zelda_i.overworld.bomb_shop", None)
    mod = importlib.import_module("zelda_i.overworld.bomb_shop")
    assert mod.BOMB_SHOP_SCREEN == 0x4A
    src = (Path(mod.__file__).read_text()).split("def bomb_shop_restock")[0]
    assert "raise" not in src


def test_restock_pair_is_the_catalog_0x49_west_leave() -> None:
    """Constructing succeeds today: ``locations._RESTOCK`` has the 0x4A pair.

    The guard is a live invariant, not dead defence — the pair is read from
    the catalog, so editing ``_RESTOCK`` is what would trip it.
    """
    assert bomb_shop_restock() == (0x49, "LEFT")
    ctl = make_bomb_shop_controller()
    assert ctl.farm.restock_neighbor_screen == 0x49
    assert ctl.farm.restock_direction == "LEFT"


def test_missing_restock_pair_raises_at_construction_not_import(monkeypatch) -> None:
    from zelda_i.overworld import bomb_shop as mod

    monkeypatch.setattr(mod, "restock_for", lambda screen: None)
    with pytest.raises(RuntimeError, match="restock pair"):
        mod.make_bomb_shop_controller()

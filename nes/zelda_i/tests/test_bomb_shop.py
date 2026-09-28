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
    unpaid = read_snapshot(
        _ram(mode=CAVE_MODE, x=BOMB_BUY_X, y=BOMB_BUY_Y, rupees=20, bombs=1)
    )
    act = ctl._buy_step(unpaid)
    assert not ctl.success
    assert list(act.action) == list(nes_action("UP"))
    done = read_snapshot(
        _ram(mode=CAVE_MODE, x=BOMB_BUY_X, y=BOMB_BUY_Y, rupees=0, bombs=1)
    )
    act = ctl._buy_step(done)
    assert ctl.success
    assert list(act.action) == list(nes_idle_action())


def test_held_bombs_require_a_paid_new_pack() -> None:
    ctl = make_bomb_shop_controller(restock_farm=False)
    ctl.phase = CaveShopBuyPhase.BUY
    ctl._rupees_at_buy = 40
    held = read_snapshot(_ram(mode=CAVE_MODE, x=BOMB_BUY_X, y=BOMB_BUY_Y,
                              bombs=4, max_bombs=8, rupees=40))
    ctl._buy_step(held)
    assert not ctl.success
    unpaid = read_snapshot(_ram(mode=CAVE_MODE, x=BOMB_BUY_X, y=BOMB_BUY_Y,
                                bombs=4, max_bombs=8, rupees=20))
    ctl._buy_step(unpaid)
    assert not ctl.success
    paid = read_snapshot(_ram(mode=CAVE_MODE, x=BOMB_BUY_X, y=BOMB_BUY_Y,
                              bombs=8, max_bombs=8, rupees=20))
    ctl._buy_step(paid)
    assert ctl.success


def test_bomb_buy_rejects_partial_pack_headroom() -> None:
    ctl = make_bomb_shop_controller(restock_farm=False)
    ctl.phase = CaveShopBuyPhase.BUY
    ctl._rupees_at_buy = 40
    fullish = read_snapshot(_ram(mode=CAVE_MODE, x=BOMB_BUY_X, y=BOMB_BUY_Y,
                                 bombs=5, max_bombs=8, rupees=40))
    ctl._buy_step(fullish)
    assert not ctl.success
    assert "shop_headroom_3_need_4" in ctl.notes[-1]


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


def _l4_restock():
    from zelda_i.level4.overworld import LEVEL4_BOMB_WALLS, LEVEL4_HOPS_VIA_SHOP_E5
    from zelda_i.overworld.bomb_shop import bomb_restock_stages

    (_, buy, _), _ = bomb_restock_stages(LEVEL4_HOPS_VIA_SHOP_E5, "l3", want=LEVEL4_BOMB_WALLS)
    return buy


def test_restock_skips_on_first_frame_when_bombs_cover_want() -> None:
    from zelda_i.level4.overworld import LEVEL4_BOMB_WALLS

    ctl = _l4_restock()
    ctl.reset()
    ctl.step(read_snapshot(_ram(screen=0x74, bombs=LEVEL4_BOMB_WALLS, rupees=46)))
    assert ctl.success
    assert ctl.frames == 1


def test_restock_short_of_want_walks_to_0x44() -> None:
    from zelda_i.overworld.bomb_shop import (
        SHOP_E5_APPROACH_Y,
        SHOP_E5_CAVE_X,
        SHOP_E5_SCREEN,
    )

    ctl = _l4_restock()
    ctl.reset()
    ctl.step(read_snapshot(_ram(screen=0x74, bombs=2, rupees=46)))
    assert not ctl.success
    assert [hop.target for hop in ctl.hops] == [0x73, 0x63, 0x64, 0x54, SHOP_E5_SCREEN]
    assert ctl.shop_screen == SHOP_E5_SCREEN
    assert (ctl.door_x, ctl.mouth_approach_y) == (SHOP_E5_CAVE_X, SHOP_E5_APPROACH_Y)
    assert (ctl.buy_x, ctl.buy_y, ctl.price) == (BOMB_BUY_X, BOMB_BUY_Y, BOMB_SHOP_PRICE)
    assert ctl.farm is None  # a short wallet fails closed, never a poke


def test_level4_walk_resumes_after_the_shop_or_the_potion() -> None:
    """One hop list serves 0x44 (after the buy) and 0x64 (after a potion)."""
    from zelda_i.level4.overworld import (
        LEVEL4_HOPS_FROM_POST_L3,
        LEVEL4_HOPS_VIA_SHOP_E5,
        OverworldToLevel4Controller,
    )

    def next_target(screen: int) -> int:
        ctl = OverworldToLevel4Controller(hops=LEVEL4_HOPS_VIA_SHOP_E5, resume_on_screen=True)
        ctl.reset()
        ctl.step(read_snapshot(_ram(screen=screen, x=120, y=141, bombs=6, rupees=26)))
        return ctl.hops[ctl.hop_index].target

    assert next_target(0x44) == 0x54
    assert next_target(0x64) == 0x65
    assert LEVEL4_HOPS_VIA_SHOP_E5[-3:] == LEVEL4_HOPS_FROM_POST_L3[-3:]


def _l9_restock(*, skip: bool):
    from zelda_i.level9.overworld import POST_L8_TO_BOMB_SHOP_HOPS
    from zelda_i.overworld.bomb_shop import bomb_restock_stages

    (_, buy, _), _ = bomb_restock_stages(
        POST_L8_TO_BOMB_SHOP_HOPS, "l8", want=8,
        shop_screen=BOMB_SHOP_SCREEN, skip_unaffordable=skip,
    )
    buy.reset()
    return buy


def test_l9_restock_skips_a_pack_the_wallet_cannot_pay_for() -> None:
    ctl = _l9_restock(skip=True)
    ctl.step(read_snapshot(_ram(bombs=4, rupees=2)))
    assert ctl.success and ctl.phase is CaveShopBuyPhase.DONE
    assert ctl.frames == 1
    assert "bomb_restock_unaffordable_2r" in ctl.notes


def test_restock_without_the_opt_in_still_fails_closed_on_a_short_wallet() -> None:
    ctl = _l9_restock(skip=False)
    ctl.step(read_snapshot(_ram(bombs=4, rupees=2)))
    assert not ctl.success
    paid = _l9_restock(skip=True)
    paid.step(read_snapshot(_ram(bombs=4, rupees=20)))
    assert not paid.success
    assert paid.phase not in (CaveShopBuyPhase.DONE, CaveShopBuyPhase.FAILED)


def test_level9_entry_opts_both_0x4a_packs_into_the_skip() -> None:
    from zelda_i.level9.dungeon import MEASURED_POST_L8_HANDOFF
    from zelda_i.level9.hops import level9_entry_chapter

    buys = {
        name: ctl for name, ctl, _ in level9_entry_chapter(handoff=MEASURED_POST_L8_HANDOFF)
        if name.startswith("bomb_restock_")
    }
    assert set(buys) == {"bomb_restock_l8", "bomb_restock_l8_second"}
    assert all(ctl.skip_unaffordable for ctl in buys.values())

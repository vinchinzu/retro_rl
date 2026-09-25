"""Unit and live-fixture tests for the natural 60R bait buy at 0x34 (rr-8t4.5).

Verifies the cave policy and controller for buying Bait at 0x34 for 60R:
- Screen: 0x34 (Armos special shop: Key 80 left, Blue Ring 250 middle, Bait 60 right)
- Cave entrance: the (64,128) Armos, pushed LEFT from (80,125); wait at (80,109)
- Cave interior: spawn (112,213), stairs UP to y=165, lateral RIGHT to x=152, touch UP
- Pedestal contact at (152,157) flips ADDR_FOOD 0->1 and debits 60R naturally
- Acceptance criteria (rr-8t4.5):
  "From leftover on 0x34 with >=60R: Food 0->1, rupees -60, food_writes=0,
   inventory_assist empty. SurvivalBaitPurchaseController off the spine."
"""

from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest
from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.level7.entry import (
    BAIT_BUY_BUDGET,
    BAIT_BUY_X,
    BAIT_BUY_Y,
    BAIT_CAVE_X,
    BAIT_CAVE_Y,
    BAIT_COST,
    BAIT_DOOR_X,
    BAIT_MAX_FRAMES,
    BAIT_SHOP_SCREEN,
    UNVERIFIED_BAIT_PLAN,
    VERIFIED_BAIT_PLAN,
    BaitPurchasePlan,
    NaturalBaitPurchaseController,
    SurvivalBaitPurchaseController,
    make_bait_purchase_controller,
)
from zelda_i.level7.hops import l7_hops, level7_entry_chapter_stages
from zelda_i.dungeon import tilemap as tm
from zelda_i.overworld.cave_shop import (
    SHOP_34_ARMOS_STAND,
    SHOP_34_ARMOS_TILE,
    SHOP_34_ARMOS_WAIT,
    STAIRS_TILE,
    CaveShopBuyPhase,
)
from zelda_i.ram import (
    ADDR_FOOD,
    ADDR_RUPEES,
    CAVE_MODE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)
from zelda_i.tests.ram_helpers import make_ram, room_tile_ram

_DEFAULTS = {
    "mode": PLAY_MODE,
    "screen": BAIT_SHOP_SCREEN,
    "x": 64,
    "y": 109,
    "sword": 1,
    "rupees": 100,
    "food": 0,
}


def _ram(**fields: int) -> np.ndarray:
    return make_ram(_DEFAULTS, **fields)


def _poke_env(ram: np.ndarray):
    calls: list[tuple[int, int]] = []

    class _Mem:
        def assign(self, addr: int, _fmt: str, val: int) -> None:
            calls.append((int(addr), int(val)))
            ram[int(addr)] = int(val) & 0xFF

    env = SimpleNamespace(
        get_ram=lambda: ram,
        unwrapped=SimpleNamespace(data=SimpleNamespace(memory=_Mem())),
    )
    return env, SimpleNamespace(calls=calls)


def test_bait_constants_and_plan_geometry() -> None:
    """Bait purchase constants match measured 0x34 geometry."""
    assert BAIT_SHOP_SCREEN == 0x34
    assert BAIT_COST == 60
    assert BAIT_CAVE_X == 64
    assert BAIT_CAVE_Y == 125
    assert BAIT_DOOR_X == 64
    assert BAIT_BUY_X == 152
    assert BAIT_BUY_Y == 165
    assert BAIT_BUY_BUDGET == 1500
    assert BAIT_MAX_FRAMES == 18000

    assert VERIFIED_BAIT_PLAN.shop_screen == 0x34
    assert VERIFIED_BAIT_PLAN.cost == 60
    assert VERIFIED_BAIT_PLAN.shop_cave_xy == (64, 125)
    assert VERIFIED_BAIT_PLAN.shop_geometry_verified is True
    assert VERIFIED_BAIT_PLAN.route_eligible is True
    assert VERIFIED_BAIT_PLAN.can_pay(60) is True
    assert VERIFIED_BAIT_PLAN.can_pay(59) is False

    assert UNVERIFIED_BAIT_PLAN.shop_geometry_verified is False
    assert UNVERIFIED_BAIT_PLAN.shop_cave_xy is None


def test_natural_bait_fails_closed_when_short_of_60r() -> None:
    """Wallet < 60R fails closed with bait_need_60_rupees and zero writes."""
    ram = _ram(rupees=42, food=0)
    env, mem = _poke_env(ram)
    ctl = make_bait_purchase_controller()
    ctl.bind_env(env)
    act = ctl.step(read_snapshot(ram))
    assert ctl.failed is True
    assert act.reason == "bait_need_60_rupees"
    assert mem.calls == []
    report = ctl.report()
    assert report["writes"] == 0
    assert report["inventory_assist"] is None


def test_bait_walk_allows_hidden_cave_payout_to_finish_counting() -> None:
    """The 0x62 payout may still be below 60R on the first walking frame."""
    from zelda_i.overworld.graph import ScreenHop

    ram = _ram(screen=0x62, rupees=34, food=0)
    env, mem = _poke_env(ram)
    ctl = make_bait_purchase_controller(hops=(ScreenHop(0x52, "UP"),))
    ctl.bind_env(env)
    act = ctl.step(read_snapshot(ram))
    assert ctl.failed is False
    assert act.reason != "bait_need_60_rupees"
    assert mem.calls == []


def test_natural_bait_fails_closed_when_geometry_unverified() -> None:
    """Unverified plan fails closed with bait_shop_geometry_unobserved."""
    ram = _ram(rupees=100, food=0)
    env, mem = _poke_env(ram)
    ctl = make_bait_purchase_controller(plan=UNVERIFIED_BAIT_PLAN)
    ctl.bind_env(env)
    act = ctl.step(read_snapshot(ram))
    assert ctl.failed is True
    assert act.reason == "bait_shop_geometry_unobserved"
    assert mem.calls == []
    assert ctl.report()["writes"] == 0


def test_natural_bait_fails_closed_when_off_shop_screen() -> None:
    """Link not on 0x34 and no approach hops fails closed."""
    ram = _ram(screen=0x42, rupees=100, food=0)
    env, mem = _poke_env(ram)
    ctl = make_bait_purchase_controller()
    ctl.bind_env(env)
    act = ctl.step(read_snapshot(ram))
    assert ctl.failed is True
    assert act.reason == "bait_not_on_shop_screen"
    assert mem.calls == []


def test_natural_bait_already_owned_finishes_without_write() -> None:
    """If Food is already owned upon arrival, passes cleanly with 0 writes."""
    ram = _ram(rupees=100, food=1)
    env, mem = _poke_env(ram)
    ctl = make_bait_purchase_controller()
    ctl.bind_env(env)
    act = ctl.step(read_snapshot(ram))
    assert ctl.success is True
    assert not ctl.failed
    assert mem.calls == []
    assert ctl.report()["writes"] == 0
    assert ram[ADDR_RUPEES] == 100  # untouched


def test_level7_bait_stage_accepts_food_carried_from_gathering() -> None:
    """Level 7 starts at the pond; the purchase happened before Level 1."""
    ram = _ram(screen=0x42, rupees=0, food=1)
    env, mem = _poke_env(ram)
    ctl = make_bait_purchase_controller()
    ctl.bind_env(env)
    action = ctl.step(read_snapshot(ram))
    assert action.reason in ("done", "bait_already_owned")
    assert ctl.success is True
    assert mem.calls == []
    assert ctl.report()["writes"] == 0
    assert ram[ADDR_RUPEES] == 0


def _shop_34(revealed: bool = False, **fields: int):
    """The captured 0x34 map (six statues) under a snapshot's fields."""
    ram = room_tile_ram("0x34", level=0)
    ram[:0x800] = _ram(**fields)
    if revealed:
        ax, ay = SHOP_34_ARMOS_TILE
        col, row = ax // tm.TILE_PX, (ay - tm.PLAYFIELD_TOP_Y) // tm.TILE_PX
        ram[tm.WRAM_RAM_OFFSET + tm.ADDR_ROOM_TILE_MAP - tm.WRAM_BASE + col * tm.TILE_ROWS + row] = STAIRS_TILE
    return ram, SimpleNamespace(get_ram=lambda: ram)


def test_natural_bait_armos_door_outside_0x34() -> None:
    """Wake only the stairs Armos from its east side, then step on the stairs.

    The old hunt climbed x=64 from y=189 and woke the (64,160) statue on top
    of Link first: 4-6 hits per visit with no refill.
    """
    ctl = make_bait_purchase_controller()
    ctl.hop_index = len(ctl.hops)
    ctl.phase = CaveShopBuyPhase.DOOR

    # 1. From the bottom of the screen: walk the lattice, not up x=64.
    ram, env = _shop_34(x=128, y=189)
    ctl.bind_env(env)
    act = ctl._simple_door_hunt(read_snapshot(ram))
    assert act.reason.startswith("armos_stand")

    # 2. On the stand: push LEFT into the statue.
    ram, env = _shop_34(x=SHOP_34_ARMOS_STAND[0], y=SHOP_34_ARMOS_STAND[1])
    ctl.bind_env(env)
    act = ctl._simple_door_hunt(read_snapshot(ram))
    assert act.reason == "armos_touch"
    assert act.action == nes_action("LEFT")

    # 3. Stairs showing, no Armos near them: walk onto (64,125).
    ram, env = _shop_34(revealed=True, x=SHOP_34_ARMOS_WAIT[0], y=SHOP_34_ARMOS_WAIT[1])
    ctl.bind_env(env)
    act = ctl._simple_door_hunt(read_snapshot(ram))
    assert act.reason == "armos_stairs"

    # 4. In cave: transitions to BUY phase
    cave = read_snapshot(_ram(mode=CAVE_MODE, screen=0x34, x=112, y=213))
    act = ctl._simple_door_hunt(cave)
    assert ctl.phase is CaveShopBuyPhase.BUY
    assert act.reason == "shop_ready"


def test_natural_bait_buy_lateral_and_pedestal_contact_in_cave() -> None:
    """Cave interior buy state machine: stairs UP, RIGHT lateral, touch UP."""
    ctl = make_bait_purchase_controller()
    ctl.phase = CaveShopBuyPhase.BUY
    ctl._rupees_at_buy = 100
    ctl.buy_frames = 200  # past dialog idle

    # 1. Stairs: Link at cave bottom climbs UP
    stairs = read_snapshot(_ram(mode=CAVE_MODE, x=112, y=213, rupees=100, food=0))
    act = ctl._buy_step(stairs)
    assert list(act.action) == list(nes_action("UP"))

    # 2. Lateral band: at buy_y (165), moves RIGHT to avoid middle pedestal
    band = read_snapshot(_ram(mode=CAVE_MODE, x=112, y=BAIT_BUY_Y, rupees=100, food=0))
    act = ctl._buy_step(band)
    assert list(act.action) == list(nes_action("RIGHT"))

    # 3. Alignment: at buy_x (152), y=165, pushes UP toward right pedestal
    aligned = read_snapshot(_ram(mode=CAVE_MODE, x=BAIT_BUY_X, y=BAIT_BUY_Y, rupees=100, food=0))
    act = ctl._buy_step(aligned)
    assert list(act.action) == list(nes_action("UP"))

    # 4. Touching pedestal before purchase confirms: continues UP
    touching = read_snapshot(_ram(mode=CAVE_MODE, x=BAIT_BUY_X, y=157, rupees=100, food=0))
    act = ctl._buy_step(touching)
    assert not ctl.success
    assert list(act.action) == list(nes_action("UP"))

    # 5. Unpaid item: Food flipped but rupees not yet debited by 60 -> continues UP
    unpaid = read_snapshot(_ram(mode=CAVE_MODE, x=BAIT_BUY_X, y=157, rupees=100, food=1))
    act = ctl._buy_step(unpaid)
    assert not ctl.success
    assert list(act.action) == list(nes_action("UP"))

    # 6. Paid item: Food=1 and rupees <= 100 - 60 = 40 -> completes with success
    paid = read_snapshot(_ram(mode=CAVE_MODE, x=BAIT_BUY_X, y=157, rupees=40, food=1))
    act = ctl._buy_step(paid)
    assert ctl.success is True
    assert ctl.phase is CaveShopBuyPhase.DONE
    assert list(act.action) == list(nes_idle_action())


def test_survival_spine_uses_natural_bait_controller_and_retires_food_fixture() -> None:
    """Survival spine has NaturalBaitPurchaseController; SurvivalBaitPurchaseController is off."""
    stages = level7_entry_chapter_stages(survival=True)
    by_name = {n: c for n, c, _f in stages}
    bait_ctl = by_name["level7_bait_purchase"]

    assert isinstance(bait_ctl, NaturalBaitPurchaseController)
    assert not isinstance(bait_ctl, SurvivalBaitPurchaseController)

    # Substantive behavioral check: stepping the survival stage on food=0 never pokes RAM
    ram = _ram(rupees=40, food=0)
    env, mem = _poke_env(ram)
    bait_ctl.bind_env(env)
    act = bait_ctl.step(read_snapshot(ram))
    assert mem.calls == []
    assert ram[ADDR_FOOD] == 0
    assert ram[ADDR_RUPEES] == 40
    assert bait_ctl.report()["writes"] == 0

    # Check hops tuple as well
    ram2 = _ram()
    env2, _ = _poke_env(ram2)
    hops = l7_hops(env2, survival=True)
    entry_hop = next(h for h in hops if h.through == "level7-entry")
    entry_stages = entry_hop.stages()
    entry_by_name = {n: c for n, c, _f in entry_stages}
    assert isinstance(entry_by_name["level7_bait_purchase"], NaturalBaitPurchaseController)
    assert not isinstance(entry_by_name["level7_bait_purchase"], SurvivalBaitPurchaseController)


def test_natural_bait_live_from_leftover_0x34() -> None:
    """Acceptance criteria rr-8t4.5: live emulator buy from leftover on 0x34.

    From GatherChain_exit_ring (OW 0x34 outside cave):
    - Starts with 100R, food=0
    - Controller steps onto stairs, enters cave, walks to right pedestal
    - Food flips 0 -> 1
    - Rupees debit naturally: 100 -> 40 (-60R)
    - food_writes = 0, progression_writes = 0, capacity_writes = 0
    - inventory_assist is empty (None)
    - SurvivalBaitPurchaseController off the spine
    """
    from retro_harness.env import make_env
    from zelda_i.paths import GAME, GAME_DIR

    try:
        env = make_env(GAME, "GatherChain_exit_ring", GAME_DIR, render_mode="rgb_array")
    except Exception as exc:
        pytest.skip(f"GatherChain_exit_ring state or ROM not available: {exc}")

    try:
        env.reset()
        mem = env.unwrapped.data.memory
        # Fund the natural purchase: set wallet to 100R (>= 60R)
        mem.assign(int(ADDR_RUPEES), "|u1", 100)
        s0 = read_snapshot(env.get_ram())
        assert s0.screen == 0x34
        assert s0.rupees == 100
        assert s0.food == 0

        ctl = make_bait_purchase_controller()
        ctl.bind_env(env)

        for _ in range(1200):
            s = read_snapshot(env.get_ram())
            act = ctl.step(s)
            env.step(act.action)
            if ctl.success:
                break
            if ctl.failed:
                pytest.fail(f"Natural bait buy failed: {ctl.notes}")

        s_final = read_snapshot(env.get_ram())
        ram_final = env.get_ram()
        food_final = read_u8(ram_final, ADDR_FOOD)
        rupees_final = read_u8(ram_final, ADDR_RUPEES)

        # Acceptance criteria assertions:
        assert ctl.success is True
        assert food_final == 1  # Food 0 -> 1
        assert rupees_final == 40  # Rupees 100 -> 40 (-60R)
        report = ctl.report()
        assert report["writes"] == 0  # food_writes = 0
        assert report["inventory_assist"] is None  # inventory_assist empty
        assert report["spec_id"] == "level7_bait_purchase"
        assert report["cost"] == 60
    finally:
        env.close()

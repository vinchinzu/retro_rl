"""Fail-closed L7 hops: OverworldHandoff gate, bait, Hungry Goriya, Red Candle."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from zelda_i.anchors import SCREEN_LEVEL6_ENTRANCE
from zelda_i.level7.entry import (
    BAIT_COST,
    BAIT_SHOP_SCREEN_HYP,
    MEASURED_POST_L6_EXIT,
    POST_L6_TRIFORCE,
    NaturalBaitPurchaseController,
    SurvivalBaitPurchaseController,
    make_bait_purchase_controller,
    make_post_l6_overworld_controller,
    make_survival_bait_purchase_controller,
)
from zelda_i.level7.hops import (
    l7_hops,
    level7_entry_chapter_stages,
    level7_red_candle_chapter_stages,
    make_entry_first_door_controller,
    make_entry_to_goriya_controller,
    make_pond_entry_controller,
    make_red_candle_controller,
)
from zelda_i.level7.path import (
    NORTH_DOOR_X,
    NORTH_DOOR_Y,
    SOUTH_MOUTH_Y,
    EntryNorthDoorController,
    north_door_79_step,
)
from zelda_i.level7.overworld import POST_L6_TO_BAIT_HOPS
from zelda_i.overworld.stitch import UNMEASURED_HANDOFF, OverworldHandoff
from retro_harness.nes import nes_idle_action
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_BOW,
    ADDR_CANDLE,
    ADDR_FOOD,
    ADDR_HEALTH,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_ROD,
    ADDR_RUPEES,
    ADDR_SCREEN,
    ADDR_SELECTED_ITEM,
    ADDR_SWORD,
    ADDR_TRIFORCE,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 0)
    ram[ADDR_SCREEN] = fields.get("screen", 0x09)
    ram[ADDR_LINK_X] = fields.get("x", 56)
    ram[ADDR_LINK_Y] = fields.get("y", 109)
    ram[ADDR_SWORD] = fields.get("sword", 1)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x1F)
    ram[ADDR_KEYS] = fields.get("keys", 3)
    ram[ADDR_BOMBS] = fields.get("bombs", 8)
    ram[ADDR_ARROWS] = fields.get("arrows", 1)
    ram[ADDR_HEALTH] = fields.get("health", 0xBB)
    ram[ADDR_WHISTLE] = fields.get("whistle", 1)
    ram[ADDR_FOOD] = fields.get("food", 0)
    ram[ADDR_ROD] = fields.get("rod", 0)
    ram[ADDR_BOW] = fields.get("bow", 1)
    ram[ADDR_CANDLE] = fields.get("candle", 1)
    ram[ADDR_RUPEES] = fields.get("rupees", 20)
    ram[ADDR_SELECTED_ITEM] = fields.get("selected", 0)
    return ram


def _env(ram: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(get_ram=lambda: ram)


class _WriteThroughMem:
    """``memory.assign`` mock that writes back into the RAM array."""

    def __init__(self, ram: np.ndarray) -> None:
        self._ram = ram
        self.calls: list[tuple[int, int]] = []

    def assign(self, addr: int, _fmt: str, val: int) -> None:
        self.calls.append((int(addr), int(val)))
        self._ram[int(addr)] = int(val) & 0xFF


def _poke_env(ram: np.ndarray) -> tuple[SimpleNamespace, _WriteThroughMem]:
    mem = _WriteThroughMem(ram)
    env = SimpleNamespace(
        get_ram=lambda: ram,
        unwrapped=SimpleNamespace(data=SimpleNamespace(memory=mem)),
    )
    return env, mem


def _measured_leave_ram() -> np.ndarray:
    """RAM matching MEASURED_POST_L6_EXIT byte-for-byte (l7p1_l6exit.json)."""
    return _ram(
        screen=SCREEN_LEVEL6_ENTRANCE,
        x=112,
        y=125,
        triforce=0x3F,
        keys=2,
        bombs=8,
        arrows=1,
        health=0x77,  # 8 containers, full
        whistle=1,
        food=0,
        rod=1,
        bow=1,
        candle=0,
        rupees=42,
        selected=2,
    )


def test_measured_post_l6_exit_is_a_verified_shared_handoff() -> None:
    h = MEASURED_POST_L6_EXIT
    assert isinstance(h, OverworldHandoff)
    assert h.screen == SCREEN_LEVEL6_ENTRANCE
    assert (h.link_x, h.link_y) == (112, 125)
    assert h.triforce == POST_L6_TRIFORCE == 0x3F
    assert h.food == 0
    assert h.selected_item == 2  # arrows, from the Gohma kill
    assert h.verified is True
    assert h.route_eligible is False  # pond/shop route past 0x25 still unobserved
    assert h.complete() is True
    ram = _measured_leave_ram()
    assert h.mismatch(read_snapshot(ram), ram) is None
    # The fixture-live bait prefix is wired in as the default hops.
    ctl = make_post_l6_overworld_controller()
    assert ctl.hops == POST_L6_TO_BAIT_HOPS


def test_unmeasured_handoff_refuses_to_move() -> None:
    ram = _ram()
    ctl = make_post_l6_overworld_controller()
    ctl.bind_env(_env(ram))
    act = ctl.step(read_snapshot(ram))
    assert ctl.failed
    assert not ctl.success
    assert act.reason == "handoff_unmeasured"
    assert list(act.action) == list(nes_idle_action())
    assert UNMEASURED_HANDOFF.route_eligible is False
    assert UNMEASURED_HANDOFF.verified is False
    assert UNMEASURED_HANDOFF.screen is None
    assert ctl.report()["route_eligible"] is False
    assert ctl.report()["writes"] == 0


def test_measured_exit_verifies_and_does_not_refuse_on_the_mouth_tile() -> None:
    """The measured leave stands on the 0x22 mouth tile; re-entry refusal only
    arms after Link steps off it, so frame 1 walks (does not fail closed)."""
    ram = _measured_leave_ram()
    ctl = make_post_l6_overworld_controller(handoff=MEASURED_POST_L6_EXIT)
    ctl.bind_env(_env(ram))
    act = ctl.step(read_snapshot(ram))
    assert not ctl.failed
    assert act.reason != "l6_cave_mouth"
    assert "post_l6_handoff_accepted" in ctl.notes
    # Once off the mouth, walking back onto it is refused.
    ctl._left_mouth = True
    act = ctl.step(read_snapshot(ram))
    assert ctl.failed
    assert act.reason == "l6_cave_mouth_reentry"


def test_recovered_l6_prefix_is_not_an_l7_start() -> None:
    """Play 0x09 (56,109) TF 0x1F Rod=0 is the L6 residual, not L7."""
    ram = _ram(screen=0x09, x=56, y=109, triforce=0x1F, rod=0, bow=1)
    fake = OverworldHandoff(
        screen=0x09,
        link_x=56,
        link_y=109,
        mode=PLAY_MODE,
        triforce=POST_L6_TRIFORCE,
        keys=3,
        bombs=8,
        rupees=20,
        heart_containers=12,
        selected_item=0,
        whistle=1,
        food=0,
        rod=0,
        bow=1,
        arrows=1,
        candle=1,
        verified=True,
        route_eligible=False,
    )
    assert fake.complete()
    assert fake.mismatch(read_snapshot(ram), ram) == "handoff_triforce_mismatch"
    assert POST_L6_TRIFORCE == 0x3F


def test_verified_handoff_without_hops_still_refuses() -> None:
    ram = _ram(
        screen=0x42,
        x=120,
        y=125,
        triforce=POST_L6_TRIFORCE,
        rod=1,
        bow=1,
        whistle=1,
        rupees=60,
    )
    handoff = OverworldHandoff(
        screen=0x42,
        link_x=120,
        link_y=125,
        mode=PLAY_MODE,
        triforce=POST_L6_TRIFORCE,
        keys=3,
        bombs=8,
        rupees=60,
        heart_containers=12,
        selected_item=0,
        whistle=1,
        food=0,
        rod=1,
        bow=1,
        arrows=1,
        candle=1,
        verified=True,
        evidence="fixture-live",
        route_eligible=False,
    )
    assert handoff.mismatch(read_snapshot(ram), ram) is None
    ctl = make_post_l6_overworld_controller(handoff=handoff, hops=())
    ctl.bind_env(_env(ram))
    act = ctl.step(read_snapshot(ram))
    assert ctl.failed
    assert act.reason == "post_l6_path_unmeasured"
    assert handoff.route_eligible is False


def test_bait_plan_fails_closed_without_60r_or_shop_geometry() -> None:
    assert BAIT_SHOP_SCREEN_HYP == 0x34
    ram = _ram(rupees=20, food=0)
    ctl = make_bait_purchase_controller()
    ctl.bind_env(_env(ram))
    act = ctl.step(read_snapshot(ram))
    assert ctl.failed
    assert act.reason == "bait_need_60_rupees"
    assert ctl.report()["writes"] == 0
    ram[ADDR_RUPEES] = BAIT_COST
    ctl = make_bait_purchase_controller()
    ctl.bind_env(_env(ram))
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "bait_shop_geometry_unobserved"


def test_survival_bait_controller_pokes_food_and_succeeds() -> None:
    ram = _ram(food=0, rupees=42)
    env, mem = _poke_env(ram)
    ctl = make_survival_bait_purchase_controller()
    ctl.bind_env(env)
    act = ctl.step(read_snapshot(ram))
    assert ctl.success and not ctl.failed
    assert act.reason == "survival_bait_food_fixture"
    assert ram[ADDR_FOOD] == 1
    # The ONLY write is ADDR_FOOD -> 1. No rupee / selected / door / TF write.
    assert mem.calls == [(ADDR_FOOD, 1)]
    report = ctl.report()
    assert report["writes"] == 1
    assert report["progression_writes"] == 0
    assert report["capacity_writes"] == 0
    assert report["route_eligible"] is False
    assert report["spec_id"] == "level7_bait_purchase"
    assert ram[ADDR_RUPEES] == 42  # untouched


def test_survival_bait_controller_no_write_when_food_already_owned() -> None:
    ram = _ram(food=1)
    env, mem = _poke_env(ram)
    ctl = make_survival_bait_purchase_controller()
    ctl.bind_env(env)
    ctl.step(read_snapshot(ram))
    assert ctl.success and not ctl.failed
    assert mem.calls == []
    assert ctl.report()["writes"] == 0


def test_survival_bait_controller_fails_closed_without_env() -> None:
    ctl = make_survival_bait_purchase_controller()
    act = ctl.step(read_snapshot(_ram()))
    assert ctl.failed and not ctl.success
    assert act.reason == "survival_bait_env_not_bound"


def test_natural_bait_stays_fail_closed_and_survival_is_opt_in() -> None:
    clean = level7_entry_chapter_stages()
    survival = level7_entry_chapter_stages(survival=True)
    assert isinstance(clean[1][1], NaturalBaitPurchaseController)
    assert isinstance(survival[1][1], SurvivalBaitPurchaseController)
    # Stage names are identical either way.
    assert [n for n, _c, _f in clean] == [n for n, _c, _f in survival]


def test_l7_hops_survival_swaps_only_the_bait_stage() -> None:
    ram = _ram()
    stages = l7_hops(_env(ram), survival=True)[0].stages()
    names = [n for n, _c, _f in stages]
    assert names == [
        "level7_post_l6_overworld",
        "level7_bait_purchase",
        "level7_pond_drain_entry",
    ]
    assert isinstance(stages[1][1], SurvivalBaitPurchaseController)
    # pond stage still fail-closed (unverified path controller)
    pond = stages[2][1]
    assert not isinstance(pond, SurvivalBaitPurchaseController)
    pond.step(read_snapshot(ram))
    assert pond.failed


def test_continue_level7_spine_uses_the_survival_bait_fixture() -> None:
    import inspect

    from zelda_i.level7 import spine

    src = inspect.getsource(spine.continue_level7_spine)
    assert "survival=True" in src


def test_hops_docstring_matches_live_facts() -> None:
    from zelda_i.level7 import hops as hops_mod

    doc = hops_mod.__doc__ or ""
    assert "verified=False" not in doc
    assert "every factory here fails closed" not in doc
    assert "MEASURED_POST_L6_EXIT" in doc
    assert "0x79" in doc
    assert "rr-8t4.4" in doc


def test_pond_missing_evidence_is_natural_whistle_drain() -> None:
    ctl = make_pond_entry_controller()
    missing = str(ctl.report()["missing_evidence"])
    assert "natural-whistle" in missing
    assert "rr-8t4.4" in missing
    assert "observed entry room" not in missing
    assert "live pond screen" not in missing


def test_red_candle_chapter_starts_at_entry_first_door() -> None:
    stages = level7_red_candle_chapter_stages()
    names = [name for name, _c, _f in stages]
    assert names == [
        "level7_entry_first_door",
        "level7_entry_to_hungry_goriya",
        "level7_tip_of_nose_stairs",
        "level7_red_candle_pickup",
    ]
    first = stages[0][1]
    assert isinstance(first, EntryNorthDoorController)
    assert first.stage_id == make_entry_first_door_controller().stage_id
    report = first.report()
    assert report["route_eligible"] is False
    assert report["dest_screen"] == 0x69
    assert report["door"] == "UP"


def test_north_door_79_step_south_mouth_and_door_x() -> None:
    mouth = read_snapshot(_ram(level=7, screen=0x79, x=NORTH_DOOR_X, y=SOUTH_MOUTH_Y))
    act = north_door_79_step(mouth)
    assert list(act.action) != list(nes_idle_action())
    assert act.reason == "north_leave_mouth"
    door = read_snapshot(_ram(level=7, screen=0x79, x=NORTH_DOOR_X, y=NORTH_DOOR_Y))
    act = north_door_79_step(door)
    assert act.reason == "north_push"
    off = read_snapshot(_ram(level=7, screen=0x79, x=NORTH_DOOR_X + 8, y=NORTH_DOOR_Y))
    act = north_door_79_step(off)
    assert act.reason == "north_align_x"
    dest = read_snapshot(_ram(level=7, screen=0x69, x=NORTH_DOOR_X, y=SOUTH_MOUTH_Y))
    ctl = EntryNorthDoorController()
    act = ctl.step(dest)
    assert ctl.success and not ctl.failed
    assert act.reason == "left_0x79"


def test_hungry_goriya_requires_food() -> None:
    ram = _ram(level=7, screen=0x10, food=0)
    ctl = make_entry_to_goriya_controller()
    ctl.bind_env(_env(ram))
    act = ctl.step(read_snapshot(ram))
    assert ctl.failed
    assert act.reason == "hungry_goriya_requires_food"
    assert any("net_hyp" in note for note in ctl.notes)
    ram[ADDR_FOOD] = 1
    ctl = make_entry_to_goriya_controller()
    ctl.bind_env(_env(ram))
    act = ctl.step(read_snapshot(ram))
    assert ctl.failed
    assert "blocked_unverified" in act.reason


def test_red_candle_does_not_write_and_fails_closed() -> None:
    ram = _ram(level=7, candle=1)
    ctl = make_red_candle_controller()
    ctl.bind_env(_env(ram))
    act = ctl.step(read_snapshot(ram))
    assert ctl.failed
    assert act.reason.startswith("red_candle_still_1")
    assert ctl.report()["writes"] == 0
    ram[ADDR_CANDLE] = 2
    ctl = make_red_candle_controller()
    ctl.bind_env(_env(ram))
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "red_candle_room_unobserved"


def test_l7_hops_use_fail_closed_entry_chapter() -> None:
    ram = _ram()
    hops = l7_hops(_env(ram))
    assert tuple(h.through for h in hops) == (
        "level7-entry",
        "level7-red-candle",
        "level7",
    )
    stages_fn = hops[0].stages
    assert callable(stages_fn)
    stages = stages_fn()
    assert [name for name, _c, _n in stages] == [
        "level7_post_l6_overworld",
        "level7_bait_purchase",
        "level7_pond_drain_entry",
    ]
    post = stages[0][1]
    post.bind_env(_env(ram))
    post.step(read_snapshot(ram))
    assert post.failed

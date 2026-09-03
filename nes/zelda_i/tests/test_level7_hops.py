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
    level7_complete_chapter_stages,
    level7_entry_chapter_stages,
    level7_red_candle_chapter_stages,
    make_aquamentus_heart_controller,
    make_entry_first_door_controller,
    make_entry_to_goriya_controller,
    make_forced_digdogger_controller,
    make_level7_shard_leave_controller,
    make_pond_entry_controller,
    make_red_candle_controller,
    make_room58_north_controller,
    make_room68_down_controller,
    make_room08_east_bomb_controller,
    make_room09_down_controller,
    make_room18_north_bomb_controller,
    make_room19_east_bomb_controller,
    make_room1a_east_bomb_controller,
    make_room1a_candle_controller,
    make_room4a_return_controller,
    make_room1b_key_east_controller,
    make_room38_up_controller,
    make_room39_left_controller,
    make_room49_up_controller,
)
from zelda_i.level7.path import (
    EAST_APPROACH_X,
    EAST_BAND_Y,
    EAST_DOOR_X,
    EAST_DOOR_Y,
    NORTH_DOOR_X,
    NORTH_DOOR_Y,
    ROOM_6A_DOOR_Y,
    ROOM_6A_EAST_COLUMN_X,
    ROOM_6A_EAST_PLANE,
    ROOM_6A_TOP_BAND_Y,
    ROOM_58_NORTH_EAST_X,
    ROOM_58_NORTH_MID_Y,
    ROOM_58_NORTH_TOP_Y,
    ROOM_58_NORTH_X,
    ROOM_68_MID_Y,
    ROOM_68_SAFE_X,
    ROOM_68_SOUTH_X,
    ROOM_68_TRAP_ROW_Y,
    SOUTH_MOUTH_Y,
    EntryNorthDoorController,
    Room6AEastController,
    Room58NorthController,
    Room68DownController,
    L7_ROOM08_EAST_BOMB,
    L7_ROOM18_NORTH_BOMB,
    L7_ROOM19_EAST_BOMB,
    L7_ROOM1A_EAST_BOMB,
    Room09DownController,
    Room1ACandleController,
    Room4AReturnController,
    Room1BKeyEastController,
    Room38UpController,
    Room39LeftController,
    Room49UpController,
    Room69EastController,
    north_door_79_step,
    room69_east_step,
    room_58_north_step,
    room_68_down_step,
    room_6a_east_step,
    room_09_down_step,
    room_38_up_step,
    room_39_left_step,
    room_49_up_step,
    room_4a_return_step,
    room_1b_key_east_step,
)
from zelda_i.level7.overworld import POST_L6_TO_BAIT_HOPS
from zelda_i.overworld.stitch import UNMEASURED_HANDOFF, OverworldHandoff
from retro_harness.nes import nes_idle_action
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_BOW,
    ADDR_CANDLE,
    ADDR_CUR_OPENED_DOORS,
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
    ram[ADDR_CUR_OPENED_DOORS] = fields.get("doors", 0)
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


def test_room69_east_south_mouth_stands_until_spawn() -> None:
    mouth = read_snapshot(_ram(level=7, screen=0x69, x=NORTH_DOOR_X, y=SOUTH_MOUTH_Y))
    act = room69_east_step(mouth)
    assert list(act.action) == list(nes_idle_action())
    assert act.reason == "spawn_wait"
    ctl = Room69EastController()
    act = ctl.step(mouth)
    assert not ctl.success and not ctl.failed
    assert act.reason == "spawn_wait"


def test_room69_east_door_band_pushes_without_door_bit() -> None:
    """0x69 east is an OPEN doorway: never wait on ``cur_opened_doors``."""
    snap = read_snapshot(_ram(level=7, screen=0x69, x=EAST_DOOR_X, y=EAST_DOOR_Y))
    act = room69_east_step(snap, saw_goriya=True)
    assert act.reason == "east_push"
    ctl = Room69EastController()
    ctl.saw_goriya = True
    act = ctl.step(snap)
    assert not ctl.success and not ctl.failed
    assert act.reason == "east_push"
    assert ctl.east_opened_frame is None


def test_room69_east_crosses_on_the_north_band_not_the_centre_row() -> None:
    """The 0x69 centre row is walled: rise to y=109 before heading east."""
    snap = read_snapshot(_ram(level=7, screen=0x69, x=120, y=EAST_DOOR_Y))
    assert room69_east_step(snap, saw_goriya=True).reason == "east_band_y"
    band = read_snapshot(_ram(level=7, screen=0x69, x=120, y=EAST_BAND_Y))
    assert room69_east_step(band, saw_goriya=True).reason == "east_band_x"
    column = read_snapshot(_ram(level=7, screen=0x69, x=EAST_APPROACH_X, y=EAST_BAND_Y))
    assert room69_east_step(column, saw_goriya=True).reason == "east_door_y"


def test_room69_east_arrived_leaves_0x69() -> None:
    dest = read_snapshot(_ram(level=7, screen=0x6A, x=16, y=EAST_DOOR_Y))
    ctl = Room69EastController()
    act = ctl.step(dest)
    assert ctl.success and not ctl.failed
    assert act.reason == "left_0x69"


def test_room_6a_east_crosses_the_top_band_not_the_centre_row() -> None:
    """0x6A centre band walls at x=48: rise to y=93 and cross the top."""
    mouth = read_snapshot(_ram(level=7, screen=0x6A, x=16, y=ROOM_6A_DOOR_Y))
    assert room_6a_east_step(mouth).reason == "east6a_leave_mouth"
    west = read_snapshot(_ram(level=7, screen=0x6A, x=48, y=ROOM_6A_DOOR_Y))
    assert room_6a_east_step(west).reason == "east6a_rise"
    band = read_snapshot(_ram(level=7, screen=0x6A, x=48, y=ROOM_6A_TOP_BAND_Y))
    assert room_6a_east_step(band).reason == "east6a_cross"
    column = read_snapshot(
        _ram(level=7, screen=0x6A, x=ROOM_6A_EAST_COLUMN_X, y=ROOM_6A_TOP_BAND_Y)
    )
    assert room_6a_east_step(column).reason == "east6a_drop_y"
    door = read_snapshot(
        _ram(level=7, screen=0x6A, x=ROOM_6A_EAST_PLANE, y=ROOM_6A_DOOR_Y)
    )
    assert room_6a_east_step(door).reason == "east6a_push"


def test_room_6a_east_needs_no_candle_and_no_door_bit() -> None:
    """The room is unlit and the east exit is OPEN: movement waits on neither."""
    snap = read_snapshot(_ram(level=7, screen=0x6A, x=ROOM_6A_EAST_PLANE, y=ROOM_6A_DOOR_Y))
    ctl = Room6AEastController()
    act = ctl.step(snap)
    assert not ctl.success and not ctl.failed
    assert act.reason == "east6a_push"


def test_room_6a_east_arrived_leaves_0x6a() -> None:
    dest = read_snapshot(_ram(level=7, screen=0x6B, x=16, y=ROOM_6A_DOOR_Y))
    ctl = Room6AEastController()
    act = ctl.step(dest)
    assert ctl.success and not ctl.failed
    assert act.reason == "left_0x6a"
    report = ctl.report()
    assert report["route_eligible"] is False
    assert report["evidence"] == "fixture-live"


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


def test_complete_chapter_factories_fail_closed_and_do_not_invent_leave() -> None:
    """L7-C stages stay unverified. Screen None; handoff verified stays False."""
    stages = level7_complete_chapter_stages()
    names = [name for name, _c, _f in stages]
    assert names == [
        "level7_forced_digdogger",
        "level7_aquamentus_heart",
        "level7_shard_and_settled_leave",
    ]
    ram = _ram(level=7, screen=0x10, candle=2, whistle=1, food=0, triforce=0x3F)
    snap = read_snapshot(ram)
    forced = make_forced_digdogger_controller()
    aqua = make_aquamentus_heart_controller()
    leave = make_level7_shard_leave_controller()
    for ctl, needle in (
        (forced, "Whistle B-slot 5"),
        (aqua, "Level1AquamentusController"),
        (leave, "MEASURED_POST_L7_EXIT"),
    ):
        report = ctl.report()
        assert report["route_eligible"] is False
        assert report["evidence"] == "hypothesis"
        assert needle in str(report["missing_evidence"])
        act = ctl.step(snap)
        assert ctl.failed and not ctl.success
        assert act.reason == "blocked_unverified"
    assert UNMEASURED_HANDOFF.verified is False
    assert UNMEASURED_HANDOFF.screen is None
    assert UNMEASURED_HANDOFF.complete() is False
    # L6 packet is the schema template; L7 has no filled packet yet.
    h = MEASURED_POST_L6_EXIT
    for name in (
        "screen",
        "link_x",
        "link_y",
        "mode",
        "triforce",
        "keys",
        "bombs",
        "rupees",
        "heart_containers",
        "selected_item",
        "whistle",
        "food",
        "rod",
        "bow",
        "arrows",
        "candle",
    ):
        assert getattr(h, name) is not None


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


def test_room_68_down_peels_then_drops_then_pushes() -> None:
    """0x68 south: off the east trap column, between trap rows, then DOWN."""
    ne = read_snapshot(_ram(level=7, screen=0x68, x=208, y=93))
    assert room_68_down_step(ne).reason == "south68_peel"
    safe = read_snapshot(_ram(level=7, screen=0x68, x=ROOM_68_SAFE_X, y=93))
    assert room_68_down_step(safe).reason == "south68_drop"
    mid = read_snapshot(
        _ram(level=7, screen=0x68, x=ROOM_68_SAFE_X, y=ROOM_68_MID_Y)
    )
    assert room_68_down_step(mid).reason == "south68_align_x"
    door = read_snapshot(
        _ram(level=7, screen=0x68, x=ROOM_68_SOUTH_X, y=ROOM_68_MID_Y)
    )
    assert room_68_down_step(door).reason == "south68_push"
    trap = read_snapshot(
        _ram(level=7, screen=0x68, x=80, y=ROOM_68_TRAP_ROW_Y)
    )
    assert room_68_down_step(trap).reason == "south68_off_trap"


def test_room_68_down_arrived_leaves_0x68() -> None:
    dest = read_snapshot(_ram(level=7, screen=0x78, x=120, y=77))
    ctl = Room68DownController()
    act = ctl.step(dest)
    assert ctl.success and not ctl.failed
    assert act.reason == "left_0x68_south"
    report = make_room68_down_controller().report()
    assert report["route_eligible"] is False
    assert report["dest_screen"] == 0x78
    assert report["door"] == "DOWN"


def test_room_58_north_east_around_then_push() -> None:
    """0x58 north: climb, east around the central mass, then x=120 UP."""
    mouth = read_snapshot(_ram(level=7, screen=0x58, x=120, y=205))
    assert room_58_north_step(mouth).reason == "north58_climb"
    mid = read_snapshot(
        _ram(level=7, screen=0x58, x=120, y=ROOM_58_NORTH_MID_Y)
    )
    assert room_58_north_step(mid).reason == "north58_east"
    east = read_snapshot(
        _ram(level=7, screen=0x58, x=ROOM_58_NORTH_EAST_X, y=ROOM_58_NORTH_MID_Y)
    )
    assert room_58_north_step(east).reason == "north58_rise"
    top = read_snapshot(
        _ram(level=7, screen=0x58, x=ROOM_58_NORTH_EAST_X, y=ROOM_58_NORTH_TOP_Y)
    )
    assert room_58_north_step(top).reason == "north58_align_x"
    door = read_snapshot(
        _ram(level=7, screen=0x58, x=ROOM_58_NORTH_X, y=ROOM_58_NORTH_TOP_Y)
    )
    assert room_58_north_step(door).reason == "north58_push"


def test_room_58_north_arrived_leaves_0x58() -> None:
    dest = read_snapshot(_ram(level=7, screen=0x48, x=120, y=205))
    ctl = Room58NorthController()
    act = ctl.step(dest)
    assert ctl.success and not ctl.failed
    assert act.reason == "left_0x58_north"
    report = make_room58_north_controller().report()
    assert report["route_eligible"] is False
    assert report["dest_screen"] == 0x48
    assert report["door"] == "UP"


def test_room_49_up_south_mouth_stands_until_spawn() -> None:
    mouth = read_snapshot(_ram(level=7, screen=0x49, x=120, y=SOUTH_MOUTH_Y))
    assert room_49_up_step(mouth).reason == "spawn_wait"
    ctl = Room49UpController()
    act = ctl.step(mouth)
    assert not ctl.success and not ctl.failed
    assert act.reason == "spawn_wait"


def test_room_49_up_aligns_x_then_pushes_north() -> None:
    """Stepladder moat: never strafe on water; align on land, then hold UP."""
    water = read_snapshot(_ram(level=7, screen=0x49, x=64, y=117))
    assert room_49_up_step(water, saw_goriya=True).reason == "up49_cross"
    south = read_snapshot(_ram(level=7, screen=0x49, x=80, y=180))
    assert room_49_up_step(south, saw_goriya=True).reason == "up49_south_align"
    off = read_snapshot(_ram(level=7, screen=0x49, x=196, y=109))
    assert room_49_up_step(off, saw_goriya=True).reason == "up49_align_x"
    door = read_snapshot(_ram(level=7, screen=0x49, x=120, y=93))
    assert room_49_up_step(door, saw_goriya=True).reason == "up49_push"
    ctl = Room49UpController()
    ctl.saw_goriya = True
    act = ctl.step(door)
    assert not ctl.success and not ctl.failed
    assert act.reason == "up49_push"


def test_room_49_up_arrived_leaves_0x49() -> None:
    dest = read_snapshot(_ram(level=7, screen=0x39, x=120, y=SOUTH_MOUTH_Y))
    ctl = Room49UpController()
    act = ctl.step(dest)
    assert ctl.success and not ctl.failed
    assert act.reason == "left_0x49_north"
    report = ctl.report()
    assert report["route_eligible"] is False
    assert report["evidence"] == "fixture-live"
    assert report["dest_screen"] == 0x39
    assert report["door"] == "UP"
    factory = make_room49_up_controller()
    assert factory.report()["dest_screen"] == 0x39
    assert "level7_room49_up" not in [
        name for name, _c, _f in level7_red_candle_chapter_stages()
    ]


def test_room_39_left_rises_centre_column_not_sw_statue() -> None:
    """SW statue boxes (48,189): stay x=120 until y=141, then LEFT."""
    mouth = read_snapshot(_ram(level=7, screen=0x39, x=120, y=SOUTH_MOUTH_Y))
    assert room_39_left_step(mouth).reason == "left39_rise"
    sw = read_snapshot(_ram(level=7, screen=0x39, x=48, y=189))
    assert room_39_left_step(sw).reason == "left39_center"
    door = read_snapshot(_ram(level=7, screen=0x39, x=16, y=141))
    assert room_39_left_step(door).reason == "left39_push"
    dest = read_snapshot(_ram(level=7, screen=0x38, x=208, y=141))
    ctl = Room39LeftController()
    act = ctl.step(dest)
    assert ctl.success and not ctl.failed
    assert act.reason == "left_0x39_west"
    assert ctl.report()["dest_screen"] == 0x38
    assert ctl.report()["route_eligible"] is False
    assert make_room39_left_controller().report()["dest_screen"] == 0x38
    assert "level7_room39_left" not in [
        name for name, _c, _f in level7_red_candle_chapter_stages()
    ]


def test_room_38_up_uses_east_pocket_not_centre_diamonds() -> None:
    """y=149 diamond row blocks centre UP; recollect x=208 then rise."""
    mid = read_snapshot(_ram(level=7, screen=0x38, x=110, y=149))
    assert room_38_up_step(mid, saw_goriya=True).reason == "up38_pocket"
    pocket = read_snapshot(_ram(level=7, screen=0x38, x=208, y=149))
    assert room_38_up_step(pocket, saw_goriya=True).reason == "up38_rise"
    door = read_snapshot(_ram(level=7, screen=0x38, x=120, y=93))
    assert room_38_up_step(door, saw_goriya=True).reason == "up38_push"
    dest = read_snapshot(_ram(level=7, screen=0x28, x=120, y=SOUTH_MOUTH_Y))
    ctl = Room38UpController()
    act = ctl.step(dest)
    assert ctl.success and not ctl.failed
    assert act.reason == "left_0x38_north"
    assert ctl.report()["dest_screen"] == 0x28
    assert ctl.report()["route_eligible"] is False
    assert make_room38_up_controller().report()["dest_screen"] == 0x28
    assert "level7_room38_up" not in [
        name for name, _c, _f in level7_red_candle_chapter_stages()
    ]


def test_room_09_down_drops_then_pushes_after_clear() -> None:
    """0x09 south shutter: kill-clear, y=189, x=120, DOWN. Not OPEN on spawn."""
    mouth = read_snapshot(_ram(level=7, screen=0x09, x=32, y=141))
    assert room_09_down_step(mouth).reason == "spawn_wait"
    drop = read_snapshot(_ram(level=7, screen=0x09, x=32, y=141))
    assert room_09_down_step(drop, saw_goriya=True).reason == "down09_drop"
    south = read_snapshot(_ram(level=7, screen=0x09, x=32, y=189))
    assert room_09_down_step(south, saw_goriya=True).reason == "down09_align_x"
    door = read_snapshot(_ram(level=7, screen=0x09, x=120, y=189))
    assert room_09_down_step(door, saw_goriya=True).reason == "down09_push"
    dest = read_snapshot(_ram(level=7, screen=0x19, x=120, y=93))
    ctl = Room09DownController()
    act = ctl.step(dest)
    assert ctl.success and not ctl.failed
    assert act.reason == "left_0x09_south"
    assert ctl.report()["dest_screen"] == 0x19
    assert ctl.report()["route_eligible"] is False
    assert make_room09_down_controller().report()["dest_screen"] == 0x19
    assert "level7_room09_down" not in [
        name for name, _c, _f in level7_red_candle_chapter_stages()
    ]


def test_map_bomb_chain_factories_are_recon_only() -> None:
    """0x18/0x08/0x19 bomb walls + 0x09 down stay off the chapter chain."""
    n18 = make_room18_north_bomb_controller()
    assert n18.wall is L7_ROOM18_NORTH_BOMB
    assert n18.to_room == 0x08
    assert n18.stand == (120, 93)
    e08 = make_room08_east_bomb_controller()
    assert e08.wall is L7_ROOM08_EAST_BOMB
    assert e08.to_room == 0x09
    assert e08.approach_waypoints[0] == (200, 189)
    e19 = make_room19_east_bomb_controller()
    assert e19.wall is L7_ROOM19_EAST_BOMB
    assert e19.to_room == 0x1A
    e1a = make_room1a_east_bomb_controller()
    assert e1a.wall is L7_ROOM1A_EAST_BOMB
    assert e1a.to_room == 0x1B
    assert e1a.approach_waypoints[0] == (96, 189)
    names = [name for name, _c, _f in level7_red_candle_chapter_stages()]
    assert "level7_room18_north_bomb" not in names
    assert "level7_room08_east_bomb" not in names
    assert "level7_room19_east_bomb" not in names
    assert "level7_room1a_east_bomb" not in names


def test_room_1a_candle_is_recon_only_and_arrives_on_candle_2() -> None:
    """Natural ADDR_CANDLE 0→2; not on the executable chapter chain."""
    mouth = read_snapshot(_ram(level=7, screen=0x1A, x=32, y=141))
    assert Room1ACandleController().step(mouth).reason == "spawn_wait"
    dest = read_snapshot(_ram(level=7, screen=0x4A, x=135, y=141, mode=9, candle=2))
    ctl = Room1ACandleController()
    act = ctl.step(dest)
    assert ctl.success and not ctl.failed
    assert act.reason == "red_candle_natural"
    factory = make_room1a_candle_controller()
    assert factory.report()["dest_screen"] == 0x4A
    assert factory.report()["route_eligible"] is False
    assert "level7_room1a_candle" not in [
        name for name, _c, _f in level7_red_candle_chapter_stages()
    ]
    assert "level7_red_candle_pickup" in [
        name for name, _c, _f in level7_red_candle_chapter_stages()
    ]


def test_room_4a_return_step_east_drop_then_west_ladder() -> None:
    """Pad RIGHT, east LEFT+DOWN, floor LEFT, west-ladder UP. Not on chapter."""
    pad = read_snapshot(_ram(level=7, screen=0x4A, x=136, y=141, mode=9, candle=2))
    assert room_4a_return_step(pad).reason == "cellar_to_east"
    east = read_snapshot(_ram(level=7, screen=0x4A, x=192, y=141, mode=9, candle=2))
    assert room_4a_return_step(east).reason == "cellar_east_drop"
    floor = read_snapshot(_ram(level=7, screen=0x4A, x=100, y=189, mode=9, candle=2))
    assert room_4a_return_step(floor).reason == "cellar_floor_west"
    west = read_snapshot(_ram(level=7, screen=0x4A, x=48, y=189, mode=9, candle=2))
    assert room_4a_return_step(west).reason == "cellar_west_climb"
    climb = read_snapshot(_ram(level=7, screen=0x4A, x=48, y=120, mode=9, candle=2))
    assert room_4a_return_step(climb).reason == "cellar_west_up"
    dest = read_snapshot(_ram(level=7, screen=0x1A, x=96, y=157, candle=2))
    ctl = Room4AReturnController()
    act = ctl.step(dest)
    assert ctl.success and not ctl.failed
    assert act.reason == "left_0x4a_stairs"
    factory = make_room4a_return_controller()
    assert factory.report()["dest_screen"] == 0x1A
    assert factory.report()["route_eligible"] is False
    assert "level7_room4a_return" not in [
        name for name, _c, _f in level7_red_candle_chapter_stages()
    ]
    assert "level7_room4a_return" not in [
        name for name, _c, _f in level7_complete_chapter_stages()
    ]


def test_room_1b_key_east_is_recon_only_and_arrives_on_0x1c() -> None:
    """y=141 RIGHT spends a key; dest 0x1C. Not on the chapter chain."""
    mouth = read_snapshot(_ram(level=7, screen=0x1B, x=32, y=141, candle=2))
    assert room_1b_key_east_step(mouth).reason == "keyeast_approach"
    dest = read_snapshot(_ram(level=7, screen=0x1C, x=16, y=141, candle=2))
    ctl = Room1BKeyEastController()
    act = ctl.step(dest)
    assert ctl.success and not ctl.failed
    assert act.reason == "left_0x1b_east"
    factory = make_room1b_key_east_controller()
    assert factory.report()["dest_screen"] == 0x1C
    assert factory.report()["route_eligible"] is False
    assert "level7_room1b_key_east" not in [
        name for name, _c, _f in level7_complete_chapter_stages()
    ]

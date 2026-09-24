"""Fail-closed L7 hops: OverworldHandoff gate, bait, Hungry Goriya, Red Candle."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from zelda_i.level7.dungeon import POST_L7_ARROW_RUPEES
from zelda_i.anchors import SCREEN_LEVEL6_ENTRANCE
from zelda_i.level7.entry import (
    BAIT_COST,
    BAIT_SHOP_SCREEN_HYP,
    MEASURED_POST_L6_EXIT,
    POST_L6_TRIFORCE,
    UNVERIFIED_BAIT_PLAN,
    NaturalBaitPurchaseController,
    SurvivalBaitPurchaseController,
    make_bait_purchase_controller,
    make_post_l6_overworld_controller,
    make_survival_bait_purchase_controller,
)
from zelda_i.level7.hops import (
    l7_hops,
    level7_bait_shop_chapter_stages,
    level7_complete_chapter_stages,
    level7_entry_chapter_stages,
    level7_red_candle_chapter_stages,
    make_aquamentus_heart_controller,
    make_entry_first_door_controller,
    make_entry_to_goriya_controller,
    make_forced_digdogger_controller,
    make_level7_shard_leave_controller,
    make_red_candle_controller,
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
    Room09DownController,
    Room1ACandleController,
    Room4AReturnController,
    Room1BKeyEastController,
    Room0DClearController,
    room_0d_clear_step,
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
from zelda_i.level7.pond import POST_L6_TO_POND_HOPS
from zelda_i.overworld.stitch import UNMEASURED_HANDOFF, OverworldHandoff
from retro_harness.controls import pressed_nes_buttons
from retro_harness.nes import nes_idle_action
from zelda_i.ram import (
    ADDR_CANDLE,
    ADDR_FOOD,
    ADDR_RUPEES,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.tests.ram_helpers import make_ram

_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 0,
    "screen": 0x09,
    "x": 56,
    "y": 109,
    "sword": 1,
    "triforce": 0x1F,
    "keys": 3,
    "bombs": 8,
    "arrows": 1,
    "health": 0xBB,
    "whistle": 1,
    "food": 0,
    "rod": 0,
    "bow": 1,
    "candle": 1,
    "ladder": 1,
    "rupees": 20,
    "selected": 0,
    "doors": 0,
    "room_all_dead": 0,
}


def _ram(**fields: int) -> np.ndarray:
    return make_ram(_DEFAULTS, **fields)


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
    """RAM matching MEASURED_POST_L6_EXIT byte-for-byte (gathered spine, 2026-09-22)."""
    return _ram(
        screen=SCREEN_LEVEL6_ENTRANCE,
        x=112,
        y=125,
        triforce=0x3F,
        keys=2,
        bombs=7,
        arrows=1,
        health=0xAA,  # 11 containers, full
        whistle=1,
        food=1,
        rod=1,
        bow=1,
        candle=1,
        rupees=42,
        selected=2,
    )


def _press(act) -> list[str]:
    return pressed_nes_buttons(list(act.action))


def test_measured_post_l6_exit_is_a_verified_shared_handoff() -> None:
    h = MEASURED_POST_L6_EXIT
    assert isinstance(h, OverworldHandoff)
    assert h.screen == SCREEN_LEVEL6_ENTRANCE
    assert (h.link_x, h.link_y) == (112, 125)
    assert h.triforce == POST_L6_TRIFORCE == 0x3F
    assert h.food == 1
    assert h.selected_item == 2  # arrows, from the Gohma kill
    assert h.verified is True
    assert h.route_eligible is False  # walk to pond 0x42 still unobserved
    assert h.complete() is True
    ram = _measured_leave_ram()
    assert h.mismatch(read_snapshot(ram), ram) is None
    ctl = make_post_l6_overworld_controller()
    assert ctl.hops == POST_L6_TO_POND_HOPS


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
    assert act.reason != "post_l6_path_unmeasured"
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
    ctl = make_bait_purchase_controller(plan=UNVERIFIED_BAIT_PLAN)
    ctl.bind_env(_env(ram))
    act = ctl.step(read_snapshot(ram))
    assert ctl.failed
    assert act.reason == "bait_need_60_rupees"
    assert ctl.report()["writes"] == 0
    ram[ADDR_RUPEES] = BAIT_COST
    ctl = make_bait_purchase_controller(plan=UNVERIFIED_BAIT_PLAN)
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
    assert isinstance(dict((n, c) for n, c, _f in clean)["level7_bait_purchase"], NaturalBaitPurchaseController)
    assert isinstance(
        dict((n, c) for n, c, _f in survival)["level7_bait_purchase"], NaturalBaitPurchaseController
    )
    # Stage names are identical either way.
    assert [n for n, _c, _f in clean] == [n for n, _c, _f in survival]


def test_l7_hops_survival_swaps_only_the_bait_stage() -> None:
    ram = _ram()
    hops = l7_hops(_env(ram), survival=True)
    entry = next(h for h in hops if h.through == "level7-entry")
    stages = entry.stages()
    names = [n for n, _c, _f in stages]
    assert names == [
        "level7_post_l6_overworld",
        "level7_recorder_warp",
        "potion_restock_l6",
        "exit_potion_l6",
        "level7_pond_approach",
        "level7_bait_purchase",
        "level7_pond_drain_entry",
    ]
    by_name = {n: c for n, c, _f in stages}
    assert isinstance(by_name["level7_bait_purchase"], NaturalBaitPurchaseController)
    assert not isinstance(by_name["level7_bait_purchase"], SurvivalBaitPurchaseController)
    pond = by_name["level7_pond_drain_entry"]
    assert not isinstance(pond, SurvivalBaitPurchaseController)
    from zelda_i.level7.pond import Level7PondDrainController

    assert isinstance(pond, Level7PondDrainController)
    pond.step(read_snapshot(ram))
    assert pond.failed  # env not bound; drain still needs 0x42 + whistle


def test_red_candle_chapter_starts_at_entry_first_door() -> None:
    stages = level7_red_candle_chapter_stages()
    names = [name for name, _c, _f in stages]
    assert names == [
        "level7_entry_first_door",
        "level7_room69_west_bomb",
        "level7_room68_north",
        "level7_room58_east",
        "level7_room59_up",
        "level7_room49_up",
        "level7_room39_left",
        "level7_room38_up",
        "level7_entry_to_hungry_goriya",
        "level7_room18_north_bomb",
        "level7_room08_east_bomb",
        "level7_room09_down",
        "level7_room19_east_bomb",
        "level7_red_candle_pickup",
    ]
    first = stages[0][1]
    assert isinstance(first, EntryNorthDoorController)
    assert first.stage_id == make_entry_first_door_controller().stage_id


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

    def press(x: int, y: int) -> list[str]:
        snap = read_snapshot(_ram(level=7, screen=0x69, x=x, y=y))
        return _press(room69_east_step(snap, saw_goriya=True))

    assert press(120, EAST_DOOR_Y) == ["UP"]
    assert press(120, EAST_BAND_Y) == ["RIGHT"]
    assert press(EAST_APPROACH_X, EAST_BAND_Y) == ["DOWN"]


def test_room69_east_arrived_leaves_0x69() -> None:
    dest = read_snapshot(_ram(level=7, screen=0x6A, x=16, y=EAST_DOOR_Y))
    ctl = Room69EastController()
    act = ctl.step(dest)
    assert ctl.success and not ctl.failed
    assert act.reason == "left_0x69"


def test_room_6a_east_crosses_the_top_band_not_the_centre_row() -> None:
    """0x6A centre band walls at x=48: rise to y=93 and cross the top."""

    def press(x: int, y: int) -> list[str]:
        return _press(room_6a_east_step(read_snapshot(_ram(level=7, screen=0x6A, x=x, y=y))))

    assert press(16, ROOM_6A_DOOR_Y) == ["RIGHT"]
    assert press(48, ROOM_6A_DOOR_Y) == ["UP"]
    assert press(48, ROOM_6A_TOP_BAND_Y) == ["RIGHT"]
    assert press(ROOM_6A_EAST_COLUMN_X, ROOM_6A_TOP_BAND_Y) == ["DOWN"]
    assert press(ROOM_6A_EAST_PLANE, ROOM_6A_DOOR_Y) == ["RIGHT"]


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


def test_hungry_goriya_requires_food() -> None:
    ram = _ram(level=7, screen=0x28, food=0)
    ctl = make_entry_to_goriya_controller()
    ctl.bind_env(_env(ram))
    act = ctl.step(read_snapshot(ram))
    assert ctl.failed
    assert act.reason == "hungry_goriya_requires_food"
    leftover = ctl.report()["leftover"]
    assert leftover is not None
    assert leftover["reason"] == "hungry_goriya_requires_food"
    assert leftover["food"] == 0
    assert leftover["screen"] == 0x28
    ram[ADDR_FOOD] = 1
    ctl = make_entry_to_goriya_controller()
    ctl.bind_env(_env(ram))
    act = ctl.step(read_snapshot(ram))
    assert not ctl.failed
    assert ctl.report()["writes"] == 0


def test_complete_chapter_follows_live_tail_and_does_not_invent_leave() -> None:
    """L7-C stage factories stay fixture-live; the Survival packet is separate."""
    from zelda_i.level7.digdogger import Level7ForcedDigdoggerController
    from zelda_i.level7.dungeon import MEASURED_POST_L7_EXIT

    stages = level7_complete_chapter_stages()
    names = [name for name, _c, _f in stages]
    assert names == [
        "level7_room4a_return",
        "level7_room1a_east_bomb",
        "level7_room1b_key_east",
        "level7_forced_digdogger",
        "level7_room0c_east_bomb",
        "level7_room0d_clear",
        "level7_tip_of_nose_stairs",
        "level7_nose_cellar_cross",
        "level7_room29_east_bomb",
        "level7_aquamentus_heart",
        "level7_shard_and_settled_leave",
    ]
    forced = make_forced_digdogger_controller()
    assert isinstance(forced, Level7ForcedDigdoggerController)
    for ctl in (
        make_aquamentus_heart_controller(),
        make_level7_shard_leave_controller(),
    ):
        assert not ctl.success
    assert (
        make_level7_shard_leave_controller().report()[
            "measured_post_l7_exit_verified"
        ]
        is False
    )
    assert MEASURED_POST_L7_EXIT.verified is True
    assert MEASURED_POST_L7_EXIT.screen == 0x42
    assert MEASURED_POST_L7_EXIT.bombs == 0
    assert MEASURED_POST_L7_EXIT.rupees == POST_L7_ARROW_RUPEES
    assert MEASURED_POST_L7_EXIT.heart_containers == 12
    assert MEASURED_POST_L7_EXIT.complete() is True
    assert MEASURED_POST_L7_EXIT.route_eligible is True
    assert UNMEASURED_HANDOFF.verified is False
    assert UNMEASURED_HANDOFF.screen is None
    assert UNMEASURED_HANDOFF.complete() is False
    # Both L6 and L7 packets are filled schemas.
    for h in (MEASURED_POST_L6_EXIT, MEASURED_POST_L7_EXIT):
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


def test_bait_shop_chapter_has_no_food_poke() -> None:
    stages = level7_bait_shop_chapter_stages()
    names = [n for n, _c, _f in stages]
    assert names == [
        "level7_post_l6_overworld",
        "level7_recorder_warp",
        "potion_restock_l6",
        "exit_potion_l6",
        "level7_shop_approach",
    ]
    assert all(
        not isinstance(ctl, SurvivalBaitPurchaseController) for _n, ctl, _f in stages
    )
    shop = {n: c for n, c, _f in stages}["level7_shop_approach"]
    assert shop.hops[-1].target == BAIT_SHOP_SCREEN_HYP


def test_l7_hops_use_fail_closed_entry_chapter() -> None:
    ram = _ram()
    hops = l7_hops(_env(ram))
    assert tuple(h.through for h in hops) == (
        "level7-bait-shop",
        "level7-entry",
        "level7-red-candle",
        "level7",
    )
    assert hops[0].dedicated and hops[0].through == "level7-bait-shop"
    stages_fn = hops[1].stages
    assert callable(stages_fn)
    stages = stages_fn()
    assert [name for name, _c, _n in stages] == [
        "level7_post_l6_overworld",
        "level7_recorder_warp",
        "potion_restock_l6",
        "exit_potion_l6",
        "level7_pond_approach",
        "level7_bait_purchase",
        "level7_pond_drain_entry",
    ]
    post = stages[0][1]
    post.bind_env(_env(ram))
    post.step(read_snapshot(ram))
    assert post.failed


def test_room_68_down_peels_then_drops_then_pushes() -> None:
    """0x68 south: off the east trap column, between trap rows, then DOWN."""

    def press(x: int, y: int) -> list[str]:
        return _press(room_68_down_step(read_snapshot(_ram(level=7, screen=0x68, x=x, y=y))))

    assert press(208, 93) == ["LEFT"]
    assert press(ROOM_68_SAFE_X, 93) == ["DOWN"]
    assert press(ROOM_68_SAFE_X, ROOM_68_MID_Y) == ["LEFT"]
    assert press(ROOM_68_SOUTH_X, ROOM_68_MID_Y) == ["DOWN"]
    assert press(80, ROOM_68_TRAP_ROW_Y) == ["UP"]


def test_room_68_down_arrived_leaves_0x68() -> None:
    dest = read_snapshot(_ram(level=7, screen=0x78, x=120, y=77))
    ctl = Room68DownController()
    act = ctl.step(dest)
    assert ctl.success and not ctl.failed
    assert act.reason == "left_0x68_south"


def test_room_58_north_east_around_then_push() -> None:
    """0x58 north: climb, east around the central mass, then x=120 UP."""

    def press(x: int, y: int) -> list[str]:
        return _press(room_58_north_step(read_snapshot(_ram(level=7, screen=0x58, x=x, y=y))))

    assert press(120, 205) == ["UP"]
    assert press(120, ROOM_58_NORTH_MID_Y) == ["RIGHT"]
    assert press(ROOM_58_NORTH_EAST_X, ROOM_58_NORTH_MID_Y) == ["UP"]
    assert press(ROOM_58_NORTH_EAST_X, ROOM_58_NORTH_TOP_Y) == ["LEFT"]
    assert press(ROOM_58_NORTH_X, ROOM_58_NORTH_TOP_Y) == ["UP"]


def test_room_58_north_arrived_leaves_0x58() -> None:
    dest = read_snapshot(_ram(level=7, screen=0x48, x=120, y=205))
    ctl = Room58NorthController()
    act = ctl.step(dest)
    assert ctl.success and not ctl.failed
    assert act.reason == "left_0x58_north"


def test_room_49_up_south_mouth_stands_until_spawn() -> None:
    mouth = read_snapshot(_ram(level=7, screen=0x49, x=120, y=SOUTH_MOUTH_Y))
    assert room_49_up_step(mouth).reason == "spawn_wait"
    ctl = Room49UpController()
    act = ctl.step(mouth)
    assert not ctl.success and not ctl.failed
    assert act.reason == "spawn_wait"


def test_room_49_up_ladder_zero_never_crosses_the_moat() -> None:
    water = read_snapshot(_ram(level=7, screen=0x49, x=64, y=117, ladder=0))
    assert room_49_up_step(water, saw_goriya=True).reason == "moat_requires_ladder"
    ctl = Room49UpController()
    ctl.saw_goriya = True
    act = ctl.step(water)
    assert ctl.failed
    assert not ctl.success
    assert act.reason == "moat_requires_ladder"


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
    assert "level7_room49_up" in [
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
    assert "level7_room39_left" in [
        name for name, _c, _f in level7_red_candle_chapter_stages()
    ]


def test_room_38_up_uses_east_pocket_not_centre_diamonds() -> None:
    """y=149 diamond row blocks centre UP; recollect x=208 then rise."""
    mid = read_snapshot(_ram(level=7, screen=0x38, x=110, y=149))
    assert room_38_up_step(mid, saw_goriya=True).reason == "up38_pocket"
    pocket = read_snapshot(_ram(level=7, screen=0x38, x=208, y=149))
    assert room_38_up_step(pocket, saw_goriya=True).reason == "up38_rise"
    col144 = read_snapshot(_ram(level=7, screen=0x38, x=144, y=109))
    assert room_38_up_step(col144, saw_goriya=True).reason == "up38_rise"
    door = read_snapshot(_ram(level=7, screen=0x38, x=120, y=93))
    assert room_38_up_step(door, saw_goriya=True).reason == "up38_push"
    dest = read_snapshot(_ram(level=7, screen=0x28, x=120, y=SOUTH_MOUTH_Y))
    ctl = Room38UpController()
    act = ctl.step(dest)
    assert ctl.success and not ctl.failed
    assert act.reason == "left_0x38_north"
    assert "level7_room38_up" in [
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
    assert "level7_room09_down" in [
        name for name, _c, _f in level7_red_candle_chapter_stages()
    ]


def test_room_0d_clear_peels_west_grab_and_arrives_on_all_dead() -> None:
    """0x0D wallmaster clear: x<44 RIGHT; room_all_dead arrives. Recon-only."""
    grab = read_snapshot(_ram(level=7, screen=0x0D, x=32, y=141))
    assert room_0d_clear_step(grab, Room0DClearController()).reason == (
        "clear0d_grab_peel"
    )
    inland = read_snapshot(_ram(level=7, screen=0x0D, x=96, y=141))
    assert room_0d_clear_step(inland, Room0DClearController()).reason == (
        "clear0d_to_nudge_y"
    )
    dead = read_snapshot(_ram(level=7, screen=0x0D, x=96, y=141, room_all_dead=1))
    done = Room0DClearController()
    act = done.step(dead)
    assert done.success and not done.failed
    assert act.reason == "left_0x0d_cleared"
    names = [name for name, _c, _f in level7_complete_chapter_stages()]
    assert "level7_room0d_clear" in names
    assert "level7_tip_of_nose_stairs" in names
    assert "level7_tip_of_nose_stairs" not in [
        n for n, _c, _f in level7_red_candle_chapter_stages()
    ]


def test_room_1a_candle_is_the_chapter_pickup_and_arrives_on_candle_2() -> None:
    """Natural ADDR_CANDLE 0→2; a pin that already has 2 never greens."""
    mouth = read_snapshot(_ram(level=7, screen=0x1A, x=32, y=141, candle=0))
    assert Room1ACandleController().step(mouth).reason == "spawn_wait"
    already = read_snapshot(
        _ram(level=7, screen=0x4A, x=135, y=141, mode=9, candle=2)
    )
    ctl = Room1ACandleController()
    act = ctl.step(already)
    assert ctl.failed and not ctl.success
    assert act.reason == "already_red_candle"
    rising = Room1ACandleController()
    pad = _ram(level=7, screen=0x4A, x=124, y=141, mode=9, candle=0)
    rising.step(read_snapshot(pad))
    pad[ADDR_CANDLE] = 2
    act = rising.step(read_snapshot(pad))
    assert rising.success and not rising.failed
    assert act.reason == "red_candle_natural"
    assert "level7_red_candle_pickup" in [
        name for name, _c, _f in level7_red_candle_chapter_stages()
    ]
    assert isinstance(make_red_candle_controller(), Room1ACandleController)


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
    assert "level7_room4a_return" not in [
        name for name, _c, _f in level7_red_candle_chapter_stages()
    ]
    assert "level7_room4a_return" in [
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
    assert "level7_room1b_key_east" in [
        name for name, _c, _f in level7_complete_chapter_stages()
    ]

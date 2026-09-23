"""Recorder overworld warp (H1) + the 0x45 island join onto the pond chain."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from zelda_i.anchors import SCREEN_LEVEL4_ENTRANCE, SCREEN_LEVEL7_POND_HYP
from zelda_i.dungeon.pause_select import B_SLOT_RECORDER
from zelda_i.level7.overworld import (
    LEVEL7_POND_HOPS,
    POND_55_DOCK_X,
    POST_L6_TO_POND_HOPS,
    POST_L6_TO_WARP_HOPS,
    POST_L6_TO_WARP_SCREENS,
    WARP_ISLAND_SCREEN,
    WARP_JOIN_TO_POND_HOPS,
    WARP_JOIN_TO_POND_SCREENS,
    WARP_LAUNCH_SCREEN,
)
from zelda_i.level7.warp import (
    MAX_BLOWS,
    WHIRLWIND_MISS_FRAMES,
    WHIRLWIND_OBJECT_TYPE,
    RecorderWarpController,
    WarpPhase,
    make_recorder_warp_controller,
)
from zelda_i.ram import (
    ADDR_OBJ_TYPE,
    ADDR_SCREEN,
    ADDR_WHIRLWIND_SUMMONED,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.tests.ram_helpers import make_ram

_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 0,
    "screen": WARP_LAUNCH_SCREEN,
    "x": 128,
    "y": 141,
    "sword": 1,
    "triforce": 0x3F,
    "keys": 2,
    "bombs": 8,
    "arrows": 1,
    "health": 0x77,
    "whistle": 1,
    "food": 0,
    "rod": 1,
    "bow": 1,
    "candle": 0,
    "raft": 1,
    "ladder": 1,
    "rupees": 42,
    "selected": B_SLOT_RECORDER,
    "doors": 0,
    "room_all_dead": 0,
}


def _ram(**fields: int) -> np.ndarray:
    return make_ram(_DEFAULTS, **fields)


def _env(ram: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(get_ram=lambda: ram)


def _drive(ctl: RecorderWarpController, ram: np.ndarray, *, land_on: int | None,
           budget: int = 6000) -> int:
    """Step the controller, dropping Link on ``land_on`` after the first blow."""
    dropped = False
    for step in range(budget):
        if land_on is not None and ctl.blows >= 1 and not dropped:
            ram[ADDR_SCREEN] = land_on  # whirlwind carry settles Link here
            dropped = True
        ctl.step(read_snapshot(ram))
        if ctl.success or ctl.failed:
            return step
    return budget


# --- hop tables -------------------------------------------------------------


def test_warp_prefix_is_the_greened_walk_truncated_to_the_launch_screen() -> None:
    assert POST_L6_TO_WARP_HOPS == POST_L6_TO_POND_HOPS[:4]
    assert POST_L6_TO_WARP_HOPS[-1].target == WARP_LAUNCH_SCREEN == 0x24
    # 0x22 -> 0x32 -> 0x33 -> 0x23 -> 0x24, the live L6 reverse.
    assert POST_L6_TO_WARP_SCREENS == (0x22, 0x32, 0x33, 0x23, 0x24)


def test_warp_join_rejoins_the_green_pond_chain_at_0x55() -> None:
    assert WARP_ISLAND_SCREEN == SCREEN_LEVEL4_ENTRANCE == 0x45
    assert WARP_JOIN_TO_POND_SCREENS == (
        0x45,
        0x55,
        0x65,
        0x64,
        0x54,
        0x53,
        0x52,
        0x42,
    )
    assert WARP_JOIN_TO_POND_HOPS[-1].target == SCREEN_LEVEL7_POND_HYP
    # Everything from 0x64 on is the untouched, already-green tail.
    assert WARP_JOIN_TO_POND_HOPS[2:] == LEVEL7_POND_HOPS[7:]


def test_join_keeps_the_dock_column_through_0x65() -> None:
    """Regression: reusing the stock align_x=112 here stalls Link at (128,103).

    The stock ``0x55 -> 0x65`` hop assumes the east ``0x56 -> 0x55`` arrival
    band; from the raft dock at x=128 it drags Link LEFT into the mid-screen
    house/tree mass (rw_full_route_t1 burned a 30,000f budget there).
    """
    to_55, to_65 = WARP_JOIN_TO_POND_HOPS[0], WARP_JOIN_TO_POND_HOPS[1]
    assert (to_55.target, to_55.direction, to_55.align_x) == (0x55, "DOWN", 128)
    assert (to_65.target, to_65.direction, to_65.align_x) == (0x65, "DOWN", 128)
    assert POND_55_DOCK_X == 128
    stock_65 = LEVEL7_POND_HOPS[6]
    assert stock_65.target == 0x65 and stock_65.align_x == 112


# --- controller -------------------------------------------------------------


def test_warp_lands_on_the_island_and_stops_blowing() -> None:
    ram = _ram()
    ctl = make_recorder_warp_controller(
        target_screen=WARP_ISLAND_SCREEN, launch_screen=WARP_LAUNCH_SCREEN
    )
    ctl.bind_env(_env(ram))
    _drive(ctl, ram, land_on=WARP_ISLAND_SCREEN)
    assert ctl.success and not ctl.failed
    assert ctl.phase is WarpPhase.DONE
    assert ctl.blows == 1
    assert ctl.landings == ["0x45"]
    report = ctl.report()
    assert report["writes"] == 0
    assert report["route_eligible"] is False
    assert report["target_screen"] == "0x45"
    assert report["launch_screen"] == "0x24"
    assert report["facing"] == "DOWN"


def test_warp_keeps_blowing_through_intermediate_door_screens() -> None:
    """Landing on 0x22 (L6 door) is a cycle step, not a stop."""
    ram = _ram()
    ctl = make_recorder_warp_controller(
        target_screen=WARP_ISLAND_SCREEN, launch_screen=WARP_LAUNCH_SCREEN
    )
    ctl.bind_env(_env(ram))
    _drive(ctl, ram, land_on=0x22, budget=3000)
    assert not ctl.success
    assert ctl.blows >= 2
    assert ctl.landings[0] == "0x22"


def test_warp_waits_for_the_whirlwind_before_the_next_blow() -> None:
    """rr-p8rg: a still Link is not a landing while $0508 is set.

    Re-facing under an inbound whirlwind made it miss Link on 0x0B and left
    $0508 stuck, so every later blow no-oped.
    """
    ram = _ram()
    ctl = make_recorder_warp_controller(
        target_screen=WARP_ISLAND_SCREEN, launch_screen=WARP_LAUNCH_SCREEN
    )
    ctl.bind_env(_env(ram))
    while ctl.blows < 1:
        ctl.step(read_snapshot(ram))
    ram[ADDR_WHIRLWIND_SUMMONED] = 1
    ram[ADDR_OBJ_TYPE + 2] = WHIRLWIND_OBJECT_TYPE  # still crossing
    for _ in range(1000):
        act = ctl.step(read_snapshot(ram))
    assert ctl.phase is WarpPhase.SETTLE and ctl.blows == 1
    assert act.reason == "warp_settle"
    ram[ADDR_WHIRLWIND_SUMMONED] = 0
    ram[ADDR_OBJ_TYPE + 2] = 0
    ram[ADDR_SCREEN] = WARP_ISLAND_SCREEN
    for _ in range(200):
        ctl.step(read_snapshot(ram))
    assert ctl.success and ctl.landings == ["0x45"]


def test_a_missed_whirlwind_walks_off_the_screen() -> None:
    """$0508 set with no $2E crossing: leave the screen (the load clears it).

    The fake RAM has no tile map, so every leave direction is unroutable and
    the controller names the boxed screen instead of blowing into a no-op.
    """
    ram = _ram()
    ctl = make_recorder_warp_controller(
        target_screen=WARP_ISLAND_SCREEN, launch_screen=WARP_LAUNCH_SCREEN
    )
    ctl.bind_env(_env(ram))
    while ctl.blows < 1:
        ctl.step(read_snapshot(ram))
    ram[ADDR_WHIRLWIND_SUMMONED] = 1
    for _ in range(WHIRLWIND_MISS_FRAMES + 1):
        ctl.step(read_snapshot(ram))
    assert ctl.failed and ctl.report()["misses"] == 1
    notes = ctl.report()["notes"]
    assert "warp_whirlwind_missed_leave" in notes
    assert notes[-1] == "warp_leave_boxed_0x24"


def test_warp_gives_up_after_max_blows() -> None:
    ram = _ram()
    ctl = make_recorder_warp_controller(
        target_screen=WARP_ISLAND_SCREEN, launch_screen=WARP_LAUNCH_SCREEN
    )
    ctl.bind_env(_env(ram))
    _drive(ctl, ram, land_on=0x0B, budget=40_000)
    assert ctl.failed and not ctl.success
    assert ctl.blows == MAX_BLOWS
    assert ctl.report()["notes"][-1].startswith("warp_target_unreached")


def test_warp_refuses_off_the_launch_screen() -> None:
    ram = _ram(screen=0x33)
    ctl = make_recorder_warp_controller(
        target_screen=WARP_ISLAND_SCREEN, launch_screen=WARP_LAUNCH_SCREEN
    )
    ctl.bind_env(_env(ram))
    act = ctl.step(read_snapshot(ram))
    assert ctl.failed
    assert act.reason == "warp_not_on_launch_0x33"


def test_warp_requires_the_owned_recorder() -> None:
    ram = _ram(whistle=0)
    ctl = make_recorder_warp_controller(
        target_screen=WARP_ISLAND_SCREEN, launch_screen=WARP_LAUNCH_SCREEN
    )
    ctl.bind_env(_env(ram))
    act = ctl.step(read_snapshot(ram))
    assert ctl.failed
    assert act.reason == "warp_requires_whistle"


def test_warp_refuses_unbound_env_and_leaving_the_overworld() -> None:
    ram = _ram()
    unbound = make_recorder_warp_controller(
        target_screen=WARP_ISLAND_SCREEN, launch_screen=WARP_LAUNCH_SCREEN
    )
    assert unbound.step(read_snapshot(ram)).reason == "warp_env_not_bound"
    assert unbound.failed

    in_dungeon = _ram(level=7, screen=0x79)
    ctl = make_recorder_warp_controller(
        target_screen=WARP_ISLAND_SCREEN, launch_screen=WARP_LAUNCH_SCREEN
    )
    ctl.bind_env(_env(in_dungeon))
    act = ctl.step(read_snapshot(in_dungeon))
    assert ctl.failed
    assert act.reason == "warp_left_overworld_L7"


def test_warp_faces_up_when_the_cycle_passed_the_target() -> None:
    """Landing on L3 0x74 while aiming at L4 0x45: the next blow faces UP."""
    ram = _ram()
    ctl = make_recorder_warp_controller(
        target_screen=WARP_ISLAND_SCREEN, launch_screen=WARP_LAUNCH_SCREEN
    )
    ctl.bind_env(_env(ram))
    _drive(ctl, ram, land_on=0x74, budget=800)
    assert ctl.landings[0] == "0x74"
    assert ctl.facings[:2] == ["DOWN", "UP"]


def test_a_missed_summon_is_spent() -> None:
    """Miss on L5 0x0B facing DOWN spends L4; the cursor moves on to it."""
    ram = _ram()
    ctl = make_recorder_warp_controller(
        target_screen=0x74, launch_screen=WARP_LAUNCH_SCREEN
    )
    ctl.bind_env(_env(ram))
    ctl._cursor = 5
    ctl._blow_face = "DOWN"
    ctl._spend_cursor(read_snapshot(ram))
    assert ctl._cursor == 4
    assert ctl._choose_facing() == "DOWN"
    ctl._cursor = 1
    ctl._spend_cursor(read_snapshot(ram))  # DOWN off the lowest owned wraps
    assert ctl._cursor == 6

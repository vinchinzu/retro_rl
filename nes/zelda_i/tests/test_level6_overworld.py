"""Unit tests for Level 6 overworld hops, L5 settle, and spine stages."""

from __future__ import annotations

import numpy as np

from zelda_i.level6_overworld import (
    LEVEL6_POST_L5_SCREENS,
    POST_L5_PATH_MAX_FRAMES,
    POST_L5_TO_LEVEL6_HOPS,
    SCREEN_POST_L5_RETURN,
    OverworldToLevel6Controller,
    PostL5SettlePhase,
    PostL5TriforceSettleController,
    level6_hops_from,
    make_post_l5_level6_controller,
    post_l5_overworld_ready,
)
from zelda_i.level6_spine import (
    L6_THROUGH,
    Level6Return79Controller,
    level6_east_key_stages,
    level6_east_key_success,
    level6_entry_stages,
    level6_entry_success,
    level6_west_stages,
    level6_west_success,
)
from zelda_i.overworld import neighbor_screens
from zelda_i.ram import (
    ADDR_LADDER,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_RAFT,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    ADDR_WHISTLE,
    ADDR_KEYS,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.survival_spine import SpineRun


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 0)
    ram[ADDR_SCREEN] = fields.get("screen", 0x0B)
    ram[ADDR_LINK_X] = fields.get("x", 112)
    ram[ADDR_LINK_Y] = fields.get("y", 125)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0x1F)
    ram[ADDR_RAFT] = fields.get("raft", 1)
    ram[ADDR_LADDER] = fields.get("ladder", 1)
    ram[ADDR_WHISTLE] = fields.get("whistle", 1)
    ram[ADDR_KEYS] = fields.get("keys", 5)
    return ram


def test_post_l5_path_is_contiguous_and_skips_lost_hills_south() -> None:
    assert SCREEN_POST_L5_RETURN == 0x0B
    assert LEVEL6_POST_L5_SCREENS[0] == 0x0B
    assert LEVEL6_POST_L5_SCREENS[1] == 0x1B
    assert LEVEL6_POST_L5_SCREENS[-1] == 0x22
    assert [h.target for h in POST_L5_TO_LEVEL6_HOPS[:3]] == [0x1B, 0x1A, 0x19]
    assert POST_L5_TO_LEVEL6_HOPS[0].direction == "DOWN"
    assert POST_L5_TO_LEVEL6_HOPS[1].direction == "LEFT"
    assert POST_L5_TO_LEVEL6_HOPS[-1].target == 0x22
    assert POST_L5_TO_LEVEL6_HOPS[-1].direction == "UP"
    assert len(POST_L5_TO_LEVEL6_HOPS) == len(LEVEL6_POST_L5_SCREENS) - 1
    for before, after in zip(LEVEL6_POST_L5_SCREENS, LEVEL6_POST_L5_SCREENS[1:]):
        assert after in neighbor_screens(before).values(), f"{before:02x}->{after:02x}"


def test_level6_hops_from_l5_door_uses_post_l5_path() -> None:
    assert level6_hops_from(0x0B) == POST_L5_TO_LEVEL6_HOPS
    assert level6_hops_from(0x1B)[0].target == 0x1A
    assert level6_hops_from(0x32)[0].target == 0x22


def test_post_l5_settle_idles_fanfare_then_l5_door() -> None:
    ctl = PostL5TriforceSettleController()
    fanfare = read_snapshot(_ram(mode=18, level=5, screen=0x14, triforce=0x1F))
    act = ctl.step(fanfare)
    assert act.reason == "settle_wait"
    assert ctl.success is False
    ready = read_snapshot(_ram())
    act = ctl.step(ready)
    assert ctl.success
    assert ctl.phase is PostL5SettlePhase.DONE
    assert act.reason == "settle_done"
    assert post_l5_overworld_ready(ready)
    assert not post_l5_overworld_ready(read_snapshot(_ram(triforce=0x0F)))
    assert not post_l5_overworld_ready(read_snapshot(_ram(screen=0x22)))


def test_level6_entry_attaches_post_l5_hops() -> None:
    stages = level6_entry_stages()
    assert [name for name, _, _ in stages] == ["settle_l5_tf", "enter_level6"]
    assert isinstance(stages[0][1], PostL5TriforceSettleController)
    enter = stages[1][1]
    assert enter.hops == POST_L5_TO_LEVEL6_HOPS
    assert enter.require_dungeon is True
    assert stages[1][2] == POST_L5_PATH_MAX_FRAMES
    ctl = make_post_l5_level6_controller()
    assert ctl.hops[0].target == 0x1B
    assert ctl.hops[0].direction == "DOWN"
    run = SpineRun(through="level6-entry", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_entry_0x79"


def test_level6_entry_stop_requires_l5_inventory() -> None:
    ram = _ram(level=6, screen=0x79, x=120, y=205)
    snap = read_snapshot(ram)
    assert level6_entry_success(snap, whistle=1)
    assert not level6_entry_success(snap, whistle=0)
    ram[ADDR_LADDER] = 0
    assert not level6_entry_success(read_snapshot(ram), whistle=1)
    ram[ADDR_LADDER] = 1
    ram[ADDR_RAFT] = 0
    assert not level6_entry_success(read_snapshot(ram), whistle=1)
    ram[ADDR_RAFT] = 1
    ram[ADDR_TRIFORCE] = 0x0F
    assert not level6_entry_success(read_snapshot(ram), whistle=1)
    ram[ADDR_TRIFORCE] = 0x1F
    ram[ADDR_SCREEN] = 0x7A
    assert not level6_entry_success(read_snapshot(ram), whistle=1)


def test_lost_hills_left_stays_on_north_arrival_edge() -> None:
    ctl = make_post_l5_level6_controller()
    hop = POST_L5_TO_LEVEL6_HOPS[1]
    assert hop.target == 0x1A
    north = read_snapshot(_ram(screen=0x1B, x=112, y=61))
    act = ctl._extra_hop_action(north, hop)
    assert act is not None
    assert act.reason.startswith("hills_north_left")


def test_level6_east_key_attaches_right_then_fight() -> None:
    stages = level6_east_key_stages()
    assert [name for name, _, _ in stages] == [
        "level6_right_0x7a",
        "level6_east_key_0x7a",
    ]
    ram = _ram(level=6, screen=0x7A, x=120, y=141, keys=6)
    snap = read_snapshot(ram)
    assert level6_east_key_success(snap, keys_before=5)
    assert not level6_east_key_success(snap, keys_before=6)
    run = SpineRun(through="level6-east-key", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_east_key_0x7a"


def test_level6_west_attaches_return_then_key_then_clear() -> None:
    from retro_harness.nes import nes_action

    stages = level6_west_stages()
    assert [name for name, _, _ in stages] == [
        "level6_return_0x79",
        "level6_west_key_0x78",
        "level6_west_clear_0x78",
    ]
    assert isinstance(stages[0][1], Level6Return79Controller)
    ctl = Level6Return79Controller()
    ram = _ram(level=6, screen=0x7A, x=120, y=141)
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("LEFT"))
    ram[ADDR_SCREEN] = 0x79
    act = ctl.step(read_snapshot(ram))
    assert ctl.success
    assert act.reason == "arrived_79"
    ram[ADDR_SCREEN] = 0x78
    ram[ADDR_LEVEL] = 6
    snap = read_snapshot(ram)
    assert level6_west_success(snap)
    ram[ADDR_SCREEN] = 0x68
    assert not level6_west_success(read_snapshot(ram))
    run = SpineRun(through="level6-west", success=True, boot_frames=199)
    assert run.report()["stop"] == "level6_west_0x78"
    assert "level6-west" in L6_THROUGH
    assert not hasattr(OverworldToLevel6Controller(), "bfs")


def test_level6_east_key_live_enemies_block_stop() -> None:
    from zelda_i.level6_overworld import WIZZROBE_ORANGE_TYPE

    ram = _ram(level=6, screen=0x7A, x=120, y=141, keys=6)
    ram[ADDR_OBJ_TYPE + 1] = WIZZROBE_ORANGE_TYPE
    ram[ADDR_OBJ_HP + 1] = 64
    assert not level6_east_key_success(read_snapshot(ram), keys_before=5)

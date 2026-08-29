"""Unit tests for Level 6 overworld hops, L5 settle, and entry stages."""

from __future__ import annotations

import numpy as np

from retro_harness.nes import nes_action
from zelda_i.level6.overworld import (
    HILLS_AISLE_X,
    HILLS_CHANNEL_Y_HI,
    HILLS_CHANNEL_Y_LO,
    HILLS_NOTCH_X,
    HILLS_ROCK_X,
    LEVEL6_POST_L5_SCREENS,
    POST_L5_TO_LEVEL6_HOPS,
    SCREEN_POST_L5_RETURN,
    WIZZROBE_ORANGE_TYPE,
    lost_hills_west_dir,
    make_post_l5_level6_controller,
)
from zelda_i.level6.path import Level6North68Controller
from zelda_i.level6.spine import (
    level6_east_key_success,
    level6_entry_success,
)
from zelda_i.overworld.graph import neighbor_screens
from zelda_i.ram import (
    ADDR_KEYS,
    ADDR_LADDER,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_RAFT,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)


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
    assert POST_L5_TO_LEVEL6_HOPS[1].align_y == 141
    assert POST_L5_TO_LEVEL6_HOPS[2].target == 0x19
    assert POST_L5_TO_LEVEL6_HOPS[2].align_y == 141
    assert all(
        h.align_y == 141
        for h in POST_L5_TO_LEVEL6_HOPS[1:7]
        if h.direction == "LEFT"
    )
    assert POST_L5_TO_LEVEL6_HOPS[7].target == 0x14
    assert POST_L5_TO_LEVEL6_HOPS[7].y_band == (165, 189)
    assert POST_L5_TO_LEVEL6_HOPS[8].target == 0x24
    assert POST_L5_TO_LEVEL6_HOPS[8].direction == "DOWN"
    assert POST_L5_TO_LEVEL6_HOPS[8].align_x == 160
    assert POST_L5_TO_LEVEL6_HOPS[10].target == 0x33
    assert POST_L5_TO_LEVEL6_HOPS[10].align_x == 208
    assert POST_L5_TO_LEVEL6_HOPS[-1].target == 0x22
    assert POST_L5_TO_LEVEL6_HOPS[-1].direction == "UP"
    assert len(POST_L5_TO_LEVEL6_HOPS) == len(LEVEL6_POST_L5_SCREENS) - 1
    for before, after in zip(LEVEL6_POST_L5_SCREENS, LEVEL6_POST_L5_SCREENS[1:]):
        assert after in neighbor_screens(before).values(), f"{before:02x}->{after:02x}"


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


def test_lost_hills_west_channel_gates_burned_leftovers() -> None:
    """Screenshot occupancy: west sand y=136–151 x<72. Not north-edge LEFT."""
    assert lost_hills_west_dir(112, 61) == "DOWN"  # v25 north arrival
    assert lost_hills_west_dir(96, 141) == "DOWN"  # v1 rock at x≈72
    assert lost_hills_west_dir(96, 165) == "DOWN"  # v26 bottom rock row
    assert lost_hills_west_dir(96, 185) == "LEFT"  # south sand of both rows
    assert lost_hills_west_dir(71, 189) == "LEFT"  # v27 under the south rock
    assert lost_hills_west_dir(48, 189) == "UP"  # v28 SW mountain, notch north
    assert lost_hills_west_dir(64, 165) == "DOWN"  # still east of notch
    assert lost_hills_west_dir(48, 165) == "LEFT"  # v29 toward v17 aisle
    assert lost_hills_west_dir(40, 181) == "UP"  # v30 west wall, climb notch
    assert lost_hills_west_dir(32, 165) == "UP"  # v17 SW notch / west aisle
    assert lost_hills_west_dir(32, 189) == "UP"
    assert lost_hills_west_dir(40, 93) == "DOWN"  # v8 NW alcove
    assert lost_hills_west_dir(64, 125) == "DOWN"  # v15 north of channel
    assert lost_hills_west_dir(64, 149) == "LEFT"  # v12 in channel band
    assert lost_hills_west_dir(32, 141) == "LEFT"
    assert HILLS_CHANNEL_Y_LO <= 141 <= HILLS_CHANNEL_Y_HI
    assert HILLS_ROCK_X == 72
    assert HILLS_AISLE_X == 32
    assert HILLS_NOTCH_X == 48

    ctl = make_post_l5_level6_controller()
    hop = POST_L5_TO_LEVEL6_HOPS[1]
    assert hop.target == 0x1A
    assert hop.align_y == 141
    north = read_snapshot(_ram(screen=0x1B, x=112, y=61))
    act = ctl._extra_hop_action(north, hop)
    assert act is not None
    assert act.reason.startswith("hills_down")
    channel = read_snapshot(_ram(screen=0x1B, x=32, y=141))
    act = ctl._extra_hop_action(channel, hop)
    assert list(act.action) == list(nes_action("LEFT"))
    assert act.reason == "hills_west_left"
    v31 = read_snapshot(_ram(screen=0x1B, x=24, y=149))
    act = ctl._extra_hop_action(v31, hop)
    assert list(act.action) == list(nes_action("UP"))
    assert act.reason == "hills_west_ay"
    north_ch = read_snapshot(_ram(screen=0x1B, x=24, y=136))
    act = ctl._extra_hop_action(north_ch, hop)
    assert list(act.action) == list(nes_action("DOWN"))
    sw = read_snapshot(_ram(screen=0x1B, x=32, y=165))
    act = ctl._extra_hop_action(sw, hop)
    assert act.reason.startswith("hills_up")
    ctl.stuck = 181
    solid = ctl._extra_hop_action(sw, hop)
    assert solid.reason.startswith("hills_solid_32_165")
    assert ctl.phase.name == "FAILED"

    ctl = make_post_l5_level6_controller()
    hop14 = POST_L5_TO_LEVEL6_HOPS[7]
    assert hop14.target == 0x14
    east = read_snapshot(_ram(screen=0x15, x=232, y=109))
    act = ctl._extra_hop_action(east, hop14)
    assert act is not None
    assert act.reason.startswith("ow15_inland")
    mid = read_snapshot(_ram(screen=0x15, x=104, y=141))
    act = ctl._extra_hop_action(mid, hop14)
    assert act.reason.startswith("ow15_south")


def test_level6_compass_fails_closed_and_east_key_live_enemies() -> None:
    ctl = Level6North68Controller()
    act = ctl.step(read_snapshot(_ram(level=6, screen=0x79, x=32, y=141)))
    assert ctl.failed
    assert act.reason == "left_0x78"

    ram = _ram(level=6, screen=0x7A, x=120, y=141, keys=6)
    ram[ADDR_OBJ_TYPE + 1] = WIZZROBE_ORANGE_TYPE
    ram[ADDR_OBJ_HP + 1] = 64
    assert not level6_east_key_success(read_snapshot(ram), keys_before=5)

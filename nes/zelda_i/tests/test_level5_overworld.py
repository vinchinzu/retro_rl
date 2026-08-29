import numpy as np

from zelda_i.level5.overworld import (
    POST_L4_TO_LEVEL5_HOPS,
    PostL4SettlePhase,
    PostL4TriforceSettleController,
    post_l4_overworld_ready,
)
from zelda_i.level5.spine import level5_entry_success
from zelda_i.ram import (
    ADDR_LADDER,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_RAFT,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)


def test_post_l4_path_returns_from_raft_island_and_joins_4a() -> None:
    targets = [hop.target for hop in POST_L4_TO_LEVEL5_HOPS]
    assert targets[:7] == [0x55, 0x56, 0x57, 0x58, 0x59, 0x49, 0x4A]
    assert POST_L4_TO_LEVEL5_HOPS[0].direction == "DOWN"
    assert POST_L4_TO_LEVEL5_HOPS[0].align_x == 128


def test_post_l4_56_entry_uses_open_center_channel() -> None:
    hop = POST_L4_TO_LEVEL5_HOPS[2]
    assert hop.target == 0x57
    assert hop.direction == "RIGHT"
    assert hop.align_y == 141


def _l4_ow_ram(*, mode: int = PLAY_MODE, screen: int = 0x45, tf: int = 0x0F, raft: int = 1):
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_LEVEL] = 0
    ram[ADDR_SCREEN] = screen
    ram[ADDR_LINK_X] = 128
    ram[ADDR_LINK_Y] = 125
    ram[ADDR_TRIFORCE] = tf
    ram[ADDR_RAFT] = raft
    ram[ADDR_LADDER] = 1
    return ram


def test_post_l4_settle_idles_fanfare_then_island() -> None:
    ctl = PostL4TriforceSettleController()
    fanfare = read_snapshot(_l4_ow_ram(mode=18, screen=0x03, tf=0x0F))
    act = ctl.step(fanfare)
    assert act.reason == "settle_wait"
    assert ctl.success is False
    ready = read_snapshot(_l4_ow_ram())
    act = ctl.step(ready)
    assert ctl.success
    assert ctl.phase is PostL4SettlePhase.DONE
    assert act.reason == "settle_done"
    assert post_l4_overworld_ready(ready)
    assert not post_l4_overworld_ready(read_snapshot(_l4_ow_ram(tf=0x07)))
    assert not post_l4_overworld_ready(read_snapshot(_l4_ow_ram(raft=0)))


def test_level5_entry_stop_requires_l4_inventory() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 5
    ram[ADDR_SCREEN] = 0x76
    ram[ADDR_LINK_X] = 120
    ram[ADDR_LINK_Y] = 205
    ram[ADDR_TRIFORCE] = 0x0F
    ram[ADDR_RAFT] = 1
    ram[ADDR_LADDER] = 1
    snap = read_snapshot(ram)
    assert level5_entry_success(snap)
    ram[ADDR_LADDER] = 0
    assert not level5_entry_success(read_snapshot(ram))
    ram[ADDR_LADDER] = 1
    ram[ADDR_RAFT] = 0
    assert not level5_entry_success(read_snapshot(ram))
    ram[ADDR_RAFT] = 1
    ram[ADDR_TRIFORCE] = 0x07
    assert not level5_entry_success(read_snapshot(ram))
    ram[ADDR_TRIFORCE] = 0x0F
    ram[ADDR_SCREEN] = 0x66
    assert not level5_entry_success(read_snapshot(ram))

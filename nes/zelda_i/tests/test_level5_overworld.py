import numpy as np
import pytest
from types import SimpleNamespace

from zelda_i.level5.overworld import (
    POST_L4_TO_LEVEL5_HOPS,
    RUPEES_67_BACK_HOPS,
    RUPEES_67_HOPS,
    PostL4SettlePhase,
    PostL4TriforceSettleController,
    make_post_l4_level5_controller,
    post_l4_overworld_ready,
)
from retro_harness.nes import nes_action
from zelda_i.level5.spine import l5_hops, level5_entry_success, rupees_67_stages
from zelda_i.overworld.graph import neighbor_screens
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_LADDER,
    ADDR_HEALTH,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_RAFT,
    ADDR_RUPEES,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    ADDR_WORLD_FLAGS,
    PLAY_MODE,
    WORLD_FLAG_ITEM,
    read_snapshot,
)


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
    ram[ADDR_HEALTH] = 0x88
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


def test_walk_67_steps_off_the_raft_before_defend() -> None:
    walk = rupees_67_stages()[0][1]
    walk.frames = 1
    walk.hop_index = 1
    dock = read_snapshot(_l4_ow_ram(screen=0x55))
    act = walk.step(dock)
    assert act.reason == "raft_dismount"
    assert list(act.action) == list(nes_action("DOWN"))
    island = read_snapshot(_l4_ow_ram(screen=0x45))
    act = walk.step(island)
    assert act.reason == "raft_south"
    assert walk.hop_index == 0


def test_post_l4_level5_walk_dismounts_raft() -> None:
    assert POST_L4_TO_LEVEL5_HOPS[1].align_y == 141
    ctl = make_post_l4_level5_controller()
    ctl.frames = 1
    ctl.hop_index = 1
    dock = read_snapshot(_l4_ow_ram(screen=0x55))
    act = ctl.step(dock)
    assert act.reason == "raft_dismount"
    assert list(act.action) == list(nes_action("DOWN"))


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


def test_rupees_67_detour_rejoins_l5_walk_after_restock_checks() -> None:
    from zelda_i.level9.hops import L9_RUPEES_67_HOPS, L9_RUPEES_67_RETURN_HOPS

    route = (0x45, *(hop.target for hop in RUPEES_67_HOPS))
    assert route == (
        0x45, 0x55, 0x56, 0x57, 0x58, 0x59, 0x49, 0x4A,
        0x49, 0x59, 0x58, 0x68, 0x78, 0x77, 0x67,
    )
    assert RUPEES_67_HOPS[-len(L9_RUPEES_67_HOPS):] == L9_RUPEES_67_HOPS
    assert RUPEES_67_BACK_HOPS == L9_RUPEES_67_RETURN_HOPS
    assert tuple(hop.target for hop in RUPEES_67_BACK_HOPS)[-1] == 0x4A
    for a, b in zip(route, route[1:]):
        assert b in neighbor_screens(a).values()
    names = [name for name, _, _ in l5_hops()[0].stages]
    assert names.index("bomb_restock_l4") < names.index("walk_67")
    assert names.index("return_4a") < names.index("enter_level5")


@pytest.mark.parametrize(
    ("bombs", "rupees", "taken", "screen"),
    [(0, 24, False, 0x45), (3, 24, True, 0x45),
     (3, 240, False, 0x45), (3, 100, False, 0x10)],
)
def test_rupees_67_stages_skip_without_cave_pay(
    bombs: int, rupees: int, taken: bool, screen: int
) -> None:
    ram = _l4_ow_ram(screen=screen)
    ram[ADDR_BOMBS] = bombs
    ram[ADDR_RUPEES] = rupees
    if taken:
        ram[ADDR_WORLD_FLAGS + 0x67] = WORLD_FLAG_ITEM
    env = SimpleNamespace(get_ram=lambda: ram)
    snap = read_snapshot(ram)
    stages = rupees_67_stages()
    assert [name for name, _, _ in stages] == [
        "walk_67", "select_bombs_67", "rupees_67", "exit_cave_67", "return_4a"
    ]
    for _, ctl, _ in stages:
        if hasattr(ctl, "bind_env"):
            ctl.bind_env(env)
        ctl.step(snap)
        assert ctl.success
        assert ctl.frames == 1

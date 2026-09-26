"""One leftover trap per burned hop. Independent RAM/screenshot evidence."""

from __future__ import annotations

import numpy as np

from retro_harness.nes import nes_action
from zelda_i.overworld.graph import LEVEL2_PATH_SCREENS, neighbor_screens
from zelda_i.overworld.nav import OverworldToLevel1Controller
from zelda_i.overworld.sword_cave import SwordCaveController
from zelda_i.ram import (
    ADDR_HEALTH,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_SWORD,
    CAVE_MODE,
    PLAY_MODE,
    SCREEN_START,
    read_snapshot,
)


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_SCREEN] = fields.get("screen", SCREEN_START)
    ram[ADDR_LINK_X] = fields.get("x", 120)
    ram[ADDR_LINK_Y] = fields.get("y", 141)
    ram[ADDR_HEALTH] = fields.get("health", 0x33)
    ram[ADDR_SWORD] = fields.get("sword", 1)
    return ram


def test_cave_exit_at_64_77_goes_down_first() -> None:
    """After sword cave exit ~(64,77) on 0x77, first travel is DOWN to y≈140."""
    ctrl = OverworldToLevel1Controller()
    act = ctrl.step(read_snapshot(_ram(screen=0x77, x=64, y=77, sword=1)))
    assert list(act.action) == list(nes_action("DOWN"))


def test_sword_pickup_at_x120_walks_up() -> None:
    """Cave pickup: x≈120 then UP. Mode 11, floor spawn, no sword yet."""
    ctrl = SwordCaveController()
    snap = read_snapshot(_ram(mode=CAVE_MODE, x=120, y=213, sword=0))
    act = None
    for _ in range(40):
        act = ctrl.step(snap)
    assert act is not None
    assert list(act.action) == list(nes_action("UP"))


def test_bait_24_sw_16_189_down_is_mountain() -> None:
    """l7_bait_33up leftover 0x24 (16,189): DOWN is south mountain; UP to y=141."""
    from zelda_i.level7.overworld import OverworldToBaitShopController

    ctl = OverworldToBaitShopController()
    hop = ctl.hops[-1]
    snap = read_snapshot(_ram(screen=0x24, x=16, y=189, sword=1))
    act = ctl._extra_hop_action(snap, hop)
    assert not ctl.failed
    assert act is not None
    assert "24_east_band" in act.reason
    assert "DOWN" not in act.reason


def test_bait_24_se_208_189_down_is_mountain() -> None:
    """l7_bait_24se leftover 0x24 (208,189): DOWN is SE mountain; UP to y=141."""
    from zelda_i.level7.overworld import OverworldToBaitShopController

    ctl = OverworldToBaitShopController()
    hop = ctl.hops[-1]
    snap = read_snapshot(_ram(screen=0x24, x=208, y=189, sword=1))
    act = ctl._extra_hop_action(snap, hop)
    assert not ctl.failed
    assert act is not None
    assert "24_east_band" in act.reason
    assert "DOWN" not in act.reason
    assert hop.target == 0x25
    assert hop.direction == "RIGHT"


def test_bait_25_arrival_0_141_is_west_mouth_not_shop() -> None:
    """l7_bait_25 leftover 0x25 (0,141): west mouth. Do not LEFT back to 0x24."""
    from zelda_i.level7.overworld import OverworldToBaitShopController

    ctl = OverworldToBaitShopController()
    assert ctl.hops[-1].target == 0x25
    assert ctl.end_screen() == 0x25
    ctl.hop_index = len(ctl.hops)
    snap = read_snapshot(_ram(screen=0x25, x=0, y=141, sword=1))
    act = ctl.step(snap)
    assert ctl.success
    assert not ctl.failed
    assert act.reason == "done"


def test_bait_33_east_208_141_is_mountain_not_0x34() -> None:
    """l7_bait_32ax leftover 0x33 (208,141): RIGHT is east mountain, not 0x34."""
    from zelda_i.level7.overworld import POST_L6_TO_BAIT_HOPS

    assert 0x34 not in {h.target for h in POST_L6_TO_BAIT_HOPS[:3]}


def test_bait_32_north_120_61_does_not_hold_down() -> None:
    """0x32 (120,61): off_north DOWN is the east wall of the x=112 corridor."""
    from zelda_i.level7.overworld import OverworldToBaitShopController

    ctl = OverworldToBaitShopController()
    hop = ctl.hops[1]
    snap = read_snapshot(_ram(screen=0x32, x=120, y=61, sword=1))
    act = ctl._extra_hop_action(snap, hop)
    assert act is not None
    assert "32_north_ax" in act.reason
    assert "DOWN" not in act.reason


def test_l6_exit_112_125_is_cave_mouth_not_leave() -> None:
    """Standing (112,125) on 0x22 starts dungeon enter (mode 16 → L6)."""
    from zelda_i.level7.overworld import OverworldToBaitShopController, at_l6_cave_mouth

    snap = read_snapshot(_ram(screen=0x22, x=112, y=125, sword=1))
    assert at_l6_cave_mouth(snap)
    ctl = OverworldToBaitShopController()
    act = ctl.step(snap)
    assert ctl.failed
    assert act.reason == "l6_cave_mouth"


def test_l6_clear29_120_77_left_is_door_channel() -> None:
    """Red 3 leftover (120,77): LEFT stays in the north door; DOWN inland."""
    from zelda_i.dungeon.engine import DungeonPhase, GenericDungeonRoomController
    from zelda_i.dungeon.ids import WIZZROBE_ORANGE_OBJECT_TYPE
    from zelda_i.level6.dungeon import ROOM_29_SPEC
    from zelda_i.ram import ADDR_OBJ_HP, ADDR_OBJ_TYPE, ADDR_LEVEL

    ram = _ram(screen=0x29, x=120, y=77, sword=1)
    ram[ADDR_LEVEL] = 6
    ram[ADDR_OBJ_TYPE + 1] = WIZZROBE_ORANGE_OBJECT_TYPE
    ram[ADDR_OBJ_HP + 1] = 64
    ctl = GenericDungeonRoomController(ROOM_29_SPEC)
    ctl.phase = DungeonPhase.FIGHT
    ctl.combat_frames = 24
    act = ctl.step(read_snapshot(ram))
    assert act.reason.startswith("leave_wall")
    assert list(act.action) == list(nes_action("DOWN")) or list(act.action) == list(
        nes_action("DOWN", "A")
    )
    assert list(act.action) != list(nes_action("LEFT"))


def test_l2_prefix_never_enters_79() -> None:
    """L2 prefix is 37→38→48→58→59→49→4A. 0x79 is a rocky dead-end."""
    assert LEVEL2_PATH_SCREENS == (0x37, 0x38, 0x48, 0x58, 0x59, 0x49, 0x4A)
    assert 0x79 not in LEVEL2_PATH_SCREENS
    for a, b in zip(LEVEL2_PATH_SCREENS, LEVEL2_PATH_SCREENS[1:]):
        assert b in neighbor_screens(a).values()

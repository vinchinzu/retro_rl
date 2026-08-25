"""Policy gates for the retargeted Level 6 0x3A northeast clip."""

from retro_harness.nes import nes_action
from zelda_i.level6_path import BLOCK_OBJECT_TYPE
from zelda_i.level6_stairs3a_ne71 import Stairs3ANE71Phase
from zelda_i.level6_stairs3a_neclip import (
    CLIP_START_X,
    DOOR_CLEAR_Y,
    level6_stairs3a_neclip_stages,
    make_stairs_3a_neclip_controller,
)
from zelda_i.ram import ADDR_OBJ_TYPE, ADDR_LINK_X, ADDR_LINK_Y, ADDR_ROD, read_snapshot

from zelda_i.tests.test_level6_stairs3a_ne71 import _ram


def _controller():
    ctl = make_stairs_3a_neclip_controller()
    ctl.phase = Stairs3ANE71Phase.TO_NE
    ctl.hole_x = 112
    return ctl


def _ne_block(ram) -> None:
    slot = 11
    ram[ADDR_OBJ_TYPE + slot] = BLOCK_OBJECT_TYPE
    ram[ADDR_LINK_X + slot] = 208
    ram[ADDR_LINK_Y + slot] = 96


def test_neclip_uses_live_right_down_corridor_through_x160() -> None:
    ram = _ram(x=160, y=147, tile=119)
    ram[ADDR_ROD] = 1
    _ne_block(ram)
    ctl = _controller()
    act = ctl.step(read_snapshot(ram))
    assert not ctl.passed_around
    assert act.reason == "ne_around_clip"
    assert list(act.action) == list(nes_action("RIGHT", "DOWN"))
    assert CLIP_START_X == 176


def test_neclip_turns_up_in_east_aisle_at_dated_x184_tile119() -> None:
    ram = _ram(x=184, y=147, tile=119)
    ram[ADDR_ROD] = 1
    _ne_block(ram)
    ctl = _controller()
    act = ctl.step(read_snapshot(ram))
    assert ctl.passed_around
    assert act.reason == "ne_north_aisle"
    assert list(act.action) == list(nes_action("UP"))
    assert ctl.walker.last_dir is None


def test_neclip_never_crosses_open_east_door_band() -> None:
    ram = _ram(x=200, y=133, tile=119)
    ram[ADDR_ROD] = 1
    _ne_block(ram)
    ctl = _controller()
    act = ctl.step(read_snapshot(ram))
    assert ctl.failed
    assert act.reason == "east_door_200_133"
    assert DOOR_CLEAR_Y == 132


def test_neclip_at_x200_above_door_holds_up() -> None:
    ram = _ram(x=200, y=125, tile=119)
    ram[ADDR_ROD] = 1
    _ne_block(ram)
    ctl = _controller()
    act = ctl.step(read_snapshot(ram))
    assert not ctl.failed
    assert act.reason == "ne_north_aisle"
    assert list(act.action) == list(nes_action("UP"))


def test_neclip_has_dedicated_stage() -> None:
    stages = level6_stairs3a_neclip_stages()
    assert stages[0][0] == "level6_stairs_0x3a_neclip"
    assert stages[0][1].spec_id == "level6_stairs_0x3a_neclip"

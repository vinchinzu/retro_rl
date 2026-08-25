"""Coordinate gates for the Level 6 0x3A under-block retarget."""

from retro_harness.nes import nes_action
from zelda_i.level6_path import BLOCK_OBJECT_TYPE
from zelda_i.level6_stairs3a_ne71 import Stairs3ANE71Phase
from zelda_i.level6_stairs3a_neunder import (
    UNDER_BLOCK_Y,
    level6_stairs3a_neunder_stages,
    make_stairs_3a_neunder_controller,
)
from zelda_i.ram import ADDR_LINK_X, ADDR_LINK_Y, ADDR_OBJ_TYPE, ADDR_ROD, read_snapshot
from zelda_i.tests.test_level6_stairs3a_ne71 import _ram


def _step(x: int, y: int, *, tile: int = 118):
    ram = _ram(x=x, y=y, tile=tile)
    ram[ADDR_ROD] = 1
    ram[ADDR_OBJ_TYPE + 11] = BLOCK_OBJECT_TYPE
    ram[ADDR_LINK_X + 11] = 208
    ram[ADDR_LINK_Y + 11] = 96
    ctl = make_stairs_3a_neunder_controller()
    ctl.phase = Stairs3ANE71Phase.TO_NE
    ctl.hole_x = 112
    return ctl, ctl.step(read_snapshot(ram))


def test_neunder_stops_climb_at_y132_then_goes_right() -> None:
    ctl, act = _step(176, 133)
    assert act.reason == "ne_north_aisle"
    assert list(act.action) == list(nes_action("UP"))
    assert ctl.walker.last_dir is None
    ctl, act = _step(176, UNDER_BLOCK_Y)
    assert act.reason == "ne_under_clip"
    assert list(act.action) == list(nes_action("RIGHT", "DOWN"))
    assert ctl.walker.last_dir is None


def test_neunder_recovers_frozen_north_face_by_going_down() -> None:
    _ctl, act = _step(178, 125, tile=117)
    assert act.reason == "ne_under_y"
    assert list(act.action) == list(nes_action("DOWN"))


def test_neunder_uses_shared_south_face_then_up() -> None:
    _ctl, act = _step(208, 132)
    assert act.reason == "ne_y"
    assert list(act.action) == list(nes_action("UP"))
    _ctl, act = _step(208, 112)
    assert act.reason == "push_ne_block"
    assert list(act.action) == list(nes_action("UP"))


def test_neunder_has_dedicated_stage() -> None:
    stages = level6_stairs3a_neunder_stages()
    assert stages[0][0] == "level6_stairs_0x3a_neunder"

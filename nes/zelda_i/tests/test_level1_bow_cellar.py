"""Unit tests for L1 0x22 west-block stairs into mode-9 cellar."""

from __future__ import annotations

import numpy as np

from zelda_i.level1_bow import LEVEL1_BOW_ROOM
from zelda_i.level1_bow_cellar import (
    BLOCK_OBJECT_TYPE,
    EAST_INLAND_X,
    SOUTH_LANE_Y,
    WEST_AISLE_X,
    level1_bow_cellar_stages,
    level1_bow_cellar_success,
    make_bow_cellar_controller,
    north_face_stand,
    westmost_block_0x68,
)
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOW,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_TYPE,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PASSAGE_MODE,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.survival_spine import SPINE_THROUGH, SpineRun


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 1)
    ram[ADDR_SCREEN] = fields.get("screen", LEVEL1_BOW_ROOM)
    ram[ADDR_LINK_X] = fields.get("x", 224)
    ram[ADDR_LINK_Y] = fields.get("y", 141)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0)
    ram[ADDR_KEYS] = fields.get("keys", 0)
    ram[ADDR_BOW] = fields.get("bow", 0)
    ram[ADDR_ARROWS] = fields.get("arrows", 0)
    return ram


def _plant_block(ram: np.ndarray, slot: int, x: int, y: int) -> None:
    ram[ADDR_OBJ_TYPE + slot] = BLOCK_OBJECT_TYPE
    ram[ADDR_LINK_X + slot] = x
    ram[ADDR_LINK_Y + slot] = y


def test_bow_cellar_through_is_wired_after_enter_stop() -> None:
    assert "level1-bow-cellar" in SPINE_THROUGH
    assert SPINE_THROUGH.index("level1-bow-cellar") == SPINE_THROUGH.index(
        "level1-bow"
    ) + 1
    names = [name for name, _, _ in level1_bow_cellar_stages()]
    assert names[-2] == "level1_bow_0x22"
    assert names[-1] == "level1_bow_cellar"
    run = SpineRun(through="level1-bow-cellar", success=True, boot_frames=199)
    assert run.report()["stop"] == "level1_bow_cellar"


def test_bow_cellar_occupancy_south_around_then_down() -> None:
    from retro_harness.nes import nes_action

    leftover = _ram(x=224, y=141)
    _plant_block(leftover, 4, 96, 144)
    ctl = make_bow_cellar_controller()
    act = ctl.step(read_snapshot(leftover))
    assert act.reason == "west_inland"
    assert list(act.action) == list(nes_action("LEFT"))
    assert list(act.action) != list(nes_action("UP"))
    inland = _ram(x=EAST_INLAND_X, y=141)
    _plant_block(inland, 4, 96, 144)
    peel = make_bow_cellar_controller()
    act = peel.step(read_snapshot(inland))
    assert act.reason == "south_peel"
    assert list(act.action) == list(nes_action("DOWN"))
    assert list(act.action) != list(nes_action("UP"))
    assert list(act.action) != list(nes_action("LEFT"))
    south = _ram(x=EAST_INLAND_X, y=SOUTH_LANE_Y)
    _plant_block(south, 4, 96, 144)
    west = make_bow_cellar_controller()
    act = west.step(read_snapshot(south))
    assert act.reason == "west_south"
    assert list(act.action) == list(nes_action("LEFT"))
    aisle = _ram(x=WEST_AISLE_X, y=SOUTH_LANE_Y)
    _plant_block(aisle, 4, 96, 144)
    climb = make_bow_cellar_controller()
    act = climb.step(read_snapshot(aisle))
    assert act.reason == "north_aisle"
    assert list(act.action) == list(nes_action("UP"))
    lane = _ram(x=WEST_AISLE_X, y=128)
    _plant_block(lane, 4, 96, 144)
    across = make_bow_cellar_controller()
    act = across.step(read_snapshot(lane))
    assert act.reason == "east_face"
    assert list(act.action) == list(nes_action("RIGHT"))
    face = north_face_stand(westmost_block_0x68(read_snapshot(leftover)))
    assert face == (96, 128)
    at_face = _ram(x=96, y=128)
    _plant_block(at_face, 4, 96, 144)
    push = make_bow_cellar_controller()
    act = push.step(read_snapshot(at_face))
    assert act.reason == "push_block"
    assert list(act.action) == list(nes_action("DOWN"))


def test_bow_cellar_requires_mode_9_not_play_22() -> None:
    dest = _ram(mode=PASSAGE_MODE, screen=0x00, x=48, y=80)
    assert level1_bow_cellar_success(read_snapshot(dest))
    still = _ram()
    assert not level1_bow_cellar_success(read_snapshot(still))
    ctl = make_bow_cellar_controller()
    act = ctl.step(read_snapshot(dest))
    assert ctl.success
    assert act.reason == "arrived_cellar"
    snap = read_snapshot(dest)
    assert snap.bow == 0
    back = _ram(screen=0x23, x=32, y=141)
    fail = make_bow_cellar_controller()
    fail.step(read_snapshot(back))
    assert fail.failed
    assert any(n.startswith("backtrack_23") for n in fail.notes)

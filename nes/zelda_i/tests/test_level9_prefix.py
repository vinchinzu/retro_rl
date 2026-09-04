"""Durable L9 0x76 north dest hop. No emulator.

x-align to 120, then UP. Dest is RAM (hyp 0x66); fail non-north and 0x07.
Natural old-man factory stays fail-closed. No RAM writes.
"""

from __future__ import annotations

import numpy as np
from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.door_graph import DoorDir, L9_ENTRY, LEVEL_9_NATURAL_DOOR_GRAPH
from zelda_i.level9.dungeon import MISSING_OLD_MAN_GATE, ROOM_OLD_MAN_TF
from zelda_i.level9.natural_path import make_old_man_tf_gate_controller
from zelda_i.level9.prefix import (
    NORTH_DEST_HYP,
    NORTH_DEST_POSE,
    NORTH_DOOR,
    NORTH_ORIGIN,
    RED_RING,
    WEST_DEST_HYP,
    WEST_DEST_POSE,
    WEST_DOOR,
    WEST_ORIGIN,
    Level9North76Controller,
    is_north_neighbor,
    is_west_neighbor,
    make_north_76_controller,
    make_west_66_controller,
    north_76_step,
    west_66_step,
)
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_COLLIDING_TILE,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MAGIC_KEY,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PASSAGE_MODE,
    PLAY_MODE,
    read_snapshot,
)

UP = list(nes_action("UP"))
DOWN = list(nes_action("DOWN"))
LEFT = list(nes_action("LEFT"))
RIGHT = list(nes_action("RIGHT"))
IDLE = list(nes_idle_action())


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 9)
    ram[ADDR_SCREEN] = fields.get("screen", NORTH_ORIGIN)
    ram[ADDR_LINK_X] = fields.get("x", 120)
    ram[ADDR_LINK_Y] = fields.get("y", 205)
    ram[ADDR_COLLIDING_TILE] = fields.get("tile", 0)
    ram[ADDR_KEYS] = fields.get("keys", 9)
    ram[ADDR_BOMBS] = fields.get("bombs", 16)
    ram[ADDR_MAGIC_KEY] = fields.get("magic_key", 1)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0xFF)
    return ram


def _step(ctl, ram: np.ndarray):
    before = ram.copy()
    act = ctl.step(read_snapshot(ram))
    assert np.array_equal(ram, before), "prefix dest hops must not write RAM"
    return act


def test_south_mouth_emits_up_not_left() -> None:
    snap = read_snapshot(_ram(x=120, y=205))
    act = north_76_step(snap)
    assert list(act.action) == UP
    assert list(act.action) != LEFT
    assert list(act.action) != DOWN
    assert act.reason == "north_76_push"

    ctl = make_north_76_controller()
    act = _step(ctl, _ram(x=120, y=205))
    assert not ctl.failed
    assert list(act.action) == UP
    assert act.reason == "north_76_push"


def test_off_center_aligns_x_then_up() -> None:
    left = make_north_76_controller()
    act = _step(left, _ram(x=160, y=205))
    assert not left.failed
    assert list(act.action) == LEFT
    assert act.reason == "north_76_align_x"

    right = make_north_76_controller()
    act = _step(right, _ram(x=80, y=141))
    assert not right.failed
    assert list(act.action) == RIGHT
    assert act.reason == "north_76_align_x"

    door = make_north_76_controller()
    act = _step(door, _ram(x=NORTH_DOOR[0], y=NORTH_DOOR[1]))
    assert not door.failed
    assert list(act.action) == UP
    assert act.reason == "north_76_push"


def test_dest_none_north_neighbor_succeeds_red_ring_and_cellar_fail() -> None:
    ok = make_north_76_controller(dest=None)
    act = _step(ok, _ram(mode=PLAY_MODE, screen=0x66, x=120, y=205))
    assert ok.success and not ok.failed
    assert list(act.action) == IDLE

    ring = make_north_76_controller(dest=None)
    act = _step(ring, _ram(mode=PLAY_MODE, screen=RED_RING, x=120, y=141))
    assert not ring.success and ring.failed
    assert "red_ring_0x07" in ring.notes
    assert list(act.action) == IDLE

    west = make_north_76_controller(dest=None)
    act = _step(west, _ram(mode=PLAY_MODE, screen=0x75, x=208, y=141))
    assert not west.success and west.failed
    assert "not_north_neighbor_0x75" in west.notes

    cellar = make_north_76_controller(dest=None)
    act = _step(cellar, _ram(mode=PASSAGE_MODE, screen=0x60, x=136, y=141))
    assert not cellar.success and cellar.failed
    assert list(act.action) == IDLE


def test_dest_0x66_accepts_66_rejects_07() -> None:
    ok = make_north_76_controller(dest=0x66)
    act = _step(ok, _ram(mode=PLAY_MODE, screen=0x66, x=120, y=205))
    assert ok.success and not ok.failed
    assert list(act.action) == IDLE

    bad = make_north_76_controller(dest=0x66)
    act = _step(bad, _ram(mode=PLAY_MODE, screen=RED_RING, x=120, y=141))
    assert not bad.success and bad.failed
    assert list(act.action) == IDLE


def test_factory_report_fixture_live_not_route_eligible() -> None:
    ctl = make_north_76_controller()
    assert isinstance(ctl, Level9North76Controller)
    report = ctl.report()
    assert report["route_eligible"] is False
    assert report["door"] == "UP"
    assert report["writes"] == 0
    assert report["evidence"] == "fixture-live"
    assert report["natural_entry"] is False
    assert report["spec_id"] == "level9_north_76"
    assert report["dest_screen"] is None
    assert report["dest_hyp"] == NORTH_DEST_HYP == ROOM_OLD_MAN_TF == 0x66
    assert NORTH_DOOR == (120, 93)
    assert NORTH_ORIGIN == 0x76
    assert NORTH_DEST_POSE == (120, 205)
    assert is_north_neighbor(0x76, 0x66)
    assert not is_north_neighbor(0x76, 0x75)
    assert not is_north_neighbor(0x76, 0x07)


def test_natural_old_man_factory_stays_fail_closed() -> None:
    ctl = make_old_man_tf_gate_controller()
    ram = _ram()
    before = ram.copy()
    act = ctl.step(read_snapshot(ram))
    assert np.array_equal(ram, before)
    assert ctl.failed
    assert act.reason == MISSING_OLD_MAN_GATE
    assert ctl.max_frames == 1


def test_west_66_south_mouth_aligns_y_then_left() -> None:
    snap = read_snapshot(_ram(screen=WEST_ORIGIN, x=120, y=205))
    act = west_66_step(snap)
    assert list(act.action) == UP
    assert act.reason == "west_66_align_y"

    ctl = make_west_66_controller()
    act = _step(ctl, _ram(screen=WEST_ORIGIN, x=120, y=WEST_DOOR[1]))
    assert not ctl.failed
    assert list(act.action) == LEFT
    assert act.reason == "west_66_approach"

    ok = make_west_66_controller(dest=None)
    act = _step(ok, _ram(mode=PLAY_MODE, screen=0x65, x=208, y=141))
    assert ok.success and not ok.failed
    assert is_west_neighbor(WEST_ORIGIN, WEST_DEST_HYP)
    assert WEST_DEST_HYP == 0x65
    assert WEST_DEST_POSE == (224, 141)
    assert WEST_DOOR == (32, 141)


def test_live_dest_edges_are_observed_on_natural_graph() -> None:
    north = LEVEL_9_NATURAL_DOOR_GRAPH.exit_between(
        L9_ENTRY, 0x66, direction=DoorDir.UP
    )
    assert north is not None
    assert north.verification == "observed"
    west = LEVEL_9_NATURAL_DOOR_GRAPH.exit_between(
        0x66, 0x65, direction=DoorDir.LEFT
    )
    assert west is not None
    assert west.verification == "observed"

"""Durable L7 nose cellar 0x7B B→A cross: DOWN, floor LEFT, never source UP.

No emulator. Fake snapshots at the right-ladder spawn (192,93) must emit
DOWN, never UP (CheckSubroom AttrB → play 0x0D). Floor LEFT to x=48, then
UP the west ladder. OccupancyWalker is banned. Spine tip-stairs factory
stays fail-closed. RAM claim dest is play 0x29.
"""

from __future__ import annotations

import numpy as np
from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.level7.cellar import (
    CELLAR_ROOM,
    DEST_ROOM,
    EAST_X,
    EXIT_STAIRS,
    FLOOR_Y,
    MOUTH_Y,
    PIT_TILE,
    RAM_CLAIM,
    SOURCE_ROOM,
    SPAWN_XY,
    WEST_X,
    Level7NoseCellarCrossController,
    make_nose_cellar_cross_controller,
    nose_cellar_cross_step,
    nose_cellar_cross_success,
)
from zelda_i.level7.hops import (
    make_aquamentus_heart_controller,
    make_level7_shard_leave_controller,
    make_nose_cellar_cross_controller as hops_make_nose_cellar_cross,
    make_tip_stairs_controller,
)
from zelda_i.level7.path import UnverifiedLevel7PathController
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_CANDLE,
    ADDR_COLLIDING_TILE,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PASSAGE_MODE,
    PLAY_MODE,
    read_snapshot,
)

DOWN = list(nes_action("DOWN"))
LEFT = list(nes_action("LEFT"))
RIGHT = list(nes_action("RIGHT"))
UP = list(nes_action("UP"))
IDLE = list(nes_idle_action())


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PASSAGE_MODE)
    ram[ADDR_LEVEL] = fields.get("level", 7)
    ram[ADDR_SCREEN] = fields.get("screen", CELLAR_ROOM)
    ram[ADDR_LINK_X] = fields.get("x", SPAWN_XY[0])
    ram[ADDR_LINK_Y] = fields.get("y", SPAWN_XY[1])
    ram[ADDR_COLLIDING_TILE] = fields.get("tile", 36)
    ram[ADDR_KEYS] = fields.get("keys", 2)
    ram[ADDR_BOMBS] = fields.get("bombs", 6)
    ram[ADDR_CANDLE] = fields.get("candle", 2)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0)
    return ram


def _step(ctl, ram: np.ndarray):
    before = ram.copy()
    act = ctl.step(read_snapshot(ram))
    assert np.array_equal(ram, before), "cellar controllers must not write RAM"
    return act


def test_ram_claim_is_play_0x29_not_0x0d() -> None:
    assert DEST_ROOM == 0x29
    assert SOURCE_ROOM == 0x0D
    assert "0x29" in RAM_CLAIM
    assert "0x0D" in RAM_CLAIM
    assert "Never UP at x>=$80" in RAM_CLAIM


def test_right_spawn_goes_down_never_up() -> None:
    snap = read_snapshot(_ram(x=EAST_X, y=MOUTH_Y))
    act = nose_cellar_cross_step(snap)
    assert act.reason == "cellar_east_drop"
    assert list(act.action) == DOWN
    assert list(act.action) != UP

    ctl = make_nose_cellar_cross_controller()
    act = _step(ctl, _ram(x=EAST_X, y=MOUTH_Y))
    assert not ctl.failed
    assert ctl.arrival_seen
    assert act.reason == "cellar_east_drop"
    assert list(act.action) == DOWN
    assert list(act.action) != UP


def test_mid_top_channel_goes_east_never_up() -> None:
    """Previous miss: UP at any x on the top channel returns to 0x0D."""
    snap = read_snapshot(_ram(x=160, y=MOUTH_Y))
    act = nose_cellar_cross_step(snap)
    assert act.reason == "cellar_to_east"
    assert list(act.action) == RIGHT
    assert list(act.action) != UP
    assert list(act.action) != LEFT


def test_floor_walks_west() -> None:
    act = nose_cellar_cross_step(read_snapshot(_ram(x=100, y=FLOOR_Y)))
    assert act.reason == "cellar_floor_west"
    assert list(act.action) == LEFT
    assert list(act.action) != UP

    ctl = make_nose_cellar_cross_controller()
    ctl.arrival_seen = True
    act = _step(ctl, _ram(x=100, y=FLOOR_Y))
    assert not ctl.failed
    assert ctl.on_floor
    assert act.reason == "cellar_floor_west"
    assert list(act.action) == LEFT


def test_west_floor_climbs() -> None:
    act = nose_cellar_cross_step(read_snapshot(_ram(x=WEST_X, y=FLOOR_Y)))
    assert act.reason == "cellar_west_climb"
    assert list(act.action) == UP

    ctl = make_nose_cellar_cross_controller()
    ctl.arrival_seen = True
    act = _step(ctl, _ram(x=WEST_X, y=FLOOR_Y))
    assert not ctl.failed
    assert act.reason == "cellar_west_climb"
    assert list(act.action) == UP


def test_west_climb_holds_up() -> None:
    act = nose_cellar_cross_step(read_snapshot(_ram(x=WEST_X, y=120)))
    assert act.reason == "cellar_west_up"
    assert list(act.action) == UP

    ctl = make_nose_cellar_cross_controller()
    ctl.arrival_seen = True
    act = _step(ctl, _ram(x=WEST_X, y=120))
    assert not ctl.failed
    assert act.reason == "cellar_west_up"
    assert list(act.action) == UP


def test_west_lip_keeps_up() -> None:
    snap = read_snapshot(_ram(x=EXIT_STAIRS[0], y=EXIT_STAIRS[1], tile=0x6F))
    act = nose_cellar_cross_step(snap)
    assert act.reason == "cellar_west_lip"
    assert list(act.action) == UP

    ctl = make_nose_cellar_cross_controller()
    ctl.arrival_seen = True
    act = _step(ctl, _ram(x=EXIT_STAIRS[0], y=EXIT_STAIRS[1], tile=0x6F))
    assert not ctl.failed
    assert act.reason == "cellar_west_lip"
    assert list(act.action) == UP


def test_stairs_tile_idles_exit_warp() -> None:
    snap = read_snapshot(_ram(x=EXIT_STAIRS[0], y=EXIT_STAIRS[1], tile=0x71))
    act = nose_cellar_cross_step(snap)
    assert act.reason == "cellar_exit_warp"
    assert list(act.action) == IDLE


def test_pit_tile_250_fails_closed_idle() -> None:
    ctl = make_nose_cellar_cross_controller()
    ctl.arrival_seen = True
    act = _step(ctl, _ram(x=112, y=141, tile=PIT_TILE))
    assert ctl.failed and not ctl.success
    assert "pit_tile_250" in ctl.notes
    assert act.reason == "pit_tile_250"
    assert list(act.action) == IDLE


def test_west_floor_tile_250_still_climbs() -> None:
    """C1: (48,189) reported tile 250; that is not the y=141 pit."""
    ctl = make_nose_cellar_cross_controller()
    ctl.arrival_seen = True
    act = _step(ctl, _ram(x=WEST_X, y=FLOOR_Y, tile=PIT_TILE))
    assert not ctl.failed
    assert act.reason == "cellar_west_climb"
    assert list(act.action) == UP


def test_pit_step_goes_east_not_left() -> None:
    act = nose_cellar_cross_step(read_snapshot(_ram(x=112, y=141, tile=PIT_TILE)))
    assert act.reason == "cellar_pit_to_east"
    assert list(act.action) == RIGHT
    assert list(act.action) != LEFT


def test_source_ladder_up_fails() -> None:
    """UP at the right/source ladder is CheckSubroom AttrB → 0x0D."""
    ctl = make_nose_cellar_cross_controller()
    ctl.arrival_seen = True
    ctl.on_floor = True
    # Force a west-climb reason at source x by being on west? No: at east floor
    # the step emits LEFT, not UP. Climb at east mid-height is east_drop DOWN.
    # Simulate a west-climb-shaped pose that is actually still on the source
    # column at mouth y after a mistaken UP: policy checks UP reasons.
    # Direct: emerge to 0x0D play is the miss.
    act = _step(ctl, _ram(mode=PLAY_MODE, screen=SOURCE_ROOM, x=96, y=157))
    assert ctl.failed and not ctl.success
    assert "returned_source_0x0d" in ctl.notes
    assert act.reason == "returned_source_0x0d"
    assert list(act.action) == IDLE


def test_emerge_requires_exact_attr_a_0x29() -> None:
    emerge = _ram(mode=PLAY_MODE, screen=DEST_ROOM, x=96, y=157)
    assert nose_cellar_cross_success(read_snapshot(emerge))
    still = _ram(mode=PASSAGE_MODE, screen=CELLAR_ROOM)
    assert not nose_cellar_cross_success(read_snapshot(still))
    back = _ram(mode=PLAY_MODE, screen=SOURCE_ROOM, x=96, y=157)
    assert not nose_cellar_cross_success(read_snapshot(back))

    ctl = make_nose_cellar_cross_controller()
    ctl.arrival_seen = True
    act = _step(ctl, emerge)
    assert ctl.success and not ctl.failed
    assert act.reason == "emerged_0x29"


def test_wrong_play_fails() -> None:
    ctl = make_nose_cellar_cross_controller()
    ctl.arrival_seen = True
    act = _step(ctl, _ram(mode=PLAY_MODE, screen=0x1A, x=96, y=157))
    assert ctl.failed and not ctl.success
    assert "wrong_play_0x1a" in ctl.notes
    assert list(act.action) == IDLE


def test_factory_report_is_fixture_live_not_route_eligible() -> None:
    ctl = make_nose_cellar_cross_controller()
    assert isinstance(ctl, Level7NoseCellarCrossController)
    report = ctl.report()
    assert report["route_eligible"] is False
    assert report["dest_screen"] == DEST_ROOM == 0x29
    assert report["writes"] == 0
    assert report["evidence"] == "fixture-live"
    assert report["natural_entry"] is False
    assert report["door"] == "STAIRS"
    assert report["spec_id"] == "level7_nose_cellar_0x7b"
    assert "OccupancyWalker" not in (ctl.__class__.__module__ + ctl.__class__.__name__)


def test_hops_factory_is_the_live_cross_not_the_spine_tip() -> None:
    """0x0D->0x7B and 0x7B->0x29 are two different live hops.

    ``make_tip_stairs_controller`` is the walk-on INTO cellar 0x7B
    (`level7.stairs0d`, live 2/2); the cross OUT of it to play 0x29 stays
    ``Level7NoseCellarCrossController``. The boss/leave factories stay
    fail-closed.
    """
    from zelda_i.level7.stairs0d import Level7Stairs0DController

    live = hops_make_nose_cellar_cross()
    assert isinstance(live, Level7NoseCellarCrossController)
    spine = make_tip_stairs_controller()
    assert isinstance(spine, Level7Stairs0DController)
    assert not isinstance(spine, Level7NoseCellarCrossController)
    assert spine.report()["dest_screen"] == 0x7B
    assert spine.report()["route_eligible"] is False
    aqua = make_aquamentus_heart_controller()
    leave = make_level7_shard_leave_controller()
    assert isinstance(aqua, UnverifiedLevel7PathController)
    assert isinstance(leave, UnverifiedLevel7PathController)


def test_no_occupancy_walker_import() -> None:
    import ast

    import zelda_i.level7.cellar as cellar

    assert not hasattr(cellar, "OccupancyWalker")
    source = cellar.__file__
    assert source is not None
    tree = ast.parse(open(source, encoding="utf-8").read())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
            if node.module:
                imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert "OccupancyWalker" not in imported
    assert "zelda_i.walk.physics" not in imported
    assert "cellar_cross_dir" not in imported

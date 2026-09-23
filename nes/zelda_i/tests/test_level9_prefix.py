"""Durable L9 0x76 north dest hop. No emulator.

x-align to 120, then UP. Dest is RAM (hyp 0x66); fail non-north and 0x07.
Natural old-man factory stays fail-closed. No RAM writes.
"""

from __future__ import annotations

import numpy as np
from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.door_graph import DoorDir, L9_ENTRY, LEVEL_9_NATURAL_DOOR_GRAPH
from zelda_i.level9.dungeon import ROOM_OLD_MAN_TF
from zelda_i.level9.natural_path import make_old_man_tf_gate_controller
from zelda_i.level9.prefix import (
    BOMB_NORTH_DEST_HYP,
    BOMB_NORTH_DEST_POSE,
    BOMB_NORTH_ORIGIN,
    BOMB_NORTH_STAND,
    CELLAR_60_DEST_HYP,
    CELLAR_60_DEST_POSE,
    CELLAR_60_ORIGIN,
    CELLAR_60_SOURCE_RETURN,
    CELLAR_70_DEST_HYP,
    CELLAR_70_DEST_POSE,
    CELLAR_70_ORIGIN,
    CELLAR_70_SOURCE_RETURN,
    NORTH_DEST_HYP,
    NORTH_DEST_POSE,
    NORTH_DOOR,
    NORTH_ORIGIN,
    RED_RING,
    STAIRS_55_DEST_HYP,
    STAIRS_55_DEST_POSE,
    STAIRS_55_ORIGIN,
    WEST_DEST_HYP,
    WEST_DEST_POSE,
    WEST_DOOR,
    WEST_ORIGIN,
    EAST_14_DEST_HYP,
    EAST_14_DEST_POSE,
    EAST_14_ORIGIN,
    EAST_15_DEST_HYP,
    EAST_15_DEST_POSE,
    EAST_15_ORIGIN,
    NORTH_16_DEST_HYP,
    NORTH_16_DEST_POSE,
    NORTH_16_ORIGIN,
    BOMB_WEST_06_DEST_HYP,
    BOMB_WEST_06_DEST_POSE,
    BOMB_WEST_06_ORIGIN,
    STAIRS_05_DEST_HYP,
    STAIRS_05_DEST_POSE,
    STAIRS_05_ORIGIN,
    STAIRS_61_DEST_HYP,
    STAIRS_61_ORIGIN,
    CELLAR_75_DEST_HYP,
    CELLAR_75_ORIGIN,
    CELLAR_75_SOURCE_RETURN,
    BOMB_NORTH_20_DEST_HYP,
    BOMB_NORTH_20_ORIGIN,
    Level9BombNorth20Controller,
    Level9BombNorth65Controller,
    Level9BombWest06Controller,
    Level9Cellar60Controller,
    Level9Cellar70Controller,
    Level9Cellar75Controller,
    Level9East14Controller,
    Level9East15Controller,
    Level9North16Controller,
    Level9North76Controller,
    Level9Stairs05Controller,
    Level9Stairs55Controller,
    Level9Stairs61Controller,
    is_east_neighbor,
    is_north_neighbor,
    is_west_neighbor,
    make_bomb_north_20_controller,
    make_bomb_north_65_controller,
    make_bomb_west_06_controller,
    make_cellar_60_controller,
    make_cellar_70_controller,
    make_cellar_75_controller,
    make_east_14_controller,
    make_east_15_controller,
    make_north_16_controller,
    make_north_76_controller,
    make_stairs_05_controller,
    make_stairs_55_controller,
    make_stairs_61_controller,
    make_west_66_controller,
    north_76_step,
    west_66_step,
)
from zelda_i.ram import PASSAGE_MODE, PLAY_MODE, read_snapshot
from zelda_i.tests.ram_helpers import make_ram

_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 9,
    "screen": NORTH_ORIGIN,
    "x": 120,
    "y": 205,
    "tile": 0,
    "keys": 9,
    "bombs": 16,
    "magic_key": 1,
    "triforce": 0xFF,
}

UP = list(nes_action("UP"))
DOWN = list(nes_action("DOWN"))
LEFT = list(nes_action("LEFT"))
RIGHT = list(nes_action("RIGHT"))
IDLE = list(nes_idle_action())


def _ram(**fields: int) -> np.ndarray:
    return make_ram(_DEFAULTS, **fields)


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


def test_natural_old_man_factory_wires_level9_north_76_controller() -> None:
    ctl = make_old_man_tf_gate_controller()
    assert isinstance(ctl, Level9North76Controller)
    assert ctl.dest == ROOM_OLD_MAN_TF == 0x66
    ram = _ram()
    before = ram.copy()
    act = ctl.step(read_snapshot(ram))
    assert np.array_equal(ram, before)
    assert not ctl.failed
    assert list(act.action) == UP
    assert act.reason == "north_76_push"
    assert ctl.max_frames == 4000


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
    bomb_n = LEVEL_9_NATURAL_DOOR_GRAPH.exit_between(
        0x65, 0x55, direction=DoorDir.UP
    )
    assert bomb_n is not None
    assert bomb_n.verification == "observed"
    stairs = LEVEL_9_NATURAL_DOOR_GRAPH.exit_between(
        0x55, 0x60, direction=DoorDir.UP
    )
    assert stairs is not None
    assert stairs.verification == "observed"
    cellar = LEVEL_9_NATURAL_DOOR_GRAPH.exit_between(
        0x60, 0x14, direction=DoorDir.LEFT
    )
    assert cellar is not None
    assert cellar.verification == "observed"
    east_14 = LEVEL_9_NATURAL_DOOR_GRAPH.exit_between(
        0x14, 0x15, direction=DoorDir.RIGHT
    )
    assert east_14 is not None
    assert east_14.verification == "observed"
    east_15 = LEVEL_9_NATURAL_DOOR_GRAPH.exit_between(
        0x15, 0x16, direction=DoorDir.RIGHT
    )
    assert east_15 is not None
    assert east_15.verification == "observed"


def test_bomb_north_65_approach_and_dest() -> None:
    ctl = make_bomb_north_65_controller()
    # East mouth leftover (224, 141) -> moves LEFT
    act = _step(ctl, _ram(screen=BOMB_NORTH_ORIGIN, x=224, y=141))
    assert not ctl.failed
    assert list(act.action) == LEFT
    assert act.reason == "approach_x"

    # Reaching waypoint (208, 141) -> switches to next waypoint, then moves UP
    act2 = _step(ctl, _ram(screen=BOMB_NORTH_ORIGIN, x=208, y=141))
    assert not ctl.failed
    assert act2.reason == "approach_next"
    act2b = _step(ctl, _ram(screen=BOMB_NORTH_ORIGIN, x=208, y=141))
    assert not ctl.failed
    assert list(act2b.action) == UP
    assert act2b.reason == "approach_y"

    # Reaching north band (208, 93) -> switches to next waypoint, then moves LEFT
    act3 = _step(ctl, _ram(screen=BOMB_NORTH_ORIGIN, x=208, y=93))
    assert not ctl.failed
    assert act3.reason == "approach_next"
    act3b = _step(ctl, _ram(screen=BOMB_NORTH_ORIGIN, x=208, y=93))
    assert not ctl.failed
    assert list(act3b.action) == LEFT
    assert act3b.reason == "approach_x"

    # Destination 0x55 settles cleanly
    ok = make_bomb_north_65_controller(dest=None)
    act_dest = _step(ok, _ram(mode=PLAY_MODE, screen=0x55, x=120, y=189))
    assert ok.success and not ok.failed
    assert is_north_neighbor(BOMB_NORTH_ORIGIN, BOMB_NORTH_DEST_HYP)
    assert BOMB_NORTH_DEST_HYP == 0x55
    assert BOMB_NORTH_DEST_POSE == (120, 189)
    assert BOMB_NORTH_STAND == (120, 93)


def test_bomb_north_65_rejects_red_ring_and_cellar() -> None:
    ring = make_bomb_north_65_controller(dest=None)
    act = _step(ring, _ram(mode=PLAY_MODE, screen=RED_RING, x=120, y=141))
    assert not ring.success and ring.failed
    assert "red_ring_0x07" in ring.notes

    cellar = make_bomb_north_65_controller(dest=None)
    act = _step(cellar, _ram(mode=PASSAGE_MODE, screen=0x60, x=136, y=141))
    assert not cellar.success and cellar.failed


def test_bomb_north_65_factory_report() -> None:
    ctl = make_bomb_north_65_controller()
    assert isinstance(ctl, Level9BombNorth65Controller)
    report = ctl.report()
    assert report["route_eligible"] is False
    assert report["natural_entry"] is False
    assert report["evidence"] == "fixture-live"
    assert report["writes"] == 0
    assert report["door"] == "UP"
    assert report["stand"] == [120, 93]
    assert report["spec_id"] == "level9_bomb_north_65"
    assert report["dest_hyp"] == BOMB_NORTH_DEST_HYP == 0x55
    assert report["dest_screen"] is None


def test_stairs_55_push_and_dest() -> None:
    ctl = make_stairs_55_controller()
    # Leftover in 0x55 with block unpushed (slot 11 at (96, 144))
    # Link at (120, 189) -> moves LEFT to align with push column x=96
    ram = _ram(screen=STAIRS_55_ORIGIN, x=120, y=189)
    ram[0x034F + 11] = 0x68
    ram[0x0485 + 11] = 176
    ram[0x0070 + 11] = 96
    ram[0x0084 + 11] = 144
    act = _step(ctl, ram)
    assert not ctl.failed
    assert list(act.action) == LEFT
    assert act.reason == "align_push_x"

    # Link at (96, 189) -> pushes UP
    ram[0x0070] = 96
    act2 = _step(ctl, ram)
    assert not ctl.failed
    assert list(act2.action) == UP
    assert act2.reason == "push_block_up"

    # Once block is pushed (y <= 128) and Link is at (96, 133), walks RIGHT to (128, 141)
    ram[0x0084 + 11] = 128
    ram[0x0070] = 96
    ram[0x0084] = 133
    act3 = _step(ctl, ram)
    assert not ctl.failed
    assert list(act3.action) == RIGHT
    assert act3.reason == "walk_stair_x"

    # Reaching (128, 133) -> moves DOWN to (128, 141)
    ram[0x0070] = 128
    act4 = _step(ctl, ram)
    assert not ctl.failed
    assert list(act4.action) == DOWN
    assert act4.reason == "walk_stair_y"

    # Destination cellar 0x60 in PASSAGE_MODE settles cleanly
    ok = make_stairs_55_controller(dest=None)
    act_dest = _step(ok, _ram(mode=PASSAGE_MODE, screen=0x60, x=192, y=93))
    assert ok.success and not ok.failed
    assert STAIRS_55_DEST_HYP == 0x60
    assert STAIRS_55_DEST_POSE == (192, 93)


def test_stairs_55_rejects_red_ring_and_play() -> None:
    ring = make_stairs_55_controller(dest=None)
    act = _step(ring, _ram(mode=PLAY_MODE, screen=RED_RING, x=120, y=141))
    assert not ring.success and ring.failed
    assert "red_ring_0x07" in ring.notes

    other_play = make_stairs_55_controller(dest=None)
    act = _step(other_play, _ram(mode=PLAY_MODE, screen=0x65, x=120, y=141))
    assert not other_play.success and other_play.failed
    assert "unexpected_play_0x65" in other_play.notes


def test_stairs_55_factory_report() -> None:
    ctl = make_stairs_55_controller()
    assert isinstance(ctl, Level9Stairs55Controller)
    report = ctl.report()
    assert report["route_eligible"] is False
    assert report["natural_entry"] is False
    assert report["evidence"] == "fixture-live"
    assert report["writes"] == 0
    assert report["spec_id"] == "level9_stairs_55"
    assert report["dest_hyp"] == STAIRS_55_DEST_HYP == 0x60
    assert report["dest_screen"] is None
    assert report["stairs"] == [128, 141]


def test_cellar_60_drop_walk_climb_and_dest() -> None:
    ctl = make_cellar_60_controller()
    # Leftover at right ladder (192, 93) -> moves DOWN to floor
    act = _step(ctl, _ram(mode=PASSAGE_MODE, screen=CELLAR_60_ORIGIN, x=192, y=93))
    assert not ctl.failed
    assert list(act.action) == DOWN
    assert act.reason == "cellar_east_drop"

    # Once on floor (y=189, x=192), walks LEFT towards x=48
    act2 = _step(ctl, _ram(mode=PASSAGE_MODE, screen=CELLAR_60_ORIGIN, x=192, y=189))
    assert not ctl.failed
    assert list(act2.action) == LEFT
    assert act2.reason == "cellar_floor_west"

    # Mid floor (y=189, x=120) -> continues LEFT
    act3 = _step(ctl, _ram(mode=PASSAGE_MODE, screen=CELLAR_60_ORIGIN, x=120, y=189))
    assert not ctl.failed
    assert list(act3.action) == LEFT
    assert act3.reason == "cellar_floor_west"

    # Arriving at west ladder (y=189, x=48) -> climbs UP
    act4 = _step(ctl, _ram(mode=PASSAGE_MODE, screen=CELLAR_60_ORIGIN, x=48, y=189))
    assert not ctl.failed
    assert list(act4.action) == UP
    assert act4.reason == "cellar_west_climb"

    # Climbing west ladder (y=140, x=48) -> moves UP
    act5 = _step(ctl, _ram(mode=PASSAGE_MODE, screen=CELLAR_60_ORIGIN, x=48, y=140))
    assert not ctl.failed
    assert list(act5.action) == UP
    assert act5.reason == "cellar_west_up"

    # Destination play 0x14 in PLAY_MODE settles cleanly
    ok = make_cellar_60_controller(dest=None)
    act_dest = _step(ok, _ram(mode=PLAY_MODE, screen=0x14, x=96, y=157))
    assert ok.success and not ok.failed
    assert CELLAR_60_DEST_HYP == 0x14
    assert CELLAR_60_DEST_POSE == (96, 157)


def test_cellar_60_rejects_red_ring_and_source_return() -> None:
    ring = make_cellar_60_controller(dest=None)
    act = _step(ring, _ram(mode=PLAY_MODE, screen=RED_RING, x=120, y=141))
    assert not ring.success and ring.failed
    assert "red_ring_0x07" in ring.notes

    source = make_cellar_60_controller(dest=None)
    act = _step(source, _ram(mode=PLAY_MODE, screen=CELLAR_60_SOURCE_RETURN, x=120, y=141))
    assert not source.success and source.failed
    assert "returned_source_0x55" in source.notes

    unexpected = make_cellar_60_controller(dest=None)
    act = _step(unexpected, _ram(mode=PLAY_MODE, screen=0x65, x=120, y=141))
    assert not unexpected.success and unexpected.failed
    assert "unexpected_dest_0x65" in unexpected.notes


def test_cellar_60_factory_report() -> None:
    ctl = make_cellar_60_controller()
    assert isinstance(ctl, Level9Cellar60Controller)
    report = ctl.report()
    assert report["route_eligible"] is False
    assert report["natural_entry"] is False
    assert report["evidence"] == "fixture-live"
    assert report["writes"] == 0
    assert report["door"] == "STAIRS"
    assert report["spec_id"] == "level9_cellar_60"
    assert report["dest_hyp"] == CELLAR_60_DEST_HYP == 0x14
    assert report["dest_screen"] is None


def test_cellar_70_drop_walk_climb_and_dest() -> None:
    ctl = make_cellar_70_controller()
    # Leftover at right ladder (192, 93) -> moves DOWN to floor
    act = _step(ctl, _ram(mode=PASSAGE_MODE, screen=CELLAR_70_ORIGIN, x=192, y=93))
    assert not ctl.failed
    assert list(act.action) == DOWN
    assert act.reason == "cellar_east_drop"

    # Once on floor (y=189, x=192), walks LEFT towards x=48
    act2 = _step(ctl, _ram(mode=PASSAGE_MODE, screen=CELLAR_70_ORIGIN, x=192, y=189))
    assert not ctl.failed
    assert list(act2.action) == LEFT
    assert act2.reason == "cellar_floor_west"

    # Arriving at west ladder (y=189, x=48) -> climbs UP
    act3 = _step(ctl, _ram(mode=PASSAGE_MODE, screen=CELLAR_70_ORIGIN, x=48, y=189))
    assert not ctl.failed
    assert list(act3.action) == UP
    assert act3.reason == "cellar_west_climb"

    # Climbing west ladder (y=140, x=48) -> moves UP
    act4 = _step(ctl, _ram(mode=PASSAGE_MODE, screen=CELLAR_70_ORIGIN, x=48, y=140))
    assert not ctl.failed
    assert list(act4.action) == UP
    assert act4.reason == "cellar_west_up"

    # Destination play 0x63 in PLAY_MODE settles cleanly
    ok = make_cellar_70_controller(dest=None)
    act_dest = _step(ok, _ram(mode=PLAY_MODE, screen=0x63, x=160, y=157))
    assert ok.success and not ok.failed
    assert CELLAR_70_DEST_HYP == 0x63
    assert CELLAR_70_DEST_POSE == (160, 157)


def test_cellar_70_rejects_red_ring_and_source_return() -> None:
    ring = make_cellar_70_controller(dest=None)
    act = _step(ring, _ram(mode=PLAY_MODE, screen=RED_RING, x=120, y=141))
    assert not ring.success and ring.failed
    assert "red_ring_0x07" in ring.notes

    source = make_cellar_70_controller(dest=None)
    act = _step(source, _ram(mode=PLAY_MODE, screen=CELLAR_70_SOURCE_RETURN, x=120, y=141))
    assert not source.success and source.failed
    assert "returned_source_0x05" in source.notes

    unexpected = make_cellar_70_controller(dest=None)
    act = _step(unexpected, _ram(mode=PLAY_MODE, screen=0x15, x=120, y=141))
    assert not unexpected.success and unexpected.failed
    assert "unexpected_dest_0x15" in unexpected.notes


def test_cellar_70_factory_report() -> None:
    ctl = make_cellar_70_controller()
    assert isinstance(ctl, Level9Cellar70Controller)
    report = ctl.report()
    assert report["route_eligible"] is False
    assert report["natural_entry"] is False
    assert report["evidence"] == "fixture-live"
    assert report["writes"] == 0
    assert report["door"] == "STAIRS"
    assert report["spec_id"] == "level9_cellar_70"
    assert report["dest_hyp"] == CELLAR_70_DEST_HYP == 0x63
    assert report["dest_screen"] is None


def test_east_14_waypoints_and_dest() -> None:
    ctl = make_east_14_controller()
    # Leftover at (96, 157) -> moves RIGHT towards WP 0 (176, 157)
    act = _step(ctl, _ram(mode=PLAY_MODE, screen=EAST_14_ORIGIN, x=96, y=157))
    assert not ctl.failed
    assert list(act.action) == RIGHT
    assert "walk_wp_0_x" in act.reason

    # Reaching WP 0 (176, 157) advances index
    act2 = _step(ctl, _ram(mode=PLAY_MODE, screen=EAST_14_ORIGIN, x=176, y=157))
    assert not ctl.failed
    assert act2.reason == "reach_wp_1"

    # From (176, 157) moves UP towards WP 1 (176, 93)
    act3 = _step(ctl, _ram(mode=PLAY_MODE, screen=EAST_14_ORIGIN, x=176, y=157))
    assert not ctl.failed
    assert list(act3.action) == UP
    assert "walk_wp_1_y" in act3.reason

    # Destination play 0x15 in PLAY_MODE settles cleanly
    ok = make_east_14_controller(dest=None)
    act_dest = _step(ok, _ram(mode=PLAY_MODE, screen=0x15, x=16, y=141))
    assert ok.success and not ok.failed
    assert EAST_14_DEST_HYP == 0x15
    assert EAST_14_DEST_POSE == (16, 141)


def test_east_14_rejects_red_ring_and_unexpected_play() -> None:
    ring = make_east_14_controller(dest=None)
    act = _step(ring, _ram(mode=PLAY_MODE, screen=RED_RING, x=120, y=141))
    assert not ring.success and ring.failed
    assert "red_ring_0x07" in ring.notes

    unexpected = make_east_14_controller(dest=None)
    act = _step(unexpected, _ram(mode=PLAY_MODE, screen=0x24, x=120, y=141))
    assert not unexpected.success and unexpected.failed
    assert "unexpected_dest_0x24" in unexpected.notes


def test_east_14_factory_report() -> None:
    ctl = make_east_14_controller()
    assert isinstance(ctl, Level9East14Controller)
    report = ctl.report()
    assert report["route_eligible"] is False
    assert report["natural_entry"] is False
    assert report["evidence"] == "fixture-live"
    assert report["writes"] == 0
    assert report["door"] == "RIGHT"
    assert report["spec_id"] == "level9_east_14"
    assert report["dest_hyp"] == EAST_14_DEST_HYP == 0x15
    assert report["dest_screen"] is None


def test_east_15_step_and_dest() -> None:
    ctl = make_east_15_controller()
    # Leftover at (16, 141) -> moves RIGHT
    act = _step(ctl, _ram(mode=PLAY_MODE, screen=EAST_15_ORIGIN, x=16, y=141))
    assert not ctl.failed
    assert list(act.action) == RIGHT
    assert "east_15" in act.reason

    # Destination play 0x16 in PLAY_MODE settles cleanly
    ok = make_east_15_controller(dest=None)
    act_dest = _step(ok, _ram(mode=PLAY_MODE, screen=0x16, x=32, y=141))
    assert ok.success and not ok.failed
    assert EAST_15_DEST_HYP == 0x16
    assert EAST_15_DEST_POSE == (32, 141)
    assert is_east_neighbor(EAST_15_ORIGIN, EAST_15_DEST_HYP)


def test_east_15_rejects_red_ring_and_unexpected_play() -> None:
    ring = make_east_15_controller(dest=None)
    act = _step(ring, _ram(mode=PLAY_MODE, screen=RED_RING, x=120, y=141))
    assert not ring.success and ring.failed
    assert "red_ring_0x07" in ring.notes

    unexpected = make_east_15_controller(dest=None)
    act = _step(unexpected, _ram(mode=PLAY_MODE, screen=0x17, x=120, y=141))
    assert not unexpected.success and unexpected.failed
    assert "unexpected_dest_0x17" in unexpected.notes


def test_east_15_factory_report() -> None:
    ctl = make_east_15_controller()
    assert isinstance(ctl, Level9East15Controller)
    report = ctl.report()
    assert report["route_eligible"] is False
    assert report["natural_entry"] is False
    assert report["evidence"] == "fixture-live"
    assert report["writes"] == 0
    assert report["door"] == "RIGHT"
    assert report["spec_id"] == "level9_east_15"
    assert report["dest_hyp"] == EAST_15_DEST_HYP == 0x16
    assert report["dest_screen"] is None


def test_north_16_step_and_dest() -> None:
    ctl = make_north_16_controller()
    # Leftover at (16, 141) -> link_x != 120 -> moves RIGHT
    act = _step(ctl, _ram(mode=PLAY_MODE, screen=NORTH_16_ORIGIN, x=16, y=141))
    assert not ctl.failed
    assert list(act.action) == RIGHT
    assert "north_16_align_x" in act.reason

    # When aligned to x=120 -> moves UP
    act_up = _step(ctl, _ram(mode=PLAY_MODE, screen=NORTH_16_ORIGIN, x=120, y=141))
    assert not ctl.failed
    assert list(act_up.action) == UP
    assert "north_16_push" in act_up.reason

    # Destination play 0x06 in PLAY_MODE settles cleanly
    ok = make_north_16_controller(dest=None)
    act_dest = _step(ok, _ram(mode=PLAY_MODE, screen=0x06, x=120, y=205))
    assert ok.success and not ok.failed
    assert NORTH_16_DEST_HYP == 0x06
    assert NORTH_16_DEST_POSE == (120, 205)
    assert is_north_neighbor(NORTH_16_ORIGIN, NORTH_16_DEST_HYP)


def test_north_16_rejects_red_ring_and_unexpected_play() -> None:
    ring = make_north_16_controller(dest=None)
    act = _step(ring, _ram(mode=PLAY_MODE, screen=RED_RING, x=120, y=141))
    assert not ring.success and ring.failed
    assert "red_ring_0x07" in ring.notes

    unexpected = make_north_16_controller(dest=None)
    act = _step(unexpected, _ram(mode=PLAY_MODE, screen=0x26, x=120, y=141))
    assert not unexpected.success and unexpected.failed
    assert "unexpected_dest_0x26" in unexpected.notes


def test_north_16_factory_report() -> None:
    ctl = make_north_16_controller()
    assert isinstance(ctl, Level9North16Controller)
    report = ctl.report()
    assert report["route_eligible"] is False
    assert report["natural_entry"] is False
    assert report["evidence"] == "fixture-live"
    assert report["writes"] == 0
    assert report["door"] == "UP"
    assert report["spec_id"] == "level9_north_16"
    assert report["dest_hyp"] == NORTH_16_DEST_HYP == 0x06
    assert report["dest_screen"] is None


def test_bomb_west_06_step_and_dest() -> None:
    ctl = make_bomb_west_06_controller()
    # Leftover at (120, 205) -> link_y > 189 -> moves UP towards first waypoint (120, 189)
    act = _step(ctl, _ram(mode=PLAY_MODE, screen=BOMB_WEST_06_ORIGIN, x=120, y=205))
    assert not ctl.failed
    assert list(act.action) == UP
    assert "approach" in act.reason

    # Destination play 0x05 in PLAY_MODE settles cleanly
    ok = make_bomb_west_06_controller(dest=None)
    act_dest = _step(ok, _ram(mode=PLAY_MODE, screen=0x05, x=208, y=141))
    assert ok.success and not ok.failed
    assert BOMB_WEST_06_DEST_HYP == 0x05
    assert BOMB_WEST_06_DEST_POSE == (208, 141)
    assert is_west_neighbor(BOMB_WEST_06_ORIGIN, BOMB_WEST_06_DEST_HYP)


def test_bomb_west_06_rejects_red_ring_and_unexpected_play() -> None:
    ring = make_bomb_west_06_controller(dest=None)
    act = _step(ring, _ram(mode=PLAY_MODE, screen=RED_RING, x=120, y=141))
    assert not ring.success and ring.failed
    assert "red_ring_0x07" in ring.notes

    unexpected = make_bomb_west_06_controller(dest=None)
    act = _step(unexpected, _ram(mode=PLAY_MODE, screen=0x15, x=120, y=141))
    assert not unexpected.success and unexpected.failed
    assert "not_west_neighbor_0x15" in unexpected.notes


def test_bomb_west_06_factory_report() -> None:
    ctl = make_bomb_west_06_controller()
    assert isinstance(ctl, Level9BombWest06Controller)
    report = ctl.report()
    assert report["route_eligible"] is False
    assert report["natural_entry"] is False
    assert report["evidence"] == "fixture-live"
    assert report["writes"] == 0
    assert report["door"] == "LEFT"
    assert report["spec_id"] == "level9_bomb_west_06"
    assert report["dest_hyp"] == BOMB_WEST_06_DEST_HYP == 0x05
    assert report["dest_screen"] is None
    assert report["stand"] == [48, 141]


def test_stairs_05_steps_off_the_bombed_door_row_first() -> None:
    """bomb_west_06 lands Link at (208,141), inside the hole it just blew.

    Two Wizzrobes camp in the east wall on that row, so they are the nearest
    target and engaging from there chases Link back out into 0x06. The hop
    must drop off the row before doing anything else.
    """
    ctl = make_stairs_05_controller()
    # Wizzrobes cleared (the hop gates the push on the ROM all-dead flag).
    ram = _ram(screen=STAIRS_05_ORIGIN, x=208, y=141, room_all_dead=1)
    ram[0x034F + 11] = 0x68
    ram[0x0485 + 11] = 176
    ram[0x0070 + 11] = 96
    ram[0x0084 + 11] = 144
    act = _step(ctl, ram)
    assert not ctl.failed
    assert list(act.action) == DOWN
    assert act.reason == "leave_east_doorway"

    # Off the row, normal behaviour resumes.
    ram[0x0084] = 173
    act2 = _step(ctl, ram)
    assert not ctl.failed
    assert act2.reason == "align_push_x"


def test_stairs_05_push_and_dest() -> None:
    ctl = make_stairs_05_controller()
    # Leftover in 0x05 at (208, 173) with block unpushed (slot 11 at (96, 144))
    # Link at (208, 173) -> moves LEFT to align with push column x=96
    ram = _ram(screen=STAIRS_05_ORIGIN, x=208, y=173, room_all_dead=1)
    ram[0x034F + 11] = 0x68
    ram[0x0485 + 11] = 176
    ram[0x0070 + 11] = 96
    ram[0x0084 + 11] = 144
    act = _step(ctl, ram)
    assert not ctl.failed
    assert list(act.action) == LEFT
    assert act.reason == "align_push_x"

    # Link at (96, 173) -> pushes UP
    ram[0x0070] = 96
    act2 = _step(ctl, ram)
    assert not ctl.failed
    assert list(act2.action) == UP
    assert act2.reason == "push_block_up"

    # Once block is pushed (y <= 128) and Link is at (96, 173), walks RIGHT to (208, 96)
    ram[0x0084 + 11] = 128
    ram[0x0070] = 96
    ram[0x0084] = 173
    act3 = _step(ctl, ram)
    assert not ctl.failed
    assert list(act3.action) == RIGHT
    assert act3.reason == "walk_stair_x"

    # Link at (208, 173) -> moves UP to (208, 96)
    ram[0x0070] = 208
    ram[0x0084] = 173
    act4 = _step(ctl, ram)
    assert not ctl.failed
    assert list(act4.action) == UP
    assert act4.reason == "walk_stair_y"

    # Link at (208, 96) -> stands on stairs holding UP to trigger stair descend
    ram[0x0070] = 208
    ram[0x0084] = 96
    act5 = _step(ctl, ram)
    assert not ctl.failed
    assert list(act5.action) == UP
    assert act5.reason == "stand_on_stairs"

    # Destination cellar 0x70 in PASSAGE_MODE settles cleanly
    ok = make_stairs_05_controller(dest=None)
    act_dest = _step(ok, _ram(mode=PASSAGE_MODE, screen=0x70, x=192, y=93))
    assert ok.success and not ok.failed
    assert STAIRS_05_DEST_HYP == 0x70
    assert STAIRS_05_DEST_POSE == (192, 93)


def test_stairs_05_rejects_red_ring_and_play() -> None:
    ring = make_stairs_05_controller(dest=None)
    act = _step(ring, _ram(mode=PLAY_MODE, screen=RED_RING, x=120, y=141))
    assert not ring.success and ring.failed
    assert "red_ring_0x07" in ring.notes

    other_play = make_stairs_05_controller(dest=None)
    act = _step(other_play, _ram(mode=PLAY_MODE, screen=0x06, x=120, y=141))
    assert not other_play.success and other_play.failed
    assert "unexpected_play_0x06" in other_play.notes


def test_stairs_05_factory_report() -> None:
    ctl = make_stairs_05_controller()
    assert isinstance(ctl, Level9Stairs05Controller)
    report = ctl.report()
    assert report["route_eligible"] is False
    assert report["natural_entry"] is False
    assert report["evidence"] == "fixture-live"
    assert report["writes"] == 0
    assert report["spec_id"] == "level9_stairs_05"
    assert report["dest_hyp"] == STAIRS_05_DEST_HYP == 0x70
    assert report["dest_screen"] is None
    assert report["stairs"] == [208, 96]
    assert report["pushed"] is False


def test_stairs_61_policy_and_factory() -> None:
    ctl = make_stairs_61_controller()
    assert isinstance(ctl, Level9Stairs61Controller)
    assert ctl.origin == STAIRS_61_ORIGIN == 0x61
    assert ctl.dest_hyp == STAIRS_61_DEST_HYP == 0x75
    report = ctl.report()
    assert report["spec_id"] == "level9_stairs_61"
    assert report["stairs"] == [128, 141]

    # Reject red ring
    ring = make_stairs_61_controller()
    _step(ring, _ram(mode=PLAY_MODE, screen=RED_RING, x=120, y=141))
    assert ring.failed and "red_ring_0x07" in ring.notes

    # Destination cellar 0x75
    ok = make_stairs_61_controller()
    _step(ok, _ram(mode=PASSAGE_MODE, screen=0x75, x=192, y=93))
    assert ok.success and not ok.failed


def test_cellar_75_policy_and_factory() -> None:
    ctl = make_cellar_75_controller()
    assert isinstance(ctl, Level9Cellar75Controller)
    assert ctl.origin == CELLAR_75_ORIGIN == 0x75
    assert ctl.dest_hyp == CELLAR_75_DEST_HYP == 0x20
    assert ctl.source_return == CELLAR_75_SOURCE_RETURN == 0x61

    # Reject source return 0x61
    ret = make_cellar_75_controller()
    _step(ret, _ram(mode=PLAY_MODE, screen=0x61, x=128, y=141))
    assert ret.failed and "returned_source_0x61" in ret.notes

    # Arrive in play 0x20
    ok = make_cellar_75_controller()
    _step(ok, _ram(mode=PLAY_MODE, screen=0x20, x=96, y=157))
    assert ok.success and not ok.failed


def test_bomb_north_20_policy_and_factory() -> None:
    ctl = make_bomb_north_20_controller()
    assert isinstance(ctl, Level9BombNorth20Controller)
    assert ctl.origin == BOMB_NORTH_20_ORIGIN == 0x20
    assert ctl.dest_hyp == BOMB_NORTH_20_DEST_HYP == 0x10
    assert ctl.stand_coords == (120, 93)

    # Reject non-north neighbor
    bad = make_bomb_north_20_controller()
    _step(bad, _ram(mode=PLAY_MODE, screen=0x21, x=120, y=93))
    assert bad.failed and "not_north_neighbor_0x21" in bad.notes

    # Arrive in play 0x10
    ok = make_bomb_north_20_controller()
    _step(ok, _ram(mode=PLAY_MODE, screen=0x10, x=120, y=189))
    assert ok.success and not ok.failed


"""Unit tests for the L2 door-path heart-cost model (no emulator)."""

from __future__ import annotations

from zelda_i.health_cost import (
    HEART_MANAGEMENT_CONSTRAINT,
    L2_DOOR_DEATH_SCREEN,
    L2_DOOR_PATH_HOP_COSTS,
    L2_DOOR_PATH_START_FILLED,
    PATH_L2_DOOR,
    PATH_L2_PREFIX,
    HopCost,
    ResourceState,
    decode_health,
    encode_health,
    hop_costs_for_path,
    requires_assist_or_farm,
    resources_from_snapshot,
    route_leg_needs_heart_management,
    simulate_corridor,
)
from zelda_i.ram import ZeldaSnapshot
from zelda_i.route_legs import level2_door_path_route_legs


def _snap(*, health: int, bombs: int = 0, keys: int = 0) -> ZeldaSnapshot:
    return ZeldaSnapshot(
        mode=5,
        level=0,
        screen=0x37,
        next_screen=0x37,
        link_x=112,
        link_y=125,
        facing=1,
        sword=1,
        bombs=bombs,
        rupees=0,
        keys=keys,
        health=health,
        triforce=1,
        compass=0,
        dialog_timer=0,
        colliding_tile=0x26,
        room_item_id=0,
        room_all_dead=0,
        room_obj_count=0,
        cur_opened_doors=0,
        open_doorway_mask=0,
        objects=(),
    )


def test_health_byte_matches_ram_nibbles() -> None:
    # HeartValues: low nibble is whole hearts (0-based), not a 0xF flag.
    assert encode_health(4, 3) == 0x32
    assert encode_health(4, 4) == 0x33
    assert encode_health(3, 3) == 0x22
    assert decode_health(0x33) == (4, 4)
    assert decode_health(0x32) == (4, 3)
    assert decode_health(0x3F) == (4, 4)
    state = ResourceState.from_health_byte(0x32, bombs=2, keys=1)
    assert state.filled == 3
    assert state.containers == 4
    assert state.health_byte == 0x32


def test_resources_from_snapshot_treats_full_sentinel() -> None:
    full = resources_from_snapshot(_snap(health=0x3F, bombs=4, keys=2))
    assert full.filled == 4
    assert full.bombs == 4
    assert full.keys == 2
    damaged = resources_from_snapshot(_snap(health=0x30))
    assert damaged.filled == 1


def test_l2_door_path_costs_match_status_arrivals() -> None:
    arrivals = {hop.screen: hop.hearts_lost for hop in L2_DOOR_PATH_HOP_COSTS}
    assert list(arrivals) == [0x38, 0x48, 0x58, 0x59, 0x5A, 0x5B, 0x5C]
    assert arrivals[0x38] == 0
    assert arrivals[0x48] == 0
    assert arrivals[0x58] == 1
    assert arrivals[0x59] == 0
    assert arrivals[0x5A] == 1
    assert arrivals[0x5B] == 0
    assert arrivals[0x5C] == 1
    assert hop_costs_for_path(PATH_L2_DOOR) == L2_DOOR_PATH_HOP_COSTS
    assert hop_costs_for_path(PATH_L2_PREFIX) == ()


def test_simulate_from_three_hearts_dies_on_0x5c() -> None:
    survives, death, remaining = simulate_corridor(L2_DOOR_PATH_START_FILLED)
    assert survives is False
    assert death == L2_DOOR_DEATH_SCREEN == 0x5C
    assert remaining.filled == 0
    assert remaining.containers == 4


def test_simulate_from_health_byte_3_of_4() -> None:
    start = ResourceState.from_health_byte(0x32)
    survives, death, remaining = simulate_corridor(start)
    assert (survives, death, remaining.filled) == (False, 0x5C, 0)


def test_simulate_from_four_hearts_reaches_5c_with_one() -> None:
    survives, death, remaining = simulate_corridor(4)
    # Measured hop losses total 3; 4/4 arrives at 0x5C with 1. Maze unmeasured.
    assert survives is True
    assert death is None
    assert remaining.filled == 1
    assert requires_assist_or_farm(PATH_L2_DOOR, 4) is True


def test_requires_assist_or_farm_for_l2_door_from_3_or_4() -> None:
    assert requires_assist_or_farm(PATH_L2_DOOR) is True
    assert requires_assist_or_farm("walk_level2_door_path", 3) is True
    assert requires_assist_or_farm("zelda_level2_door_path", 4) is True
    assert requires_assist_or_farm(PATH_L2_PREFIX) is False
    assert requires_assist_or_farm("unknown_path") is False


def test_consumes_existing_door_path_constraint() -> None:
    walk = next(
        leg
        for leg in level2_door_path_route_legs()
        if leg.leg_id == "walk_level2_door_path"
    )
    assert HEART_MANAGEMENT_CONSTRAINT in walk.constraints
    assert route_leg_needs_heart_management(walk)
    enter = next(
        leg
        for leg in level2_door_path_route_legs()
        if leg.leg_id == "enter_level2_dungeon"
    )
    assert not route_leg_needs_heart_management(enter)


def test_empty_or_already_dead_corridor() -> None:
    survives, death, remaining = simulate_corridor(3, hops=())
    assert survives is True
    assert death is None
    assert remaining.filled == 3
    dead = simulate_corridor(0)
    assert dead[0] is False
    assert dead[1] is None


def test_custom_hops_preserve_bombs_and_keys() -> None:
    start = ResourceState(containers=4, filled=2, bombs=3, keys=1)
    hops = (HopCost(0x10, 1, "probe", "unit"),)
    survives, death, remaining = simulate_corridor(start, hops)
    assert survives is True
    assert death is None
    assert remaining == ResourceState(containers=4, filled=1, bombs=3, keys=1)

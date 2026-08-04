from __future__ import annotations

from zelda_i.overworld import (
    LEVEL1_PATH_SCREENS,
    LEVEL2_PATH_SCREENS,
    NODE_LEVEL1_DUNGEON,
    NODE_LEVEL1_COMPLETE,
    NODE_LEVEL1_ENTRANCE,
    NODE_LEVEL1_EXIT_OVERWORLD,
    NODE_LEVEL1_FIRST_KEY,
    NODE_LEVEL1_FIRST_KEY_ROOM,
    NODE_LEVEL1_NORTH_CLEARED,
    NODE_LEVEL1_NORTH_ROOM,
    NODE_LEVEL1_ROOM_53,
    NODE_LEVEL1_ROOM_53_CLEARED,
    NODE_LEVEL1_ROOM_54,
    NODE_LEVEL1_ROOM_54_CLEARED,
    NODE_LEVEL2_PATH_4A,
    NODE_START,
    NODE_SWORD_CAVE,
    SCREEN_START,
    build_early_route_graph,
    neighbor_screens,
    screen_to_grid,
)
from zelda_i.route_legs import (
    early_route_plan,
    level1_clear53_route_plan,
    level1_clear54_route_plan,
    level1_complete_route_plan,
    level1_clear63_route_plan,
    level1_first_key_route_plan,
    level1_north_route_plan,
    level1_route_plan,
    level2_path_prefix_route_plan,
    sword_cave_route_legs,
)


def test_start_screen_grid() -> None:
    col, row = screen_to_grid(SCREEN_START)
    assert (col, row) == (7, 7)
    neighbors = neighbor_screens(SCREEN_START)
    assert neighbors["north"] == 0x67
    assert neighbors["south"] is None


def test_early_graph_has_sword_cave_portal() -> None:
    graph = build_early_route_graph()
    assert NODE_SWORD_CAVE in graph.nodes
    edge = graph.edge_for(NODE_START, NODE_SWORD_CAVE)
    assert edge is not None
    assert edge.verification == "observed"


def test_sword_route_plan_acquires_sword() -> None:
    planned = early_route_plan()
    assert len(planned) == 2
    assert planned[-1].capabilities_after == frozenset({"wooden_sword"})
    assert [leg.leg.leg_id for leg in planned] == [
        "enter_sword_cave",
        "take_wooden_sword_and_exit",
    ]


def test_sword_legs_match_route_helper() -> None:
    legs = sword_cave_route_legs()
    assert legs[0].source_id == NODE_START
    assert legs[0].target_id == NODE_SWORD_CAVE


def test_level1_path_screens_chain() -> None:
    assert LEVEL1_PATH_SCREENS[0] == 0x77
    assert LEVEL1_PATH_SCREENS[-1] == 0x37
    # consecutive screens are grid neighbors
    for a, b in zip(LEVEL1_PATH_SCREENS, LEVEL1_PATH_SCREENS[1:]):
        assert b in neighbor_screens(a).values()


def test_level1_route_plan_reaches_dungeon() -> None:
    planned = level1_route_plan()
    assert planned[-1].leg.target_id == NODE_LEVEL1_DUNGEON
    assert "wooden_sword" in planned[-1].capabilities_before
    assert any(leg.leg.acquires for leg in planned)  # sword acquire


def test_level1_portal_edge_observed() -> None:
    graph = build_early_route_graph()
    edge = graph.edge_for(NODE_LEVEL1_ENTRANCE, NODE_LEVEL1_DUNGEON)
    assert edge is not None
    assert edge.verification == "observed"
    assert "wooden_sword" in edge.requires


def test_level1_first_key_route_plan_acquires_key() -> None:
    planned = level1_first_key_route_plan()
    assert planned[-1].leg.target_id == NODE_LEVEL1_FIRST_KEY
    assert planned[-1].leg.source_id == NODE_LEVEL1_FIRST_KEY_ROOM
    assert "keys" in planned[-1].capabilities_after
    assert planned[-1].edge.verification == "observed"


def test_level1_north_route_plan_spends_key_at_observed_door() -> None:
    planned = level1_north_route_plan()
    assert planned[-1].leg.target_id == NODE_LEVEL1_NORTH_ROOM
    assert "keys" in planned[-1].effective_requires
    assert planned[-1].leg.constraints == ("consumes_one_key",)
    assert planned[-1].edge.verification == "observed"


def test_level1_clear63_route_plan_clears_north_room() -> None:
    planned = level1_clear63_route_plan()
    assert planned[-1].leg.target_id == NODE_LEVEL1_NORTH_CLEARED
    assert planned[-1].leg.source_id == NODE_LEVEL1_NORTH_ROOM
    assert planned[-1].edge.verification == "observed"
    assert planned[-1].edge.meta.get("reward") == "none"


def test_level1_clear53_route_plan_collects_room_key() -> None:
    planned = level1_clear53_route_plan()
    assert planned[-2].leg.target_id == NODE_LEVEL1_ROOM_53
    assert planned[-1].leg.source_id == NODE_LEVEL1_ROOM_53
    assert planned[-1].leg.target_id == NODE_LEVEL1_ROOM_53_CLEARED
    assert planned[-1].edge.verification == "observed"
    assert planned[-1].edge.meta.get("reward") == "small_key"
    assert "keys" in planned[-1].capabilities_after


def test_level1_clear54_route_plan_clears_east_keese_room() -> None:
    planned = level1_clear54_route_plan()
    assert planned[-2].leg.target_id == NODE_LEVEL1_ROOM_54
    assert planned[-1].leg.source_id == NODE_LEVEL1_ROOM_54
    assert planned[-1].leg.target_id == NODE_LEVEL1_ROOM_54_CLEARED
    assert planned[-1].edge.verification == "observed"
    assert planned[-1].edge.meta.get("enemies") == "8_keese"


def test_level1_complete_route_plan_reaches_first_triforce_shard() -> None:
    planned = level1_complete_route_plan()
    assert planned[-1].leg.target_id == NODE_LEVEL1_COMPLETE
    assert planned[-1].edge.verification == "observed"
    assert planned[-1].edge.meta.get("segment") == "level1_complete"
    assert "triforce_shard_1" in planned[-1].capabilities_after


def test_level2_path_prefix_route_plan_reaches_4a() -> None:
    planned = level2_path_prefix_route_plan()
    assert planned[-1].leg.target_id == NODE_LEVEL2_PATH_4A
    assert planned[-1].leg.source_id == NODE_LEVEL1_EXIT_OVERWORLD
    assert "triforce_shard_1" in planned[-1].capabilities_before
    assert LEVEL2_PATH_SCREENS[-1] == 0x4A

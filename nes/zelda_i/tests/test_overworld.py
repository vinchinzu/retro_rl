from __future__ import annotations

from zelda_i.overworld import (
    LEVEL1_PATH_SCREENS,
    LEVEL2_DOOR_SCREENS,
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
    NODE_LEVEL2_DUNGEON,
    NODE_LEVEL2_ENTRANCE,
    NODE_LEVEL2_PATH_4A,
    NODE_START,
    NODE_SWORD_CAVE,
    SCREEN_START,
    build_early_route_graph,
    neighbor_screens,
    node_id_for_screen,
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
    level2_door_path_route_plan,
    level2_path_prefix_route_plan,
    sword_cave_route_legs,
)
from zelda_i.routes import get_route, list_routes


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


def test_level2_door_path_screens_in_graph() -> None:
    """Door path 0x37→…→0x3C is fully node/edge-covered (planned beyond prefix)."""
    graph = build_early_route_graph()
    assert LEVEL2_DOOR_SCREENS[0] == 0x37
    assert LEVEL2_DOOR_SCREENS[-1] == 0x3C
    assert LEVEL2_DOOR_SCREENS == (
        0x37,
        0x38,
        0x48,
        0x58,
        0x59,
        0x5A,
        0x5B,
        0x5C,
        0x5D,
        0x4D,
        0x4C,
        0x3C,
    )
    for screen in LEVEL2_DOOR_SCREENS:
        assert node_id_for_screen(screen) in graph.nodes

    # Shared prefix hops stay observed (Clean 0x4A walk).
    prefix_pairs = set(zip(LEVEL2_PATH_SCREENS, LEVEL2_PATH_SCREENS[1:]))
    for a, b in zip(LEVEL2_DOOR_SCREENS, LEVEL2_DOOR_SCREENS[1:]):
        edge = graph.edge_for(node_id_for_screen(a), node_id_for_screen(b))
        assert edge is not None, f"missing edge {a:02X}->{b:02X}"
        if (a, b) in prefix_pairs:
            assert edge.verification == "observed"
            assert edge.meta.get("segment") == "to_level2_prefix"
        else:
            assert edge.verification == "planned"
            assert edge.provenance == "probe_geometry"
            assert edge.meta.get("segment") == "to_level2_door"

    # Maze hop is tagged (controller still open).
    maze = graph.edge_for("ow_5c", "ow_5d")
    assert maze is not None
    assert maze.meta.get("requires_maze") is True


def test_level2_door_path_route_plan_reaches_dungeon() -> None:
    planned = level2_door_path_route_plan()
    assert planned[-1].leg.target_id == NODE_LEVEL2_DUNGEON
    assert planned[-1].leg.source_id == NODE_LEVEL2_ENTRANCE
    assert planned[-1].edge.verification == "planned"
    assert planned[-2].leg.target_id == NODE_LEVEL2_ENTRANCE
    assert planned[-2].edge.verification == "planned"
    assert planned[-2].edge.meta.get("segment") == "to_level2_door"
    assert "triforce_shard_1" in planned[-1].capabilities_before


def test_named_routes_align_with_status_milestones() -> None:
    """L1 complete + L2 prefix verified; door path present as planned scaffold."""
    ids = {r.route_id for r in list_routes()}
    assert "zelda_level1_complete" in ids
    assert "zelda_level2_path_prefix" in ids
    assert "zelda_level2_door_path" in ids

    complete = get_route("level1_complete")
    assert complete.milestones[-1].node_id == NODE_LEVEL1_COMPLETE
    assert complete.milestones[-1].stop_predicate == "triforce & 0x01"

    prefix = get_route("level2_prefix")
    assert prefix.milestones[-1].node_id == NODE_LEVEL2_PATH_4A
    assert prefix.milestones[-1].stop_predicate == "level2_path_prefix_success"

    door = get_route("to_level2")
    assert door.milestones[-2].node_id == NODE_LEVEL2_ENTRANCE
    assert door.milestones[-1].node_id == NODE_LEVEL2_DUNGEON
    # Intermediate planned stop predicates (future controllers).
    mid_ids = {m.milestone_id for m in door.milestones}
    assert "level2_door_5a" in mid_ids
    assert "level2_door_5c" in mid_ids
    assert "level2_entrance" in mid_ids

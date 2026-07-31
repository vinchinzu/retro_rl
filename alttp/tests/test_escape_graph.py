"""Offline unit tests for the ALTTP castle-escape capability graph."""

from __future__ import annotations

import pytest

from adventure_common.graph import RouteLeg
from alttp.escape_graph import (
    CAP_FIGHTER_SWORD,
    CAP_LAMP,
    CAP_SMALL_KEY,
    CAP_ZELDA_FOLLOWER,
    N_CASTLE_GROUNDS,
    N_ROOM_55_KEYED,
    N_ROOM_55_SOUTH,
    N_ROOM_55_SWORD,
    N_ROOM_55_UNCLE,
    N_ROOM_61,
    N_ROOM_80,
    N_SANCTUARY,
    N_SEWERS_DARK,
    NATURAL_HOUSE_EXIT_CAPABILITIES,
    capabilities_from_snapshot,
    escape_route_graph,
    escape_route_legs,
    escape_route_legs_from_room_55,
    plan_escape_to_sanctuary,
)
from alttp.ram import (
    HYRULE_CASTLE_MAIN_HALL_ROOM,
    SANCTUARY_ROOM,
    SECRET_PASSAGE_ROOM,
    ZELDA_CELL_ROOM,
    AlttpSnapshot,
)


def _snap(**kwargs: object) -> AlttpSnapshot:
    base: dict[str, object] = dict(
        game_mode=0x07,
        submodule=0x00,
        room_id=SECRET_PASSAGE_ROOM,
        indoors=True,
        screen_id=0,
        link_x=0,
        link_y=0,
        link_direction=0,
        link_action=0,
        camera_x=0,
        camera_y=0,
        dark_world=False,
        sword_level=0,
        lamp_level=0,
        num_keys=0xFF,
        follower=0,
    )
    base.update(kwargs)
    return AlttpSnapshot(**base)  # type: ignore[arg-type]


def test_escape_graph_builds_and_nodes_cover_edges() -> None:
    graph = escape_route_graph()
    assert len(graph.nodes) >= 10
    assert len(graph.edges) >= 8
    # RouteGraph constructor already rejects missing endpoints; re-check pairs.
    for edge in graph.edges:
        assert edge.source_id in graph.nodes
        assert edge.target_id in graph.nodes

    # RAM room constants present on indoor nodes.
    assert graph.nodes[N_ROOM_55_UNCLE].meta["room_base_id"] == SECRET_PASSAGE_ROOM
    assert graph.nodes[N_ROOM_55_SOUTH].meta["room_base_id"] == SECRET_PASSAGE_ROOM
    assert graph.nodes[N_ROOM_61].meta["room_base_id"] == HYRULE_CASTLE_MAIN_HALL_ROOM
    assert graph.nodes[N_ROOM_80].meta["room_base_id"] == ZELDA_CELL_ROOM
    assert graph.nodes[N_SANCTUARY].meta["room_base_id"] == SANCTUARY_ROOM


def test_verified_edges_are_continuous() -> None:
    graph = escape_route_graph()
    continuous_pairs = {
        (N_CASTLE_GROUNDS, N_ROOM_55_UNCLE),
        (N_ROOM_55_UNCLE, N_ROOM_55_SWORD),
        (N_ROOM_55_SWORD, N_ROOM_55_SOUTH),
    }
    for edge in graph.edges:
        pair = (edge.source_id, edge.target_id)
        if pair in continuous_pairs:
            assert edge.verification == "continuous", edge.edge_id
        else:
            assert edge.verification == "planned", edge.edge_id


def test_multi_screen_55_connected() -> None:
    graph = escape_route_graph()
    edge = graph.edge_for(N_ROOM_55_SWORD, N_ROOM_55_SOUTH)
    assert edge is not None
    assert CAP_FIGHTER_SWORD in edge.requires
    assert edge.verification == "continuous"


def test_full_plan_with_natural_lamp_reaches_sanctuary() -> None:
    planned = plan_escape_to_sanctuary()
    assert planned[0].leg.source_id == N_CASTLE_GROUNDS
    assert planned[-1].leg.target_id == N_SANCTUARY
    assert planned[-1].capabilities_after >= frozenset(
        {CAP_LAMP, CAP_FIGHTER_SWORD, CAP_SMALL_KEY, CAP_ZELDA_FOLLOWER}
    )
    # Sword acquired on uncle leg before south chamber.
    sword_leg = next(p for p in planned if p.leg.leg_id == "uncle_fighter_sword")
    assert CAP_FIGHTER_SWORD in sword_leg.capabilities_after
    assert CAP_FIGHTER_SWORD not in sword_leg.capabilities_before
    path = [p.leg.source_id for p in planned] + [planned[-1].leg.target_id]
    assert path[0] == N_CASTLE_GROUNDS
    assert path[-1] == N_SANCTUARY
    assert N_SEWERS_DARK in path


def test_plan_from_room_55_with_sword_and_lamp() -> None:
    """Resume at 0x55 with sword+lamp; keys/Zelda still come from leg.acquires."""
    planned = plan_escape_to_sanctuary(
        frozenset({CAP_FIGHTER_SWORD, CAP_LAMP}),
        legs=escape_route_legs_from_room_55(),
    )
    assert planned[0].leg.source_id == N_ROOM_55_UNCLE
    assert planned[-1].leg.target_id == N_SANCTUARY
    # Key and follower are not in the initial set — acquired on the route.
    assert CAP_SMALL_KEY not in planned[0].capabilities_before
    assert CAP_ZELDA_FOLLOWER not in planned[0].capabilities_before
    assert CAP_SMALL_KEY in planned[-1].capabilities_after
    assert CAP_ZELDA_FOLLOWER in planned[-1].capabilities_after


def test_plan_without_sword_cannot_leave_combat_section() -> None:
    """South chamber → keyed exit requires fighter_sword."""
    graph = escape_route_graph()
    leave_combat = (
        RouteLeg(
            leg_id="leave_south",
            source_id=N_ROOM_55_SOUTH,
            target_id=N_ROOM_55_KEYED,
            requires=frozenset({CAP_FIGHTER_SWORD}),
        ),
    )
    with pytest.raises(ValueError, match="missing capabilities"):
        graph.plan_legs(
            leave_combat,
            initial_capabilities=frozenset({CAP_LAMP}),
        )
    # Same edge pathfinding is blocked without sword.
    assert (
        graph.shortest_path(
            N_ROOM_55_SOUTH,
            N_ROOM_61,
            capabilities=frozenset({CAP_LAMP, CAP_SMALL_KEY}),
        )
        is None
    )
    path = graph.shortest_path(
        N_ROOM_55_SOUTH,
        N_ROOM_61,
        capabilities=frozenset({CAP_FIGHTER_SWORD, CAP_SMALL_KEY}),
    )
    assert path is not None
    assert len(path) == 2


def test_plan_without_lamp_fails_at_sewers() -> None:
    """Mantle / dark sewers require lamp even with sword, key, and Zelda."""
    graph = escape_route_graph()
    legs = escape_route_legs()
    with pytest.raises(ValueError, match="lamp"):
        graph.plan_legs(legs, initial_capabilities=frozenset())


def test_escape_route_legs_are_contiguous() -> None:
    legs = escape_route_legs()
    assert legs[0].source_id == N_CASTLE_GROUNDS
    for prev, nxt in zip(legs, legs[1:]):
        assert prev.target_id == nxt.source_id
    assert legs[-1].target_id == N_SANCTUARY
    # Every leg has a matching edge.
    graph = escape_route_graph()
    for leg in legs:
        assert graph.edge_for(leg.source_id, leg.target_id) is not None


def test_capabilities_from_snapshot() -> None:
    empty = capabilities_from_snapshot(_snap())
    assert empty == frozenset()

    full = capabilities_from_snapshot(
        _snap(
            sword_level=1,
            lamp_level=1,
            num_keys=2,
            follower=1,
        )
    )
    assert full == frozenset(
        {CAP_FIGHTER_SWORD, CAP_LAMP, CAP_SMALL_KEY, CAP_ZELDA_FOLLOWER}
    )

    blank_keys = capabilities_from_snapshot(
        _snap(sword_level=1, lamp_level=1, num_keys=0xFF)
    )
    assert CAP_SMALL_KEY not in blank_keys
    assert CAP_FIGHTER_SWORD in blank_keys
    assert CAP_LAMP in blank_keys


def test_natural_default_capabilities_include_lamp() -> None:
    assert CAP_LAMP in NATURAL_HOUSE_EXIT_CAPABILITIES
    assert CAP_FIGHTER_SWORD not in NATURAL_HOUSE_EXIT_CAPABILITIES

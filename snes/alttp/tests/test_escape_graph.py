"""Offline unit tests for the ALTTP castle-escape capability graph."""

from __future__ import annotations

import pytest

from retro_harness.adventure.graph import RouteLeg
from alttp.escape_graph import (
    CAP_FIGHTER_SWORD,
    CAP_LAMP,
    CAP_SMALL_KEY,
    CAP_ZELDA_FOLLOWER,
    N_CASTLE_GROUNDS,
    N_COURTYARD_SECRET_POCKET,
    N_ROOM_01,
    N_ROOM_55_KEYED,
    N_ROOM_55_SOUTH,
    N_ROOM_55_SWORD,
    N_ROOM_55_UNCLE,
    N_ROOM_50,
    N_ROOM_60,
    N_ROOM_61,
    N_ROOM_80,
    N_SANCTUARY,
    N_SEWERS_DARK,
    NATURAL_HOUSE_EXIT_CAPABILITIES,
    VERIFICATION_CONTINUOUS,
    VERIFICATION_NATURAL_ENTRY,
    VERIFICATION_PLANNED,
    capabilities_from_snapshot,
    continuous_spine_legs,
    escape_route_graph,
    escape_route_legs,
    escape_route_legs_from_room_55,
    escape_route_legs_key_path,
    plan_escape_to_sanctuary,
)
from alttp.ram import (
    HYRULE_CASTLE_MAIN_HALL_ROOM,
    HYRULE_CASTLE_MAIN_WEST_ROOM,
    HYRULE_CASTLE_NORTH_CONNECTOR_ROOM,
    HYRULE_CASTLE_NW_ROOM,
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
    assert len(graph.nodes) >= 13
    assert len(graph.edges) >= 12
    # RouteGraph constructor already rejects missing endpoints; re-check pairs.
    for edge in graph.edges:
        assert edge.source_id in graph.nodes
        assert edge.target_id in graph.nodes

    # RAM room constants present on indoor nodes.
    assert graph.nodes[N_ROOM_55_UNCLE].meta["room_base_id"] == SECRET_PASSAGE_ROOM
    assert graph.nodes[N_ROOM_55_SOUTH].meta["room_base_id"] == SECRET_PASSAGE_ROOM
    assert graph.nodes[N_ROOM_61].meta["room_base_id"] == HYRULE_CASTLE_MAIN_HALL_ROOM
    assert graph.nodes[N_ROOM_60].meta["room_base_id"] == HYRULE_CASTLE_MAIN_WEST_ROOM
    assert graph.nodes[N_ROOM_50].meta["room_base_id"] == HYRULE_CASTLE_NW_ROOM
    assert graph.nodes[N_ROOM_01].meta["room_base_id"] == HYRULE_CASTLE_NORTH_CONNECTOR_ROOM
    assert graph.nodes[N_ROOM_80].meta["room_base_id"] == ZELDA_CELL_ROOM
    assert graph.nodes[N_SANCTUARY].meta["room_base_id"] == SANCTUARY_ROOM
    assert graph.nodes[N_COURTYARD_SECRET_POCKET].meta["screen_id"] == 0x1B


def test_verified_edges_are_continuous() -> None:
    graph = escape_route_graph()
    continuous_pairs = {
        (N_CASTLE_GROUNDS, N_ROOM_55_UNCLE),
        (N_ROOM_55_UNCLE, N_ROOM_55_SWORD),
        (N_ROOM_55_SWORD, N_ROOM_55_SOUTH),
        (N_ROOM_55_SOUTH, N_COURTYARD_SECRET_POCKET),
        (N_COURTYARD_SECRET_POCKET, N_ROOM_61),
    }
    continuous_pairs.update(
        {
            (N_ROOM_61, N_ROOM_60),
            (N_ROOM_60, N_ROOM_50),
        }
    )
    natural_entry_pairs = {
        (N_ROOM_50, N_ROOM_01),
    }
    for edge in graph.edges:
        pair = (edge.source_id, edge.target_id)
        if pair in continuous_pairs:
            assert edge.verification == VERIFICATION_CONTINUOUS, edge.edge_id
        elif pair in natural_entry_pairs:
            assert edge.verification == VERIFICATION_NATURAL_ENTRY, edge.edge_id
        else:
            assert edge.verification == VERIFICATION_PLANNED, edge.edge_id
    assert graph.edge_for(N_ROOM_61, N_ROOM_60).direction == "west"  # type: ignore[union-attr]
    assert graph.edge_for(N_ROOM_60, N_ROOM_50).direction == "north"  # type: ignore[union-attr]
    assert graph.edge_for(N_ROOM_50, N_ROOM_01).direction == "east"  # type: ignore[union-attr]
    assert graph.edge_for(N_ROOM_50, N_ROOM_01).meta["door_label"] == "east_to_0x01"  # type: ignore[union-attr]


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
    # Outdoor primary path: lamp + sword + Zelda; small_key is alternate only.
    assert planned[-1].capabilities_after >= frozenset(
        {CAP_LAMP, CAP_FIGHTER_SWORD, CAP_ZELDA_FOLLOWER}
    )
    assert CAP_SMALL_KEY not in planned[-1].capabilities_after
    # Sword acquired on uncle leg before south chamber.
    sword_leg = next(p for p in planned if p.leg.leg_id == "uncle_fighter_sword")
    assert CAP_FIGHTER_SWORD in sword_leg.capabilities_after
    assert CAP_FIGHTER_SWORD not in sword_leg.capabilities_before
    path = [p.leg.source_id for p in planned] + [planned[-1].leg.target_id]
    assert path[0] == N_CASTLE_GROUNDS
    assert path[-1] == N_SANCTUARY
    assert N_COURTYARD_SECRET_POCKET in path
    assert N_SEWERS_DARK in path


def test_continuous_spine_ends_at_nw_chamber() -> None:
    legs = continuous_spine_legs()
    assert legs[0].source_id == N_CASTLE_GROUNDS
    assert legs[-1].target_id == N_ROOM_50
    assert any(leg.target_id == N_COURTYARD_SECRET_POCKET for leg in legs)
    graph = escape_route_graph()
    for leg in legs:
        edge = graph.edge_for(leg.source_id, leg.target_id)
        assert edge is not None
        assert edge.verification == VERIFICATION_CONTINUOUS
    pocket_edge = graph.edge_for(N_COURTYARD_SECRET_POCKET, N_ROOM_61)
    assert pocket_edge is not None
    assert pocket_edge.edge_id == "pocket_to_main_hall"
    assert any(leg.target_id == N_ROOM_60 for leg in legs)
    assert any(leg.target_id == N_ROOM_50 for leg in legs)


def test_key_path_plan_still_acquires_small_key() -> None:
    planned = plan_escape_to_sanctuary(legs=escape_route_legs_key_path())
    assert CAP_SMALL_KEY in planned[-1].capabilities_after
    assert N_ROOM_55_KEYED in (
        [p.leg.source_id for p in planned] + [planned[-1].leg.target_id]
    )


def test_key_path_and_primary_share_post_main_hall_tail() -> None:
    """Both Sanctuary plans derive the same shared hop-table tail after room_61."""
    primary = escape_route_legs()
    key = escape_route_legs_key_path()

    def tail_from_main(legs: tuple[RouteLeg, ...]) -> tuple[RouteLeg, ...]:
        idx = next(i for i, leg in enumerate(legs) if leg.source_id == N_ROOM_61)
        return legs[idx:]

    assert tail_from_main(primary) == tail_from_main(key)
    assert CAP_SMALL_KEY not in {cap for leg in primary for cap in leg.acquires}
    assert any(CAP_SMALL_KEY in leg.acquires for leg in key)
    assert not any(leg.target_id == N_COURTYARD_SECRET_POCKET for leg in key)
    assert any(leg.target_id == N_COURTYARD_SECRET_POCKET for leg in primary)
    # West + north exits on shared tail; tip east hop then planned Zelda.
    assert any(leg.target_id == N_ROOM_60 for leg in primary)
    assert any(leg.target_id == N_ROOM_50 for leg in primary)
    assert any(
        leg.source_id == N_ROOM_50 and leg.target_id == N_ROOM_01 for leg in primary
    )
    assert any(
        leg.source_id == N_ROOM_01 and leg.target_id == N_ROOM_80 for leg in primary
    )


def test_plan_from_room_55_with_sword_and_lamp() -> None:
    """Resume at 0x55 with sword+lamp; Zelda acquired on route (outdoor path)."""
    planned = plan_escape_to_sanctuary(
        frozenset({CAP_FIGHTER_SWORD, CAP_LAMP}),
        legs=escape_route_legs_from_room_55(),
    )
    assert planned[0].leg.source_id == N_ROOM_55_UNCLE
    assert planned[-1].leg.target_id == N_SANCTUARY
    assert CAP_ZELDA_FOLLOWER not in planned[0].capabilities_before
    assert CAP_ZELDA_FOLLOWER in planned[-1].capabilities_after
    assert CAP_SMALL_KEY not in planned[-1].capabilities_after


def test_plan_without_sword_cannot_leave_combat_section() -> None:
    """South chamber exits require fighter_sword (pocket or keyed)."""
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
    # Pathfinding blocked without sword (both pocket and key routes need it).
    assert (
        graph.shortest_path(
            N_ROOM_55_SOUTH,
            N_ROOM_61,
            capabilities=frozenset({CAP_LAMP, CAP_SMALL_KEY}),
        )
        is None
    )
    # With sword only: outdoor pocket path (2 hops: south→pocket→main).
    path = graph.shortest_path(
        N_ROOM_55_SOUTH,
        N_ROOM_61,
        capabilities=frozenset({CAP_FIGHTER_SWORD}),
    )
    assert path is not None
    nodes_on_path = {path[0].source_id} | {e.target_id for e in path}
    assert N_COURTYARD_SECRET_POCKET in nodes_on_path
    # Key path still works with sword + key.
    key_path = graph.shortest_path(
        N_ROOM_55_SOUTH,
        N_ROOM_61,
        capabilities=frozenset({CAP_FIGHTER_SWORD, CAP_SMALL_KEY}),
    )
    assert key_path is not None


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

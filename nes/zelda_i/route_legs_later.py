"""RouteLeg tables and plan helpers for Zelda I L3–L5 + L9 fixture.

Coarse NamedRoute hops (not the dungeon door-graph). L6–L8 stay stubs in
``routes_later``. Verification is observed / assisted — not Clean.
"""

from __future__ import annotations

from retro_harness.adventure.graph import GraphEdge, GraphNode, RouteGraph, RouteLeg

from zelda_i.later_nodes import (
    NODE_LEVEL3_BOSS,
    NODE_LEVEL3_COMPLETE,
    NODE_LEVEL3_ENTRANCE,
    NODE_LEVEL3_ENTRY_ROOM,
    NODE_LEVEL3_RAFT,
    NODE_LEVEL3_WEST_KEY,
    NODE_LEVEL4_BOSS,
    NODE_LEVEL4_COMPLETE,
    NODE_LEVEL4_ENTRANCE,
    NODE_LEVEL4_ENTRY_ROOM,
    NODE_LEVEL4_STEPLADDER,
    NODE_LEVEL5_BOSS,
    NODE_LEVEL5_COMPLETE,
    NODE_LEVEL5_EAST_77,
    NODE_LEVEL5_ENTRANCE,
    NODE_LEVEL5_ENTRY_ROOM,
    NODE_LEVEL5_KEY_66,
    NODE_LEVEL5_WHISTLE,
    NODE_LEVEL9_CELLAR_67,
    NODE_LEVEL9_GANON,
    NODE_LEVEL9_PATRA,
    NODE_LEVEL9_ROOM_03,
    NODE_LEVEL9_ROOM_04,
    NODE_LEVEL9_ROOM_30,
    NODE_LEVEL9_ROOM_31,
    NODE_LEVEL9_ROOM_41,
    NODE_LEVEL9_ZELDA,
    NODE_LOST_HILLS,
    NODE_RAFT_L4_DOCK,
)


def _node(node_id: str, name: str, area: str, **meta: object) -> GraphNode:
    return GraphNode(node_id=node_id, name=name, area=area, meta=meta)


def _edge(
    source_id: str,
    target_id: str,
    *,
    requires: frozenset[str] = frozenset(),
    acquires: frozenset[str] = frozenset(),
    verification: str = "observed",
    **meta: object,
) -> GraphEdge:
    return GraphEdge(
        source_id=source_id,
        target_id=target_id,
        requires=requires,
        acquires=acquires,
        verification=verification,
        provenance="LEVELN_ROUTE",
        meta=meta,
    )


def level3_complete_route_legs() -> tuple[RouteLeg, ...]:
    """OW 0x74 → entry 0x7c → west key → Raft → Manhandla → TF 0x04."""
    return (
        RouteLeg(
            leg_id="enter_level3",
            source_id=NODE_LEVEL3_ENTRANCE,
            target_id=NODE_LEVEL3_ENTRY_ROOM,
            goal="inside_level3_room_7c",
        ),
        RouteLeg(
            leg_id="level3_west_key",
            source_id=NODE_LEVEL3_ENTRY_ROOM,
            target_id=NODE_LEVEL3_WEST_KEY,
            acquires=frozenset({"keys"}),
            goal="level3_room_7b_key",
        ),
        RouteLeg(
            leg_id="level3_raft",
            source_id=NODE_LEVEL3_WEST_KEY,
            target_id=NODE_LEVEL3_RAFT,
            acquires=frozenset({"raft"}),
            goal="level3_raft_collected",
            constraints=("assisted",),
        ),
        RouteLeg(
            leg_id="level3_manhandla",
            source_id=NODE_LEVEL3_RAFT,
            target_id=NODE_LEVEL3_BOSS,
            requires=frozenset({"raft"}),
            goal="level3_manhandla_room",
            constraints=("assisted",),
        ),
        RouteLeg(
            leg_id="level3_triforce",
            source_id=NODE_LEVEL3_BOSS,
            target_id=NODE_LEVEL3_COMPLETE,
            acquires=frozenset({"triforce_shard_3"}),
            goal="level3_triforce_bit_2_set",
            constraints=("assisted",),
        ),
    )


def level4_complete_route_legs() -> tuple[RouteLeg, ...]:
    """Dock 0x55 → island 0x45 → entry 0x71 → Stepladder → Gleeok → TF 0x08."""
    return (
        RouteLeg(
            leg_id="level4_raft_to_island",
            source_id=NODE_RAFT_L4_DOCK,
            target_id=NODE_LEVEL4_ENTRANCE,
            requires=frozenset({"raft"}),
            goal="reach_screen_45_island",
            constraints=("assisted",),
        ),
        RouteLeg(
            leg_id="enter_level4",
            source_id=NODE_LEVEL4_ENTRANCE,
            target_id=NODE_LEVEL4_ENTRY_ROOM,
            requires=frozenset({"raft"}),
            goal="inside_level4_room_71",
        ),
        RouteLeg(
            leg_id="level4_stepladder",
            source_id=NODE_LEVEL4_ENTRY_ROOM,
            target_id=NODE_LEVEL4_STEPLADDER,
            acquires=frozenset({"stepladder"}),
            goal="level4_stepladder_collected",
        ),
        RouteLeg(
            leg_id="level4_gleeok",
            source_id=NODE_LEVEL4_STEPLADDER,
            target_id=NODE_LEVEL4_BOSS,
            requires=frozenset({"stepladder"}),
            goal="level4_gleeok_room",
        ),
        RouteLeg(
            leg_id="level4_triforce",
            source_id=NODE_LEVEL4_BOSS,
            target_id=NODE_LEVEL4_COMPLETE,
            acquires=frozenset({"triforce_shard_4"}),
            goal="level4_triforce_bit_3_set",
        ),
    )


def level5_complete_route_legs() -> tuple[RouteLeg, ...]:
    """Lost Hills → door → entry → 0x66 key → 0x77 → whistle → Digdogger → TF."""
    return (
        RouteLeg(
            leg_id="level5_hills_to_door",
            source_id=NODE_LOST_HILLS,
            target_id=NODE_LEVEL5_ENTRANCE,
            goal="reach_screen_0b_l5_door",
        ),
        RouteLeg(
            leg_id="enter_level5",
            source_id=NODE_LEVEL5_ENTRANCE,
            target_id=NODE_LEVEL5_ENTRY_ROOM,
            goal="inside_level5_room_76",
        ),
        RouteLeg(
            leg_id="level5_first_key",
            source_id=NODE_LEVEL5_ENTRY_ROOM,
            target_id=NODE_LEVEL5_KEY_66,
            acquires=frozenset({"keys"}),
            goal="level5_room_66_cleared",
        ),
        RouteLeg(
            leg_id="level5_east_key",
            source_id=NODE_LEVEL5_KEY_66,
            target_id=NODE_LEVEL5_EAST_77,
            requires=frozenset({"keys"}),
            goal="level5_room_77_key",
        ),
        RouteLeg(
            leg_id="level5_whistle",
            source_id=NODE_LEVEL5_EAST_77,
            target_id=NODE_LEVEL5_WHISTLE,
            acquires=frozenset({"whistle"}),
            goal="level5_whistle_collected",
            constraints=("assisted",),
        ),
        RouteLeg(
            leg_id="level5_digdogger",
            source_id=NODE_LEVEL5_WHISTLE,
            target_id=NODE_LEVEL5_BOSS,
            requires=frozenset({"whistle"}),
            goal="level5_digdogger_room",
            constraints=("assisted",),
        ),
        RouteLeg(
            leg_id="level5_triforce",
            source_id=NODE_LEVEL5_BOSS,
            target_id=NODE_LEVEL5_COMPLETE,
            requires=frozenset({"whistle"}),
            acquires=frozenset({"triforce_shard_5"}),
            goal="level5_triforce_bit_4_set",
            constraints=("assisted",),
        ),
    )


_L9_FIXTURE = ("fixture_only", "route_eligible=false")


def level9_fixture_route_legs() -> tuple[RouteLeg, ...]:
    """Fixture suffix 0x41 → … → Patra → Ganon → Zelda. Not route-ready."""
    hops = (
        ("level9_41_to_31", NODE_LEVEL9_ROOM_41, NODE_LEVEL9_ROOM_31, "level9_in_room_31"),
        ("level9_31_to_30", NODE_LEVEL9_ROOM_31, NODE_LEVEL9_ROOM_30, "level9_in_room_30"),
        ("level9_30_to_67", NODE_LEVEL9_ROOM_30, NODE_LEVEL9_CELLAR_67, "level9_in_cellar_67"),
        ("level9_67_to_04", NODE_LEVEL9_CELLAR_67, NODE_LEVEL9_ROOM_04, "level9_in_room_04"),
        ("level9_04_to_03", NODE_LEVEL9_ROOM_04, NODE_LEVEL9_ROOM_03, "level9_in_room_03"),
        ("level9_03_to_patra", NODE_LEVEL9_ROOM_03, NODE_LEVEL9_PATRA, "level9_patra_room"),
        ("level9_patra_to_ganon", NODE_LEVEL9_PATRA, NODE_LEVEL9_GANON, "level9_ganon_defeated"),
        ("level9_ganon_to_zelda", NODE_LEVEL9_GANON, NODE_LEVEL9_ZELDA, "level9_ending"),
    )
    return tuple(
        RouteLeg(
            leg_id=leg_id,
            source_id=src,
            target_id=dst,
            goal=goal,
            constraints=_L9_FIXTURE,
        )
        for leg_id, src, dst, goal in hops
    )


def build_later_route_graph() -> RouteGraph:
    """Coarse L3–L5 + L9-fixture graph that can ``plan_legs`` those routes."""
    nodes = (
        _node(NODE_LEVEL3_ENTRANCE, "level3_ow_door", "overworld", screen=0x74),
        _node(NODE_LEVEL3_ENTRY_ROOM, "level3_entry", "level3", room=0x7C),
        _node(NODE_LEVEL3_WEST_KEY, "level3_west_key", "level3", room=0x7B),
        _node(NODE_LEVEL3_RAFT, "level3_raft", "level3", room=0x0F, item="raft"),
        _node(NODE_LEVEL3_BOSS, "level3_manhandla", "level3", room=0x4D),
        _node(NODE_LEVEL3_COMPLETE, "level3_triforce", "level3", room=0x3D),
        _node(NODE_RAFT_L4_DOCK, "level4_dock", "overworld", screen=0x55),
        _node(NODE_LEVEL4_ENTRANCE, "level4_island", "overworld", screen=0x45),
        _node(NODE_LEVEL4_ENTRY_ROOM, "level4_entry", "level4", room=0x71),
        _node(NODE_LEVEL4_STEPLADDER, "level4_stepladder", "level4", room=0x60),
        _node(NODE_LEVEL4_BOSS, "level4_gleeok", "level4", room=0x13),
        _node(NODE_LEVEL4_COMPLETE, "level4_triforce", "level4", room=0x03),
        _node(NODE_LOST_HILLS, "lost_hills", "overworld", screen=0x1B),
        _node(NODE_LEVEL5_ENTRANCE, "level5_ow_door", "overworld", screen=0x0B),
        _node(NODE_LEVEL5_ENTRY_ROOM, "level5_entry", "level5", room=0x76),
        _node(NODE_LEVEL5_KEY_66, "level5_key_66", "level5", room=0x66),
        _node(NODE_LEVEL5_EAST_77, "level5_east_77", "level5", room=0x77),
        _node(NODE_LEVEL5_WHISTLE, "level5_whistle", "level5", room=0x04),
        _node(NODE_LEVEL5_BOSS, "level5_digdogger", "level5", room=0x24),
        _node(NODE_LEVEL5_COMPLETE, "level5_triforce", "level5", room=0x14),
        _node(NODE_LEVEL9_ROOM_41, "level9_41", "level9", room=0x41, fixture_only=True),
        _node(NODE_LEVEL9_ROOM_31, "level9_31", "level9", room=0x31, fixture_only=True),
        _node(NODE_LEVEL9_ROOM_30, "level9_30", "level9", room=0x30, fixture_only=True),
        _node(NODE_LEVEL9_CELLAR_67, "level9_67", "level9", room=0x67, fixture_only=True),
        _node(NODE_LEVEL9_ROOM_04, "level9_04", "level9", room=0x04, fixture_only=True),
        _node(NODE_LEVEL9_ROOM_03, "level9_03", "level9", room=0x03, fixture_only=True),
        _node(NODE_LEVEL9_PATRA, "level9_patra", "level9", room=0x52, fixture_only=True),
        _node(NODE_LEVEL9_GANON, "level9_ganon", "level9", room=0x42, fixture_only=True),
        _node(NODE_LEVEL9_ZELDA, "level9_zelda", "level9", room=0x32, fixture_only=True),
    )
    edges = (
        _edge(NODE_LEVEL3_ENTRANCE, NODE_LEVEL3_ENTRY_ROOM, segment="level3"),
        _edge(
            NODE_LEVEL3_ENTRY_ROOM,
            NODE_LEVEL3_WEST_KEY,
            acquires=frozenset({"keys"}),
            segment="level3",
        ),
        _edge(
            NODE_LEVEL3_WEST_KEY,
            NODE_LEVEL3_RAFT,
            acquires=frozenset({"raft"}),
            verification="assisted",
            segment="level3",
        ),
        _edge(
            NODE_LEVEL3_RAFT,
            NODE_LEVEL3_BOSS,
            requires=frozenset({"raft"}),
            verification="assisted",
            segment="level3",
        ),
        _edge(
            NODE_LEVEL3_BOSS,
            NODE_LEVEL3_COMPLETE,
            acquires=frozenset({"triforce_shard_3"}),
            verification="assisted",
            segment="level3",
        ),
        _edge(
            NODE_RAFT_L4_DOCK,
            NODE_LEVEL4_ENTRANCE,
            requires=frozenset({"raft"}),
            verification="assisted",
            segment="level4",
        ),
        _edge(
            NODE_LEVEL4_ENTRANCE,
            NODE_LEVEL4_ENTRY_ROOM,
            requires=frozenset({"raft"}),
            verification="assisted",
            segment="level4",
        ),
        _edge(
            NODE_LEVEL4_ENTRY_ROOM,
            NODE_LEVEL4_STEPLADDER,
            acquires=frozenset({"stepladder"}),
            segment="level4",
        ),
        _edge(
            NODE_LEVEL4_STEPLADDER,
            NODE_LEVEL4_BOSS,
            requires=frozenset({"stepladder"}),
            segment="level4",
        ),
        _edge(
            NODE_LEVEL4_BOSS,
            NODE_LEVEL4_COMPLETE,
            acquires=frozenset({"triforce_shard_4"}),
            segment="level4",
        ),
        _edge(NODE_LOST_HILLS, NODE_LEVEL5_ENTRANCE, segment="level5"),
        _edge(NODE_LEVEL5_ENTRANCE, NODE_LEVEL5_ENTRY_ROOM, segment="level5"),
        _edge(
            NODE_LEVEL5_ENTRY_ROOM,
            NODE_LEVEL5_KEY_66,
            acquires=frozenset({"keys"}),
            segment="level5",
        ),
        _edge(
            NODE_LEVEL5_KEY_66,
            NODE_LEVEL5_EAST_77,
            requires=frozenset({"keys"}),
            segment="level5",
        ),
        _edge(
            NODE_LEVEL5_EAST_77,
            NODE_LEVEL5_WHISTLE,
            acquires=frozenset({"whistle"}),
            verification="assisted",
            segment="level5",
        ),
        _edge(
            NODE_LEVEL5_WHISTLE,
            NODE_LEVEL5_BOSS,
            requires=frozenset({"whistle"}),
            verification="assisted",
            segment="level5",
        ),
        _edge(
            NODE_LEVEL5_BOSS,
            NODE_LEVEL5_COMPLETE,
            requires=frozenset({"whistle"}),
            acquires=frozenset({"triforce_shard_5"}),
            verification="assisted",
            segment="level5",
        ),
        _edge(
            NODE_LEVEL9_ROOM_41,
            NODE_LEVEL9_ROOM_31,
            verification="observed",
            fixture_only=True,
        ),
        _edge(
            NODE_LEVEL9_ROOM_31,
            NODE_LEVEL9_ROOM_30,
            verification="observed",
            fixture_only=True,
        ),
        _edge(
            NODE_LEVEL9_ROOM_30,
            NODE_LEVEL9_CELLAR_67,
            verification="observed",
            fixture_only=True,
        ),
        _edge(
            NODE_LEVEL9_CELLAR_67,
            NODE_LEVEL9_ROOM_04,
            verification="observed",
            fixture_only=True,
        ),
        _edge(
            NODE_LEVEL9_ROOM_04,
            NODE_LEVEL9_ROOM_03,
            verification="observed",
            fixture_only=True,
        ),
        _edge(
            NODE_LEVEL9_ROOM_03,
            NODE_LEVEL9_PATRA,
            verification="observed",
            fixture_only=True,
        ),
        _edge(
            NODE_LEVEL9_PATRA,
            NODE_LEVEL9_GANON,
            verification="observed",
            fixture_only=True,
        ),
        _edge(
            NODE_LEVEL9_GANON,
            NODE_LEVEL9_ZELDA,
            verification="observed",
            fixture_only=True,
        ),
    )
    return RouteGraph(nodes, edges)


def level3_complete_route_plan(graph: RouteGraph | None = None):
    g = graph or build_later_route_graph()
    return g.plan_legs(level3_complete_route_legs(), initial_capabilities=frozenset())


def level4_complete_route_plan(graph: RouteGraph | None = None):
    g = graph or build_later_route_graph()
    return g.plan_legs(
        level4_complete_route_legs(),
        initial_capabilities=frozenset({"raft"}),
    )


def level5_complete_route_plan(graph: RouteGraph | None = None):
    g = graph or build_later_route_graph()
    return g.plan_legs(level5_complete_route_legs(), initial_capabilities=frozenset())


def level9_fixture_route_plan(graph: RouteGraph | None = None):
    g = graph or build_later_route_graph()
    return g.plan_legs(level9_fixture_route_legs(), initial_capabilities=frozenset())


LEVEL3_COMPLETE_LEGS = level3_complete_route_legs()
LEVEL4_COMPLETE_LEGS = level4_complete_route_legs()
LEVEL5_COMPLETE_LEGS = level5_complete_route_legs()
LEVEL9_FIXTURE_LEGS = level9_fixture_route_legs()

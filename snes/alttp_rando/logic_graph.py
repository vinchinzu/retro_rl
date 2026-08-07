"""Early ALTTP item-logic graph (single-game rando scaffold).

Vanilla room/OW physics are seed-invariant; item placement varies.
Practice ground for L4 before SMZ3. Skills live in ``alttp``.
"""

from __future__ import annotations

from retro_harness.adventure.graph import (
    GraphEdge,
    GraphNode,
    RouteGraph,
    shortest_path,
)

N_LINKS_HOUSE = "z3_links_house"
N_UNCLE = "z3_uncle_sword"
N_CASTLE_YARD = "z3_castle_yard"
N_SEWERS = "z3_sewers"
N_SANCTUARY = "z3_sanctuary"
N_EASTERN = "z3_eastern_palace"
N_EASTERN_BOW = "z3_eastern_bow"
N_HYRULE_CASTLE = "z3_hyrule_castle_escape"


def build_early_graph() -> RouteGraph:
    """Coarse opening + Eastern Palace tip (planned edges)."""
    nodes = (
        GraphNode(N_LINKS_HOUSE, name="Link's House", area="light_world"),
        GraphNode(N_UNCLE, name="Uncle / Fighter Sword", area="hyrule_castle", tags=frozenset({"item"})),
        GraphNode(N_CASTLE_YARD, name="Castle Courtyard", area="hyrule_castle"),
        GraphNode(N_HYRULE_CASTLE, name="Hyrule Castle", area="hyrule_castle"),
        GraphNode(N_SEWERS, name="Sewers", area="hyrule_castle"),
        GraphNode(N_SANCTUARY, name="Sanctuary", area="light_world", tags=frozenset({"milestone"})),
        GraphNode(N_EASTERN, name="Eastern Palace", area="eastern"),
        GraphNode(
            N_EASTERN_BOW,
            name="Eastern Big Chest (Bow tip)",
            area="eastern",
            tags=frozenset({"item", "tip"}),
        ),
    )
    edges = (
        GraphEdge(
            N_LINKS_HOUSE,
            N_UNCLE,
            edge_id="house_to_uncle",
            verification="planned",
            provenance="opening route skill in alttp",
        ),
        GraphEdge(
            N_UNCLE,
            N_CASTLE_YARD,
            edge_id="uncle_to_yard",
            requires=frozenset({"sword"}),
            verification="planned",
        ),
        GraphEdge(
            N_CASTLE_YARD,
            N_HYRULE_CASTLE,
            edge_id="yard_to_castle",
            requires=frozenset({"sword"}),
            verification="planned",
        ),
        GraphEdge(
            N_HYRULE_CASTLE,
            N_SEWERS,
            edge_id="castle_to_sewers",
            requires=frozenset({"sword"}),
            verification="planned",
        ),
        GraphEdge(
            N_SEWERS,
            N_SANCTUARY,
            edge_id="sewers_to_sanctuary",
            requires=frozenset({"sword", "lamp"}),
            verification="planned",
            provenance="lamp often required in dark; scaffold gate",
        ),
        GraphEdge(
            N_SANCTUARY,
            N_EASTERN,
            edge_id="sanctuary_to_eastern",
            requires=frozenset({"sword"}),
            cost=5.0,
            verification="planned",
        ),
        GraphEdge(
            N_EASTERN,
            N_EASTERN_BOW,
            edge_id="eastern_to_bow",
            requires=frozenset({"sword"}),
            cost=10.0,
            verification="planned",
            provenance="aggregate dungeon tip; expand with room skills",
        ),
    )
    return RouteGraph(nodes, edges)


EARLY_GRAPH = build_early_graph()


def path_with_capabilities(
    source_id: str,
    target_id: str,
    capabilities: frozenset[str],
) -> tuple[GraphEdge, ...] | None:
    return shortest_path(EARLY_GRAPH.edges, source_id, target_id, capabilities)


def plan_to_eastern_bow(
    capabilities: frozenset[str] | None = None,
) -> tuple[GraphEdge, ...] | None:
    caps = capabilities if capabilities is not None else frozenset({"sword", "lamp"})
    return path_with_capabilities(N_LINKS_HOUSE, N_EASTERN_BOW, caps)

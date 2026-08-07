"""Early Super Metroid item-logic graph (single-game rando scaffold).

Vanilla room physics are seed-invariant; only placement order changes.
This graph is the practice ground for the shared L4 solver before SMZ3.
Skills for each hop live in ``super_metroid`` — this module owns connectivity
and capability gates only.
"""

from __future__ import annotations

from retro_harness.adventure.graph import (
    GraphEdge,
    GraphNode,
    RouteGraph,
    shortest_path,
)

# --- Node ids ---------------------------------------------------------------

N_SHIP = "sm_landing_ship"
N_MORPH = "sm_morph_ball"
N_CLIMB = "sm_climb"
N_PARLOR = "sm_parlor"
N_BOMBS = "sm_bombs"
N_EARLY_MISSILES = "sm_early_missiles"
N_BRINSTAR = "sm_brinstar_elevator"
N_VARIA = "sm_varia"  # long-range tip placeholder

# --- Capability tokens (normalized by adventure.graph) ----------------------
# morph_ball, bombs, missiles, …


def build_early_graph() -> RouteGraph:
    """Coarse early-game graph for fixture seeds (not full VARIA logic)."""
    nodes = (
        GraphNode(N_SHIP, name="Landing Site / Ship", area="crateria"),
        GraphNode(N_MORPH, name="Morph Ball", area="crateria", tags=frozenset({"item"})),
        GraphNode(N_CLIMB, name="Climb", area="crateria"),
        GraphNode(N_PARLOR, name="Parlor", area="crateria"),
        GraphNode(N_BOMBS, name="Bombs", area="crateria", tags=frozenset({"item"})),
        GraphNode(
            N_EARLY_MISSILES,
            name="Early Missiles",
            area="crateria",
            tags=frozenset({"item"}),
        ),
        GraphNode(N_BRINSTAR, name="Brinstar Elevator", area="brinstar"),
        GraphNode(N_VARIA, name="Varia Suit (tip)", area="norfair", tags=frozenset({"item", "tip"})),
    )
    edges = (
        GraphEdge(N_SHIP, N_MORPH, edge_id="ship_to_morph", cost=1.0, verification="planned"),
        GraphEdge(
            N_MORPH,
            N_CLIMB,
            edge_id="morph_to_climb",
            requires=frozenset({"morph_ball"}),
            verification="planned",
        ),
        GraphEdge(N_CLIMB, N_PARLOR, edge_id="climb_to_parlor", verification="planned"),
        GraphEdge(
            N_PARLOR,
            N_BOMBS,
            edge_id="parlor_to_bombs",
            requires=frozenset({"morph_ball"}),
            verification="planned",
        ),
        GraphEdge(
            N_PARLOR,
            N_EARLY_MISSILES,
            edge_id="parlor_to_missiles",
            requires=frozenset({"morph_ball"}),
            verification="planned",
        ),
        GraphEdge(
            N_PARLOR,
            N_BRINSTAR,
            edge_id="parlor_to_brinstar",
            requires=frozenset({"morph_ball", "missiles"}),
            verification="planned",
            provenance="red door style gate (scaffold)",
        ),
        # Placeholder long edge — real path is many pure hops in super_metroid.
        GraphEdge(
            N_BRINSTAR,
            N_VARIA,
            edge_id="brinstar_to_varia_tip",
            requires=frozenset({"morph_ball", "missiles", "bombs"}),
            cost=50.0,
            verification="planned",
            provenance="aggregate tip; expand with pure skills",
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


def plan_to_varia(capabilities: frozenset[str] | None = None) -> tuple[GraphEdge, ...] | None:
    caps = capabilities if capabilities is not None else frozenset(
        {"morph_ball", "missiles", "bombs"}
    )
    return path_with_capabilities(N_SHIP, N_VARIA, caps)

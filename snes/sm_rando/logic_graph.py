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
from retro_harness.adventure.progression import (
    CapabilityId,
    Has,
    ItemCheck,
    SeedPlacement,
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

# Namespaced offline fixture used by the progression-core tests.  The legacy
# graph above intentionally remains string-compatible for existing callers.
SM_BOMBS = CapabilityId("sm", "bombs")
SM_MISSILES = CapabilityId("sm", "missiles")
N_PROGRESSION_START = "sm_fixture_start"
N_PROGRESSION_BOMBS = "sm_fixture_bombs_check"
N_PROGRESSION_MISSILES = "sm_fixture_missiles_check"
N_PROGRESSION_GOAL = "sm_fixture_goal"
SM_BOMBS_CHECK = "sm_fixture_bombs"
SM_MISSILES_CHECK = "sm_fixture_missiles"

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


def build_progression_graph() -> RouteGraph:
    """Build the small namespaced placement fixture, without emulator state."""
    nodes = (
        GraphNode(N_PROGRESSION_START, name="SM fixture start", area="crateria"),
        GraphNode(N_PROGRESSION_BOMBS, name="SM bombs check", area="crateria"),
        GraphNode(N_PROGRESSION_MISSILES, name="SM missiles check", area="crateria"),
        GraphNode(N_PROGRESSION_GOAL, name="SM fixture goal", area="brinstar"),
    )
    edges = (
        GraphEdge(
            N_PROGRESSION_START,
            N_PROGRESSION_BOMBS,
            edge_id="sm_fixture_to_bombs_check",
        ),
        GraphEdge(
            N_PROGRESSION_BOMBS,
            N_PROGRESSION_GOAL,
            edge_id="sm_fixture_bombs_to_goal",
            requires=Has(SM_BOMBS),
        ),
        GraphEdge(
            N_PROGRESSION_START,
            N_PROGRESSION_MISSILES,
            edge_id="sm_fixture_to_missiles_check",
        ),
        GraphEdge(
            N_PROGRESSION_MISSILES,
            N_PROGRESSION_GOAL,
            edge_id="sm_fixture_missiles_to_goal",
            requires=Has(SM_BOMBS),
        ),
    )
    checks = (
        ItemCheck(SM_BOMBS_CHECK, N_PROGRESSION_BOMBS),
        ItemCheck(SM_MISSILES_CHECK, N_PROGRESSION_MISSILES),
    )
    return RouteGraph(nodes, edges, checks)


PROGRESSION_GRAPH = build_progression_graph()
PLACEMENT_OVERLAY_A = SeedPlacement(
    {
        SM_BOMBS_CHECK: SM_BOMBS,
        SM_MISSILES_CHECK: SM_MISSILES,
    },
    seed_id="sm-fixture-a",
)
PLACEMENT_OVERLAY_B = SeedPlacement(
    {
        SM_BOMBS_CHECK: SM_MISSILES,
        SM_MISSILES_CHECK: SM_BOMBS,
    },
    seed_id="sm-fixture-b",
)
SEED_OVERLAY_A = PLACEMENT_OVERLAY_A
SEED_OVERLAY_B = PLACEMENT_OVERLAY_B


def plan_with_placement(
    placement: SeedPlacement = PLACEMENT_OVERLAY_A,
) -> tuple[GraphEdge, ...] | None:
    """Return the valid offline plan for one SM fixture placement overlay."""
    return PROGRESSION_GRAPH.progression_path(
        N_PROGRESSION_START,
        N_PROGRESSION_GOAL,
        placement,
    )


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

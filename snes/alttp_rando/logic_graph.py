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
from retro_harness.adventure.progression import (
    CapabilityId,
    Has,
    ItemCheck,
    SeedPlacement,
)

N_LINKS_HOUSE = "z3_links_house"
N_UNCLE = "z3_uncle_sword"
N_CASTLE_YARD = "z3_castle_yard"
N_SEWERS = "z3_sewers"
N_SANCTUARY = "z3_sanctuary"
N_EASTERN = "z3_eastern_palace"
N_EASTERN_BOW = "z3_eastern_bow"
N_HYRULE_CASTLE = "z3_hyrule_castle_escape"

# Namespaced offline fixture used by the progression-core tests.  The legacy
# graph above remains available to callers using frozenset[str] inventories.
Z3_SWORD = CapabilityId("z3", "sword")
Z3_LAMP = CapabilityId("z3", "lamp")
N_PROGRESSION_START = "z3_fixture_start"
N_PROGRESSION_SWORD = "z3_fixture_sword_check"
N_PROGRESSION_LAMP = "z3_fixture_lamp_check"
N_PROGRESSION_GOAL = "z3_fixture_goal"
Z3_SWORD_CHECK = "z3_fixture_sword"
Z3_LAMP_CHECK = "z3_fixture_lamp"


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
            verification="natural_entry",
            provenance=(
                "alttp opening skills from FirstPlay (wake/lamp/exit + "
                "OW walk + castle_to_sword); binding in solver_bindings"
            ),
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


def build_progression_graph() -> RouteGraph:
    """Build the small namespaced placement fixture, without emulator state."""
    nodes = (
        GraphNode(N_PROGRESSION_START, name="ALTTP fixture start", area="light_world"),
        GraphNode(N_PROGRESSION_SWORD, name="ALTTP sword check", area="castle"),
        GraphNode(N_PROGRESSION_LAMP, name="ALTTP lamp check", area="castle"),
        GraphNode(N_PROGRESSION_GOAL, name="ALTTP fixture goal", area="eastern"),
    )
    edges = (
        GraphEdge(
            N_PROGRESSION_START,
            N_PROGRESSION_SWORD,
            edge_id="z3_fixture_to_sword_check",
        ),
        GraphEdge(
            N_PROGRESSION_SWORD,
            N_PROGRESSION_GOAL,
            edge_id="z3_fixture_sword_to_goal",
            requires=Has(Z3_SWORD),
        ),
        GraphEdge(
            N_PROGRESSION_START,
            N_PROGRESSION_LAMP,
            edge_id="z3_fixture_to_lamp_check",
        ),
        GraphEdge(
            N_PROGRESSION_LAMP,
            N_PROGRESSION_GOAL,
            edge_id="z3_fixture_lamp_to_goal",
            requires=Has(Z3_SWORD),
        ),
    )
    checks = (
        ItemCheck(Z3_SWORD_CHECK, N_PROGRESSION_SWORD),
        ItemCheck(Z3_LAMP_CHECK, N_PROGRESSION_LAMP),
    )
    return RouteGraph(nodes, edges, checks)


PROGRESSION_GRAPH = build_progression_graph()
PLACEMENT_OVERLAY_A = SeedPlacement(
    {
        Z3_SWORD_CHECK: Z3_SWORD,
        Z3_LAMP_CHECK: Z3_LAMP,
    },
    seed_id="z3-fixture-a",
)
PLACEMENT_OVERLAY_B = SeedPlacement(
    {
        Z3_SWORD_CHECK: Z3_LAMP,
        Z3_LAMP_CHECK: Z3_SWORD,
    },
    seed_id="z3-fixture-b",
)
SEED_OVERLAY_A = PLACEMENT_OVERLAY_A
SEED_OVERLAY_B = PLACEMENT_OVERLAY_B


def plan_with_placement(
    placement: SeedPlacement = PLACEMENT_OVERLAY_A,
) -> tuple[GraphEdge, ...] | None:
    """Return the valid offline plan for one ALTTP fixture placement overlay."""
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


def plan_to_eastern_bow(
    capabilities: frozenset[str] | None = None,
) -> tuple[GraphEdge, ...] | None:
    caps = capabilities if capabilities is not None else frozenset({"sword", "lamp"})
    return path_with_capabilities(N_LINKS_HOUSE, N_EASTERN_BOW, caps)

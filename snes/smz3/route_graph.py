"""Capability-aware early SMZ3 progression graph.

Nodes are rooms/screens/portals; edges carry ``requires`` so future legs
compose without another seed-specific phase machine. Controllers for each
edge live in ``early_route`` / ``portal_route`` / ``outdoor_route`` /
``house_route``; this module only owns connectivity + capability gates.
"""

from __future__ import annotations

from retro_harness.adventure.graph import (
    GraphEdge,
    GraphNode,
    RouteGraph,
    RouteLeg,
)
from retro_harness.adventure.progression import (
    CapabilityId,
    Has,
    ItemCheck,
    SeedPlacement,
)
from retro_harness.adventure.routes import NamedRoute, RouteMilestone, RouteRegistry

# --- Node ids (stable string keys) ----------------------------------------

N_LANDING = "sm_landing_site"
N_PARLOR = "sm_parlor"
N_RED_DOOR = "sm_parlor_red_door"
N_PORTAL_SETTLED = "z3_fortune_teller_ow"
N_LINKS_HOUSE_OW = "z3_links_house_ow"
N_LINKS_HOUSE_IN = "z3_links_house_interior"
N_LINKS_HOUSE_CHEST = "z3_links_house_chest"

# Intermediate OW screens (optional path detail).
N_OW_2D = "z3_ow_0x2d"

# Namespaced offline fixture.  SMZ3 deliberately keeps ``sm:bombs`` and
# ``z3:bombs`` distinct even though both worlds use the word "bombs".
SM_BOMBS = CapabilityId("sm", "bombs")
Z3_BOMBS = CapabilityId("z3", "bombs")
N_PROGRESSION_START = "smz3_fixture_start"
N_PROGRESSION_SM_BOMBS = "smz3_fixture_sm_bombs_check"
N_PROGRESSION_Z3_BOMBS = "smz3_fixture_z3_bombs_check"
N_PROGRESSION_GOAL = "smz3_fixture_goal"
SM_BOMBS_CHECK = "smz3_fixture_sm_bombs"
Z3_BOMBS_CHECK = "smz3_fixture_z3_bombs"


def build_early_graph() -> RouteGraph:
    """Verified early quest graph (test seed 1337 geometry; fixed portals)."""
    nodes = (
        GraphNode(N_LANDING, name="Landing Site", area="crateria", tags=frozenset({"sm"})),
        GraphNode(N_PARLOR, name="Parlor and Alcatraz", area="crateria", tags=frozenset({"sm"})),
        GraphNode(
            N_RED_DOOR,
            name="Parlor red door (portal)",
            area="crateria",
            tags=frozenset({"sm", "portal"}),
            meta={"door_ptr": 0x8976},
        ),
        GraphNode(
            N_PORTAL_SETTLED,
            name="Fortune Teller OW",
            area="light_world",
            tags=frozenset({"z3", "overworld"}),
            meta={"screen_id": 0x35},
        ),
        GraphNode(
            N_OW_2D,
            name="OW screen $2D",
            area="light_world",
            tags=frozenset({"z3", "overworld"}),
            meta={"screen_id": 0x2D},
        ),
        GraphNode(
            N_LINKS_HOUSE_OW,
            name="Link's House OW",
            area="light_world",
            tags=frozenset({"z3", "overworld"}),
            meta={"screen_id": 0x2C},
        ),
        GraphNode(
            N_LINKS_HOUSE_IN,
            name="Link's House interior",
            area="light_world",
            tags=frozenset({"z3", "indoors"}),
            meta={"room_id": 0x0004},
        ),
        GraphNode(
            N_LINKS_HOUSE_CHEST,
            name="Link's House chest open",
            area="light_world",
            tags=frozenset({"z3", "chest"}),
        ),
    )
    edges = (
        GraphEdge(
            N_LANDING,
            N_PARLOR,
            edge_id="landing_to_parlor",
            direction="LEFT",
            verification="continuous",
            provenance="early_route.leave_landing_site_to_parlor",
        ),
        GraphEdge(
            N_PARLOR,
            N_RED_DOOR,
            edge_id="parlor_to_red_door",
            direction="DOWN",
            requires=frozenset({"missiles"}),
            verification="continuous",
            provenance="portal_route.descend_left_shaft_to_red_door",
            meta={"assist": "missile_red_door"},
        ),
        GraphEdge(
            N_RED_DOOR,
            N_PORTAL_SETTLED,
            edge_id="portal_sm_to_z3",
            verification="continuous",
            provenance="portal_route.open_red_door_portal",
            meta={"door_ptr": 0x8976, "z3_cave": 0x0122},
        ),
        GraphEdge(
            N_PORTAL_SETTLED,
            N_OW_2D,
            edge_id="fortune_to_mid",
            direction="UP",
            verification="continuous",
            provenance="outdoor_route corridor north",
            meta={"screen_path": [0x35, 0x2D]},
        ),
        GraphEdge(
            N_OW_2D,
            N_LINKS_HOUSE_OW,
            edge_id="mid_to_links_house_ow",
            direction="LEFT",
            verification="continuous",
            provenance="outdoor_route UP+LEFT",
            meta={"screen_path": [0x2D, 0x2C]},
        ),
        GraphEdge(
            N_LINKS_HOUSE_OW,
            N_LINKS_HOUSE_IN,
            edge_id="enter_links_house",
            direction="UP",
            verification="continuous",
            provenance="house_route.enter_links_house",
        ),
        GraphEdge(
            N_LINKS_HOUSE_IN,
            N_LINKS_HOUSE_CHEST,
            edge_id="open_links_house_chest",
            verification="continuous",
            provenance="house_route.open_links_house_chest",
            meta={"item": "seed_randomized"},
        ),
    )
    return RouteGraph(nodes, edges)


EARLY_GRAPH = build_early_graph()

EARLY_LEGS: tuple[RouteLeg, ...] = (
    RouteLeg("landing_to_parlor", N_LANDING, N_PARLOR, goal="landing_to_parlor"),
    RouteLeg(
        "parlor_to_red_door",
        N_PARLOR,
        N_RED_DOOR,
        requires=frozenset({"missiles"}),
        goal="landing_to_red_door",
    ),
    RouteLeg("portal_sm_to_z3", N_RED_DOOR, N_PORTAL_SETTLED, goal="landing_to_portal"),
    RouteLeg(
        "fortune_to_links_house_ow",
        N_PORTAL_SETTLED,
        N_LINKS_HOUSE_OW,
        goal="fortune_teller_to_links_house",
    ),
    RouteLeg(
        "enter_links_house",
        N_LINKS_HOUSE_OW,
        N_LINKS_HOUSE_IN,
        goal="enter_links_house",
    ),
    RouteLeg(
        "open_links_house_chest",
        N_LINKS_HOUSE_IN,
        N_LINKS_HOUSE_CHEST,
        goal="links_house_chest",
        acquires=frozenset({"links_house_chest_item"}),
    ),
)

# Direct fortune → house OW edge for pathfinding (skips mid-screen node when
# planning coarse legs). Outdoor controller still walks $35→$2D→$2C.
_COARSE_PATCH_EDGES = (
    GraphEdge(
        N_PORTAL_SETTLED,
        N_LINKS_HOUSE_OW,
        edge_id="fortune_to_links_house_ow",
        direction="UP",
        verification="continuous",
        provenance="outdoor_route.run_fortune_teller_to_links_house",
        meta={"screen_path": [0x35, 0x2D, 0x2C]},
    ),
)


def build_coarse_graph() -> RouteGraph:
    """Early graph plus a direct Fortune Teller → Link's House OW edge."""
    g = build_early_graph()
    return RouteGraph(g.nodes.values(), (*g.edges, *_COARSE_PATCH_EDGES))


COARSE_GRAPH = build_coarse_graph()


def build_progression_graph() -> RouteGraph:
    """Build the two-world placement fixture, without emulator state."""
    nodes = (
        GraphNode(N_PROGRESSION_START, name="SMZ3 fixture start"),
        GraphNode(N_PROGRESSION_SM_BOMBS, name="SM bombs check", area="crateria"),
        GraphNode(N_PROGRESSION_Z3_BOMBS, name="Z3 bombs check", area="light_world"),
        GraphNode(N_PROGRESSION_GOAL, name="SMZ3 fixture goal"),
    )
    edges = (
        GraphEdge(
            N_PROGRESSION_START,
            N_PROGRESSION_SM_BOMBS,
            edge_id="smz3_fixture_to_sm_bombs_check",
        ),
        GraphEdge(
            N_PROGRESSION_SM_BOMBS,
            N_PROGRESSION_GOAL,
            edge_id="smz3_fixture_sm_bombs_to_goal",
            requires=Has(SM_BOMBS),
        ),
        GraphEdge(
            N_PROGRESSION_START,
            N_PROGRESSION_Z3_BOMBS,
            edge_id="smz3_fixture_to_z3_bombs_check",
        ),
        GraphEdge(
            N_PROGRESSION_Z3_BOMBS,
            N_PROGRESSION_GOAL,
            edge_id="smz3_fixture_z3_bombs_to_goal",
            requires=Has(SM_BOMBS),
        ),
    )
    checks = (
        ItemCheck(SM_BOMBS_CHECK, N_PROGRESSION_SM_BOMBS),
        ItemCheck(Z3_BOMBS_CHECK, N_PROGRESSION_Z3_BOMBS),
    )
    return RouteGraph(nodes, edges, checks)


PROGRESSION_GRAPH = build_progression_graph()
PLACEMENT_OVERLAY_A = SeedPlacement(
    {
        SM_BOMBS_CHECK: SM_BOMBS,
        Z3_BOMBS_CHECK: Z3_BOMBS,
    },
    seed_id="smz3-fixture-a",
)
PLACEMENT_OVERLAY_B = SeedPlacement(
    {
        SM_BOMBS_CHECK: Z3_BOMBS,
        Z3_BOMBS_CHECK: SM_BOMBS,
    },
    seed_id="smz3-fixture-b",
)
SEED_OVERLAY_A = PLACEMENT_OVERLAY_A
SEED_OVERLAY_B = PLACEMENT_OVERLAY_B


def plan_with_placement(
    placement: SeedPlacement = PLACEMENT_OVERLAY_A,
) -> tuple[GraphEdge, ...] | None:
    """Return the valid offline plan for one SMZ3 fixture placement overlay."""
    return PROGRESSION_GRAPH.progression_path(
        N_PROGRESSION_START,
        N_PROGRESSION_GOAL,
        placement,
    )


def path_with_capabilities(
    source: str,
    target: str,
    capabilities: frozenset[str] | set[str] = frozenset(),
    *,
    coarse: bool = True,
) -> tuple[GraphEdge, ...] | None:
    graph = COARSE_GRAPH if coarse else EARLY_GRAPH
    return graph.shortest_path(source, target, capabilities=capabilities)


def plan_early_legs(
    *,
    stop_node: str = N_LINKS_HOUSE_CHEST,
    initial_capabilities: frozenset[str] | set[str] = frozenset({"missiles"}),
) -> tuple:
    """Plan contiguous early legs until *stop_node* (requires missiles by default)."""
    graph = COARSE_GRAPH
    selected: list[RouteLeg] = []
    for leg in EARLY_LEGS:
        selected.append(leg)
        if leg.target_id == stop_node:
            break
    else:
        raise ValueError(f"stop_node {stop_node!r} not on early leg chain")
    return graph.plan_legs(selected, initial_capabilities=initial_capabilities)


# Named catalog for tooling.
ROUTE_REGISTRY = RouteRegistry()
ROUTE_REGISTRY.register(
    NamedRoute(
        route_id="early_quest",
        display_name="Early quest: Landing → portal → Link's House chest",
        description=(
            "Verified on test seed 1337. Red door requires missiles "
            "(dev assist until natural path)."
        ),
        milestones=(
            RouteMilestone("parlor", N_PARLOR, "Parlor", "sm_parlor_controllable"),
            RouteMilestone("red_door", N_RED_DOOR, "Red door", "sm_red_door_band"),
            RouteMilestone(
                "portal", N_PORTAL_SETTLED, "Fortune Teller OW", "z3_controllable_ow_35"
            ),
            RouteMilestone(
                "links_house_ow",
                N_LINKS_HOUSE_OW,
                "Link's House OW",
                "z3_screen_2c",
            ),
            RouteMilestone(
                "chest",
                N_LINKS_HOUSE_CHEST,
                "House chest open",
                "chest_flag_or_inv_delta",
            ),
        ),
    )
)

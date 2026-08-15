"""NamedRoute / RouteLeg catalog for early OW item gates.

Lifts ``item_gate_hops`` geometry into first-class adventure routes so
dungeon catalogs do not hard-assume candle / white sword / bombs.

Hops are assisted pathing (not a STATUS promote). Shop buys and the white
sword cave tile are residual. Infinite-life assist fills the health low
nibble only and does **not** satisfy the white-sword container gate.
"""

from __future__ import annotations

from retro_harness.adventure.graph import GraphEdge, GraphNode, RouteGraph, RouteLeg
from retro_harness.adventure.routes import (
    NamedRoute,
    RouteMilestone,
    get_route as _get_route,
    list_routes as _list_routes,
    register_routes,
)
from zelda_i.item_gate_hops import (
    BOMB_CAPACITY_UPGRADE_LOCATIONS_SOURCE,
    BOMB_SHOP_HOPS,
    BOMB_SHOP_PRICE_SOURCE,
    BOMB_SHOP_VERIFICATION,
    CANDLE_SHOP_NEAR_HOPS,
    CANDLE_SHOP_NEAR_VERIFICATION,
    CANDLE_SHOP_PRICE_SOURCE,
    SCREEN_BOMB_SHOP,
    SCREEN_CANDLE_SHOP_NEAR,
    SCREEN_LOST_HILLS,
    SCREEN_WHITE_SWORD_CAVE,
    SCREEN_WHITE_SWORD_REGION,
    WHITE_SWORD_HOPS,
    WHITE_SWORD_LOST_HILLS_UPS,
    WHITE_SWORD_MIN_CONTAINERS,
    WHITE_SWORD_VERIFICATION,
)
from zelda_i.overworld import (
    NODE_START,
    SCREEN_START,
    ScreenHop,
    build_overworld_grid_graph,
    node_id_for_screen,
    path_screens_from_hops,
)

# Explicit Old Man gate: containers, not filled-hearts / infinite-life.
WHITE_SWORD_CONTAINER_GATE = f"heart_containers>={WHITE_SWORD_MIN_CONTAINERS}"

CAP_BLUE_CANDLE = "blue_candle"
CAP_WHITE_SWORD = "white_sword"
CAP_BOMBS = "bombs"

NODE_CANDLE_SHOP_NEAR = node_id_for_screen(SCREEN_CANDLE_SHOP_NEAR)
NODE_CANDLE_SHOP_CAVE = "cave_candle_66"
NODE_BOMB_SHOP = node_id_for_screen(SCREEN_BOMB_SHOP)
NODE_BOMB_SHOP_CAVE = "cave_bomb_shop_4a"
NODE_LOST_HILLS = node_id_for_screen(SCREEN_LOST_HILLS)
NODE_WHITE_SWORD_REGION = node_id_for_screen(SCREEN_WHITE_SWORD_REGION)
NODE_WHITE_SWORD_CAVE = "cave_white_sword_0a"

_ASSISTED_NOTE = (
    "Assisted hop geometry from item_gate_hops; inventory buy residual. "
    "Not a STATUS promote."
)


def _legs_from_hops(
    hops: tuple[ScreenHop, ...],
    start: int,
    *,
    prefix: str = "ow",
) -> tuple[RouteLeg, ...]:
    screens = path_screens_from_hops(start, hops)
    return tuple(
        RouteLeg(
            leg_id=f"{prefix}_{src:02x}_to_{dst:02x}",
            source_id=node_id_for_screen(src),
            target_id=node_id_for_screen(dst),
            goal=f"reach_screen_{dst:02X}",
        )
        for src, dst in zip(screens, screens[1:])
    )


def candle_shop_route_legs() -> tuple[RouteLeg, ...]:
    """Start 0x77 → near-start Blue Candle shop 0x66, then buy."""
    return (
        *_legs_from_hops(CANDLE_SHOP_NEAR_HOPS, SCREEN_START),
        RouteLeg(
            leg_id="buy_blue_candle",
            source_id=NODE_CANDLE_SHOP_NEAR,
            target_id=NODE_CANDLE_SHOP_CAVE,
            acquires=frozenset({CAP_BLUE_CANDLE}),
            goal="has_blue_candle",
            constraints=(f"costs_{CANDLE_SHOP_PRICE_SOURCE}_rupees",),
        ),
    )


def bomb_shop_route_legs() -> tuple[RouteLeg, ...]:
    """Start 0x77 → early bomb shop 0x4A (K-5), then buy inventory bombs."""
    return (
        *_legs_from_hops(BOMB_SHOP_HOPS, SCREEN_START),
        RouteLeg(
            leg_id="buy_bombs",
            source_id=NODE_BOMB_SHOP,
            target_id=NODE_BOMB_SHOP_CAVE,
            acquires=frozenset({CAP_BOMBS}),
            goal="has_bombs",
            constraints=(f"costs_{BOMB_SHOP_PRICE_SOURCE}_rupees",),
        ),
    )


def white_sword_route_legs() -> tuple[RouteLeg, ...]:
    """Start 0x77 → Lost Hills → region 0x0B → source cave 0x0A.

    The acquire leg requires ``WHITE_SWORD_CONTAINER_GATE``. Assist fill
    (infinite-life) is not that capability.
    """
    return (
        *_legs_from_hops(WHITE_SWORD_HOPS, SCREEN_START),
        RouteLeg(
            leg_id="lost_hills_to_white_sword_region",
            source_id=NODE_LOST_HILLS,
            target_id=NODE_WHITE_SWORD_REGION,
            goal="white_sword_region_0B",
            constraints=(f"lost_hills_up_x{WHITE_SWORD_LOST_HILLS_UPS}",),
        ),
        RouteLeg(
            leg_id="take_white_sword",
            source_id=NODE_WHITE_SWORD_REGION,
            target_id=NODE_WHITE_SWORD_CAVE,
            requires=frozenset({WHITE_SWORD_CONTAINER_GATE}),
            acquires=frozenset({CAP_WHITE_SWORD}),
            goal="has_white_sword",
            constraints=("source_cave_0a_residual",),
        ),
    )


def _promote_hop_edges(
    graph: RouteGraph,
    hops: tuple[ScreenHop, ...],
    start: int,
    *,
    verification: str,
    segment: str,
) -> list[GraphEdge]:
    screens = path_screens_from_hops(start, hops)
    pairs = {
        (node_id_for_screen(src), node_id_for_screen(dst)): hop
        for hop, src, dst in zip(hops, screens, screens[1:])
    }
    promoted: list[GraphEdge] = []
    for edge in graph.edges:
        hop = pairs.get((edge.source_id, edge.target_id))
        if hop is None:
            promoted.append(edge)
            continue
        promoted.append(
            GraphEdge(
                source_id=edge.source_id,
                target_id=edge.target_id,
                edge_id=edge.edge_id,
                direction=edge.direction,
                requires=edge.requires,
                cost=edge.cost,
                verification=verification,
                provenance="item_gate_hops",
                meta={
                    **dict(edge.meta),
                    "segment": segment,
                    "align_x": hop.align_x,
                    "align_y": hop.align_y,
                },
            )
        )
    return promoted


def build_item_gate_route_graph() -> RouteGraph:
    """Tiny local graph: hop screens + shop/cave portals."""
    screens: set[int] = {SCREEN_START, SCREEN_WHITE_SWORD_REGION}
    for hops in (CANDLE_SHOP_NEAR_HOPS, BOMB_SHOP_HOPS, WHITE_SWORD_HOPS):
        screens.update(path_screens_from_hops(SCREEN_START, hops))
    graph = build_overworld_grid_graph(screens=sorted(screens))
    edges = _promote_hop_edges(
        graph,
        CANDLE_SHOP_NEAR_HOPS,
        SCREEN_START,
        verification=CANDLE_SHOP_NEAR_VERIFICATION,
        segment="candle_shop_near",
    )
    tmp = RouteGraph(graph.nodes.values(), edges)
    edges = _promote_hop_edges(
        tmp,
        BOMB_SHOP_HOPS,
        SCREEN_START,
        verification=BOMB_SHOP_VERIFICATION,
        segment="bomb_shop",
    )
    tmp = RouteGraph(graph.nodes.values(), edges)
    edges = _promote_hop_edges(
        tmp,
        WHITE_SWORD_HOPS,
        SCREEN_START,
        verification=WHITE_SWORD_VERIFICATION,
        segment="white_sword",
    )

    extra_nodes = (
        GraphNode(
            node_id=NODE_CANDLE_SHOP_CAVE,
            name="blue_candle_shop_cave",
            area="cave",
            tags=frozenset({"cave", "item", "shop", "candle"}),
            meta={
                "overworld_screen": SCREEN_CANDLE_SHOP_NEAR,
                "item": CAP_BLUE_CANDLE,
                "price_source": CANDLE_SHOP_PRICE_SOURCE,
            },
        ),
        GraphNode(
            node_id=NODE_BOMB_SHOP_CAVE,
            name="bomb_shop_cave",
            area="cave",
            tags=frozenset({"cave", "item", "shop", "bombs"}),
            meta={
                "overworld_screen": SCREEN_BOMB_SHOP,
                "item": CAP_BOMBS,
                "price_source": BOMB_SHOP_PRICE_SOURCE,
            },
        ),
        GraphNode(
            node_id=NODE_WHITE_SWORD_CAVE,
            name="white_sword_cave",
            area="cave",
            tags=frozenset({"cave", "item", "sword"}),
            meta={
                "overworld_screen": SCREEN_WHITE_SWORD_CAVE,
                "region_screen": SCREEN_WHITE_SWORD_REGION,
                "item": CAP_WHITE_SWORD,
                "requires": WHITE_SWORD_CONTAINER_GATE,
                "note": "source cave 0x0A; OW west off 0x0B sealed live",
            },
        ),
    )
    extra_edges = (
        GraphEdge(
            source_id=NODE_CANDLE_SHOP_NEAR,
            target_id=NODE_CANDLE_SHOP_CAVE,
            edge_id="buy_blue_candle",
            acquires=frozenset({CAP_BLUE_CANDLE}),
            verification="planned",
            provenance="item_gate_shop",
            meta={"price_source": CANDLE_SHOP_PRICE_SOURCE},
        ),
        GraphEdge(
            source_id=NODE_BOMB_SHOP,
            target_id=NODE_BOMB_SHOP_CAVE,
            edge_id="buy_bombs",
            acquires=frozenset({CAP_BOMBS}),
            verification="planned",
            provenance="item_gate_shop",
            meta={"price_source": BOMB_SHOP_PRICE_SOURCE},
        ),
        GraphEdge(
            source_id=NODE_LOST_HILLS,
            target_id=NODE_WHITE_SWORD_REGION,
            edge_id="lost_hills_up_x4",
            direction="UP",
            verification=WHITE_SWORD_VERIFICATION,
            provenance="item_gate_hops",
            meta={"lost_hills_ups": WHITE_SWORD_LOST_HILLS_UPS},
        ),
        GraphEdge(
            source_id=NODE_WHITE_SWORD_REGION,
            target_id=NODE_WHITE_SWORD_CAVE,
            edge_id="enter_white_sword_cave",
            requires=frozenset({WHITE_SWORD_CONTAINER_GATE}),
            acquires=frozenset({CAP_WHITE_SWORD}),
            verification="planned",
            provenance="source_gathering",
            meta={
                "source_cave_screen": SCREEN_WHITE_SWORD_CAVE,
                "requires": WHITE_SWORD_CONTAINER_GATE,
                "note": "OW west off 0x0B sealed; infinite-life does not unlock",
            },
        ),
    )
    # Grid already has 0x1B→0x0B; replace that pair with the Lost Hills edge.
    kept = [
        edge
        for edge in edges
        if (edge.source_id, edge.target_id)
        != (NODE_LOST_HILLS, NODE_WHITE_SWORD_REGION)
    ]
    return RouteGraph(
        (*graph.nodes.values(), *extra_nodes),
        (*kept, *extra_edges),
    )


def candle_shop_route_plan(graph: RouteGraph | None = None):
    g = graph or build_item_gate_route_graph()
    return g.plan_legs(candle_shop_route_legs(), initial_capabilities=frozenset())


def bomb_shop_route_plan(graph: RouteGraph | None = None):
    g = graph or build_item_gate_route_graph()
    return g.plan_legs(bomb_shop_route_legs(), initial_capabilities=frozenset())


def white_sword_route_plan(
    graph: RouteGraph | None = None,
    *,
    initial_capabilities: frozenset[str] | None = None,
):
    """Plan white-sword legs. Default inventory is empty (gate blocks)."""
    g = graph or build_item_gate_route_graph()
    caps = (
        frozenset()
        if initial_capabilities is None
        else frozenset(initial_capabilities)
    )
    return g.plan_legs(white_sword_route_legs(), initial_capabilities=caps)


ROUTE_CANDLE_SHOP = NamedRoute(
    route_id="zelda_candle_shop",
    display_name="Start → Near-Start Blue Candle Shop",
    description=(
        f"{_ASSISTED_NOTE} Start 0x77 → Blue Candle open shop 0x66 "
        f"(near-start; avoid 0x67 trap). Acquires {CAP_BLUE_CANDLE}. "
        f"Price {CANDLE_SHOP_PRICE_SOURCE}R."
    ),
    milestones=(
        RouteMilestone(
            "start_overworld",
            NODE_START,
            "Start overworld",
            "is_on_start_overworld",
        ),
        RouteMilestone(
            "candle_shop_near",
            NODE_CANDLE_SHOP_NEAR,
            "Near-start Blue Candle shop screen 0x66",
            "candle_shop_near_screen",
        ),
        RouteMilestone(
            "blue_candle",
            NODE_CANDLE_SHOP_CAVE,
            "Blue Candle purchased",
            "has_blue_candle",
        ),
    ),
)

ROUTE_WHITE_SWORD = NamedRoute(
    route_id="zelda_white_sword",
    display_name="Start → White Sword Cave",
    description=(
        f"{_ASSISTED_NOTE} Assisted hops to Lost Hills 0x1B then "
        f"↑×{WHITE_SWORD_LOST_HILLS_UPS} into region 0x0B. Source cave 0x0A "
        f"residual (OW west off 0x0B sealed). Old Man requires "
        f"{WHITE_SWORD_CONTAINER_GATE}; unlimited-health assist only fills "
        f"the low nibble and does not grant containers. Acquires {CAP_WHITE_SWORD}."
    ),
    milestones=(
        RouteMilestone(
            "start_overworld",
            NODE_START,
            "Start overworld",
            "is_on_start_overworld",
        ),
        RouteMilestone(
            "white_sword_region",
            NODE_WHITE_SWORD_REGION,
            "White sword region 0x0B (L5 mouth)",
            "white_sword_region_reached",
        ),
        RouteMilestone(
            "white_sword_container_gate",
            NODE_WHITE_SWORD_CAVE,
            "White sword Old Man container gate",
            WHITE_SWORD_CONTAINER_GATE,
        ),
        RouteMilestone(
            "white_sword",
            NODE_WHITE_SWORD_CAVE,
            "White sword obtained (source cave 0x0A)",
            "has_white_sword",
        ),
    ),
)

_BOMB_CAP_NOTE = " / ".join(BOMB_CAPACITY_UPGRADE_LOCATIONS_SOURCE)

ROUTE_BOMB_SHOP = NamedRoute(
    route_id="zelda_bomb_shop",
    display_name="Start → Early Bomb Shop",
    description=(
        f"{_ASSISTED_NOTE} Start 0x77 → early bomb shop 0x4A (K-5). "
        f"Acquires {CAP_BOMBS} (inventory, not capacity). "
        f"Price {BOMB_SHOP_PRICE_SOURCE}R. Capacity upgrades 8→12→16 are "
        f"{_BOMB_CAP_NOTE} Old Men at 100R each — not this route."
    ),
    milestones=(
        RouteMilestone(
            "start_overworld",
            NODE_START,
            "Start overworld",
            "is_on_start_overworld",
        ),
        RouteMilestone(
            "bomb_shop",
            NODE_BOMB_SHOP,
            "Early bomb shop screen 0x4A (K-5)",
            "bomb_shop_screen",
        ),
        RouteMilestone(
            "bombs",
            NODE_BOMB_SHOP_CAVE,
            "Bombs purchased (inventory)",
            "has_bombs",
        ),
    ),
)

ROUTE_REGISTRY_ITEM_GATE: dict[str, NamedRoute] = {}
register_routes(
    ROUTE_REGISTRY_ITEM_GATE,
    ROUTE_CANDLE_SHOP,
    "candle",
    "candle_shop",
    "candle_shop_near",
    "blue_candle",
)
register_routes(
    ROUTE_REGISTRY_ITEM_GATE,
    ROUTE_WHITE_SWORD,
    "white_sword",
    "white",
)
register_routes(
    ROUTE_REGISTRY_ITEM_GATE,
    ROUTE_BOMB_SHOP,
    "bomb_shop",
    "bombs",
)


def get_item_gate_route(route_id: str) -> NamedRoute:
    return _get_route(ROUTE_REGISTRY_ITEM_GATE, route_id)


def list_item_gate_routes() -> list[NamedRoute]:
    return _list_routes(ROUTE_REGISTRY_ITEM_GATE)


__all__ = [
    "WHITE_SWORD_CONTAINER_GATE",
    "CAP_BLUE_CANDLE",
    "CAP_WHITE_SWORD",
    "CAP_BOMBS",
    "NODE_CANDLE_SHOP_NEAR",
    "NODE_CANDLE_SHOP_CAVE",
    "NODE_BOMB_SHOP",
    "NODE_BOMB_SHOP_CAVE",
    "NODE_LOST_HILLS",
    "NODE_WHITE_SWORD_REGION",
    "NODE_WHITE_SWORD_CAVE",
    "ROUTE_CANDLE_SHOP",
    "ROUTE_WHITE_SWORD",
    "ROUTE_BOMB_SHOP",
    "ROUTE_REGISTRY_ITEM_GATE",
    "build_item_gate_route_graph",
    "candle_shop_route_legs",
    "white_sword_route_legs",
    "bomb_shop_route_legs",
    "candle_shop_route_plan",
    "white_sword_route_plan",
    "bomb_shop_route_plan",
    "get_item_gate_route",
    "list_item_gate_routes",
]

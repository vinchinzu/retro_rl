"""Shared route-graph primitives for nonlinear adventure games.

Consumers: ``zelda_i``, ``metroid``, ``super_metroid`` (via thin loaders),
and ``alttp`` opening-route graphs. Game-specific stop predicates and WRAM
maps stay local.
"""

from adventure_common.graph import (
    GraphEdge,
    GraphNode,
    ObservedTransition,
    PlannedLeg,
    ProgressionMilestone,
    RouteGraph,
    RouteLeg,
    RoutePatch,
    apply_milestones,
    normalize_capability,
    promote_edge_verification,
    shortest_path,
)
from adventure_common.hashutil import sha256_file
from adventure_common.nav import (
    Waypoint,
    WaypointFollower,
    direction_to_waypoint,
    direction_toward,
    manhattan,
    reached_waypoint,
)
from adventure_common.routes import (
    NamedRoute,
    RouteMilestone,
    RouteRegistry,
    get_route,
    list_routes,
    register_routes,
)

__all__ = [
    "GraphEdge",
    "GraphNode",
    "NamedRoute",
    "ObservedTransition",
    "PlannedLeg",
    "ProgressionMilestone",
    "RouteGraph",
    "RouteLeg",
    "RouteMilestone",
    "RoutePatch",
    "RouteRegistry",
    "Waypoint",
    "WaypointFollower",
    "apply_milestones",
    "direction_to_waypoint",
    "direction_toward",
    "get_route",
    "list_routes",
    "manhattan",
    "normalize_capability",
    "promote_edge_verification",
    "reached_waypoint",
    "register_routes",
    "sha256_file",
    "shortest_path",
]

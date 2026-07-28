"""Shared route-graph primitives for nonlinear adventure games.

Consumers: ``zelda_i`` (NES), ``metroid`` (NES). Super Metroid reuses the same
capability spelling and shortest-path core via thin wrappers in
``super_metroid.progression`` / ``super_metroid.map_planning``.

Second consumer (Metroid) proved the Phase 4 graph core; keep game-specific
stop predicates and WRAM maps local.
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

__all__ = [
    "GraphEdge",
    "GraphNode",
    "ObservedTransition",
    "PlannedLeg",
    "ProgressionMilestone",
    "RouteGraph",
    "RouteLeg",
    "RoutePatch",
    "apply_milestones",
    "normalize_capability",
    "promote_edge_verification",
    "shortest_path",
]

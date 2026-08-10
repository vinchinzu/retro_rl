"""L4 item-logic planner consumer over the ALTTP escape capability graph.

Offline / dry-run surface for :mod:`retro_harness.adventure.planner` using
real opening-route edges (not synthetic fixtures). Emulator play stays in
segments; this only plans a skill/edge sequence from inventory.
"""

from __future__ import annotations

from typing import Iterable

from retro_harness.adventure.planner import (
    PlanBudget,
    PlanRequest,
    PlanResult,
    PlanStatus,
    plan,
)

from alttp.opening_route.escape_graph import (
    CAP_LAMP,
    N_CASTLE_GROUNDS,
    N_ROOM_50,
    N_SANCTUARY,
    NATURAL_HOUSE_EXIT_CAPABILITIES,
    PATH_PRIMARY,
    escape_route_graph,
    normalize_capability,
)


def escape_plan_request(
    source_id: str = N_CASTLE_GROUNDS,
    target_id: str = N_ROOM_50,
    *,
    capabilities: Iterable[str] | None = None,
    path: str = PATH_PRIMARY,
    budget: PlanBudget | None = None,
) -> PlanRequest:
    """Build a :class:`PlanRequest` from the live escape graph.

    Default target is the continuous tip (``room_50``). Sanctuary remains a
    planned abstract goal and may be unreachable under continuous-only edges.
    """
    graph = escape_route_graph()
    if path == "all":
        edges = tuple(graph.edges)
    else:
        edges = tuple(
            edge
            for edge in graph.edges
            if str(edge.meta.get("path", PATH_PRIMARY)) == path
        )
    if capabilities is None:
        caps = NATURAL_HOUSE_EXIT_CAPABILITIES
    else:
        caps = frozenset(normalize_capability(v) for v in capabilities)
    return PlanRequest(
        edges,
        source_id,
        target_id,
        capabilities=caps,
        budget=budget or PlanBudget(max_expansions=2000),
    )


def plan_escape(
    source_id: str = N_CASTLE_GROUNDS,
    target_id: str = N_ROOM_50,
    *,
    capabilities: Iterable[str] | None = None,
    path: str = PATH_PRIMARY,
    budget: PlanBudget | None = None,
) -> PlanResult:
    """Plan a short escape edge sequence given inventory (offline dry plan)."""
    return plan(
        escape_plan_request(
            source_id,
            target_id,
            capabilities=capabilities,
            path=path,
            budget=budget,
        )
    )


def plan_escape_summary(
    source_id: str = N_CASTLE_GROUNDS,
    target_id: str = N_ROOM_50,
    *,
    capabilities: Iterable[str] | None = None,
) -> dict[str, object]:
    """Compact dry-run record for tests / CLI."""
    result = plan_escape(
        source_id, target_id, capabilities=capabilities
    )
    return {
        "game": "alttp",
        "subgraph": "escape_opening",
        "source": source_id,
        "target": target_id,
        "capabilities": sorted(
            NATURAL_HOUSE_EXIT_CAPABILITIES
            if capabilities is None
            else frozenset(normalize_capability(v) for v in capabilities)
        ),
        "status": result.status.value,
        "path_edge_ids": [e.edge_id for e in result.path],
        "total_cost": result.total_cost,
        "final_capabilities": sorted(
            str(c) for c in result.final_progression.capabilities
        ),
        "found": result.found,
    }


__all__ = [
    "CAP_LAMP",
    "N_CASTLE_GROUNDS",
    "N_ROOM_50",
    "N_SANCTUARY",
    "PlanResult",
    "PlanStatus",
    "escape_plan_request",
    "plan_escape",
    "plan_escape_summary",
]

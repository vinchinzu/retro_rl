"""L4 item-logic planner consumer over Super Metroid progression graphs.

Offline / dry-run surface for :mod:`retro_harness.adventure.planner` using
real Morph-stage room edges (Ceres → Morph Ball). No emulator required.
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
from retro_harness.adventure.graph import normalize_capability
from super_metroid.progression.stages.morph import MORPH_GRAPH
from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_ELEVATOR,
    ROOM_LANDING_SITE,
    ROOM_MORPH,
)


def morph_plan_request(
    source_room_id: int = ROOM_CERES_ELEVATOR,
    target_room_id: int = ROOM_MORPH,
    *,
    capabilities: Iterable[str] = (),
    budget: PlanBudget | None = None,
) -> PlanRequest:
    """Build a :class:`PlanRequest` from the Morph progression stage graph."""
    edges = tuple(edge.as_graph_edge() for edge in MORPH_GRAPH.edges)
    caps = frozenset(normalize_capability(v) for v in capabilities)
    return PlanRequest(
        edges,
        source_room_id,
        target_room_id,
        capabilities=caps,
        budget=budget or PlanBudget(max_expansions=2000),
    )


def plan_morph(
    source_room_id: int = ROOM_CERES_ELEVATOR,
    target_room_id: int = ROOM_MORPH,
    *,
    capabilities: Iterable[str] = (),
    budget: PlanBudget | None = None,
) -> PlanResult:
    """Plan Ceres → Morph (or other Morph-subgraph endpoints) offline."""
    return plan(
        morph_plan_request(
            source_room_id,
            target_room_id,
            capabilities=capabilities,
            budget=budget,
        )
    )


def plan_morph_summary(
    source_room_id: int = ROOM_CERES_ELEVATOR,
    target_room_id: int = ROOM_MORPH,
    *,
    capabilities: Iterable[str] = (),
) -> dict[str, object]:
    """Compact dry-run record for tests / CLI."""
    result = plan_morph(
        source_room_id,
        target_room_id,
        capabilities=capabilities,
    )
    return {
        "game": "super_metroid",
        "subgraph": "morph",
        "source": source_room_id,
        "source_hex": f"0x{source_room_id:04X}",
        "target": target_room_id,
        "target_hex": f"0x{target_room_id:04X}",
        "capabilities": sorted(
            normalize_capability(v) for v in capabilities
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
    "ROOM_CERES_ELEVATOR",
    "ROOM_LANDING_SITE",
    "ROOM_MORPH",
    "PlanResult",
    "PlanStatus",
    "morph_plan_request",
    "plan_morph",
    "plan_morph_summary",
]

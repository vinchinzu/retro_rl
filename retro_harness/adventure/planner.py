"""Bounded, deterministic planning over monotonic progression state."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from heapq import heappop, heappush
from itertools import count
from math import isfinite
from typing import Any, Hashable, Iterable, Mapping

from retro_harness.adventure.graph import (
    GraphCapability,
    GraphEdge,
    _collect_node_checks,
    _edge_acquires,
    _edge_order_key,
    _edge_requires,
    _normalize_graph_capabilities,
)
from retro_harness.adventure.progression import (
    CapabilityValue,
    ItemCheck,
    ProgressionState,
    Requirement,
    SeedPlacement,
    coerce_placement,
)


@dataclass(frozen=True, slots=True)
class PlanBudget:
    """Hard search limits; exhausted budgets are a result, not `None`."""

    max_expansions: int = 500

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_expansions, bool)
            or not isinstance(self.max_expansions, int)
            or self.max_expansions < 1
        ):
            raise ValueError("max_expansions must be a positive integer")

    def to_record(self) -> dict[str, int]:
        return {"max_expansions": self.max_expansions}


class PlanStatus(str, Enum):
    FOUND = "FOUND"
    UNREACHABLE = "UNREACHABLE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


@dataclass(frozen=True, slots=True)
class FrontierBlocker:
    """One stable explanation for an observed non-traversable edge."""

    edge_id: str
    source_id: Hashable
    target_id: Hashable
    requirement: Requirement | frozenset[str]

    def to_record(self) -> dict[str, Any]:
        requires: Any
        if isinstance(self.requirement, Requirement):
            requires = self.requirement.to_dict()
        else:
            requires = sorted(self.requirement)
        return {
            "edge_id": self.edge_id,
            "source": _node_record(self.source_id),
            "target": _node_record(self.target_id),
            "requires": requires,
        }


@dataclass(frozen=True, slots=True)
class ResourceBlocker:
    """A traversable graph edge rejected by a bounded resource constraint."""

    edge_id: str
    source_id: Hashable
    target_id: Hashable
    resource: str
    required: float
    available: float
    reason: str

    def to_record(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source": _node_record(self.source_id),
            "target": _node_record(self.target_id),
            "resource": self.resource,
            "required": self.required,
            "available": self.available,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True, init=False)
class PlanRequest:
    """Canonical search input independent of iterable ordering."""

    edges: tuple[GraphEdge, ...]
    source_id: Hashable
    target_id: Hashable
    checks: tuple[ItemCheck, ...]
    placements: SeedPlacement
    capabilities: frozenset[GraphCapability]
    collected_checks: frozenset[str]
    budget: PlanBudget

    def __init__(
        self,
        edges: Iterable[GraphEdge],
        source_id: Hashable,
        target_id: Hashable,
        *,
        checks: Iterable[ItemCheck] = (),
        placements: SeedPlacement
        | Mapping[str, CapabilityValue]
        | Iterable[SeedPlacement]
        | Iterable[tuple[str, CapabilityValue]]
        | None = None,
        capabilities: Iterable[GraphCapability] = (),
        collected_checks: Iterable[str] = (),
        budget: PlanBudget | None = None,
    ) -> None:
        canonical_edges = tuple(sorted(tuple(edges), key=_edge_order_key))
        canonical_checks = tuple(sorted(tuple(checks), key=lambda value: value.check_id))
        edge_ids: set[str] = set()
        for edge in canonical_edges:
            if edge.edge_id in edge_ids:
                raise ValueError(f"duplicate edge ID: {edge.edge_id!r}")
            edge_ids.add(edge.edge_id)
            if not isfinite(edge.cost) or edge.cost < 0:
                raise ValueError("plan requires finite, non-negative edge costs")
        check_ids: set[str] = set()
        for check in canonical_checks:
            if check.check_id in check_ids:
                raise ValueError(f"duplicate item check ID: {check.check_id!r}")
            check_ids.add(check.check_id)
        normalized_checks = frozenset(collected_checks)
        if not all(isinstance(value, str) for value in normalized_checks):
            raise TypeError("collected check IDs must be strings")
        object.__setattr__(self, "edges", canonical_edges)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "checks", canonical_checks)
        object.__setattr__(self, "placements", coerce_placement(placements))
        object.__setattr__(
            self,
            "capabilities",
            _normalize_graph_capabilities(capabilities),
        )
        object.__setattr__(self, "collected_checks", normalized_checks)
        object.__setattr__(self, "budget", budget or PlanBudget())

    def to_record(self) -> dict[str, Any]:
        return {
            "source": _node_record(self.source_id),
            "target": _node_record(self.target_id),
            "edges": [_edge_record(edge) for edge in self.edges],
            "checks": [check.to_dict() for check in self.checks],
            "placements": self.placements.to_dict(),
            "capabilities": sorted(str(value) for value in self.capabilities),
            "collected_checks": sorted(self.collected_checks),
            "budget": self.budget.to_record(),
        }

    @property
    def identity_digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_record()).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PlanResult:
    """Explainable bounded-search outcome."""

    status: PlanStatus
    path: tuple[GraphEdge, ...]
    total_cost: float | None
    final_progression: ProgressionState
    expanded_count: int
    dominated_pruned: int
    frontier_blockers: tuple[FrontierBlocker, ...] = field(default_factory=tuple)
    request_digest: str = ""
    resource_trajectory: tuple[Mapping[str, float], ...] = field(default_factory=tuple)
    resource_blockers: tuple[ResourceBlocker, ...] = field(default_factory=tuple)
    risk_adjusted_cost: float | None = None

    @property
    def found(self) -> bool:
        return self.status is PlanStatus.FOUND

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "path": [_edge_record(edge) for edge in self.path],
            "path_edge_ids": [edge.edge_id for edge in self.path],
            "total_cost": self.total_cost,
            "final_progression": _progression_record(self.final_progression),
            "expanded_count": self.expanded_count,
            "dominated_pruned": self.dominated_pruned,
            "frontier_blockers": [
                blocker.to_record() for blocker in self.frontier_blockers
            ],
            "request_digest": self.request_digest,
            "resource_trajectory": [dict(value) for value in self.resource_trajectory],
            "resource_blockers": [
                blocker.to_record() for blocker in self.resource_blockers
            ],
            "risk_adjusted_cost": self.risk_adjusted_cost,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_record())


StateKey = tuple[Hashable, frozenset[GraphCapability], frozenset[str]]
PathKey = tuple[tuple[str, ...], ...]
Rank = tuple[float, PathKey]


def plan(request: PlanRequest) -> PlanResult:
    """Run deterministic Dijkstra search with same-node dominance pruning."""
    if not isinstance(request, PlanRequest):
        raise TypeError("request must be a PlanRequest")

    checks_by_node = _checks_by_node(request.checks)
    initial = _collect_node_checks(
        ProgressionState(
            request.source_id,
            request.capabilities,
            request.collected_checks,
        ),
        checks_by_node,
        request.placements,
    )
    initial_key = _state_key(initial)
    initial_rank: Rank = (0.0, ())
    best: dict[StateKey, Rank] = {initial_key: initial_rank}
    labels: dict[Hashable, set[StateKey]] = defaultdict(set)
    labels[initial.node].add(initial_key)
    parents: dict[StateKey, tuple[StateKey, GraphEdge]] = {}
    sequence = count()
    pending: list[tuple[float, PathKey, int, StateKey]] = [
        (0.0, (), next(sequence), initial_key)
    ]
    outgoing: dict[Hashable, list[GraphEdge]] = defaultdict(list)
    for edge in request.edges:
        outgoing[edge.source_id].append(edge)
    blockers: dict[str, FrontierBlocker] = {}
    expanded = 0
    dominated_pruned = 0
    last_state = initial

    while pending:
        cost, path_key, _sequence, state_key = heappop(pending)
        if best.get(state_key) != (cost, path_key):
            continue
        node_id, current_caps, current_checks = state_key
        current = ProgressionState(node_id, current_caps, current_checks)
        last_state = current
        if node_id == request.target_id:
            return _result(
                request,
                PlanStatus.FOUND,
                state_key,
                cost,
                current,
                expanded,
                dominated_pruned,
                blockers,
                parents,
            )
        if expanded >= request.budget.max_expansions:
            return _result(
                request,
                PlanStatus.BUDGET_EXHAUSTED,
                state_key,
                None,
                current,
                expanded,
                dominated_pruned,
                blockers,
                parents,
                include_path=False,
            )
        expanded += 1

        for edge in outgoing.get(node_id, ()):
            if not _edge_requires(edge, current_caps):
                blockers[edge.edge_id] = FrontierBlocker(
                    edge.edge_id,
                    edge.source_id,
                    edge.target_id,
                    edge.requires,
                )
                continue
            blockers.pop(edge.edge_id, None)
            next_progression = _collect_node_checks(
                ProgressionState(
                    edge.target_id,
                    current_caps | _edge_acquires(edge),
                    current_checks,
                ),
                checks_by_node,
                request.placements,
            )
            next_key = _state_key(next_progression)
            next_rank: Rank = (
                cost + edge.cost,
                path_key + (_edge_order_key(edge),),
            )
            if _is_dominated(next_key, next_rank, labels, best):
                dominated_pruned += 1
                continue
            dominated_pruned += _remove_dominated(
                next_key,
                next_rank,
                labels,
                best,
            )
            best[next_key] = next_rank
            labels[next_key[0]].add(next_key)
            parents[next_key] = (state_key, edge)
            heappush(
                pending,
                (next_rank[0], next_rank[1], next(sequence), next_key),
            )

    return _result(
        request,
        PlanStatus.UNREACHABLE,
        _state_key(last_state),
        None,
        last_state,
        expanded,
        dominated_pruned,
        blockers,
        parents,
        include_path=False,
    )


def _checks_by_node(
    checks: tuple[ItemCheck, ...],
) -> dict[Hashable, tuple[ItemCheck, ...]]:
    grouped: dict[Hashable, list[ItemCheck]] = defaultdict(list)
    for check in checks:
        grouped[check.node_id].append(check)
    return {
        node: tuple(sorted(values, key=lambda value: value.check_id))
        for node, values in grouped.items()
    }


def _state_key(state: ProgressionState) -> StateKey:
    return (state.node, state.capabilities, state.collected_checks)


def _state_dominates(
    first: StateKey,
    first_rank: Rank,
    second: StateKey,
    second_rank: Rank,
) -> bool:
    return (
        first[0] == second[0]
        and first[1].issuperset(second[1])
        and first[2].issuperset(second[2])
        and first_rank <= second_rank
    )


def _is_dominated(
    candidate: StateKey,
    rank: Rank,
    labels: Mapping[Hashable, set[StateKey]],
    best: Mapping[StateKey, Rank],
) -> bool:
    return any(
        _state_dominates(existing, best[existing], candidate, rank)
        for existing in labels.get(candidate[0], ())
        if existing in best
    )


def _remove_dominated(
    candidate: StateKey,
    rank: Rank,
    labels: dict[Hashable, set[StateKey]],
    best: dict[StateKey, Rank],
) -> int:
    removed = 0
    for existing in tuple(labels.get(candidate[0], ())):
        existing_rank = best.get(existing)
        if existing_rank is None:
            labels[candidate[0]].discard(existing)
            continue
        if _state_dominates(candidate, rank, existing, existing_rank) and (
            candidate != existing or rank < existing_rank
        ):
            labels[candidate[0]].discard(existing)
            best.pop(existing, None)
            removed += 1
    return removed


def _result(
    request: PlanRequest,
    status: PlanStatus,
    state_key: StateKey,
    total_cost: float | None,
    final_progression: ProgressionState,
    expanded: int,
    dominated_pruned: int,
    blockers: Mapping[str, FrontierBlocker],
    parents: Mapping[StateKey, tuple[StateKey, GraphEdge]],
    *,
    include_path: bool = True,
) -> PlanResult:
    path: list[GraphEdge] = []
    if include_path:
        cursor = state_key
        while cursor in parents:
            cursor, edge = parents[cursor]
            path.append(edge)
        path.reverse()
    return PlanResult(
        status=status,
        path=tuple(path),
        total_cost=total_cost,
        final_progression=final_progression,
        expanded_count=expanded,
        dominated_pruned=dominated_pruned,
        frontier_blockers=tuple(blockers[key] for key in sorted(blockers)),
        request_digest=request.identity_digest,
    )


def _node_record(value: Hashable) -> dict[str, str]:
    return {"type": type(value).__qualname__, "repr": repr(value)}


def _edge_record(edge: GraphEdge) -> dict[str, Any]:
    requires: Any
    if isinstance(edge.requires, Requirement):
        requires = edge.requires.to_dict()
    else:
        requires = sorted(edge.requires)
    return {
        "edge_id": edge.edge_id,
        "source": _node_record(edge.source_id),
        "target": _node_record(edge.target_id),
        "direction": edge.direction,
        "requires": requires,
        "acquires": sorted(str(value) for value in edge.acquires),
        "cost": edge.cost,
        "verification": edge.verification,
        "provenance": edge.provenance,
        "meta": dict(edge.meta),
    }


def _progression_record(state: ProgressionState) -> dict[str, Any]:
    return {
        "node": _node_record(state.node),
        "capabilities": sorted(str(value) for value in state.capabilities),
        "collected_checks": sorted(state.collected_checks),
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )

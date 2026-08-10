"""Bounded, deterministic planning over monotonic progression state."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from heapq import heappop, heappush
from itertools import count
from math import isfinite
from typing import Any, Hashable, Iterable, Mapping, Protocol

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
from retro_harness.identity import canonical_json as _canonical_json, sha256_bytes


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
        return sha256_bytes(_canonical_json(self.to_record()).encode("utf-8"))


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


SearchState = tuple[
    Hashable,
    frozenset[GraphCapability],
    frozenset[str],
    Hashable,
]
PathKey = tuple[tuple[str, ...], ...]
Rank = tuple[float, PathKey]


@dataclass(frozen=True, slots=True)
class SearchTransition:
    """One extension-owned edge transition for the shared search core."""

    state: Hashable | None
    cost: float | None
    resource_blockers: tuple[ResourceBlocker, ...] = ()

    def __post_init__(self) -> None:
        if self.resource_blockers:
            if self.state is not None or self.cost is not None:
                raise ValueError("blocked transitions cannot carry state or cost")
            return
        if self.state is None or self.cost is None:
            raise ValueError("traversable transitions require state and cost")
        if not isfinite(self.cost) or self.cost < 0:
            raise ValueError("transition cost must be finite and non-negative")


class PlanSearchDimension(Protocol):
    """Optional state/cost dimension consumed by :func:`search`."""

    request_digest: str
    initial_state: Hashable
    risk_adjusted: bool

    def transition(self, edge: GraphEdge, state: Hashable) -> SearchTransition: ...

    def dominates(self, first: Hashable, second: Hashable) -> bool: ...

    def state_record(self, state: Hashable) -> Mapping[str, float] | None: ...


@dataclass(frozen=True, slots=True)
class _ProgressionOnlyDimension:
    request_digest: str
    initial_state: Hashable = ()
    risk_adjusted: bool = False

    def transition(self, edge: GraphEdge, state: Hashable) -> SearchTransition:
        return SearchTransition(state, edge.cost)

    def dominates(self, first: Hashable, second: Hashable) -> bool:
        return first == second

    def state_record(self, state: Hashable) -> None:
        return None


def plan(request: PlanRequest) -> PlanResult:
    """Plan over progression only using the shared bounded search core."""
    if not isinstance(request, PlanRequest):
        raise TypeError("request must be a PlanRequest")
    return search(request, _ProgressionOnlyDimension(request.identity_digest))


def search(
    request: PlanRequest,
    dimension: PlanSearchDimension,
) -> PlanResult:
    """Run one deterministic bounded search over progression plus a dimension."""
    if not isinstance(request, PlanRequest):
        raise TypeError("request must be a PlanRequest")

    node_checks = checks_by_node(request.checks)
    initial = _collect_node_checks(
        ProgressionState(
            request.source_id,
            request.capabilities,
            request.collected_checks,
        ),
        node_checks,
        request.placements,
    )
    initial_key: SearchState = (
        initial.node,
        initial.capabilities,
        initial.collected_checks,
        dimension.initial_state,
    )
    initial_rank: Rank = (0.0, ())
    best: dict[SearchState, Rank] = {initial_key: initial_rank}
    labels: dict[Hashable, set[SearchState]] = defaultdict(set)
    labels[initial.node].add(initial_key)
    parents: dict[SearchState, tuple[SearchState, GraphEdge]] = {}
    sequence = count()
    pending: list[tuple[float, PathKey, int, SearchState]] = [
        (0.0, (), next(sequence), initial_key)
    ]
    outgoing: dict[Hashable, list[GraphEdge]] = defaultdict(list)
    for edge in request.edges:
        outgoing[edge.source_id].append(edge)
    frontier_blockers: dict[str, FrontierBlocker] = {}
    resource_blockers: dict[tuple[str, str], ResourceBlocker] = {}
    expanded = 0
    dominated_pruned = 0
    last_key = initial_key

    while pending:
        cost, path_key, _sequence, state_key = heappop(pending)
        if best.get(state_key) != (cost, path_key):
            continue
        last_key = state_key
        node_id, current_caps, current_checks, dimension_state = state_key
        current = ProgressionState(node_id, current_caps, current_checks)
        if node_id == request.target_id:
            return _search_result(
                status=PlanStatus.FOUND,
                state_key=state_key,
                total_cost=cost,
                expanded=expanded,
                dominated_pruned=dominated_pruned,
                frontier_blockers=frontier_blockers,
                resource_blockers=resource_blockers,
                parents=parents,
                dimension=dimension,
            )
        if expanded >= request.budget.max_expansions:
            return _search_result(
                status=PlanStatus.BUDGET_EXHAUSTED,
                state_key=state_key,
                total_cost=None,
                expanded=expanded,
                dominated_pruned=dominated_pruned,
                frontier_blockers=frontier_blockers,
                resource_blockers=resource_blockers,
                parents=parents,
                dimension=dimension,
                include_path=False,
            )
        expanded += 1

        for edge in outgoing.get(node_id, ()):
            if not _edge_requires(edge, current_caps):
                frontier_blockers[edge.edge_id] = FrontierBlocker(
                    edge.edge_id,
                    edge.source_id,
                    edge.target_id,
                    edge.requires,
                )
                continue
            frontier_blockers.pop(edge.edge_id, None)
            transition = dimension.transition(edge, dimension_state)
            if transition.resource_blockers:
                for blocker in transition.resource_blockers:
                    resource_blockers[(blocker.edge_id, blocker.resource)] = blocker
                continue
            for key in tuple(resource_blockers):
                if key[0] == edge.edge_id:
                    resource_blockers.pop(key)
            next_progression = _collect_node_checks(
                ProgressionState(
                    edge.target_id,
                    current_caps | _edge_acquires(edge),
                    current_checks,
                ),
                node_checks,
                request.placements,
            )
            next_key: SearchState = (
                next_progression.node,
                next_progression.capabilities,
                next_progression.collected_checks,
                transition.state,
            )
            next_rank: Rank = (
                cost + transition.cost,  # type: ignore[operator]
                path_key + (_edge_order_key(edge),),
            )
            if _is_dominated(next_key, next_rank, labels, best, dimension):
                dominated_pruned += 1
                continue
            dominated_pruned += _remove_dominated(
                next_key,
                next_rank,
                labels,
                best,
                dimension,
            )
            best[next_key] = next_rank
            labels[next_key[0]].add(next_key)
            parents[next_key] = (state_key, edge)
            heappush(
                pending,
                (next_rank[0], next_rank[1], next(sequence), next_key),
            )

    return _search_result(
        status=PlanStatus.UNREACHABLE,
        state_key=last_key,
        total_cost=None,
        expanded=expanded,
        dominated_pruned=dominated_pruned,
        frontier_blockers=frontier_blockers,
        resource_blockers=resource_blockers,
        parents=parents,
        dimension=dimension,
        include_path=False,
    )


def checks_by_node(
    checks: tuple[ItemCheck, ...],
) -> dict[Hashable, tuple[ItemCheck, ...]]:
    grouped: dict[Hashable, list[ItemCheck]] = defaultdict(list)
    for check in checks:
        grouped[check.node_id].append(check)
    return {
        node: tuple(sorted(values, key=lambda value: value.check_id))
        for node, values in grouped.items()
    }


def _state_dominates(
    first: SearchState,
    first_rank: Rank,
    second: SearchState,
    second_rank: Rank,
    dimension: PlanSearchDimension,
) -> bool:
    return (
        first[0] == second[0]
        and first[1].issuperset(second[1])
        and first[2].issuperset(second[2])
        and dimension.dominates(first[3], second[3])
        and first_rank <= second_rank
    )


def _is_dominated(
    candidate: SearchState,
    rank: Rank,
    labels: Mapping[Hashable, set[SearchState]],
    best: Mapping[SearchState, Rank],
    dimension: PlanSearchDimension,
) -> bool:
    return any(
        _state_dominates(existing, best[existing], candidate, rank, dimension)
        for existing in labels.get(candidate[0], ())
        if existing in best
    )


def _remove_dominated(
    candidate: SearchState,
    rank: Rank,
    labels: dict[Hashable, set[SearchState]],
    best: dict[SearchState, Rank],
    dimension: PlanSearchDimension,
) -> int:
    removed = 0
    for existing in tuple(labels.get(candidate[0], ())):
        existing_rank = best.get(existing)
        if existing_rank is None:
            labels[candidate[0]].discard(existing)
            continue
        if _state_dominates(
            candidate,
            rank,
            existing,
            existing_rank,
            dimension,
        ) and (candidate != existing or rank < existing_rank):
            labels[candidate[0]].discard(existing)
            best.pop(existing, None)
            removed += 1
    return removed


def _search_result(
    *,
    status: PlanStatus,
    state_key: SearchState,
    total_cost: float | None,
    expanded: int,
    dominated_pruned: int,
    frontier_blockers: Mapping[str, FrontierBlocker],
    resource_blockers: Mapping[tuple[str, str], ResourceBlocker],
    parents: Mapping[SearchState, tuple[SearchState, GraphEdge]],
    dimension: PlanSearchDimension,
    include_path: bool = True,
) -> PlanResult:
    path: list[GraphEdge] = []
    states: list[SearchState] = []
    if include_path:
        cursor = state_key
        states.append(cursor)
        while cursor in parents:
            cursor, edge = parents[cursor]
            path.append(edge)
            states.append(cursor)
        path.reverse()
        states.reverse()
    resource_trajectory = tuple(
        record
        for state in states
        if (record := dimension.state_record(state[3])) is not None
    )
    progression = ProgressionState(state_key[0], state_key[1], state_key[2])
    return PlanResult(
        status=status,
        path=tuple(path),
        total_cost=total_cost,
        final_progression=progression,
        expanded_count=expanded,
        dominated_pruned=dominated_pruned,
        frontier_blockers=tuple(
            frontier_blockers[key] for key in sorted(frontier_blockers)
        ),
        request_digest=dimension.request_digest,
        resource_trajectory=resource_trajectory,
        resource_blockers=tuple(
            resource_blockers[key] for key in sorted(resource_blockers)
        ),
        risk_adjusted_cost=total_cost if dimension.risk_adjusted else None,
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

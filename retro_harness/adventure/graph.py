"""Capability-aware directed route graphs.

Game-agnostic core used for room/door/overworld graphs. Edges may require or
acquire capabilities (items, events, boss flags), and route legs may acquire
new ones explicitly.
Verification status stays ``planned`` until emulator evidence promotes a
transition.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count
from math import isfinite
from typing import Hashable, Iterable, Mapping, TypeAlias

from retro_harness.adventure.progression import (
    AllOf,
    CapabilityId,
    CapabilityValue,
    ItemCheck,
    ProgressionState,
    Requirement,
    SeedPlacement,
    coerce_placement,
)

NodeId = Hashable
GraphCapability: TypeAlias = CapabilityId | str


# Shared spelling for Metroid-family and adventure inventories.
_CAPABILITY_ALIASES = {
    "missile": "missiles",
    "super_missile": "super_missiles",
    "power_bomb": "power_bombs",
    "maru_mari": "morph_ball",
    "morph": "morph_ball",
    "bomb": "bombs",
    "longbeam": "long_beam",
    "icebeam": "ice_beam",
    "wavebeam": "wave_beam",
    "varia": "varia_suit",
    "hijump": "hi_jump",
    "high_jump": "hi_jump",
    "screwattack": "screw_attack",
}


def normalize_capability(value: str) -> str:
    """Normalize a capability / ability token to snake_case.

    Accepts Super Metroid editor spellings (``missile`` → ``missiles``) and
    NES Metroid aliases (``maru_mari`` → ``morph_ball``).
    """
    normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
    return _CAPABILITY_ALIASES.get(normalized, normalized)


def _normalize_graph_capability(value: GraphCapability) -> GraphCapability:
    if isinstance(value, CapabilityId):
        return value
    if ":" in value:
        return CapabilityId.parse(value)
    return normalize_capability(value)


def _normalize_graph_capabilities(
    values: Iterable[GraphCapability],
) -> frozenset[GraphCapability]:
    return frozenset(_normalize_graph_capability(value) for value in values)


def _edge_requires(edge: GraphEdge, capabilities: frozenset[GraphCapability]) -> bool:
    if isinstance(edge.requires, Requirement):
        return edge.requires.satisfied_by(capabilities)
    return edge.requires.issubset(capabilities)


def _edge_acquires(edge: GraphEdge) -> frozenset[GraphCapability]:
    return frozenset(edge.acquires)


def _requirement_order_key(requirement: Requirement | frozenset[str]) -> str:
    if isinstance(requirement, Requirement):
        return requirement.canonical_json()
    return ",".join(sorted(requirement))


def _requirement_size(requirement: Requirement | frozenset[str]) -> int:
    if not isinstance(requirement, Requirement):
        return len(requirement)
    children = getattr(requirement, "requirements", None)
    return len(children) if children is not None else 1


def _coerce_edge_requires(
    value: Requirement
    | CapabilityId
    | str
    | Iterable[Requirement | CapabilityId | str],
) -> Requirement | frozenset[str]:
    if isinstance(value, Requirement):
        return value
    if isinstance(value, CapabilityId):
        return AllOf(value)
    if isinstance(value, str):
        if ":" in value:
            return AllOf(value)
        return frozenset({normalize_capability(value)})
    values = tuple(value)
    if not values:
        return frozenset()
    if any(
        isinstance(item, (Requirement, CapabilityId))
        or (isinstance(item, str) and ":" in item)
        for item in values
    ):
        return AllOf(values)
    return frozenset(normalize_capability(item) for item in values)  # type: ignore[arg-type]


def _coerce_edge_acquires(
    value: Iterable[GraphCapability] | GraphCapability,
) -> frozenset[GraphCapability]:
    if isinstance(value, (CapabilityId, str)):
        values = (value,)
    else:
        values = value
    return _normalize_graph_capabilities(values)


@dataclass(frozen=True)
class GraphNode:
    """One room, screen, or abstract route node."""

    node_id: NodeId
    name: str = ""
    area: str = ""
    tags: frozenset[str] = field(default_factory=frozenset)
    meta: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "nodeId": self.node_id,
            "name": self.name,
            "area": self.area,
            "tags": sorted(self.tags),
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class GraphEdge:
    """Directed transition between two nodes."""

    source_id: NodeId
    target_id: NodeId
    edge_id: str = ""
    direction: str = ""
    requires: Requirement | frozenset[str] = field(default_factory=frozenset)
    cost: float = 1.0
    verification: str = "planned"
    provenance: str = ""
    meta: Mapping[str, object] = field(default_factory=dict)
    acquires: frozenset[GraphCapability] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.edge_id:
            object.__setattr__(
                self,
                "edge_id",
                f"{self.source_id}->{self.target_id}",
            )
        object.__setattr__(self, "requires", _coerce_edge_requires(self.requires))
        object.__setattr__(
            self,
            "acquires",
            _coerce_edge_acquires(self.acquires),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "edgeId": self.edge_id,
            "sourceId": self.source_id,
            "targetId": self.target_id,
            "direction": self.direction,
            "requires": (
                self.requires.to_dict()
                if isinstance(self.requires, Requirement)
                else sorted(self.requires)
            ),
            "acquires": sorted(str(value) for value in self.acquires),
            "cost": self.cost,
            "verification": self.verification,
            "provenance": self.provenance,
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class RoutePatch:
    """Explicit directed edge to merge into a base graph."""

    source_id: NodeId
    target_id: NodeId
    direction: str = ""
    requires: Requirement | frozenset[str] = field(default_factory=frozenset)
    support: str = ""
    meta: Mapping[str, object] = field(default_factory=dict)
    acquires: frozenset[GraphCapability] = field(default_factory=frozenset)

    def as_edge(self) -> GraphEdge:
        edge_meta: dict[str, object] = dict(self.meta)
        if self.support:
            edge_meta.setdefault("support", self.support)
        return GraphEdge(
            source_id=self.source_id,
            target_id=self.target_id,
            direction=self.direction,
            requires=self.requires,
            acquires=self.acquires,
            verification="planned",
            provenance="explicit_route_patch",
            meta=edge_meta,
        )


@dataclass(frozen=True)
class RouteLeg:
    """One planned hop on a route (may acquire capabilities on clear)."""

    leg_id: str
    source_id: NodeId
    target_id: NodeId
    requires: Requirement | frozenset[str] = field(default_factory=frozenset)
    acquires: frozenset[GraphCapability] = field(default_factory=frozenset)
    goal: str = ""
    constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlannedLeg:
    """Resolved leg with capability bookkeeping."""

    leg: RouteLeg
    edge: GraphEdge
    capabilities_before: frozenset[GraphCapability]
    effective_requires: Requirement | frozenset[str]
    capabilities_after: frozenset[GraphCapability]

    def to_dict(self, nodes: Mapping[NodeId, GraphNode]) -> dict[str, object]:
        source = nodes.get(self.leg.source_id)
        target = nodes.get(self.leg.target_id)
        acquires = _edge_acquires(self.edge) | _normalize_graph_capabilities(
            self.leg.acquires
        )
        return {
            "legId": self.leg.leg_id,
            "source": source.to_dict() if source else {"nodeId": self.leg.source_id},
            "target": target.to_dict() if target else {"nodeId": self.leg.target_id},
            "edge": self.edge.to_dict(),
            "capabilitiesBefore": sorted(str(value) for value in self.capabilities_before),
            "effectiveRequires": (
                self.effective_requires.to_dict()
                if isinstance(self.effective_requires, Requirement)
                else sorted(self.effective_requires)
            ),
            "acquires": sorted(str(value) for value in acquires),
            "capabilitiesAfter": sorted(str(value) for value in self.capabilities_after),
            "goal": self.leg.goal,
            "constraints": list(self.leg.constraints),
            "status": "planned_not_continuous",
        }


def shortest_path(
    edges: Iterable[GraphEdge],
    source_id: NodeId,
    target_id: NodeId,
    capabilities: frozenset[GraphCapability] | Iterable[GraphCapability] = frozenset(),
) -> tuple[GraphEdge, ...] | None:
    """BFS shortest path (edge count) respecting capability gates."""
    caps = _normalize_graph_capabilities(capabilities)
    if source_id == target_id:
        return ()
    outgoing: dict[NodeId, list[GraphEdge]] = defaultdict(list)
    for edge in edges:
        outgoing[edge.source_id].append(edge)
    queue: deque[NodeId] = deque([source_id])
    seen = {source_id}
    parent: dict[NodeId, tuple[NodeId, GraphEdge]] = {}
    while queue:
        node = queue.popleft()
        for edge in outgoing.get(node, ()):
            if not _edge_requires(edge, caps):
                continue
            if edge.target_id in seen:
                continue
            seen.add(edge.target_id)
            parent[edge.target_id] = (node, edge)
            if edge.target_id == target_id:
                path: list[GraphEdge] = []
                cursor = target_id
                while cursor != source_id:
                    previous, used = parent[cursor]
                    path.append(used)
                    cursor = previous
                return tuple(reversed(path))
            queue.append(edge.target_id)
    return None


def _edge_order_key(edge: GraphEdge) -> tuple[str, ...]:
    """Stable ordering key for deterministic capability search tie breaks."""
    return (
        type(edge.target_id).__qualname__,
        repr(edge.target_id),
        edge.edge_id,
        edge.direction,
        _requirement_order_key(edge.requires),
        ",".join(sorted(str(value) for value in edge.acquires)),
        repr(edge.cost),
    )


def inventory_aware_path(
    edges: Iterable[GraphEdge],
    source_id: NodeId,
    target_id: NodeId,
    capabilities: frozenset[GraphCapability] | Iterable[GraphCapability] = frozenset(),
) -> tuple[GraphEdge, ...] | None:
    """Find a least-cost path while capabilities grow on traversed edges.

    Search state is ``(node_id, capabilities)``.  An edge is traversable when
    its ``requires`` are in the current inventory, and its ``acquires`` are
    added after the transition.  Costs must be non-negative.  Equal-cost
    paths use stable edge metadata as a tie break so a graph's result does not
    depend on input edge order.

    This is intentionally monotonic item logic: capabilities are set-valued
    items, events, or defeated-boss flags and are never consumed.  Resource
    counts and game-specific stop predicates belong above this shared layer.
    """
    graph_edges = tuple(edges)
    for edge in graph_edges:
        if not isfinite(edge.cost) or edge.cost < 0:
            raise ValueError(
                "inventory-aware path requires finite, non-negative edge costs"
            )

    if source_id == target_id:
        return ()

    normalized = _normalize_graph_capabilities(capabilities)
    outgoing: dict[NodeId, list[GraphEdge]] = defaultdict(list)
    for edge in graph_edges:
        outgoing[edge.source_id].append(edge)
    for candidates in outgoing.values():
        candidates.sort(key=_edge_order_key)

    State = tuple[NodeId, frozenset[GraphCapability]]
    PathKey = tuple[tuple[str, ...], ...]
    initial_state: State = (source_id, normalized)
    initial_rank: tuple[float, PathKey] = (0.0, ())
    best: dict[State, tuple[float, PathKey]] = {initial_state: initial_rank}
    parent: dict[State, tuple[State, GraphEdge]] = {}
    sequence = count()
    pending: list[tuple[float, PathKey, int, NodeId, frozenset[GraphCapability]]] = [
        (0.0, (), next(sequence), source_id, normalized)
    ]

    while pending:
        cost, path_key, _sequence, node_id, current_caps = heappop(pending)
        state = (node_id, current_caps)
        if best.get(state) != (cost, path_key):
            continue
        if node_id == target_id:
            path: list[GraphEdge] = []
            cursor = state
            while cursor in parent:
                previous, edge = parent[cursor]
                path.append(edge)
                cursor = previous
            return tuple(reversed(path))

        for edge in outgoing.get(node_id, ()):
            if not _edge_requires(edge, current_caps):
                continue
            next_caps = current_caps | _edge_acquires(edge)
            next_state: State = (edge.target_id, next_caps)
            next_cost = cost + edge.cost
            next_path_key = path_key + (_edge_order_key(edge),)
            next_rank = (next_cost, next_path_key)
            if next_state in best and next_rank >= best[next_state]:
                continue
            best[next_state] = next_rank
            parent[next_state] = (state, edge)
            heappush(
                pending,
                (
                    next_cost,
                    next_path_key,
                    next(sequence),
                    edge.target_id,
                    next_caps,
                ),
            )
    return None


def _collect_node_checks(
    state: ProgressionState,
    checks_by_node: Mapping[NodeId, tuple[ItemCheck, ...]],
    placement: SeedPlacement,
) -> ProgressionState:
    """Collect every currently available check at a node in stable order."""
    current = state
    checks = checks_by_node.get(current.node, ())
    changed = True
    while changed:
        changed = False
        for check in checks:
            if check.can_collect(current):
                current = current.collect(check, placement)
                changed = True
    return current


def progression_plan(
    edges: Iterable[GraphEdge],
    checks: Iterable[ItemCheck],
    source_id: NodeId,
    target_id: NodeId,
    placements: SeedPlacement
    | Mapping[str, CapabilityValue]
    | Iterable[SeedPlacement]
    | Iterable[tuple[str, CapabilityValue]]
    | None = None,
    *,
    capabilities: Iterable[GraphCapability] = frozenset(),
    collected_checks: Iterable[str] = frozenset(),
) -> tuple[tuple[GraphEdge, ...], ProgressionState] | None:
    """Plan over monotonic capabilities and item-check collection state.

    A check at the current node is collected before outgoing edges are
    expanded.  The returned state is post-collection at the target.  This is
    intentionally a small deterministic search surface; bounded planner
    budgets and richer result reporting belong to a later layer.
    """
    graph_edges = tuple(edges)
    graph_checks = tuple(checks)
    placement = coerce_placement(placements)
    checks_by_node: dict[NodeId, tuple[ItemCheck, ...]] = defaultdict(tuple)
    grouped: dict[NodeId, list[ItemCheck]] = defaultdict(list)
    seen_check_ids: set[str] = set()
    for check in graph_checks:
        if check.check_id in seen_check_ids:
            raise ValueError(f"duplicate item check ID: {check.check_id!r}")
        seen_check_ids.add(check.check_id)
        grouped[check.node_id].append(check)
    for node_id, node_checks in grouped.items():
        checks_by_node[node_id] = tuple(
            sorted(node_checks, key=lambda check: check.check_id)
        )
    for edge in graph_edges:
        if not isfinite(edge.cost) or edge.cost < 0:
            raise ValueError(
                "progression plan requires finite, non-negative edge costs"
            )

    initial = ProgressionState(
        source_id,
        _normalize_graph_capabilities(capabilities),
        collected_checks,
    )
    initial = _collect_node_checks(initial, checks_by_node, placement)
    if source_id == target_id:
        return (), initial

    outgoing: dict[NodeId, list[GraphEdge]] = defaultdict(list)
    for edge in graph_edges:
        outgoing[edge.source_id].append(edge)
    for candidates in outgoing.values():
        candidates.sort(key=_edge_order_key)

    State = tuple[NodeId, frozenset[GraphCapability], frozenset[str]]
    PathKey = tuple[tuple[str, ...], ...]
    initial_state: State = (
        initial.node,
        initial.capabilities,
        initial.collected_checks,
    )
    best: dict[State, tuple[float, PathKey]] = {initial_state: (0.0, ())}
    parent: dict[State, tuple[State, GraphEdge]] = {}
    sequence = count()
    pending: list[
        tuple[float, PathKey, int, NodeId, frozenset[GraphCapability], frozenset[str]]
    ] = [(0.0, (), next(sequence), *initial_state)]

    while pending:
        cost, path_key, _sequence, node_id, current_caps, current_checks = heappop(
            pending
        )
        state_key: State = (node_id, current_caps, current_checks)
        if best.get(state_key) != (cost, path_key):
            continue
        if node_id == target_id:
            path: list[GraphEdge] = []
            cursor = state_key
            while cursor in parent:
                previous, edge = parent[cursor]
                path.append(edge)
                cursor = previous
            final_state = ProgressionState(
                node_id,
                current_caps,
                current_checks,
            )
            return tuple(reversed(path)), final_state

        for edge in outgoing.get(node_id, ()):
            if not _edge_requires(edge, current_caps):
                continue
            next_state = ProgressionState(
                edge.target_id,
                current_caps | _edge_acquires(edge),
                current_checks,
            )
            next_state = _collect_node_checks(next_state, checks_by_node, placement)
            next_key: State = (
                next_state.node,
                next_state.capabilities,
                next_state.collected_checks,
            )
            next_cost = cost + edge.cost
            next_path_key = path_key + (_edge_order_key(edge),)
            next_rank = (next_cost, next_path_key)
            if next_key in best and next_rank >= best[next_key]:
                continue
            best[next_key] = next_rank
            parent[next_key] = (state_key, edge)
            heappush(
                pending,
                (
                    next_cost,
                    next_path_key,
                    next(sequence),
                    *next_key,
                ),
            )
    return None


def progression_path(
    edges: Iterable[GraphEdge],
    checks: Iterable[ItemCheck],
    source_id: NodeId,
    target_id: NodeId,
    placements: SeedPlacement
    | Mapping[str, CapabilityValue]
    | Iterable[SeedPlacement]
    | Iterable[tuple[str, CapabilityValue]]
    | None = None,
    *,
    capabilities: Iterable[GraphCapability] = frozenset(),
    collected_checks: Iterable[str] = frozenset(),
) -> tuple[GraphEdge, ...] | None:
    """Return only the edge sequence from :func:`progression_plan`."""
    result = progression_plan(
        edges,
        checks,
        source_id,
        target_id,
        placements,
        capabilities=capabilities,
        collected_checks=collected_checks,
    )
    return None if result is None else result[0]


class RouteGraph:
    """Validated node/edge set with pathfinding and leg planning."""

    def __init__(
        self,
        nodes: Iterable[GraphNode],
        edges: Iterable[GraphEdge],
        checks: Iterable[ItemCheck] = (),
    ) -> None:
        self.nodes = {node.node_id: node for node in nodes}
        self.edges = tuple(edges)
        self.checks = tuple(checks)
        self.item_checks = self.checks
        self._outgoing: dict[NodeId, list[GraphEdge]] = defaultdict(list)
        self._by_pair: dict[tuple[NodeId, NodeId], list[GraphEdge]] = defaultdict(list)
        check_ids: set[str] = set()
        for check in self.checks:
            if check.check_id in check_ids:
                raise ValueError(f"duplicate item check ID: {check.check_id!r}")
            if check.node_id not in self.nodes:
                raise ValueError(
                    f"item check node {check.node_id!r} is not a node"
                )
            check_ids.add(check.check_id)
        for edge in self.edges:
            if edge.source_id not in self.nodes:
                raise ValueError(f"edge source {edge.source_id!r} is not a node")
            if edge.target_id not in self.nodes:
                raise ValueError(f"edge target {edge.target_id!r} is not a node")
            pair = (edge.source_id, edge.target_id)
            self._by_pair[pair].append(edge)
            self._outgoing[edge.source_id].append(edge)

    def edge_for(
        self,
        source_id: NodeId,
        target_id: NodeId,
    ) -> GraphEdge | None:
        candidates = self._by_pair.get((source_id, target_id), ())
        return min(
            candidates,
            key=lambda edge: (
                _requirement_size(edge.requires),
                _requirement_order_key(edge.requires),
                edge.direction,
                edge.edge_id,
            ),
            default=None,
        )

    def add_patches(self, patches: Iterable[RoutePatch]) -> RouteGraph:
        added: list[GraphEdge] = []
        for patch in patches:
            pair = (patch.source_id, patch.target_id)
            if pair in self._by_pair:
                raise ValueError(
                    f"route patch would hide an existing edge: {pair[0]!r}->{pair[1]!r}"
                )
            added.append(patch.as_edge())
        return RouteGraph(self.nodes.values(), (*self.edges, *added), self.checks)

    def shortest_path(
        self,
        source_id: NodeId,
        target_id: NodeId,
        capabilities: frozenset[GraphCapability] | Iterable[GraphCapability] = frozenset(),
    ) -> tuple[GraphEdge, ...] | None:
        return shortest_path(
            self.edges,
            source_id,
            target_id,
            capabilities=capabilities,
        )

    def inventory_aware_path(
        self,
        source_id: NodeId,
        target_id: NodeId,
        capabilities: frozenset[GraphCapability] | Iterable[GraphCapability] = frozenset(),
    ) -> tuple[GraphEdge, ...] | None:
        """Plan a least-cost path while collecting edge ``acquires`` items."""
        return inventory_aware_path(
            self.edges,
            source_id,
            target_id,
            capabilities=capabilities,
        )

    def progression_plan(
        self,
        source_id: NodeId,
        target_id: NodeId,
        placements: SeedPlacement
        | Mapping[str, CapabilityValue]
        | Iterable[SeedPlacement]
        | Iterable[tuple[str, CapabilityValue]]
        | None = None,
        *,
        capabilities: Iterable[GraphCapability] = frozenset(),
        collected_checks: Iterable[str] = frozenset(),
    ) -> tuple[tuple[GraphEdge, ...], ProgressionState] | None:
        """Plan while collecting this graph's item checks from an overlay."""
        return progression_plan(
            self.edges,
            self.checks,
            source_id,
            target_id,
            placements,
            capabilities=capabilities,
            collected_checks=collected_checks,
        )

    def progression_path(
        self,
        source_id: NodeId,
        target_id: NodeId,
        placements: SeedPlacement
        | Mapping[str, CapabilityValue]
        | Iterable[SeedPlacement]
        | Iterable[tuple[str, CapabilityValue]]
        | None = None,
        *,
        capabilities: Iterable[GraphCapability] = frozenset(),
        collected_checks: Iterable[str] = frozenset(),
    ) -> tuple[GraphEdge, ...] | None:
        """Return only the edge sequence for a placement-aware plan."""
        result = self.progression_plan(
            source_id,
            target_id,
            placements,
            capabilities=capabilities,
            collected_checks=collected_checks,
        )
        return None if result is None else result[0]

    plan_with_placement = progression_path
    plan_progression = progression_plan

    def plan_legs(
        self,
        legs: Iterable[RouteLeg],
        *,
        initial_capabilities: frozenset[GraphCapability] | Iterable[GraphCapability] = frozenset(),
    ) -> tuple[PlannedLeg, ...]:
        capabilities = _normalize_graph_capabilities(initial_capabilities)
        planned: list[PlannedLeg] = []
        previous_target: NodeId | None = None
        for leg in legs:
            if previous_target is not None and leg.source_id != previous_target:
                raise ValueError(
                    f"route leg {leg.leg_id} is not contiguous with its predecessor"
                )
            edge = self.edge_for(leg.source_id, leg.target_id)
            if edge is None:
                raise ValueError(
                    f"route leg {leg.leg_id} has no edge {leg.source_id!r}->{leg.target_id!r}"
                )
            if isinstance(leg.requires, Requirement):
                effective: Requirement | frozenset[str] = AllOf(
                    edge.requires,
                    leg.requires,
                ) if isinstance(edge.requires, Requirement) else leg.requires
                satisfied = _edge_requires(edge, capabilities) and leg.requires.satisfied_by(
                    capabilities
                )
                missing: frozenset[str] = frozenset()
            else:
                explicit = frozenset(normalize_capability(v) for v in leg.requires)
                if isinstance(edge.requires, Requirement):
                    effective = edge.requires
                    satisfied = edge.requires.satisfied_by(capabilities) and not explicit - capabilities
                else:
                    effective = edge.requires | explicit
                    missing = effective - capabilities
                    satisfied = not missing
            if not satisfied:
                if isinstance(effective, Requirement):
                    missing_text = effective.canonical_json()
                else:
                    missing_text = ", ".join(sorted(missing))
                raise ValueError(
                    f"route leg {leg.leg_id} is missing capabilities: "
                    f"{missing_text}"
                )
            after = capabilities | _edge_acquires(edge) | _normalize_graph_capabilities(
                leg.acquires
            )
            planned.append(
                PlannedLeg(
                    leg=leg,
                    edge=edge,
                    capabilities_before=capabilities,
                    effective_requires=effective,
                    capabilities_after=after,
                )
            )
            capabilities = after
            previous_target = leg.target_id
        return tuple(planned)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
        }
        if self.checks:
            result["checks"] = [check.to_dict() for check in self.checks]
        return result


@dataclass(frozen=True)
class ProgressionMilestone:
    """Inventory / event gate on a route (game-agnostic bookkeeping).

    Stop predicates stay game-local; this only tracks capability requirements
    and acquisitions so NES Metroid and Super Metroid share the same schema.
    """

    milestone_id: str
    label: str
    node_id: NodeId | None = None
    requires: frozenset[str] = field(default_factory=frozenset)
    acquires: frozenset[str] = field(default_factory=frozenset)
    timeout_frames: int = 0
    policy_id: str = ""
    goal: str = ""
    meta: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.requires:
            object.__setattr__(
                self,
                "requires",
                frozenset(normalize_capability(v) for v in self.requires),
            )
        if self.acquires:
            object.__setattr__(
                self,
                "acquires",
                frozenset(normalize_capability(v) for v in self.acquires),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "milestoneId": self.milestone_id,
            "label": self.label,
            "nodeId": self.node_id,
            "requires": sorted(self.requires),
            "acquires": sorted(self.acquires),
            "timeoutFrames": self.timeout_frames,
            "policyId": self.policy_id,
            "goal": self.goal,
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class ObservedTransition:
    """Live room/screen transition observed during an emulator run."""

    frame: int
    source_id: NodeId
    target_id: NodeId
    edge_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "frame": self.frame,
            "sourceId": self.source_id,
            "targetId": self.target_id,
            "edgeId": self.edge_id,
        }


def apply_milestones(
    milestones: Iterable[ProgressionMilestone],
    *,
    initial_capabilities: frozenset[str] | Iterable[str] = frozenset(),
) -> tuple[frozenset[str], tuple[ProgressionMilestone, ...]]:
    """Validate milestone order and return final capability set.

    Raises ``ValueError`` if a milestone is missing required capabilities.
    """
    capabilities = frozenset(normalize_capability(v) for v in initial_capabilities)
    ordered: list[ProgressionMilestone] = []
    for milestone in milestones:
        missing = milestone.requires - capabilities
        if missing:
            raise ValueError(
                f"milestone {milestone.milestone_id} is missing capabilities: "
                f"{', '.join(sorted(missing))}"
            )
        capabilities = capabilities | milestone.acquires
        ordered.append(milestone)
    return capabilities, tuple(ordered)


def promote_edge_verification(
    graph: RouteGraph,
    source_id: NodeId,
    target_id: NodeId,
    *,
    verification: str = "continuous",
) -> RouteGraph:
    """Return a copy of ``graph`` with one edge's verification upgraded."""
    new_edges: list[GraphEdge] = []
    found = False
    for edge in graph.edges:
        if edge.source_id == source_id and edge.target_id == target_id and not found:
            new_edges.append(
                GraphEdge(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    edge_id=edge.edge_id,
                    direction=edge.direction,
                    requires=edge.requires,
                    cost=edge.cost,
                    verification=verification,
                    provenance=edge.provenance,
                    meta=edge.meta,
                    acquires=edge.acquires,
                )
            )
            found = True
        else:
            new_edges.append(edge)
    if not found:
        raise ValueError(f"no edge {source_id!r}->{target_id!r} to promote")
    return RouteGraph(graph.nodes.values(), new_edges, graph.checks)

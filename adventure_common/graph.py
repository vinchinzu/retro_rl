"""Capability-aware directed route graphs.

Game-agnostic core used for room/door/overworld graphs. Edges may require
capabilities (items, events, boss flags) and route legs may acquire new ones.
Verification status stays ``planned`` until emulator evidence promotes a
transition.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Hashable, Iterable, Mapping

NodeId = Hashable


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
    requires: frozenset[str] = field(default_factory=frozenset)
    cost: float = 1.0
    verification: str = "planned"
    provenance: str = ""
    meta: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.edge_id:
            object.__setattr__(
                self,
                "edge_id",
                f"{self.source_id}->{self.target_id}",
            )
        if self.requires:
            object.__setattr__(
                self,
                "requires",
                frozenset(normalize_capability(v) for v in self.requires),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "edgeId": self.edge_id,
            "sourceId": self.source_id,
            "targetId": self.target_id,
            "direction": self.direction,
            "requires": sorted(self.requires),
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
    requires: frozenset[str] = field(default_factory=frozenset)
    support: str = ""
    meta: Mapping[str, object] = field(default_factory=dict)

    def as_edge(self) -> GraphEdge:
        edge_meta: dict[str, object] = dict(self.meta)
        if self.support:
            edge_meta.setdefault("support", self.support)
        return GraphEdge(
            source_id=self.source_id,
            target_id=self.target_id,
            direction=self.direction,
            requires=frozenset(normalize_capability(v) for v in self.requires),
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
    requires: frozenset[str] = field(default_factory=frozenset)
    acquires: frozenset[str] = field(default_factory=frozenset)
    goal: str = ""
    constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlannedLeg:
    """Resolved leg with capability bookkeeping."""

    leg: RouteLeg
    edge: GraphEdge
    capabilities_before: frozenset[str]
    effective_requires: frozenset[str]
    capabilities_after: frozenset[str]

    def to_dict(self, nodes: Mapping[NodeId, GraphNode]) -> dict[str, object]:
        source = nodes.get(self.leg.source_id)
        target = nodes.get(self.leg.target_id)
        return {
            "legId": self.leg.leg_id,
            "source": source.to_dict() if source else {"nodeId": self.leg.source_id},
            "target": target.to_dict() if target else {"nodeId": self.leg.target_id},
            "edge": self.edge.to_dict(),
            "capabilitiesBefore": sorted(self.capabilities_before),
            "effectiveRequires": sorted(self.effective_requires),
            "acquires": sorted(self.leg.acquires),
            "capabilitiesAfter": sorted(self.capabilities_after),
            "goal": self.leg.goal,
            "constraints": list(self.leg.constraints),
            "status": "planned_not_continuous",
        }


def shortest_path(
    edges: Iterable[GraphEdge],
    source_id: NodeId,
    target_id: NodeId,
    capabilities: frozenset[str] | Iterable[str] = frozenset(),
) -> tuple[GraphEdge, ...] | None:
    """BFS shortest path (edge count) respecting capability gates."""
    caps = frozenset(normalize_capability(v) for v in capabilities)
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
            if not edge.requires.issubset(caps):
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


class RouteGraph:
    """Validated node/edge set with pathfinding and leg planning."""

    def __init__(
        self,
        nodes: Iterable[GraphNode],
        edges: Iterable[GraphEdge],
    ) -> None:
        self.nodes = {node.node_id: node for node in nodes}
        self.edges = tuple(edges)
        self._outgoing: dict[NodeId, list[GraphEdge]] = defaultdict(list)
        self._by_pair: dict[tuple[NodeId, NodeId], list[GraphEdge]] = defaultdict(list)
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
                len(edge.requires),
                sorted(edge.requires),
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
        return RouteGraph(self.nodes.values(), (*self.edges, *added))

    def shortest_path(
        self,
        source_id: NodeId,
        target_id: NodeId,
        capabilities: frozenset[str] | Iterable[str] = frozenset(),
    ) -> tuple[GraphEdge, ...] | None:
        return shortest_path(
            self.edges,
            source_id,
            target_id,
            capabilities=capabilities,
        )

    def plan_legs(
        self,
        legs: Iterable[RouteLeg],
        *,
        initial_capabilities: frozenset[str] | Iterable[str] = frozenset(),
    ) -> tuple[PlannedLeg, ...]:
        capabilities = frozenset(
            normalize_capability(value) for value in initial_capabilities
        )
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
            explicit = frozenset(normalize_capability(v) for v in leg.requires)
            effective = edge.requires | explicit
            missing = effective - capabilities
            if missing:
                raise ValueError(
                    f"route leg {leg.leg_id} is missing capabilities: "
                    f"{', '.join(sorted(missing))}"
                )
            after = capabilities | frozenset(
                normalize_capability(v) for v in leg.acquires
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
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
        }


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
                )
            )
            found = True
        else:
            new_edges.append(edge)
    if not found:
        raise ValueError(f"no edge {source_id!r}->{target_id!r} to promote")
    return RouteGraph(graph.nodes.values(), new_edges)

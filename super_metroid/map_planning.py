"""Ability-aware route planning over Super Metroid editor exports.

The editor graph is a research input, not acceptance evidence.  Edges loaded
from it and explicit patches therefore remain ``planned`` until a continuous
emulator run observes the corresponding room transition.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping


_ABILITY_ALIASES = {
    "missile": "missiles",
    "super_missile": "super_missiles",
    "power_bomb": "power_bombs",
}


def normalize_ability(value: str) -> str:
    """Return the capability spelling used by the runtime route graph."""
    normalized = value.strip().lower().replace(" ", "_")
    return _ABILITY_ALIASES.get(normalized, normalized)


def _room_id(value: int | str) -> int:
    if isinstance(value, int):
        return value
    return int(value, 0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class EditorRoom:
    room_id: int
    name: str
    area: str
    handle: str
    map_x: int
    map_y: int
    width_screens: int
    height_screens: int

    def to_dict(self) -> dict[str, object]:
        return {
            "roomId": self.room_id,
            "roomIdHex": f"0x{self.room_id:04X}",
            "name": self.name,
            "area": self.area,
            "handle": self.handle,
            "mapX": self.map_x,
            "mapY": self.map_y,
            "widthScreens": self.width_screens,
            "heightScreens": self.height_screens,
        }


@dataclass(frozen=True)
class EditorEdge:
    source_room_id: int
    target_room_id: int
    direction: str
    is_elevator: bool = False
    door_cap_color: str | None = None
    requires: frozenset[str] = field(default_factory=frozenset)
    provenance: str = "super_metroid_editor"
    provenance_detail: str = ""
    verification: str = "planned"

    @property
    def edge_id(self) -> str:
        return (
            f"editor_{self.source_room_id:04x}_to_"
            f"{self.target_room_id:04x}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "edgeId": self.edge_id,
            "sourceRoomId": self.source_room_id,
            "sourceRoomIdHex": f"0x{self.source_room_id:04X}",
            "targetRoomId": self.target_room_id,
            "targetRoomIdHex": f"0x{self.target_room_id:04X}",
            "direction": self.direction,
            "isElevator": self.is_elevator,
            "doorCapColor": self.door_cap_color,
            "requires": sorted(self.requires),
            "provenance": self.provenance,
            "provenanceDetail": self.provenance_detail,
            "verification": self.verification,
        }


@dataclass(frozen=True)
class RoutePatch:
    source_room_id: int
    target_room_id: int
    direction: str
    requires: frozenset[str] = field(default_factory=frozenset)
    door_cap_color: str | None = None
    support: str = ""

    def as_edge(self) -> EditorEdge:
        return EditorEdge(
            source_room_id=self.source_room_id,
            target_room_id=self.target_room_id,
            direction=self.direction,
            door_cap_color=self.door_cap_color,
            requires=frozenset(normalize_ability(value) for value in self.requires),
            provenance="explicit_route_patch",
            provenance_detail=self.support,
            verification="planned",
        )


@dataclass(frozen=True)
class RouteLeg:
    leg_id: str
    source_room_id: int
    target_room_id: int
    requires: frozenset[str] = field(default_factory=frozenset)
    acquires: frozenset[str] = field(default_factory=frozenset)
    goal: str = ""
    constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlannedLeg:
    leg: RouteLeg
    edge: EditorEdge
    capabilities_before: frozenset[str]
    effective_requires: frozenset[str]
    capabilities_after: frozenset[str]

    def to_dict(
        self,
        rooms: Mapping[int, EditorRoom],
    ) -> dict[str, object]:
        source = rooms[self.leg.source_room_id]
        target = rooms[self.leg.target_room_id]
        return {
            "legId": self.leg.leg_id,
            "source": source.to_dict(),
            "target": target.to_dict(),
            "edge": self.edge.to_dict(),
            "capabilitiesBefore": sorted(self.capabilities_before),
            "effectiveRequires": sorted(self.effective_requires),
            "acquires": sorted(self.leg.acquires),
            "capabilitiesAfter": sorted(self.capabilities_after),
            "goal": self.leg.goal,
            "constraints": list(self.leg.constraints),
            "status": "planned_not_continuous",
        }


class EditorNavigationGraph:
    """A validated editor export with optional, explicit directed patches."""

    def __init__(
        self,
        rooms: Iterable[EditorRoom],
        edges: Iterable[EditorEdge],
        *,
        source_path: Path,
        source_sha256: str,
    ) -> None:
        self.rooms = {room.room_id: room for room in rooms}
        self.edges = tuple(edges)
        self.source_path = source_path
        self.source_sha256 = source_sha256
        self._outgoing: dict[int, list[EditorEdge]] = defaultdict(list)
        self._by_pair: dict[tuple[int, int], list[EditorEdge]] = defaultdict(list)
        for edge in self.edges:
            if edge.source_room_id not in self.rooms:
                raise ValueError(
                    f"edge source 0x{edge.source_room_id:04X} is not a node"
                )
            if edge.target_room_id not in self.rooms:
                raise ValueError(
                    f"edge target 0x{edge.target_room_id:04X} is not a node"
                )
            pair = (edge.source_room_id, edge.target_room_id)
            # A room pair can have more than one physical door.  Retain every
            # exported edge for pathfinding while exposing a stable preferred
            # edge for room-sequence plans.
            self._by_pair[pair].append(edge)
            self._outgoing[edge.source_room_id].append(edge)

    @classmethod
    def load(cls, path: str | Path) -> EditorNavigationGraph:
        source = Path(path).expanduser().resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        nodes = payload.get("nodes")
        edges = payload.get("edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise ValueError("editor nav graph must contain nodes and edges lists")

        rooms = (
            EditorRoom(
                room_id=_room_id(node["roomId"]),
                name=str(node["name"]),
                area=str(node["areaName"]),
                handle=str(node["handle"]),
                map_x=int(node["mapX"]),
                map_y=int(node["mapY"]),
                width_screens=int(node["widthScreens"]),
                height_screens=int(node["heightScreens"]),
            )
            for node in nodes
        )
        parsed_edges = (
            EditorEdge(
                source_room_id=_room_id(edge["fromRoomId"]),
                target_room_id=_room_id(edge["toRoomId"]),
                direction=str(edge["direction"]),
                is_elevator=bool(edge["isElevator"]),
                door_cap_color=edge.get("doorCapColor"),
                requires=(
                    frozenset(
                        {normalize_ability(str(edge["requiredAbility"]))}
                    )
                    if edge.get("requiredAbility")
                    else frozenset()
                ),
            )
            for edge in edges
        )
        return cls(
            rooms,
            parsed_edges,
            source_path=source,
            source_sha256=sha256_file(source),
        )

    def edge_for(
        self,
        source_room_id: int,
        target_room_id: int,
    ) -> EditorEdge | None:
        candidates = self._by_pair.get((source_room_id, target_room_id), ())
        return min(
            candidates,
            key=lambda edge: (
                len(edge.requires),
                sorted(edge.requires),
                edge.direction,
            ),
            default=None,
        )

    def add_patches(
        self,
        patches: Iterable[RoutePatch],
    ) -> EditorNavigationGraph:
        added: list[EditorEdge] = []
        for patch in patches:
            pair = (patch.source_room_id, patch.target_room_id)
            if pair in self._by_pair:
                raise ValueError(
                    "route patch would hide an editor edge: "
                    f"0x{pair[0]:04X}->0x{pair[1]:04X}"
                )
            added.append(patch.as_edge())
        return EditorNavigationGraph(
            self.rooms.values(),
            (*self.edges, *added),
            source_path=self.source_path,
            source_sha256=self.source_sha256,
        )

    def shortest_path(
        self,
        source_room_id: int,
        target_room_id: int,
        capabilities: frozenset[str] = frozenset(),
    ) -> tuple[EditorEdge, ...] | None:
        normalized = frozenset(normalize_ability(value) for value in capabilities)
        if source_room_id == target_room_id:
            return ()
        queue: deque[int] = deque([source_room_id])
        seen = {source_room_id}
        parent: dict[int, tuple[int, EditorEdge]] = {}
        while queue:
            room_id = queue.popleft()
            for edge in self._outgoing.get(room_id, ()):
                if not edge.requires.issubset(normalized):
                    continue
                if edge.target_room_id in seen:
                    continue
                seen.add(edge.target_room_id)
                parent[edge.target_room_id] = (room_id, edge)
                if edge.target_room_id == target_room_id:
                    path: list[EditorEdge] = []
                    cursor = target_room_id
                    while cursor != source_room_id:
                        previous, used = parent[cursor]
                        path.append(used)
                        cursor = previous
                    return tuple(reversed(path))
                queue.append(edge.target_room_id)
        return None

    def plan_legs(
        self,
        legs: Iterable[RouteLeg],
        *,
        initial_capabilities: frozenset[str],
    ) -> tuple[PlannedLeg, ...]:
        capabilities = frozenset(
            normalize_ability(value) for value in initial_capabilities
        )
        planned: list[PlannedLeg] = []
        previous_target: int | None = None
        for leg in legs:
            if (
                previous_target is not None
                and leg.source_room_id != previous_target
            ):
                raise ValueError(
                    f"route leg {leg.leg_id} is not contiguous with its predecessor"
                )
            edge = self.edge_for(leg.source_room_id, leg.target_room_id)
            if edge is None:
                raise ValueError(
                    f"route leg {leg.leg_id} has no editor edge or route patch"
                )
            explicit_requires = frozenset(
                normalize_ability(value) for value in leg.requires
            )
            effective_requires = edge.requires | explicit_requires
            missing = effective_requires - capabilities
            if missing:
                raise ValueError(
                    f"route leg {leg.leg_id} is missing capabilities: "
                    f"{', '.join(sorted(missing))}"
                )
            after = capabilities | frozenset(
                normalize_ability(value) for value in leg.acquires
            )
            planned.append(
                PlannedLeg(
                    leg=leg,
                    edge=edge,
                    capabilities_before=capabilities,
                    effective_requires=effective_requires,
                    capabilities_after=after,
                )
            )
            capabilities = after
            previous_target = leg.target_room_id
        return tuple(planned)

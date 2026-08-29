"""Ability-aware route planning over Super Metroid editor exports.

The editor graph is a research input, not acceptance evidence. Edges loaded
from it and explicit patches therefore remain ``planned`` until a continuous
emulator run observes the corresponding room transition.

Graph operations (:meth:`edge_for`, :meth:`add_patches`, :meth:`shortest_path`,
:meth:`plan_legs`) are delegated to :class:`retro_harness.adventure.RouteGraph`.
This module only owns editor JSON load/parse and SM-shaped serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping

from retro_harness.adventure.graph import (
    GraphEdge,
    GraphNode,
    PlannedLeg,
    RouteGraph,
    RouteLeg,
    RoutePatch,
    normalize_capability,
)
from retro_harness.adventure.hashutil import sha256_file

# Re-exports for historical imports.
normalize_ability = normalize_capability

__all__ = [
    "EditorNavigationGraph",
    "EditorRoom",
    "PlannedLeg",
    "POST_TORIZO_CAPABILITIES",
    "POST_TORIZO_ROUTE_PATCHES",
    "POST_TORIZO_TO_SPORE_SPAWN",
    "RouteLeg",
    "RoutePatch",
    "normalize_ability",
    "sha256_file",
]


def _room_id(value: int | str) -> int:
    if isinstance(value, int):
        return value
    return int(value, 0)


@dataclass(frozen=True)
class EditorRoom:
    """SM editor room payload (geometry + labels)."""

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

    def as_graph_node(self) -> GraphNode:
        return GraphNode(
            node_id=self.room_id,
            name=self.name,
            area=self.area,
            tags=frozenset({"editor_room"}),
            meta={
                "handle": self.handle,
                "mapX": self.map_x,
                "mapY": self.map_y,
                "widthScreens": self.width_screens,
                "heightScreens": self.height_screens,
            },
        )


def _editor_edge_id(source_room_id: int, target_room_id: int) -> str:
    return f"editor_{source_room_id:04x}_to_{target_room_id:04x}"


def _graph_edge_from_editor(
    *,
    source_room_id: int,
    target_room_id: int,
    direction: str,
    is_elevator: bool = False,
    door_cap_color: str | None = None,
    requires: frozenset[str] = frozenset(),
    provenance: str = "super_metroid_editor",
    provenance_detail: str = "",
    verification: str = "planned",
    edge_id: str | None = None,
) -> GraphEdge:
    meta: dict[str, object] = {
        "isElevator": is_elevator,
        "doorCapColor": door_cap_color,
        "provenanceDetail": provenance_detail,
    }
    return GraphEdge(
        source_id=source_room_id,
        target_id=target_room_id,
        edge_id=edge_id or _editor_edge_id(source_room_id, target_room_id),
        direction=direction,
        requires=frozenset(normalize_capability(v) for v in requires),
        verification=verification,
        provenance=provenance,
        meta=meta,
    )


def edge_to_editor_dict(edge: GraphEdge) -> dict[str, object]:
    """Serialize a graph edge in the historical SM editor plan shape."""
    meta = dict(edge.meta)
    return {
        "edgeId": edge.edge_id,
        "sourceRoomId": edge.source_id,
        "sourceRoomIdHex": f"0x{int(edge.source_id):04X}",
        "targetRoomId": edge.target_id,
        "targetRoomIdHex": f"0x{int(edge.target_id):04X}",
        "direction": edge.direction,
        "isElevator": bool(meta.get("isElevator", False)),
        "doorCapColor": meta.get("doorCapColor"),
        "requires": sorted(edge.requires),
        "acquires": sorted(edge.acquires),
        "provenance": edge.provenance,
        "provenanceDetail": meta.get("provenanceDetail") or meta.get("support", ""),
        "verification": edge.verification,
    }


def planned_leg_to_editor_dict(
    planned: PlannedLeg,
    rooms: Mapping[int, EditorRoom],
) -> dict[str, object]:
    source = rooms[int(planned.leg.source_id)]
    target = rooms[int(planned.leg.target_id)]
    acquires = planned.edge.acquires | frozenset(
        normalize_capability(v) for v in planned.leg.acquires
    )
    return {
        "legId": planned.leg.leg_id,
        "source": source.to_dict(),
        "target": target.to_dict(),
        "edge": edge_to_editor_dict(planned.edge),
        "capabilitiesBefore": sorted(planned.capabilities_before),
        "effectiveRequires": sorted(planned.effective_requires),
        "acquires": sorted(acquires),
        "capabilitiesAfter": sorted(planned.capabilities_after),
        "goal": planned.leg.goal,
        "constraints": list(planned.leg.constraints),
        "status": "planned_not_continuous",
    }


def sm_route_patch(
    source_room_id: int,
    target_room_id: int,
    direction: str,
    requires: frozenset[str] = frozenset(),
    *,
    door_cap_color: str | None = None,
    support: str = "",
) -> RoutePatch:
    """Build a :class:`RoutePatch` with optional SM door-cap metadata."""
    meta: dict[str, object] = {}
    if door_cap_color is not None:
        meta["doorCapColor"] = door_cap_color
    if support:
        meta["provenanceDetail"] = support
    return RoutePatch(
        source_id=source_room_id,
        target_id=target_room_id,
        direction=direction,
        requires=requires,
        support=support,
        meta=meta,
    )


POST_TORIZO_CAPABILITIES = frozenset(
    {
        "morph_ball",
        "bombs",
        "missiles",
    }
)

POST_TORIZO_ROUTE_PATCHES = (
    sm_route_patch(
        0x9AD9,
        0x9CB3,
        "Right",
        frozenset({"missiles"}),
        door_cap_color="red",
        support=(
            "Missing from SMEDIT nav_graph; supported by exported inverse "
            "0x9CB3->0x9AD9 and the editor practice-route door record."
        ),
    ),
)

POST_TORIZO_TO_SPORE_SPAWN = (
    RouteLeg(
        "parlor_to_terminator",
        0x92FD,
        0x990D,
        frozenset({"bombs"}),
        goal="Bomb through upper Parlor and exit left.",
    ),
    RouteLeg(
        "terminator",
        0x990D,
        0x99BD,
        acquires=frozenset({"terminator_energy_tank"}),
        goal="Collect the Energy Tank naturally and continue left.",
    ),
    RouteLeg(
        "green_pirates",
        0x99BD,
        0x9969,
        goal="Descend Green Pirates Shaft into Lower Mushrooms.",
    ),
    RouteLeg(
        "lower_mushrooms",
        0x9969,
        0x9938,
        goal="Cross Lower Mushrooms to the Green Brinstar elevator.",
    ),
    RouteLeg(
        "green_elevator",
        0x9938,
        0x9AD9,
        goal="Ride down to Green Brinstar Main Shaft.",
    ),
    RouteLeg(
        "dachora_entry",
        0x9AD9,
        0x9CB3,
        frozenset({"missiles"}),
        goal="Open the lower-right red door into Dachora Room.",
    ),
    RouteLeg(
        "dachora",
        0x9CB3,
        0x9D19,
        goal="Traverse Dachora Room into Big Pink.",
    ),
    RouteLeg(
        "big_pink",
        0x9D19,
        0x9D9C,
        frozenset({"missiles"}),
        goal="Open the right red door toward Spore Spawn.",
    ),
    RouteLeg(
        "spore_kihunters",
        0x9D9C,
        0x9DC7,
        goal="Clear the Kihunters to open the forward gray boss-room lock.",
        constraints=(
            "Kill the Kihunters to clear the two gray locks. The forward "
            "Kihunter-to-boss edge has no ammo gate in the editor export; "
            "the green Super gate is on the reverse boss-to-Kihunter edge.",
        ),
    ),
    RouteLeg(
        "spore_spawn",
        0x9DC7,
        0x9B5B,
        frozenset({"missiles"}),
        frozenset({"spore_spawn_defeated"}),
        goal=(
            "Defeat Spore Spawn with naturally unlocked Missiles and reach "
            "the first playable post-boss room."
        ),
    ),
)


class EditorNavigationGraph:
    """Editor export loaded as a :class:`RouteGraph` plus room geometry."""

    def __init__(
        self,
        rooms: Iterable[EditorRoom],
        graph: RouteGraph,
        *,
        source_path: Path,
        source_sha256: str,
    ) -> None:
        self.rooms = {room.room_id: room for room in rooms}
        self.graph = graph
        self.source_path = source_path
        self.source_sha256 = source_sha256
        missing = set(self.graph.nodes) - set(self.rooms)
        if missing:
            raise ValueError(f"graph nodes without EditorRoom: {sorted(missing)[:5]}")

    @property
    def edges(self) -> tuple[GraphEdge, ...]:
        return self.graph.edges

    @classmethod
    def load(cls, path: str | Path) -> EditorNavigationGraph:
        source = Path(path).expanduser().resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        nodes = payload.get("nodes")
        edges = payload.get("edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise ValueError("editor nav graph must contain nodes and edges lists")

        rooms = tuple(
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
        graph_nodes = [room.as_graph_node() for room in rooms]
        graph_edges = tuple(
            _graph_edge_from_editor(
                source_room_id=_room_id(edge["fromRoomId"]),
                target_room_id=_room_id(edge["toRoomId"]),
                direction=str(edge["direction"]),
                is_elevator=bool(edge["isElevator"]),
                door_cap_color=edge.get("doorCapColor"),
                requires=(
                    frozenset({normalize_capability(str(edge["requiredAbility"]))})
                    if edge.get("requiredAbility")
                    else frozenset()
                ),
            )
            for edge in edges
        )
        return cls(
            rooms,
            RouteGraph(graph_nodes, graph_edges),
            source_path=source,
            source_sha256=sha256_file(source),
        )

    def edge_for(
        self,
        source_room_id: int,
        target_room_id: int,
    ) -> GraphEdge | None:
        return self.graph.edge_for(source_room_id, target_room_id)

    def add_patches(
        self,
        patches: Iterable[RoutePatch],
    ) -> EditorNavigationGraph:
        return EditorNavigationGraph(
            self.rooms.values(),
            self.graph.add_patches(patches),
            source_path=self.source_path,
            source_sha256=self.source_sha256,
        )

    def shortest_path(
        self,
        source_room_id: int,
        target_room_id: int,
        capabilities: frozenset[str] = frozenset(),
    ) -> tuple[GraphEdge, ...] | None:
        return self.graph.shortest_path(
            source_room_id,
            target_room_id,
            capabilities=capabilities,
        )

    def plan_legs(
        self,
        legs: Iterable[RouteLeg],
        *,
        initial_capabilities: frozenset[str],
    ) -> tuple[PlannedLeg, ...]:
        return self.graph.plan_legs(
            legs,
            initial_capabilities=initial_capabilities,
        )

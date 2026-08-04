"""Physical room topology from sm-json-data reference corpus.

Authoritative source for room connections, door endpoints, and undirected
physical components. Editor geometry and problem generation live elsewhere.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

from super_metroid.rooms.capabilities import (
    _DOOR_REQUIREMENTS,
    _LOCK_REQUIREMENTS,
    normalize_ability,
)


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _room_id(room_address: str) -> int:
    return int(room_address, 0) & 0xFFFF


def _mask_cells(mask: object) -> list[tuple[int, int]]:
    if not isinstance(mask, list):
        return []
    cells: list[tuple[int, int]] = []
    for y, row in enumerate(mask):
        if not isinstance(row, list):
            continue
        cells.extend((x, y) for x, value in enumerate(row) if value == 2)
    return cells


def _node_block(node: Mapping[str, object]) -> tuple[int, int] | None:
    cells = _mask_cells(node.get("mapTileMask"))
    if not cells:
        return None
    screen_x, screen_y = min(cells)
    orientation = str(node.get("doorOrientation", "")).lower()
    if orientation == "left":
        return screen_x * 16, screen_y * 16 + 7
    if orientation == "right":
        return screen_x * 16 + 15, screen_y * 16 + 7
    if orientation == "up":
        return screen_x * 16 + 7, screen_y * 16
    if orientation == "down":
        return screen_x * 16 + 7, screen_y * 16 + 15
    return screen_x * 16 + 7, screen_y * 16 + 7


def _requirement_strings(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, str):
        found.add(value)
    elif isinstance(value, dict):
        for child in value.values():
            found.update(_requirement_strings(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_requirement_strings(child))
    return found


def _requirements_for_node(
    node: Mapping[str, object],
) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    capabilities: set[str] = set()
    local: set[str] = set()
    impossible = False
    subtype = str(node.get("nodeSubType", "")).lower()
    if subtype in _DOOR_REQUIREMENTS:
        capabilities.add(_DOOR_REQUIREMENTS[subtype])
    elif subtype in {"gray", "eye"}:
        local.add("clear_local_lock")

    for lock in node.get("locks", []) or []:
        if not isinstance(lock, dict):
            continue
        lock_type = str(lock.get("lockType", ""))
        activation = _requirement_strings(lock.get("lock", []))
        raw_requirements: set[str] = set()
        for strategy in lock.get("unlockStrats", []) or []:
            if isinstance(strategy, dict):
                raw_requirements.update(
                    _requirement_strings(strategy.get("requires", []))
                )

        # Escape funnels and other conditional locks do not block ordinary
        # traversal merely because their endgame activation condition exists.
        if "f_ZebesSetAblaze" in activation:
            local.add("conditional_endgame_lock")
            continue
        if lock_type == "escapeFunnel":
            local.add("conditional_endgame_lock")
            continue
        if lock_type == "killEnemies":
            local.add("clear_room_enemies")
        elif lock_type in {"coloredDoor", "cutscene"}:
            local.add(f"resolve_{lock_type}")
        elif lock_type == "permanent" and not activation:
            usable = raw_requirements - {"never"}
            if not usable:
                impossible = True
                continue
            local.add("special_one_way_exit")

        for raw in raw_requirements - {"never"}:
            mapped = _LOCK_REQUIREMENTS.get(raw)
            if lock_type in {"coloredDoor", "cutscene"}:
                local.add(raw)
            elif mapped is not None:
                capabilities.add(mapped)
            elif raw.startswith(("h_", "f_KilledMetroidRoom")):
                local.add(raw)
            elif lock_type in {"killEnemies", "permanent"}:
                local.add(raw)
            elif raw != "h_allItemsSpawned":
                capabilities.add(normalize_ability(raw))
    return tuple(sorted(capabilities)), tuple(sorted(local)), impossible


@dataclass(frozen=True)
class ReferenceRoom:
    logical_id: int
    room_id: int
    name: str
    path: Path
    nodes: Mapping[int, Mapping[str, object]]


@dataclass(frozen=True)
class PhysicalEndpoint:
    room_id: int
    logical_room_id: int
    node_id: int
    node_name: str
    position: str
    orientation: str
    subtype: str
    block: tuple[int, int] | None
    requires: tuple[str, ...]
    local_requirements: tuple[str, ...]
    impossible_exit: bool
    # Bank-$83 door definition pointer (sm-json-data nodeAddress low 16 bits).
    # On the *source* side of a hop this is the door_warp argument that enters
    # the peer room.
    door_ptr: int | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "roomId": self.room_id,
            "roomIdHex": f"0x{self.room_id:04X}",
            "logicalRoomId": self.logical_room_id,
            "nodeId": self.node_id,
            "nodeName": self.node_name,
            "position": self.position,
            "orientation": self.orientation,
            "subtype": self.subtype,
            "block": list(self.block) if self.block is not None else None,
            "requires": list(self.requires),
            "localRequirements": list(self.local_requirements),
            "impossibleExit": self.impossible_exit,
        }
        if self.door_ptr is not None:
            payload["doorPtr"] = self.door_ptr
            payload["doorPtrHex"] = f"0x{self.door_ptr:04X}"
        return payload


@dataclass(frozen=True)
class PhysicalConnection:
    connection_id: str
    connection_type: str
    description: str
    direction: str
    first: PhysicalEndpoint
    second: PhysicalEndpoint

    def directed_edges(self) -> tuple[dict[str, object], ...]:
        pairs = [(self.first, self.second)]
        if self.direction == "Bidirectional":
            pairs.append((self.second, self.first))
        edges = []
        for index, (source, target) in enumerate(pairs):
            edges.append(
                {
                    "edgeId": f"{self.connection_id}_{index}",
                    "connectionId": self.connection_id,
                    "source": source.to_dict(),
                    "target": target.to_dict(),
                    "connectionType": self.connection_type,
                    "direction": self.direction,
                    "requires": list(source.requires),
                    "localRequirements": list(source.local_requirements),
                    "impossible": source.impossible_exit,
                    "verification": "reference_topology",
                }
            )
        return tuple(edges)

    def to_dict(self) -> dict[str, object]:
        return {
            "connectionId": self.connection_id,
            "connectionType": self.connection_type,
            "description": self.description,
            "direction": self.direction,
            "endpoints": [self.first.to_dict(), self.second.to_dict()],
        }


def _load_reference_rooms(reference_root: Path) -> dict[int, ReferenceRoom]:
    rooms: dict[int, ReferenceRoom] = {}
    for path in (reference_root / "region").rglob("*.json"):
        payload = _json(path)
        if "roomAddress" not in payload or "id" not in payload:
            continue
        logical_id = int(payload["id"])
        nodes = {
            int(node["id"]): node
            for node in payload.get("nodes", [])
            if isinstance(node, dict) and "id" in node
        }
        if logical_id in rooms:
            raise ValueError(f"duplicate logical room id {logical_id}")
        rooms[logical_id] = ReferenceRoom(
            logical_id=logical_id,
            room_id=_room_id(str(payload["roomAddress"])),
            name=str(payload["name"]),
            path=path,
            nodes=nodes,
        )
    return rooms


def _door_ptr_from_node(node: Mapping[str, object]) -> int | None:
    raw = node.get("nodeAddress")
    if raw is None or raw == "" or raw == "null":
        return None
    return int(str(raw), 0) & 0xFFFF


def _endpoint(
    raw: Mapping[str, object],
    rooms: Mapping[int, ReferenceRoom],
) -> PhysicalEndpoint:
    logical_id = int(raw["roomid"])
    room = rooms[logical_id]
    node_id = int(raw["nodeid"])
    node = room.nodes[node_id]
    requires, local, impossible = _requirements_for_node(node)
    return PhysicalEndpoint(
        room_id=room.room_id,
        logical_room_id=logical_id,
        node_id=node_id,
        node_name=str(raw.get("nodeName", node.get("name", ""))),
        position=str(raw.get("position", "")),
        orientation=str(node.get("doorOrientation", "")),
        subtype=str(node.get("nodeSubType", "")),
        block=_node_block(node),
        requires=requires,
        local_requirements=local,
        impossible_exit=impossible,
        door_ptr=_door_ptr_from_node(node),
    )


def _load_connections(
    reference_root: Path,
    rooms: Mapping[int, ReferenceRoom],
) -> tuple[PhysicalConnection, ...]:
    result: list[PhysicalConnection] = []
    index = 0
    for path in sorted((reference_root / "connection").rglob("*.json")):
        payload = _json(path)
        for raw in payload.get("connections", []):
            if not isinstance(raw, dict):
                continue
            nodes = raw.get("nodes")
            if not isinstance(nodes, list) or len(nodes) != 2:
                raise ValueError(f"connection must have two endpoints: {path}")
            first = _endpoint(nodes[0], rooms)
            second = _endpoint(nodes[1], rooms)
            connection_id = (
                f"connection_{index:03d}_{first.room_id:04x}_"
                f"{first.node_id}_to_{second.room_id:04x}_{second.node_id}"
            )
            result.append(
                PhysicalConnection(
                    connection_id=connection_id,
                    connection_type=str(raw["connectionType"]),
                    description=str(raw.get("description", "")),
                    direction=str(raw["direction"]),
                    first=first,
                    second=second,
                )
            )
            index += 1
    return tuple(result)


def _physical_components(
    room_ids: set[int],
    connections: Sequence[PhysicalConnection],
) -> list[set[int]]:
    adjacency: dict[int, set[int]] = {room_id: set() for room_id in room_ids}
    for connection in connections:
        first = connection.first.room_id
        second = connection.second.room_id
        if first in adjacency and second in adjacency:
            adjacency[first].add(second)
            adjacency[second].add(first)

    remaining = set(room_ids)
    components: list[set[int]] = []
    while remaining:
        seed = min(remaining)
        component = {seed}
        queue = deque([seed])
        remaining.remove(seed)
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current] & remaining:
                remaining.remove(neighbor)
                component.add(neighbor)
                queue.append(neighbor)
        components.append(component)
    return components

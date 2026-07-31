"""Full-game room topology and canonical room-clear problem generation.

The SM editor export is excellent geometry data, but its directed navigation
graph omits several legitimate door endpoints.  The bundled ``sm-json-data``
reference corpus is authoritative for physical room connections and explicitly
marks the handful of one-way story/sand transitions.  This module merges the
two sources:

* reference room/connection JSON supplies physical topology;
* editor room JSON supplies collision grids, exact item blocks, and counts;
* generated artifacts stay planning/development inputs until emulator evidence
  promotes an individual room clear.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

from adventure_common.hashutil import sha256_file
from super_metroid.rooms.capabilities import (
    _DOOR_REQUIREMENTS,
    _ITEM_CAPABILITIES,
    _LOCK_REQUIREMENTS,
    normalize_ability,
)

_BOSS_ROOM_IDS = {
    0x9804,  # Bomb Torizo
    0x9DC7,  # Spore Spawn
    0xA59F,  # Kraid
    0xA98D,  # Crocomire
    0xB283,  # Golden Torizo
    0xB32E,  # Ridley
    0xCD13,  # Phantoon
    0xD95E,  # Botwoon
    0xDA60,  # Draygon
    0xDD58,  # Mother Brain
    0xE0B5,  # Ceres Ridley
}

_SPECIAL_LATE_NAMES = (
    "Escape",
    "Metroid Room",
    "Statues Room",
    "Tourian",
    "The Worst Room",
)


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _aggregate_sha256(root: Path, paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


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


def _item_capabilities(items: Sequence[Mapping[str, object]]) -> list[str]:
    found: set[str] = set()
    for item in items:
        name = str(item.get("name", "")).split(" (", 1)[0].lower()
        capability = _ITEM_CAPABILITIES.get(name)
        if capability:
            found.add(capability)
    return sorted(found)


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


def _nearest_open(
    collision: Sequence[Sequence[int]],
    point: tuple[int, int],
) -> tuple[int, int] | None:
    if not collision or not collision[0]:
        return None
    width = len(collision[0])
    height = len(collision)
    start = (
        min(max(point[0], 0), width - 1),
        min(max(point[1], 0), height - 1),
    )
    queue = deque([start])
    seen = {start}
    while queue:
        x, y = queue.popleft()
        if int(collision[y][x]) == 0:
            return x, y
        for nxt in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            nx, ny = nxt
            if 0 <= nx < width and 0 <= ny < height and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return None


def _grid_path(
    collision: Sequence[Sequence[int]],
    start: tuple[int, int],
    target: tuple[int, int],
) -> list[tuple[int, int]] | None:
    source = _nearest_open(collision, start)
    goal = _nearest_open(collision, target)
    if source is None or goal is None:
        return None
    queue = deque([source])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {source: None}
    width = len(collision[0])
    height = len(collision)
    while queue and goal not in parent:
        x, y = queue.popleft()
        for nxt in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            nx, ny = nxt
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if nxt in parent or int(collision[ny][nx]) != 0:
                continue
            parent[nxt] = (x, y)
            queue.append(nxt)
    if goal not in parent:
        return None
    path = []
    cursor: tuple[int, int] | None = goal
    while cursor is not None:
        path.append(cursor)
        cursor = parent[cursor]
    return list(reversed(path))


def _compress_path(path: Sequence[tuple[int, int]]) -> list[list[int]]:
    if len(path) <= 2:
        return [list(point) for point in path]
    result = [path[0]]
    previous_direction = (
        path[1][0] - path[0][0],
        path[1][1] - path[0][1],
    )
    for index in range(1, len(path) - 1):
        direction = (
            path[index + 1][0] - path[index][0],
            path[index + 1][1] - path[index][1],
        )
        if direction != previous_direction:
            result.append(path[index])
        previous_direction = direction
    result.append(path[-1])
    return [list(point) for point in result]


def _distance(
    first: PhysicalEndpoint,
    second: PhysicalEndpoint,
) -> int:
    if first.block is None or second.block is None:
        return 0
    return abs(first.block[0] - second.block[0]) + abs(first.block[1] - second.block[1])


def _exit_cost(endpoint: PhysicalEndpoint) -> tuple[int, int, str]:
    return (
        int(endpoint.impossible_exit) * 100
        + len(endpoint.requires) * 10
        + len(endpoint.local_requirements) * 3,
        endpoint.node_id,
        endpoint.node_name,
    )


def _canonical_endpoints(
    room_id: int,
    connections: Sequence[PhysicalConnection],
) -> tuple[
    tuple[PhysicalEndpoint, PhysicalEndpoint] | None,
    tuple[PhysicalEndpoint, PhysicalEndpoint] | None,
]:
    """Pick canonical (local, peer) endpoint pairs for entry and exit.

    ``peer.door_ptr`` on the entry pair is the door-warp argument that enters
    this room from the peer (source) room.
    """
    # (local, peer, can_enter_from_peer, can_exit_to_peer)
    incident: list[tuple[PhysicalEndpoint, PhysicalEndpoint, bool, bool]] = []
    for connection in connections:
        if connection.first.room_id == room_id:
            incident.append(
                (
                    connection.first,
                    connection.second,
                    connection.direction == "Bidirectional",
                    True,
                )
            )
        if connection.second.room_id == room_id:
            incident.append(
                (
                    connection.second,
                    connection.first,
                    True,
                    connection.direction == "Bidirectional",
                )
            )
    if not incident:
        return None, None

    entry_candidates = [item for item in incident if item[2]]
    exit_candidates = [
        item for item in incident if item[3] and not item[0].impossible_exit
    ]
    if not entry_candidates or not exit_candidates:
        return None, None
    if len(incident) == 1:
        item = incident[0]
        pair = (item[0], item[1])
        return pair, pair

    candidates = [
        (entry, exit_)
        for entry in entry_candidates
        for exit_ in exit_candidates
        if entry[0].node_id != exit_[0].node_id
    ]
    if not candidates:
        entry = min(entry_candidates, key=lambda item: item[0].node_id)
        exit_ = min(exit_candidates, key=lambda item: _exit_cost(item[0]))
    else:
        maximum = max(_distance(entry[0], exit_[0]) for entry, exit_ in candidates)
        furthest = [
            pair for pair in candidates if _distance(pair[0][0], pair[1][0]) == maximum
        ]
        entry, exit_ = min(
            furthest,
            key=lambda pair: (
                _exit_cost(pair[1][0]),
                pair[0][0].node_id,
            ),
        )
    return (entry[0], entry[1]), (exit_[0], exit_[1])


def _objective(room: Mapping[str, object], boss: bool, endpoint_count: int) -> str:
    if boss:
        return "defeat_boss_and_exit"
    name = str(room["name"])
    if "Escape" in name:
        return "scripted_escape"
    if room.get("items"):
        return "collect_items_and_exit" if endpoint_count > 1 else "collect_and_return"
    if any(token in name for token in ("Save Room", "Refill", "Recharge", "Map Room")):
        return "visit_station_and_return"
    if endpoint_count <= 1:
        return "enter_objective_and_return"
    return "traverse_to_exit"


def _difficulty(
    room: Mapping[str, object],
    *,
    boss: bool,
    endpoint_count: int,
    exit_endpoint: PhysicalEndpoint | None,
    static_path_found: bool,
) -> tuple[str, int, list[str]]:
    name = str(room["name"])
    area = str(room["areaName"])
    screens = int(room["widthScreens"]) * int(room["heightScreens"])
    enemies = len(room.get("enemies", []))
    reasons: list[str] = []
    if boss:
        return "boss_late", 4, ["boss room"]
    if (
        area in {"Ceres", "Tourian"}
        or any(token in name for token in _SPECIAL_LATE_NAMES)
        or endpoint_count == 0
    ):
        return "late_special", 3, ["scripted/late-game room"]
    if exit_endpoint and (exit_endpoint.requires or exit_endpoint.local_requirements):
        reasons.append("gated exit")
    if screens >= 12 or int(room["heightScreens"]) >= 4:
        reasons.append("large/vertical geometry")
    if enemies >= 12:
        reasons.append("dense enemies")
    if not static_path_found and endpoint_count > 1:
        reasons.append("static collision path unresolved")
    if reasons:
        return "tough", 3, reasons
    if screens <= 3 and enemies <= 5:
        return "easy", 1, ["small low-enemy room"]
    return "standard", 2, ["ordinary traversal"]


def _static_plan(
    room: Mapping[str, object],
    entry: PhysicalEndpoint | None,
    exit_: PhysicalEndpoint | None,
) -> dict[str, object]:
    collision = room.get("collision", [])
    if entry is None or exit_ is None or entry.block is None or exit_.block is None:
        return {
            "status": "unavailable",
            "reason": "missing entry/exit geometry",
            "waypointsBlocks": [],
        }
    target = exit_.block
    if entry.node_id == exit_.node_id:
        items = room.get("items", [])
        if items:
            target = max(
                (
                    (int(item["blockX"]), int(item["blockY"]))
                    for item in items
                    if isinstance(item, dict)
                ),
                key=lambda point: abs(point[0] - entry.block[0])
                + abs(point[1] - entry.block[1]),
                default=target,
            )
        else:
            target = (
                int(room["widthBlocks"]) // 2,
                int(room["heightBlocks"]) // 2,
            )
    path = _grid_path(collision, entry.block, target)
    if path is None:
        return {
            "status": "unresolved",
            "reason": (
                "no air-only collision path; dynamic blocks or movement "
                "abilities need runtime planning"
            ),
            "entryBlock": list(entry.block),
            "objectiveBlock": list(target),
            "exitBlock": list(exit_.block),
            "waypointsBlocks": [],
        }
    return {
        "status": "planned_static",
        "warning": (
            "Air-cell connectivity only; jumps, physics, enemies, and dynamic "
            "blocks remain runtime work."
        ),
        "entryBlock": list(entry.block),
        "objectiveBlock": list(target),
        "exitBlock": list(exit_.block),
        "pathBlocks": len(path),
        "waypointsBlocks": _compress_path(path),
    }


def _problem_for_room(
    room: Mapping[str, object],
    connections: Sequence[PhysicalConnection],
    *,
    state_names: set[str],
    verified_policy_ids: set[str],
) -> dict[str, object]:
    room_id = int(room["roomId"])
    entry_pair, exit_pair = _canonical_endpoints(room_id, connections)
    entry = entry_pair[0] if entry_pair else None
    entry_peer = entry_pair[1] if entry_pair else None
    entry_source = entry_peer.room_id if entry_peer is not None else None
    exit_ = exit_pair[0] if exit_pair else None
    exit_peer = exit_pair[1] if exit_pair else None
    exit_target = exit_peer.room_id if exit_peer is not None else None
    endpoint_count = sum(
        connection.first.room_id == room_id or connection.second.room_id == room_id
        for connection in connections
    )
    boss = room_id in _BOSS_ROOM_IDS
    plan = _static_plan(room, entry, exit_)
    tier, queue, reasons = _difficulty(
        room,
        boss=boss,
        endpoint_count=endpoint_count,
        exit_endpoint=exit_,
        static_path_found=plan["status"] == "planned_static",
    )
    if entry_source is None or exit_target is None:
        problem_id = f"room_{room_id:04x}_scripted"
        state_name = f"room_{room_id:04x}_entry"
    else:
        problem_id = f"room_{room_id:04x}_from_{entry_source:04x}_to_{exit_target:04x}"
        state_name = f"room_{room_id:04x}_from_{entry_source:04x}"
    state_ready = state_name in state_names
    policy_ready = problem_id in verified_policy_ids
    if state_ready and policy_ready:
        status = "ready"
        queue = 0
    elif state_ready:
        status = "state_ready"
    else:
        status = "unstarted"
    entry_payload: dict[str, object] | None = None
    if entry is not None and entry_source is not None and entry_peer is not None:
        entry_payload = {
            "sourceRoomId": entry_source,
            "sourceRoomIdHex": f"0x{entry_source:04X}",
            "endpoint": entry.to_dict(),
        }
        # Source-side door definition used by door_warp into this room.
        if entry_peer.door_ptr is not None:
            entry_payload["doorPtr"] = entry_peer.door_ptr
            entry_payload["doorPtrHex"] = f"0x{entry_peer.door_ptr:04X}"
    exit_payload: dict[str, object] | None = None
    if exit_ is not None and exit_target is not None:
        exit_payload = {
            "targetRoomId": exit_target,
            "targetRoomIdHex": f"0x{exit_target:04X}",
            "endpoint": exit_.to_dict(),
        }
    return {
        "problemId": problem_id,
        "roomId": room_id,
        "roomIdHex": f"0x{room_id:04X}",
        "roomName": room["name"],
        "area": room["areaName"],
        "objective": _objective(room, boss, endpoint_count),
        "tier": tier,
        "queue": queue,
        "difficultyReasons": reasons,
        "endpointCount": endpoint_count,
        "entry": entry_payload,
        "exit": exit_payload,
        "acquires": _item_capabilities(room.get("items", [])),
        "items": room.get("items", []),
        "geometry": {
            "widthScreens": room["widthScreens"],
            "heightScreens": room["heightScreens"],
            "widthBlocks": room["widthBlocks"],
            "heightBlocks": room["heightBlocks"],
            "enemyCount": len(room.get("enemies", [])),
        },
        "staticPlan": plan,
        "practice": {
            "status": status,
            "stateName": state_name,
            "stateFile": (f"custom_integrations/SuperMetroid-Snes/{state_name}.state"),
            "policyFile": f"policies/room_clears/{problem_id}.json",
            "reportFile": f"recordings/room_clears/{problem_id}.json",
        },
        "verification": "planned_not_continuous",
    }


def _capability_path(
    edges: Sequence[Mapping[str, object]],
    source: int,
    target: int,
    capabilities: set[str],
) -> list[Mapping[str, object]] | None:
    if source == target:
        return []
    outgoing: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for edge in edges:
        outgoing[int(edge["source"]["roomId"])].append(edge)
    queue = deque([source])
    parent: dict[int, tuple[int, Mapping[str, object]]] = {}
    seen = {source}
    while queue:
        room_id = queue.popleft()
        for edge in outgoing.get(room_id, []):
            if edge.get("impossible"):
                continue
            if not set(edge.get("requires", [])).issubset(capabilities):
                continue
            next_room = int(edge["target"]["roomId"])
            if next_room in seen:
                continue
            seen.add(next_room)
            parent[next_room] = (room_id, edge)
            if next_room == target:
                path: list[Mapping[str, object]] = []
                cursor = target
                while cursor != source:
                    previous, used = parent[cursor]
                    path.append(used)
                    cursor = previous
                return list(reversed(path))
            queue.append(next_room)
    return None


def _completion_anchors(legacy_route: Mapping[str, object]) -> list[dict[str, object]]:
    anchors = [dict(item) for item in legacy_route["anchors"]]
    bomb_index = next(
        index for index, item in enumerate(anchors) if item["id"] == "bomb_torizo"
    )
    bomb_acquires = set(anchors[bomb_index].get("acquires", []))
    bomb_acquires.add("bomb_torizo_defeated")
    anchors[bomb_index]["acquires"] = sorted(bomb_acquires)
    early_supers_index = next(
        index for index, item in enumerate(anchors) if item["id"] == "early_supers"
    )
    spore_anchors = [
        {
            "id": "spore_spawn",
            "label": "Spore Spawn",
            "roomId": "0x9DC7",
            "kind": "boss",
            "requires": ["morph_ball", "bombs", "missiles"],
            "acquires": ["spore_spawn_defeated"],
            "constraints": {"hard": ["defeat Spore Spawn and exit naturally"]},
            "verification": "continuous",
        },
        {
            "id": "spore_spawn_supers",
            "label": "Spore Spawn Super Missiles",
            "roomId": "0x9B5B",
            "kind": "item",
            "requires": ["spore_spawn_defeated"],
            "acquires": ["super_missiles"],
            "constraints": {"hard": ["collect Super Missile expansion"]},
            "verification": "next_unverified",
        },
    ]
    return anchors[: bomb_index + 1] + spore_anchors + anchors[early_supers_index + 1 :]


def _completion_sequence(
    edges: Sequence[Mapping[str, object]],
    legacy_route: Mapping[str, object],
    rooms: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    anchors = _completion_anchors(legacy_route)
    capabilities: set[str] = set()
    legs: list[dict[str, object]] = []
    for index, anchor in enumerate(anchors):
        capabilities.update(anchor.get("acquires", []))
        if index + 1 >= len(anchors):
            break
        target = anchors[index + 1]
        source_id = int(anchor["roomId"], 0)
        target_id = int(target["roomId"], 0)
        path = _capability_path(edges, source_id, target_id, capabilities)
        legs.append(
            {
                "sourceAnchor": anchor["id"],
                "targetAnchor": target["id"],
                "capabilitiesBefore": sorted(capabilities),
                "status": "planned" if path is not None else "topology_gap",
                "roomPath": (
                    [
                        f"0x{source_id:04X}",
                        *(f"0x{int(edge['target']['roomId']):04X}" for edge in path),
                    ]
                    if path is not None
                    else []
                ),
                "roomNames": (
                    [
                        rooms[source_id]["name"],
                        *(
                            rooms[int(edge["target"]["roomId"])]["name"]
                            for edge in path
                        ),
                    ]
                    if path is not None
                    else []
                ),
                "edgeIds": (
                    [str(edge["edgeId"]) for edge in path] if path is not None else []
                ),
            }
        )
    return {
        "status": "research_route_not_acceptance_evidence",
        "warning": (
            "Door topology and color/flag gates are modeled. In-room terrain "
            "tech, dynamic events, resource floors, and route optimality are not."
        ),
        "anchors": anchors,
        "legs": legs,
    }


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


def export_full_room_catalog(**kwargs):
    """Compatibility re-export — implementation lives in room_catalog."""
    from super_metroid.rooms.room_catalog import (
        export_full_room_catalog as _export,
    )

    return _export(**kwargs)


def load_problem_catalog(path: Path) -> dict[str, object]:
    payload = _json(path)
    problems = payload.get("problems")
    if not isinstance(problems, list):
        raise ValueError("room problem catalog has no problems list")
    return payload


def problem_by_id(
    catalog: Mapping[str, object],
    problem_id: str,
) -> Mapping[str, object]:
    for problem in catalog["problems"]:
        if problem["problemId"] == problem_id:
            return problem
    raise KeyError(f"unknown room problem: {problem_id}")


def shortest_room_path(
    graph: Mapping[str, object],
    source_room_id: int,
    target_room_id: int,
    capabilities: Iterable[str] = (),
) -> list[Mapping[str, object]] | None:
    normalized = {normalize_ability(value) for value in capabilities}
    return _capability_path(
        graph["edges"],
        source_room_id,
        target_room_id,
        normalized,
    )

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

Topology loading lives in :mod:`super_metroid.rooms.topology`; grid and
capability pathfinding in :mod:`super_metroid.rooms.pathfind`. Public symbols
are re-exported here for stable import paths.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

from retro_harness.adventure.hashutil import sha256_file
from super_metroid.rooms.capabilities import _ITEM_CAPABILITIES
from super_metroid.rooms.pathfind import (
    _capability_path,
    _compress_path,
    _grid_path,
    shortest_room_path,
)
from super_metroid.rooms.topology import (
    PhysicalConnection,
    PhysicalEndpoint,
    ReferenceRoom,
    _json,
)

# Public re-exports (stable import path for tests, scripts, rooms/__init__).
__all__ = [
    "PhysicalConnection",
    "PhysicalEndpoint",
    "ReferenceRoom",
    "export_full_room_catalog",
    "load_problem_catalog",
    "problem_by_id",
    "shortest_room_path",
]

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


def _aggregate_sha256(root: Path, paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _item_capabilities(items: Sequence[Mapping[str, object]]) -> list[str]:
    found: set[str] = set()
    for item in items:
        name = str(item.get("name", "")).split(" (", 1)[0].lower()
        capability = _ITEM_CAPABILITIES.get(name)
        if capability:
            found.add(capability)
    return sorted(found)


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

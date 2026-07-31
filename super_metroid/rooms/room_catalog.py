"""Full room catalog / problem export (sm-json-data + editor merge).

Topology helpers and pathfinding live in :mod:`super_metroid.rooms.room_graph`.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from adventure_common.hashutil import sha256_file
from super_metroid.rooms.room_graph import (
    _aggregate_sha256,
    _completion_sequence,
    _item_capabilities,
    _json,
    _load_connections,
    _load_reference_rooms,
    _physical_components,
    _problem_for_room,
)

def export_full_room_catalog(
    *,
    editor_nav: Path,
    reference_root: Path,
    legacy_route: Path,
    graph_output: Path,
    problems_output: Path,
    states_dir: Path,
    policy_dir: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Merge sources and write self-contained graph/problem artifacts."""
    editor_nav = editor_nav.expanduser().resolve()
    editor_root = editor_nav.parent
    editor_rooms_dir = editor_root / "rooms"
    reference_root = reference_root.expanduser().resolve()
    legacy_route = legacy_route.expanduser().resolve()

    nav = _json(editor_nav)
    editor_rooms = {
        int(node["roomId"]): _json(
            editor_rooms_dir / f"room_{int(node['roomId']):04X}.json"
        )
        for node in nav["nodes"]
    }
    reference_rooms = _load_reference_rooms(reference_root)
    connections = _load_connections(reference_root, reference_rooms)
    edges = [edge for connection in connections for edge in connection.directed_edges()]

    reference_ids = {room.room_id for room in reference_rooms.values()}
    editor_ids = set(editor_rooms)
    if reference_ids - editor_ids:
        missing = ", ".join(
            f"0x{room_id:04X}" for room_id in reference_ids - editor_ids
        )
        raise ValueError(f"reference rooms missing editor geometry: {missing}")

    state_names = {path.stem for path in states_dir.glob("*.state")}
    verified_policy_ids = {
        path.stem
        for path in policy_dir.glob("*.json")
        if _json(path).get("status") == "verified_development_state"
    }
    problems = [
        _problem_for_room(
            editor_rooms[room_id],
            connections,
            state_names=state_names,
            verified_policy_ids=verified_policy_ids,
        )
        for room_id in sorted(editor_rooms)
    ]
    tier_counts = Counter(problem["tier"] for problem in problems)
    status_counts = Counter(problem["practice"]["status"] for problem in problems)
    queue_counts = Counter(problem["queue"] for problem in problems)
    static_plan_counts = Counter(
        problem["staticPlan"]["status"] for problem in problems
    )
    direction_counts = Counter(connection.direction for connection in connections)
    type_counts = Counter(connection.connection_type for connection in connections)
    editor_components = _physical_components(editor_ids, connections)
    reference_components = _physical_components(reference_ids, connections)
    isolated_editor_ids = sorted(
        next(iter(component)) for component in editor_components if len(component) == 1
    )

    reference_paths = [room.path for room in reference_rooms.values()] + list(
        (reference_root / "connection").rglob("*.json")
    )
    editor_room_paths = list(editor_rooms_dir.glob("room_*.json"))
    source = {
        "editorNavPath": str(editor_nav),
        "editorNavSha256": sha256_file(editor_nav),
        "editorRoomsPath": str(editor_rooms_dir),
        "editorRoomsAggregateSha256": _aggregate_sha256(
            editor_rooms_dir, editor_room_paths
        ),
        "referenceRoot": str(reference_root),
        "referenceAggregateSha256": _aggregate_sha256(reference_root, reference_paths),
        "legacyRoutePath": str(legacy_route),
        "legacyRouteSha256": sha256_file(legacy_route),
    }
    generated_at = datetime.now(timezone.utc).isoformat()
    completion_sequence = _completion_sequence(
        edges,
        _json(legacy_route),
        editor_rooms,
    )
    graph_payload = {
        "schemaVersion": 1,
        "graphId": "super_metroid_full_room_completion",
        "status": "planned_not_continuous",
        "acceptanceWarning": (
            "Reference topology and editor geometry are planning inputs. "
            "Individual edges become accepted only after emulator observation."
        ),
        "generatedAt": generated_at,
        "source": source,
        "summary": {
            "roomCount": len(editor_rooms),
            "vanillaReferenceRoomCount": len(reference_rooms),
            "editorOnlyRoomCount": len(editor_ids - reference_ids),
            "physicalConnectionCount": len(connections),
            "directedEdgeCount": len(edges),
            "editorPhysicalComponentCount": len(editor_components),
            "vanillaPhysicalComponentCount": len(reference_components),
            "isolatedEditorRoomIds": [
                f"0x{room_id:04X}" for room_id in isolated_editor_ids
            ],
            "directionCounts": dict(sorted(direction_counts.items())),
            "connectionTypeCounts": dict(sorted(type_counts.items())),
            "completionAnchorCount": len(completion_sequence["anchors"]),
            "completionLegCount": len(completion_sequence["legs"]),
            "completionTopologyGapCount": sum(
                leg["status"] != "planned" for leg in completion_sequence["legs"]
            ),
        },
        "rooms": [
            {
                "roomId": room_id,
                "roomIdHex": f"0x{room_id:04X}",
                "name": room["name"],
                "handle": room["handle"],
                "area": room["areaName"],
                "mapX": room["mapX"],
                "mapY": room["mapY"],
                "widthScreens": room["widthScreens"],
                "heightScreens": room["heightScreens"],
                "widthBlocks": room["widthBlocks"],
                "heightBlocks": room["heightBlocks"],
                "items": room.get("items", []),
                "acquires": _item_capabilities(room.get("items", [])),
                "enemyCount": len(room.get("enemies", [])),
                "referenceTopology": room_id in reference_ids,
            }
            for room_id, room in sorted(editor_rooms.items())
        ],
        "connections": [connection.to_dict() for connection in connections],
        "edges": edges,
        "completionSequence": completion_sequence,
    }
    problems_payload = {
        "schemaVersion": 1,
        "catalogId": "super_metroid_room_problems",
        "status": "development_catalog",
        "generatedAt": generated_at,
        "sourceGraph": {
            "path": str(graph_output.resolve()),
            "graphId": graph_payload["graphId"],
        },
        "summary": {
            "problemCount": len(problems),
            "tierCounts": dict(sorted(tier_counts.items())),
            "practiceStatusCounts": dict(sorted(status_counts.items())),
            "queueCounts": {
                str(queue): count for queue, count in sorted(queue_counts.items())
            },
            "staticPlanStatusCounts": dict(sorted(static_plan_counts.items())),
        },
        "queuePolicy": [
            {"queue": 0, "meaning": "state and policy ready; run now"},
            {"queue": 1, "meaning": "easy/small rooms"},
            {"queue": 2, "meaning": "standard traversal"},
            {"queue": 3, "meaning": "tough, scripted, or unresolved geometry"},
            {"queue": 4, "meaning": "bosses held for later"},
        ],
        "problems": sorted(
            problems,
            key=lambda problem: (
                int(problem["queue"]),
                str(problem["area"]),
                int(problem["roomId"]),
            ),
        ),
    }
    graph_output.parent.mkdir(parents=True, exist_ok=True)
    problems_output.parent.mkdir(parents=True, exist_ok=True)
    graph_output.write_text(
        json.dumps(graph_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    problems_output.write_text(
        json.dumps(problems_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return graph_payload, problems_payload



"""Validate opening-route checkpoints against Z3 JSON and emit artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alttp.opening_route_data import (
    CATALOG_KIND,
    CATALOG_VERSION,
    DEFAULT_ARTIFACT,
    DISCLAIMER,
    OVERWORLD_SCREEN_PATH,
    ExpectedConnection,
    ExpectedNode,
    OpeningCheckpoint,
    opening_checkpoints,
)
from alttp.paths import RECORDINGS_DIR, Z3_JSON_DATA_PIN
from alttp.ram import (
    HYRULE_CASTLE_SCREEN,
    LINKS_HOUSE_ROOM,
    LINKS_HOUSE_SCREEN,
)
from alttp.z3_json_data import (
    OPENING_ROUTE_ROOM_NAMES,
    Z3Connection,
    Z3JsonData,
    Z3JsonDataError,
    Z3JsonDataNotFoundError,
    Z3Room,
    source_status,
)

@dataclass
class CheckResult:
    """One validation finding."""

    ok: bool
    kind: str  # room | node | connection | source
    name: str
    required: bool
    detail: str = ""


@dataclass
class CatalogValidation:
    """Aggregate validation of catalog expectations against loaded z3 data."""

    ok: bool
    required_ok: bool
    results: list[CheckResult] = field(default_factory=list)
    rooms_present: list[str] = field(default_factory=list)
    rooms_missing: list[str] = field(default_factory=list)
    nodes_present: list[str] = field(default_factory=list)
    nodes_missing: list[str] = field(default_factory=list)
    connections_present: list[str] = field(default_factory=list)
    connections_missing: list[str] = field(default_factory=list)
    connections_optional_missing: list[str] = field(default_factory=list)

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "required_ok": self.required_ok,
            "rooms_present": list(self.rooms_present),
            "rooms_missing": list(self.rooms_missing),
            "nodes_present": list(self.nodes_present),
            "nodes_missing": list(self.nodes_missing),
            "connections_present": list(self.connections_present),
            "connections_missing": list(self.connections_missing),
            "connections_optional_missing": list(
                self.connections_optional_missing
            ),
            "result_count": len(self.results),
            "failed_required": [
                {
                    "kind": r.kind,
                    "name": r.name,
                    "detail": r.detail,
                }
                for r in self.results
                if r.required and not r.ok
            ],
        }


def _conn_key(origin: str, destination: str) -> str:
    return f"{origin} -> {destination}"


def _connection_exists(
    connections: Sequence[Z3Connection],
    origin: str,
    destination: str,
) -> bool:
    for conn in connections:
        if conn.origin == origin and conn.destination == destination:
            return True
        # Undirected physical doors often appear one-way in the file; accept
        # reverse match for presence checks only.
        if conn.origin == destination and conn.destination == origin:
            return True
    return False


def _node_in_room(room: Z3Room, node_name: str) -> bool:
    return any(n.name == node_name for n in room.nodes)


def validate_against_z3(data: Z3JsonData) -> CatalogValidation:
    """Check curated opening expectations against a loaded Z3JsonData."""
    results: list[CheckResult] = []
    rooms_present: list[str] = []
    rooms_missing: list[str] = []
    nodes_present: list[str] = []
    nodes_missing: list[str] = []
    connections_present: list[str] = []
    connections_missing: list[str] = []
    connections_optional_missing: list[str] = []

    room_index: dict[str, Z3Room] = {}
    for room in data.rooms:
        room_index.setdefault(room.name, room)

    expected_rooms: set[str] = set()
    expected_nodes: list[ExpectedNode] = []
    expected_conns: list[ExpectedConnection] = []
    for cp in opening_checkpoints():
        expected_rooms.update(cp.z3_rooms)
        expected_nodes.extend(cp.z3_nodes)
        expected_conns.extend(cp.z3_connections)

    # Always require the curated OPENING_ROUTE_ROOM_NAMES subset that the
    # loader already surfaces for the boot path.
    expected_rooms.update(OPENING_ROUTE_ROOM_NAMES)

    for name in sorted(expected_rooms):
        if name in room_index:
            rooms_present.append(name)
            results.append(
                CheckResult(
                    ok=True,
                    kind="room",
                    name=name,
                    required=True,
                    detail=room_index[name].source_path,
                )
            )
        else:
            rooms_missing.append(name)
            results.append(
                CheckResult(
                    ok=False,
                    kind="room",
                    name=name,
                    required=True,
                    detail="room not found in loaded regions",
                )
            )

    seen_nodes: set[tuple[str, str]] = set()
    for node in expected_nodes:
        key = (node.room_name, node.node_name)
        if key in seen_nodes:
            continue
        seen_nodes.add(key)
        label = f"{node.room_name}::{node.node_name}"
        room = room_index.get(node.room_name)
        if room is None:
            nodes_missing.append(label)
            results.append(
                CheckResult(
                    ok=False,
                    kind="node",
                    name=label,
                    required=node.required,
                    detail=f"room {node.room_name!r} missing",
                )
            )
            continue
        if _node_in_room(room, node.node_name):
            nodes_present.append(label)
            results.append(
                CheckResult(
                    ok=True,
                    kind="node",
                    name=label,
                    required=node.required,
                )
            )
        else:
            nodes_missing.append(label)
            results.append(
                CheckResult(
                    ok=False,
                    kind="node",
                    name=label,
                    required=node.required,
                    detail="node not present on room",
                )
            )

    seen_conns: set[tuple[str, str]] = set()
    for conn in expected_conns:
        key = (conn.origin, conn.destination)
        if key in seen_conns:
            continue
        seen_conns.add(key)
        label = _conn_key(conn.origin, conn.destination)
        found = _connection_exists(data.connections, conn.origin, conn.destination)
        if found:
            connections_present.append(label)
            results.append(
                CheckResult(
                    ok=True,
                    kind="connection",
                    name=label,
                    required=conn.required,
                    detail=conn.note,
                )
            )
        else:
            if conn.required:
                connections_missing.append(label)
            else:
                connections_optional_missing.append(label)
            results.append(
                CheckResult(
                    ok=False,
                    kind="connection",
                    name=label,
                    required=conn.required,
                    detail=conn.note or "connection not found",
                )
            )

    failed_required = [r for r in results if r.required and not r.ok]
    # Overall ok: required checks pass; optional missing is reported but OK.
    required_ok = not failed_required
    return CatalogValidation(
        ok=required_ok,
        required_ok=required_ok,
        results=results,
        rooms_present=rooms_present,
        rooms_missing=rooms_missing,
        nodes_present=nodes_present,
        nodes_missing=nodes_missing,
        connections_present=connections_present,
        connections_missing=connections_missing,
        connections_optional_missing=connections_optional_missing,
    )


def _checkpoint_public(cp: OpeningCheckpoint) -> dict[str, Any]:
    return {
        "id": cp.id,
        "label": cp.label,
        "role": cp.role,
        "gameplay": cp.gameplay,
        "z3_rooms": list(cp.z3_rooms),
        "z3_nodes": [
            {
                "room_name": n.room_name,
                "node_name": n.node_name,
                "required": n.required,
            }
            for n in cp.z3_nodes
        ],
        "z3_connections": [
            {
                "origin": c.origin,
                "destination": c.destination,
                "required": c.required,
                "note": c.note,
            }
            for c in cp.z3_connections
        ],
        "notes": cp.notes,
        "coordinate_claim": (
            "none — z3 names are logic associations only; "
            "gameplay fields use stable-retro RAM IDs"
        ),
    }


def correlate_boot_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only real observed milestone facts from a boot_to_castle report.

    Does not invent intermediate checkpoints the report never measured.
    """
    observed: list[dict[str, Any]] = []
    facts = {
        "phase": report.get("phase"),
        "frames": report.get("frames"),
        "game_mode": report.get("game_mode"),
        "submodule": report.get("submodule"),
        "screen_id": report.get("screen_id"),
        "screen_hex": report.get("screen_hex"),
        "indoors": report.get("indoors"),
        "dark_world": report.get("dark_world"),
        "link_x": report.get("link_x"),
        "link_y": report.get("link_y"),
        "has_control": report.get("has_control"),
        "on_castle_grounds": report.get("on_castle_grounds"),
    }

    on_grounds = bool(report.get("on_castle_grounds"))
    screen_id = report.get("screen_id")
    indoors = report.get("indoors")
    dark = report.get("dark_world")
    has_control = report.get("has_control")

    if (
        on_grounds
        and screen_id == HYRULE_CASTLE_SCREEN
        and indoors is False
        and dark is False
        and has_control is True
    ):
        observed.append(
            {
                "checkpoint_id": "hyrule_castle_grounds",
                "status": "observed_gameplay",
                "source": "boot_to_castle_report",
                "facts": facts,
            }
        )
    elif report.get("phase") or screen_id is not None:
        observed.append(
            {
                "checkpoint_id": "hyrule_castle_grounds",
                "status": "not_confirmed",
                "source": "boot_to_castle_report",
                "facts": facts,
                "reason": (
                    "report present but acceptance fields do not all match "
                    "castle-grounds goal "
                    f"(on_castle_grounds={on_grounds}, "
                    f"screen_id={screen_id!r}, indoors={indoors!r}, "
                    f"dark_world={dark!r}, has_control={has_control!r})"
                ),
            }
        )

    return {
        "kind": "boot_report_correlation",
        "note": (
            "Only milestones measured in the report are listed. Intermediate "
            "screens (0x2C/0x24/0x1C) are not inferred from a final snapshot."
        ),
        "observed_milestones": observed,
        "proven_gameplay": bool(
            observed
            and observed[0].get("status") == "observed_gameplay"
        ),
    }


def build_catalog_artifact(
    data: Z3JsonData,
    *,
    boot_report: Mapping[str, Any] | None = None,
    validation: CatalogValidation | None = None,
) -> dict[str, Any]:
    """Build the structured opening-route catalog / progress artifact."""
    validation = validation or validate_against_z3(data)
    opening_rooms = [
        {
            "id": r.id,
            "name": r.name,
            "room_type": r.room_type,
            "node_count": len(r.nodes),
            "source_path": r.source_path,
            "nodes": [
                {
                    "id": n.id,
                    "name": n.name,
                    "node_type": n.node_type,
                    "area": n.area,
                }
                for n in r.nodes
            ],
        }
        for r in data.opening_route_rooms()
    ]
    opening_conns = [
        {
            "connection_type": c.connection_type,
            "origin": c.origin,
            "destination": c.destination,
            "description": c.description,
        }
        for c in data.opening_route_connections()
    ]

    artifact: dict[str, Any] = {
        "kind": CATALOG_KIND,
        "version": CATALOG_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": DISCLAIMER,
        "goal": {
            "id": "links_house_to_castle_grounds",
            "description": (
                "Title → fresh file → Link's House exit → controllable on "
                "Hyrule Castle grounds (light-world screen 0x1B)."
            ),
            "gameplay_acceptance": {
                "indoors": False,
                "dark_world": False,
                "screen_id": HYRULE_CASTLE_SCREEN,
                "screen_hex": f"0x{HYRULE_CASTLE_SCREEN:02X}",
                "has_control": True,
                "on_castle_grounds": True,
            },
            "overworld_screen_path": list(OVERWORLD_SCREEN_PATH),
        },
        "z3_source": {
            "root": str(data.root),
            "revision": data.revision,
            "pin": Z3_JSON_DATA_PIN,
            "revision_matches_pin": data.revision == Z3_JSON_DATA_PIN
            if data.revision
            else None,
            "opening_room_count": len(opening_rooms),
            "opening_connection_count": len(opening_conns),
            "total_rooms": len(data.rooms),
            "total_connections": len(data.connections),
        },
        "checkpoints": [_checkpoint_public(cp) for cp in opening_checkpoints()],
        "z3_opening_rooms": opening_rooms,
        "z3_opening_connections": opening_conns,
        "validation": validation.summary,
        "observed": None,
        "metrics": {
            "checkpoint_count": len(opening_checkpoints()),
            "required_validation_ok": validation.required_ok,
            "rooms_present": len(validation.rooms_present),
            "rooms_missing": len(validation.rooms_missing),
            "nodes_present": len(validation.nodes_present),
            "nodes_missing": len(validation.nodes_missing),
            "connections_present": len(validation.connections_present),
            "connections_missing_required": len(validation.connections_missing),
            "connections_optional_missing": len(
                validation.connections_optional_missing
            ),
        },
    }
    if boot_report is not None:
        artifact["observed"] = correlate_boot_report(boot_report)
        artifact["metrics"]["proven_gameplay_from_boot_report"] = bool(
            artifact["observed"].get("proven_gameplay")
        )
    return artifact


def load_and_validate(
    root: Path | None = None,
) -> tuple[Z3JsonData, CatalogValidation]:
    """Load z3 data (actionable error if absent) and validate the catalog."""
    data = Z3JsonData.load(root)
    return data, validate_against_z3(data)


def write_artifact(artifact: Mapping[str, Any], path: Path) -> Path:
    """Write the catalog JSON artifact; create parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return path

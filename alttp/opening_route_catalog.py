"""Data-driven opening-route catalog: Link's House → Hyrule Castle grounds.

Maps the confirmed boot goal (title → fresh file → Link's House exit →
controllable on light-world screen ``0x1B``) onto local ``z3-json-data``
regions/connections for developer validation and progress artifacts.

**Authority split (do not collapse these):**

- Gameplay routing uses stable-retro RAM fields from ``alttp.ram`` /
  ``alttp.overworld`` (screen id, room id, indoors, control).
- z3 room/node names are randomizer-oriented **logic labels**. They are
  *associated* with route segments for naming and graph checks; they are
  **not** exact stable-retro screen coordinates or RAM screen IDs.

Never auto-downloads. Uses ``alttp.z3_json_data`` only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alttp.paths import (
    RECORDINGS_DIR,
    Z3_JSON_DATA_PIN,
)
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

CATALOG_KIND = "alttp_opening_route_catalog"
CATALOG_VERSION = 1
DEFAULT_ARTIFACT = RECORDINGS_DIR / "opening_route_catalog.json"

DISCLAIMER = (
    "z3 room/node names are logic labels from vg-json-data/z3-json-data; "
    "they are associated with route segments and are NOT exact stable-retro "
    "screen coordinates or RAM screen IDs. Gameplay proof requires RAM "
    "snapshots from a real boot/route run (alttp.ram / boot_to_castle)."
)

# Overworld screen path used by the scripted BFS (authoritative gameplay IDs).
# Labels are local; not z3 names.
OVERWORLD_SCREEN_PATH: tuple[dict[str, Any], ...] = (
    {
        "screen_id": LINKS_HOUSE_SCREEN,
        "screen_hex": f"0x{LINKS_HOUSE_SCREEN:02X}",
        "label": "links_house",
    },
    {
        "screen_id": 0x24,
        "screen_hex": "0x24",
        "label": "north_field",
    },
    {
        "screen_id": 0x1C,
        "screen_hex": "0x1C",
        "label": "castle_approach",
    },
    {
        "screen_id": HYRULE_CASTLE_SCREEN,
        "screen_hex": f"0x{HYRULE_CASTLE_SCREEN:02X}",
        "label": "hyrule_castle",
    },
)


@dataclass(frozen=True)
class ExpectedConnection:
    """A directed z3 connection we care about for the opening route."""

    origin: str
    destination: str
    required: bool = True
    note: str = ""


@dataclass(frozen=True)
class ExpectedNode:
    """A named door/item node expected inside a z3 room."""

    room_name: str
    node_name: str
    required: bool = True


@dataclass(frozen=True)
class OpeningCheckpoint:
    """One actionable checkpoint on Link's House → castle grounds.

    ``gameplay_*`` fields describe the stable-retro RAM acceptance for that
    segment when observed. ``z3_*`` fields are logic associations only.
    """

    id: str
    label: str
    role: str  # start | transit | goal | post_goal_context
    gameplay: dict[str, Any]
    z3_rooms: tuple[str, ...] = ()
    z3_nodes: tuple[ExpectedNode, ...] = ()
    z3_connections: tuple[ExpectedConnection, ...] = ()
    notes: str = ""


def opening_checkpoints() -> tuple[OpeningCheckpoint, ...]:
    """Return the curated Link's House → castle grounds checkpoint list."""
    return (
        OpeningCheckpoint(
            id="links_house_interior",
            label="Link's House (interior)",
            role="start",
            gameplay={
                "indoors": True,
                "room_base_id": LINKS_HOUSE_ROOM,
                "room_hex": f"0x{LINKS_HOUSE_ROOM:04X}",
                "dark_world": False,
            },
            z3_rooms=("Links House",),
            z3_nodes=(
                ExpectedNode("Links House", "Links House Exit"),
                ExpectedNode("Links House", "Link's House"),  # lamp item node
            ),
            # Pin has the cave room/nodes but no Door edge for the exit in
            # connections/main.json — keep optional so validate stays honest.
            z3_connections=(
                ExpectedConnection(
                    origin="Links House Exit",
                    destination="Light World",
                    required=False,
                    note=(
                        "Expected door edge for house exit; absent from "
                        "connections/main.json at the pinned revision "
                        "(nodes still exist on Links House / Light World)."
                    ),
                ),
            ),
            notes=(
                "Fresh-file spawn. z3 room 'Links House' is a cave/logic "
                "region, not screen 0x2C."
            ),
        ),
        OpeningCheckpoint(
            id="links_house_overworld",
            label="Link's House overworld porch",
            role="transit",
            gameplay={
                "indoors": False,
                "screen_id": LINKS_HOUSE_SCREEN,
                "screen_hex": f"0x{LINKS_HOUSE_SCREEN:02X}",
                "dark_world": False,
            },
            z3_rooms=("Light World", "Links House"),
            z3_nodes=(
                ExpectedNode("Light World", "Links House"),
            ),
            notes=(
                "stable-retro overworld screen 0x2C after house exit. "
                "Light World door node 'Links House' is a logic label, "
                "not a pixel coordinate."
            ),
        ),
        OpeningCheckpoint(
            id="overworld_to_castle",
            label="Overworld screens toward castle",
            role="transit",
            gameplay={
                "indoors": False,
                "dark_world": False,
                "screen_path": list(OVERWORLD_SCREEN_PATH),
            },
            z3_rooms=(),
            notes=(
                "Scripted BFS on the 8×8 light-world grid "
                "(alttp.overworld). No 1:1 z3 room per intermediate screen."
            ),
        ),
        OpeningCheckpoint(
            id="hyrule_castle_grounds",
            label="Hyrule Castle grounds (goal)",
            role="goal",
            gameplay={
                "indoors": False,
                "screen_id": HYRULE_CASTLE_SCREEN,
                "screen_hex": f"0x{HYRULE_CASTLE_SCREEN:02X}",
                "dark_world": False,
                "has_control": True,
                "on_castle_grounds": True,
            },
            # Courtyard is the nearest logic region for the castle exterior;
            # it is associated with screen 0x1B, not identical to it.
            z3_rooms=(
                "Hyrule Castle Courtyard",
                "Hyrule Castle Ledge",
            ),
            z3_nodes=(
                ExpectedNode(
                    "Hyrule Castle Courtyard",
                    "Hyrule Castle Entrance (South)",
                ),
                ExpectedNode(
                    "Hyrule Castle Courtyard",
                    "Hyrule Castle Secret Entrance Stairs",
                ),
                ExpectedNode("Light World", "Hyrule Castle Main Gate"),
            ),
            z3_connections=(
                ExpectedConnection(
                    origin="Hyrule Castle Main Gate",
                    destination="Hyrule Castle Courtyard",
                    required=True,
                    note="Logic edge for the courtyard / main gate.",
                ),
                ExpectedConnection(
                    origin="Hyrule Castle Entrance (South)",
                    destination="Hyrule Castle",
                    required=True,
                    note="South door into the castle interior (post-goal).",
                ),
            ),
            notes=(
                "Acceptance for boot_to_castle: controllable outdoors on "
                "light-world screen 0x1B. z3 'Hyrule Castle Courtyard' is "
                "an associated logic region name, not the screen id."
            ),
        ),
        OpeningCheckpoint(
            id="castle_interior_context",
            label="Hyrule Castle interior (context, not boot goal)",
            role="post_goal_context",
            gameplay={
                "indoors": True,
                "dark_world": False,
            },
            z3_rooms=(
                "Hyrule Castle",
                "Hyrule Castle Secret Entrance",
            ),
            z3_nodes=(
                ExpectedNode("Hyrule Castle", "Hyrule Castle Exit (South)"),
            ),
            z3_connections=(
                ExpectedConnection(
                    origin="Hyrule Castle Exit (South)",
                    destination="Light World",
                    required=True,
                ),
                ExpectedConnection(
                    origin="Hyrule Castle Secret Entrance Stairs",
                    destination="Hyrule Castle Secret Entrance",
                    required=True,
                    note="Uncle hole / secret entrance (next segment).",
                ),
            ),
            notes=(
                "Not required for boot_to_castle acceptance. Curated for "
                "the next uncle / fighter-sword experiment."
            ),
        ),
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


def _load_boot_report(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    text = Path(path).read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"boot report must be a JSON object: {path}")
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_status(args: argparse.Namespace) -> int:
    status = source_status(args.root)
    print(f"root:      {status.root}")
    print(f"present:   {status.present}")
    print(f"shape_ok:  {status.shape_ok}")
    print(f"revision:  {status.revision or '(unknown)'}")
    print(f"pin:       {status.pin}")
    if not status.present:
        print(
            "\nFetch with:\n"
            "  uv run python alttp/scripts/setup_z3_json_data.py"
        )
        return 1
    if not status.shape_ok:
        for issue in status.issues:
            print(f"  - {issue}")
        return 2
    try:
        data, validation = load_and_validate(args.root)
    except Z3JsonDataError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"catalog_required_ok: {validation.required_ok}")
    print(f"opening_rooms:       {len(data.opening_route_rooms())}")
    print(f"opening_connections: {len(data.opening_route_connections())}")
    if validation.connections_optional_missing:
        print(
            "optional_missing_connections: "
            + ", ".join(validation.connections_optional_missing)
        )
    return 0 if validation.required_ok else 3


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        data, validation = load_and_validate(args.root)
    except Z3JsonDataNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    except Z3JsonDataError as exc:
        print(exc, file=sys.stderr)
        return 2

    print(f"z3 root:   {data.root}")
    print(f"revision:  {data.revision or '(unknown)'}")
    print(f"pin:       {Z3_JSON_DATA_PIN}")
    print(f"required_ok: {validation.required_ok}")
    print(f"rooms:     present={len(validation.rooms_present)} "
          f"missing={len(validation.rooms_missing)}")
    print(f"nodes:     present={len(validation.nodes_present)} "
          f"missing={len(validation.nodes_missing)}")
    print(
        f"connections: present={len(validation.connections_present)} "
        f"missing_required={len(validation.connections_missing)} "
        f"optional_missing={len(validation.connections_optional_missing)}"
    )
    for r in validation.results:
        if r.ok:
            continue
        tag = "REQUIRED" if r.required else "optional"
        print(f"  [{tag}] {r.kind}: {r.name} — {r.detail}")
    if args.verbose:
        for r in validation.results:
            if r.ok:
                print(f"  [ok] {r.kind}: {r.name}")
    return 0 if validation.required_ok else 3


def _cmd_emit(args: argparse.Namespace) -> int:
    try:
        data, validation = load_and_validate(args.root)
    except Z3JsonDataNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    except Z3JsonDataError as exc:
        print(exc, file=sys.stderr)
        return 2

    try:
        boot_report = _load_boot_report(args.from_boot_report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"failed to load boot report: {exc}", file=sys.stderr)
        return 2

    artifact = build_catalog_artifact(
        data,
        boot_report=boot_report,
        validation=validation,
    )
    out = Path(args.out) if args.out else DEFAULT_ARTIFACT
    write_artifact(artifact, out)
    print(f"wrote {out}")
    print(
        json.dumps(
            {
                "required_ok": validation.required_ok,
                "metrics": artifact["metrics"],
                "observed": artifact.get("observed"),
            },
            indent=2,
        )
    )
    if args.require_ok and not validation.required_ok:
        return 3
    return 0


def _cmd_list_checkpoints(args: argparse.Namespace) -> int:
    for cp in opening_checkpoints():
        print(f"{cp.id:28s}  role={cp.role:18s}  {cp.label}")
        if args.verbose:
            print(f"  gameplay: {json.dumps(cp.gameplay)}")
            if cp.z3_rooms:
                print(f"  z3_rooms: {', '.join(cp.z3_rooms)}")
            for c in cp.z3_connections:
                req = "required" if c.required else "optional"
                print(f"  conn[{req}]: {c.origin} -> {c.destination}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m alttp.opening_route_catalog",
        description=(
            "Validate and emit the Link's House → castle opening-route "
            "catalog against local z3-json-data (no silent download)."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="override z3-json-data root",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser(
        "status", help="checkout + catalog required-check summary"
    )
    p_status.set_defaults(func=_cmd_status)

    p_val = sub.add_parser(
        "validate",
        help="validate expected opening rooms/nodes/connections against z3",
    )
    p_val.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="also print successful checks",
    )
    p_val.set_defaults(func=_cmd_validate)

    p_emit = sub.add_parser(
        "emit",
        help="write structured catalog/progress JSON under alttp/",
    )
    p_emit.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_ARTIFACT,
        help=f"output path (default: {DEFAULT_ARTIFACT})",
    )
    p_emit.add_argument(
        "--from-boot-report",
        type=Path,
        default=None,
        help=(
            "optional boot_to_castle.json to attach only real observed "
            "milestone facts (no invented intermediate screens)"
        ),
    )
    p_emit.add_argument(
        "--require-ok",
        action="store_true",
        help="exit non-zero if required z3 checks fail",
    )
    p_emit.set_defaults(func=_cmd_emit)

    p_list = sub.add_parser(
        "list-checkpoints", help="list curated opening checkpoints"
    )
    p_list.add_argument("-v", "--verbose", action="store_true")
    p_list.set_defaults(func=_cmd_list_checkpoints)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

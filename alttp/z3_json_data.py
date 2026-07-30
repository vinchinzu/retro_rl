"""Project-native loader for a local vg-json-data / z3-json-data checkout.

The upstream tree is **not** vendored. Fetch it explicitly with
``alttp/scripts/setup_z3_json_data.py`` into the gitignored path
``alttp/refs/z3-json-data``. Normal imports never download.

This module validates a small **source shape** (expected files/keys) and
exposes a typed subset useful for the title→castle opening route. It does
not claim full upstream schema validation or game-semantic correctness.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alttp.paths import (
    Z3_JSON_DATA_DIR,
    Z3_JSON_DATA_PIN,
    Z3_JSON_DATA_REPO,
)

# Rooms that appear on the boot / opening route (Link's House → castle).
OPENING_ROUTE_ROOM_NAMES: frozenset[str] = frozenset(
    {
        "Links House",
        "Light World",
        "Hyrule Castle Courtyard",
        "Hyrule Castle Ledge",
        "Hyrule Castle Secret Entrance",
        "Hyrule Castle",
    }
)

_REQUIRED_ROOT_FILES = (
    "items.json",
    "connections/main.json",
    "enemies/main.json",
)
_REQUIRED_ROOT_DIRS = (
    "regions",
    "schema",
    "connections",
    "enemies",
)


class Z3JsonDataError(RuntimeError):
    """Base error for z3-json-data access."""


class Z3JsonDataNotFoundError(Z3JsonDataError):
    """Raised when the local checkout is missing."""


class Z3JsonDataShapeError(Z3JsonDataError):
    """Raised when the checkout is present but fails shape checks."""


@dataclass(frozen=True)
class Z3Node:
    """A named node inside a room (door, item, drop, …)."""

    id: int
    name: str
    area: str
    node_type: str
    node_item: str | None = None
    node_address: str | None = None


@dataclass(frozen=True)
class Z3Room:
    """A region/room entry loaded from a regions/*.json file."""

    id: int
    name: str
    room_type: str
    nodes: tuple[Z3Node, ...]
    source_path: str


@dataclass(frozen=True)
class Z3Connection:
    """A physical connection between two named nodes."""

    connection_type: str
    description: str
    origin: str
    destination: str


@dataclass(frozen=True)
class Z3Enemy:
    """Enemy stats row from enemies/main.json."""

    id: int
    names: tuple[str, ...]
    hp: int | None = None


@dataclass(frozen=True)
class Z3ItemEntry:
    """A single named item (or progressive) with optional data blob."""

    name: str
    category: str
    data: str | None = None


@dataclass(frozen=True)
class Z3SourceStatus:
    """Presence / shape status for a local checkout."""

    root: Path
    present: bool
    shape_ok: bool
    revision: str | None
    pin: str
    issues: tuple[str, ...]


def default_data_root() -> Path:
    """Return the gitignored game-local checkout path."""
    return Z3_JSON_DATA_DIR


def resolve_data_root(root: Path | None = None) -> Path:
    """Resolve the data root; fail with an actionable message if missing."""
    path = Path(root) if root is not None else default_data_root()
    if not path.is_dir():
        raise Z3JsonDataNotFoundError(
            f"z3-json-data not found at {path}.\n"
            "Fetch it once (no silent download on import):\n"
            "  uv run python alttp/scripts/setup_z3_json_data.py\n"
            f"Upstream: {Z3_JSON_DATA_REPO}\n"
            f"Pinned revision: {Z3_JSON_DATA_PIN}\n"
            "Docs: alttp/docs/Z3_JSON_DATA.md"
        )
    return path.resolve()


def read_git_revision(root: Path) -> str | None:
    """Best-effort HEAD revision for a git checkout (None if unavailable)."""
    head = root / ".git" / "HEAD"
    if not head.is_file():
        # Worktree or bare checkout without .git as a dir — try .git file.
        git_file = root / ".git"
        if git_file.is_file():
            # Not resolving gitdir for simplicity; status can still report shape.
            return None
        return None
    try:
        text = head.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if text.startswith("ref:"):
        ref = text.split(":", 1)[1].strip()
        ref_path = root / ".git" / ref
        if ref_path.is_file():
            try:
                return ref_path.read_text(encoding="utf-8").strip()
            except OSError:
                return None
        return None
    return text or None


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Z3JsonDataShapeError(f"invalid JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise Z3JsonDataShapeError(f"cannot read {path}: {exc}") from exc


def discover_shape_issues(root: Path) -> list[str]:
    """Return human-readable issues for expected source layout (not full schema)."""
    issues: list[str] = []
    for rel in _REQUIRED_ROOT_DIRS:
        if not (root / rel).is_dir():
            issues.append(f"missing directory: {rel}/")
    for rel in _REQUIRED_ROOT_FILES:
        path = root / rel
        if not path.is_file():
            issues.append(f"missing file: {rel}")
            continue
        payload = _load_json(path)
        if rel == "items.json":
            if not isinstance(payload, dict):
                issues.append("items.json: expected a JSON object")
            else:
                for key in ("inventory", "progressives", "base"):
                    if key not in payload:
                        issues.append(f"items.json: missing key {key!r}")
        elif rel == "connections/main.json":
            if not isinstance(payload, dict):
                issues.append("connections/main.json: expected a JSON object")
            elif "connections" not in payload:
                issues.append("connections/main.json: missing key 'connections'")
            elif not isinstance(payload["connections"], list):
                issues.append("connections/main.json: 'connections' must be a list")
        elif rel == "enemies/main.json":
            if not isinstance(payload, dict):
                issues.append("enemies/main.json: expected a JSON object")
            elif "enemies" not in payload:
                issues.append("enemies/main.json: missing key 'enemies'")
            elif not isinstance(payload["enemies"], list):
                issues.append("enemies/main.json: 'enemies' must be a list")

    regions = root / "regions"
    if regions.is_dir():
        region_files = list(regions.rglob("*.json"))
        if not region_files:
            issues.append("regions/: no *.json files found")
        else:
            # Spot-check a few files for room payload shape.
            for path in sorted(region_files)[:5]:
                payload = _load_json(path)
                if not isinstance(payload, dict):
                    issues.append(f"{path.relative_to(root)}: expected a JSON object")
                    continue
                if "rooms" in payload:
                    if not isinstance(payload["rooms"], list):
                        issues.append(
                            f"{path.relative_to(root)}: 'rooms' must be a list"
                        )
                elif "name" not in payload or "id" not in payload:
                    issues.append(
                        f"{path.relative_to(root)}: expected 'rooms' list "
                        "or a single room with id/name"
                    )
    return issues


def validate_source_shape(root: Path | None = None) -> Path:
    """Ensure *root* exists and matches the expected layout; return resolved path."""
    path = resolve_data_root(root)
    issues = discover_shape_issues(path)
    if issues:
        joined = "\n  - ".join(issues)
        raise Z3JsonDataShapeError(
            f"z3-json-data at {path} failed shape checks:\n  - {joined}\n"
            "Re-run setup or inspect upstream layout. "
            "This check is structural only, not full schema validation."
        )
    return path


def source_status(root: Path | None = None) -> Z3SourceStatus:
    """Report presence, shape, and revision without raising on missing data."""
    path = Path(root) if root is not None else default_data_root()
    present = path.is_dir()
    issues: list[str] = []
    shape_ok = False
    revision: str | None = None
    if present:
        try:
            issues = discover_shape_issues(path)
            shape_ok = not issues
        except Z3JsonDataError as exc:
            issues = [str(exc)]
            shape_ok = False
        revision = read_git_revision(path)
    else:
        issues = ["checkout directory does not exist"]
    return Z3SourceStatus(
        root=path,
        present=present,
        shape_ok=shape_ok,
        revision=revision,
        pin=Z3_JSON_DATA_PIN,
        issues=tuple(issues),
    )


def _parse_node(raw: Mapping[str, Any]) -> Z3Node:
    return Z3Node(
        id=int(raw.get("id", -1)),
        name=str(raw.get("name", "")),
        area=str(raw.get("area", "")),
        node_type=str(raw.get("nodeType", "")),
        node_item=(
            str(raw["nodeItem"]) if raw.get("nodeItem") is not None else None
        ),
        node_address=(
            str(raw["nodeAddress"]) if raw.get("nodeAddress") is not None else None
        ),
    )


def _parse_room(raw: Mapping[str, Any], source_path: str) -> Z3Room:
    nodes_raw = raw.get("nodes") or []
    if not isinstance(nodes_raw, list):
        nodes_raw = []
    nodes = tuple(
        _parse_node(n) for n in nodes_raw if isinstance(n, Mapping)
    )
    return Z3Room(
        id=int(raw.get("id", -1)),
        name=str(raw.get("name", "")),
        room_type=str(raw.get("roomType", "")),
        nodes=nodes,
        source_path=source_path,
    )


def _iter_rooms_from_file(path: Path, root: Path) -> Iterable[Z3Room]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return
    rel = str(path.relative_to(root)).replace("\\", "/")
    if isinstance(payload.get("rooms"), list):
        for entry in payload["rooms"]:
            if isinstance(entry, Mapping) and "name" in entry:
                yield _parse_room(entry, rel)
    elif "name" in payload and "id" in payload:
        yield _parse_room(payload, rel)


def _parse_connection(raw: Mapping[str, Any]) -> Z3Connection | None:
    nodes = raw.get("nodes")
    if not isinstance(nodes, list) or len(nodes) < 2:
        return None
    origin = destination = ""
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        pos = str(node.get("position", "")).lower()
        name = str(node.get("name", ""))
        if pos == "origin":
            origin = name
        elif pos == "destination":
            destination = name
    if not origin and isinstance(nodes[0], Mapping):
        origin = str(nodes[0].get("name", ""))
    if not destination and isinstance(nodes[1], Mapping):
        destination = str(nodes[1].get("name", ""))
    return Z3Connection(
        connection_type=str(raw.get("connectionType", "")),
        description=str(raw.get("description", "")),
        origin=origin,
        destination=destination,
    )


def _parse_item_catalog(payload: Mapping[str, Any]) -> tuple[Z3ItemEntry, ...]:
    entries: list[Z3ItemEntry] = []
    for name in payload.get("base") or []:
        if isinstance(name, str):
            entries.append(Z3ItemEntry(name=name, category="base"))
    for category in (
        "inventory",
        "progressives",
        "ammo",
        "bottleContents",
        "currency",
        "drops",
        "dungeonItems",
        "dungeonPrizes",
        "expansions",
        "bosses",
        "flags",
        "settings",
    ):
        block = payload.get(category)
        if isinstance(block, Mapping):
            for name, meta in block.items():
                data = None
                if isinstance(meta, Mapping) and meta.get("data") is not None:
                    data = str(meta["data"])
                elif isinstance(meta, str):
                    data = meta
                entries.append(
                    Z3ItemEntry(name=str(name), category=category, data=data)
                )
        elif isinstance(block, list):
            for name in block:
                if isinstance(name, str):
                    entries.append(Z3ItemEntry(name=name, category=category))
    return tuple(entries)


@dataclass(frozen=True)
class Z3JsonData:
    """In-memory view of a useful subset of the local z3-json-data tree."""

    root: Path
    rooms: tuple[Z3Room, ...]
    connections: tuple[Z3Connection, ...]
    items: tuple[Z3ItemEntry, ...]
    enemies: tuple[Z3Enemy, ...]
    revision: str | None

    @classmethod
    def load(cls, root: Path | None = None, *, validate: bool = True) -> Z3JsonData:
        """Load data from *root* (default game-local path). Never downloads."""
        path = validate_source_shape(root) if validate else resolve_data_root(root)
        rooms = tuple(
            room
            for region_path in sorted((path / "regions").rglob("*.json"))
            for room in _iter_rooms_from_file(region_path, path)
        )
        conn_payload = _load_json(path / "connections" / "main.json")
        connections: list[Z3Connection] = []
        if isinstance(conn_payload, dict):
            for raw in conn_payload.get("connections") or []:
                if isinstance(raw, Mapping):
                    parsed = _parse_connection(raw)
                    if parsed is not None:
                        connections.append(parsed)
        items_payload = _load_json(path / "items.json")
        items = (
            _parse_item_catalog(items_payload)
            if isinstance(items_payload, Mapping)
            else ()
        )
        enemies_payload = _load_json(path / "enemies" / "main.json")
        enemies: list[Z3Enemy] = []
        if isinstance(enemies_payload, dict):
            for raw in enemies_payload.get("enemies") or []:
                if not isinstance(raw, Mapping):
                    continue
                names_raw = raw.get("names") or []
                names = (
                    tuple(str(n) for n in names_raw)
                    if isinstance(names_raw, list)
                    else (str(names_raw),)
                )
                hp = raw.get("hp")
                enemies.append(
                    Z3Enemy(
                        id=int(raw.get("id", -1)),
                        names=names,
                        hp=int(hp) if isinstance(hp, int) else None,
                    )
                )
        return cls(
            root=path,
            rooms=rooms,
            connections=tuple(connections),
            items=items,
            enemies=tuple(enemies),
            revision=read_git_revision(path),
        )

    def rooms_by_name(self, name: str, *, exact: bool = True) -> list[Z3Room]:
        """Return rooms matching *name* (exact or case-insensitive substring)."""
        if exact:
            return [r for r in self.rooms if r.name == name]
        needle = name.casefold()
        return [r for r in self.rooms if needle in r.name.casefold()]

    def room(self, name: str) -> Z3Room:
        """Return the unique exact-name room or raise KeyError."""
        matches = self.rooms_by_name(name, exact=True)
        if not matches:
            raise KeyError(f"no room named {name!r}")
        if len(matches) > 1:
            paths = ", ".join(m.source_path for m in matches)
            raise KeyError(f"ambiguous room name {name!r} in: {paths}")
        return matches[0]

    def find_rooms(self, query: str) -> list[Z3Room]:
        """Case-insensitive substring search over room names."""
        return self.rooms_by_name(query, exact=False)

    def find_connections(self, query: str) -> list[Z3Connection]:
        """Substring search over connection description and endpoint names."""
        needle = query.casefold()
        out: list[Z3Connection] = []
        for conn in self.connections:
            blob = f"{conn.description} {conn.origin} {conn.destination}".casefold()
            if needle in blob:
                out.append(conn)
        return out

    def find_items(self, query: str) -> list[Z3ItemEntry]:
        """Case-insensitive substring search over item names."""
        needle = query.casefold()
        return [i for i in self.items if needle in i.name.casefold()]

    def find_enemies(self, query: str) -> list[Z3Enemy]:
        """Case-insensitive substring search over enemy names."""
        needle = query.casefold()
        return [
            e
            for e in self.enemies
            if any(needle in n.casefold() for n in e.names)
        ]

    def opening_route_rooms(self) -> list[Z3Room]:
        """Rooms that matter for the title→castle opening route."""
        out: list[Z3Room] = []
        for name in sorted(OPENING_ROUTE_ROOM_NAMES):
            out.extend(self.rooms_by_name(name, exact=True))
        return out

    def opening_route_endpoint_names(self) -> set[str]:
        """Node/room names used to filter opening-route connections.

        The top-level ``Light World`` room is a hub with dozens of doors; only
        its Link/castle/sanctuary-related nodes are included so listings stay
        focused on the boot path.
        """
        names: set[str] = set()
        for room in self.opening_route_rooms():
            if room.name != "Light World":
                names.add(room.name)
            for node in room.nodes:
                if room.name == "Light World":
                    key = node.name.casefold()
                    if not any(
                        token in key
                        for token in (
                            "link",
                            "hyrule castle",
                            "sanctuary",
                        )
                    ):
                        continue
                names.add(node.name)
        return names

    def connections_for_room(self, room: Z3Room) -> list[Z3Connection]:
        """Connections whose endpoints match a node name (or the room name)."""
        node_names = {n.name for n in room.nodes} | {room.name}
        out: list[Z3Connection] = []
        for conn in self.connections:
            if conn.origin in node_names or conn.destination in node_names:
                out.append(conn)
        return out

    def opening_route_connections(self) -> list[Z3Connection]:
        """Connections touching opening-route endpoints (excludes LW noise)."""
        names = self.opening_route_endpoint_names()
        out: list[Z3Connection] = []
        seen: set[tuple[str, str, str]] = set()
        for conn in self.connections:
            if conn.origin in names or conn.destination in names:
                key = (conn.connection_type, conn.origin, conn.destination)
                if key not in seen:
                    seen.add(key)
                    out.append(conn)
        return out


# ---------------------------------------------------------------------------
# CLI (also ``python -m alttp.z3_json_data``)
# ---------------------------------------------------------------------------


def _print_status(status: Z3SourceStatus) -> int:
    print(f"root:      {status.root}")
    print(f"present:   {status.present}")
    print(f"shape_ok:  {status.shape_ok}")
    print(f"revision:  {status.revision or '(unknown)'}")
    print(f"pin:       {status.pin}")
    if status.revision and status.revision != status.pin:
        print("note:      working tree revision differs from pin")
    if status.issues:
        print("issues:")
        for issue in status.issues:
            print(f"  - {issue}")
    if not status.present:
        print(
            "\nFetch with:\n"
            "  uv run python alttp/scripts/setup_z3_json_data.py"
        )
        return 1
    if not status.shape_ok:
        return 2
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    return _print_status(source_status(args.root))


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        path = validate_source_shape(args.root)
    except Z3JsonDataNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    except Z3JsonDataShapeError as exc:
        print(exc, file=sys.stderr)
        return 2
    rev = read_git_revision(path)
    print(f"OK: shape checks passed for {path}")
    print(f"revision: {rev or '(unknown)'}")
    print(f"pin:      {Z3_JSON_DATA_PIN}")
    return 0


def _load_or_exit(root: Path | None) -> Z3JsonData:
    try:
        return Z3JsonData.load(root)
    except Z3JsonDataError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc


def _cmd_list_regions(args: argparse.Namespace) -> int:
    data = _load_or_exit(args.root)
    rooms: Sequence[Z3Room]
    if args.opening:
        rooms = data.opening_route_rooms()
    elif args.query:
        rooms = data.find_rooms(args.query)
    else:
        rooms = data.rooms
    for room in rooms:
        print(
            f"{room.id:4d}  {room.room_type:12s}  {room.name}  "
            f"({len(room.nodes)} nodes)  [{room.source_path}]"
        )
    print(f"# {len(rooms)} room(s)", file=sys.stderr)
    return 0


def _cmd_list_connections(args: argparse.Namespace) -> int:
    data = _load_or_exit(args.root)
    if args.query:
        conns = data.find_connections(args.query)
    elif args.opening:
        conns = data.opening_route_connections()
    else:
        conns = list(data.connections)
    for conn in conns:
        print(
            f"{conn.connection_type:8s}  {conn.origin}  ->  {conn.destination}"
        )
        if args.verbose and conn.description:
            print(f"         {conn.description}")
    print(f"# {len(conns)} connection(s)", file=sys.stderr)
    return 0


def _cmd_list_items(args: argparse.Namespace) -> int:
    data = _load_or_exit(args.root)
    items = data.find_items(args.query) if args.query else list(data.items)
    for item in items:
        data_s = f"  data={item.data}" if item.data else ""
        print(f"{item.category:16s}  {item.name}{data_s}")
    print(f"# {len(items)} item(s)", file=sys.stderr)
    return 0


def _cmd_list_enemies(args: argparse.Namespace) -> int:
    data = _load_or_exit(args.root)
    enemies = data.find_enemies(args.query) if args.query else list(data.enemies)
    for enemy in enemies:
        names = ", ".join(enemy.names)
        hp = f"  hp={enemy.hp}" if enemy.hp is not None else ""
        print(f"{enemy.id:4d}  {names}{hp}")
    print(f"# {len(enemies)} enemy/ies", file=sys.stderr)
    return 0


def _cmd_show_room(args: argparse.Namespace) -> int:
    data = _load_or_exit(args.root)
    rooms = data.rooms_by_name(args.name, exact=not args.fuzzy)
    if not rooms and not args.fuzzy:
        rooms = data.find_rooms(args.name)
    if not rooms:
        print(f"no room matching {args.name!r}", file=sys.stderr)
        return 1
    for room in rooms:
        print(f"id:        {room.id}")
        print(f"name:      {room.name}")
        print(f"type:      {room.room_type}")
        print(f"source:    {room.source_path}")
        print(f"nodes ({len(room.nodes)}):")
        for node in room.nodes:
            extra = ""
            if node.node_item:
                extra += f" item={node.node_item}"
            if node.node_address:
                extra += f" addr={node.node_address}"
            print(
                f"  {node.id:3d}  {node.node_type:8s}  {node.name}  "
                f"[{node.area}]{extra}"
            )
        conns = data.connections_for_room(room)
        if conns:
            print(f"connections ({len(conns)}):")
            for conn in conns:
                print(
                    f"  {conn.connection_type:8s}  "
                    f"{conn.origin}  ->  {conn.destination}"
                )
        print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m alttp.z3_json_data",
        description=(
            "Inspect a local vg-json-data/z3-json-data checkout "
            "(no ROM/emulator required)."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=f"override data root (default: {Z3_JSON_DATA_DIR})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="report checkout presence and shape")
    p_status.set_defaults(func=_cmd_status)

    p_validate = sub.add_parser(
        "validate", help="exit non-zero if missing or shape-invalid"
    )
    p_validate.set_defaults(func=_cmd_validate)

    p_regions = sub.add_parser("list-regions", help="list rooms/regions")
    p_regions.add_argument("-q", "--query", help="substring filter on room name")
    p_regions.add_argument(
        "--opening",
        action="store_true",
        help="only rooms used by the title→castle opening route",
    )
    p_regions.set_defaults(func=_cmd_list_regions)

    p_conn = sub.add_parser("list-connections", help="list connections")
    p_conn.add_argument("-q", "--query", help="substring filter")
    p_conn.add_argument(
        "--opening",
        action="store_true",
        help="connections touching opening-route rooms",
    )
    p_conn.add_argument("-v", "--verbose", action="store_true")
    p_conn.set_defaults(func=_cmd_list_connections)

    p_items = sub.add_parser("list-items", help="list item catalog entries")
    p_items.add_argument("-q", "--query", help="substring filter on item name")
    p_items.set_defaults(func=_cmd_list_items)

    p_enemies = sub.add_parser("list-enemies", help="list enemies")
    p_enemies.add_argument("-q", "--query", help="substring filter on enemy name")
    p_enemies.set_defaults(func=_cmd_list_enemies)

    p_show = sub.add_parser("show-room", help="show one room and its connections")
    p_show.add_argument("name", help="exact room name (or substring with --fuzzy)")
    p_show.add_argument(
        "--fuzzy",
        action="store_true",
        help="substring match instead of exact name",
    )
    p_show.set_defaults(func=_cmd_show_room)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

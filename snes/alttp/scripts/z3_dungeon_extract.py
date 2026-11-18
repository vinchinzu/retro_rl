#!/usr/bin/env python3
"""Extract per-dungeon regions, locations, requirements, and connections.

Reads the local z3-json-data pin (refs checkout when present, else the
committed dump). Never downloads. Writes ``alttp/dungeon/route_data/<slug>.yaml``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from alttp.paths import (
    GAME_DIR,
    Z3_JSON_DATA_DIR,
    Z3_JSON_DATA_PIN,
    Z3_JSON_DATA_WORKSPACE_DIR,
)
from alttp.z3_json_data import read_git_revision, resolve_data_root

ROUTE_DATA_DIR = GAME_DIR / "dungeon" / "route_data"
SCHEMA_VERSION = 1

# Slug → region file. ``castle`` is the escape dungeon (Hyrule Castle through
# Sanctuary), not the courtyard. One file holds every logic area for that dungeon.
DUNGEON_FILES: dict[str, str] = {
    "castle": "regions/dungeons/escape/main.json",
    "eastern": "regions/dungeons/eastern/main.json",
    "desert": "regions/dungeons/desert/main.json",
    "hera": "regions/dungeons/hera/main.json",
    "castle_tower": "regions/dungeons/castletower/main.json",
    "pod": "regions/dungeons/pod/main.json",
    "swamp": "regions/dungeons/swamp/main.json",
    "skull": "regions/dungeons/skwoods/main.json",
    "thieves": "regions/dungeons/thieves/main.json",
    "ice": "regions/dungeons/ice/main.json",
    "mire": "regions/dungeons/mire/main.json",
    "turtle_rock": "regions/dungeons/trock/main.json",
    "gt": "regions/dungeons/gt/main.json",
}

# ``count`` is the max of ``KeyX`` (1) and ``KeyX:N`` on that node, not a sum.
_KEY_TOKEN = re.compile(
    r"^(?P<name>(?:BigKey|Key)[A-Za-z0-9]*)(?::(?P<count>\d+))?$"
)
_STRAT_KEYS = (
    "name",
    "link",
    "collectsItems",
    "setsFlags",
    "requires",
    "yields",
    "id",
    "note",
    "devNote",
)


def route_path(slug: str) -> Path:
    """Path of the YAML extract for *slug*."""
    if slug not in DUNGEON_FILES:
        raise KeyError(f"unknown dungeon slug {slug!r}")
    return ROUTE_DATA_DIR / f"{slug}.yaml"


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def _root_kind(root: Path) -> str:
    resolved = root.resolve()
    if resolved == Z3_JSON_DATA_DIR.resolve():
        return "refs"
    if resolved == Z3_JSON_DATA_WORKSPACE_DIR.resolve():
        return "dump"
    return "other"


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)


def _address(raw: Any) -> str | list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return str(raw)


def _classify(requires: Any) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Split a requires tree into key entries, item tokens, and raw key strings.

    Raw strings are returned only when stripping whitespace changes the token
    (upstream ``BigKeyD3 `` on the Skull Woods big chest).
    """
    keys: list[dict[str, Any]] = []
    items: list[str] = []
    dirty: list[str] = []
    seen_keys: set[tuple[str, int, str]] = set()
    seen_items: set[str] = set()
    for raw in _strings(requires):
        stripped = raw.strip()
        if not stripped:
            continue
        match = _KEY_TOKEN.match(stripped)
        if match:
            name = match.group("name")
            count = int(match.group("count") or "1")
            entry: dict[str, Any] = {"name": name, "count": count}
            if raw != stripped:
                entry["raw"] = raw
                dirty.append(raw)
            ident = (name, count, raw)
            if ident not in seen_keys:
                seen_keys.add(ident)
                keys.append(entry)
        elif stripped not in seen_items:
            seen_items.add(stripped)
            items.append(stripped)
    return keys, items, dirty


def _node_doc(raw: Mapping[str, Any], room_name: str) -> tuple[dict[str, Any], list[str]]:
    requires = raw.get("requires") or []
    if not isinstance(requires, list):
        requires = []
    keys, items, dirty = _classify(requires)
    region = str(raw.get("area") or room_name)
    name = str(raw.get("name", ""))
    doc: dict[str, Any] = {
        "id": int(raw.get("id", -1)),
        "name": name,
        "region": region,
        "node_type": str(raw.get("nodeType", "")),
    }
    if raw.get("nodeItem") is not None:
        doc["item"] = str(raw["nodeItem"])
    address = _address(raw.get("nodeAddress"))
    if address is not None:
        doc["address"] = address
    doc["requires"] = requires
    doc["keys"] = keys
    doc["items"] = items
    if raw.get("note"):
        doc["note"] = str(raw["note"])
    for extra in ("yields", "locks", "view"):
        if raw.get(extra) is not None:
            doc[extra] = raw[extra]
    notes = [
        f"{name}: requirement token {token!r} normalizes to {token.strip()!r}"
        for token in dirty
    ]
    return doc, notes


def _regions(nodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    order: list[str] = []
    grouped: dict[str, list[str]] = {}
    for node in nodes:
        region = str(node["region"])
        if region not in grouped:
            order.append(region)
            grouped[region] = []
        grouped[region].append(str(node["name"]))
    return [{"name": region, "nodes": grouped[region]} for region in order]


def _rollup(nodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Max key count per name, plus every non-key string token, across nodes."""
    key_count: dict[str, int] = {}
    key_nodes: dict[str, list[str]] = {}
    item_nodes: dict[str, list[str]] = {}
    for node in nodes:
        node_name = str(node["name"])
        for key in node["keys"]:
            name = str(key["name"])
            key_count[name] = max(key_count.get(name, 0), int(key["count"]))
            names = key_nodes.setdefault(name, [])
            if node_name not in names:
                names.append(node_name)
        for item in node["items"]:
            names = item_nodes.setdefault(str(item), [])
            if node_name not in names:
                names.append(node_name)
    return {
        "keys": [
            {"name": name, "count": key_count[name], "nodes": key_nodes[name]}
            for name in sorted(key_count)
        ],
        "items": [
            {"name": name, "nodes": item_nodes[name]}
            for name in sorted(item_nodes)
        ],
    }


def _strat(raw: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in _STRAT_KEYS:
        if key in raw:
            out[key] = raw[key]
    for key, value in raw.items():
        if key not in out:
            out[key] = value
    return out


def _connections(payload: Mapping[str, Any], names: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for raw in payload.get("connections") or []:
        if not isinstance(raw, Mapping):
            continue
        ends: dict[str, str] = {}
        for node in raw.get("nodes") or []:
            if not isinstance(node, Mapping):
                continue
            pos = str(node.get("position", "")).lower()
            ends[pos] = str(node.get("name", ""))
        origin = ends.get("origin", "")
        destination = ends.get("destination", "")
        if origin not in names and destination not in names:
            continue
        row = {
            "connection_type": str(raw.get("connectionType", "")),
            "description": str(raw.get("description", "")),
            "origin": origin,
            "destination": destination,
            "internal": origin in names and destination in names,
        }
        ident = (
            row["connection_type"],
            row["description"],
            row["origin"],
            row["destination"],
        )
        if ident in seen:
            continue
        seen.add(ident)
        rows.append(row)
    return rows


def build_dungeon(slug: str, root: Path | None = None) -> dict[str, Any]:
    """Build one dungeon document from the local z3-json-data tree."""
    if slug not in DUNGEON_FILES:
        raise KeyError(f"unknown dungeon slug {slug!r}")
    data_root = resolve_data_root(root)
    rel = DUNGEON_FILES[slug]
    room = _load_json(data_root / rel)
    if not isinstance(room, Mapping):
        raise ValueError(f"{rel} is not a JSON object")
    room_name = str(room.get("name", ""))
    raw_nodes = room.get("nodes") or []
    if not isinstance(raw_nodes, list):
        raise ValueError(f"{rel} nodes is not a list")
    nodes: list[dict[str, Any]] = []
    notes: list[str] = []
    for raw in raw_nodes:
        if not isinstance(raw, Mapping):
            continue
        doc, dirty = _node_doc(raw, room_name)
        nodes.append(doc)
        notes.extend(dirty)
    names = {room_name}
    names.update(str(node["name"]) for node in nodes)
    names.update(str(node["region"]) for node in nodes)
    conn_payload = _load_json(data_root / "connections" / "main.json")
    if not isinstance(conn_payload, Mapping):
        raise ValueError("connections/main.json is not a JSON object")
    strats_raw = room.get("strats") or []
    strats = [
        _strat(raw)
        for raw in strats_raw
        if isinstance(raw, Mapping)
    ]
    doc: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "slug": slug,
        "name": room_name,
        "z3_id": int(room.get("id", -1)),
        "room_type": str(room.get("roomType", "")),
        "source_file": rel,
        "data_root": _root_kind(data_root),
        "pin": Z3_JSON_DATA_PIN,
    }
    revision = read_git_revision(data_root)
    if revision:
        doc["revision"] = revision
    if room.get("note"):
        doc["note"] = str(room["note"])
    doc["regions"] = _regions(nodes)
    doc["locations"] = [node for node in nodes if node["node_type"] == "item"]
    doc["gates"] = [node for node in nodes if node["node_type"] != "item"]
    doc["requirements"] = _rollup(nodes)
    doc["connections"] = _connections(conn_payload, names)
    doc["strats"] = strats
    doc["notes"] = notes
    return doc


def dump_route(doc: Mapping[str, Any]) -> str:
    """Serialize *doc* to YAML that round-trips through ``safe_load``."""
    text = yaml.safe_dump(
        doc,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=4096,
    )
    if not text.endswith("\n"):
        text += "\n"
    return text


def load_route(slug: str) -> dict[str, Any]:
    """Load one written dungeon YAML."""
    path = route_path(slug)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not load as a mapping")
    return payload


def write_routes(root: Path | None = None) -> list[Path]:
    """Write all 13 dungeon YAML files. Raises if a file would not round-trip."""
    ROUTE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for slug in DUNGEON_FILES:
        doc = build_dungeon(slug, root)
        text = dump_route(doc)
        loaded = yaml.safe_load(text)
        if loaded != doc:
            raise RuntimeError(f"YAML round-trip mismatch for {slug}")
        path = route_path(slug)
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written


def check_routes(root: Path | None = None) -> list[str]:
    """Return problems if a YAML is missing or differs from a fresh extract."""
    problems: list[str] = []
    if len(DUNGEON_FILES) != 13:
        problems.append(f"expected 13 slugs, found {len(DUNGEON_FILES)}")
    for slug in DUNGEON_FILES:
        path = route_path(slug)
        if not path.is_file():
            problems.append(f"missing {path.name}")
            continue
        try:
            loaded = load_route(slug)
            fresh = build_dungeon(slug, root)
        except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
            problems.append(f"{slug}: {exc}")
            continue
        if loaded != fresh:
            problems.append(f"drift {slug}")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Load each YAML and compare it to a fresh extract.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="z3-json-data root (default: refs checkout, else committed dump).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.check:
        problems = check_routes(args.root)
        if problems:
            print("\n".join(problems), file=sys.stderr)
            return 1
        print(f"OK: {len(DUNGEON_FILES)} dungeon route files")
        return 0
    for path in write_routes(args.root):
        print(path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

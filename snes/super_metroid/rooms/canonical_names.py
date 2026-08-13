"""Map Rando / sm-json-data canonical room names.

Map Rando logic (https://maprando.com/logic) is built on sm-json-data.
This package already vendors that corpus at ``refs/sm-json-data/``.

Two identifiers matter:

- **maprandoId** — integer on ``/logic/room/<id>`` pages (sm-json-data ``id``).
- **roomId** — SNES room pointer low 16 bits (``roomAddress & 0xFFFF``), what
  RAM ``room_id`` reports and what continuous/human traces store.

Prefer these names over ad-hoc labels. Rebuild the on-disk index with::

    uv run python snes/super_metroid/scripts/export/maprando_catalog.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from super_metroid.paths import GAME_DIR, MAPS_DIR

SM_JSON_DATA_ROOT = GAME_DIR / "refs" / "sm-json-data"
MAPRANDO_CATALOG_PATH = MAPS_DIR / "maprando_room_catalog.json"
MAPRANDO_NAMES_PATH = MAPS_DIR / "maprando_room_names.json"
MAPRANDO_LOGIC_URL = "https://maprando.com/logic"


@dataclass(frozen=True)
class CanonicalRoom:
    """One vanilla room as named by Map Rando / sm-json-data."""

    maprando_id: int
    name: str
    area: str | None
    subarea: str | None
    room_id: int | None
    room_id_hex: str | None
    room_address: str | None = None
    ref_path: str | None = None
    node_count: int = 0
    item_node_count: int = 0

    @property
    def logic_url(self) -> str:
        return f"{MAPRANDO_LOGIC_URL}/room/{self.maprando_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "maprandoId": self.maprando_id,
            "name": self.name,
            "area": self.area,
            "subarea": self.subarea,
            "roomId": self.room_id,
            "roomIdHex": self.room_id_hex,
            "roomAddress": self.room_address,
            "refPath": self.ref_path,
            "nodeCount": self.node_count,
            "itemNodeCount": self.item_node_count,
            "logicUrl": self.logic_url,
        }


def _room_ptr(room_address: object) -> int | None:
    if not isinstance(room_address, str) or not room_address:
        return None
    try:
        return int(room_address, 0) & 0xFFFF
    except ValueError:
        return None


def parse_rooms_from_sm_json_data(
    reference_root: Path | None = None,
) -> list[CanonicalRoom]:
    """Parse every room JSON under ``region/`` (no roomDiagrams)."""
    root = (reference_root or SM_JSON_DATA_ROOT).expanduser().resolve()
    region = root / "region"
    if not region.is_dir():
        raise FileNotFoundError(f"sm-json-data region dir missing: {region}")

    rooms: list[CanonicalRoom] = []
    for path in sorted(region.rglob("*.json")):
        if "roomDiagrams" in path.parts or path.name == "roomDiagrams.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        name = data.get("name")
        mid = data.get("id")
        if name is None or mid is None:
            continue
        addr = data.get("roomAddress")
        room_ptr = _room_ptr(addr)
        nodes = data.get("nodes") or []
        item_nodes = 0
        if isinstance(nodes, list):
            for node in nodes:
                if isinstance(node, dict) and str(node.get("nodeType", "")).lower() == "item":
                    item_nodes += 1
        rooms.append(
            CanonicalRoom(
                maprando_id=int(mid),
                name=str(name),
                area=str(data["area"]) if data.get("area") is not None else None,
                subarea=str(data["subarea"]) if data.get("subarea") is not None else None,
                room_id=room_ptr,
                room_id_hex=f"0x{room_ptr:04X}" if room_ptr is not None else None,
                room_address=str(addr) if addr is not None else None,
                ref_path=str(path.relative_to(root)),
                node_count=len(nodes) if isinstance(nodes, list) else 0,
                item_node_count=item_nodes,
            )
        )
    rooms.sort(key=lambda r: (r.area or "", r.maprando_id))
    return rooms


def build_catalog_payload(
    reference_root: Path | None = None,
) -> dict[str, Any]:
    """Full catalog dict suitable for ``maprando_room_catalog.json``."""
    root = (reference_root or SM_JSON_DATA_ROOT).expanduser().resolve()
    rooms = parse_rooms_from_sm_json_data(root)
    items_path = root / "items.json"
    upgrades: list[Any] = []
    expansions: list[Any] = []
    flags: list[Any] = []
    if items_path.is_file():
        items = json.loads(items_path.read_text(encoding="utf-8"))
        if isinstance(items, dict):
            upgrades = list(items.get("upgradeItems") or [])
            expansions = list(items.get("expansionItems") or [])
            flags = list(items.get("gameFlags") or [])
    area_counts: dict[str, int] = {}
    for room in rooms:
        key = room.area or "Unknown"
        area_counts[key] = area_counts.get(key, 0) + 1
    return {
        "schemaVersion": 1,
        "source": {
            "maprandoLogicUrl": MAPRANDO_LOGIC_URL,
            "smJsonData": "refs/sm-json-data",
            "note": (
                "Canonical names match Map Rando / sm-json-data. "
                "maprandoId is the /logic/room/<id> index; roomId is SNES room "
                "pointer low 16 bits (RAM room_id)."
            ),
        },
        "summary": {
            "roomCount": len(rooms),
            "areaCounts": dict(sorted(area_counts.items())),
            "upgradeItemCount": len(upgrades),
            "expansionItemCount": len(expansions),
            "gameFlagCount": len(flags),
        },
        "rooms": [
            {
                "maprandoId": r.maprando_id,
                "name": r.name,
                "area": r.area,
                "subarea": r.subarea,
                "roomAddress": r.room_address,
                "roomId": r.room_id,
                "roomIdHex": r.room_id_hex,
                "refPath": r.ref_path,
                "nodeCount": r.node_count,
                "itemNodeCount": r.item_node_count,
            }
            for r in rooms
        ],
        "upgradeItems": upgrades,
        "expansionItems": expansions,
        "gameFlags": flags,
    }


def build_names_index(catalog: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Compact lookup tables: byRoomIdHex / byMaprandoId / byName."""
    if catalog is None:
        catalog = build_catalog_payload()
    rooms = catalog.get("rooms") or []
    by_hex: dict[str, dict[str, Any]] = {}
    by_mid: dict[str, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    for row in rooms:
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("name") or "")
        mid = row.get("maprandoId")
        hex_id = row.get("roomIdHex")
        area = row.get("area")
        if not name or mid is None:
            continue
        entry_hex = {
            "name": name,
            "maprandoId": int(mid),
            "area": area,
        }
        entry_mid = {
            "name": name,
            "roomIdHex": hex_id,
            "area": area,
        }
        entry_name = {
            "maprandoId": int(mid),
            "roomIdHex": hex_id,
            "area": area,
        }
        if isinstance(hex_id, str) and hex_id:
            by_hex[hex_id] = entry_hex
        by_mid[str(int(mid))] = entry_mid
        by_name[name] = entry_name
    return {
        "schemaVersion": 1,
        "byRoomIdHex": by_hex,
        "byMaprandoId": by_mid,
        "byName": by_name,
    }


def write_catalog(
    *,
    catalog_path: Path | None = None,
    names_path: Path | None = None,
    reference_root: Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Rebuild catalog + names index on disk. Returns paths and catalog."""
    catalog_path = catalog_path or MAPRANDO_CATALOG_PATH
    names_path = names_path or MAPRANDO_NAMES_PATH
    catalog = build_catalog_payload(reference_root)
    names = build_names_index(catalog)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    names_path.write_text(json.dumps(names, indent=2) + "\n", encoding="utf-8")
    return catalog_path, names_path, catalog


@lru_cache(maxsize=4)
def _load_names_file(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def load_canonical_names(
    *,
    names_path: Path | None = None,
    catalog_path: Path | None = None,
    prefer_rebuild: bool = False,
) -> dict[int, str]:
    """room_id (SNES ptr) → Map Rando name.

    Prefers the compact names index; falls back to full catalog or live parse.
    """
    if prefer_rebuild:
        write_catalog(catalog_path=catalog_path, names_path=names_path)

    path = names_path or MAPRANDO_NAMES_PATH
    data = _load_names_file(str(path.resolve())) if path.is_file() else {}
    by_hex = data.get("byRoomIdHex") if data else None
    out: dict[int, str] = {}
    if isinstance(by_hex, dict):
        for hex_id, row in by_hex.items():
            try:
                rid = int(str(hex_id), 0)
            except ValueError:
                continue
            if isinstance(row, Mapping) and row.get("name"):
                out[rid] = str(row["name"])
        if out:
            return out

    cat_path = catalog_path or MAPRANDO_CATALOG_PATH
    if cat_path.is_file():
        try:
            catalog = json.loads(cat_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            catalog = None
        if isinstance(catalog, dict):
            for row in catalog.get("rooms") or []:
                if not isinstance(row, Mapping):
                    continue
                rid = row.get("roomId")
                name = row.get("name")
                if rid is not None and name:
                    out[int(rid)] = str(name)
            if out:
                return out

    # Last resort: parse sm-json-data directly (no cache write).
    try:
        for room in parse_rooms_from_sm_json_data():
            if room.room_id is not None:
                out[room.room_id] = room.name
    except FileNotFoundError:
        pass
    return out


def load_canonical_rooms(
    *,
    catalog_path: Path | None = None,
) -> list[CanonicalRoom]:
    """Load full room records from the on-disk catalog (or rebuild from source)."""
    path = catalog_path or MAPRANDO_CATALOG_PATH
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = None
        if isinstance(data, dict):
            rooms: list[CanonicalRoom] = []
            for row in data.get("rooms") or []:
                if not isinstance(row, Mapping):
                    continue
                mid = row.get("maprandoId")
                name = row.get("name")
                if mid is None or not name:
                    continue
                rid = row.get("roomId")
                rooms.append(
                    CanonicalRoom(
                        maprando_id=int(mid),
                        name=str(name),
                        area=str(row["area"]) if row.get("area") is not None else None,
                        subarea=(
                            str(row["subarea"]) if row.get("subarea") is not None else None
                        ),
                        room_id=int(rid) if rid is not None else None,
                        room_id_hex=(
                            str(row["roomIdHex"])
                            if row.get("roomIdHex") is not None
                            else (
                                f"0x{int(rid):04X}" if rid is not None else None
                            )
                        ),
                        room_address=(
                            str(row["roomAddress"])
                            if row.get("roomAddress") is not None
                            else None
                        ),
                        ref_path=str(row["refPath"]) if row.get("refPath") else None,
                        node_count=int(row.get("nodeCount") or 0),
                        item_node_count=int(row.get("itemNodeCount") or 0),
                    )
                )
            if rooms:
                return rooms
    return parse_rooms_from_sm_json_data()


def room_name(room_id: int, *, names: Mapping[int, str] | None = None) -> str:
    """Lookup helper; falls back to hex if unknown."""
    table = names if names is not None else load_canonical_names()
    return table.get(int(room_id), f"0x{int(room_id):04X}")


def clear_name_cache() -> None:
    """Drop LRU cache (tests / after rebuild)."""
    _load_names_file.cache_clear()

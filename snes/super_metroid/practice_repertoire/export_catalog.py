"""Build ``maps/practice_repertoire.json`` from sm_practice_hack menus.

One catalog core: menu/data parsers + parent-chain resolve live here.
ROM pointers are attached by ``data_label`` (word-hash / ``snes``), not a
5-tuple fingerprint. Teleports stay in ``source_extract``.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path

from super_metroid.paths import (
    PRACTICE_REPERTOIRE_PATH,
    SHARED_PRACTICE_ROM,
    VANILLA_ROM_SHA1,
)
from super_metroid.practice_repertoire.rom_map import (
    join_blobs_by_label,
    walk_preset_blobs,
    word_hash,
)
from super_metroid.practice_repertoire.source_extract import (
    parse_teleports,
    validate_category_order,
)

UPSTREAM_COMMIT = "181c76b1a5e6e86eef6e1b1e9ba82c8a6c38e1f6"
RAW_ROOT = (
    "https://raw.githubusercontent.com/tewtal/sm_practice_hack/"
    f"{UPSTREAM_COMMIT}/src"
)
RAW = f"{RAW_ROOT}/presets"
DEFAULT_CACHE = Path("/tmp/sm_practice/presets")

CATEGORIES = [
    ("kpdr20", "Any% KPDR 20%", "kpdr20"),
    ("kpdr21", "Any% KPDR 21%", "kpdr21"),
    ("kpdr22", "Any% KPDR 22%", "kpdr22"),
    ("kpdr23", "Any% KPDR 23%", "kpdr23"),
    ("kpdr25", "Any% KPDR - Early Ice", "kpdr25"),
    ("prkd19", "Any% PRKD 19%", "prkd19"),
    ("prkd20", "Any% PRKD 20%", "prkd20"),
    ("pkrd", "Any% PKRD", "pkrd"),
    ("gtclassic", "GT Classic", "gtclassic"),
    ("gtmax", "GT Max%", "gtmax"),
    ("100early", "100% Early Ice", "100early"),
    ("hundo", "100%", "hundo"),
    ("100map", "100% Map", "100map"),
    ("spazermap", "Spazer Map", "spazermap"),
    ("14ice", "14% Ice", "14ice"),
    ("14speed", "14% Speed", "14speed"),
    ("rbo", "RBO", "rbo"),
    ("suitless", "Suitless", "suitless"),
    ("ngplasma", "NG+ Plasma", "ngplasma"),
    ("nghyper", "NG+ Hyper", "nghyper"),
    ("nintendopower", "Nintendo Power", "nintendopower"),
    ("allbosskpdr", "All Bosses KPDR", "allbosskpdr"),
    ("allbosspkdr", "All Bosses PKDR", "allbosspkdr"),
    ("allbossprkd", "All Bosses PRKD", "allbossprkd"),
    ("nodropskpdr", "No Drops KPDR", "nodropskpdr"),
    ("rando", "Rando", "rando"),
]

CM_HEADER = re.compile(r'%cm_header\("([^"]+)"\)')
CM_SUBMENU = re.compile(r'%cm_submenu\("([^"]+)",\s*#(\w+)\)')
CM_PRESET = re.compile(
    r'^(presets_[A-Za-z0-9_]+):\s*\n\s*%cm_preset\("([^"]+)"',
    re.MULTILINE,
)
LABEL_START = re.compile(r"^(preset_[A-Za-z0-9_]+):\s*$", re.MULTILINE)
WORD_WRITE_RE = re.compile(
    r"^\s*dw\s+\$([0-9A-Fa-f]{4}),\s*\$([0-9A-Fa-f]{4})(?:\s*;\s*(.*))?$",
    re.MULTILINE,
)

CORE_FIELDS = {
    0x078D: "ddb",
    0x079B: "room_id",
    0x09A2: "items_equipped",
    0x09A4: "items_collected",
    0x09A6: "beams_equipped",
    0x09A8: "beams_collected",
    0x09C0: "reserve_mode",
    0x09C2: "health",
    0x09C4: "max_health",
    0x09C6: "missiles",
    0x09C8: "max_missiles",
    0x09CA: "supers",
    0x09CC: "max_supers",
    0x09CE: "power_bombs",
    0x09D0: "max_power_bombs",
    0x09D2: "selected_item",
    0x09D4: "max_reserves",
    0x09D6: "reserves",
    0x0A1C: "pose",
    0x0A1E: "pose_direction",
    0x0AF6: "x",
    0x0AF8: "x_subpixel",
    0x0AFA: "y",
    0x0AFC: "y_subpixel",
}


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mainmenu_path(cache: Path) -> Path:
    sibling = cache.parent / "mainmenu.asm"
    return sibling if sibling.is_file() else cache / "mainmenu.asm"


def ensure_cache(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    need = []
    for _, _, stem in CATEGORIES:
        for suffix in ("_menu.asm", "_data.asm"):
            path = cache / f"{stem}{suffix}"
            if not path.is_file() or path.stat().st_size < 50:
                need.append(f"{stem}{suffix}")
    mainmenu = _mainmenu_path(cache)
    if not mainmenu.is_file() or mainmenu.stat().st_size < 50:
        need.append("mainmenu.asm")
    if need:
        print(f"fetching {len(need)} practice-hack source files…")
    for name in need:
        url = f"{RAW_ROOT}/{name}" if name == "mainmenu.asm" else f"{RAW}/{name}"
        dest = cache / name
        print(f"  {name}")
        urllib.request.urlretrieve(url, dest)


def parse_data_file(path: Path) -> dict[str, dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    labels = list(LABEL_START.finditer(text))
    if not labels:
        raise ValueError(f"no preset data labels in {path}")
    raw: dict[str, dict] = {}
    for i, m in enumerate(labels):
        name = m.group(1)
        start = m.end()
        end = labels[i + 1].start() if i + 1 < len(labels) else len(text)
        body = text[start:end]
        first = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
        if name in raw:
            raise ValueError(f"duplicate preset data label {name} in {path}")
        pm = re.match(r"dw\s+#(preset_[A-Za-z0-9_]+)", first)
        root = re.match(r"dw\s+#\$0000", first)
        if not pm and not root:
            raise ValueError(f"{name}: expected parent or root marker, got {first!r}")
        if not re.search(r"^\s*dw\s+#\$FFFF\s*$", body, re.MULTILINE):
            raise ValueError(f"{name}: missing $FFFF terminator")
        pairs: list[tuple[int, int]] = []
        overrides: dict[int, int] = {}
        for write in WORD_WRITE_RE.finditer(body):
            address = int(write.group(1), 16)
            value = int(write.group(2), 16)
            pairs.append((address, value))
            overrides[address] = value
        raw[name] = {
            "parent": pm.group(1) if pm else None,
            "overrides": overrides,
            "pairs": pairs,
        }

    resolved: dict[str, dict[int, int]] = {}

    def resolve(name: str, chain: tuple[str, ...] = ()) -> dict[int, int]:
        if name in resolved:
            return resolved[name]
        if name not in raw:
            raise ValueError(f"missing preset parent {name} in {path}")
        if name in chain:
            raise ValueError(f"preset parent cycle: {' -> '.join((*chain, name))}")
        node = raw[name]
        out = dict(resolve(node["parent"], (*chain, name))) if node["parent"] else {}
        out.update(node["overrides"])
        resolved[name] = out
        return out

    out: dict[str, dict] = {}
    for name, definition in raw.items():
        words = resolve(name)
        fields = {
            field_name: words.get(address)
            for address, field_name in CORE_FIELDS.items()
        }
        out[name] = {
            "parent": definition["parent"],
            "overrides": definition["overrides"],
            "pairs": definition["pairs"],
            "words": words,
            **fields,
        }
    return out


def parse_menu_file(
    path: Path, cat_id: str
) -> tuple[str, list[dict], list[dict]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    headers = CM_HEADER.findall(text)
    cat_title = headers[0] if headers else cat_id

    area_order: list[tuple[str, str]] = []
    prefix = f"presets_submenu_{cat_id}_"
    for display, target in CM_SUBMENU.findall(text):
        if target.startswith(prefix):
            area_order.append((target[len(prefix) :], display))

    leaf_prefix = f"presets_{cat_id}_"
    sessions: list[dict] = []
    area_indices = {area_id: index for index, (area_id, _) in enumerate(area_order)}
    preset_indices: dict[str, int] = {}
    for m in CM_PRESET.finditer(text):
        full_label, display = m.groups()
        if not full_label.startswith(leaf_prefix):
            continue
        rest = full_label[len(leaf_prefix) :]
        area = None
        slug = ""
        for ak, _ in sorted(area_order, key=lambda t: -len(t[0])):
            if rest == ak:
                area, slug = ak, ""
                break
            if rest.startswith(ak + "_"):
                area = ak
                slug = rest[len(ak) + 1 :]
                break
        if area is None:
            parts = rest.split("_", 1)
            area = parts[0]
            slug = parts[1] if len(parts) > 1 else ""
        data_label = "preset_" + full_label[len("presets_") :]
        preset_index = preset_indices.get(area, 0)
        preset_indices[area] = preset_index + 1
        sessions.append(
            {
                "id": f"{cat_id}/{area}/{slug}",
                "kind": "category_preset",
                "category": cat_id,
                "area": area,
                "slug": slug,
                "name": display,
                "menu_label": full_label,
                "data_label": data_label,
                "area_index": area_indices.get(area, -1),
                "preset_index": preset_index,
                "browse_index": len(sessions),
            }
        )

    area_sessions: dict[str, list[str]] = {ak: [] for ak, _ in area_order}
    for s in sessions:
        area_sessions.setdefault(s["area"], []).append(s["id"])
    areas = [
        {
            "id": ak,
            "name": disp,
            "session_count": len(area_sessions.get(ak, [])),
        }
        for ak, disp in area_order
    ]
    return cat_title, areas, sessions


def _source_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _attach_rom_pointers(sessions: list[dict], rom: bytes) -> None:
    blobs = walk_preset_blobs(rom)
    found, _missing = join_blobs_by_label(sessions, blobs)
    for session in sessions:
        row = found.get(str(session.get("data_label") or ""))
        if row is None:
            continue
        session["snes"] = row["snes"]
        session["snes_hex"] = row["snes_hex"]
        session["rom_offset"] = row["offset"]


def build_catalog(
    cache: Path,
    *,
    mainmenu: Path | None = None,
    practice_rom: Path = SHARED_PRACTICE_ROM,
    source_commit: str = UPSTREAM_COMMIT,
) -> dict:
    menu_source = mainmenu or _mainmenu_path(cache)
    if not menu_source.is_file():
        raise FileNotFoundError(f"practice mainmenu source missing: {menu_source}")
    if not practice_rom.is_file():
        raise FileNotFoundError(
            f"practice ROM missing: {practice_rom}\n"
            "Run setup_practice_rom.py; it is required to resolve the 52 "
            "Save Stations load-table records."
        )
    validate_category_order(menu_source, [stem for _, _, stem in CATEGORIES])
    categories: list[dict] = []
    all_sessions: list[dict] = []
    source_paths = [menu_source]
    for category_index, (cat_id, default_title, stem) in enumerate(CATEGORIES):
        menu = cache / f"{stem}_menu.asm"
        data = cache / f"{stem}_data.asm"
        if not menu.is_file() or not data.is_file():
            raise FileNotFoundError(f"category sources missing: {menu}, {data}")
        source_paths.extend((menu, data))
        title, areas, sessions = parse_menu_file(menu, cat_id)
        definitions = parse_data_file(data)
        ids_by_label = {session["data_label"]: session["id"] for session in sessions}
        if set(ids_by_label) != set(definitions):
            missing_data = sorted(set(ids_by_label) - set(definitions))
            missing_menu = sorted(set(definitions) - set(ids_by_label))
            raise ValueError(
                f"{cat_id}: menu/data mismatch missing_data={missing_data[:5]} "
                f"missing_menu={missing_menu[:5]}"
            )
        for s in sessions:
            definition = definitions[s["data_label"]]
            words = definition["words"]
            s["category_index"] = category_index
            parent_label = definition["parent"]
            s["parent_id"] = ids_by_label[parent_label] if parent_label else None
            s["override_words"] = {
                f"0x{address:04X}": value
                for address, value in sorted(definition["overrides"].items())
            }
            s["effective_word_count"] = len(words)
            s["effective_state_sha256"] = word_hash(words)
            for field_name in CORE_FIELDS.values():
                value = definition.get(field_name)
                if value is not None:
                    s[field_name] = value
            s["items"] = s["items_collected"]
            s["beams"] = s["beams_collected"]
            s["room_hex"] = f"0x{s['room_id']:04X}"
            s["canonical_state"] = f"practice_repertoire/{s['id']}.state"
            s["canonical_demo"] = f"recordings/practice_repertoire/{s['id']}"
            all_sessions.append(s)
        categories.append(
            {
                "id": cat_id,
                "name": title or default_title,
                "menu_index": len(categories),
                "areas": areas,
                "session_count": len(sessions),
            }
        )
        print(f"{cat_id}: {len(sessions)} sessions, {len(areas)} areas")

    _attach_rom_pointers(all_sessions, practice_rom.read_bytes())
    teleports = parse_teleports(menu_source, practice_rom)

    return {
        "schema_version": 2,
        "source": "https://github.com/tewtal/sm_practice_hack",
        "source_commit": source_commit,
        "source_sha256": _source_digest(source_paths),
        "patch_site": "https://smpractice.speedga.me/",
        "vanilla_rom": "roms/SuperMetroid.sfc",
        "vanilla_sha1": VANILLA_ROM_SHA1,
        "practice_rom": "roms/SuperMetroid_Practice.sfc",
        "practice_rom_sha1": _sha1(practice_rom),
        "practice_rom_tinystates": "roms/SuperMetroid_Practice_tinystates.sfc",
        "practice_rom_size": 4_194_304,
        "ips_emulator": "https://smpractice.speedga.me/patches/emulator-ntsc.ips",
        "ips_tinystates": "https://smpractice.speedga.me/patches/tinystates-ntsc.ips",
        "product_category": "kpdr25",
        "note": (
            "Practice-only training starts, not main-spine evidence. Category "
            "presets store compact exact inherited WRAM definitions. Teleports "
            "have exact destinations but preserve caller state. canonical_state "
            "is an artifact target, not proof that a file exists."
        ),
        "exactness": {
            "category_presets": (
                "Exact practice-hack preset writes before loader-derived room/enemy/PLM "
                "initialization; default category adjustments and preset options required."
            ),
            "teleports": (
                "Exact load-station destination only; inventory/progression/pose are "
                "parameterized by the caller."
            ),
            "runtime_state": (
                "A bootable exact state requires selecting the action in the pinned ROM "
                "and capturing after the loader returns to game state 8."
            ),
        },
        "counts": {
            "categories": len(categories),
            "category_areas": sum(len(category["areas"]) for category in categories),
            "category_presets": len(all_sessions),
            "teleports": len(teleports),
            "static_starts": len(all_sessions) + len(teleports),
        },
        "categories": categories,
        "sessions": all_sessions,
        "teleports": teleports,
    }


def write_catalog(catalog: dict, path: Path = PRACTICE_REPERTOIRE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = [
    "CATEGORIES",
    "CORE_FIELDS",
    "DEFAULT_CACHE",
    "UPSTREAM_COMMIT",
    "build_catalog",
    "ensure_cache",
    "parse_data_file",
    "parse_menu_file",
    "word_hash",
    "write_catalog",
]

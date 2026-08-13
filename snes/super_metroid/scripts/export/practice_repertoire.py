#!/usr/bin/env python3
"""Regenerate ``maps/practice_repertoire.json`` from sm_practice_hack menus.

Fetches preset ``*_menu.asm`` / ``*_data.asm`` from GitHub (or uses ``--cache``)
and rebuilds the full practice-hack repertoire catalog used by
``super_metroid.practice_repertoire``.

```bash
uv run python snes/super_metroid/scripts/export/practice_repertoire.py
uv run python snes/super_metroid/scripts/export/practice_repertoire.py --cache /tmp/sm_practice/presets
```
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[3]
for _p in (_REPO_ROOT, _SNES_IMPORT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from super_metroid.paths import (  # noqa: E402
    PRACTICE_REPERTOIRE_PATH,
    VANILLA_ROM_SHA1,
)

GITHUB_API = "https://api.github.com/repos/tewtal/sm_practice_hack/contents/src/presets"
RAW = "https://raw.githubusercontent.com/tewtal/sm_practice_hack/master/src/presets"

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
MDB_RE = re.compile(r"dw\s+\$079B,\s*\$([0-9A-Fa-f]{4})\s*;\s*MDB")
ITEMS_EQ_RE = re.compile(r"dw\s+\$09A2,\s*\$([0-9A-Fa-f]{4})\s*;\s*Equipped Items")
ITEMS_COL_RE = re.compile(r"dw\s+\$09A4,\s*\$([0-9A-Fa-f]{4})\s*;\s*Collected Items")
BEAMS_EQ_RE = re.compile(r"dw\s+\$09A6,\s*\$([0-9A-Fa-f]{4})\s*;\s*Equipped Beams")
BEAMS_COL_RE = re.compile(r"dw\s+\$09A8,\s*\$([0-9A-Fa-f]{4})\s*;\s*Collected Beams")
SAMUS_X_RE = re.compile(r"dw\s+\$0AF6,\s*\$([0-9A-Fa-f]{4})\s*;\s*Samus X")
SAMUS_Y_RE = re.compile(r"dw\s+\$0AFA,\s*\$([0-9A-Fa-f]{4})\s*;\s*Samus Y")
POSE_RE = re.compile(r"dw\s+\$0A1C,\s*\$([0-9A-Fa-f]{4})\s*;\s*Samus position/state")
LABEL_START = re.compile(r"^(preset_[A-Za-z0-9_]+):\s*$", re.MULTILINE)


def _ensure_cache(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    need = []
    for _, _, stem in CATEGORIES:
        for suffix in ("_menu.asm", "_data.asm"):
            path = cache / f"{stem}{suffix}"
            if not path.is_file() or path.stat().st_size < 50:
                need.append(f"{stem}{suffix}")
    if not need:
        return
    print(f"fetching {len(need)} preset asm files…")
    for name in need:
        url = f"{RAW}/{name}"
        dest = cache / name
        print(f"  {name}")
        urllib.request.urlretrieve(url, dest)


def parse_data_file(path: Path) -> dict[str, dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    labels = list(LABEL_START.finditer(text))
    raw: dict[str, dict] = {}
    for i, m in enumerate(labels):
        name = m.group(1)
        start = m.end()
        end = labels[i + 1].start() if i + 1 < len(labels) else len(text)
        body = text[start:end]
        first = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
        parent = None
        pm = re.match(r"dw\s+#(preset_[A-Za-z0-9_]+)", first)
        if pm:
            parent = pm.group(1)

        def grab(rx: re.Pattern[str], src: str = body) -> int | None:
            mm = rx.search(src)
            return int(mm.group(1), 16) if mm else None

        raw[name] = {
            "parent": parent,
            "room_id": grab(MDB_RE),
            "items_equipped": grab(ITEMS_EQ_RE),
            "items_collected": grab(ITEMS_COL_RE),
            "beams_equipped": grab(BEAMS_EQ_RE),
            "beams_collected": grab(BEAMS_COL_RE),
            "x": grab(SAMUS_X_RE),
            "y": grab(SAMUS_Y_RE),
            "pose": grab(POSE_RE),
        }

    def resolve(name: str, seen: set[str] | None = None) -> dict:
        if seen is None:
            seen = set()
        if name in seen or name not in raw:
            return {}
        seen.add(name)
        node = raw[name]
        base = resolve(node["parent"], seen) if node.get("parent") else {}
        out = dict(base)
        for k, v in node.items():
            if k != "parent" and v is not None:
                out[k] = v
        return out

    return {name: resolve(name) for name in raw}


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
        sessions.append(
            {
                "id": f"{cat_id}/{area}/{slug}",
                "category": cat_id,
                "area": area,
                "slug": slug,
                "name": display,
                "menu_label": full_label,
                "data_label": data_label,
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


def build_catalog(cache: Path) -> dict:
    categories: list[dict] = []
    all_sessions: list[dict] = []
    for cat_id, default_title, stem in CATEGORIES:
        menu = cache / f"{stem}_menu.asm"
        data = cache / f"{stem}_data.asm"
        if not menu.is_file():
            print(f"skip missing {menu.name}", file=sys.stderr)
            continue
        title, areas, sessions = parse_menu_file(menu, cat_id)
        fingerprints = parse_data_file(data) if data.is_file() else {}
        for s in sessions:
            fp = fingerprints.get(s["data_label"], {})
            if fp.get("room_id") is not None:
                s["room_id"] = fp["room_id"]
                s["room_hex"] = f"0x{fp['room_id']:04X}"
            items = fp.get("items_collected")
            if items is None:
                items = fp.get("items_equipped")
            if items is not None:
                s["items"] = items
            beams = fp.get("beams_collected")
            if beams is None:
                beams = fp.get("beams_equipped")
            if beams is not None:
                s["beams"] = beams
            for k in ("x", "y", "pose"):
                if fp.get(k) is not None:
                    s[k] = fp[k]
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

    return {
        "source": "https://github.com/tewtal/sm_practice_hack",
        "patch_site": "https://smpractice.speedga.me/",
        "vanilla_rom": "roms/SuperMetroid.sfc",
        "vanilla_sha1": VANILLA_ROM_SHA1,
        "practice_rom": "roms/SuperMetroid_Practice.sfc",
        "practice_rom_tinystates": "roms/SuperMetroid_Practice_tinystates.sfc",
        "practice_rom_size": 4_194_304,
        "ips_emulator": "https://smpractice.speedga.me/patches/emulator-ntsc.ips",
        "ips_tinystates": "https://smpractice.speedga.me/patches/tinystates-ntsc.ips",
        "product_category": "kpdr25",
        "note": (
            "Repertoire = practice-hack preset menus/saves. "
            "canonical_state/demo are harness targets; practice-ROM WRAM "
            "presets are not drop-in vanilla save states."
        ),
        "categories": categories,
        "sessions": all_sessions,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cache",
        type=Path,
        default=Path("/tmp/sm_practice/presets"),
        help="directory of *_menu.asm / *_data.asm",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=PRACTICE_REPERTOIRE_PATH,
        help="output JSON path",
    )
    p.add_argument(
        "--no-fetch",
        action="store_true",
        help="do not download missing asm files",
    )
    args = p.parse_args(argv)
    if not args.no_fetch:
        _ensure_cache(args.cache)
    catalog = build_catalog(args.cache)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {args.out} sessions={len(catalog['sessions'])} "
        f"cats={len(catalog['categories'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

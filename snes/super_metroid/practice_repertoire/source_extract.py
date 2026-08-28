"""Strict source/ROM extraction helpers for the practice start manifest."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

CATEGORY_BLOCK_RE = re.compile(
    r"preset_category_submenus:\s*\{(?P<body>.*?)dw\s+#\$0000",
    re.DOTALL,
)
CATEGORY_ENTRY_RE = re.compile(r"dw\s+#PresetsMenu([A-Za-z0-9]+)")
TELEPORT_RE = re.compile(
    r"^(tel_[A-Za-z0-9_]+):\s*\n\s*"
    r'%cm_jsl\("([^"]+)",\s*#action_teleport,\s*#\$([0-9A-Fa-f]{4})\)',
    re.MULTILINE,
)

TELEPORT_AREAS = {
    0: ("crateria", "Crateria"),
    1: ("brinstar", "Brinstar"),
    2: ("norfair", "Norfair"),
    3: ("wrecked_ship", "Wrecked Ship"),
    4: ("maridia", "Maridia"),
    5: ("tourian", "Tourian"),
    6: ("ceres", "Ceres"),
}


def _slug(value: str) -> str:
    value = value.lower().replace("debug ", "")
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def validate_category_order(mainmenu: Path, expected_stems: Sequence[str]) -> None:
    """Fail closed when upstream adds, removes, or reorders a category."""

    text = mainmenu.read_text(encoding="utf-8", errors="replace")
    match = CATEGORY_BLOCK_RE.search(text)
    if not match:
        raise ValueError(f"preset_category_submenus missing from {mainmenu}")
    actual = [value.lower() for value in CATEGORY_ENTRY_RE.findall(match.group("body"))]
    expected = [value.lower() for value in expected_stems]
    if actual != expected:
        raise ValueError(
            "category inventory drifted from exporter:\n"
            f"  source={actual}\n"
            f"  exporter={expected}"
        )


def parse_teleports(mainmenu: Path, practice_rom: Path) -> list[dict]:
    """Extract all Save Stations actions and their fixed destination records.

    A selector ``0xAAII`` supplies load-station area ``AA`` and index ``II``.
    The patched ROM retains the vanilla seven-word load-station record table.
    Teleports preserve the caller's progression, so the resulting starts are
    explicitly parameterized even though destination room/DDB/position is exact.
    """

    text = mainmenu.read_text(encoding="utf-8", errors="replace")
    rom = practice_rom.read_bytes()

    def le16(offset: int) -> int:
        if offset < 0 or offset + 2 > len(rom):
            raise ValueError(f"load-station ROM offset out of range: 0x{offset:X}")
        return int.from_bytes(rom[offset : offset + 2], "little")

    rows: list[dict] = []
    area_counts: dict[int, int] = {}
    for menu_label, display_name, selector_hex in TELEPORT_RE.findall(text):
        selector = int(selector_hex, 16)
        area_index = selector >> 8
        station_index = selector & 0xFF
        if area_index not in TELEPORT_AREAS:
            raise ValueError(f"{menu_label}: unknown teleport area {area_index}")
        area_id, area_name = TELEPORT_AREAS[area_index]
        pointer = le16(0x44B5 + 2 * area_index)
        record_offset = (pointer & 0x7FFF) + 14 * station_index
        words = [le16(record_offset + 2 * index) for index in range(7)]
        room_id, ddb, door_bts, screen_x, screen_y, y_offset, x_offset = words
        signed_x_offset = x_offset if x_offset < 0x8000 else x_offset - 0x10000
        menu_index = area_counts.get(area_index, 0)
        area_counts[area_index] = menu_index + 1
        rows.append(
            {
                "id": f"teleport/{area_id}/{_slug(display_name)}",
                "kind": "teleport",
                "area": area_id,
                "area_name": area_name,
                "name": display_name,
                "menu_label": menu_label,
                "area_index": area_index,
                "menu_index": menu_index,
                "selector": selector,
                "selector_hex": f"0x{selector:04X}",
                "station_index": station_index,
                "room_id": room_id,
                "room_hex": f"0x{room_id:04X}",
                "ddb": ddb,
                "ddb_hex": f"0x{ddb:04X}",
                "door_bts": door_bts,
                "x": (screen_x + 0x80 + signed_x_offset) & 0xFFFF,
                "y": (screen_y + y_offset) & 0xFFFF,
                "parameterized_state": True,
                "preserves": [
                    "inventory",
                    "ammo",
                    "events",
                    "bosses",
                    "doors",
                    "pose",
                    "subpixels",
                ],
            }
        )
    if len(rows) != 52:
        raise ValueError(f"expected 52 Save Stations actions, found {len(rows)}")
    return rows


__all__ = ["parse_teleports", "validate_category_order"]

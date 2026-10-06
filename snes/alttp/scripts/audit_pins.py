"""Audit every save state under ``custom_integrations/Zelda3-Snes/``.

One subprocess per pin (stable-retro keeps one emulator per process).
A pin is incoherent when HP exceeds max HP, max HP is not a multiple of 8,
sword/shield/armor leave the vanilla range, the module is not play, or the
room does not belong to the dungeon id. Prints provenance kind. Exit 1
when any pin is incoherent.

    SDL_VIDEODRIVER=dummy uv run python alttp/scripts/audit_pins.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from retro_harness.env import integration_dir, state_path

from alttp.paths import GAME, GAME_DIR
from alttp.spine.tape import read_pin_provenance

# $040C is the palace index times 2, or 0xFF off a palace (overworld / cave).
# Room groups are the usdasm rooms.asm section headers (vanilla USA).
_PALACE_ROOMS: dict[int, tuple[int, ...]] = {
    0x00: (0x12, 0x02, 0x11, 0x21, 0x22, 0x32, 0x42, 0x41),
    0x02: (
        0x51, 0x50, 0x01, 0x52, 0x60, 0x61, 0x62,
        0x72, 0x82, 0x81, 0x71, 0x70, 0x80,
    ),
    0x04: (
        0xC8, 0xD8, 0xD9, 0xDA, 0x99, 0xA8, 0xA9, 0xAA,
        0xB8, 0xB9, 0xBA, 0xC9, 0x89,
    ),
    0x06: (0x33, 0x43, 0x53, 0x63, 0x73, 0x74, 0x75, 0x83, 0x84, 0x85),
    0x08: (0x30, 0x40, 0xB0, 0xC0, 0xD0, 0xE0, 0x0D),
    0x0A: (
        0x28, 0x38, 0x37, 0x36, 0x35, 0x34, 0x54,
        0x46, 0x26, 0x76, 0x66, 0x16, 0x06,
    ),
    0x0C: (
        0x4A, 0x3A, 0x2A, 0x1A, 0x0A, 0x6A, 0x5A,
        0x4B, 0x3B, 0x2B, 0x1B, 0x0B, 0x19, 0x09,
    ),
    0x0E: (
        0x98, 0xD2, 0xC2, 0xC1, 0xD1, 0x97, 0xB1, 0xB2, 0xC3,
        0xB3, 0xA3, 0xA2, 0xA1, 0x93, 0x92, 0x91, 0xA0, 0x90,
    ),
    0x10: (0x56, 0x57, 0x58, 0x67, 0x68, 0x59, 0x49, 0x39, 0x29),
    0x12: (
        0x0E, 0x1E, 0x1F, 0x2E, 0x3E, 0x3F, 0x4E, 0x5E, 0x5F,
        0x6E, 0x7E, 0x7F, 0x8E, 0x9E, 0x9F, 0xAE, 0xAF, 0xBE,
        0x4F, 0xCE, 0xDE,
    ),
    0x14: (0x07, 0x17, 0x27, 0x31, 0x77, 0x87, 0xA7),
    0x16: (
        0xDB, 0xDC, 0xCB, 0xCC, 0xBB, 0xBC,
        0xAB, 0xAC, 0x64, 0x65, 0x44, 0x45,
    ),
    0x18: (
        0xB6, 0xB7, 0xC6, 0xC7, 0xD6, 0x04, 0x13, 0x14, 0x15,
        0x24, 0xA4, 0xB4, 0xB5, 0xC4, 0xC5, 0x23, 0xD5,
    ),
    0x1A: (
        0x0C, 0x7B, 0x7C, 0x7D, 0x8B, 0x8C, 0x8D, 0x9B, 0x9C,
        0x9D, 0x1C, 0x6B, 0x5B, 0x5C, 0x5D, 0x6D, 0x6C, 0xA5,
        0x95, 0x96, 0x3D, 0x4D, 0x4C, 0x1D, 0xA6,
    ),
}

SWORD_MAX = 4  # 0 none, 1 fighter, 2 master, 3 tempered, 4 gold
SHIELD_MAX = 3  # 0 none, 1 fighter, 2 fire, 3 mirror
ARMOR_MAX = 2  # 0 green, 1 blue, 2 red
PLAY_MODULES = frozenset({0x07, 0x09})


def _room_to_dungeon() -> dict[int, int]:
    table: dict[int, int] = {}
    for dungeon, rooms in _PALACE_ROOMS.items():
        for room in rooms:
            if room in table:
                raise RuntimeError(
                    f"room 0x{room:04X} listed in 0x{table[room]:02X} and 0x{dungeon:02X}"
                )
            table[room] = dungeon
    return table


ROOM_TO_DUNGEON = _room_to_dungeon()
PALACE_IDS = frozenset(_PALACE_ROOMS)


def room_dungeon_issues(indoors: bool, room_id: int, dungeon_id: int) -> list[str]:
    """Room word versus ``$040C``. Overworld must be 0xFF. Listed rooms must match."""
    room = int(room_id) & 0xFFFF
    dungeon = int(dungeon_id) & 0xFF
    if dungeon != 0xFF and dungeon not in PALACE_IDS:
        return [f"dungeon 0x{dungeon:02X} is not a palace index"]
    if not indoors:
        if dungeon != 0xFF:
            return [f"overworld room 0x{room:04X} has dungeon 0x{dungeon:02X}"]
        return []
    expected = ROOM_TO_DUNGEON.get(room)
    if expected is None and room <= 0xFF:
        expected = ROOM_TO_DUNGEON.get(room & 0xFF)
    if expected is None:
        if room >= 0x100 and dungeon != 0xFF:
            return [f"cave room 0x{room:04X} has dungeon 0x{dungeon:02X}"]
        return []
    if dungeon != expected:
        return [
            f"room 0x{room:04X} belongs to dungeon 0x{expected:02X}, "
            f"pin has 0x{dungeon:02X}"
        ]
    return []


def coherence_issues(row: dict[str, Any]) -> list[str]:
    """HP, gear range, play module, and room-versus-dungeon. Empty when coherent."""
    issues: list[str] = []
    hp = int(row["hp"])
    max_hp = int(row["max_hp"])
    if hp > max_hp:
        issues.append(f"hp 0x{hp:02X} > max 0x{max_hp:02X}")
    if max_hp % 8 != 0:
        issues.append(f"max_hp 0x{max_hp:02X} is not a multiple of 8")
    sword = int(row["sword"])
    shield = int(row["shield"])
    armor = int(row["armor"])
    if not 0 <= sword <= SWORD_MAX:
        issues.append(f"sword {sword} out of range 0..{SWORD_MAX}")
    if not 0 <= shield <= SHIELD_MAX:
        issues.append(f"shield {shield} out of range 0..{SHIELD_MAX}")
    if not 0 <= armor <= ARMOR_MAX:
        issues.append(f"armor {armor} out of range 0..{ARMOR_MAX}")
    module = int(row["module"])
    if module not in PLAY_MODULES:
        issues.append(f"module 0x{module:02X} is not play")
    issues.extend(
        room_dungeon_issues(bool(row["indoors"]), int(row["room_id"]), int(row["dungeon_id"]))
    )
    return issues


_CHILD = """
import json, sys
from retro_harness.env import make_env, reset_obs, resync_custom_state
from alttp.paths import GAME, GAME_DIR
from alttp.ram import read_inventory, read_snapshot

name = sys.argv[1]
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
env = make_env(GAME, name, GAME_DIR, render_mode=None)
reset_obs(env)
resync_custom_state(env, GAME_DIR, GAME, name)
ram = env.get_ram()
snap = read_snapshot(ram)
inv = read_inventory(ram)
print(json.dumps({
    "pin": name,
    "hp": int(inv.hp),
    "max_hp": int(inv.max_hp),
    "sword": int(inv.sword),
    "shield": int(inv.shield),
    "armor": int(inv.armor),
    "module": int(snap.game_mode),
    "submodule": int(snap.submodule),
    "indoors": bool(snap.indoors),
    "room_id": int(snap.room_id),
    "dungeon_id": int(snap.dungeon_id),
}))
env.close()
"""


def pin_names() -> list[str]:
    integration = integration_dir(GAME_DIR, GAME)
    return sorted(path.stem for path in integration.glob("*.state"))


def provenance_kind(name: str) -> str | None:
    data = read_pin_provenance(state_path(GAME_DIR, GAME, name))
    if data is None:
        return None
    kind = data.get("kind")
    return None if kind is None else str(kind)


def read_pin(name: str) -> dict[str, Any]:
    """One emulator per process."""
    env = os.environ.copy()
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    env.setdefault("SDL_AUDIODRIVER", "dummy")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD, name],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[2]),
            env=env,
            timeout=90,
        )
    except subprocess.TimeoutExpired:
        return {"pin": name, "error": "timeout"}
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return {"pin": name, "error": (tail[-1] if tail else "emulator failed")[:200]}
    line = proc.stdout.strip().splitlines()[-1]
    row = json.loads(line)
    row["kind"] = provenance_kind(name)
    row["issues"] = coherence_issues(row)
    return row


def _format_row(row: dict[str, Any]) -> str:
    name = str(row["pin"])
    if "error" in row:
        return f"{name} ERROR {row['error']}"
    flag = "  " if not row["issues"] else "  <-- " + "; ".join(row["issues"])
    kind = row.get("kind") or "none"
    return (
        f"{name} hp=0x{int(row['hp']):02X}/0x{int(row['max_hp']):02X} "
        f"kind={kind} room=0x{int(row['room_id']):04X} "
        f"dungeon=0x{int(row['dungeon_id']):02X} "
        f"module=0x{int(row['module']):02X}{flag}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", action="append", help="audit only these pins")
    parser.add_argument("--json", action="store_true", help="emit rows as JSON")
    args = parser.parse_args(argv)
    names = args.pin or pin_names()
    rows = [read_pin(name) for name in names]
    bad = [
        row
        for row in rows
        if row.get("error") or row.get("issues")
    ]
    if args.json:
        print(json.dumps({"rows": rows, "incoherent": bad}, indent=2))
    else:
        for row in rows:
            print(_format_row(row))
        print(f"\n{len(rows)} pins, {len(bad)} incoherent")
        print("entry:")
        for row in rows:
            if not str(row["pin"]).endswith("Entry"):
                continue
            if "error" in row:
                print(f"  {row['pin']} ERROR {row['error']}")
                continue
            kind = row.get("kind") or "none"
            print(
                f"  {row['pin']} hp=0x{int(row['hp']):02X}/0x{int(row['max_hp']):02X} "
                f"kind={kind}"
            )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the room census from tape-scan pins.

Reads ``recordings/scan/index.json`` and each ``Scan_*.state``. One
emulator, load only: no RAM writes. Doors, chests, and stair/pit tiles
come from the loaded pin. Sprite names must match the scan ``.txt``.
Writes ``docs/routes/ROOM_CENSUS.md``.

A castle room is palace index ``0x02`` (``audit_pins.ROOM_TO_DUNGEON``,
full room id, caves not masked) or a ``hyrule_castle*`` label. ``0xFF``
is not a palace. Stair and pit tiles are ``tile_kind`` only.

    cd snes && SDL_VIDEODRIVER=dummy uv run python alttp/scripts/room_census.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from alttp.dungeon.rom_scan import chest_lines, door_lines
from alttp.dungeon.tilemap import read_room_tiles, tile_kind
from alttp.paths import DOCS_DIR, GAME, GAME_DIR, RECORDINGS_DIR
from alttp.primitives import active_sprites
from alttp.ram import read_snapshot, room_label
from alttp.scripts.audit_pins import ROOM_TO_DUNGEON
from alttp.sprite_ids import sprite_name
from retro_harness.env import make_env, read_state_bytes
from retro_harness.segment_runner import configure_headless

SCAN_DIR = RECORDINGS_DIR / "scan"
OUT_PATH = DOCS_DIR / "routes" / "ROOM_CENSUS.md"
# ram.py measured names. Other palace indexes stay unnamed.
_NAMED_PALACE = {0x02: "castle", 0x04: "eastern"}
_SPRITE_RE = re.compile(
    r"^slot\s+\d+\s+(.*?)\s+state=0x[0-9A-Fa-f]+\s+hp=-?\d+\s+\(-?\d+,-?\d+\)\s*$"
)
_SHOW_CELLS = 8


def palace_of(room_id: int) -> int | None:
    """Palace index for a dungeon room. Cave ids (``>= 0x100``) are not masked."""
    room = int(room_id) & 0xFFFF
    found = ROOM_TO_DUNGEON.get(room)
    if found is None and room <= 0xFF:
        found = ROOM_TO_DUNGEON.get(room & 0xFF)
    return found


def castle_kind(room_id: int) -> str:
    """``palace`` for index 0x02, ``named`` for a castle label, else empty."""
    if palace_of(room_id) == 0x02:
        return "palace"
    if room_label(int(room_id) & 0xFF).startswith("hyrule_castle"):
        return "named"
    return ""


def dungeon_text(indoors: bool, dungeon_id: int, floor: int) -> str:
    if not indoors:
        return "n/a"
    dung = int(dungeon_id) & 0xFF
    if dung == 0xFF:
        name = "not a palace"
    else:
        name = _NAMED_PALACE.get(dung, "palace index")
    return f"0x{dung:02X} {name}; floor 0x{int(floor) & 0xFF:02X}"


def _cell(text: str) -> str:
    return " ".join(text.replace("|", "/").split())


def compact_doors(lines: list[str]) -> str:
    """Primary slots and object locks. Layer copies of the same door are dropped."""
    if not lines or "outdoors" in lines[0]:
        return "n/a"
    parts = [
        line.strip()
        for line in lines[1:]
        if line.strip() and " layer of slot " not in line and not line.strip().startswith("(no door")
    ]
    return "; ".join(parts) if parts else "none"


def compact_chests(lines: list[str]) -> str:
    if not lines or "outdoors" in lines[0]:
        return "n/a"
    if "(none)" in lines[0]:
        return "none"
    parts = [line.strip() for line in lines[1:] if line.strip()]
    return "; ".join(parts) if parts else "none"


def _dump_sprite_names(text: str) -> list[str]:
    names: list[str] = []
    for line in text.splitlines():
        match = _SPRITE_RE.match(line.strip())
        if match is not None:
            names.append(match.group(1).strip())
    return names


def sprite_text(env: object, dump: str) -> str:
    """Unique live types. Refuses when the scan text names a different set."""
    live = active_sprites(env)
    live_names = [sprite_name(int(sprite.sprite_type)) for sprite in live]
    if set(live_names) != set(_dump_sprite_names(dump)):
        raise SystemExit(
            "sprite names on the loaded pin do not match the scan text: "
            f"live={sorted(set(live_names))} dump={sorted(set(_dump_sprite_names(dump)))}"
        )
    seen: list[str] = []
    have: set[int] = set()
    for sprite in live:
        type_id = int(sprite.sprite_type)
        if type_id in have:
            continue
        have.add(type_id)
        seen.append(f"{sprite_name(type_id)} 0x{type_id:02X}")
    return ", ".join(seen) if seen else "none"


def _kind_cells(grid: np.ndarray, kind: str) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    rows, cols = grid.shape
    for row in range(int(rows)):
        for col in range(int(cols)):
            if tile_kind(int(grid[row, col])) == kind:
                found.append((col, row))
    return found


def _fmt_cells(cells: list[tuple[int, int]]) -> str:
    if not cells:
        return "none"
    shown = " ".join(f"({col},{row})" for col, row in cells[:_SHOW_CELLS])
    extra = len(cells) - _SHOW_CELLS
    more = f" +{extra}" if extra > 0 else ""
    return f"n={len(cells)} {shown}{more}"


def stairs_and_pits(ram: np.ndarray, indoors: bool, doors: str) -> str:
    """Door slots typed stairs, plus legend stair and pit cells. Unknown bytes stay out."""
    if not indoors:
        return "n/a"
    bits: list[str] = []
    if doors not in ("n/a", "none"):
        stair_doors = [part for part in doors.split("; ") if "stairs" in part]
        if stair_doors:
            bits.append("door-slots " + "; ".join(stair_doors))
    tile_bits: list[str] = []
    for layer in (0, 1):
        grid = read_room_tiles(ram, layer)
        stairs = _kind_cells(grid, "stairs")
        pits = _kind_cells(grid, "pit")
        if not stairs and not pits:
            continue
        tile_bits.append(
            f"layer {layer} stairs {_fmt_cells(stairs)}; pits {_fmt_cells(pits)}"
        )
    if tile_bits:
        bits.append("; ".join(tile_bits))
    elif not bits:
        return "none"
    else:
        bits.append("tiles none")
    return "; ".join(bits)


def row_from_env(pin: dict[str, Any], env: object, dump: str) -> dict[str, str]:
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    snap = read_snapshot(ram)
    if snap.indoors:
        room = f"0x{snap.room_id:04X} {room_label(snap.room_base_id)}"
        kind = castle_kind(snap.room_id)
    else:
        world = "dark world" if snap.dark_world else "light world"
        room = f"screen 0x{snap.screen_id:02X} ({world})"
        kind = ""
    doors = compact_doors(door_lines(env))
    return {
        "pin": str(pin["pin"]),
        "room": room,
        "dungeon": dungeon_text(snap.indoors, snap.dungeon_id, snap.floor),
        "doors": doors,
        "chests": compact_chests(chest_lines(env)),
        "sprites": sprite_text(env, dump),
        "stairs": stairs_and_pits(ram, snap.indoors, doors),
        "castle": kind,
    }


def _visit_text(location: list[Any]) -> str:
    if location[0] == "in":
        return f"room 0x{int(location[1]):04X}"
    world = "dw" if location[1] else "ow"
    return f"{world} screen 0x{int(location[2]):02X}"


def _castle_sentence(rows: list[dict[str, str]]) -> str:
    palace = [row for row in rows if row["castle"] == "palace"]
    named = [row for row in rows if row["castle"] == "named"]
    parts: list[str] = []
    if palace:
        listed = ", ".join(f"{row['pin']} {row['room']}" for row in palace)
        parts.append(f"Palace 0x02 rooms: {listed}.")
    else:
        parts.append("No palace 0x02 room is in this scan.")
    if named:
        listed = ", ".join(f"{row['pin']} {row['room']}" for row in named)
        parts.append(f"Castle-named rooms outside that palace: {listed}.")
    elif not palace:
        parts.append("No hyrule_castle-labeled room is in this scan.")
    return " ".join(parts)


def render_census(index: dict[str, Any], rows: list[dict[str, str]]) -> str:
    """Markdown table. ``rows`` follow ``index['pins']`` order."""
    tape = Path(str(index.get("tape") or "")).name
    unsettled = [
        _visit_text(list(row["location"]))
        for row in index.get("unsettled") or []
        if row.get("location")
    ]
    skipped = ", ".join(unsettled) if unsettled else "none"
    header = (
        "# Room census",
        "",
        "Generated by `alttp/scripts/room_census.py` from `recordings/scan`.",
        "Do not edit by hand.",
        "",
        (
            f"Tape `{tape}`: frames {index.get('frames')}, "
            f"desync {index.get('desynced')}, writes {index.get('writes')}, "
            f"loads {index.get('loads')}."
        ),
        f"Settled pins: {len(rows)}. Not pinned: {skipped}.",
        "",
        _castle_sentence(rows),
        "Stairs and pits are door slots whose type name is stairs, plus",
        "`tile_kind` cells (`0x1D` stairs, `0x20` pit). Other bytes are not stairs.",
        "",
        "| pin | room | dungeon | doors | chests | sprites | stairs and pits |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    )
    body = [
        "| "
        + " | ".join(
            _cell(row[key])
            for key in ("pin", "room", "dungeon", "doors", "chests", "sprites", "stairs")
        )
        + " |"
        for row in rows
    ]
    return "\n".join((*header, *body, ""))


def collect_rows(index: dict[str, Any], scan_dir: Path) -> list[dict[str, str]]:
    """Load each pin on one env. Closes the env before returning."""
    pins = list(index.get("pins") or [])
    if not pins:
        raise SystemExit(f"no pins in {scan_dir / 'index.json'}")
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    rows: list[dict[str, str]] = []
    try:
        for pin in pins:
            name = str(pin["pin"])
            state = scan_dir / f"{name}.state"
            text_path = scan_dir / f"{name}.txt"
            if not state.is_file() or not text_path.is_file():
                raise SystemExit(f"missing scan dump for {name}")
            env.reset()
            env.em.set_state(read_state_bytes(state))  # type: ignore[attr-defined]
            rows.append(row_from_env(pin, env, text_path.read_text(encoding="utf-8")))
    finally:
        env.close()
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, default=SCAN_DIR)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)
    index_path = args.scan / "index.json"
    if not index_path.is_file():
        raise SystemExit(f"missing {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    rows = collect_rows(index, args.scan)
    text = render_census(index, rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    castle = [row["pin"] for row in rows if row["castle"]]
    print(f"wrote {args.out} pins={len(rows)} castle={','.join(castle) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

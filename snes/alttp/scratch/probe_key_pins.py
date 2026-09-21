"""Census Zelda3-Snes work-queue pins for dungeon keys leftover.

Load → settle_control → RAM glance. Does not walk rooms. Does not poke keys.

Usage:
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_key_pins.py
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from alttp.opening_route.work_queue import build_catalog
from alttp.paths import GAME_DIR, INTEGRATION, INTEGRATION_DIR, RECORDINGS_DIR
from alttp.primitives import settle_control
from alttp.ram import (
    EQUIP_SWORD,
    FOLLOWER,
    LINK_HP,
    LINK_ITEM_LAMP,
    LINK_MAX_HP,
    NUM_KEYS,
    room_label,
    snapshot_to_diag,
    wram_index,
)
from alttp.startup import build_boot_env, snapshot_env
from retro_harness.env import resync_custom_state

OUT_DIR = RECORDINGS_DIR / "probe_key_pins"
KEYS_BLANK = 0xFF
# maps/room_81.json west_to_0x80_approach (cell-door y-band).
WEST_WALL_ROOM = 0x81
WEST_WALL_XY = (608, 4168)
WEST_WALL_X_MAX = 640
WEST_WALL_Y_TOL = 32
# 0x72 north-ledge south lip vs lower floor (b1-to-zelda residual).
ROOM_72 = 0x72
LEDGE_Y_MAX = 3776
LOWER_Y_MIN = 3900


def discover_pins() -> list[tuple[str, int, str]]:
    """Work-queue state names in ranked order. Fail if a catalog pin is missing."""
    items = build_catalog()
    names = [i.state_name for i in items]
    missing = [
        n for n in names if not (INTEGRATION_DIR / f"{n}.state").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"work-queue pins missing: {missing}")
    return [(i.state_name, int(i.rank), i.group) for i in items]


def leftover_bytes(env: object) -> dict[str, int]:
    ram = env.get_ram()  # type: ignore[attr-defined]
    return {
        "F359_sword": int(ram[wram_index(EQUIP_SWORD)]),
        "F34A_lamp": int(ram[wram_index(LINK_ITEM_LAMP)]),
        "F36F_keys": int(ram[wram_index(NUM_KEYS)]),
        "F3CC_follower": int(ram[wram_index(FOLLOWER)]),
        "F36D_hp": int(ram[wram_index(LINK_HP)]),
        "F36C_max_hp": int(ram[wram_index(LINK_MAX_HP)]),
    }


def keys_held(keys: int) -> bool:
    return int(keys) != KEYS_BLANK and int(keys) >= 1


def on_west_wall(room: int, x: int, y: int) -> bool:
    return (
        int(room) == WEST_WALL_ROOM
        and int(x) <= WEST_WALL_X_MAX
        and abs(int(y) - WEST_WALL_XY[1]) <= WEST_WALL_Y_TOL
    )


def floor_band_72(room: int, y: int) -> str | None:
    if int(room) != ROOM_72:
        return None
    if int(y) <= LEDGE_Y_MAX:
        return "north_ledge"
    if int(y) >= LOWER_Y_MIN:
        return "lower_floor"
    return "mid"


def pin_record(
    name: str,
    rank: int,
    group: str,
    settle: Any,
    snap: Any,
    leftover: dict[str, int],
) -> dict[str, Any]:
    keys = int(snap.num_keys)
    room = int(snap.room_base_id)
    x = int(snap.link_x)
    y = int(snap.link_y)
    return {
        "state": name,
        "rank": rank,
        "group": group,
        "module": int(snap.game_mode),
        "module_hex": f"0x{int(snap.game_mode):02X}",
        "submodule": int(snap.submodule),
        "indoors": bool(snap.indoors),
        "room_base_id": room,
        "room_hex": f"0x{room:02X}",
        "room_label": room_label(room),
        "screen": int(snap.screen_id),
        "screen_hex": f"0x{int(snap.screen_id):02X}",
        "link_x": x,
        "link_y": y,
        "sword": int(snap.sword_level),
        "lamp": int(snap.lamp_level),
        "keys": keys,
        "keys_blank": keys == KEYS_BLANK,
        "keys_held": keys_held(keys),
        "dungeon_key_count": None if keys == KEYS_BLANK else keys,
        "follower": int(snap.follower),
        "has_zelda_follower": bool(snap.has_zelda_follower),
        "in_zelda_cell": bool(snap.in_zelda_cell),
        "hp": leftover["F36D_hp"],
        "max_hp": leftover["F36C_max_hp"],
        "has_control": bool(snap.has_control),
        "settle_ok": bool(settle.ok),
        "settle_frames": int(settle.frames),
        "settle_reason": settle.reason,
        "on_0x81_west_wall": on_west_wall(room, x, y),
        "room_72_band": floor_band_72(room, y),
        "leftover": leftover,
        "diag": snapshot_to_diag(snap),
    }


def census_pins(pins: list[tuple[str, int, str]]) -> list[dict[str, Any]]:
    env = build_boot_env(pins[0][0], render_mode="rgb_array")
    records: list[dict[str, Any]] = []
    try:
        env.reset()
        for name, rank, group in pins:
            if not resync_custom_state(env, GAME_DIR, INTEGRATION, name):
                raise RuntimeError(f"failed to load {name}")
            settle = settle_control(env)
            snap = snapshot_env(env)
            leftover = leftover_bytes(env)
            rec = pin_record(name, rank, group, settle, snap, leftover)
            if leftover["F36F_keys"] != rec["keys"]:
                raise RuntimeError(f"{name}: keys snapshot/leftover mismatch")
            records.append(rec)
            print(
                f"{name:32} room {rec['room_hex']} "
                f"xy=({rec['link_x']},{rec['link_y']}) "
                f"mod={rec['module_hex']}/{rec['submodule']} "
                f"ctrl={int(rec['has_control'])} "
                f"sword={rec['sword']} lamp={rec['lamp']} "
                f"keys={rec['keys']} hp={rec['hp']} "
                f"F3CC={rec['follower']} "
                f"held={int(rec['keys_held'])} "
                f"west={int(rec['on_0x81_west_wall'])} "
                f"72={rec['room_72_band'] or '-'}"
            )
    finally:
        env.close()
    return records


def _brief(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "state": rec["state"],
        "room_hex": rec["room_hex"],
        "link_x": rec["link_x"],
        "link_y": rec["link_y"],
        "keys": rec["keys"],
        "follower": rec["follower"],
        "hp": rec["hp"],
        "on_0x81_west_wall": rec["on_0x81_west_wall"],
        "room_72_band": rec["room_72_band"],
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    held = [r for r in records if r["keys_held"]]
    blank = [r for r in records if r["keys_blank"]]
    west = [r for r in records if r["on_0x81_west_wall"]]
    held_west = [r for r in held if r["on_0x81_west_wall"]]
    held_81 = [r for r in held if r["room_base_id"] == 0x81]
    held_80 = [r for r in held if r["room_base_id"] == 0x80]
    held_82 = [r for r in held if r["room_base_id"] == 0x82]
    held_72_ledge = [r for r in held if r["room_72_band"] == "north_ledge"]
    held_72_lower = [r for r in held if r["room_72_band"] == "lower_floor"]
    held_72_mid = [r for r in held if r["room_72_band"] == "mid"]
    none_west = not held_west
    return {
        "pinCount": len(records),
        "keysHeldCount": len(held),
        "keysBlankCount": len(blank),
        "keysHeld": [_brief(r) for r in held],
        "keysBlank": [r["state"] for r in blank],
        "on0x81WestWall": [_brief(r) for r in west],
        "keysHeldIn0x81": [_brief(r) for r in held_81],
        "keysHeldOn0x81WestWall": [_brief(r) for r in held_west],
        "keysHeldIn0x80": [_brief(r) for r in held_80],
        "keysHeldIn0x82": [_brief(r) for r in held_82],
        "keysHeldIn0x72NorthLedge": [_brief(r) for r in held_72_ledge],
        "keysHeldIn0x72LowerFloor": [_brief(r) for r in held_72_lower],
        "keysHeldIn0x72Mid": [_brief(r) for r in held_72_mid],
        "noneOn0x81WestWall": none_west,
        "honest": (
            "none on 0x81 west wall"
            if none_west
            else "keys>=1 pin sits on 0x81 west wall"
        ),
    }


def write_payload(records: list[dict[str, Any]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "alttp_key_pin_census",
        "schemaVersion": 1,
        "measured": datetime.now(timezone.utc).isoformat(),
        "source": "state_load_dev",
        "poked": False,
        "walked": False,
        "westWallXy": list(WEST_WALL_XY),
        "westWallXMax": WEST_WALL_X_MAX,
        "westWallYTol": WEST_WALL_Y_TOL,
        "room72LedgeYMax": LEDGE_Y_MAX,
        "room72LowerYMin": LOWER_Y_MIN,
        "summary": summarize(records),
        "pins": records,
    }
    path = out_dir / "key_pin_census.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="RAM-census work-queue pins for dungeon keys leftover."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help=f"JSON dump directory (default {OUT_DIR})",
    )
    args = parser.parse_args(argv)
    pins = discover_pins()
    print(f"pins: {len(pins)}")
    records = census_pins(pins)
    path = write_payload(records, args.out_dir)
    summary = summarize(records)
    print(f"wrote {path}")
    print(f"keys>=1: {len(summary['keysHeld'])}  blank: {len(summary['keysBlank'])}")
    print(f"honest: {summary['honest']}")
    for rec in summary["keysHeld"]:
        print(
            f"  HELD {rec['state']:32} {rec['room_hex']} "
            f"({rec['link_x']},{rec['link_y']}) keys={rec['keys']} "
            f"west={int(rec['on_0x81_west_wall'])} 72={rec['room_72_band'] or '-'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

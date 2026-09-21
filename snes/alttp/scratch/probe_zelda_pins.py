"""Census Zelda* / follower Zelda3-Snes pins. RAM glance only; no walks.

Usage:
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_zelda_pins.py
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from alttp.paths import GAME_DIR, INTEGRATION, INTEGRATION_DIR, RECORDINGS_DIR
from alttp.primitives import settle_control
from alttp.ram import (
    EQUIP_SWORD,
    FOLLOWER,
    LINK_ITEM_LAMP,
    NUM_KEYS,
    room_label,
    snapshot_to_diag,
    wram_index,
)
from alttp.startup import build_boot_env, snapshot_env
from retro_harness.env import resync_custom_state

OUT_DIR = RECORDINGS_DIR / "probe_zelda_pins"
QUEUE_PINS = (
    "CastleMainZeldaBoomerang",
    "CastleMainZeldaReady",
    "CastleZeldaFollower",
    "CastleRoom51Zelda",
    "CastleRoom52ZeldaBoomerang",
    "CastleZeldaB1East",
    "CastleZeldaB1Island",
    "CastleZeldaB1Pit",
    "CastleZeldaB1West",
    "CastleMantleZelda",
)


def discover_pins(integration_dir: Path | None = None) -> list[str]:
    root = integration_dir or INTEGRATION_DIR
    names = sorted(
        p.stem
        for p in root.glob("*.state")
        if p.is_file() and ("Zelda" in p.stem or "Follower" in p.stem)
    )
    extra = [n for n in QUEUE_PINS if n not in names]
    if extra:
        raise FileNotFoundError(f"work-queue Zelda pins missing: {extra}")
    return names


def leftover_bytes(env: object) -> dict[str, int]:
    ram = env.get_ram()  # type: ignore[attr-defined]
    return {
        "F359_sword": int(ram[wram_index(EQUIP_SWORD)]),
        "F34A_lamp": int(ram[wram_index(LINK_ITEM_LAMP)]),
        "F36F_keys": int(ram[wram_index(NUM_KEYS)]),
        "F3CC_follower": int(ram[wram_index(FOLLOWER)]),
    }


def pin_record(name: str, settle: Any, snap: Any, leftover: dict[str, int]) -> dict[str, Any]:
    return {
        "state": name,
        "module": int(snap.game_mode),
        "module_hex": f"0x{int(snap.game_mode):02X}",
        "submodule": int(snap.submodule),
        "indoors": bool(snap.indoors),
        "room_base_id": int(snap.room_base_id),
        "room_hex": f"0x{int(snap.room_base_id):02X}",
        "room_label": room_label(snap.room_base_id),
        "screen": int(snap.screen_id),
        "screen_hex": f"0x{int(snap.screen_id):02X}",
        "link_x": int(snap.link_x),
        "link_y": int(snap.link_y),
        "sword": int(snap.sword_level),
        "lamp": int(snap.lamp_level),
        "keys": int(snap.num_keys),
        "follower": int(snap.follower),
        "has_zelda_follower": bool(snap.has_zelda_follower),
        "in_zelda_cell": bool(snap.in_zelda_cell),
        "has_control": bool(snap.has_control),
        "settle_ok": bool(settle.ok),
        "settle_frames": int(settle.frames),
        "settle_reason": settle.reason,
        "leftover": leftover,
        "diag": snapshot_to_diag(snap),
    }


def census_pins(names: list[str]) -> list[dict[str, Any]]:
    env = build_boot_env(names[0], render_mode="rgb_array")
    records: list[dict[str, Any]] = []
    try:
        env.reset()
        for name in names:
            if not resync_custom_state(env, GAME_DIR, INTEGRATION, name):
                raise RuntimeError(f"failed to load {name}")
            settle = settle_control(env)
            snap = snapshot_env(env)
            leftover = leftover_bytes(env)
            rec = pin_record(name, settle, snap, leftover)
            if leftover["F3CC_follower"] != rec["follower"]:
                raise RuntimeError(f"{name}: follower snapshot/leftover mismatch")
            records.append(rec)
            print(
                f"{name:28} room {rec['room_hex']} "
                f"xy=({rec['link_x']},{rec['link_y']}) "
                f"mod={rec['module_hex']}/{rec['submodule']} "
                f"ctrl={int(rec['has_control'])} "
                f"sword={rec['sword']} lamp={rec['lamp']} keys={rec['keys']} "
                f"F3CC={rec['follower']} cell={int(rec['in_zelda_cell'])}"
            )
    finally:
        env.close()
    return records


def write_payload(records: list[dict[str, Any]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "alttp_zelda_pin_census",
        "schemaVersion": 1,
        "measured": datetime.now(timezone.utc).isoformat(),
        "source": "state_load_dev",
        "poked": False,
        "walked": False,
        "pinCount": len(records),
        "pins": records,
        "inZeldaCell": [r["state"] for r in records if r["in_zelda_cell"]],
        "followerTrue": [
            r["state"] for r in records if r["follower"] == 1
        ],
    }
    path = out_dir / "zelda_pin_census.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAM-census Zelda* save-state pins.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help=f"JSON dump directory (default {OUT_DIR})",
    )
    args = parser.parse_args(argv)
    names = discover_pins()
    print(f"pins: {len(names)} -> {', '.join(names)}")
    records = census_pins(names)
    path = write_payload(records, args.out_dir)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

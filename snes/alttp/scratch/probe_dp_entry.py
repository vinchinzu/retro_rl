"""Pin Desert Palace main/book entrance and isolate the first indoor door.

Dev / state-load only. Does not poke $F3CC / sword-from-zero on a STATUS path.
Does not write docs/STATUS.md or verified_tip_run.json.

Usage:
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_dp_entry.py
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_dp_entry.py --scan
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_dp_entry.py --edges
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from PIL import Image

from alttp.paths import GAME_DIR, INTEGRATION, RECORDINGS_DIR
from alttp.primitives import Waypoint, active_sprites, fight_nearby, move_to, settle_control
from alttp.ram import (
    FOLLOWER,
    LINK_HP,
    LINK_ITEM_LAMP,
    LINK_MAX_HP,
    NUM_KEYS,
    snapshot_to_diag,
    wram_index,
)
from alttp.room_sense import detect_edge, overlay_from_env
from alttp.screen_glance import leftover_from_snapshot
from alttp.startup import action_for, build_boot_env, no_action, snapshot_env, step_frames
from retro_harness.env import resync_custom_state, save_state

OUT_DIR = RECORDINGS_DIR / "probe_dp_entry"
PIN_NAME = "DesertPalaceEntry"
SOURCE_STATE = "FighterSword"
FALLBACK_STATE = "HyruleCastleGrounds"

# Inventory / progress (WRAM $7EF3xx). Offset == $F3xx for memory.assign.
BOW = 0xF340
LAMP = 0xF34A
BOOK = 0xF34E
GLOVES = 0xF354
BOOTS = 0xF355
SWORD = 0xF359
ARROWS = 0xF377
ENTRANCE_ID = 0x010E
MODULE = 0x10
SUBMODULE = 0x11
Y_ITEM = 0x0202
LAYER = 0x00EE
DUNGEON = 0x040C

# ROM $02C813 dungeon entrance table (US): index 0x09 → room 0x0084 (main/book).
# Overworld door_index 0x08 is a different table (that index is Eastern Palace 0xC9).
DESERT_MAIN_ENTRANCE = 0x09
HYPOTHESIS_ROOM = 0x84
PREDUNGEON_MODULE = 0x06
DUNGEON_MODULE = 0x07

LOADOUT = (
    (BOW, 2, "bow+arrows"),
    (LAMP, 1, "lamp"),
    (BOOK, 1, "book"),
    (BOOTS, 1, "boots"),
    (SWORD, 1, "fighter sword"),
    (ARROWS, 30, "arrows"),
)

CARDINALS = ("UP", "DOWN", "LEFT", "RIGHT")


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _memory(env: object) -> Any:
    return env.unwrapped.data.memory  # type: ignore[attr-defined]


def read_u8(env: object, offset: int) -> int:
    ram = env.get_ram()  # type: ignore[attr-defined]
    if offset >= 0x2000:
        return int(ram[wram_index(offset)])
    return int(ram[int(offset)])


def read_u16(env: object, offset: int) -> int:
    return read_u8(env, offset) | (read_u8(env, offset + 1) << 8)


def assign_addr(offset: int) -> int:
    """stable-retro SNES WRAM: low $0000-$1FFF as-is, high via bank $7E."""
    off = int(offset)
    return off if off < 0x2000 else 0x7E0000 + off


def poke_u8(env: object, offset: int, value: int) -> dict[str, Any]:
    """Write one WRAM byte via memory.assign. Confirm with get_ram() read-after-write."""
    before = read_u8(env, offset)
    mapped = assign_addr(offset)
    _memory(env).assign(mapped, "|u1", int(value) & 0xFF)
    after = read_u8(env, offset)
    return {
        "offset": f"0x{offset:04X}",
        "want": int(value) & 0xFF,
        "before": before,
        "after": after,
        "ok": after == (int(value) & 0xFF),
        "assign_addr": f"0x{mapped:X}",
    }


def poke_u16(env: object, offset: int, value: int) -> dict[str, Any]:
    lo = poke_u8(env, offset, int(value) & 0xFF)
    hi = poke_u8(env, offset + 1, (int(value) >> 8) & 0xFF)
    return {"lo": lo, "hi": hi, "ok": bool(lo["ok"] and hi["ok"])}


def leftover_bytes(env: object) -> dict[str, int]:
    ram = env.get_ram()  # type: ignore[attr-defined]
    return {
        "F340_bow": int(ram[wram_index(BOW)]),
        "F34A_lamp": int(ram[wram_index(LAMP)]),
        "F34E_book": int(ram[wram_index(BOOK)]),
        "F354_gloves": int(ram[wram_index(GLOVES)]),
        "F355_boots": int(ram[wram_index(BOOTS)]),
        "F359_sword": int(ram[wram_index(SWORD)]),
        "F377_arrows": int(ram[wram_index(ARROWS)]),
        "F36F_keys": int(ram[wram_index(NUM_KEYS)]),
        "F3CC_follower": int(ram[wram_index(FOLLOWER)]),
        "F36D_hp": int(ram[wram_index(LINK_HP)]),
        "F36C_max_hp": int(ram[wram_index(LINK_MAX_HP)]),
        "010E_entrance": read_u16(env, ENTRANCE_ID),
        "0202_yitem": int(ram[Y_ITEM]),
        "00EE_layer": int(ram[LAYER]),
        "040C_dungeon": int(ram[DUNGEON]),
        "2F_dir": int(ram[0x2F]),
        "5D_act": int(ram[0x5D]),
    }


def glance(env: object) -> dict[str, Any]:
    snap = snapshot_env(env)
    sprites = [
        {"type": f"0x{s.sprite_type:02X}", "xy": [s.x, s.y], "hp": s.hp}
        for s in active_sprites(env)
    ]
    rec = leftover_from_snapshot(snap)
    rec.update(
        {
            "module": int(snap.game_mode),
            "module_hex": f"0x{int(snap.game_mode):02X}",
            "submodule": int(snap.submodule),
            "indoors": bool(snap.indoors),
            "room_base_id": int(snap.room_base_id),
            "room_hex": f"0x{int(snap.room_base_id):02X}",
            "screen": int(snap.screen_id),
            "screen_hex": f"0x{int(snap.screen_id):02X}",
            "link_x": int(snap.link_x),
            "link_y": int(snap.link_y),
            "sword": int(snap.sword_level),
            "lamp": int(snap.lamp_level),
            "keys": int(snap.num_keys),
            "follower": int(snap.follower),
            "has_control": bool(snap.has_control),
            "leftover": leftover_bytes(env),
            "boot": leftover_from_snapshot(snap),
            "sprites": sprites[:16],
            "diag": snapshot_to_diag(snap),
        }
    )
    return rec


def save_overlay(env: object, path: Path, title: str = "") -> None:
    img = overlay_from_env(env, include_all_sprites=True, title=title)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_pin(env: object, name: str) -> None:
    if not resync_custom_state(env, GAME_DIR, INTEGRATION, name):
        raise RuntimeError(f"failed to load {name}")


def apply_loadout(env: object) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for offset, value, label in LOADOUT:
        rec = poke_u8(env, offset, value)
        rec["label"] = label
        docs.append(rec)
    gloves = poke_u8(env, GLOVES, 0)
    gloves["label"] = "gloves cleared (prize not granted)"
    docs.append(gloves)
    return docs


def predungeon_warp(env: object, entrance: int) -> dict[str, Any]:
    """Trigger a real dungeon load via module 0x06 (PreDungeon) + $010E."""
    before = glance(env)
    ent = poke_u16(env, ENTRANCE_ID, entrance)
    mod = poke_u8(env, MODULE, PREDUNGEON_MODULE)
    sub = poke_u8(env, SUBMODULE, 0)
    settle = settle_control(env, max_frames=720)
    after = glance(env)
    return {
        "entrance_poke": ent,
        "module_poke": mod,
        "submodule_poke": sub,
        "settle_ok": bool(settle.ok),
        "settle_frames": int(settle.frames),
        "settle_reason": settle.reason,
        "before": before,
        "after": after,
    }


def push_cardinal(
    env: object, direction: str, *, room: int, max_frames: int = 480
) -> dict[str, Any]:
    start = snapshot_env(env)
    prev = (int(start.link_x), int(start.link_y))
    stuck = 0
    frames = 0
    edge = None
    while frames < max_frames:
        step_frames(env, action_for(direction), 4)
        frames += 4
        after = snapshot_env(env)
        edge = detect_edge(
            start,
            after,
            expected_room=room,
            frames=frames,
            label=direction,
            preferred_direction=direction,
        )
        if edge is not None:
            break
        xy = (int(after.link_x), int(after.link_y))
        if xy == prev:
            stuck += 1
        else:
            stuck = 0
            prev = xy
            start = after
        if stuck >= 16:
            break
        if after.game_mode == 0x12:
            break
    snap = snapshot_env(env)
    return {
        "direction": direction,
        "frames": frames,
        "stuck": stuck,
        "edge": None if edge is None else edge.to_dict(),
        "final": glance(env),
        "left_room": int(snap.room_base_id) != room or (not snap.indoors),
        "xy": [int(snap.link_x), int(snap.link_y)],
        "room_hex": f"0x{int(snap.room_base_id):02X}",
        "indoors": bool(snap.indoors),
        "module": int(snap.game_mode),
        "submodule": int(snap.submodule),
    }


def walk_north_scan(env: object, *, room: int) -> list[dict[str, Any]]:
    """Walk UP in y-steps from spawn; record xy + room at each halt."""
    samples: list[dict[str, Any]] = []
    start = snapshot_env(env)
    y = int(start.link_y)
    for target_y in range(y - 32, y - 520, -32):
        res = move_to(
            env,
            Waypoint(int(start.link_x), target_y, tolerance=16, room=room, label=f"n{target_y}"),
            max_frames=240,
        )
        snap = res.snapshot
        samples.append(
            {
                "target_y": target_y,
                "ok": bool(res.ok),
                "reason": res.reason,
                "xy": [int(snap.link_x), int(snap.link_y)],
                "room_hex": f"0x{int(snap.room_base_id):02X}",
                "indoors": bool(snap.indoors),
                "module": int(snap.game_mode),
                "submodule": int(snap.submodule),
            }
        )
        if int(snap.room_base_id) != room or not snap.indoors:
            break
        if not res.ok and "stuck" in (res.reason or ""):
            break
    return samples


def cmd_scan() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {"when": _utc(), "states": {}}
    env = build_boot_env(SOURCE_STATE, render_mode="rgb_array")
    try:
        env.reset()
        for name in (SOURCE_STATE, FALLBACK_STATE, "CastleMain"):
            load_pin(env, name)
            settle_control(env)
            rec = glance(env)
            records["states"][name] = rec
            save_overlay(env, OUT_DIR / f"scan_{name}.png", title=name)
            print(
                f"{name}: room={rec['room_hex']} mod={rec['module_hex']}/"
                f"{rec['submodule']} xy=({rec['link_x']},{rec['link_y']}) "
                f"in={rec['indoors']} ctrl={rec['has_control']} "
                f"loadout={ {k: rec['leftover'][k] for k in ('F340_bow','F34A_lamp','F34E_book','F354_gloves','F355_boots','F359_sword')} }"
            )
    finally:
        env.close()
    save_json(OUT_DIR / "scan.json", records)
    print(f"Wrote {OUT_DIR / 'scan.json'}")
    return 0


def cmd_pin() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = build_boot_env(SOURCE_STATE, render_mode="rgb_array")
    report: dict[str, Any] = {
        "when": _utc(),
        "bead": "rr-alwe",
        "source": SOURCE_STATE,
        "pin": PIN_NAME,
        "hypothesis_room": f"0x{HYPOTHESIS_ROOM:02X}",
        "entrance": DESERT_MAIN_ENTRANCE,
    }
    try:
        env.reset()
        load_pin(env, SOURCE_STATE)
        settle_control(env)
        report["source_glance"] = glance(env)
        save_overlay(env, OUT_DIR / "source.png", title=SOURCE_STATE)

        pokes = apply_loadout(env)
        report["pokes"] = pokes
        poke_fail = [p for p in pokes if not p["ok"]]
        if poke_fail:
            report["poke_fail"] = poke_fail
            save_json(OUT_DIR / "pin_report.json", report)
            print(f"POKE FAIL: {poke_fail}")
            return 2
        step_frames(env, no_action(), 4)
        report["after_pokes"] = glance(env)

        warp = predungeon_warp(env, DESERT_MAIN_ENTRANCE)
        report["warp"] = warp
        after = warp["after"]
        save_overlay(env, OUT_DIR / "after_warp.png", title="after PreDungeon 0x08")
        print(
            f"warp settle={warp['settle_ok']} room={after['room_hex']} "
            f"mod={after['module_hex']}/{after['submodule']} "
            f"xy=({after['link_x']},{after['link_y']}) in={after['indoors']} "
            f"ctrl={after['has_control']} dungeon={after['leftover']['040C_dungeon']}"
        )

        room = int(after["room_base_id"])
        ok_pin = (
            bool(after["indoors"])
            and bool(after["has_control"])
            and int(after["module"]) == DUNGEON_MODULE
            and int(after["submodule"]) == 0
            and room == HYPOTHESIS_ROOM
        )
        report["pin_ok"] = ok_pin
        report["confirmed_room"] = f"0x{room:02X}"
        report["pin_glance"] = after

        if not ok_pin:
            save_json(OUT_DIR / "pin_report.json", report)
            print("PIN NOT CONFIRMED — not saving DesertPalaceEntry.state")
            return 3

        path = save_state(env, GAME_DIR, INTEGRATION, PIN_NAME)
        report["state_path"] = str(path)
        print(f"Saved {path}")

        leftover = leftover_bytes(env)
        save_json(OUT_DIR / "leftover.json", {"glance": after, "leftover": leftover})
        save_json(OUT_DIR / "pin_report.json", report)
    finally:
        env.close()
    print(f"Wrote {OUT_DIR / 'pin_report.json'}")
    return 0


def walk_chain(
    env: object,
    steps: list[tuple[int, int, str]],
    *,
    room: int,
    fight: bool = True,
) -> list[dict[str, Any]]:
    """Walk waypoints; optional skirmish. Stops if the room changes."""
    log: list[dict[str, Any]] = []
    for x, y, label in steps:
        if fight:
            fight_nearby(env, room=room, max_distance=48, attack_distance=28, max_cycles=40)
        res = move_to(
            env,
            Waypoint(x, y, tolerance=16, room=room, label=label),
            max_frames=400,
        )
        snap = res.snapshot
        rec = {
            "label": label,
            "target": [x, y],
            "ok": bool(res.ok),
            "reason": res.reason,
            "xy": [int(snap.link_x), int(snap.link_y)],
            "room_hex": f"0x{int(snap.room_base_id):02X}",
            "indoors": bool(snap.indoors),
            "module": int(snap.game_mode),
            "submodule": int(snap.submodule),
            "has_control": bool(snap.has_control),
        }
        log.append(rec)
        if int(snap.room_base_id) != room or not snap.indoors:
            break
        if snap.game_mode == 0x12:
            break
    return log


def cmd_hop() -> int:
    """From spawn: alcove north, then west/east hall, push for a room-change door."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = build_boot_env(PIN_NAME, render_mode="rgb_array")
    report: dict[str, Any] = {"when": _utc(), "pin": PIN_NAME, "routes": {}}
    routes: dict[str, list[tuple[int, int, str]]] = {
        "west_hall": [
            (2296, 4480, "alcove_south"),
            (2296, 4400, "alcove_north"),
            (2160, 4400, "west_doorway"),
            (2104, 4320, "west_hall_mid"),
            (2104, 4200, "west_hall_north"),
            (2064, 4200, "west_door_approach"),
        ],
        "east_hall": [
            (2296, 4480, "alcove_south"),
            (2296, 4400, "alcove_north"),
            (2432, 4400, "east_doorway"),
            (2496, 4320, "east_hall_mid"),
            (2496, 4200, "east_hall_north"),
            (2544, 4200, "east_door_approach"),
        ],
        "north_hall": [
            (2296, 4480, "alcove_south"),
            (2296, 4400, "alcove_north"),
            (2160, 4400, "west_doorway"),
            (2160, 4200, "hall_west"),
            (2296, 4180, "hall_center"),
            (2296, 4120, "hall_north"),
        ],
    }
    pushes = {
        "west_hall": "LEFT",
        "east_hall": "RIGHT",
        "north_hall": "UP",
    }
    try:
        env.reset()
        for name, steps in routes.items():
            load_pin(env, PIN_NAME)
            settle_control(env)
            start = glance(env)
            room = int(start["room_base_id"])
            chain = walk_chain(env, steps, room=room)
            save_overlay(env, OUT_DIR / f"hop_{name}_path.png", title=f"{name} path")
            snap = snapshot_env(env)
            still_in = int(snap.room_base_id) == room and snap.indoors
            push = None
            if still_in:
                push = push_cardinal(env, pushes[name], room=room, max_frames=360)
                save_overlay(
                    env,
                    OUT_DIR / f"hop_{name}_push.png",
                    title=f"{name} {pushes[name]} {push['room_hex']} {push['xy']}",
                )
            report["routes"][name] = {
                "start": {"xy": [start["link_x"], start["link_y"]], "room": start["room_hex"]},
                "chain": chain,
                "push": push,
            }
            last = chain[-1] if chain else {}
            print(
                f"{name}: last={last.get('label')} xy={last.get('xy')} "
                f"room={last.get('room_hex')} in={last.get('indoors')} "
                f"push={None if push is None else (push['room_hex'], push['xy'], push['left_room'])}"
            )
        save_json(OUT_DIR / "hop.json", report)
        print(f"Wrote {OUT_DIR / 'hop.json'}")
    finally:
        env.close()
    return 0


def cmd_edges() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = build_boot_env(PIN_NAME, render_mode="rgb_array")
    report: dict[str, Any] = {"when": _utc(), "pin": PIN_NAME, "rays": {}}
    try:
        env.reset()
        load_pin(env, PIN_NAME)
        settle = settle_control(env)
        start = glance(env)
        report["start"] = start
        save_overlay(env, OUT_DIR / "pin_spawn.png", title="DesertPalaceEntry spawn")
        room = int(start["room_base_id"])
        print(
            f"pin room={start['room_hex']} xy=({start['link_x']},{start['link_y']}) "
            f"mod={start['module_hex']}/{start['submodule']} ctrl={start['has_control']}"
        )
        if room != HYPOTHESIS_ROOM or not start["indoors"]:
            save_json(OUT_DIR / "edges.json", report)
            print("pin is not hypothesis room 0x84")
            return 3

        for direction in CARDINALS:
            load_pin(env, PIN_NAME)
            settle_control(env)
            rec = push_cardinal(env, direction, room=room)
            report["rays"][direction] = rec
            save_overlay(
                env,
                OUT_DIR / f"ray_{direction.lower()}.png",
                title=f"{direction} {rec['room_hex']} {rec['xy']}",
            )
            print(
                f"  {direction:5} frames={rec['frames']:4} left={rec['left_room']} "
                f"room={rec['room_hex']} xy={rec['xy']} edge={rec['edge']}"
            )

        load_pin(env, PIN_NAME)
        settle_control(env)
        report["north_scan"] = walk_north_scan(env, room=room)
        save_overlay(env, OUT_DIR / "north_scan.png", title="north scan end")
        save_json(OUT_DIR / "edges.json", report)
        print(f"Wrote {OUT_DIR / 'edges.json'}")
    finally:
        env.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", action="store_true", help="glance source states only")
    parser.add_argument("--edges", action="store_true", help="cardinal rays from saved pin")
    parser.add_argument("--hop", action="store_true", help="alcove → hall → first room door")
    args = parser.parse_args()
    if args.scan:
        return cmd_scan()
    if args.edges:
        return cmd_edges()
    if args.hop:
        return cmd_hop()
    return cmd_pin()


if __name__ == "__main__":
    raise SystemExit(main())

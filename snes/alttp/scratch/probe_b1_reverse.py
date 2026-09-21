"""Reverse-walk B1 from CastleB2Landing / CastleB1* pins toward F1 stairs.

Usage:
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_b1_reverse.py
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_b1_reverse.py --no-scan
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
from alttp.primitives import Waypoint, move_to, settle_control
from alttp.ram import EQUIP_SWORD, FOLLOWER, LINK_ITEM_LAMP, NUM_KEYS, room_label, wram_index
from alttp.room_sense import detect_edge
from alttp.startup import action_for, build_boot_env, snapshot_env, step_frames
from retro_harness.env import resync_custom_state

OUT_DIR = RECORDINGS_DIR / "probe_b1_reverse"
F1_ROOMS = frozenset({0x01, 0x50, 0x51, 0x52, 0x60, 0x61, 0x62})
B1_ROOMS = frozenset({0x70, 0x71, 0x72, 0x80, 0x81, 0x82})
DIRS = ("UP", "DOWN", "LEFT", "RIGHT")
SCAN_STATES = (
    "CastleB2Landing",
    "CastleB1Guard",
    "CastleB1UpperCleared",
    "CastleB1FarWest",
    "CastleB1FarDoor",
    "CastleB1Key",
    "CastleB1Pit",
    "CastleB1South",
    "CastleB1West",
    "CastleB1WestRoom",
    "CastleB1Bridge",
)
OFFSETS = ((0, 0), (-48, 0), (48, 0), (0, -48), (0, 48), (-96, 0), (96, 0), (0, -96))


def leftover_bytes(env: object) -> dict[str, int]:
    ram = env.get_ram()  # type: ignore[attr-defined]
    return {
        "F359_sword": int(ram[wram_index(EQUIP_SWORD)]),
        "F34A_lamp": int(ram[wram_index(LINK_ITEM_LAMP)]),
        "F36F_keys": int(ram[wram_index(NUM_KEYS)]),
        "F3CC_follower": int(ram[wram_index(FOLLOWER)]),
    }


def glance(env: object, settle: Any | None = None) -> dict[str, Any]:
    snap = snapshot_env(env)
    rec: dict[str, Any] = {
        "module": int(snap.game_mode),
        "module_hex": f"0x{int(snap.game_mode):02X}",
        "submodule": int(snap.submodule),
        "indoors": bool(snap.indoors),
        "room_base_id": int(snap.room_base_id),
        "room_hex": f"0x{int(snap.room_base_id):02X}",
        "room_label": room_label(snap.room_base_id),
        "link_x": int(snap.link_x),
        "link_y": int(snap.link_y),
        "sword": int(snap.sword_level),
        "lamp": int(snap.lamp_level),
        "keys": int(snap.num_keys),
        "follower": int(snap.follower),
        "has_control": bool(snap.has_control),
        "leftover": leftover_bytes(env),
    }
    if settle is not None:
        rec["settle_ok"] = bool(settle.ok)
        rec["settle_frames"] = int(settle.frames)
    return rec


def discover_pins() -> list[str]:
    names = sorted(
        p.stem
        for p in INTEGRATION_DIR.glob("*.state")
        if p.is_file() and (p.stem.startswith("CastleB1") or p.stem == "CastleB2Landing")
    )
    if "CastleB2Landing" not in names:
        raise FileNotFoundError("CastleB2Landing.state missing")
    return names


def load_pin(env: object, name: str) -> None:
    if not resync_custom_state(env, GAME_DIR, INTEGRATION, name):
        raise RuntimeError(f"failed to load {name}")


def hold_watch(env: object, direction: str, *, max_frames: int = 720, step: int = 4) -> dict[str, Any]:
    """Hold one direction; log room/submodule/xy jumps. Stops on F1 or idle control."""
    start = snapshot_env(env)
    events: list[dict[str, Any]] = []
    prev = (start.room_base_id, start.submodule, start.link_x, start.link_y)
    idle = 0
    frames = 0
    f1: dict[str, Any] | None = None
    while frames < max_frames:
        step_frames(env, action_for(direction), step)
        frames += step
        snap = snapshot_env(env)
        cur = (snap.room_base_id, snap.submodule, snap.link_x, snap.link_y)
        if cur != prev:
            idle = 0
            ev = {
                "frames": frames,
                "dir": direction,
                "room": f"0x{snap.room_base_id:02X}",
                "sub": int(snap.submodule),
                "xy": [snap.link_x, snap.link_y],
                "ctrl": bool(snap.has_control),
                "indoors": bool(snap.indoors),
                "f1": snap.room_base_id in F1_ROOMS,
            }
            events.append(ev)
            print(
                f"  hold {direction} f={frames:4d} room={ev['room']} sub={ev['sub']} "
                f"xy=({snap.link_x},{snap.link_y}) ctrl={int(snap.has_control)}"
            )
            if snap.room_base_id in F1_ROOMS:
                f1 = ev
                break
            if snap.has_control and snap.room_base_id != start.room_base_id:
                break
        elif snap.has_control:
            idle += 1
            if idle >= 8:
                break
        prev = cur
    return {
        "dir": direction,
        "frames": frames,
        "start": {"room": f"0x{start.room_base_id:02X}", "xy": [start.link_x, start.link_y]},
        "end": glance(env),
        "events": events,
        "f1": f1,
    }


def ray(env: object, direction: str, *, max_frames: int = 240, step: int = 4) -> dict[str, Any]:
    before = snapshot_env(env)
    room = before.room_base_id
    prev = (before.link_x, before.link_y)
    stuck = 0
    frames = 0
    while frames < max_frames:
        step_frames(env, action_for(direction), step)
        frames += step
        after = snapshot_env(env)
        edge = detect_edge(
            before, after, expected_room=room, frames=frames, preferred_direction=direction
        )
        if edge is not None:
            dest = 0 if not after.indoors else after.room_base_id
            return {
                "dir": direction,
                "frames": frames,
                "from_xy": [before.link_x, before.link_y],
                "to_xy": [after.link_x, after.link_y],
                "from_room": f"0x{room:02X}",
                "to_room": None if not after.indoors else f"0x{dest:02X}",
                "f1": dest in F1_ROOMS,
                "b1": dest in B1_ROOMS,
                "stuck": False,
            }
        xy = (after.link_x, after.link_y)
        stuck = stuck + 1 if xy == prev else 0
        prev = xy
        if stuck >= 10:
            break
        before = after
    snap = snapshot_env(env)
    return {
        "dir": direction,
        "frames": frames,
        "from_xy": [before.link_x, before.link_y],
        "to_xy": [snap.link_x, snap.link_y],
        "from_room": f"0x{room:02X}",
        "to_room": f"0x{snap.room_base_id:02X}",
        "f1": False,
        "b1": snap.room_base_id in B1_ROOMS,
        "stuck": True,
    }


def scan_pose(env: object) -> list[dict[str, Any]]:
    origin = env.em.get_state()  # type: ignore[attr-defined]
    spawn = snapshot_env(env)
    hits: list[dict[str, Any]] = []
    for dx, dy in OFFSETS:
        env.em.set_state(origin)  # type: ignore[attr-defined]
        if dx or dy:
            move_to(
                env,
                Waypoint(spawn.link_x + dx, spawn.link_y + dy, tolerance=12),
                max_frames=160,
            )
        pose = env.em.get_state()  # type: ignore[attr-defined]
        here = snapshot_env(env)
        if here.room_base_id != spawn.room_base_id:
            continue
        for direction in DIRS:
            env.em.set_state(pose)  # type: ignore[attr-defined]
            rec = ray(env, direction)
            rec["offset"] = [dx, dy]
            rec["pose_xy"] = [here.link_x, here.link_y]
            hits.append(rec)
    env.em.set_state(origin)  # type: ignore[attr-defined]
    return hits


def exits_only(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for hit in hits:
        if hit.get("stuck") or hit.get("to_room") == hit.get("from_room"):
            continue
        key = (str(hit.get("from_room")), str(hit.get("dir")), str(hit.get("to_room")))
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
    return out


def holds_from_here(env: object) -> list[dict[str, Any]]:
    origin = env.em.get_state()  # type: ignore[attr-defined]
    holds: list[dict[str, Any]] = []
    for direction in DIRS:
        env.em.set_state(origin)  # type: ignore[attr-defined]
        settle_control(env)
        holds.append(hold_watch(env, direction))
    env.em.set_state(origin)  # type: ignore[attr-defined]
    return holds


def record_holds(env: object, name: str, start: dict[str, Any]) -> dict[str, Any]:
    holds = holds_from_here(env)
    f1 = [h for h in holds if h.get("f1")]
    exits = []
    for h in holds:
        end = h["end"]
        if end["room_hex"] != start["room_hex"]:
            exits.append(
                {
                    "dir": h["dir"],
                    "from_room": start["room_hex"],
                    "to_room": end["room_hex"],
                    "to_xy": [end["link_x"], end["link_y"]],
                    "f1": end["room_base_id"] in F1_ROOMS,
                }
            )
    print(f"  exits={[(e['dir'], e['to_room']) for e in exits]} f1={len(f1)}")
    return {"state": name, "start": start, "holds": holds, "exits": exits, "f1": f1}


def census_pins(env: object, names: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name in names:
        load_pin(env, name)
        settle = settle_control(env)
        rec = glance(env, settle)
        rec["state"] = name
        records.append(rec)
        print(
            f"{name:28} room {rec['room_hex']} xy=({rec['link_x']},{rec['link_y']}) "
            f"mod={rec['module_hex']}/{rec['submodule']} ctrl={int(rec['has_control'])} "
            f"sword={rec['sword']} keys={rec['keys']} F3CC={rec['follower']}"
        )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B1 reverse-walk toward F1 stairs.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--no-scan", action="store_true")
    parser.add_argument("--landing", action="store_true", help="Scan settled 0x71 spiral landing only.")
    args = parser.parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    names = discover_pins()
    print(f"pins: {len(names)}")
    env = build_boot_env(names[0], render_mode="rgb_array")
    try:
        env.reset()
        pins = census_pins(env, names)
        (out_dir / "pin_census.json").write_text(
            json.dumps(
                {
                    "schema": "alttp_b1_reverse_pin_census",
                    "schemaVersion": 1,
                    "measured": datetime.now(timezone.utc).isoformat(),
                    "source": "state_load_dev",
                    "poked": False,
                    "pinCount": len(pins),
                    "pins": pins,
                },
                indent=2,
            )
            + "\n"
        )
        scans: list[dict[str, Any]] = []
        f1_hits: list[dict[str, Any]] = []
        if args.landing:
            load_pin(env, "CastleB2Landing")
            settle_control(env)
            spiral = hold_watch(env, "UP", max_frames=400)
            land = glance(env)
            print(
                f"spiral landing room={land['room_hex']} xy=({land['link_x']},{land['link_y']}) "
                f"ctrl={int(land['has_control'])} sub={land['submodule']}"
            )
            (out_dir / "spiral_70_to_71.json").write_text(json.dumps(spiral, indent=2) + "\n")
            if land["room_base_id"] != 0x71 or not land["has_control"]:
                print("HALT: spiral did not settle in 0x71 with control")
                return 1
            hits = scan_pose(env)
            exits = exits_only(hits)
            f1 = [h for h in hits if h.get("f1")]
            scans.append(
                {
                    "state": "CastleB2Landing+spiral_0x71",
                    "start": land,
                    "exits": exits,
                    "f1": f1,
                    "rays": len(hits),
                }
            )
            f1_hits.extend({"state": "spiral_0x71", **h} for h in f1)
            print(f"landing grid exits={[(e['dir'], e['to_room']) for e in exits]} f1={len(f1)}")
            extra = ("CastleB1SecondKey", "CastleB1SingleGreen", "CastleB1Guard", "CastleB1FarDoor")
            if not f1:
                for name in extra:
                    if name not in names:
                        continue
                    load_pin(env, name)
                    settle_control(env)
                    start = glance(env)
                    print(f"hold {name} room {start['room_hex']} xy=({start['link_x']},{start['link_y']})")
                    rec = record_holds(env, name, start)
                    scans.append(rec)
                    f1_hits.extend({"state": name, **h["f1"], "end": h["end"]} for h in rec["holds"] if h.get("f1"))
                    if rec["f1"]:
                        print("F1 STAIR HIT — halt")
                        break
        elif not args.no_scan:
            for name in SCAN_STATES:
                if name not in names:
                    continue
                load_pin(env, name)
                settle_control(env)
                start = glance(env)
                print(f"hold {name} room {start['room_hex']} xy=({start['link_x']},{start['link_y']})")
                rec = record_holds(env, name, start)
                scans.append(rec)
                f1_hits.extend({"state": name, **h["f1"], "end": h["end"]} for h in rec["holds"] if h.get("f1"))
                if rec["f1"]:
                    print("F1 STAIR HIT — halt")
                    break
        payload = {
            "schema": "alttp_b1_reverse_scan",
            "schemaVersion": 1,
            "measured": datetime.now(timezone.utc).isoformat(),
            "source": "state_load_dev",
            "poked": False,
            "f1Rooms": [f"0x{r:02X}" for r in sorted(F1_ROOMS)],
            "f1Hits": f1_hits,
            "scans": [
                {k: v for k, v in s.items() if k != "holds"} | {"holdCount": len(s.get("holds", []))}
                for s in scans
            ],
            "exits": [e for s in scans for e in s.get("exits", [])],
        }
        (out_dir / "scan.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {out_dir} f1Hits={len(f1_hits)}")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())

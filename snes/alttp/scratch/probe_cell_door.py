"""Isolate 0x81 west_to_0x80 (Zelda cell) from a keys>=1 / already-open pin.

Does not poke keys / $F3CC. Does not drop the 0x72 north ledge.

Usage:
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_cell_door.py
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from alttp.paths import GAME_DIR, INTEGRATION, RECORDINGS_DIR
from alttp.primitives import (
    SPRITE_SMALL_KEY,
    Waypoint,
    active_sprites,
    fight_nearby,
    move_to,
    settle_control,
    sprites_of_type,
)
from alttp.ram import (
    EQUIP_SWORD,
    FOLLOWER,
    LINK_HP,
    LINK_ITEM_LAMP,
    LINK_MAX_HP,
    NUM_KEYS,
    room_label,
    wram_index,
)
from alttp.screen_glance import leftover_from_snapshot
from alttp.startup import action_for, build_boot_env, snapshot_env, step_frames
from retro_harness.env import resync_custom_state

OUT_DIR = RECORDINGS_DIR / "probe_cell_door"
ROOM_81 = 0x81
ROOM_80 = 0x80
APPROACH = (608, 4168)
# Candidate 0x81-side pins from the work queue / last sitting.
CANDIDATE_PINS = (
    "CastleZeldaB1West",
    "CastleB1FarDoor",
    "CastleB1WestRoom",
    "CastleB1West",
    "CastleB1FarWest",
    "CastleB1GreenRoom",
    "CastleB1GreenRoomCleared",
    "CastleB1Shutter",
)
Y_SWEEP = tuple(range(4144, 4217, 8))


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


def glance(env: object) -> dict[str, Any]:
    snap = snapshot_env(env)
    sprites = [
        {"type": f"0x{s.sprite_type:02X}", "xy": [s.x, s.y], "hp": s.hp}
        for s in active_sprites(env)
    ]
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
        "boot": leftover_from_snapshot(snap),
        "sprites": sprites[:12],
        "keys_on_floor": [
            {"xy": [s.x, s.y]} for s in sprites_of_type(env, (SPRITE_SMALL_KEY,))
        ],
    }
    return rec


def load_pin(env: object, name: str) -> None:
    if not resync_custom_state(env, GAME_DIR, INTEGRATION, name):
        raise RuntimeError(f"failed to load {name}")


def walk(env: object, x: int, y: int, *, room: int | None = None, frames: int = 500) -> Any:
    return move_to(env, Waypoint(x, y, tolerance=12, room=room), max_frames=frames)


def axis_walk(env: object, x: int, y: int, *, room: int | None = None) -> Any:
    here = snapshot_env(env)
    r = walk(env, x, here.link_y, room=room)
    if not r.ok:
        return r
    return walk(env, x, y, room=room)


def hold_until_room(
    env: object, direction: str, dest: int | None, *, max_frames: int = 480
) -> dict[str, Any]:
    start = snapshot_env(env)
    start_keys = int(start.num_keys)
    frames = 0
    idle = 0
    prev = (start.link_x, start.link_y, start.room_base_id, start.submodule)
    events: list[dict[str, Any]] = []
    while frames < max_frames:
        step_frames(env, action_for(direction), 4)
        frames += 4
        snap = snapshot_env(env)
        cur = (snap.link_x, snap.link_y, snap.room_base_id, snap.submodule)
        if cur != prev or int(snap.num_keys) != start_keys:
            idle = 0
            events.append(
                {
                    "f": frames,
                    "room": f"0x{snap.room_base_id:02X}",
                    "sub": int(snap.submodule),
                    "xy": [snap.link_x, snap.link_y],
                    "keys": int(snap.num_keys),
                    "ctrl": bool(snap.has_control),
                }
            )
            if dest is not None and snap.room_base_id == dest and snap.has_control:
                break
            if snap.room_base_id != start.room_base_id and snap.has_control:
                break
        elif snap.has_control:
            idle += 1
            if idle >= 12:
                break
        prev = cur
    settle_control(env, max_frames=180)
    return {"dir": direction, "frames": frames, "events": events[-16:], "end": glance(env)}


def approach_west_wall(env: object) -> dict[str, Any]:
    """From wherever we are in 0x81, walk the west-wall cell-door band."""
    fight_nearby(env, room=ROOM_81, max_distance=90, max_cycles=80)
    settle_control(env)
    here = snapshot_env(env)
    if here.room_base_id != ROOM_81:
        return {"ok": False, "reason": f"left 0x81 during fight now=0x{here.room_base_id:02X}", "pose": glance(env)}
    r1 = axis_walk(env, 632, here.link_y, room=ROOM_81)
    r2 = axis_walk(env, 632, APPROACH[1], room=ROOM_81)
    r3 = walk(env, APPROACH[0], APPROACH[1], room=ROOM_81, frames=240)
    pose = glance(env)
    ok = pose["room_base_id"] == ROOM_81 and pose["link_x"] <= 640
    return {
        "ok": ok,
        "reason": f"{r1.reason}; {r2.reason}; {r3.reason}",
        "pose": pose,
    }


def try_left_at_wall(env: object) -> dict[str, Any]:
    """LEFT at map approach, then slide the west wall if still in 0x81."""
    attempts: list[dict[str, Any]] = []
    first = hold_until_room(env, "LEFT", ROOM_80, max_frames=360)
    attempts.append({"label": "left_approach", **first})
    end = first["end"]
    if end["room_base_id"] == ROOM_80:
        return {"opened": True, "attempts": attempts, "end": end}
    # Slide north then south along the wall while holding LEFT.
    for y in Y_SWEEP:
        here = snapshot_env(env)
        if here.room_base_id != ROOM_81:
            break
        walk(env, APPROACH[0], y, room=ROOM_81, frames=180)
        h = hold_until_room(env, "LEFT", ROOM_80, max_frames=200)
        attempts.append({"label": f"left_y{y}", "pose_y": y, **h})
        if h["end"]["room_base_id"] == ROOM_80:
            return {"opened": True, "attempts": attempts, "end": h["end"]}
    return {"opened": False, "attempts": attempts, "end": glance(env)}


def census_one(env: object, name: str) -> dict[str, Any]:
    load_pin(env, name)
    settle_control(env)
    rec = glance(env)
    rec["state"] = name
    print(
        f"{name:28} room {rec['room_hex']} xy=({rec['link_x']},{rec['link_y']}) "
        f"keys={rec['keys']} F3CC={rec['follower']} "
        f"mod={rec['module_hex']}/{rec['submodule']} "
        f"floor_keys={rec['keys_on_floor']}"
    )
    return rec


def isolate_from_pin(env: object, name: str) -> dict[str, Any]:
    load_pin(env, name)
    settle_control(env)
    start = glance(env)
    print(
        f"TRY {name} start {start['room_hex']} ({start['link_x']},{start['link_y']}) "
        f"keys={start['keys']} F3CC={start['follower']}"
    )
    if start["room_base_id"] != ROOM_81:
        return {
            "pin": name,
            "start": start,
            "skipped": True,
            "reason": f"not on 0x81 (room {start['room_hex']})",
            "opened": False,
        }
    approach = approach_west_wall(env)
    print(
        f"  approach ok={approach['ok']} "
        f"xy=({approach['pose']['link_x']},{approach['pose']['link_y']}) "
        f"keys={approach['pose']['keys']}"
    )
    result = try_left_at_wall(env)
    end = result["end"]
    print(
        f"  LEFT → {end['room_hex']} ({end['link_x']},{end['link_y']}) "
        f"keys={end['keys']} F3CC={end['follower']} opened={result['opened']}"
    )
    return {
        "pin": name,
        "start": start,
        "approach": approach,
        "opened": result["opened"],
        "attempts": result["attempts"],
        "end": end,
    }


def _maybe_open_cell(env: object) -> dict[str, Any] | None:
    here = snapshot_env(env)
    if here.room_base_id != ROOM_81:
        return None
    approach = approach_west_wall(env)
    cell = try_left_at_wall(env)
    cell["approach"] = approach
    print(
        f"  cell LEFT → {cell['end']['room_hex']} "
        f"({cell['end']['link_x']},{cell['end']['link_y']}) "
        f"keys={cell['end']['keys']} opened={cell['opened']}"
    )
    return cell


def second_key_walk(env: object) -> dict[str, Any]:
    """Bounded walk of CastleB1SecondKey (0x71 keys=1) toward 0x81. No 0x72 drop."""
    origin_holds: list[dict[str, Any]] = []
    load_pin(env, "CastleB1SecondKey")
    settle_control(env)
    start = glance(env)
    origin = env.em.get_state()  # type: ignore[attr-defined]
    print(
        f"SecondKey start {start['room_hex']} ({start['link_x']},{start['link_y']}) "
        f"keys={start['keys']}"
    )
    for direction in ("DOWN", "LEFT", "RIGHT", "UP"):
        env.em.set_state(origin)  # type: ignore[attr-defined]
        settle_control(env)
        h = hold_until_room(env, direction, ROOM_81, max_frames=400)
        origin_holds.append(h)
        e = h["end"]
        print(
            f"  {direction:5} → {e['room_hex']} ({e['link_x']},{e['link_y']}) "
            f"keys={e['keys']} sub={e['submodule']}"
        )
        if e["room_base_id"] == ROOM_81:
            cell = _maybe_open_cell(env)
            return {
                "pin": "CastleB1SecondKey",
                "start": start,
                "holds": origin_holds,
                "detours": [],
                "cell": cell,
                "opened": bool(cell and cell["opened"]),
                "end": cell["end"] if cell else e,
            }

    detours: list[dict[str, Any]] = []
    for label, x, y in (
        ("west_column", 632, start["link_y"]),
        ("south_lip", start["link_x"], 4100),
        ("north_around", start["link_x"], 3856),
        ("nw_column", 632, 3856),
        ("sw_column", 632, 4100),
        ("mid_west", 768, start["link_y"]),
        ("west_832", 832, start["link_y"]),
    ):
        env.em.set_state(origin)  # type: ignore[attr-defined]
        settle_control(env)
        fight_nearby(env, room=0x71, max_distance=80, max_cycles=40)
        r = axis_walk(env, x, y, room=0x71)
        pose = glance(env)
        rec = {"label": label, "target": [x, y], "ok": r.ok, "reason": r.reason, "pose": pose}
        print(
            f"  {label:12} → {pose['room_hex']} ({pose['link_x']},{pose['link_y']}) "
            f"keys={pose['keys']} ok={r.ok}"
        )
        if pose["room_base_id"] == ROOM_81:
            cell = _maybe_open_cell(env)
            rec["cell"] = cell
            detours.append(rec)
            return {
                "pin": "CastleB1SecondKey",
                "start": start,
                "holds": origin_holds,
                "detours": detours,
                "cell": cell,
                "opened": bool(cell and cell["opened"]),
                "end": cell["end"] if cell else pose,
            }
        if pose["link_x"] <= 700 and pose["room_base_id"] == 0x71:
            south = hold_until_room(env, "DOWN", ROOM_81, max_frames=360)
            rec["south"] = south
            print(
                f"    DOWN → {south['end']['room_hex']} "
                f"({south['end']['link_x']},{south['end']['link_y']}) "
                f"keys={south['end']['keys']}"
            )
            if south["end"]["room_base_id"] == ROOM_81:
                cell = _maybe_open_cell(env)
                rec["cell"] = cell
                detours.append(rec)
                return {
                    "pin": "CastleB1SecondKey",
                    "start": start,
                    "holds": origin_holds,
                    "detours": detours,
                    "cell": cell,
                    "opened": bool(cell and cell["opened"]),
                    "end": cell["end"] if cell else south["end"],
                }
        detours.append(rec)
    return {
        "pin": "CastleB1SecondKey",
        "start": start,
        "holds": origin_holds,
        "detours": detours,
        "cell": None,
        "opened": False,
        "end": detours[-1]["pose"] if detours else start,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Isolate 0x81 west_to_0x80")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--pin", action="append", default=[], help="Only these 0x81 pins")
    parser.add_argument(
        "--second-key",
        action="store_true",
        help="Walk CastleB1SecondKey (0x71 keys=1) toward 0x81",
    )
    parser.add_argument("--census", action="store_true", help="Load/glance pins; do not walk")
    args = parser.parse_args(argv)
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    pins = tuple(args.pin) if args.pin else (
        CANDIDATE_PINS if (args.census or not args.second_key) else ()
    )
    boot = pins[0] if pins else "CastleB1SecondKey"
    env = build_boot_env(boot, render_mode="rgb_array")
    census: list[dict[str, Any]] = []
    tries: list[dict[str, Any]] = []
    opened: dict[str, Any] | None = None
    try:
        env.reset()
        for name in pins:
            census.append(census_one(env, name))
        if args.second_key or not args.pin:
            census.append(census_one(env, "CastleB1SecondKey"))
        key_pins = [r for r in census if r["keys"] >= 1 and r["keys"] < 0xFF]
        on_81 = [r for r in census if r["room_base_id"] == ROOM_81]
        print(
            f"census: {len(census)} pins, on_0x81={len(on_81)}, "
            f"keys>=1={[r['state'] for r in key_pins]}"
        )
        second = None
        if not args.census:
            for name in pins:
                rec = isolate_from_pin(env, name)
                tries.append(rec)
                if rec.get("opened"):
                    opened = rec
                    break
            if opened is None and (args.second_key or not args.pin):
                second = second_key_walk(env)
                if second.get("opened"):
                    opened = second
        if opened:
            end = opened["end"]
        elif second is not None:
            end = second["end"]
        elif tries:
            end = tries[-1]["end"]
        else:
            end = {}
        payload = {
            "schema": "alttp_cell_door",
            "schemaVersion": 1,
            "measured": datetime.now(timezone.utc).isoformat(),
            "source": "state_load_dev",
            "poked": False,
            "approach": list(APPROACH),
            "census": census,
            "tries": tries,
            "secondKey": second,
            "opened": opened is not None,
            "openPin": opened["pin"] if opened else None,
            "end": end,
        }
        if args.census:
            (out / "pin_census.json").write_text(
                json.dumps(
                    {
                        "schema": "alttp_cell_door_pin_census",
                        "schemaVersion": 1,
                        "measured": payload["measured"],
                        "source": "state_load_dev",
                        "poked": False,
                        "pins": census,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        if second is not None:
            (out / "second_key.json").write_text(
                json.dumps(second, indent=2) + "\n", encoding="utf-8"
            )
        if len(tries) == 1:
            (out / f"{tries[0]['pin']}.json").write_text(
                json.dumps(tries[0], indent=2) + "\n", encoding="utf-8"
            )
        path = out / "cell_door.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(
            f"wrote {path} opened={payload['opened']} pin={payload['openPin']} "
            f"end_room={end.get('room_hex')} xy=({end.get('link_x')},{end.get('link_y')}) "
            f"keys={end.get('keys')} F3CC={end.get('follower')}"
        )
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())

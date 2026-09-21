"""Isolate 0x81 west_to_0x80 from CastleB1SecondKey leftover (keys=1).

Replay rr-ccxt.18: NW lip UP+LEFT through the 0x71 west door, south_to_0x81
DOWN, then LEFT at west_to_0x80_approach. Does not poke keys / $F3CC.
Does not mash Zelda dialogue (this is not a rescue).

Usage:
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_cell_key.py
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
from alttp.primitives import Waypoint, active_sprites, move_to, settle_control
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
from alttp.room_map import load_room_map
from alttp.room_sense import overlay_from_env
from alttp.screen_glance import leftover_from_snapshot
from alttp.startup import action_for, build_boot_env, no_action, snapshot_env, step_frames
from retro_harness.env import resync_custom_state

OUT_DIR = RECORDINGS_DIR / "probe_cell_key"
PIN = "CastleB1SecondKey"
ROOM_71 = 0x71
ROOM_81 = 0x81
ROOM_80 = 0x80
# maps/room_71.json — sibling wrap
NW_LIP = (832, 3976)
WRAP_XY = (688, 3960)
SOUTH_APPROACH = (632, 4100)
# maps/room_81.json west_to_0x80
CELL_APPROACH = (608, 4168)
OPEN_X = 688
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
        "F366_bigkey1": int(ram[wram_index(0xF366)]),
        "F367_bigkey2": int(ram[wram_index(0xF367)]),
        "00EE_layer": int(ram[0x00EE]),
        "040C_dungeon": int(ram[0x040C]),
        "0400_doors": int(ram[0x0400]),
        "0401_doors": int(ram[0x0401]),
        "2F_dir": int(ram[0x2F]),
        "5D_act": int(ram[0x5D]),
    }


def glance(env: object) -> dict[str, Any]:
    snap = snapshot_env(env)
    rec = leftover_from_snapshot(snap)
    rec.update(
        {
            "module_hex": f"0x{int(snap.game_mode):02X}",
            "room_hex": f"0x{int(snap.room_base_id):02X}",
            "room_label": room_label(snap.room_base_id),
            "link_x": int(snap.link_x),
            "link_y": int(snap.link_y),
            "link_action": int(snap.link_action),
            "has_control": bool(snap.has_control),
            "lamp": int(snap.lamp_level),
            "leftover": leftover_bytes(env),
            "sprites": [
                {"type": f"0x{s.sprite_type:02X}", "xy": [s.x, s.y], "hp": s.hp}
                for s in active_sprites(env)
            ][:12],
        }
    )
    return rec


def brief(g: dict[str, Any]) -> str:
    return (
        f"{g['room_hex']} ({g['link_x']},{g['link_y']}) keys={g['keys']} "
        f"sub={g['submodule']} F3CC={g['follower']} ctrl={g['has_control']}"
    )


def save_overlay(env: object, path: Path, title: str, *, map_id: str) -> None:
    m = load_room_map(map_id)
    img = overlay_from_env(env, include_all_sprites=True, points=m.points, title=title)
    Image.fromarray(img).save(path)


def walk(env: object, x: int, y: int, *, room: int | None, frames: int = 500) -> Any:
    return move_to(env, Waypoint(x, y, tolerance=6, room=room), max_frames=frames)


def hold(
    env: object,
    buttons: tuple[str, ...],
    *,
    max_frames: int = 280,
    stop_room: int | None = None,
    stop_x_le: int | None = None,
) -> dict[str, Any]:
    start = snapshot_env(env)
    start_keys = int(start.num_keys)
    frames = 0
    idle = 0
    prev = (start.link_x, start.link_y, start.room_base_id, start.submodule, start.game_mode)
    events: list[dict[str, Any]] = []
    action = no_action() if not buttons or buttons == ("NONE",) else action_for(*buttons)
    while frames < max_frames:
        step_frames(env, action, 4)
        frames += 4
        snap = snapshot_env(env)
        cur = (snap.link_x, snap.link_y, snap.room_base_id, snap.submodule, snap.game_mode)
        if cur != prev or int(snap.num_keys) != start_keys:
            idle = 0
            events.append(
                {
                    "f": frames,
                    "room": f"0x{snap.room_base_id:02X}",
                    "sub": int(snap.submodule),
                    "act": int(snap.link_action),
                    "xy": [snap.link_x, snap.link_y],
                    "keys": int(snap.num_keys),
                    "ctrl": bool(snap.has_control),
                    "F3CC": int(snap.follower),
                }
            )
            if snap.game_mode == 0x12:
                break
            if stop_room is not None and snap.room_base_id == stop_room and snap.has_control:
                break
            if (
                stop_x_le is not None
                and snap.room_base_id == start.room_base_id
                and snap.link_x <= stop_x_le
                and snap.num_keys >= 1
                and snap.has_control
            ):
                break
            if snap.room_base_id != start.room_base_id and snap.has_control:
                break
        elif snap.has_control:
            idle += 1
            if idle >= 12:
                break
        prev = cur
    settle_control(env, max_frames=180)
    return {"buttons": list(buttons), "frames": frames, "events": events[-20:], "end": glance(env)}


def _write(out: Path, payload: dict[str, Any]) -> None:
    path = out / "leftover.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    end = payload.get("end") or payload.get("start") or {}
    print(
        f"HALT {payload.get('halt')} leftover {brief(end) if end else '?'} "
        f"opened={payload.get('opened')} wrote {path}"
    )


def replay_wrap(env: object, out: Path) -> dict[str, Any]:
    """Sibling acts: NW lip (832,3976) then UP+LEFT through y=3960."""
    r = walk(env, NW_LIP[0], NW_LIP[1], room=ROOM_71)
    lip = glance(env)
    rec: dict[str, Any] = {
        "act": "nw_lip",
        "ok": r.ok,
        "reason": r.reason,
        "end": lip,
    }
    print(f"  nw_lip {NW_LIP} → {brief(lip)} {r.reason}")
    if lip["room"] != ROOM_71 or lip["keys"] < 1:
        rec["miss"] = "not_0x71_keys"
        return rec
    h = hold(env, ("UP", "LEFT"), max_frames=280, stop_x_le=OPEN_X)
    wrap = h["end"]
    rec["hold"] = h
    rec["wrap"] = wrap
    opened = wrap["room"] == ROOM_71 and wrap["keys"] >= 1 and wrap["link_x"] <= OPEN_X
    rec["opened"] = opened
    print(f"  UP+LEFT → {brief(wrap)} opened={opened}")
    if opened:
        save_overlay(env, out / "wrap.png", f"wrap {brief(wrap)}", map_id="room_71")
    return rec


def south_to_81(env: object, out: Path) -> dict[str, Any]:
    here = snapshot_env(env)
    r1 = walk(env, SOUTH_APPROACH[0], here.link_y, room=ROOM_71)
    r2 = walk(env, SOUTH_APPROACH[0], SOUTH_APPROACH[1], room=ROOM_71)
    pose = glance(env)
    push = hold(env, ("DOWN",), max_frames=240, stop_room=ROOM_81)
    end = push["end"]
    rec = {
        "act": "south_to_0x81",
        "walk": [r1.reason, r2.reason],
        "approach": pose,
        "push": push,
        "end": end,
        "ok": end["room"] == ROOM_81 and end["keys"] >= 1,
    }
    print(f"  south → {brief(end)} ok={rec['ok']}")
    if rec["ok"]:
        save_overlay(env, out / "room_81.png", f"0x81 {brief(end)}", map_id="room_81")
    return rec


def approach_cell(env: object) -> dict[str, Any]:
    here = snapshot_env(env)
    r1 = walk(env, 632, here.link_y, room=ROOM_81)
    r2 = walk(env, 632, CELL_APPROACH[1], room=ROOM_81)
    r3 = walk(env, CELL_APPROACH[0], CELL_APPROACH[1], room=ROOM_81, frames=240)
    pose = glance(env)
    ok = pose["room"] == ROOM_81 and pose["link_x"] <= 640 and pose["keys"] >= 1
    rec = {
        "act": "west_to_0x80_approach",
        "ok": ok,
        "reason": f"{r1.reason}; {r2.reason}; {r3.reason}",
        "end": pose,
    }
    print(f"  approach {CELL_APPROACH} → {brief(pose)} ok={ok}")
    return rec


def try_left(env: object) -> dict[str, Any]:
    """LEFT at map approach. If still 0x81, slide the west wall (locked vs y)."""
    attempts: list[dict[str, Any]] = []
    first = hold(env, ("LEFT",), max_frames=360, stop_room=ROOM_80)
    attempts.append({"label": "left_approach", **first})
    end = first["end"]
    print(f"  LEFT approach → {brief(end)}")
    if end["room"] == ROOM_80:
        return {"opened": True, "attempts": attempts, "end": end}
    for y in Y_SWEEP:
        here = snapshot_env(env)
        if here.room_base_id != ROOM_81:
            break
        walk(env, CELL_APPROACH[0], y, room=ROOM_81, frames=180)
        h = hold(env, ("LEFT",), max_frames=200, stop_room=ROOM_80)
        attempts.append({"label": f"left_y{y}", "pose_y": y, **h})
        print(f"  LEFT y={y} → {brief(h['end'])}")
        if h["end"]["room"] == ROOM_80:
            return {"opened": True, "attempts": attempts, "end": h["end"]}
    return {"opened": False, "attempts": attempts, "end": glance(env)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SecondKey leftover → 0x81 west_to_0x80")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    env = build_boot_env(PIN, render_mode="rgb_array")
    payload: dict[str, Any] = {
        "schema": "alttp_cell_key",
        "schemaVersion": 1,
        "measured": datetime.now(timezone.utc).isoformat(),
        "source": "state_load_dev",
        "poked": False,
        "pin": PIN,
        "nwLip": list(NW_LIP),
        "wrapXy": list(WRAP_XY),
        "southApproach": list(SOUTH_APPROACH),
        "cellApproach": list(CELL_APPROACH),
        "opened": False,
    }
    try:
        env.reset()
        if not resync_custom_state(env, GAME_DIR, INTEGRATION, PIN):
            raise RuntimeError(f"failed to load {PIN}")
        settle_control(env)
        start = glance(env)
        payload["start"] = start
        save_overlay(env, out / "pin.png", f"SecondKey {brief(start)}", map_id="room_71")
        print(f"PIN {brief(start)}")
        pin_ok = (
            start["room"] == ROOM_71
            and start["keys"] >= 1
            and start["module"] == 0x07
            and start["submodule"] == 0
            and start["follower"] == 0
            and abs(start["link_x"] - 904) <= 8
            and abs(start["link_y"] - 3988) <= 8
        )
        if not pin_ok:
            payload["halt"] = "pin"
            payload["end"] = start
            _write(out, payload)
            return 1

        wrap = replay_wrap(env, out)
        payload["wrap"] = wrap
        if not wrap.get("opened"):
            payload["halt"] = "wrap_miss"
            payload["end"] = wrap.get("wrap") or wrap["end"]
            _write(out, payload)
            return 0

        south = south_to_81(env, out)
        payload["south"] = south
        if not south["ok"]:
            payload["halt"] = "south_miss"
            payload["end"] = south["end"]
            _write(out, payload)
            return 0
        payload["room81"] = south["end"]

        approach = approach_cell(env)
        payload["approach"] = approach
        save_overlay(
            env, out / "approach.png", f"cell approach {brief(approach['end'])}", map_id="room_81"
        )
        if not approach["ok"]:
            payload["halt"] = "approach_miss"
            payload["end"] = approach["end"]
            _write(out, payload)
            return 0

        cell = try_left(env)
        payload["cell"] = cell
        end = cell["end"]
        payload["end"] = end
        payload["opened"] = bool(cell["opened"] and end["room"] == ROOM_80)
        # Glance only — do not A / mash Zelda. Follower must stay 0.
        if payload["opened"]:
            payload["halt"] = "ok"
            payload["landingXy"] = [end["link_x"], end["link_y"]]
            save_overlay(env, out / "land.png", f"0x80 {brief(end)}", map_id="room_80")
        else:
            payload["halt"] = "locked"
            save_overlay(env, out / "locked.png", f"locked {brief(end)}", map_id="room_81")
        print(
            f"LEFT dest={end['room_hex']} landing=({end['link_x']},{end['link_y']}) "
            f"keys={end['keys']} F3CC={end['follower']} sword={end['sword']}"
        )
        _write(out, payload)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())

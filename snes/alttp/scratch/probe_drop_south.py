"""From 0x72 key-door drop leftover, isolate south_to_0x82.

Replay CastleB1Key south key door (spends the key) → lower floor
(1272, 3945), then walk to south_door_approach and hold DOWN into 0x82.
Does not poke keys / $F3CC. Isolated only.

Usage:
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_drop_south.py
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

OUT_DIR = RECORDINGS_DIR / "probe_drop_south"
ROOM_72 = 0x72
ROOM_82 = 0x82
DEATH_MODULE = 0x12
PIT_SUB = 20
LIP_Y = 3776
LOWER_Y = 3944
DROP_XY = (1272, 3945)
PIT_GUARD = (1230, 3989)
ZELDA_PIT = (1184, 4016)  # west-wall corridor; south railing blocks x=1224
SOUTH_APPROACH = (1184, 4080)


def leftover_bytes(env: object) -> dict[str, int]:
    ram = env.get_ram()  # type: ignore[attr-defined]
    return {
        "F359_sword": int(ram[wram_index(EQUIP_SWORD)]),
        "F34A_lamp": int(ram[wram_index(LINK_ITEM_LAMP)]),
        "F36F_keys": int(ram[wram_index(NUM_KEYS)]),
        "F3CC_follower": int(ram[wram_index(FOLLOWER)]),
        "F36D_hp": int(ram[wram_index(LINK_HP)]),
        "F36C_max_hp": int(ram[wram_index(LINK_MAX_HP)]),
        "0400_doors": int(ram[0x0400]),
        "0401_doors": int(ram[0x0401]),
        "0403_roomflags": int(ram[0x0403]),
        "00EE_layer": int(ram[0x00EE]),
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
            "module_hex": f"0x{int(snap.game_mode):02X}",
            "room_hex": f"0x{int(snap.room_base_id):02X}",
            "room_label": room_label(snap.room_base_id),
            "link_x": int(snap.link_x),
            "link_y": int(snap.link_y),
            "link_action": int(snap.link_action),
            "has_control": bool(snap.has_control),
            "lamp": int(snap.lamp_level),
            "leftover": leftover_bytes(env),
            "sprites": sprites[:12],
        }
    )
    return rec


def save_overlay(env: object, path: Path, title: str, *, map_id: str) -> None:
    m = load_room_map(map_id)
    img = overlay_from_env(env, include_all_sprites=True, points=m.points, title=title)
    Image.fromarray(img).save(path)


def pit_or_dead(g: dict[str, Any]) -> str | None:
    if g["module"] == DEATH_MODULE or g["leftover"]["F36D_hp"] <= 0:
        return "death"
    if g["submodule"] == PIT_SUB:
        return "pit_submodule_20"
    return None


def watch(
    env: object,
    buttons: tuple[str, ...],
    *,
    max_frames: int,
    stop_on_key_spend: bool = False,
    stop_y: int | None = None,
    stop_room: int | None = None,
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
        if cur != prev:
            idle = 0
            events.append(
                {
                    "f": frames,
                    "room": f"0x{snap.room_base_id:02X}",
                    "mod": int(snap.game_mode),
                    "sub": int(snap.submodule),
                    "act": int(snap.link_action),
                    "xy": [snap.link_x, snap.link_y],
                    "ctrl": bool(snap.has_control),
                    "hp": leftover_bytes(env)["F36D_hp"],
                    "keys": int(snap.num_keys),
                }
            )
            if snap.game_mode == DEATH_MODULE:
                break
            if snap.submodule == PIT_SUB:
                break
            if stop_on_key_spend and (snap.num_keys < start_keys or snap.submodule == 4):
                break
            if stop_y is not None and snap.room_base_id == ROOM_72 and snap.link_y >= stop_y:
                break
            if stop_room is not None and snap.room_base_id == stop_room and snap.has_control:
                break
        elif snap.has_control:
            idle += 1
            if idle >= 12:
                break
        prev = cur
    if snapshot_env(env).submodule == PIT_SUB:
        settle_control(env, max_frames=400)
    return {"buttons": list(buttons), "frames": frames, "events": events[-16:], "end": glance(env)}


def axis_claim(
    env: object, label: str, x: int, y: int, *, room: int = ROOM_72, south_first: bool = True
) -> dict[str, Any]:
    """Axis-aligned claim. South-first keeps the pit (north of the drop) behind us."""
    before = glance(env)
    first = (before["x"], y) if south_first else (x, before["y"])
    r1 = move_to(env, Waypoint(first[0], first[1], tolerance=8, room=room, label=f"{label}_a"), max_frames=400)
    mid = glance(env)
    hit = pit_or_dead(mid)
    if hit or (not r1.ok and mid["room"] == room) or mid["module"] == DEATH_MODULE:
        return {
            "label": label,
            "target": [x, y],
            "ok": False,
            "miss": hit or r1.reason,
            "before": before,
            "end": mid,
            "phase": "a",
        }
    if mid["room"] == ROOM_82:
        return {
            "label": label,
            "target": [x, y],
            "ok": True,
            "miss": None,
            "before": before,
            "end": mid,
            "crossed_to_0x82": True,
            "phase": "a",
        }
    r2 = move_to(env, Waypoint(x, y, tolerance=8, room=room, label=label), max_frames=400)
    end = glance(env)
    hit = pit_or_dead(end)
    crossed = end["room"] == ROOM_82
    ok = crossed or (
        r2.ok
        and hit is None
        and end["room"] == room
        and abs(end["x"] - x) <= 12
        and abs(end["y"] - y) <= 12
    )
    return {
        "label": label,
        "target": [x, y],
        "ok": ok,
        "miss": None if ok else (hit or r2.reason),
        "before": before,
        "end": end,
        "reason": r2.reason,
        "frames": r1.frames + r2.frames,
        "crossed_to_0x82": crossed,
    }


def _write(out: Path, payload: dict[str, Any]) -> None:
    (out / "leftover.json").write_text(json.dumps(payload, indent=2) + "\n")
    end = payload.get("end") or payload.get("drop") or payload.get("start") or {}
    print(
        f"HALT {payload.get('halt')} leftover room={end.get('room_hex')} "
        f"xy=({end.get('link_x')},{end.get('link_y')}) "
        f"mod={end.get('module_hex')}/{end.get('submodule')} "
        f"hp={end.get('leftover', {}).get('F36D_hp')} keys={end.get('keys')} "
        f"F3CC={end.get('follower')} wrote {out / 'leftover.json'}"
    )


def replay_drop(env: object, out: Path, start: dict[str, Any]) -> dict[str, Any] | None:
    """Measured CastleB1Key acts: DOWN to lip, DOWN+LEFT unlock, DOWN through."""
    acts: list[dict[str, Any]] = []
    down = watch(env, ("DOWN",), max_frames=240, stop_y=LIP_Y)
    g1 = {"act": "down_lip", "watch": down, "end": down["end"]}
    acts.append(g1)
    save_overlay(env, out / "lip.png", "key-door lip", map_id="room_72")
    print(
        f"lip xy=({down['end']['link_x']},{down['end']['link_y']}) "
        f"keys={down['end']['keys']} room={down['end']['room_hex']}"
    )
    if down["end"]["room"] != ROOM_72 or down["end"]["keys"] < 1 or down["end"]["link_y"] < LIP_Y - 8:
        return {"halt": "keydoor_lip_miss", "start": start, "acts": acts, "end": down["end"]}

    unlock = watch(env, ("DOWN", "LEFT"), max_frames=200, stop_on_key_spend=True)
    settle_control(env, max_frames=200)
    after = glance(env)
    g2 = {"act": "unlock", "watch": unlock, "end": unlock["end"], "settled": after}
    acts.append(g2)
    save_overlay(env, out / "unlocked.png", "after south key door", map_id="room_72")
    print(
        f"unlock settled xy=({after['link_x']},{after['link_y']}) "
        f"keys={after['keys']} sub={after['submodule']}"
    )
    hit = pit_or_dead(after)
    if hit:
        return {"halt": hit, "start": start, "acts": acts, "end": after}

    through = watch(env, ("DOWN",), max_frames=360, stop_y=LOWER_Y)
    settle_control(env, max_frames=400)
    drop = glance(env)
    g3 = {"act": "through", "watch": through, "end": through["end"], "settled": drop}
    acts.append(g3)
    save_overlay(env, out / "drop.png", "key-door drop leftover", map_id="room_72")
    print(
        f"drop xy=({drop['link_x']},{drop['link_y']}) keys={drop['keys']} "
        f"hp={drop['leftover']['F36D_hp']} act={drop['link_action']} "
        f"room={drop['room_hex']} F3CC={drop['follower']}"
    )
    hit = pit_or_dead(drop)
    if hit:
        return {"halt": hit, "start": start, "acts": acts, "end": drop}
    on_floor = (
        drop["room"] == ROOM_72
        and drop["link_y"] >= LOWER_Y
        and drop["keys"] == 0
        and drop["module"] == 0x07
    )
    if not on_floor:
        return {"halt": "drop_miss", "start": start, "acts": acts, "end": drop}
    return {"halt": None, "start": start, "acts": acts, "drop": drop, "end": drop}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="0x72 key-door drop leftover → south 0x82")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--no-fight", action="store_true")
    args = parser.parse_args(argv)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    env = build_boot_env("CastleB1Key", render_mode="rgb_array")
    payload: dict[str, Any] = {
        "schema": "alttp_drop_south",
        "schemaVersion": 1,
        "measured": datetime.now(timezone.utc).isoformat(),
        "source": "state_load_dev",
        "pin": "CastleB1Key",
        "poked": False,
        "claims": [
            ["key_ledge_drop", *DROP_XY],
            ["pit_guard_cleared", *PIT_GUARD],
            ["zelda_pit", *ZELDA_PIT],
            ["south_door_approach", *SOUTH_APPROACH],
        ],
    }
    try:
        env.reset()
        assert resync_custom_state(env, GAME_DIR, INTEGRATION, "CastleB1Key")
        settle_control(env)
        start = glance(env)
        payload["start"] = start
        print(
            f"PIN CastleB1Key room={start['room_hex']} xy=({start['link_x']},{start['link_y']}) "
            f"keys={start['keys']} hp={start['leftover']['F36D_hp']} "
            f"mod={start['module_hex']}/{start['submodule']} F3CC={start['follower']}"
        )
        save_overlay(env, out / "pin.png", "CastleB1Key", map_id="room_72")
        pin_ok = (
            start["room"] == ROOM_72
            and start["keys"] >= 1
            and start["module"] == 0x07
            and start["submodule"] == 0
            and abs(start["link_x"] - 1320) <= 8
            and abs(start["link_y"] - 3656) <= 8
        )
        if not pin_ok:
            payload["halt"] = "pin"
            payload["end"] = start
            _write(out, payload)
            return 1

        drop_rec = replay_drop(env, out, start)
        payload["acts"] = drop_rec["acts"]
        if drop_rec["halt"]:
            payload["halt"] = drop_rec["halt"]
            payload["end"] = drop_rec["end"]
            _write(out, payload)
            return 0
        payload["drop"] = drop_rec["drop"]

        # Nudge south off the pit rim before fighting (pit is north of the drop).
        claims: list[dict[str, Any]] = []
        payload["walk"] = claims
        rim = axis_claim(env, "off_rim", DROP_XY[0], PIT_GUARD[1], south_first=True)
        claims.append(rim)
        print(
            f"off_rim ok={rim['ok']} miss={rim.get('miss')} "
            f"now=({rim['end']['link_x']},{rim['end']['link_y']}) room={rim['end']['room_hex']}"
        )
        if rim.get("crossed_to_0x82"):
            payload["halt"] = "ok"
            payload["end"] = rim["end"]
            save_overlay(env, out / "land.png", "0x82 from drop leftover", map_id="room_82")
            _write(out, payload)
            return 0
        if not rim["ok"]:
            payload["halt"] = rim.get("miss") or "stuck"
            payload["end"] = rim["end"]
            save_overlay(env, out / "stuck.png", "stuck after drop", map_id="room_72")
            _write(out, payload)
            return 0

        if not args.no_fight:
            fight = fight_nearby(env, room=ROOM_72, max_distance=120, attack_distance=48, max_cycles=120)
            settle_control(env, max_frames=80)
            fight_end = glance(env)
            rec = {
                "label": "fight",
                "ok": bool(fight.ok) and fight_end["room"] == ROOM_72 and pit_or_dead(fight_end) is None,
                "reason": fight.reason,
                "frames": fight.frames,
                "end": fight_end,
            }
            claims.append(rec)
            print(
                f"fight ok={rec['ok']} reason={fight.reason} "
                f"xy=({fight_end['link_x']},{fight_end['link_y']}) "
                f"hp={fight_end['leftover']['F36D_hp']} room={fight_end['room_hex']}"
            )
            hit = pit_or_dead(fight_end)
            if hit or fight_end["room"] != ROOM_72:
                payload["halt"] = hit or f"left_room_{fight_end['room_hex']}"
                payload["end"] = fight_end
                save_overlay(env, out / "stuck.png", "after fight", map_id="room_72")
                _write(out, payload)
                return 0

        # West-wall door. South-first at pit_guard x≈1224 hits the south
        # railing (stuck 1224,4032). Isolated hop diagonals SW; we go west
        # onto zelda_pit then south into the door.
        for label, xy, south_first in (
            ("zelda_pit", ZELDA_PIT, False),
            ("south_door_approach", SOUTH_APPROACH, True),
        ):
            rec = axis_claim(env, label, xy[0], xy[1], south_first=south_first)
            claims.append(rec)
            print(
                f"{label} ok={rec['ok']} miss={rec.get('miss')} "
                f"now=({rec['end']['link_x']},{rec['end']['link_y']}) room={rec['end']['room_hex']}"
            )
            if rec.get("crossed_to_0x82"):
                settle_control(env, max_frames=180)
                end = glance(env)
                payload["halt"] = "ok"
                payload["end"] = end
                save_overlay(env, out / "land.png", "0x82 from drop leftover", map_id="room_82")
                _write(out, payload)
                return 0
            if not rec["ok"]:
                payload["halt"] = rec.get("miss") or "stuck"
                payload["end"] = rec["end"]
                save_overlay(env, out / "stuck.png", f"stuck at {label}", map_id="room_72")
                _write(out, payload)
                return 0

        save_overlay(env, out / "approach.png", "south_door_approach", map_id="room_72")
        push = watch(env, ("DOWN",), max_frames=240, stop_room=ROOM_82)
        settle_control(env, max_frames=180)
        end = glance(env)
        payload["push"] = {"watch": push, "end": end}
        print(
            f"push room={end['room_hex']} xy=({end['link_x']},{end['link_y']}) "
            f"mod={end['module_hex']}/{end['submodule']} keys={end['keys']} "
            f"F3CC={end['follower']} hp={end['leftover']['F36D_hp']}"
        )
        hit = pit_or_dead(end)
        if hit:
            payload["halt"] = hit
            payload["end"] = end
            save_overlay(env, out / "stuck.png", "after DOWN push", map_id="room_72")
            _write(out, payload)
            return 0
        if end["room"] == ROOM_82 and end["module"] == 0x07:
            payload["halt"] = "ok"
            payload["end"] = end
            save_overlay(env, out / "land.png", "0x82 from drop leftover", map_id="room_82")
            _write(out, payload)
            return 0
        payload["halt"] = "stuck"
        payload["end"] = end
        save_overlay(env, out / "stuck.png", "still 0x72 after DOWN", map_id="room_72")
        _write(out, payload)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())

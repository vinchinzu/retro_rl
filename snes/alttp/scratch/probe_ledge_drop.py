"""Probe a drop from 0x72 north ledge (CastleB1Key) to the lower floor.

Claims, then one act. Default: grab / star / diagonal (no drop). --key-door:
south key door then DOWN to the lower floor (spends the key). Does not poke keys.

Usage:
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_ledge_drop.py
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_ledge_drop.py --key-door
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
from alttp.startup import action_for, build_boot_env, no_action, snapshot_env, step_frames
from retro_harness.env import resync_custom_state

OUT_DIR = RECORDINGS_DIR / "probe_ledge_drop"
LOWER_Y = 3944  # maps/room_72.json north_extent; south of the pit
LIP_Y = 3776
PIT_SUB = 20
DEATH_MODULE = 0x12


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
        "0402": int(ram[0x0402]),
        "0403_roomflags": int(ram[0x0403]),
        "00AE_tag1": int(ram[0x00AE]),
        "00EE_layer": int(ram[0x00EE]),
    }


def glance(env: object) -> dict[str, Any]:
    snap = snapshot_env(env)
    sprites = [
        {"type": f"0x{s.sprite_type:02X}", "xy": [s.x, s.y], "hp": s.hp}
        for s in active_sprites(env)
    ]
    return {
        "module": int(snap.game_mode),
        "module_hex": f"0x{int(snap.game_mode):02X}",
        "submodule": int(snap.submodule),
        "indoors": bool(snap.indoors),
        "room_base_id": int(snap.room_base_id),
        "room_hex": f"0x{int(snap.room_base_id):02X}",
        "room_label": room_label(snap.room_base_id),
        "link_x": int(snap.link_x),
        "link_y": int(snap.link_y),
        "link_action": int(snap.link_action),
        "sword": int(snap.sword_level),
        "lamp": int(snap.lamp_level),
        "keys": int(snap.num_keys),
        "follower": int(snap.follower),
        "has_control": bool(snap.has_control),
        "leftover": leftover_bytes(env),
        "sprites": sprites[:12],
    }


def save_overlay(env: object, path: Path, title: str) -> None:
    m = load_room_map("room_72")
    img = overlay_from_env(env, include_all_sprites=True, points=m.points, title=title)
    Image.fromarray(img).save(path)


def on_lower_floor(g: dict[str, Any]) -> bool:
    return (
        g["room_base_id"] == 0x72
        and g["module"] != DEATH_MODULE
        and g["leftover"]["F36D_hp"] > 0
        and g["link_y"] >= LOWER_Y
    )


def in_pit_anim(g: dict[str, Any]) -> bool:
    return g["submodule"] == PIT_SUB or (
        g["room_base_id"] == 0x72 and LIP_Y < g["link_y"] < LOWER_Y
    )


def dead(g: dict[str, Any]) -> bool:
    return g["module"] == DEATH_MODULE or g["leftover"]["F36D_hp"] <= 0


def still_on_ledge(g: dict[str, Any]) -> bool:
    return (
        g["room_base_id"] == 0x72
        and g["module"] != DEATH_MODULE
        and g["link_y"] <= LIP_Y + 4
        and g["link_x"] >= 1184
    )


def watch(
    env: object,
    buttons: tuple[str, ...],
    *,
    max_frames: int,
    step: int = 4,
    stop_on_key_spend: bool = False,
) -> dict[str, Any]:
    start = snapshot_env(env)
    start_keys = int(start.num_keys)
    frames = 0
    idle = 0
    prev = (start.link_x, start.link_y, start.room_base_id, start.submodule, start.game_mode)
    events: list[dict[str, Any]] = []
    action = no_action() if not buttons or buttons == ("NONE",) else action_for(*buttons)
    while frames < max_frames:
        step_frames(env, action, step)
        frames += step
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
            if snap.link_y >= LOWER_Y and snap.room_base_id == 0x72:
                break
            if stop_on_key_spend and snap.num_keys < start_keys:
                break
            if stop_on_key_spend and snap.submodule == 4:
                break
        elif snap.has_control:
            idle += 1
            if idle >= 12:
                break
        prev = cur
    if snapshot_env(env).submodule == PIT_SUB:
        settle_control(env, max_frames=400)
    return {"buttons": list(buttons), "frames": frames, "events": events[-16:], "end": glance(env)}


def grade_drop(name: str, claim: str, end: dict[str, Any]) -> dict[str, Any]:
    if dead(end):
        status = "death"
    elif on_lower_floor(end):
        status = "drop"
    elif in_pit_anim(end) and end["module"] != DEATH_MODULE:
        status = "pit"
    elif still_on_ledge(end):
        status = "miss"
    else:
        status = "other"
    rec = {
        "act": name,
        "claim": claim,
        "status": status,
        "end": end,
    }
    print(
        f"{name} {status} room={end['room_hex']} xy=({end['link_x']},{end['link_y']}) "
        f"mod={end['module_hex']}/{end['submodule']} act={end['link_action']} "
        f"hp={end['leftover']['F36D_hp']} keys={end['keys']} F3CC={end['follower']}"
    )
    return rec


def _key_door(env: object, out: Path, start: dict[str, Any]) -> int:
    """Claim: south key door at the lip opens (sub=4) then DOWN reaches lower floor."""
    acts: list[dict[str, Any]] = []
    claim_lip = "DOWN from CastleB1Key reaches east lip y≈3776 keys>=1"
    down = watch(env, ("DOWN",), max_frames=240)
    g1 = grade_drop("down_lip", claim_lip, down["end"])
    g1["watch"] = down
    acts.append(g1)
    save_overlay(env, out / "keydoor_lip.png", "key-door lip")
    if not still_on_ledge(down["end"]) or down["end"]["keys"] < 1:
        _write(out, start, acts, "keydoor_lip_miss")
        return 0

    claim_unlock = "DOWN+LEFT from east lip hits south key door: sub=4 and keys 1→0"
    unlock = watch(env, ("DOWN", "LEFT"), max_frames=200, stop_on_key_spend=True)
    g2 = grade_drop("unlock", claim_unlock, unlock["end"])
    g2["watch"] = unlock
    g2["unlocked"] = unlock["end"]["keys"] == 0 and any(
        ev.get("sub") == 4 for ev in unlock.get("events", [])
    )
    acts.append(g2)
    settle_control(env, max_frames=200)
    after_unlock = glance(env)
    g2["settled"] = after_unlock
    save_overlay(env, out / "keydoor_unlocked.png", "after south key door")
    print(
        f"unlock settled xy=({after_unlock['link_x']},{after_unlock['link_y']}) "
        f"keys={after_unlock['keys']} sub={after_unlock['submodule']} "
        f"doors={after_unlock['leftover'].get('0400_doors')} "
        f"flags={after_unlock['leftover'].get('0403_roomflags')}"
    )
    if dead(after_unlock):
        _write(out, start, acts, "death")
        return 0

    claim_through = "DOWN through opened south key door reaches lower 0x72 (y>=3944)"
    through = watch(env, ("DOWN",), max_frames=360)
    g3 = grade_drop("through", claim_through, through["end"])
    g3["watch"] = through
    acts.append(g3)
    settle_control(env, max_frames=400)
    end = glance(env)
    g3["settled"] = end
    save_overlay(env, out / "keydoor_through.png", "after DOWN through key door")
    if on_lower_floor(end):
        halt = "drop" if end["keys"] >= 1 else "drop_spent_key"
    elif dead(end):
        halt = "death"
    elif in_pit_anim(end):
        halt = "pit"
    else:
        halt = "blocked"
    _write(out, start, acts, halt)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="0x72 north-ledge drop from CastleB1Key")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--key-door",
        action="store_true",
        help="LEFT into south key door (sub=4) then DOWN through it.",
    )
    args = parser.parse_args(argv)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    env = build_boot_env("CastleB1Key", render_mode="rgb_array")
    acts: list[dict[str, Any]] = []
    try:
        env.reset()
        assert resync_custom_state(env, GAME_DIR, INTEGRATION, "CastleB1Key")
        settle_control(env)
        start = glance(env)
        print(
            f"PIN CastleB1Key room={start['room_hex']} xy=({start['link_x']},{start['link_y']}) "
            f"keys={start['keys']} hp={start['leftover']['F36D_hp']} "
            f"mod={start['module_hex']}/{start['submodule']} F3CC={start['follower']}"
        )
        save_overlay(env, out / "pin.png", "CastleB1Key")
        pin_ok = (
            start["room_base_id"] == 0x72
            and start["keys"] >= 1
            and start["module"] == 0x07
            and start["submodule"] == 0
            and abs(start["link_x"] - 1320) <= 8
            and abs(start["link_y"] - 3656) <= 8
        )
        if not pin_ok:
            print("PIN miss — halt")
            (out / "leftover.json").write_text(
                json.dumps({"start": start, "halt": "pin", "acts": []}, indent=2) + "\n"
            )
            return 1

        origin = env.em.get_state()  # type: ignore[attr-defined]
        if args.key_door:
            return _key_door(env, out, start)

        # Act 1: DOWN from the key pin to the east lip (known Guard geometry).
        claim1 = "DOWN from CastleB1Key reaches east lip y≈3776, still 0x72 keys>=1, not dead"
        down = watch(env, ("DOWN",), max_frames=240)
        g1 = grade_drop("down_lip", claim1, down["end"])
        g1["watch"] = down
        acts.append(g1)
        save_overlay(env, out / "after_down_lip.png", "after DOWN to lip")
        if g1["status"] == "drop":
            _write(out, start, acts, "drop")
            return 0
        if g1["status"] in {"death", "pit"}:
            _write(out, start, acts, g1["status"])
            return 0
        if g1["status"] != "miss" and not still_on_ledge(down["end"]):
            _write(out, start, acts, g1["status"])
            return 0
        if down["end"]["link_y"] < LIP_Y - 8:
            print("DOWN did not reach lip — halt")
            _write(out, start, acts, "lip_unreached")
            return 0

        lip = env.em.get_state()  # type: ignore[attr-defined]

        # Act 2: grab-drop at the east lip.
        claim2 = "A+DOWN at east lip drops to lower 0x72 floor (y>=3944) keys>=1, or pit sub=20"
        grab = watch(env, ("A", "DOWN"), max_frames=180)
        g2 = grade_drop("grab_lip", claim2, grab["end"])
        g2["watch"] = grab
        acts.append(g2)
        save_overlay(env, out / "after_grab_lip.png", "after A+DOWN at lip")
        if g2["status"] == "drop":
            _write(out, start, acts, "drop")
            return 0
        if g2["status"] in {"death", "pit"}:
            _write(out, start, acts, g2["status"])
            return 0

        # Grab missed (still on ledge). Restore lip; walk onto the star tile
        # south of the north block (~1272, 3712) — not reached from the north.
        env.em.set_state(lip)  # type: ignore[attr-defined]
        settle_control(env)
        claim3 = "from east lip, walk onto star ~ (1272,3712); star drops/warps to lower floor keys>=1"
        move_to(env, Waypoint(1312, 3744, tolerance=8, room=0x72), max_frames=240)
        star_walk = move_to(env, Waypoint(1272, 3712, tolerance=8, room=0x72), max_frames=320)
        step_frames(env, no_action(), 24)
        g3 = grade_drop("star", claim3, glance(env))
        g3["walk_ok"] = bool(star_walk.ok)
        g3["walk_reason"] = star_walk.reason
        acts.append(g3)
        save_overlay(env, out / "after_star.png", "after star walk")
        if g3["status"] == "drop":
            _write(out, start, acts, "drop")
            return 0
        if g3["status"] in {"death", "pit"}:
            _write(out, start, acts, g3["status"])
            return 0

        # Star miss. Restore lip; one diagonal into the pit corner.
        env.em.set_state(lip)  # type: ignore[attr-defined]
        settle_control(env)
        claim4 = "DOWN+LEFT from east lip enters pit (sub=20) or lands y>=3944 keys>=1 without dying"
        diag = watch(env, ("DOWN", "LEFT"), max_frames=200)
        g4 = grade_drop("diagonal", claim4, diag["end"])
        g4["watch"] = diag
        acts.append(g4)
        save_overlay(env, out / "after_diagonal.png", "after DOWN+LEFT")
        if g4["status"] == "drop":
            _write(out, start, acts, "drop")
            return 0
        _write(out, start, acts, g4["status"] if g4["status"] != "miss" else "blocked")
        return 0
    finally:
        env.close()


def _write(out: Path, start: dict[str, Any], acts: list[dict[str, Any]], halt: str) -> None:
    end = acts[-1]["end"] if acts else start
    payload = {
        "schema": "alttp_ledge_drop",
        "schemaVersion": 1,
        "measured": datetime.now(timezone.utc).isoformat(),
        "source": "state_load_dev",
        "pin": "CastleB1Key",
        "poked": False,
        "halt": halt,
        "start": start,
        "end": end,
        "acts": acts,
        "drop": halt == "drop",
    }
    (out / "leftover.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"HALT {halt} leftover room={end['room_hex']} xy=({end['link_x']},{end['link_y']}) "
        f"mod={end['module_hex']}/{end['submodule']} hp={end['leftover']['F36D_hp']} "
        f"keys={end['keys']} F3CC={end['follower']} wrote {out / 'leftover.json'}"
    )


if __name__ == "__main__":
    raise SystemExit(main())

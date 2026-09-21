"""Palace of Darkness entry pin: poke kit, real room-load, map lobby.

Dev pin from HyruleCastleGrounds. Isolated / state-load only.
Does not poke $F3CC. Does not grant hammer.

Usage:
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_pod_entry.py --census
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_pod_entry.py --pin
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_pod_entry.py --load-entrance
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_pod_entry.py --rays
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_pod_entry.py --map-lobby
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

from alttp.paths import GAME_DIR, INTEGRATION, INTEGRATION_DIR, RECORDINGS_DIR
from alttp.primitives import Waypoint, active_sprites, fight_nearby, move_to, settle_control
from alttp.ram import (
    DARK_WORLD_FLAG,
    FOLLOWER,
    LINK_HP,
    LINK_MAX_HP,
    MODULE,
    NUM_KEYS,
    SUBMODULE,
    room_label,
    wram_index,
)
from alttp.screen_glance import leftover_from_snapshot
from alttp.startup import action_for, build_boot_env, no_action, snapshot_env, step_frames
from retro_harness.env import resync_custom_state, write_state_bytes

OUT_DIR = RECORDINGS_DIR / "probe_pod_entry"
SAVE_NAME = "PalaceOfDarknessEntry"
SOURCE_STATE = "HyruleCastleGrounds"
POD_ROOM_HYPOTHESIS = 0x4A
POD_ENTRANCE_ID = 0x26
DEATH_MODULE = 0x12

# WRAM inventory (handoff table). Do not import extra offsets from ram.py.
F340_BOW = 0xF340
F343_BOMBS = 0xF343
F34A_LAMP = 0xF34A
F34B_HAMMER = 0xF34B
F353_MIRROR = 0xF353
F354_GLOVES = 0xF354
F355_BOOTS = 0xF355
F357_PEARL = 0xF357
F359_SWORD = 0xF359
F35A_SHIELD = 0xF35A
F377_ARROWS = 0xF377
F379_ABILITY = 0xF379
F3C5_PROGRESS = 0xF3C5
F3C7_MAPPROG = 0xF3C7
F3C8_SPAWN = 0xF3C8
F3CA_WORLD = 0xF3CA
F3CC_FOLLOWER = FOLLOWER
ENTRANCE_ID = 0x010E
DUNGEON_ID = 0x040C

SWORD_MASTER = 2
SHIELD_FIGHTER = 1
BOW_ARROWS = 2
GLOVES_POWER = 1
BOOTS_ON = 1
PEARL_ON = 1
LAMP_ON = 1
HAMMER_NONE = 0
PROGRESS_AGA1 = 3
WORLD_DW = 0x40
ABILITY_DASH = 0x04
HP_SEVEN = 0x38
ARROWS_COUNT = 30

LOADOUT_OFFSETS = (
    F340_BOW,
    F34A_LAMP,
    F34B_HAMMER,
    F354_GLOVES,
    F355_BOOTS,
    F357_PEARL,
    F359_SWORD,
    F35A_SHIELD,
    F377_ARROWS,
    F379_ABILITY,
    F3C5_PROGRESS,
    F3C7_MAPPROG,
    F3C8_SPAWN,
    F3CA_WORLD,
    F3CC_FOLLOWER,
    NUM_KEYS,
    LINK_HP,
    LINK_MAX_HP,
    DARK_WORLD_FLAG,
    DUNGEON_ID,
    ENTRANCE_ID,
)


def leftover_bytes(env: object) -> dict[str, int]:
    ram = env.get_ram()  # type: ignore[attr-defined]
    rec: dict[str, int] = {}
    for off in LOADOUT_OFFSETS:
        rec[f"{off:04X}"] = int(ram[wram_index(off)])
    rec["00EE_layer"] = int(ram[0x00EE])
    rec["0403_roomflags"] = int(ram[0x0403])
    rec["0FFF_dw"] = int(ram[DARK_WORLD_FLAG])
    return rec


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
            "screen_hex": f"0x{int(snap.screen_id):02X}",
            "link_x": int(snap.link_x),
            "link_y": int(snap.link_y),
            "link_action": int(snap.link_action),
            "has_control": bool(snap.has_control),
            "dark_world": bool(snap.dark_world),
            "lamp": int(snap.lamp_level),
            "leftover": leftover_bytes(env),
            "sprites": sprites[:16],
        }
    )
    return rec


def save_png(env: object, path: Path, title: str = "") -> None:
    from alttp.room_sense import overlay_from_env

    img = overlay_from_env(env, include_all_sprites=True, title=title)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


def ram_u8(env: object, offset: int) -> int:
    ram = env.get_ram()  # type: ignore[attr-defined]
    return int(ram[wram_index(offset)])


def poke_u8(env: object, offset: int, value: int) -> dict[str, Any]:
    """Write WRAM via memory.assign; confirm with get_ram() index.

    Handoff offset is ``0xF3xx`` (not the get_ram index). High WRAM on this
    stable-retro mapping is bank ``$7E``, so ``0xF359`` may raise
    ``IndexError: No known mapping`` and we retry ``0x7EF359``.
    """
    before = ram_u8(env, offset)
    mem = env.unwrapped.data.memory  # type: ignore[attr-defined]
    want = int(value) & 0xFF
    rec: dict[str, Any] = {
        "offset": f"0x{offset:04X}",
        "want": want,
        "before": before,
        "after": before,
        "ok": False,
        "via": "memory.assign",
    }
    candidates = [int(offset)]
    if offset >= 0x2000:
        candidates.append(0x7E0000 + int(offset))
    last_err = None
    for addr in candidates:
        try:
            mem.assign(addr, "|u1", want)
        except (IndexError, KeyError, ValueError) as exc:
            last_err = repr(exc)
            rec["via"] = f"memory.assign fail {addr:#x}: {last_err}"
            continue
        after = ram_u8(env, offset)
        rec["after"] = after
        rec["ok"] = after == want
        rec["via"] = f"memory.assign {addr:#x}"
        if rec["ok"]:
            return rec
    rec["error"] = last_err
    return rec


def ensure_pod_kit(env: object) -> list[dict[str, Any]]:
    """Poke PoD entrance kit. Never grant hammer. Never write $F3CC."""
    hammer = ram_u8(env, F34B_HAMMER)
    if hammer != HAMMER_NONE:
        rec = poke_u8(env, F34B_HAMMER, HAMMER_NONE)
        rec["item"] = "hammer_clear"
        if ram_u8(env, F34B_HAMMER) != HAMMER_NONE:
            raise RuntimeError(f"hammer already granted $F34B={hammer}; refuse to pin")
    wanted = (
        (F359_SWORD, SWORD_MASTER, "sword"),
        (F35A_SHIELD, SHIELD_FIGHTER, "shield"),
        (F34A_LAMP, LAMP_ON, "lamp"),
        (F340_BOW, BOW_ARROWS, "bow"),
        (F354_GLOVES, GLOVES_POWER, "gloves"),
        (F355_BOOTS, BOOTS_ON, "boots"),
        (F357_PEARL, PEARL_ON, "moon_pearl"),
        (F3C5_PROGRESS, PROGRESS_AGA1, "progress"),
        (F3CA_WORLD, WORLD_DW, "world_dw"),
        (F377_ARROWS, ARROWS_COUNT, "arrows"),
        (LINK_MAX_HP, HP_SEVEN, "max_hp"),
        (LINK_HP, HP_SEVEN, "hp"),
    )
    pokes: list[dict[str, Any]] = []
    for offset, value, name in wanted:
        current = ram_u8(env, offset)
        if current == value:
            pokes.append(
                {
                    "item": name,
                    "offset": f"0x{offset:04X}",
                    "want": value,
                    "before": current,
                    "after": current,
                    "ok": True,
                    "via": "already",
                }
            )
            continue
        rec = poke_u8(env, offset, value)
        rec["item"] = name
        if not rec["ok"]:
            raise RuntimeError(f"poke {name} ${offset:04X} failed: {rec}")
        pokes.append(rec)
    ability = ram_u8(env, F379_ABILITY)
    if ability & ABILITY_DASH == 0:
        rec = poke_u8(env, F379_ABILITY, ability | ABILITY_DASH)
        rec["item"] = "ability_dash"
        pokes.append(rec)
        if not rec["ok"]:
            raise RuntimeError(f"poke ability $F379 failed: {rec}")
    else:
        pokes.append(
            {
                "item": "ability_dash",
                "offset": "0xF379",
                "want": ability,
                "before": ability,
                "after": ability,
                "ok": True,
                "via": "already",
            }
        )
    if ram_u8(env, F34B_HAMMER) != HAMMER_NONE:
        raise RuntimeError("hammer granted during kit poke")
    return pokes


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_source(env: object | None = None) -> object:
    if env is None:
        env = build_boot_env(SOURCE_STATE, render_mode="rgb_array")
        env.reset()  # type: ignore[attr-defined]
    else:
        if not resync_custom_state(env, GAME_DIR, INTEGRATION, SOURCE_STATE):
            raise RuntimeError(f"failed to load {SOURCE_STATE}")
    settle_control(env)
    return env


def wait_control(env: object, *, max_frames: int = 900) -> dict[str, Any]:
    frames = 0
    events: list[dict[str, Any]] = []
    prev = None
    while frames < max_frames:
        step_frames(env, no_action(), 4)
        frames += 4
        snap = snapshot_env(env)
        cur = (
            snap.game_mode,
            snap.submodule,
            snap.room_base_id,
            snap.indoors,
            snap.has_control,
            snap.link_x,
            snap.link_y,
        )
        if cur != prev:
            events.append(
                {
                    "f": frames,
                    "mod": int(snap.game_mode),
                    "sub": int(snap.submodule),
                    "room": f"0x{snap.room_base_id:02X}",
                    "in": bool(snap.indoors),
                    "ctrl": bool(snap.has_control),
                    "dw": bool(snap.dark_world),
                    "xy": [snap.link_x, snap.link_y],
                }
            )
            prev = cur
        if snap.game_mode == DEATH_MODULE:
            break
        if snap.has_control and snap.indoors:
            break
        if len(events) > 120:
            break
    return {"frames": frames, "events": events, "end": glance(env)}


def hold(
    env: object,
    buttons: tuple[str, ...],
    *,
    max_frames: int,
    stop_room: int | None = None,
) -> dict[str, Any]:
    start = snapshot_env(env)
    frames = 0
    events: list[dict[str, Any]] = []
    action = no_action() if not buttons or buttons == ("NONE",) else action_for(*buttons)
    prev = (
        start.link_x,
        start.link_y,
        start.screen_id,
        start.room_base_id,
        start.indoors,
        start.submodule,
        start.game_mode,
    )
    start_room = start.room_base_id
    while frames < max_frames:
        step_frames(env, action, 4)
        frames += 4
        snap = snapshot_env(env)
        cur = (
            snap.link_x,
            snap.link_y,
            snap.screen_id,
            snap.room_base_id,
            snap.indoors,
            snap.submodule,
            snap.game_mode,
        )
        if cur != prev:
            events.append(
                {
                    "f": frames,
                    "screen": f"0x{snap.screen_id:02X}",
                    "room": f"0x{snap.room_base_id:02X}",
                    "in": bool(snap.indoors),
                    "mod": int(snap.game_mode),
                    "sub": int(snap.submodule),
                    "xy": [snap.link_x, snap.link_y],
                    "ctrl": bool(snap.has_control),
                    "dw": bool(snap.dark_world),
                }
            )
            prev = cur
        if snap.game_mode == DEATH_MODULE:
            break
        if stop_room is not None and snap.indoors and snap.room_base_id != start_room:
            if snap.has_control or snap.submodule != 0:
                if snap.room_base_id == stop_room or True:
                    settle_control(env, max_frames=480)
                    events.append({"f": frames, "settled": glance(env)})
                    break
        if len(events) > 80:
            break
    return {"frames": frames, "events": events, "end": glance(env)}


def cmd_census() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = [SOURCE_STATE, "FighterSword", "FighterSwordLamp"]
    env = build_boot_env(names[0], render_mode="rgb_array")
    records: list[dict[str, Any]] = []
    try:
        env.reset()
        for name in names:
            path = INTEGRATION_DIR / f"{name}.state"
            if not path.is_file():
                records.append({"state": name, "missing": True})
                continue
            if not resync_custom_state(env, GAME_DIR, INTEGRATION, name):
                records.append({"state": name, "load_failed": True})
                continue
            settle_control(env)
            rec = glance(env)
            rec["state"] = name
            save_png(env, OUT_DIR / f"census_{name}.png", title=f"{name} census")
            records.append(rec)
    finally:
        env.close()
    write_json(OUT_DIR / "census.json", {"when": utc_now(), "records": records})
    print(json.dumps(records, indent=2))
    return 0


def cmd_pin() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = load_source()
    try:
        before = glance(env)
        save_png(env, OUT_DIR / "grounds_before_poke.png", title="grounds before poke")
        pokes = ensure_pod_kit(env)
        step_frames(env, no_action(), 2)
        after = glance(env)
        save_png(env, OUT_DIR / "grounds_after_poke.png", title="grounds after poke")
        write_json(
            OUT_DIR / "pokes.json",
            {
                "when": utc_now(),
                "source": SOURCE_STATE,
                "pokes": pokes,
                "before": before,
                "after": after,
            },
        )
        print(json.dumps({"pokes": pokes, "after": after}, indent=2))
    finally:
        env.close()
    return 0


def trigger_entrance(env: object, entrance: int) -> list[dict[str, Any]]:
    """Poke entrance id + dungeon-load module. Real room-load, not a freeze poke of $A0."""
    recs: list[dict[str, Any]] = []
    dw = poke_u8(env, DARK_WORLD_FLAG, 1)
    dw["item"] = "dark_world_flag"
    recs.append(dw)
    ent = poke_u8(env, ENTRANCE_ID, entrance & 0xFF)
    ent["item"] = "entrance_id"
    recs.append(ent)
    # High byte of $010E if the game stores a word.
    ent_hi = poke_u8(env, ENTRANCE_ID + 1, 0)
    ent_hi["item"] = "entrance_id_hi"
    recs.append(ent_hi)
    sub = poke_u8(env, SUBMODULE, 0)
    sub["item"] = "submodule"
    recs.append(sub)
    mod = poke_u8(env, MODULE, 0x06)
    mod["item"] = "module_underworld_load"
    recs.append(mod)
    return recs


def indoor_ok(g: dict[str, Any]) -> bool:
    leftover = g.get("leftover") or {}
    sub = g.get("submodule")
    return bool(
        g.get("indoors")
        and g.get("has_control")
        and int(g.get("module") or 0) == 0x07
        and int(sub) == 0
        and int(leftover.get("F34B", 1)) == 0
        and int(leftover.get("F359", 0)) == SWORD_MASTER
        and int(leftover.get("F357", 0)) == PEARL_ON
    )


def cmd_load_entrance(*, entrance: int, save: bool) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = load_source()
    try:
        pokes = ensure_pod_kit(env)
        step_frames(env, no_action(), 2)
        after_kit = glance(env)
        save_png(env, OUT_DIR / "kit_before_load.png", title="kit before entrance load")
        load_pokes = trigger_entrance(env, entrance)
        waited = wait_control(env, max_frames=1200)
        end = waited["end"]
        save_png(
            env,
            OUT_DIR / f"entrance_{entrance:02x}.png",
            title=f"entrance 0x{entrance:02X} {end.get('room_hex')}",
        )
        saved = None
        if save and indoor_ok(end):
            path = INTEGRATION_DIR / f"{SAVE_NAME}.state"
            write_state_bytes(path, env.em.get_state())  # type: ignore[attr-defined]
            saved = str(path)
        payload = {
            "when": utc_now(),
            "source": SOURCE_STATE,
            "entrance": f"0x{entrance:02X}",
            "pokes": pokes,
            "load_pokes": load_pokes,
            "after_kit": after_kit,
            "waited": waited,
            "end": end,
            "saved": saved,
            "indoor_ok": indoor_ok(end),
        }
        write_json(OUT_DIR / f"entrance_{entrance:02x}.json", payload)
        print(json.dumps(payload, indent=2))
        return 0 if indoor_ok(end) else 1
    finally:
        env.close()


def cmd_rays(*, frames: int) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pin = INTEGRATION_DIR / f"{SAVE_NAME}.state"
    name = SAVE_NAME if pin.is_file() else SOURCE_STATE
    env = build_boot_env(name, render_mode="rgb_array")
    try:
        env.reset()
        settle_control(env)
        start = glance(env)
        save_png(env, OUT_DIR / "rays_start.png", title=f"rays {start.get('room_hex')}")
        blob = env.em.get_state()  # type: ignore[attr-defined]
        rays: dict[str, Any] = {}
        for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
            env.em.set_state(blob)  # type: ignore[attr-defined]
            step_frames(env, no_action(), 1)
            rec = hold(env, (direction,), max_frames=frames)
            save_png(
                env,
                OUT_DIR / f"ray_{direction.lower()}.png",
                title=f"hold {direction}",
            )
            rays[direction] = rec
        payload = {"when": utc_now(), "state": name, "start": start, "rays": rays}
        write_json(OUT_DIR / "rays.json", payload)
        print(json.dumps({"state": name, "start": start, "summary": {
            d: {"frames": rays[d]["frames"], "end": {
                k: rays[d]["end"].get(k)
                for k in ("room_hex", "xy", "module_hex", "submodule", "indoors", "has_control")
            }}
            for d in rays
        }}, indent=2))
    finally:
        env.close()
    return 0


def cmd_goto(*, x: int, y: int, max_frames: int, push: str | None) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pin = INTEGRATION_DIR / f"{SAVE_NAME}.state"
    name = SAVE_NAME if pin.is_file() else SOURCE_STATE
    env = build_boot_env(name, render_mode="rgb_array")
    try:
        env.reset()
        settle_control(env)
        start = glance(env)
        room = int(start.get("room") or 0)
        move = move_to(
            env,
            Waypoint(x, y, tolerance=8, room=room, label="goto"),
            max_frames=max_frames,
        )
        settle_control(env)
        mid = glance(env)
        pushed = None
        if push:
            pushed = hold(env, (push.upper(),), max_frames=360, stop_room=-1)
            settle_control(env, max_frames=480)
        end = glance(env)
        save_png(env, OUT_DIR / "goto_end.png", title=f"goto ({x},{y})")
        payload = {
            "when": utc_now(),
            "state": name,
            "start": start,
            "move_ok": bool(move.ok),
            "move_reason": move.reason,
            "move_frames": int(move.frames),
            "mid": mid,
            "push": push,
            "pushed": pushed,
            "end": end,
        }
        write_json(OUT_DIR / "goto.json", payload)
        print(json.dumps(payload, indent=2))
    finally:
        env.close()
    return 0


def cmd_map_lobby() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pin = INTEGRATION_DIR / f"{SAVE_NAME}.state"
    name = SAVE_NAME if pin.is_file() else SOURCE_STATE
    env = build_boot_env(name, render_mode="rgb_array")
    try:
        env.reset()
        settle_control(env)
        g = glance(env)
        save_png(env, OUT_DIR / "lobby.png", title=f"lobby {g.get('room_hex')}")
        write_json(OUT_DIR / "lobby.json", {"when": utc_now(), "state": name, "glance": g})
        print(json.dumps({"state": name, "glance": g}, indent=2))
    finally:
        env.close()
    return 0


SWEEPS: tuple[tuple[str, int, int, str], ...] = (
    ("switch_a", 5312, 2504, "NONE"),
    ("switch_b", 5296, 2496, "NONE"),
    ("switch_c", 5328, 2512, "NONE"),
    ("stair_w1", 5184, 2180, "UP"),
    ("stair_w2", 5160, 2144, "UP"),
    ("stair_w3", 5216, 2112, "UP"),
    ("stair_w4", 5248, 2088, "UP"),
    ("stair_w5", 5160, 2080, "DOWN"),
    ("cage_n", 5184, 2064, "UP"),
    ("green_w", 5160, 2240, "LEFT"),
    ("east_open", 5504, 2424, "RIGHT"),
    ("east_n", 5504, 2360, "RIGHT"),
)


def cmd_sweep() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pin = INTEGRATION_DIR / f"{SAVE_NAME}.state"
    name = SAVE_NAME if pin.is_file() else SOURCE_STATE
    env = build_boot_env(name, render_mode="rgb_array")
    try:
        env.reset()
        settle_control(env)
        blob = env.em.get_state()  # type: ignore[attr-defined]
        start = glance(env)
        rows: list[dict[str, Any]] = []
        for label, x, y, direction in SWEEPS:
            env.em.set_state(blob)  # type: ignore[attr-defined]
            step_frames(env, no_action(), 1)
            room = int(start.get("room") or POD_ROOM_HYPOTHESIS)
            move = move_to(
                env,
                Waypoint(x, y, tolerance=6, room=room, label=label),
                max_frames=500,
            )
            if label.startswith("switch"):
                step_frames(env, action_for("A"), 8)
                step_frames(env, no_action(), 24)
            pushed = hold(
                env,
                () if direction == "NONE" else (direction,),
                max_frames=360,
                stop_room=-1,
            )
            settle_control(env, max_frames=480)
            end = glance(env)
            save_png(env, OUT_DIR / f"sweep_{label}.png", title=f"{label} {direction}")
            row = {
                "label": label,
                "target": [x, y],
                "dir": direction,
                "move_ok": bool(move.ok),
                "move_reason": move.reason,
                "move_xy": [move.snapshot.link_x, move.snapshot.link_y],
                "end_room": end.get("room_hex"),
                "end_xy": end.get("xy"),
                "end_mod": end.get("module_hex"),
                "end_sub": end.get("submodule"),
                "end_in": end.get("indoors"),
                "end_ctrl": end.get("has_control"),
                "left_room": int(end.get("room") or room) != room or not end.get("indoors"),
            }
            rows.append(row)
            print(json.dumps(row))
        payload = {"when": utc_now(), "state": name, "start": start, "sweeps": rows}
        write_json(OUT_DIR / "sweep.json", payload)
    finally:
        env.close()
    return 0


PATHS: tuple[tuple[str, tuple[tuple[int, int], ...], str], ...] = (
    ("e_cage_n", ((5504, 2424), (5504, 2300), (5512, 2192), (5512, 2104)), "UP"),
)


def cmd_paths() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pin = INTEGRATION_DIR / f"{SAVE_NAME}.state"
    name = SAVE_NAME if pin.is_file() else SOURCE_STATE
    env = build_boot_env(name, render_mode="rgb_array")
    try:
        env.reset()
        settle_control(env)
        blob = env.em.get_state()  # type: ignore[attr-defined]
        start = glance(env)
        room = int(start.get("room") or POD_ROOM_HYPOTHESIS)
        rows: list[dict[str, Any]] = []
        for label, pts, direction in PATHS:
            env.em.set_state(blob)  # type: ignore[attr-defined]
            step_frames(env, no_action(), 1)
            hops: list[dict[str, Any]] = []
            blocked = False
            for x, y in pts:
                move = move_to(
                    env,
                    Waypoint(x, y, tolerance=8, room=room, label=f"{label}_{x}_{y}"),
                    max_frames=600,
                )
                hops.append(
                    {
                        "xy": [x, y],
                        "ok": bool(move.ok),
                        "reason": move.reason,
                        "at": [move.snapshot.link_x, move.snapshot.link_y],
                        "room": f"0x{move.snapshot.room_base_id:02X}",
                    }
                )
                if move.snapshot.room_base_id != room or not move.snapshot.indoors:
                    blocked = True
                    break
                if not move.ok:
                    blocked = True
                    break
            if not blocked:
                hold(env, (direction,), max_frames=400, stop_room=-1)
                settle_control(env, max_frames=480)
            end = glance(env)
            save_png(env, OUT_DIR / f"path_{label}.png", title=f"{label} {direction}")
            row = {
                "label": label,
                "hops": hops,
                "end_room": end.get("room_hex"),
                "end_xy": end.get("xy"),
                "end_mod": end.get("module_hex"),
                "end_sub": end.get("submodule"),
                "end_in": end.get("indoors"),
                "end_ctrl": end.get("has_control"),
                "left_room": int(end.get("room") or room) != room or not end.get("indoors"),
            }
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("label", "end_room", "end_xy", "end_mod", "end_sub", "left_room", "hops")}))
        write_json(
            OUT_DIR / "paths.json",
            {"when": utc_now(), "state": name, "start": start, "paths": rows},
        )
    finally:
        env.close()
    return 0


ISOLATE_PATH: tuple[tuple[int, int], ...] = (
    (5504, 2424),
    (5504, 2300),
    (5512, 2192),
    (5512, 2104),
)
SPRITE_MINI_HELMASAUR = 0x13


def cmd_isolate() -> int:
    """Replay east-cage stairs 0x4A → 0x09 from the pin. Isolated only."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = build_boot_env(SAVE_NAME, render_mode="rgb_array")
    try:
        env.reset()
        settle_control(env)
        blob = env.em.get_state()  # type: ignore[attr-defined]
        env.em.set_state(blob)  # type: ignore[attr-defined]
        step_frames(env, no_action(), 1)
        start = glance(env)
        save_png(env, OUT_DIR / "isolate_start.png", title="isolate start")
        room = POD_ROOM_HYPOTHESIS
        hops: list[dict[str, Any]] = []
        for x, y in ISOLATE_PATH:
            move = move_to(
                env,
                Waypoint(x, y, tolerance=8, room=room, label=f"iso_{x}_{y}"),
                max_frames=600,
            )
            hops.append(
                {
                    "xy": [x, y],
                    "ok": bool(move.ok),
                    "reason": move.reason,
                    "at": [move.snapshot.link_x, move.snapshot.link_y],
                    "room": f"0x{move.snapshot.room_base_id:02X}",
                    "mod": int(move.snapshot.game_mode),
                }
            )
            if move.snapshot.game_mode == DEATH_MODULE:
                break
            if move.snapshot.room_base_id != room:
                break
            if not move.ok:
                break
        pushed = hold(env, ("UP",), max_frames=400, stop_room=-1)
        settle_control(env, max_frames=480)
        end = glance(env)
        save_png(
            env,
            OUT_DIR / "isolate_end.png",
            title=f"isolate {end.get('room_hex')} {end.get('xy')}",
        )
        ok = (
            int(end.get("room") or 0) == 0x09
            and bool(end.get("indoors"))
            and bool(end.get("has_control"))
            and int(end.get("module") or 0) == 0x07
            and int(end.get("submodule") if end.get("submodule") is not None else -1) == 0
            and int((end.get("leftover") or {}).get("F34B", 1)) == 0
            and int((end.get("leftover") or {}).get("F359", 0)) == SWORD_MASTER
        )
        payload = {
            "when": utc_now(),
            "state": SAVE_NAME,
            "door": "east_stairs_to_0x09",
            "verification": "isolated",
            "ok": ok,
            "start": start,
            "hops": hops,
            "push_end": pushed.get("end") if isinstance(pushed, dict) else None,
            "end": end,
        }
        write_json(OUT_DIR / "isolate_0x09.json", payload)
        print(json.dumps({"ok": ok, "end": {
            k: end.get(k)
            for k in (
                "room_hex", "xy", "module_hex", "submodule", "indoors",
                "has_control", "dark_world", "sword", "leftover",
            )
        }}, indent=2))
        return 0 if ok else 1
    finally:
        env.close()


def cmd_save() -> int:
    pin = INTEGRATION_DIR / f"{SAVE_NAME}.state"
    name = SAVE_NAME if pin.is_file() else SOURCE_STATE
    env = build_boot_env(name, render_mode="rgb_array")
    try:
        env.reset()
        settle_control(env)
        g = glance(env)
        path = INTEGRATION_DIR / f"{SAVE_NAME}.state"
        write_state_bytes(path, env.em.get_state())  # type: ignore[attr-defined]
        payload = {"when": utc_now(), "saved": str(path), "glance": g}
        write_json(OUT_DIR / "saved.json", payload)
        print(json.dumps(payload, indent=2))
    finally:
        env.close()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Palace of Darkness entry pin probe")
    p.add_argument("--census", action="store_true")
    p.add_argument("--pin", action="store_true", help="poke kit on HyruleCastleGrounds")
    p.add_argument("--load-entrance", action="store_true")
    p.add_argument("--entrance", type=lambda s: int(s, 0), default=POD_ENTRANCE_ID)
    p.add_argument("--save", action="store_true")
    p.add_argument("--rays", action="store_true")
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--goto", nargs=2, type=int, metavar=("X", "Y"))
    p.add_argument("--push", default=None, help="hold direction after --goto")
    p.add_argument("--map-lobby", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--paths", action="store_true")
    p.add_argument("--isolate", action="store_true")
    args = p.parse_args()
    if args.census:
        return cmd_census()
    if args.pin:
        return cmd_pin()
    if args.load_entrance:
        return cmd_load_entrance(entrance=int(args.entrance), save=bool(args.save))
    if args.rays:
        return cmd_rays(frames=args.frames)
    if args.goto is not None:
        return cmd_goto(
            x=args.goto[0],
            y=args.goto[1],
            max_frames=args.frames,
            push=args.push,
        )
    if args.map_lobby:
        return cmd_map_lobby()
    if args.sweep:
        return cmd_sweep()
    if args.paths:
        return cmd_paths()
    if args.isolate:
        return cmd_isolate()
    if args.save:
        return cmd_save()
    return cmd_pin()


if __name__ == "__main__":
    raise SystemExit(main())

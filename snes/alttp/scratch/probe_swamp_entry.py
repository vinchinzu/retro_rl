"""Swamp Palace entry pin (rr-q41r). Isolated / state-load only.

Load FighterSword, poke RTA-typical pre-hookshot kit + $F3C5=3, then trigger a
real dungeon load into the lobby. Confirm $A0 before naming the map.

Usage:
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_swamp_entry.py
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_swamp_entry.py --scan
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_swamp_entry.py --doors
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_swamp_entry.py --edge
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
from alttp.primitives import Waypoint, active_sprites, move_to, settle_control
from alttp.ram import (
    EQUIP_SWORD,
    FOLLOWER,
    LINK_HP,
    LINK_MAX_HP,
    NUM_KEYS,
    room_label,
    snapshot_to_diag,
    wram_index,
)
from alttp.room_sense import overlay_from_env
from alttp.screen_glance import leftover_from_snapshot
from alttp.startup import action_for, build_boot_env, no_action, snapshot_env, step_frames
from retro_harness.env import resync_custom_state, write_state_bytes

OUT_DIR = RECORDINGS_DIR / "probe_swamp_entry"
SAVE_NAME = "SwampPalaceEntry"
SOURCE_STATE = "FighterSword"
HYPOTHESIS_ROOM = 0x28
SNES_WRAM_BANK = 0x7E0000
ENTRANCE_ID = 0x010E
MODULE = 0x10
SUBMODULE = 0x11

# RTA-typical kit before Swamp prize (hookshot). Do not grant $F342.
LOADOUT: tuple[tuple[int, int, str], ...] = (
    (0xF340, 0, "bow none"),
    (0xF342, 0, "hookshot none (prize)"),
    (0xF343, 10, "bombs"),
    (0xF34B, 1, "hammer"),
    (0xF353, 1, "mirror"),
    (0xF354, 1, "power gloves"),
    (0xF355, 1, "boots"),
    (0xF357, 1, "moon pearl"),
    (0xF359, 2, "master sword"),
    (0xF35A, 2, "fire shield"),
    (0xF36C, 0x18, "max HP 3 hearts"),
    (0xF36D, 0x18, "current HP full"),
    (0xF36E, 0x80, "magic"),
    (0xF37B, 0, "magic consumption normal"),
    (0xF3C5, 3, "Agahnim 1 / DW open"),
    (0xF2BB, 0x20, "LW dam overlay bit5 (screen 0x3B)"),
    (0xF2FB, 0x20, "DW swamp overlay bit5 (screen 0x7B)"),
    (0xF051, 0x01, "room 0x28 save_dung_info bit8 (water)"),
    (0xF216, 0x80, "dam interior room 267 bit7 (switch pulled)"),
)

# Common dungeon entrance IDs; --scan tries 0x00-0x7F.
WARP_CANDIDATES: tuple[int, ...] = (
    0x25,
    0x28,
    0x49,
    0x4A,
    0x4B,
    0x5E,
    0x48,
    0x24,
    0x26,
    0x08,
)

HOLD_DIRS: tuple[str, ...] = ("UP", "DOWN", "LEFT", "RIGHT")


def _memory(env: object) -> Any:
    unwrapped = getattr(env, "unwrapped", env)
    return unwrapped.data.memory


def poke_u8(env: object, offset: int, value: int) -> dict[str, Any]:
    """Write one WRAM byte. High WRAM is bank $7E (assign 0x7EF3xx, not get_ram index)."""
    mem = _memory(env)
    off = int(offset)
    mapped = SNES_WRAM_BANK + off if off >= 0x2000 else off
    mem.assign(mapped, "|u1", int(value) & 0xFF)
    got = read_u8(env, off)
    return {
        "offset": f"0x{off:04X}",
        "assign": f"0x{mapped:06X}",
        "want": int(value) & 0xFF,
        "got": got,
        "ok": got == (int(value) & 0xFF),
    }


def poke_u16(env: object, offset: int, value: int) -> dict[str, Any]:
    lo = poke_u8(env, offset, int(value) & 0xFF)
    hi = poke_u8(env, offset + 1, (int(value) >> 8) & 0xFF)
    return {
        "offset": f"0x{int(offset):04X}",
        "want": int(value) & 0xFFFF,
        "got": read_u16(env, offset),
        "ok": lo["ok"] and hi["ok"],
        "lo": lo,
        "hi": hi,
    }


def read_u8(env: object, offset: int) -> int:
    ram = env.get_ram()  # type: ignore[attr-defined]
    return int(ram[wram_index(int(offset))])


def read_u16(env: object, offset: int) -> int:
    return read_u8(env, offset) | (read_u8(env, offset + 1) << 8)


def leftover_bytes(env: object) -> dict[str, int]:
    ram = env.get_ram()  # type: ignore[attr-defined]
    return {
        "F340_bow": int(ram[wram_index(0xF340)]),
        "F342_hookshot": int(ram[wram_index(0xF342)]),
        "F343_bombs": int(ram[wram_index(0xF343)]),
        "F34B_hammer": int(ram[wram_index(0xF34B)]),
        "F353_mirror": int(ram[wram_index(0xF353)]),
        "F354_gloves": int(ram[wram_index(0xF354)]),
        "F355_boots": int(ram[wram_index(0xF355)]),
        "F357_pearl": int(ram[wram_index(0xF357)]),
        "F359_sword": int(ram[wram_index(EQUIP_SWORD)]),
        "F35A_shield": int(ram[wram_index(0xF35A)]),
        "F36C_max_hp": int(ram[wram_index(LINK_MAX_HP)]),
        "F36D_hp": int(ram[wram_index(LINK_HP)]),
        "F36E_magic": int(ram[wram_index(0xF36E)]),
        "F36F_keys": int(ram[wram_index(NUM_KEYS)]),
        "F3C5_progress": int(ram[wram_index(0xF3C5)]),
        "F3CA_dw_save": int(ram[wram_index(0xF3CA)]),
        "F3CC_follower": int(ram[wram_index(FOLLOWER)]),
        "F2BB_ow_3b": int(ram[wram_index(0xF2BB)]),
        "F2FB_ow_7b": int(ram[wram_index(0xF2FB)]),
        "F051_room28": int(ram[wram_index(0xF051)]),
        "F216_dam": int(ram[wram_index(0xF216)]),
        "F356_flippers": int(ram[wram_index(0xF356)]),
        "010E_entrance": read_u16(env, ENTRANCE_ID),
        "00EE_layer": int(ram[0x00EE]),
        "0403_roomflags": int(ram[0x0403]),
        "0FFF_dw": int(ram[0x0FFF]),
    }


def glance(env: object) -> dict[str, Any]:
    snap = snapshot_env(env)
    rec = leftover_from_snapshot(snap)
    sprites = [
        {"type": f"0x{s.sprite_type:02X}", "xy": [s.x, s.y], "hp": s.hp}
        for s in active_sprites(env)
    ]
    rec.update(
        {
            "module_hex": f"0x{int(snap.game_mode):02X}",
            "room_hex": f"0x{int(snap.room_base_id):02X}",
            "room_label": room_label(snap.room_base_id),
            "has_control": bool(snap.has_control),
            "link_action": int(snap.link_action),
            "dark_world": bool(snap.dark_world),
            "bytes": leftover_bytes(env),
            "sprites": sprites[:16],
            "diag": snapshot_to_diag(snap),
        }
    )
    return rec


def apply_loadout(env: object) -> list[dict[str, Any]]:
    results = []
    for offset, value, note in LOADOUT:
        rec = poke_u8(env, offset, value)
        rec["note"] = note
        results.append(rec)
        print(
            f"  poke ${rec['offset']}={rec['want']:02X} got={rec['got']:02X} "
            f"ok={int(rec['ok'])} ({note})"
        )
    step_frames(env, no_action(), 2)
    return results


def save_overlay(env: object, path: Path, title: str, points: tuple = ()) -> None:
    img = overlay_from_env(
        env, include_all_sprites=True, points=points, title=title
    )
    Image.fromarray(img).save(path)


def dump_png(env: object, path: Path) -> None:
    Image.fromarray(env.render()).save(path)  # type: ignore[attr-defined]


def wait_load(env: object, *, max_frames: int = 720) -> dict[str, Any]:
    frames = 0
    while frames < max_frames:
        snap = snapshot_env(env)
        if snap.has_control and not snap.is_hold_up_item and not snap.is_text_mode:
            return {"ok": True, "frames": frames, "glance": glance(env)}
        if snap.is_text_mode:
            step_frames(env, action_for("A"), 2)
            step_frames(env, no_action(), 2)
            frames += 4
        else:
            step_frames(env, no_action(), 4)
            frames += 4
    return {"ok": False, "frames": frames, "glance": glance(env)}


def trigger_dungeon_load(env: object, entrance: int, *, module: int = 0x06) -> dict[str, Any]:
    poke_u16(env, ENTRANCE_ID, entrance)
    poke_u8(env, SUBMODULE, 0)
    poke_u8(env, MODULE, module)
    step_frames(env, no_action(), 2)
    result = wait_load(env, max_frames=900)
    g = result["glance"]
    result.update(
        {
            "entrance": entrance,
            "entrance_hex": f"0x{entrance:02X}",
            "module_written": module,
            "landed_room": g.get("room"),
            "landed_room_hex": g.get("room_hex"),
            "landed_module": g.get("module"),
            "indoors": g.get("indoors"),
            "xy": g.get("xy"),
            "has_control": g.get("has_control"),
        }
    )
    return result


def lobby_ok(g: dict[str, Any]) -> bool:
    return (
        bool(g.get("indoors"))
        and int(g.get("room", -1)) == HYPOTHESIS_ROOM
        and bool(g.get("has_control"))
        and int(g.get("module", -1)) == 0x07
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def boot_and_poke(state_name: str = SOURCE_STATE) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    env = build_boot_env(state_name, render_mode="rgb_array")
    env.reset()
    if not resync_custom_state(env, GAME_DIR, INTEGRATION, state_name):
        env.close()
        raise RuntimeError(f"failed to load {state_name}")
    settle_control(env)
    print(f"loaded {state_name} {glance(env)['room_hex']} xy={glance(env)['xy']}")
    pokes = apply_loadout(env)
    settle_control(env)
    return env, pokes, glance(env)


def cmd_pin(scan: bool) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env, pokes, before = boot_and_poke()
    failed = [p for p in pokes if not p["ok"]]
    if failed:
        print(f"FAIL poke: {failed}")
        write_json(OUT_DIR / "probe.json", {"ok": False, "pokes": pokes, "before": before})
        env.close()
        return 1
    origin = env.em.get_state()  # type: ignore[attr-defined]
    dump_png(env, OUT_DIR / "after_poke.png")

    ids = list(range(0x80)) if scan else list(WARP_CANDIDATES)
    hits: list[dict[str, Any]] = []
    tries: list[dict[str, Any]] = []
    landed: dict[str, Any] | None = None
    try:
        for eid in ids:
            env.em.set_state(origin)  # type: ignore[attr-defined]
            rec = trigger_dungeon_load(env, eid)
            summary = {
                "entrance_hex": rec["entrance_hex"],
                "ok": rec["ok"],
                "room_hex": rec["landed_room_hex"],
                "module": rec["landed_module"],
                "indoors": rec["indoors"],
                "xy": rec["xy"],
                "ctrl": rec["has_control"],
                "frames": rec["frames"],
            }
            tries.append(summary)
            print(
                f"  entrance {rec['entrance_hex']}: room={rec['landed_room_hex']} "
                f"mod=0x{int(rec['landed_module'] or 0):02X} indoor={rec['indoors']} "
                f"ctrl={int(bool(rec['has_control']))} xy={rec['xy']}"
            )
            if lobby_ok(rec["glance"]):
                hits.append(summary)
                landed = rec
                break
        if landed is None:
            print("no lobby hit via module 0x06; trying 0x05")
            for eid in WARP_CANDIDATES:
                env.em.set_state(origin)  # type: ignore[attr-defined]
                rec = trigger_dungeon_load(env, eid, module=0x05)
                summary = {
                    "entrance_hex": rec["entrance_hex"],
                    "module_written": 0x05,
                    "ok": rec["ok"],
                    "room_hex": rec["landed_room_hex"],
                    "module": rec["landed_module"],
                    "indoors": rec["indoors"],
                    "xy": rec["xy"],
                    "ctrl": rec["has_control"],
                    "frames": rec["frames"],
                }
                tries.append(summary)
                print(
                    f"  m05 entrance {rec['entrance_hex']}: room={rec['landed_room_hex']} "
                    f"mod=0x{int(rec['landed_module'] or 0):02X} indoor={rec['indoors']} "
                    f"ctrl={int(bool(rec['has_control']))}"
                )
                if lobby_ok(rec["glance"]):
                    hits.append(summary)
                    landed = rec
                    break

        payload: dict[str, Any] = {
            "schema": "alttp_swamp_entry",
            "schemaVersion": 1,
            "measured": datetime.now(timezone.utc).isoformat(),
            "source": "state_load_dev",
            "sourceState": SOURCE_STATE,
            "save": SAVE_NAME,
            "hypothesisRoom": f"0x{HYPOTHESIS_ROOM:02X}",
            "pokes": pokes,
            "before": before,
            "tries": tries,
            "hits": hits,
        }
        if landed is None:
            payload["ok"] = False
            payload["blocker"] = "no real 0x28 load"
            write_json(OUT_DIR / "probe.json", payload)
            print("no 0x28 lobby — not writing SwampPalaceEntry.state")
            return 1

        g = landed["glance"]
        save_overlay(env, OUT_DIR / "lobby.png", f"swamp lobby 0x{g['room']:02X}")
        dump_png(env, OUT_DIR / "lobby_raw.png")
        write_state_bytes(INTEGRATION_DIR / f"{SAVE_NAME}.state", env.em.get_state())
        payload["ok"] = True
        payload["pin"] = g
        payload["entranceUsed"] = landed["entrance_hex"]
        payload["waterState"] = {
            "F2BB": g["bytes"]["F2BB_ow_3b"],
            "F2FB": g["bytes"]["F2FB_ow_7b"],
            "note": "bit5=0x20 is dam/swamp overlay; confirm visually on overlay PNG",
        }
        write_json(OUT_DIR / "probe.json", payload)
        write_json(OUT_DIR / "leftover.json", g)
        print(
            f"PIN {SAVE_NAME} room={g['room_hex']} xy={g['xy']} "
            f"mod={g['module_hex']}/{g['submodule']} sword={g['sword']} "
            f"keys={g['keys']} F3CC={g['bytes']['F3CC_follower']} "
            f"hookshot={g['bytes']['F342_hookshot']}"
        )
        return 0
    finally:
        env.close()


def cmd_doors() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = INTEGRATION_DIR / f"{SAVE_NAME}.state"
    if not path.is_file():
        print(f"missing {path}")
        return 2
    env = build_boot_env(SAVE_NAME, render_mode="rgb_array")
    try:
        env.reset()
        resync_custom_state(env, GAME_DIR, INTEGRATION, SAVE_NAME)
        settle_control(env)
        start = glance(env)
        origin = env.em.get_state()  # type: ignore[attr-defined]
        holds: dict[str, Any] = {"start": start, "holds": {}}
        for d in HOLD_DIRS:
            env.em.set_state(origin)  # type: ignore[attr-defined]
            settle_control(env)
            before = snapshot_env(env)
            for _ in range(200):
                step_frames(env, action_for(d), 3)
                snap = snapshot_env(env)
                left = snap.room_base_id != before.room_base_id or snap.indoors != before.indoors
                if left or not snap.has_control:
                    break
            end = glance(env)
            holds["holds"][d] = {
                "end": end,
                "left_room": end["room"] != start["room"] or end["indoors"] != start["indoors"],
            }
            save_overlay(env, OUT_DIR / f"hold_{d.lower()}.png", f"hold {d}")
            print(
                f"  {d:5} -> room={end['room_hex']} indoor={end['indoors']} "
                f"xy={end['xy']} ctrl={int(end['has_control'])} "
                f"mod={end['module_hex']}/{end['submodule']}"
            )
        write_json(OUT_DIR / "holds.json", holds)
        return 0
    finally:
        env.close()


def cmd_edge() -> int:
    from alttp.opening_route.room_engine import run_room_edge
    from alttp.room_map import load_room_map

    map_id = f"room_{HYPOTHESIS_ROOM:02x}"
    try:
        room_map = load_room_map(map_id)
    except FileNotFoundError as exc:
        print(f"map missing: {exc}")
        return 2
    door = next((d for d in room_map.doors if d.role in ("primary", "zelda_path")), None)
    if door is None and room_map.doors:
        door = room_map.doors[0]
    if door is None:
        print("no doors in map")
        return 2
    env = build_boot_env(SAVE_NAME, render_mode="rgb_array")
    try:
        env.reset()
        resync_custom_state(env, GAME_DIR, INTEGRATION, SAVE_NAME)
        settle_control(env)
        result = run_room_edge(
            env, map_id, door.label, clear=True, source="state_load_dev"
        )
        end = glance(env)
        save_overlay(env, OUT_DIR / f"{map_id}_{door.label}.png", door.label)
        payload = {
            "ok": bool(result.ok),
            "phase": result.phase,
            "frames": result.frames,
            "blocker": result.blocker,
            "door": door.label,
            "end": end,
        }
        write_json(OUT_DIR / f"{map_id}_{door.label}.json", payload)
        print(
            f"edge {door.label} ok={result.ok} phase={result.phase} "
            f"blocker={result.blocker!r} leftover room={end['room_hex']} xy={end['xy']}"
        )
        return 0 if result.ok else 1
    finally:
        env.close()


def _hold_until_settle(
    env: object, button: str, *, max_frames: int = 720
) -> dict[str, Any]:
    before = snapshot_env(env)
    frames = 0
    saw_busy = False
    while frames < max_frames:
        step_frames(env, action_for(button), 3)
        frames += 3
        snap = snapshot_env(env)
        left = (
            snap.room_base_id != before.room_base_id
            or snap.indoors != before.indoors
            or snap.game_mode != before.game_mode
        )
        if not snap.has_control or snap.submodule != 0:
            saw_busy = True
            step_frames(env, no_action(), 4)
            frames += 4
            settled = settle_control(env, max_frames=480)
            frames += settled.frames
            return {
                "frames": frames,
                "left": True,
                "busy": True,
                "settle_ok": settled.ok,
                "end": glance(env),
            }
        if left:
            settled = settle_control(env, max_frames=480)
            frames += settled.frames
            return {
                "frames": frames,
                "left": True,
                "busy": saw_busy,
                "settle_ok": settled.ok,
                "end": glance(env),
            }
    return {
        "frames": frames,
        "left": False,
        "busy": saw_busy,
        "settle_ok": True,
        "end": glance(env),
    }


def cmd_claim() -> int:
    """Walk spawn through the flooded canal toward chest / north stairs."""
    env = build_boot_env(SAVE_NAME, render_mode="rgb_array")
    try:
        env.reset()
        resync_custom_state(env, GAME_DIR, INTEGRATION, SAVE_NAME)
        settle_control(env)
        start = glance(env)
        print(f"claim start {start['room_hex']} xy={start['xy']}")
        room = int(start["room"])
        claims: list[dict[str, Any]] = []
        spawn_x, spawn_y = int(start["x"]), int(start["y"])
        # West lip then DOWN the ladder into the canal well, then UP/LEFT.
        targets = [
            ("canal", spawn_x, 1410),
            ("lip", spawn_x, 1352),
            ("lip_west", 4160, 1352),
        ]
        for label, x, y in targets:
            mv = move_to(
                env,
                Waypoint(x, y, tolerance=8, room=room, label=label),
                max_frames=480,
            )
            if not snapshot_env(env).has_control:
                settle_control(env, max_frames=480)
            g = glance(env)
            rec = {
                "label": label,
                "want": [x, y],
                "ok": bool(mv.ok),
                "reason": mv.reason,
                "end": {
                    "room_hex": g["room_hex"],
                    "xy": g["xy"],
                    "module_hex": g["module_hex"],
                    "submodule": g["submodule"],
                    "ctrl": g["has_control"],
                    "keys": g["keys"],
                },
            }
            claims.append(rec)
            save_overlay(env, OUT_DIR / f"claim_{label}.png", label)
            print(
                f"  {label} ok={int(mv.ok)} -> {g['room_hex']} {g['xy']} "
                f"mod={g['module_hex']}/{g['submodule']} {mv.reason}"
            )
            if g["room"] != room or g["module"] != 0x07:
                break
        rec_dn = _hold_until_settle(env, "DOWN", max_frames=480)
        claims.append({"label": "hold_down_ladder", "hold": rec_dn})
        save_overlay(env, OUT_DIR / "claim_down_ladder.png", "DOWN west ladder")
        end = rec_dn["end"]
        print(
            f"  hold DOWN -> {end['room_hex']} {end['xy']} "
            f"mod={end['module_hex']}/{end['submodule']} ctrl={int(end['has_control'])}"
        )
        rec_up = _hold_until_settle(env, "UP", max_frames=720)
        claims.append({"label": "hold_up_ladder", "hold": rec_up})
        save_overlay(env, OUT_DIR / "claim_up_ladder.png", "UP west ladder")
        end = rec_up["end"]
        print(
            f"  hold UP -> {end['room_hex']} {end['xy']} "
            f"mod={end['module_hex']}/{end['submodule']} ctrl={int(end['has_control'])} "
            f"keys={end['keys']}"
        )
        rec_l = _hold_until_settle(env, "LEFT", max_frames=360)
        claims.append({"label": "hold_left_well", "hold": rec_l})
        save_overlay(env, OUT_DIR / "claim_left_well.png", "LEFT from ladder")
        end = rec_l["end"]
        print(
            f"  hold LEFT -> {end['room_hex']} {end['xy']} "
            f"mod={end['module_hex']}/{end['submodule']} ctrl={int(end['has_control'])}"
        )
        write_json(OUT_DIR / "claims.json", {"start": start, "claims": claims})
        return 0
    finally:
        env.close()


def cmd_lipscan() -> int:
    """On the canal lip, hold UP at each x to find the north-floor climb."""
    env = build_boot_env(SAVE_NAME, render_mode="rgb_array")
    try:
        env.reset()
        resync_custom_state(env, GAME_DIR, INTEGRATION, SAVE_NAME)
        settle_control(env)
        origin = env.em.get_state()  # type: ignore[attr-defined]
        room = HYPOTHESIS_ROOM
        rows: list[dict[str, Any]] = []
        for x in range(4344, 4144, -16):
            env.em.set_state(origin)  # type: ignore[attr-defined]
            settle_control(env)
            move_to(env, Waypoint(4344, 1410, tolerance=8, room=room), max_frames=300)
            move_to(env, Waypoint(4344, 1352, tolerance=8, room=room), max_frames=300)
            move_to(env, Waypoint(x, 1352, tolerance=8, room=room), max_frames=360)
            before = glance(env)
            for _ in range(40):
                step_frames(env, action_for("UP"), 3)
            settle_control(env, max_frames=120)
            after = glance(env)
            rec = {
                "want_x": x,
                "before": before["xy"],
                "after": after["xy"],
                "dy": after["y"] - before["y"],
                "sub": after["submodule"],
                "room_hex": after["room_hex"],
                "ctrl": after["has_control"],
            }
            rows.append(rec)
            moved = rec["dy"] != 0 or after["xy"] != before["xy"]
            print(
                f"  x={x} before={before['xy']} after={after['xy']} "
                f"dy={rec['dy']} sub={after['submodule']} moved={int(moved)}"
            )
            if after["y"] < 1320 or after["room"] != room:
                save_overlay(env, OUT_DIR / f"lipscan_{x}.png", f"lipscan {x}")
        write_json(OUT_DIR / "lipscan.json", {"rows": rows})
        return 0
    finally:
        env.close()


def cmd_reflood(*, flippers: bool = False) -> int:
    """Poke dungeon water flags on the pin, reload entrance 0x25, glance."""
    extra = [
        (0xF051, 0x01, "room 0x28 water bit"),
        (0xF216, 0x80, "dam switch pulled"),
        (0xF2BB, 0x20, "LW dam overlay"),
        (0xF2FB, 0x20, "DW swamp overlay"),
    ]
    if flippers:
        extra.append((0xF356, 1, "flippers"))
    env = build_boot_env(SAVE_NAME, render_mode="rgb_array")
    try:
        env.reset()
        resync_custom_state(env, GAME_DIR, INTEGRATION, SAVE_NAME)
        settle_control(env)
        before = glance(env)
        pokes = []
        for offset, value, note in extra:
            rec = poke_u8(env, offset, value)
            rec["note"] = note
            pokes.append(rec)
            print(f"  poke ${rec['offset']}={rec['want']:02X} got={rec['got']:02X} ok={int(rec['ok'])} ({note})")
        rec = trigger_dungeon_load(env, 0x25)
        end = rec["glance"]
        save_overlay(env, OUT_DIR / "reflood.png", "reflood 0x25")
        dump_png(env, OUT_DIR / "reflood_raw.png")
        write_json(
            OUT_DIR / "reflood.json",
            {"before": before, "pokes": pokes, "load": rec, "end": end},
        )
        print(
            f"reflood room={end['room_hex']} xy={end['xy']} "
            f"mod={end['module_hex']}/{end['submodule']} ctrl={int(end['has_control'])} "
            f"F051={end['bytes'].get('F051', 'n/a')}"
        )
        if lobby_ok(end):
            write_state_bytes(INTEGRATION_DIR / f"{SAVE_NAME}.state", env.em.get_state())
            print("rewrote SwampPalaceEntry.state from reflood")
        return 0
    finally:
        env.close()


def cmd_south() -> int:
    """Hold DOWN through the south palace door; record overworld leftover."""
    env = build_boot_env(SAVE_NAME, render_mode="rgb_array")
    try:
        env.reset()
        resync_custom_state(env, GAME_DIR, INTEGRATION, SAVE_NAME)
        settle_control(env)
        start = glance(env)
        rec = _hold_until_settle(env, "DOWN", max_frames=900)
        end = rec["end"]
        save_overlay(env, OUT_DIR / "south_exit.png", "south exit")
        dump_png(env, OUT_DIR / "south_exit_raw.png")
        payload = {"start": start, "hold": rec, "end": end}
        write_json(OUT_DIR / "south_exit.json", payload)
        print(
            f"south -> room={end['room_hex']} indoor={end['indoors']} "
            f"xy={end['xy']} screen=0x{int(end.get('screen', 0)):02X} "
            f"mod={end['module_hex']}/{end['submodule']} ctrl={int(end['has_control'])}"
        )
        return 0
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan", action="store_true", help="Try entrance IDs 0x00-0x7F")
    p.add_argument("--doors", action="store_true", help="Hold-cardinal from pin")
    p.add_argument("--edge", action="store_true", help="room_engine first door")
    p.add_argument("--claim", action="store_true", help="Walk claims from pin")
    p.add_argument("--south", action="store_true", help="Hold DOWN through south exit")
    p.add_argument("--lipscan", action="store_true", help="UP-scan the canal lip")
    p.add_argument("--reflood", action="store_true", help="Reload 0x25 after water flags")
    p.add_argument("--flippers", action="store_true", help="Also poke flippers with --reflood")
    args = p.parse_args(argv)
    if args.doors:
        return cmd_doors()
    if args.edge:
        return cmd_edge()
    if args.claim:
        return cmd_claim()
    if args.south:
        return cmd_south()
    if args.lipscan:
        return cmd_lipscan()
    if args.reflood:
        return cmd_reflood(flippers=args.flippers)
    return cmd_pin(scan=args.scan)


if __name__ == "__main__":
    raise SystemExit(main())

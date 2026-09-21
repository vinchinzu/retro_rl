"""Skull Woods front-entrance pin (rr-lvnu). Isolated / state-load only.

Load HyruleCastleGrounds, poke RTA-typical kit (no fire rod), enter the
southeast skull door (entrance 0x29 → hypothesized room 0x58). Not the
boss hut. Indoor pin only after a real room-load.

Usage:
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_skull_entry.py
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_skull_entry.py --phase poke
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_skull_entry.py --phase enter
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_skull_entry.py --phase map
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_skull_entry.py --phase hop --dir LEFT
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

from alttp.paths import GAME_SPEC, RECORDINGS_DIR
from alttp.primitives import Waypoint, active_sprites, move_to, settle_control
from alttp.ram import (
    DARK_WORLD_FLAG,
    EQUIP_SWORD,
    FOLLOWER,
    INDOORS,
    LINK_HP,
    LINK_MAX_HP,
    LINK_X,
    LINK_Y,
    MODULE,
    NUM_KEYS,
    ROOM_ID,
    SCREEN_ID,
    SUBMODULE,
    snapshot_to_diag,
    wram_index,
)
from alttp.room_sense import detect_edge, overlay_from_env
from alttp.screen_glance import leftover_from_snapshot
from alttp.startup import action_for, build_boot_env, no_action, snapshot_env, step_frames
from retro_harness.env import read_state_bytes, write_state_bytes

OUT_DIR = RECORDINGS_DIR / "probe_skull_entry"
STATE_NAME = "SkullWoodsEntry"
SOURCE_STATE = "HyruleCastleGrounds"
# Vanilla entrance 0x29 = Skull Woods First Section Door (southeast skull).
# Archipelago: room 0x58, OW screen 0x40. Boss hut is 0x2A / room 0x59.
FRONT_ENTRANCE_ID = 0x29
HYPOTHESIS_ROOM = 0x58
OW_SCREEN = 0x40

# Loadout pokes (WRAM $F3xx). Prize (fire rod $F345) stays 0.
POKES: tuple[tuple[int, int, str], ...] = (
    (0xF342, 0x01, "hookshot"),
    (0xF343, 0x0A, "bombs"),
    (0xF345, 0x00, "fire_rod_absent"),
    (0xF34B, 0x01, "hammer"),
    (0xF354, 0x01, "power_gloves"),
    (0xF355, 0x01, "boots"),
    (0xF357, 0x01, "moon_pearl"),
    (0xF359, 0x02, "master_sword"),
    (0xF35A, 0x01, "fighter_shield"),
    (0xF36C, 0x38, "max_hp_7_hearts"),
    (0xF36D, 0x38, "hp_full"),
    (0xF36E, 0x80, "magic_full"),
    (0xF3C5, 0x03, "progress_agahnim1"),
    (0xF3C8, 0x03, "spawn_post_agahnim"),
    (0xF3CA, 0x40, "sram_dark_world"),
)

LOADOUT_READS: tuple[int, ...] = (
    0xF340, 0xF342, 0xF343, 0xF345, 0xF34A, 0xF34B, 0xF353,
    0xF354, 0xF355, 0xF356, 0xF357, 0xF359, 0xF35A, 0xF35B,
    0xF36C, 0xF36D, 0xF36E, 0xF36F, 0xF3C5, 0xF3C8, 0xF3CA, 0xF3CC,
)


def _mem(env: object) -> Any:
    base = getattr(env, "unwrapped", env)
    data = getattr(base, "data", None) or getattr(env, "data", None)
    if data is None:
        raise RuntimeError("env has no data.memory")
    return data.memory


def _assign(env: object, offset: int, fmt: str, value: int) -> int:
    """Assign WRAM. $F3xx is not a get_ram() index; SNES maps high WRAM at $7Exxxx."""
    mem = _mem(env)
    candidates = [int(offset)]
    if 0x2000 <= int(offset) < 0x20000:
        candidates.append(0x7E0000 + int(offset))
    last = None
    for addr in candidates:
        try:
            mem.assign(addr, fmt, value)
            return addr
        except (IndexError, KeyError, ValueError) as exc:
            last = exc
    raise RuntimeError(f"memory.assign failed for 0x{offset:04X}: {last}")


def poke_u8(env: object, offset: int, value: int) -> int:
    return _assign(env, offset, "|u1", int(value) & 0xFF)


def poke_u16(env: object, offset: int, value: int) -> int:
    return _assign(env, offset, "<u2", int(value) & 0xFFFF)


def read_u8(env: object, offset: int) -> int:
    ram = env.get_ram()  # type: ignore[attr-defined]
    return int(ram[wram_index(offset)])


def read_u16(env: object, offset: int) -> int:
    ram = env.get_ram()  # type: ignore[attr-defined]
    idx = wram_index(offset)
    return int(ram[idx]) | (int(ram[idx + 1]) << 8)


def apply_pokes(env: object) -> list[dict[str, Any]]:
    log: list[dict[str, Any]] = []
    for offset, value, name in POKES:
        before = read_u8(env, offset)
        mapped = poke_u8(env, offset, value)
        step_frames(env, no_action(), 1)
        after = read_u8(env, offset)
        log.append(
            {
                "name": name,
                "offset": f"0x{offset:04X}",
                "assign": f"0x{mapped:06X}",
                "want": value,
                "before": before,
                "after": after,
                "ok": after == value,
            }
        )
    return log


def loadout_bytes(env: object) -> dict[str, int]:
    return {f"0x{off:04X}": read_u8(env, off) for off in LOADOUT_READS}


def glance(env: object) -> dict[str, Any]:
    snap = snapshot_env(env)
    leftover = leftover_from_snapshot(snap)
    sprites = [
        {"type": f"0x{s.sprite_type:02X}", "xy": [s.x, s.y], "hp": s.hp}
        for s in active_sprites(env)
    ]
    return {
        "room": leftover["room"],
        "room_hex": f"0x{leftover['room']:02X}",
        "module": leftover["module"],
        "module_hex": f"0x{leftover['module'] & 0xFF:02X}",
        "submodule": leftover["submodule"],
        "x": leftover["x"],
        "y": leftover["y"],
        "xy": leftover["xy"],
        "sword": leftover["sword"],
        "follower": leftover["follower"],
        "keys": leftover["keys"],
        "indoors": leftover["indoors"],
        "screen": leftover["screen"],
        "screen_hex": f"0x{leftover['screen']:02X}",
        "dark_world": bool(snap.dark_world),
        "has_control": bool(snap.has_control),
        "F3CC": read_u8(env, FOLLOWER),
        "F359": read_u8(env, EQUIP_SWORD),
        "F36F_keys": read_u8(env, NUM_KEYS),
        "F0FFF_dw": read_u8(env, DARK_WORLD_FLAG),
        "010E_entrance": read_u16(env, 0x010E),
        "040C_dungeon": read_u8(env, 0x040C),
        "00EE_layer": read_u8(env, 0x00EE),
        "loadout": loadout_bytes(env),
        "sprites": sprites[:12],
        "diag": snapshot_to_diag(snap),
    }


def save_overlay(env: object, name: str, title: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    img = overlay_from_env(env, include_all_sprites=True, title=title)
    Image.fromarray(img).save(path)
    return path


def wait_control(env: object, *, max_frames: int = 720) -> dict[str, Any]:
    sc = settle_control(env, max_frames=max_frames)
    return {"ok": sc.ok, "reason": sc.reason, "frames": sc.frames, "glance": glance(env)}


def try_assign_offsets(env: object) -> list[dict[str, Any]]:
    """Read-after-write for 0xF359 vs 0x7EF359. Restores the original byte."""
    orig = read_u8(env, EQUIP_SWORD)
    mem = _mem(env)
    trials: list[dict[str, Any]] = []
    for offset in (EQUIP_SWORD, 0x7E0000 + EQUIP_SWORD):
        err = None
        try:
            mem.assign(int(offset), "|u1", 2)
        except (IndexError, KeyError, ValueError) as exc:
            err = str(exc)
        step_frames(env, no_action(), 1)
        got = read_u8(env, EQUIP_SWORD)
        trials.append(
            {
                "assign_offset": f"0x{offset:06X}",
                "got_F359": got,
                "ok": got == 2,
                "error": err,
            }
        )
        poke_u8(env, EQUIP_SWORD, orig)
        step_frames(env, no_action(), 1)
    return trials


def trigger_entrance(env: object, entrance_id: int) -> dict[str, Any]:
    """Ask the game to load a building entrance (module 0x0F). Real room-load."""
    before = glance(env)
    poke_u16(env, 0x010E, entrance_id)
    poke_u8(env, MODULE, 0x0F)
    poke_u8(env, SUBMODULE, 0x00)
    step_frames(env, no_action(), 2)
    frames = 2
    snap = snapshot_env(env)
    idle = 0
    prev = (snap.game_mode, snap.submodule, snap.room_base_id, snap.indoors)
    while frames < 900:
        step_frames(env, no_action(), 4)
        frames += 4
        snap = snapshot_env(env)
        cur = (snap.game_mode, snap.submodule, snap.room_base_id, snap.indoors)
        if cur == prev:
            idle += 1
        else:
            idle = 0
            prev = cur
        if snap.has_control and snap.indoors and snap.game_mode == 0x07:
            break
        if idle >= 80:
            break
    sc = settle_control(env, max_frames=480)
    return {
        "entrance_id": entrance_id,
        "before": before,
        "frames": frames + sc.frames,
        "settle": {"ok": sc.ok, "reason": sc.reason},
        "after": glance(env),
    }


def wait_mode(env: object, *, max_frames: int = 720) -> int:
    frames = 0
    while frames < max_frames:
        snap = snapshot_env(env)
        if snap.has_control and snap.game_mode in (0x07, 0x09) and snap.submodule == 0:
            break
        step_frames(env, no_action(), 4)
        frames += 4
    sc = settle_control(env, max_frames=240)
    return frames + sc.frames


def load_overworld(env: object, screen: int, x: int, y: int) -> dict[str, Any]:
    """Module 0x08 rebuilds OW from $8A — real screen-load, not a coord poke."""
    before = glance(env)
    poke_u8(env, INDOORS, 0)
    poke_u8(env, SCREEN_ID, screen)
    poke_u8(env, 0x040A, screen)
    poke_u16(env, LINK_X, x)
    poke_u16(env, LINK_Y, y)
    poke_u8(env, DARK_WORLD_FLAG, 1)
    poke_u8(env, 0xF3CA, 0x40)
    poke_u8(env, 0x040C, 0xFF)
    poke_u8(env, MODULE, 0x08)
    poke_u8(env, SUBMODULE, 0x00)
    frames = wait_mode(env, max_frames=900)
    return {
        "method": "module_08_ow",
        "want_screen": screen,
        "want_xy": [x, y],
        "before": before,
        "frames": frames,
        "after": glance(env),
    }


def load_dungeon_room(env: object, room: int, x: int, y: int, dungeon_id: int = 0x10) -> dict[str, Any]:
    """Module 0x07 with $A0 room — real dungeon load if overlay is the dungeon."""
    before = glance(env)
    poke_u16(env, ROOM_ID, room)
    poke_u8(env, INDOORS, 1)
    poke_u8(env, 0x040C, dungeon_id)
    poke_u16(env, LINK_X, x)
    poke_u16(env, LINK_Y, y)
    poke_u8(env, MODULE, 0x07)
    poke_u8(env, SUBMODULE, 0x00)
    frames = wait_mode(env, max_frames=900)
    return {
        "method": "module_07_uw",
        "want_room": room,
        "want_xy": [x, y],
        "dungeon_id": dungeon_id,
        "before": before,
        "frames": frames,
        "after": glance(env),
    }


def load_entrance(env: object, entrance_id: int, *, module: int = 0x0F) -> dict[str, Any]:
    before = glance(env)
    poke_u16(env, 0x010E, entrance_id)
    poke_u8(env, INDOORS, 1)
    poke_u8(env, MODULE, module)
    poke_u8(env, SUBMODULE, 0x00)
    frames = wait_mode(env, max_frames=900)
    return {
        "method": f"module_{module:02X}_entrance",
        "entrance_id": entrance_id,
        "before": before,
        "frames": frames,
        "after": glance(env),
    }


def hold_dir(env: object, direction: str, *, max_frames: int = 360) -> dict[str, Any]:
    start = snapshot_env(env)
    rec: dict[str, Any] = {
        "dir": direction,
        "fromXy": [start.link_x, start.link_y],
        "fromRoom": f"0x{start.room_base_id:02X}",
        "fromScreen": f"0x{start.screen_id:02X}",
        "frames": 0,
        "stuck": False,
    }
    prev = (start.link_x, start.link_y)
    stuck = 0
    frames = 0
    before = start
    while frames < max_frames:
        step_frames(env, action_for(direction), 4)
        frames += 4
        after = snapshot_env(env)
        edge = detect_edge(
            before,
            after,
            expected_room=start.room_base_id,
            frames=frames,
            label=direction,
            preferred_direction=direction,
        )
        if edge is not None:
            rec.update(
                {
                    "frames": frames,
                    "toRoom": f"0x{edge.to_room:02X}",
                    "toXy": list(edge.to_xy),
                    "outdoors": edge.outdoors,
                    "approachXy": list(edge.from_xy),
                }
            )
            return rec
        if after.indoors != start.indoors or after.screen_id != start.screen_id:
            rec.update(
                {
                    "frames": frames,
                    "toRoom": f"0x{after.room_base_id:02X}",
                    "toScreen": f"0x{after.screen_id:02X}",
                    "toXy": [after.link_x, after.link_y],
                    "outdoors": not after.indoors,
                    "module": after.game_mode,
                    "submodule": after.submodule,
                }
            )
            return rec
        xy = (after.link_x, after.link_y)
        if xy == prev:
            stuck += 1
        else:
            stuck = 0
            prev = xy
        if stuck >= 12:
            rec["stuck"] = True
            rec["frames"] = frames
            rec["endXy"] = [after.link_x, after.link_y]
            return rec
        before = after
    rec["frames"] = frames
    rec["endXy"] = [snapshot_env(env).link_x, snapshot_env(env).link_y]
    rec["timeout"] = True
    return rec


def write_json(name: str, payload: Any) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def phase_poke(env: object) -> dict[str, Any]:
    settle_control(env)
    trials = try_assign_offsets(env)
    # Prefer the offset that actually wrote $F359.
    working = next((t for t in trials if t["ok"]), None)
    pokes = apply_pokes(env)
    step_frames(env, no_action(), 8)
    g = glance(env)
    overlay = save_overlay(env, "poke_grounds.png", "poked HyruleCastleGrounds")
    return {
        "assign_trials": trials,
        "working_assign": working,
        "pokes": pokes,
        "pokes_ok": all(p["ok"] for p in pokes),
        "glance": g,
        "overlay": str(overlay),
    }


def _pin_if_skull(env: object, g: dict[str, Any]) -> str | None:
    """Save SkullWoodsEntry only after a controllable indoor load that is not the boss hut."""
    if not (g["indoors"] and g["module"] == 0x07 and g["has_control"]):
        return None
    if g["room"] in (0x29, 0x39, 0x59):
        return None
    GAME_SPEC.save_state(env, STATE_NAME)
    return STATE_NAME


def phase_enter(env: object) -> dict[str, Any]:
    poke_info = phase_poke(env)
    blob = env.em.get_state()  # type: ignore[attr-defined]
    attempts: list[dict[str, Any]] = []

    def restore() -> None:
        env.em.set_state(blob)  # type: ignore[attr-defined]
        step_frames(env, no_action(), 1)

    # 1) Real OW rebuild onto Skull Woods (southeast skull at ~744,584).
    restore()
    ow = load_overworld(env, OW_SCREEN, 744, 640)
    save_overlay(env, "warp_mod08.png", f"mod08 {ow['after']['screen_hex']} {ow['after']['xy']}")
    attempts.append(ow)
    if ow["after"]["screen"] == OW_SCREEN and not ow["after"]["indoors"]:
        held = hold_dir(env, "UP", max_frames=480)
        sc = settle_control(env, max_frames=480)
        g = glance(env)
        save_overlay(env, "enter_from_ow.png", f"UP {g['room_hex']} {g['xy']}")
        attempts.append({"method": "hold_up_into_skull", "hold": held, "after": g})
        saved = _pin_if_skull(env, g)
        if saved:
            return {
                "poke": poke_info,
                "attempts": attempts,
                "indoor_ok": True,
                "room_ok": g["room"] == HYPOTHESIS_ROOM,
                "saved": saved,
                "glance": g,
            }

    # 2) Entrance table via module 0x0F / 0x08 / 0x10.
    for module in (0x0F, 0x08, 0x10):
        restore()
        rec = load_entrance(env, FRONT_ENTRANCE_ID, module=module)
        g = rec["after"]
        save_overlay(
            env,
            f"enter_mod{module:02x}.png",
            f"mod {module:#x} {g['room_hex']} {g['xy']}",
        )
        attempts.append(rec)
        saved = _pin_if_skull(env, g)
        if saved:
            return {
                "poke": poke_info,
                "attempts": attempts,
                "indoor_ok": True,
                "room_ok": g["room"] == HYPOTHESIS_ROOM,
                "saved": saved,
                "glance": g,
            }

    # 3) Direct dungeon load of hypothesized 0x58 (must look like Skull Woods).
    restore()
    uw = load_dungeon_room(env, HYPOTHESIS_ROOM, 744, 584)
    g = uw["after"]
    save_overlay(env, "enter_uw58.png", f"uw 0x58 {g['room_hex']} {g['xy']}")
    attempts.append(uw)
    saved = _pin_if_skull(env, g)
    return {
        "poke": poke_info,
        "attempts": attempts,
        "indoor_ok": bool(saved),
        "room_ok": bool(saved) and g["room"] == HYPOTHESIS_ROOM,
        "saved": saved,
        "glance": g,
    }


def walk_screens(env: object, direction: str, *, n: int = 6, max_frames: int = 1200) -> list[dict[str, Any]]:
    """Hold one cardinal across overworld screens; snapshot each change."""
    logs: list[dict[str, Any]] = []
    for i in range(n):
        start = snapshot_env(env)
        rec = hold_dir(env, direction, max_frames=max_frames)
        sc = settle_control(env, max_frames=240)
        g = glance(env)
        name = f"walk_{direction.lower()}_{i}_{g['screen_hex']}.png"
        save_overlay(env, name, f"{direction} {g['screen_hex']} {g['xy']}")
        logs.append(
            {
                "i": i,
                "from_screen": f"0x{start.screen_id:02X}",
                "from_xy": [start.link_x, start.link_y],
                "hold": rec,
                "glance": {
                    "screen_hex": g["screen_hex"],
                    "xy": g["xy"],
                    "module_hex": g["module_hex"],
                    "submodule": g["submodule"],
                    "indoors": g["indoors"],
                    "dark_world": g["dark_world"],
                    "has_control": g["has_control"],
                },
                "overlay": name,
            }
        )
        if rec.get("stuck") or rec.get("timeout"):
            break
        if g["indoors"]:
            break
    return logs


def reach_dw_6b(env: object) -> dict[str, Any]:
    """Poke kit, module-08 to house, south off porch, west onto DW 0x6B."""
    poke_info = phase_poke(env)
    ow = load_overworld(env, OW_SCREEN, 744, 640)
    save_overlay(env, "walk_start.png", f"walk start {ow['after']['screen_hex']} {ow['after']['xy']}")
    clear = move_to(
        env,
        Waypoint(2394, 2940, tolerance=8, label="house_clear"),
        max_frames=900,
    )
    save_overlay(env, "walk_clear.png", f"clear {glance(env)['xy']}")
    west = walk_screens(env, "LEFT", n=2, max_frames=1500)
    g = glance(env)
    return {
        "poke": poke_info,
        "start": ow["after"],
        "clear": {
            "ok": clear.ok,
            "reason": clear.reason,
            "xy": [clear.snapshot.link_x, clear.snapshot.link_y],
        },
        "west": west,
        "glance": g,
    }


def phase_walk(env: object, direction: str) -> dict[str, Any]:
    prefix = reach_dw_6b(env)
    to_shop = walk_screens(env, "RIGHT", n=1, max_frames=800)
    save_overlay(env, "dw_6c.png", f"shop {glance(env)['screen_hex']} {glance(env)['xy']}")
    # South dirt path of 0x6C; slope up to the bomb shop is at house x≈2394.
    along = move_to(
        env,
        Waypoint(2394, 2937, tolerance=10, label="below_shop"),
        max_frames=900,
    )
    save_overlay(env, "dw_6c_below.png", f"below {glance(env)['xy']}")
    up = hold_dir(env, "UP", max_frames=700)
    sc = settle_control(env, max_frames=240)
    g_up = glance(env)
    save_overlay(env, "dw_6c_up_slope.png", f"upslope {g_up['screen_hex']} {g_up['xy']}")
    blob = env.em.get_state()  # type: ignore[attr-defined]
    write_state_bytes(OUT_DIR / "dw_6c_north.state", blob)
    probes: dict[str, Any] = {}
    for d in ("UP", "DOWN", "LEFT", "RIGHT"):
        env.em.set_state(blob)  # type: ignore[attr-defined]
        step_frames(env, no_action(), 1)
        rec = hold_dir(env, d, max_frames=400)
        g = glance(env)
        save_overlay(env, f"shop_{d.lower()}.png", f"shop {d} {g['screen_hex']} {g['xy']}")
        probes[d] = {
            "hold": rec,
            "glance": {"screen_hex": g["screen_hex"], "xy": g["xy"], "indoors": g["indoors"]},
        }
    env.em.set_state(blob)  # type: ignore[attr-defined]
    step_frames(env, no_action(), 1)
    march_dir = direction if direction in ("UP", "DOWN", "LEFT", "RIGHT") else "UP"
    march = walk_screens(env, march_dir, n=6, max_frames=1500)
    g = glance(env)
    prefix.update(
        {
            "to_shop": to_shop,
            "along": {"ok": along.ok, "reason": along.reason, "xy": [along.snapshot.link_x, along.snapshot.link_y]},
            "up_slope": up,
            "after_up": g_up,
            "probes": probes,
            "march": march,
            "glance": g,
        }
    )
    return prefix


def phase_pyramid(env: object) -> dict[str, Any]:
    """From dw_6c_north (pyramid south) find a west passage toward 0x40."""
    blob_path = OUT_DIR / "dw_6c_north.state"
    env.em.set_state(read_state_bytes(blob_path))  # type: ignore[attr-defined]
    step_frames(env, no_action(), 1)
    settle_control(env)
    start = glance(env)
    save_overlay(env, "pyr_start.png", f"pyr {start['screen_hex']} {start['xy']}")
    base = env.em.get_state()  # type: ignore[attr-defined]
    scans: list[dict[str, Any]] = []
    # Sweep Y along the south face and hold LEFT.
    for y in (2472, 2488, 2504, 2520, 2536, 2552, 2440, 2408, 2360, 2300):
        env.em.set_state(base)  # type: ignore[attr-defined]
        step_frames(env, no_action(), 1)
        moved = move_to(
            env,
            Waypoint(start["x"], y, tolerance=6, label=f"y{y}"),
            max_frames=400,
        )
        held = hold_dir(env, "LEFT", max_frames=500)
        g = glance(env)
        save_overlay(env, f"pyr_left_y{y}.png", f"y{y} {g['screen_hex']} {g['xy']}")
        scans.append(
            {
                "y": y,
                "move_ok": moved.ok,
                "move_xy": [moved.snapshot.link_x, moved.snapshot.link_y],
                "hold": {
                    k: held.get(k)
                    for k in ("stuck", "frames", "endXy", "toScreen", "toXy")
                },
                "glance": {
                    "screen_hex": g["screen_hex"],
                    "xy": g["xy"],
                    "indoors": g["indoors"],
                },
            }
        )
    # Slash-walk west through the SW bush patch (y=2472 ended in bushes).
    env.em.set_state(base)  # type: ignore[attr-defined]
    step_frames(env, no_action(), 1)
    move_to(env, Waypoint(2408, 2480, tolerance=6, label="bush_y"), max_frames=300)
    hold_dir(env, "LEFT", max_frames=120)
    save_overlay(env, "pyr_pre_slash.png", f"pre-slash {glance(env)['xy']}")
    frames = 0
    while frames < 500:
        step_frames(env, action_for("LEFT", "B"), 8)
        step_frames(env, action_for("LEFT"), 4)
        frames += 12
        snap = snapshot_env(env)
        if snap.screen_id != start["screen"] or snap.link_x < 1800:
            break
        if frames > 60 and snap.link_x >= 2330:
            # not moving; try UP+LEFT slash
            step_frames(env, action_for("UP", "LEFT", "B"), 8)
    sc = settle_control(env, max_frames=180)
    g = glance(env)
    save_overlay(env, "pyr_slash_left.png", f"slash {g['screen_hex']} {g['xy']}")
    env.em.set_state(base)  # type: ignore[attr-defined]
    step_frames(env, no_action(), 1)
    return {
        "start": start,
        "scans": scans,
        "slash": {"frames": frames, "glance": g},
        "glance": start,
    }


def phase_shop(env: object) -> dict[str, Any]:
    """Enter bomb shop, poke $8A=0x40, walk out — real OW load of Skull Woods."""
    env.em.set_state(read_state_bytes(OUT_DIR / "dw_6c_north.state"))  # type: ignore[attr-defined]
    step_frames(env, no_action(), 1)
    settle_control(env)
    down = hold_dir(env, "DOWN", max_frames=400)
    save_overlay(env, "shop_from_pyr.png", f"down {glance(env)['screen_hex']} {glance(env)['xy']}")
    moved = move_to(
        env,
        Waypoint(2394, 2820, tolerance=8, label="shop_path"),
        max_frames=900,
    )
    save_overlay(env, "shop_door.png", f"path {glance(env)['xy']}")
    blob = env.em.get_state()  # type: ignore[attr-defined]
    # Path is west of the shop; cliff blocks RIGHT at y=2813. South, east, then north.
    around = [
        move_to(env, Waypoint(2400, 2936, tolerance=8, label="south"), max_frames=700),
        move_to(env, Waypoint(2504, 2936, tolerance=8, label="se"), max_frames=700),
        move_to(env, Waypoint(2504, 2820, tolerance=8, label="east_door"), max_frames=700),
    ]
    save_overlay(env, "shop_around.png", f"around {glance(env)['xy']}")
    held = hold_dir(env, "UP", max_frames=300)
    sc = settle_control(env, max_frames=360)
    g_in = glance(env)
    save_overlay(env, "shop_inside.png", f"in {g_in['room_hex']} in={g_in['indoors']} {g_in['xy']}")
    rec_tries = [
        {"label": a.reason, "ok": a.ok, "xy": [a.snapshot.link_x, a.snapshot.link_y]}
        for a in around
    ]
    entered = held
    rec: dict[str, Any] = {
        "down": down,
        "door_move": {"ok": moved.ok, "reason": moved.reason},
        "door_tries": rec_tries,
        "enter_hold": entered,
        "inside": g_in,
    }
    if g_in["indoors"] and g_in["has_control"]:
        poke_u8(env, SCREEN_ID, OW_SCREEN)
        poke_u8(env, 0x040A, OW_SCREEN)
        poke_u8(env, DARK_WORLD_FLAG, 1)
        poke_u8(env, 0xF3CA, 0x40)
        poke_u16(env, LINK_X, 744)
        poke_u16(env, LINK_Y, 640)
        out = hold_dir(env, "DOWN", max_frames=400)
        sc2 = settle_control(env, max_frames=480)
        g_out = glance(env)
        save_overlay(env, "shop_exit_40.png", f"exit {g_out['screen_hex']} {g_out['xy']} in={g_out['indoors']}")
        rec["exit"] = out
        rec["outside"] = g_out
        rec["glance"] = g_out
        if (not g_out["indoors"]) and g_out["screen"] == OW_SCREEN:
            held = hold_dir(env, "UP", max_frames=480)
            sc3 = settle_control(env, max_frames=480)
            g_d = glance(env)
            save_overlay(env, "skull_from_shop.png", f"skull {g_d['room_hex']} {g_d['xy']}")
            rec["into_skull"] = held
            rec["dungeon"] = g_d
            rec["glance"] = g_d
            saved = _pin_if_skull(env, g_d)
            rec["saved"] = saved
    else:
        rec["glance"] = g_in
    return rec


def phase_dash(env: object) -> dict[str, Any]:
    """Dash west off the pyramid plateau with boots."""
    env.em.set_state(read_state_bytes(OUT_DIR / "dw_6c_north.state"))  # type: ignore[attr-defined]
    step_frames(env, no_action(), 1)
    settle_control(env)
    start = glance(env)
    logs: list[dict[str, Any]] = []
    blob = env.em.get_state()  # type: ignore[attr-defined]
    for y, label in ((2528, "south"), (2504, "mid"), (2480, "bush")):
        env.em.set_state(blob)  # type: ignore[attr-defined]
        step_frames(env, no_action(), 1)
        move_to(env, Waypoint(start["x"], y, tolerance=6, label=label), max_frames=300)
        step_frames(env, action_for("LEFT"), 8)  # face left
        step_frames(env, action_for("A"), 4)
        step_frames(env, action_for("LEFT", "A"), 20)
        frames = 24
        prev = snapshot_env(env).link_x
        while frames < 360:
            step_frames(env, action_for("LEFT", "A"), 8)
            frames += 8
            snap = snapshot_env(env)
            if snap.screen_id != start["screen"] or snap.link_x < 1600:
                break
            if snap.link_x == prev:
                break
            prev = snap.link_x
        sc = settle_control(env, max_frames=180)
        g = glance(env)
        save_overlay(env, f"dash_{label}.png", f"dash {label} {g['screen_hex']} {g['xy']}")
        logs.append({"label": label, "frames": frames, "glance": {
            "screen_hex": g["screen_hex"], "xy": g["xy"], "indoors": g["indoors"],
        }})
    env.em.set_state(blob)  # type: ignore[attr-defined]
    step_frames(env, no_action(), 1)
    # From furthest west terrace, walk down then left.
    move_to(env, Waypoint(2408, 2504, tolerance=6, label="mid"), max_frames=300)
    hold_dir(env, "LEFT", max_frames=300)
    save_overlay(env, "dash_west_face.png", f"west {glance(env)['xy']}")
    down = hold_dir(env, "DOWN", max_frames=240)
    left = hold_dir(env, "LEFT", max_frames=400)
    g = glance(env)
    save_overlay(env, "dash_down_left.png", f"dl {g['screen_hex']} {g['xy']}")
    # Hookshot north over the moat ($0303 = hookshot).
    poke_u8(env, 0x0303, 0x03)
    step_frames(env, action_for("UP"), 8)
    step_frames(env, action_for("Y"), 12)
    step_frames(env, action_for("UP", "Y"), 40)
    sc = settle_control(env, max_frames=240)
    g2 = glance(env)
    save_overlay(env, "hook_north.png", f"hook {g2['screen_hex']} {g2['xy']}")
    along = hold_dir(env, "LEFT", max_frames=600)
    g3 = glance(env)
    save_overlay(env, "fence_left.png", f"fenceL {g3['screen_hex']} {g3['xy']}")
    west_blob = env.em.get_state()  # type: ignore[attr-defined]
    write_state_bytes(OUT_DIR / "dw_pyr_west.state", west_blob)
    probes: dict[str, Any] = {}
    for d in ("UP", "DOWN", "LEFT", "RIGHT"):
        env.em.set_state(west_blob)  # type: ignore[attr-defined]
        step_frames(env, no_action(), 1)
        rec = hold_dir(env, d, max_frames=500)
        g = glance(env)
        save_overlay(env, f"pyrwest_{d.lower()}.png", f"pw {d} {g['screen_hex']} {g['xy']}")
        probes[d] = {
            "hold": {k: rec.get(k) for k in ("stuck", "frames", "endXy", "toScreen", "toXy")},
            "glance": {"screen_hex": g["screen_hex"], "xy": g["xy"]},
        }
    env.em.set_state(west_blob)  # type: ignore[attr-defined]
    step_frames(env, no_action(), 1)
    return {
        "start": start,
        "dashes": logs,
        "down_left": {"down": down, "left": left, "glance": g},
        "hook": g2,
        "fence_left": along,
        "probes": probes,
        "glance": g3,
    }


def phase_west(env: object) -> dict[str, Any]:
    """From pyr-west pocket, slash south/west through bushes toward 0x40."""
    env.em.set_state(read_state_bytes(OUT_DIR / "dw_pyr_west.state"))  # type: ignore[attr-defined]
    step_frames(env, no_action(), 1)
    settle_control(env)
    start = glance(env)
    save_overlay(env, "west_start.png", f"west {start['xy']}")
    hold_dir(env, "DOWN", max_frames=80)
    frames = 0
    while frames < 480:
        step_frames(env, action_for("DOWN", "LEFT", "B"), 8)
        step_frames(env, action_for("LEFT", "B"), 4)
        frames += 12
        snap = snapshot_env(env)
        if snap.link_x < 1400 or snap.screen_id != start["screen"]:
            break
    sc = settle_control(env, max_frames=180)
    g = glance(env)
    save_overlay(env, "west_slash.png", f"slash {g['screen_hex']} {g['xy']}")
    blob = env.em.get_state()  # type: ignore[attr-defined]
    probes: dict[str, Any] = {}
    for d in ("UP", "DOWN", "LEFT", "RIGHT"):
        env.em.set_state(blob)  # type: ignore[attr-defined]
        step_frames(env, no_action(), 1)
        rec = hold_dir(env, d, max_frames=400)
        gg = glance(env)
        save_overlay(env, f"wslash_{d.lower()}.png", f"{d} {gg['screen_hex']} {gg['xy']}")
        probes[d] = {
            "hold": {k: rec.get(k) for k in ("stuck", "frames", "endXy", "toScreen", "toXy")},
            "glance": {"screen_hex": gg["screen_hex"], "xy": gg["xy"]},
        }
    env.em.set_state(read_state_bytes(OUT_DIR / "dw_pyr_west.state"))  # type: ignore[attr-defined]
    step_frames(env, no_action(), 1)
    step_frames(env, action_for("LEFT"), 6)
    step_frames(env, action_for("A"), 20)
    step_frames(env, action_for("LEFT", "A"), 40)
    hold_dir(env, "LEFT", max_frames=200)
    g_lift = glance(env)
    save_overlay(env, "west_lift.png", f"lift {g_lift['screen_hex']} {g_lift['xy']}")
    poke_u8(env, 0x0303, 0x03)
    step_frames(env, action_for("UP"), 8)
    step_frames(env, action_for("Y"), 16)
    step_frames(env, action_for("UP", "Y"), 40)
    sc = settle_control(env, max_frames=200)
    g_hook = glance(env)
    save_overlay(env, "west_hook.png", f"hook {g_hook['screen_hex']} {g_hook['xy']}")
    return {
        "start": start,
        "slash": g,
        "probes": probes,
        "lift": g_lift,
        "hook": g_hook,
        "glance": g_hook,
    }


def phase_spawn(env: object) -> dict[str, Any]:
    """Poke kit + $F3C8=3, module 0x08 reload; overlay says where we landed."""
    poke_info = phase_poke(env)
    ow = load_overworld(env, OW_SCREEN, 744, 640)
    g = ow["after"]
    overlay = save_overlay(
        env,
        "spawn_mod08.png",
        f"spawn {g['screen_hex']} {g['xy']} dw={int(g['dark_world'])}",
    )
    holds: dict[str, Any] = {}
    blob = env.em.get_state()  # type: ignore[attr-defined]
    for d in ("UP", "DOWN", "LEFT", "RIGHT"):
        env.em.set_state(blob)  # type: ignore[attr-defined]
        step_frames(env, no_action(), 1)
        holds[d] = hold_dir(env, d, max_frames=200)
        hg = glance(env)
        holds[d]["glance"] = {
            "screen_hex": hg["screen_hex"],
            "xy": hg["xy"],
            "module_hex": hg["module_hex"],
            "indoors": hg["indoors"],
        }
        save_overlay(env, f"spawn_hold_{d.lower()}.png", f"{d} {hg['screen_hex']} {hg['xy']}")
    env.em.set_state(blob)  # type: ignore[attr-defined]
    step_frames(env, no_action(), 1)
    return {"poke": poke_info, "ow": ow, "overlay": str(overlay), "holds": holds, "glance": g}


def phase_map(env: object) -> dict[str, Any]:
    settle_control(env)
    g0 = glance(env)
    save_overlay(env, "pin_spawn.png", f"pin {g0['room_hex']} {g0['xy']}")
    blob = env.em.get_state()  # type: ignore[attr-defined]
    holds: dict[str, Any] = {}
    for d in ("UP", "DOWN", "LEFT", "RIGHT"):
        env.em.set_state(blob)  # type: ignore[attr-defined]
        step_frames(env, no_action(), 1)
        holds[d] = hold_dir(env, d, max_frames=400)
        g = glance(env)
        holds[d]["glance"] = g
        save_overlay(env, f"map_hold_{d.lower()}.png", f"{d} {g['room_hex']} {g['xy']}")
    env.em.set_state(blob)  # type: ignore[attr-defined]
    step_frames(env, no_action(), 1)
    return {"spawn": g0, "holds": holds}


def phase_hop(env: object, direction: str, *, ax: int, ay: int) -> dict[str, Any]:
    settle_control(env)
    start = glance(env)
    save_overlay(env, "hop_start.png", f"hop start {start['xy']}")
    moved = move_to(env, Waypoint(ax, ay, tolerance=8, label="approach"), max_frames=900)
    save_overlay(env, "hop_approach.png", f"approach ({ax},{ay})")
    held = hold_dir(env, direction, max_frames=360)
    sc = settle_control(env, max_frames=480)
    end = glance(env)
    save_overlay(env, "hop_land.png", f"land {end['room_hex']} {end['xy']}")
    return {
        "start": start,
        "approach": {"want": [ax, ay], "ok": moved.ok, "reason": moved.reason},
        "hold": held,
        "settle": {"ok": sc.ok, "reason": sc.reason, "frames": sc.frames},
        "end": end,
        "leftover": leftover_from_snapshot(snapshot_env(env)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("poke", "enter", "map", "hop", "spawn", "walk", "pyramid", "shop", "dash", "west"),
        default="enter",
    )
    parser.add_argument("--state", default=SOURCE_STATE)
    parser.add_argument("--dir", default="LEFT")
    parser.add_argument("--ax", type=int, default=0)
    parser.add_argument("--ay", type=int, default=0)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = build_boot_env(args.state)
    try:
        env.reset()  # type: ignore[attr-defined]
        if args.phase == "poke":
            payload = phase_poke(env)
        elif args.phase == "enter":
            payload = phase_enter(env)
        elif args.phase == "spawn":
            payload = phase_spawn(env)
        elif args.phase == "walk":
            payload = phase_walk(env, args.dir)
        elif args.phase == "pyramid":
            payload = phase_pyramid(env)
        elif args.phase == "shop":
            payload = phase_shop(env)
        elif args.phase == "dash":
            payload = phase_dash(env)
        elif args.phase == "west":
            payload = phase_west(env)
        elif args.phase == "map":
            payload = phase_map(env)
        else:
            payload = phase_hop(env, args.dir, ax=args.ax, ay=args.ay)
        payload["utc"] = datetime.now(timezone.utc).isoformat()
        payload["phase"] = args.phase
        payload["state"] = args.state
        path = write_json(f"{args.phase}.json", payload)
        print(json.dumps({"wrote": str(path), "phase": args.phase}, indent=2))
        g = payload.get("glance") or payload.get("spawn") or payload.get("end")
        if isinstance(g, dict):
            print(
                f"room={g.get('room_hex')} mod={g.get('module_hex')} "
                f"sub={g.get('submodule')} xy={g.get('xy')} "
                f"indoors={g.get('indoors')} sword={g.get('sword')} "
                f"keys={g.get('keys')} F3CC={g.get('F3CC')}"
            )
        return 0
    finally:
        env.close()  # type: ignore[attr-defined]


if __name__ == "__main__":
    raise SystemExit(main())

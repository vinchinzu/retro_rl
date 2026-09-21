"""Eastern Palace entry pin: poke kit, walk OW door, map lobby.

Dev pin from HyruleCastleGrounds. Isolated / state-load only. No flute.
Does not poke $F3CC. Does not grant bow.

Usage:
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_ep_entry.py --census
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_ep_entry.py --pin
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_ep_entry.py --walk
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_ep_entry.py --map-lobby
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
from alttp.primitives import (
    Waypoint,
    active_sprites,
    fight_nearby,
    move_to,
    settle_control,
)
from alttp.ram import (
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

OUT_DIR = RECORDINGS_DIR / "probe_ep_entry"
SAVE_NAME = "EasternPalaceEntry"
SOURCE_STATE = "HyruleCastleGrounds"
_ACTIVE_SOURCE = SOURCE_STATE
EP_SCREEN = 0x1E
EP_ROOM_HYPOTHESIS = 0xC9
# Vanilla entrance table: 0x08 = Eastern Palace (lobby 0xC9).
EP_ENTRANCE_ID = 0x08
ENTRANCE_WORD = 0x010E

# WRAM inventory (handoff table). Do not import extra offsets from ram.py.
F340_BOW = 0xF340
F34A_LAMP = 0xF34A
F359_SWORD = 0xF359
F35A_SHIELD = 0xF35A
F3C5_PROGRESS = 0xF3C5
F3CC_FOLLOWER = FOLLOWER

SWORD_FIGHTER = 1
SHIELD_FIGHTER = 1
LAMP_ON = 1
BOW_NONE = 0

# Peek these on every glance so the residual can list loadout bytes.
LOADOUT_OFFSETS = (
    F340_BOW,
    F34A_LAMP,
    F359_SWORD,
    F35A_SHIELD,
    F3C5_PROGRESS,
    F3CC_FOLLOWER,
    NUM_KEYS,
    LINK_HP,
    LINK_MAX_HP,
)


def leftover_bytes(env: object) -> dict[str, int]:
    ram = env.get_ram()  # type: ignore[attr-defined]
    rec: dict[str, int] = {}
    for off in LOADOUT_OFFSETS:
        rec[f"{off:04X}"] = int(ram[wram_index(off)])
    rec["00EE_layer"] = int(ram[0x00EE])
    rec["0403_roomflags"] = int(ram[0x0403])
    rec["040C_dungeon"] = int(ram[0x040C])
    rec["010E_entrance"] = int(ram[0x010E]) | (int(ram[0x010F]) << 8)
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


def poke_u16(env: object, offset: int, value: int) -> dict[str, Any]:
    before = ram_u8(env, offset) | (ram_u8(env, offset + 1) << 8)
    mem = env.unwrapped.data.memory  # type: ignore[attr-defined]
    want = int(value) & 0xFFFF
    rec: dict[str, Any] = {
        "offset": f"0x{offset:04X}",
        "want": want,
        "before": before,
        "after": before,
        "ok": False,
        "via": "memory.assign u16",
    }
    candidates = [int(offset)]
    if offset >= 0x2000:
        candidates.append(0x7E0000 + int(offset))
    last_err = None
    for addr in candidates:
        try:
            mem.assign(addr, "<u2", want)
        except (IndexError, KeyError, ValueError) as exc:
            last_err = repr(exc)
            rec["via"] = f"memory.assign u16 fail {addr:#x}: {last_err}"
            continue
        after = ram_u8(env, offset) | (ram_u8(env, offset + 1) << 8)
        rec["after"] = after
        rec["ok"] = after == want
        rec["via"] = f"memory.assign u16 {addr:#x}"
        if rec["ok"]:
            return rec
    rec["error"] = last_err
    return rec


def poke_u8(env: object, offset: int, value: int) -> dict[str, Any]:
    """Write WRAM $F3xx via memory.assign; confirm with get_ram() index.

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


def ensure_ep_kit(env: object, *, progress: int | None = 1) -> list[dict[str, Any]]:
    """Poke missing lamp/sword/shield. Lift rain lock with $F3C5=1 if needed.

    Never grant bow. Never write $F3CC. $F3C5=1 is uncle/sword progress so the
    opening rain soldiers let Link leave screen 0x2C (not Zelda-rescued).
    """
    wanted = [
        (F359_SWORD, SWORD_FIGHTER, "sword"),
        (F35A_SHIELD, SHIELD_FIGHTER, "shield"),
        (F34A_LAMP, LAMP_ON, "lamp"),
    ]
    if progress is not None:
        wanted.append((F3C5_PROGRESS, int(progress), "progress"))
    pokes: list[dict[str, Any]] = []
    bow = ram_u8(env, F340_BOW)
    if bow != BOW_NONE:
        raise RuntimeError(f"bow already granted $F340={bow}; refuse to pin")
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
            raise RuntimeError(f"poke {name} $ {offset:04X} failed: {rec}")
        pokes.append(rec)
    bow_after = ram_u8(env, F340_BOW)
    if bow_after != BOW_NONE:
        raise RuntimeError(f"bow changed during kit poke $F340={bow_after}")
    return pokes


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_source(env: object | None = None, *, state: str | None = None) -> object:
    name = state or _ACTIVE_SOURCE
    if env is None:
        env = build_boot_env(name, render_mode="rgb_array")
        env.reset()  # type: ignore[attr-defined]
    else:
        if not resync_custom_state(env, GAME_DIR, INTEGRATION, name):
            raise RuntimeError(f"failed to load {name}")
    settle_control(env)
    return env


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
    """Load grounds, poke kit, save a mid-walk checkpoint glance (no indoor yet)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = load_source()
    try:
        before = glance(env)
        save_png(env, OUT_DIR / "grounds_before_poke.png", title="grounds before poke")
        pokes = ensure_ep_kit(env)
        step_frames(env, no_action(), 2)
        after = glance(env)
        save_png(env, OUT_DIR / "grounds_after_poke.png", title="grounds after poke")
        write_json(
            OUT_DIR / "pokes.json",
            {"when": utc_now(), "source": SOURCE_STATE, "pokes": pokes, "before": before, "after": after},
        )
        print(json.dumps({"pokes": pokes, "after": after}, indent=2))
    finally:
        env.close()
    return 0


def _event(frames: int, snap: Any, *, note: str = "") -> dict[str, Any]:
    rec = {
        "f": frames,
        "screen": f"0x{snap.screen_id:02X}",
        "room": f"0x{snap.room_base_id:02X}",
        "in": bool(snap.indoors),
        "mod": int(snap.game_mode),
        "sub": int(snap.submodule),
        "xy": [snap.link_x, snap.link_y],
        "ctrl": bool(snap.has_control),
    }
    if note:
        rec["note"] = note
    return rec


def hold(
    env: object,
    buttons: tuple[str, ...],
    *,
    max_frames: int,
    stop_indoors: bool = False,
    stop_room_change: bool = False,
    stuck_frames: int = 90,
) -> dict[str, Any]:
    start = snapshot_env(env)
    frames = 0
    events: list[dict[str, Any]] = [_event(0, start, note="start")]
    action = no_action() if not buttons or buttons == ("NONE",) else action_for(*buttons)
    prev_key = (start.screen_id, start.room_base_id, start.indoors, start.game_mode, start.submodule)
    prev_xy = (start.link_x, start.link_y)
    idle = 0
    started_indoors = bool(start.indoors)
    start_room = int(start.room_base_id)
    while frames < max_frames:
        snap = snapshot_env(env)
        if snap.game_mode == 0x0E:
            button = "A" if (frames // 4) % 2 == 0 else "B"
            step_frames(env, action_for(button), 2)
            step_frames(env, no_action(), 2)
            frames += 4
        else:
            step_frames(env, action, 4)
            frames += 4
        snap = snapshot_env(env)
        key = (snap.screen_id, snap.room_base_id, snap.indoors, snap.game_mode, snap.submodule)
        xy = (snap.link_x, snap.link_y)
        if key != prev_key:
            events.append(_event(frames, snap, note="transition"))
            prev_key = key
            idle = 0
        if xy != prev_xy:
            idle = 0
            prev_xy = xy
        elif snap.has_control and snap.game_mode in (0x07, 0x09) and snap.submodule == 0:
            idle += 4
            if idle >= stuck_frames:
                events.append(_event(frames, snap, note="stuck"))
                break
        else:
            idle = 0
        if snap.game_mode == 0x12:
            events.append(_event(frames, snap, note="death"))
            break
        if (
            stop_indoors
            and (not started_indoors)
            and snap.indoors
            and snap.has_control
            and snap.submodule == 0
        ):
            events.append(_event(frames, snap, note="indoors"))
            break
        if (
            stop_room_change
            and snap.has_control
            and snap.submodule == 0
            and (int(snap.room_base_id) != start_room or bool(snap.indoors) != started_indoors)
        ):
            events.append(_event(frames, snap, note="room_change"))
            break
    return {"frames": frames, "events": events, "end": glance(env)}


def parse_script(text: str) -> list[tuple[str, int]]:
    steps: list[tuple[str, int]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"bad script step {part!r}; want DIR:frames")
        direction, raw = part.split(":", 1)
        steps.append((direction.upper(), int(raw)))
    return steps


def _maybe_save_indoor(env: object, end: dict[str, Any]) -> str | None:
    if not (end.get("indoors") and end.get("has_control") and int(end.get("submodule") or 0) == 0):
        return None
    path = INTEGRATION_DIR / f"{SAVE_NAME}.state"
    write_state_bytes(path, env.em.get_state())  # type: ignore[attr-defined]
    return str(path)


def cmd_walk(*, hold_dir: str, frames: int, save_if_indoor: bool) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = load_source()
    try:
        pokes = ensure_ep_kit(env)
        step_frames(env, no_action(), 2)
        start = glance(env)
        save_png(env, OUT_DIR / "walk_start.png", title="walk start")
        result = hold(
            env,
            (hold_dir,),
            max_frames=frames,
            stop_indoors=True,
            stop_room_change=True,
        )
        end = result["end"]
        save_png(env, OUT_DIR / f"walk_{hold_dir.lower()}.png", title=f"hold {hold_dir}")
        saved = _maybe_save_indoor(env, end) if save_if_indoor else None
        payload = {
            "when": utc_now(),
            "source": SOURCE_STATE,
            "pokes": pokes,
            "hold": hold_dir,
            "start": start,
            "result": result,
            "saved": saved,
        }
        write_json(OUT_DIR / f"walk_{hold_dir.lower()}.json", payload)
        print(json.dumps(payload, indent=2))
    finally:
        env.close()
    return 0


def cmd_script(*, script: str, save_if_indoor: bool) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    steps = parse_script(script)
    env = load_source()
    try:
        pokes = ensure_ep_kit(env)
        step_frames(env, no_action(), 2)
        start = glance(env)
        save_png(env, OUT_DIR / "script_start.png", title="script start")
        legs: list[dict[str, Any]] = []
        start_indoors = bool(start.get("indoors"))
        start_room = int(start.get("room") or 0)
        left = False
        for i, (direction, frames) in enumerate(steps):
            result = hold(
                env,
                (direction,),
                max_frames=frames,
                stop_indoors=not start_indoors,
                stop_room_change=True,
                stuck_frames=360 if i == len(steps) - 1 else 90,
            )
            shot = OUT_DIR / f"script_{i:02d}_{direction.lower()}.png"
            save_png(env, shot, title=f"{i} {direction}")
            legs.append({"dir": direction, "want_frames": frames, "shot": str(shot), **result})
            end_leg = result["end"]
            if int(end_leg.get("room") or 0) != start_room or bool(end_leg.get("indoors")) != start_indoors:
                left = True
                break
        end = glance(env)
        saved = _maybe_save_indoor(env, end) if save_if_indoor and left else None
        payload = {
            "when": utc_now(),
            "source": SOURCE_STATE,
            "pokes": pokes,
            "script": script,
            "start": start,
            "legs": legs,
            "end": end,
            "saved": saved,
        }
        write_json(OUT_DIR / "script.json", payload)
        print(json.dumps(payload, indent=2))
    finally:
        env.close()
    return 0


def cmd_path(spec: str) -> int:
    """Follow x,y;x,y waypoints from the loaded pin. Fight if hostiles block."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wps: list[tuple[int, int]] = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        xs, ys = part.split(",", 1)
        wps.append((int(xs), int(ys)))
    env = load_source()
    try:
        start = glance(env)
        room = int(start.get("room") or 0)
        save_png(env, OUT_DIR / "path_start.png", title="path start")
        legs: list[dict[str, Any]] = []
        for i, (x, y) in enumerate(wps):
            fight_nearby(env, room=room, max_distance=64, max_cycles=40)
            move = move_to(
                env,
                Waypoint(x, y, tolerance=8, room=room, label=f"p{i}"),
                max_frames=700,
            )
            g = glance(env)
            save_png(env, OUT_DIR / f"path_{i:02d}.png", title=f"p{i} ({x},{y})")
            legs.append(
                {
                    "want": [x, y],
                    "ok": bool(move.ok),
                    "reason": move.reason,
                    "frames": int(move.frames),
                    "xy": g.get("xy"),
                    "room_hex": g.get("room_hex"),
                    "indoors": g.get("indoors"),
                    "module": g.get("module"),
                    "submodule": g.get("submodule"),
                }
            )
            if int(g.get("room") or 0) != room or not g.get("indoors"):
                break
            if not move.ok:
                # Keep going; later points may still be useful.
                pass
        end = glance(env)
        payload = {"when": utc_now(), "start": start, "legs": legs, "end": end}
        write_json(OUT_DIR / "path.json", payload)
        print(json.dumps(payload, indent=2))
    finally:
        env.close()
    return 0


def cmd_goto(*, x: int, y: int, max_frames: int) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = load_source()
    try:
        pokes = ensure_ep_kit(env)
        step_frames(env, no_action(), 2)
        start = glance(env)
        save_png(env, OUT_DIR / "goto_start.png", title="goto start")
        fight_nearby(env, room=int(start.get("room") or 0), max_distance=80, max_cycles=40)
        move = move_to(env, Waypoint(x, y, tolerance=8, label="goto"), max_frames=max_frames)
        settle_control(env)
        end = glance(env)
        save_png(env, OUT_DIR / "goto_end.png", title=f"goto ({x},{y})")
        payload = {
            "when": utc_now(),
            "pokes": pokes,
            "start": start,
            "move_ok": bool(move.ok),
            "move_reason": move.reason,
            "move_frames": int(move.frames),
            "end": end,
        }
        write_json(OUT_DIR / "goto.json", payload)
        print(json.dumps(payload, indent=2))
    finally:
        env.close()
    return 0


def trigger_entrance(env: object, entrance_id: int) -> dict[str, Any]:
    """Module 0x06 PreDungeon + $010E. Real indoor room-load, not a freeze."""
    before = glance(env)
    word = poke_u16(env, ENTRANCE_WORD, entrance_id)
    mod = poke_u8(env, MODULE, 0x06)
    sub = poke_u8(env, SUBMODULE, 0x00)
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
        "pokes": [word, mod, sub],
        "before": before,
        "frames": frames + sc.frames,
        "settle": {"ok": sc.ok, "reason": sc.reason, "frames": sc.frames},
        "after": glance(env),
    }


def cmd_enter() -> int:
    """Poke kit on grounds, then entrance 0x08 (real EP room-load). Save if lobby."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = load_source()
    try:
        pokes = ensure_ep_kit(env)
        step_frames(env, no_action(), 2)
        save_png(env, OUT_DIR / "enter_before.png", title="before PreDungeon 0x08")
        result = trigger_entrance(env, EP_ENTRANCE_ID)
        after = result["after"]
        save_png(
            env,
            OUT_DIR / "enter_after.png",
            title=f"PreDungeon 0x08 {after.get('room_hex')} {after.get('xy')}",
        )
        indoor_ok = bool(
            after.get("indoors")
            and after.get("has_control")
            and int(after.get("module") or 0) == 0x07
        )
        saved = None
        if indoor_ok:
            path = INTEGRATION_DIR / f"{SAVE_NAME}.state"
            write_state_bytes(path, env.em.get_state())  # type: ignore[attr-defined]
            saved = str(path)
        payload = {
            "when": utc_now(),
            "source": _ACTIVE_SOURCE,
            "kit_pokes": pokes,
            "entrance": result,
            "indoor_ok": indoor_ok,
            "room_hex": after.get("room_hex"),
            "saved": saved,
        }
        write_json(OUT_DIR / "enter.json", payload)
        print(json.dumps(payload, indent=2))
    finally:
        env.close()
    return 0


def cmd_save() -> int:
    env = load_source()
    try:
        ensure_ep_kit(env)
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


def cmd_map_lobby() -> int:
    """Load the saved pin (or source) and dump lobby overlay + sprite census."""
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


def main() -> int:
    p = argparse.ArgumentParser(description="Eastern Palace entry pin probe")
    p.add_argument("--census", action="store_true")
    p.add_argument("--pin", action="store_true", help="poke kit on HyruleCastleGrounds")
    p.add_argument("--walk", action="store_true")
    p.add_argument("--dir", default="RIGHT", help="hold direction for --walk")
    p.add_argument("--frames", type=int, default=1800)
    p.add_argument("--save-if-indoor", action="store_true")
    p.add_argument("--script", default="", help="comma DIR:frames steps, e.g. DOWN:400,RIGHT:2000")
    p.add_argument("--goto", nargs=2, type=int, metavar=("X", "Y"))
    p.add_argument(
        "--path",
        default="",
        help="semicolon waypoints x,y;x,y from the loaded pin (no kit poke)",
    )
    p.add_argument("--save", action="store_true")
    p.add_argument("--enter", action="store_true", help="module 0x0F entrance 0x08 room-load")
    p.add_argument("--map-lobby", action="store_true")
    p.add_argument("--from", dest="from_state", default=SOURCE_STATE)
    args = p.parse_args()
    global _ACTIVE_SOURCE
    _ACTIVE_SOURCE = args.from_state
    if args.census:
        return cmd_census()
    if args.pin:
        return cmd_pin()
    if args.script:
        return cmd_script(script=args.script, save_if_indoor=args.save_if_indoor)
    if args.walk:
        return cmd_walk(hold_dir=args.dir.upper(), frames=args.frames, save_if_indoor=args.save_if_indoor)
    if args.path:
        return cmd_path(args.path)
    if args.goto is not None:
        return cmd_goto(x=args.goto[0], y=args.goto[1], max_frames=args.frames)
    if args.enter:
        return cmd_enter()
    if args.save:
        return cmd_save()
    if args.map_lobby:
        return cmd_map_lobby()
    return cmd_census()


if __name__ == "__main__":
    raise SystemExit(main())

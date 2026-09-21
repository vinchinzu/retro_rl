"""Tower of Hera entry pin. Isolated / state-load. Not continuous.

Load FighterSword, poke Hera kit (no moon pearl), real dungeon load then
Death Mountain door, save HeraEntry.state, hop west stairs to 0x87.

Usage:
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_hera_entry.py --pin
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_hera_entry.py --reenter
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_hera_entry.py --hop
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
from alttp.primitives import Waypoint, active_sprites, move_to, settle_control, spin_attack
from alttp.ram import snapshot_to_diag, wram_index
from alttp.room_sense import overlay_from_env
from alttp.screen_glance import leftover_from_snapshot
from alttp.startup import action_for, build_boot_env, no_action, snapshot_env, step_frames
from retro_harness.env import save_state

OUT_DIR = RECORDINGS_DIR / "probe_hera_entry"
STATE_NAME = "HeraEntry"
HERA_ROOM = 0x77
HERA_1F = 0x87
HERA_ENTRANCE_ID = 0x33
SNES_WRAM_BANK = 0x7E0000

POKES: tuple[tuple[int, int, str], ...] = (
    (0xF340, 2, "bow+arrows"),
    (0xF377, 30, "arrows"),
    (0xF34A, 1, "lamp"),
    (0xF34E, 1, "book"),
    (0xF354, 1, "power_gloves"),
    (0xF355, 1, "boots"),
    (0xF357, 0, "moon_pearl_absent"),
    (0xF359, 1, "fighter_sword"),
    (0xF36C, 0x18, "max_hp_3_hearts_min"),
    (0xF36D, 0x18, "hp_full_3"),
)

LOADOUT_ADDRS: tuple[tuple[str, int], ...] = (
    ("F340_bow", 0xF340),
    ("F34A_lamp", 0xF34A),
    ("F34E_book", 0xF34E),
    ("F354_gloves", 0xF354),
    ("F355_boots", 0xF355),
    ("F357_pearl", 0xF357),
    ("F359_sword", 0xF359),
    ("F35A_shield", 0xF35A),
    ("F36C_max_hp", 0xF36C),
    ("F36D_hp", 0xF36D),
    ("F36F_keys", 0xF36F),
    ("F377_arrows", 0xF377),
    ("F3C5_progress", 0xF3C5),
    ("F3CC_follower", 0xF3CC),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ram_u8(env: object, offset: int) -> int:
    ram = env.get_ram()  # type: ignore[attr-defined]
    return int(ram[wram_index(offset)])


def ram_u16(env: object, offset: int) -> int:
    ram = env.get_ram()  # type: ignore[attr-defined]
    return int(ram[wram_index(offset)]) | (int(ram[wram_index(offset) + 1]) << 8)


def loadout_bytes(env: object) -> dict[str, int]:
    out = {name: ram_u8(env, addr) for name, addr in LOADOUT_ADDRS}
    out["A0_room"] = ram_u16(env, 0x00A0) & 0xFFFF
    out["010E_entrance"] = ram_u16(env, 0x010E)
    out["008A_screen"] = ram_u8(env, 0x008A)
    out["00EE_layer"] = ram_u8(env, 0x00EE)
    return out


def glance(env: object) -> dict[str, Any]:
    snap = snapshot_env(env)
    rec = leftover_from_snapshot(snap)
    rec.update(
        {
            "module_hex": f"0x{int(snap.game_mode):02X}",
            "room_hex": f"0x{int(snap.room_base_id):02X}",
            "screen_hex": f"0x{int(snap.screen_id):02X}",
            "link_x": int(snap.link_x),
            "link_y": int(snap.link_y),
            "has_control": bool(snap.has_control),
            "lamp": int(snap.lamp_level),
            "loadout": loadout_bytes(env),
            "diag": snapshot_to_diag(snap),
            "sprites": [
                {"type": f"0x{s.sprite_type:02X}", "xy": [s.x, s.y], "hp": s.hp}
                for s in active_sprites(env)[:12]
            ],
        }
    )
    return rec


def save_png(env: object, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = overlay_from_env(env, include_all_sprites=True, title=title)
    Image.fromarray(img).save(path)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def try_assign(env: object, offset: int, value: int) -> str:
    """Poke one WRAM byte. Handoff offset 0xF3xx first, then bank $7E."""
    mem = env.unwrapped.data.memory
    value = int(value) & 0xFF
    last = "no_assign"
    for name, mapped in (
        ("wram_offset", int(offset)),
        ("bank7e", SNES_WRAM_BANK + int(offset)),
    ):
        try:
            mem.assign(mapped, "|u1", value)
            got = ram_u8(env, offset)
            if got == value:
                return name
            last = f"{name}_mismatch_got_{got:#04x}"
        except Exception as exc:  # noqa: BLE001 — probe documents poke path
            last = f"{name}_fail={exc!r}"
    return last


def apply_pokes(env: object) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, want, label in POKES:
        before = ram_u8(env, offset)
        if before == want:
            method, after = "already", before
        else:
            method = try_assign(env, offset, want)
            after = ram_u8(env, offset)
        rows.append(
            {
                "offset": f"0x{offset:04X}",
                "label": label,
                "before": before,
                "want": want,
                "after": after,
                "method": method,
                "ok": after == want,
            }
        )
    return rows


def poke_u8(env: object, offset: int, value: int) -> str:
    return try_assign(env, offset, int(value) & 0xFF)


def poke_u16(env: object, offset: int, value: int) -> None:
    poke_u8(env, offset, int(value) & 0xFF)
    poke_u8(env, offset + 1, (int(value) >> 8) & 0xFF)


def wait_control(env: object, *, max_frames: int = 720) -> Any:
    return settle_control(env, max_frames=max_frames)


def hold(
    env: object,
    *buttons: str,
    frames: int,
    stop_indoors: bool | None = None,
) -> dict[str, Any]:
    start = snapshot_env(env)
    used = 0
    action = no_action() if not buttons else action_for(*buttons)
    for _ in range(max(0, frames)):
        env.step(action)  # type: ignore[attr-defined]
        used += 1
        snap = snapshot_env(env)
        if stop_indoors is not None and bool(snap.indoors) == stop_indoors:
            break
        if snap.game_mode == 0x12:
            break
    settle = wait_control(env, max_frames=480)
    end = snapshot_env(env)
    return {
        "buttons": list(buttons),
        "used": used,
        "start": leftover_from_snapshot(start),
        "end": leftover_from_snapshot(end),
        "settle_ok": bool(settle.ok),
        "settle_frames": int(settle.frames),
    }


def slash(env: object, *, taps: int = 4) -> None:
    for _ in range(taps):
        step_frames(env, action_for("B"), 8)
        step_frames(env, no_action(), 10)


def trigger_dungeon_entrance(env: object, entrance: int) -> dict[str, Any]:
    poke_u16(env, 0x010E, entrance)
    poke_u8(env, 0x0010, 0x06)
    poke_u8(env, 0x0011, 0x00)
    step_frames(env, no_action(), 8)
    settle = wait_control(env, max_frames=900)
    after = glance(env)
    room = int(after.get("room") or 0) & 0xFF
    return {
        "entrance": entrance,
        "after": after,
        "settle_ok": bool(settle.ok),
        "ok": bool(
            after.get("indoors")
            and after.get("has_control")
            and int(after.get("module", -1)) == 0x07
            and room == HERA_ROOM
        ),
    }


def cmd_pin(source: str) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = build_boot_env(source, render_mode="rgb_array")
    try:
        env.reset()  # type: ignore[attr-defined]
        wait_control(env)
        pre = glance(env)
        pokes = apply_pokes(env)
        step_frames(env, no_action(), 2)
        dungeon = trigger_dungeon_entrance(env, HERA_ENTRANCE_ID)
        after = dungeon["after"]
        save_png(env, OUT_DIR / "pin_after_load.png", "module 0x06 entrance")
        ok = bool(dungeon.get("ok"))
        saved = None
        if ok:
            wait_control(env, max_frames=480)
            saved = str(save_state(env, GAME_DIR, INTEGRATION, STATE_NAME))
            save_png(env, OUT_DIR / "HeraEntry_module06.png", "module06 pin")
        payload = {
            "when": utc_now(),
            "source": source,
            "label": "isolated/state-load",
            "pre": pre,
            "pokes": pokes,
            "pokes_ok": all(r["ok"] for r in pokes),
            "dungeon_load": dungeon,
            "ok": ok,
            "saved": saved,
            "glance": after,
        }
        dump_json(OUT_DIR / "pin.json", payload)
        print(
            f"pin ok={ok} room={after.get('room_hex')} "
            f"xy=({after.get('x')},{after.get('y')}) saved={saved}"
        )
        return payload
    finally:
        env.close()  # type: ignore[attr-defined]


def cmd_reenter() -> dict[str, Any]:
    """DOWN to Death Mountain, UP through the real Hera door, re-save."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = build_boot_env(STATE_NAME, render_mode="rgb_array")
    try:
        env.reset()  # type: ignore[attr-defined]
        wait_control(env)
        indoor = glance(env)
        down = hold(env, "DOWN", frames=360, stop_indoors=False)
        wait_control(env, max_frames=480)
        mountain = glance(env)
        save_png(env, OUT_DIR / "re_mountain.png", "Death Mountain Hera door")
        hold(env, "UP", frames=360, stop_indoors=True)
        wait_control(env, max_frames=720)
        back = glance(env)
        save_png(env, OUT_DIR / "HeraEntry.png", "HeraEntry real door")
        ok = bool(
            back.get("indoors")
            and (int(back.get("room") or 0) & 0xFF) == HERA_ROOM
            and int(back.get("module") or 0) == 0x07
            and int(back.get("submodule", 0)) == 0
            and bool(back.get("has_control"))
        )
        saved = None
        if ok:
            saved = str(save_state(env, GAME_DIR, INTEGRATION, STATE_NAME))
        payload = {
            "when": utc_now(),
            "indoor": indoor,
            "down": down,
            "mountain": mountain,
            "back": back,
            "ok": ok,
            "saved": saved,
        }
        dump_json(OUT_DIR / "reenter.json", payload)
        print(
            f"reenter ok={ok} room={back.get('room_hex')} "
            f"xy=({back.get('x')},{back.get('y')}) saved={saved}"
        )
        return payload
    finally:
        env.close()  # type: ignore[attr-defined]


def cmd_hop() -> dict[str, Any]:
    """Slash crystal, walk west stairs, settle in 0x87."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = build_boot_env(STATE_NAME, render_mode="rgb_array")
    try:
        env.reset()  # type: ignore[attr-defined]
        wait_control(env)
        start = glance(env)
        save_png(env, OUT_DIR / "hop_start.png", "hop start")
        move_to(
            env,
            Waypoint(3840, 3968, tolerance=16, room=HERA_ROOM, label="crystal_ledge"),
            max_frames=400,
        )
        slash(env, taps=4)
        spin_attack(env, charge_frames=80)
        wait_control(env, max_frames=120)
        save_png(env, OUT_DIR / "hop_crystal.png", "after crystal")
        for label, x, y in (
            ("west_alcove", 3693, 3968),
            ("west_stair_lip", 3696, 3928),
            ("west_stair_approach", 3728, 3904),
        ):
            moved = move_to(
                env,
                Waypoint(x, y, tolerance=12, room=HERA_ROOM, label=label),
                max_frames=600,
            )
            print(
                f"{label} ok={moved.ok} "
                f"xy=({moved.snapshot.link_x},{moved.snapshot.link_y}) "
                f"room=0x{moved.snapshot.room_base_id:02X}"
            )
            if moved.snapshot.room_base_id != HERA_ROOM:
                break
        save_png(env, OUT_DIR / "hop_approach.png", "stair approach")
        before = snapshot_env(env)
        if before.room_base_id == HERA_ROOM and before.indoors:
            hold(env, "UP", frames=240)
        settle = settle_control(env, max_frames=480)
        leftover = glance(env)
        save_png(env, OUT_DIR / "hop_0x87.png", "0x87 leftover")
        ok = bool(
            leftover.get("indoors")
            and (int(leftover.get("room") or 0) & 0xFF) == HERA_1F
            and int(leftover.get("module") or 0) == 0x07
            and int(leftover.get("submodule", 0)) == 0
            and bool(leftover.get("has_control"))
        )
        payload = {
            "when": utc_now(),
            "ok": ok,
            "start": start,
            "settle_ok": bool(settle.ok),
            "settle_frames": int(settle.frames),
            "leftover": leftover,
            "verification": "isolated",
        }
        dump_json(OUT_DIR / "hop_west_down_to_0x87.json", payload)
        print(
            f"hop ok={ok} room={leftover.get('room_hex')} "
            f"xy=({leftover.get('x')},{leftover.get('y')}) "
            f"mod={leftover.get('module_hex')}/{leftover.get('submodule')}"
        )
        return payload
    finally:
        env.close()  # type: ignore[attr-defined]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="FighterSword")
    p.add_argument("--pin", action="store_true")
    p.add_argument("--reenter", action="store_true")
    p.add_argument("--hop", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.pin:
        cmd_pin(args.source)
        return 0
    if args.reenter:
        cmd_reenter()
        return 0
    if args.hop:
        cmd_hop()
        return 0
    cmd_pin(args.source)
    cmd_reenter()
    cmd_hop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

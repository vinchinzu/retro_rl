"""Walk CastleB1SecondKey out of the 0x71 east pocket with keys>=1.

Sibling cardinals boxed x=832-944 y=3976-4008; LEFT at 832 is a wall, not a
key door. This sitting tries wraps (north lip then west, south, lift, dash).
Does not poke keys. Halt a claimed wrap at the first miss; restore origin
between independent wraps.

Usage:
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_second_key.py
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
from alttp.primitives import (
    SPRITE_SMALL_KEY,
    Waypoint,
    active_sprites,
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
from alttp.room_map import load_room_map
from alttp.room_sense import overlay_from_env
from alttp.screen_glance import leftover_from_snapshot
from alttp.startup import action_for, build_boot_env, no_action, snapshot_env, step_frames
from retro_harness.env import resync_custom_state

OUT_DIR = RECORDINGS_DIR / "probe_second_key"
PIN = "CastleB1SecondKey"
ROOM_71 = 0x71
ROOM_81 = 0x81
# maps/room_71.json
SOUTH_APPROACH = (632, 4100)
EAST_EXTENT_X = 688
POCKET = (832, 3976, 944, 4008)
OPEN_X = 688
PIN_XY = (904, 3988)

BOMBS = 0xF343
BOOTS = 0xF355
BOOMERANG = 0xF341
CARRY = 0x0308
LAYER = 0x00EE
DOOR0 = 0x0400
DOOR1 = 0x0401
TAG1 = 0x00AE
Y_ITEM = 0x0202
LINK_DIR = 0x2F
LINK_ACT = 0x5D


def leftover_bytes(env: object) -> dict[str, int]:
    ram = env.get_ram()  # type: ignore[attr-defined]
    return {
        "F359_sword": int(ram[wram_index(EQUIP_SWORD)]),
        "F34A_lamp": int(ram[wram_index(LINK_ITEM_LAMP)]),
        "F36F_keys": int(ram[wram_index(NUM_KEYS)]),
        "F3CC_follower": int(ram[wram_index(FOLLOWER)]),
        "F36D_hp": int(ram[wram_index(LINK_HP)]),
        "F36C_max_hp": int(ram[wram_index(LINK_MAX_HP)]),
        "F343_bombs": int(ram[wram_index(BOMBS)]),
        "F355_boots": int(ram[wram_index(BOOTS)]),
        "F341_boomerang": int(ram[wram_index(BOOMERANG)]),
        "0308_carry": int(ram[CARRY]),
        "00EE_layer": int(ram[LAYER]),
        "0400_doors": int(ram[DOOR0]),
        "0401_doors": int(ram[DOOR1]),
        "00AE_tag1": int(ram[TAG1]),
        "0202_yitem": int(ram[Y_ITEM]),
        "2F_dir": int(ram[LINK_DIR]),
        "5D_act": int(ram[LINK_ACT]),
    }


def glance(env: object) -> dict[str, Any]:
    snap = snapshot_env(env)
    sprites = [
        {"type": f"0x{s.sprite_type:02X}", "xy": [s.x, s.y], "hp": s.hp, "state": s.state}
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
        "boot": leftover_from_snapshot(snap),
        "sprites": sprites[:16],
        "keys_on_floor": [
            {"xy": [s.x, s.y]} for s in sprites_of_type(env, (SPRITE_SMALL_KEY,))
        ],
        "in_pocket": in_pocket(int(snap.room_base_id), int(snap.link_x), int(snap.link_y)),
        "opened": opened(int(snap.room_base_id), int(snap.link_x), int(snap.num_keys)),
    }


def in_pocket(room: int, x: int, y: int) -> bool:
    x0, y0, x1, y1 = POCKET
    return int(room) == ROOM_71 and x0 <= int(x) <= x1 and y0 <= int(y) <= y1


def opened(room: int, x: int, keys: int) -> bool:
    if int(keys) < 1:
        return False
    if int(room) == ROOM_81:
        return True
    return int(room) == ROOM_71 and int(x) <= OPEN_X


def save_overlay(env: object, path: Path, title: str) -> None:
    m = load_room_map("room_71")
    img = overlay_from_env(
        env, include_all_sprites=True, points=m.points, title=title
    )
    Image.fromarray(img).save(path)


def load_pin(env: object, name: str = PIN) -> None:
    if not resync_custom_state(env, GAME_DIR, INTEGRATION, name):
        raise RuntimeError(f"failed to load {name}")


def pin_state(env: object) -> bytes:
    return env.em.get_state()  # type: ignore[attr-defined]


def restore(env: object, blob: bytes) -> None:
    env.em.set_state(blob)  # type: ignore[attr-defined]
    settle_control(env)


def walk(env: object, x: int, y: int, *, room: int | None = ROOM_71, frames: int = 360) -> Any:
    return move_to(env, Waypoint(x, y, tolerance=6, room=room), max_frames=frames)


def hold(
    env: object,
    buttons: tuple[str, ...],
    *,
    max_frames: int = 280,
    step: int = 4,
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
                }
            )
            if snap.game_mode == 0x12:
                break
            if opened(snap.room_base_id, snap.link_x, snap.num_keys) and snap.has_control:
                break
            if snap.room_base_id != start.room_base_id and snap.has_control:
                break
        elif snap.has_control:
            idle += 1
            if idle >= 12:
                break
        prev = cur
    settle_control(env, max_frames=180)
    return {
        "buttons": list(buttons),
        "frames": frames,
        "events": events[-20:],
        "end": glance(env),
    }


def brief(g: dict[str, Any]) -> str:
    return (
        f"{g['room_hex']} ({g['link_x']},{g['link_y']}) keys={g['keys']} "
        f"sub={g['submodule']} pocket={g['in_pocket']} opened={g['opened']}"
    )


def try_south_door(env: object) -> dict[str, Any]:
    here = snapshot_env(env)
    if here.room_base_id != ROOM_71 or here.num_keys < 1:
        return {"ok": False, "reason": "not in 0x71 with keys", "end": glance(env)}
    r1 = walk(env, SOUTH_APPROACH[0], here.link_y, frames=500)
    r2 = walk(env, SOUTH_APPROACH[0], SOUTH_APPROACH[1], frames=500)
    pose = glance(env)
    push = hold(env, ("DOWN",), max_frames=240)
    end = push["end"]
    return {
        "ok": end["room_base_id"] == ROOM_81 and end["keys"] >= 1,
        "walk": [r1.reason, r2.reason],
        "approach": pose,
        "push": push,
        "end": end,
    }


def one_wrap(
    env: object,
    origin: bytes,
    *,
    label: str,
    claim: str,
    corner: tuple[int, int] | None,
    buttons: tuple[str, ...],
    max_frames: int = 280,
) -> dict[str, Any]:
    restore(env, origin)
    rec: dict[str, Any] = {"label": label, "claim": claim, "corner": list(corner) if corner else None}
    if corner is not None:
        r = walk(env, corner[0], corner[1])
        rec["corner_walk"] = {
            "ok": r.ok,
            "reason": r.reason,
            "pose": glance(env),
        }
        print(f"  {label} corner {corner} → {brief(rec['corner_walk']['pose'])} {r.reason}")
    h = hold(env, buttons, max_frames=max_frames)
    rec["hold"] = h
    rec["end"] = h["end"]
    rec["miss"] = not h["end"]["opened"]
    rec["still_pocket"] = h["end"]["in_pocket"]
    print(f"  {label} {buttons} → {brief(h['end'])} claim={claim!r} miss={rec['miss']}")
    if h["end"]["opened"]:
        rec["south"] = try_south_door(env)
        print(f"    south → {brief(rec['south']['end'])} ok={rec['south']['ok']}")
    return rec


def lift_then_hold(
    env: object,
    origin: bytes,
    *,
    label: str,
    claim: str,
    corner: tuple[int, int],
    buttons: tuple[str, ...],
) -> dict[str, Any]:
    restore(env, origin)
    r = walk(env, corner[0], corner[1])
    before = glance(env)
    step_frames(env, action_for("A"), 12)
    settle_control(env, max_frames=60)
    lifted = glance(env)
    h = hold(env, buttons, max_frames=240)
    rec = {
        "label": label,
        "claim": claim,
        "corner": list(corner),
        "corner_walk": {"ok": r.ok, "reason": r.reason, "pose": before},
        "after_a": lifted,
        "hold": h,
        "end": h["end"],
        "miss": not h["end"]["opened"],
        "still_pocket": h["end"]["in_pocket"],
        "carry_before": before["leftover"]["0308_carry"],
        "carry_after": lifted["leftover"]["0308_carry"],
    }
    print(
        f"  {label} A@{corner} carry {rec['carry_before']}→{rec['carry_after']} "
        f"then {buttons} → {brief(h['end'])} miss={rec['miss']}"
    )
    return rec


def wrap_plan() -> list[dict[str, Any]]:
    """Independent wraps from origin. Not the four cardinals from pin xy."""
    x0, y0, x1, y1 = POCKET
    return [
        {
            "label": "nw_up",
            "claim": "NW lip (832,3976) hold UP → y<3976 keys>=1 (north toward spiral)",
            "corner": (x0, y0),
            "buttons": ("UP",),
        },
        {
            "label": "nw_left",
            "claim": "NW lip (832,3976) hold LEFT → x<=688 keys>=1 (west door at north lip)",
            "corner": (x0, y0),
            "buttons": ("LEFT",),
        },
        {
            "label": "nw_up_left",
            "claim": "NW lip hold UP+LEFT → x<=688 or y<3976 keys>=1",
            "corner": (x0, y0),
            "buttons": ("UP", "LEFT"),
        },
        {
            "label": "sw_down",
            "claim": "SW lip (832,4008) hold DOWN → y>4008 keys>=1 (south wrap)",
            "corner": (x0, y1),
            "buttons": ("DOWN",),
        },
        {
            "label": "sw_left",
            "claim": "SW lip (832,4008) hold LEFT → x<=688 keys>=1",
            "corner": (x0, y1),
            "buttons": ("LEFT",),
        },
        {
            "label": "sw_down_left",
            "claim": "SW lip hold DOWN+LEFT → x<=688 or y>4008 keys>=1",
            "corner": (x0, y1),
            "buttons": ("DOWN", "LEFT"),
        },
        {
            "label": "ne_up",
            "claim": "NE lip (944,3976) hold UP → y<3976 keys>=1",
            "corner": (x1, y0),
            "buttons": ("UP",),
        },
        {
            "label": "se_down",
            "claim": "SE lip (944,4008) hold DOWN → y>4008 keys>=1",
            "corner": (x1, y1),
            "buttons": ("DOWN",),
        },
        {
            "label": "mid_north_left",
            "claim": "north wall x=880 hold LEFT → x<=688 keys>=1",
            "corner": (880, y0),
            "buttons": ("LEFT",),
        },
        {
            "label": "dash_nw_left",
            "claim": "NW lip A+LEFT dash/lift → x<=688 keys>=1",
            "corner": (x0, y0),
            "buttons": ("A", "LEFT"),
        },
        {
            "label": "dash_nw_up",
            "claim": "NW lip A+UP dash → y<3976 keys>=1",
            "corner": (x0, y0),
            "buttons": ("A", "UP"),
        },
    ]


def y_sweep_left(env: object, origin: bytes) -> list[dict[str, Any]]:
    """LEFT at every 4px y in the pocket (door may sit off the pin y)."""
    hits: list[dict[str, Any]] = []
    for y in range(POCKET[1], POCKET[3] + 1, 4):
        restore(env, origin)
        r = walk(env, PIN_XY[0], y)
        pose = glance(env)
        h = hold(env, ("LEFT",), max_frames=220)
        rec = {
            "label": f"left_y{y}",
            "claim": f"LEFT at y={y} → x<=688 keys>=1",
            "corner": [PIN_XY[0], y],
            "corner_walk": {"ok": r.ok, "reason": r.reason, "pose": pose},
            "hold": h,
            "end": h["end"],
            "miss": not h["end"]["opened"],
            "still_pocket": h["end"]["in_pocket"],
        }
        print(
            f"  left_y{y} → {brief(h['end'])} miss={rec['miss']}"
        )
        hits.append(rec)
        if h["end"]["opened"]:
            rec["south"] = try_south_door(env)
            break
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Walk SecondKey out of 0x71 east pocket")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--glance-only", action="store_true")
    args = parser.parse_args(argv)
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    env = build_boot_env(PIN, render_mode="rgb_array")
    wraps: list[dict[str, Any]] = []
    lifts: list[dict[str, Any]] = []
    sweep: list[dict[str, Any]] = []
    escaped: dict[str, Any] | None = None
    try:
        env.reset()
        load_pin(env)
        settle_control(env)
        start = glance(env)
        origin = pin_state(env)
        save_overlay(env, out / "pin.png", f"SecondKey pin {brief(start)}")
        print(f"PIN {brief(start)} sprites={start['sprites']} bombs={start['leftover']['F343_bombs']} "
              f"boots={start['leftover']['F355_boots']} carry={start['leftover']['0308_carry']}")
        if args.glance_only:
            payload = {
                "schema": "alttp_second_key",
                "schemaVersion": 1,
                "measured": datetime.now(timezone.utc).isoformat(),
                "source": "state_load_dev",
                "poked": False,
                "start": start,
                "glanceOnly": True,
            }
            (out / "leftover.json").write_text(json.dumps(payload, indent=2) + "\n")
            return 0

        for spec in wrap_plan():
            rec = one_wrap(
                env,
                origin,
                label=spec["label"],
                claim=spec["claim"],
                corner=spec["corner"],
                buttons=spec["buttons"],
            )
            wraps.append(rec)
            if rec["end"]["opened"]:
                escaped = rec
                save_overlay(env, out / "opened.png", f"opened {brief(rec['end'])}")
                break

        if escaped is None:
            sweep = y_sweep_left(env, origin)
            if sweep and sweep[-1]["end"]["opened"]:
                escaped = sweep[-1]
                save_overlay(env, out / "opened.png", f"opened {brief(escaped['end'])}")

        if escaped is None:
            lifts.append(
                lift_then_hold(
                    env,
                    origin,
                    label="lift_west",
                    claim="A at west wall then LEFT (pot/block) → x<=688 keys>=1",
                    corner=(POCKET[0], PIN_XY[1]),
                    buttons=("LEFT",),
                )
            )
            if lifts[-1]["end"]["opened"]:
                escaped = lifts[-1]
            else:
                lifts.append(
                    lift_then_hold(
                        env,
                        origin,
                        label="lift_north",
                        claim="A at north wall then UP (pot) → y<3976 keys>=1",
                        corner=(PIN_XY[0], POCKET[1]),
                        buttons=("UP",),
                    )
                )
                if lifts[-1]["end"]["opened"]:
                    escaped = lifts[-1]

        if escaped is None:
            restore(env, origin)
            end = glance(env)
            save_overlay(env, out / "boxed.png", f"boxed {brief(end)}")
        else:
            end = escaped.get("south", {}).get("end") or escaped["end"]
            if end["room_base_id"] == ROOM_81:
                save_overlay(env, out / "room_81.png", f"0x81 {brief(end)}")

        payload = {
            "schema": "alttp_second_key",
            "schemaVersion": 1,
            "measured": datetime.now(timezone.utc).isoformat(),
            "source": "state_load_dev",
            "poked": False,
            "pin": PIN,
            "pocket": list(POCKET),
            "openX": OPEN_X,
            "southApproach": list(SOUTH_APPROACH),
            "start": start,
            "wraps": wraps,
            "ySweepLeft": sweep,
            "lifts": lifts,
            "escaped": escaped is not None,
            "escapeLabel": escaped["label"] if escaped else None,
            "end": end,
        }
        path = out / "leftover.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(
            f"wrote {path} escaped={payload['escaped']} label={payload['escapeLabel']} "
            f"end={brief(end)}"
        )
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())

"""Pin Thieves' Town lobby (hyp 0xDB) from HyruleCastleGrounds + inventory pokes.

Dev / isolated / state-load. Does not poke $F3CC, prize bits, or titans.
Indoor pin only after the game's own PreDungeon module ($10=0x06) room-load.

Usage:
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_thieves_entry.py
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_thieves_entry.py --save
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_thieves_entry.py --scout
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_thieves_entry.py --edge east
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
from alttp.ram import FOLLOWER, NUM_KEYS, room_label, wram_index
from alttp.room_sense import overlay_from_env
from alttp.screen_glance import leftover_from_snapshot
from alttp.startup import action_for, build_boot_env, no_action, snapshot_env, step_frames
from retro_harness.env import resync_custom_state, save_state

OUT_DIR = RECORDINGS_DIR / "probe_thieves_entry"
SOURCE_PIN = "HyruleCastleGrounds"
SAVE_NAME = "ThievesTownEntry"
HYP_ROOM = 0xDB
# Archipelago door_addresses: 'Thieves Town': (0x33, (0x00db, 0x58, ...))
THIEVES_ENTRANCE = 0x34  # kEntranceNames: [Room #219] Thieves's Town (0x33 is Hera)
MODULE_PREDUNGEON = 0x06
MODULE_DUNGEON = 0x07

# Handoff loadout (prize not granted). $F354 stays 1 (power gloves, not titans).
POKES: tuple[tuple[int, int, str], ...] = (
    (0xF342, 1, "hookshot"),
    (0xF343, 10, "bombs"),
    (0xF345, 1, "fire_rod"),
    (0xF34B, 1, "hammer"),
    (0xF353, 1, "mirror"),
    (0xF354, 1, "gloves_power"),
    (0xF355, 1, "boots"),
    (0xF357, 1, "moon_pearl"),
    (0xF359, 2, "master_sword"),
    (0xF35A, 1, "fighter_shield"),
    (0xF36C, 0x38, "max_hp_7"),
    (0xF36D, 0x38, "hp_7"),
    (0xF36E, 0x80, "magic_full"),
    # bit1 lift (power gloves) + bit4 dash (boots). Not bit2 titans.
    (0xF379, 0x12, "ability_lift_dash"),
    (0xF3C5, 3, "progress_agahnim1"),
)

LOADOUT_ADDRS: tuple[int, ...] = (
    0xF340,
    0xF342,
    0xF343,
    0xF345,
    0xF34A,
    0xF34B,
    0xF34E,
    0xF353,
    0xF354,
    0xF355,
    0xF356,
    0xF357,
    0xF359,
    0xF35A,
    0xF35B,
    0xF36C,
    0xF36D,
    0xF36E,
    0xF36F,
    0xF379,
    0xF3C5,
    0xF3CC,
    0xF366,
    0xF367,
)


def _memory(env: object) -> Any:
    unwrapped = getattr(env, "unwrapped", env)
    data = getattr(unwrapped, "data", None)
    mem = getattr(data, "memory", None) if data is not None else None
    if mem is None or not hasattr(mem, "assign"):
        raise RuntimeError("env has no data.memory.assign")
    return mem


def peek_u8(env: object, offset: int) -> int:
    ram = env.get_ram()  # type: ignore[attr-defined]
    return int(ram[wram_index(offset)])


def bus_addr(offset: int) -> int:
    """WRAM offset → stable-retro assign address (bank $7E for >= $2000)."""
    off = int(offset)
    return 0x7E0000 + off if off >= 0x2000 else off


def poke_u8(env: object, offset: int, value: int) -> dict[str, Any]:
    """WRAM poke via memory.assign (offset 0xF3xx, not get_ram index)."""
    want = int(value) & 0xFF
    before = peek_u8(env, offset)
    mem = _memory(env)
    addr = bus_addr(offset)
    mem.assign(addr, "|u1", want)
    after = peek_u8(env, offset)
    return {
        "offset": hex(offset),
        "assign": hex(addr),
        "before": before,
        "want": want,
        "after": after,
        "ok": after == want,
    }


def leftover_bytes(env: object) -> dict[str, int]:
    ram = env.get_ram()  # type: ignore[attr-defined]
    out: dict[str, int] = {}
    for addr in LOADOUT_ADDRS:
        out[f"{addr:04X}"] = int(ram[wram_index(addr)])
    out["040C_dungeon"] = int(ram[0x040C])
    out["00EE_layer"] = int(ram[0x00EE])
    out["0FFF_dw"] = int(ram[0x0FFF])
    out["010E_entrance"] = int(ram[0x010E])
    out["2F_dir"] = int(ram[0x2F])
    out["5D_act"] = int(ram[0x5D])
    out["F3CC_follower"] = int(ram[wram_index(FOLLOWER)])
    out["F36F_keys"] = int(ram[wram_index(NUM_KEYS)])
    return out


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
            "dark_world": bool(snap.dark_world),
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
        f"{g.get('room_hex')} ({g.get('link_x')},{g.get('link_y')}) "
        f"mod={g.get('module_hex')}/{g.get('submodule')} "
        f"sw={g.get('sword')} keys={g.get('keys')} "
        f"F3CC={g.get('follower')} dw={g.get('dark_world')} "
        f"ctrl={g.get('has_control')}"
    )


def save_png(env: object, path: Path, title: str) -> None:
    img = overlay_from_env(env, include_all_sprites=True, title=title)
    Image.fromarray(img).save(path)


def apply_loadout(env: object) -> list[dict[str, Any]]:
    recs = []
    for offset, value, name in POKES:
        rec = poke_u8(env, offset, value)
        rec["name"] = name
        recs.append(rec)
        print(f"  poke {name} {rec['offset']} {rec['before']}→{rec['after']} ok={rec['ok']}")
    return recs


def trigger_entrance(env: object, entrance: int, *, module: int = MODULE_PREDUNGEON) -> list[dict[str, Any]]:
    """which_entrance=$010E then Module_PreDungeon ($10=0x06). Real room-load."""
    recs = [
        poke_u8(env, 0x010E, entrance),
        poke_u8(env, 0x010F, 0),
        poke_u8(env, 0x010C, MODULE_PREDUNGEON),
        poke_u8(env, 0x0011, 0),
        poke_u8(env, 0x00B0, 0),
        poke_u8(env, 0x0010, module),
    ]
    return recs


def wait_indoor_control(env: object, *, max_frames: int = 900) -> dict[str, Any]:
    frames = 0
    idle = no_action()
    while frames < max_frames:
        step_frames(env, idle, 4)
        frames += 4
        snap = snapshot_env(env)
        if snap.is_text_mode:
            step_frames(env, action_for("A"), 2)
            step_frames(env, idle, 2)
            frames += 4
            continue
        if (
            snap.indoors
            and snap.has_control
            and not snap.is_hold_up_item
            and snap.game_mode == MODULE_DUNGEON
        ):
            return {"ok": True, "frames": frames, "glance": glance(env)}
    return {"ok": False, "frames": frames, "glance": glance(env)}


def preload_thieves(env: object, entrance: int = THIEVES_ENTRANCE) -> dict[str, Any]:
    recs = trigger_entrance(env, entrance)
    wait = wait_indoor_control(env)
    end = wait["glance"]
    rec = {
        "entrance": hex(entrance),
        "trigger": recs,
        "wait": {"ok": wait["ok"], "frames": wait["frames"]},
        "end": end,
    }
    print(f"  entrance {hex(entrance)} wait={wait['ok']}/{wait['frames']}f → {brief(end)}")
    return rec


def hold(
    env: object,
    buttons: tuple[str, ...],
    *,
    max_frames: int = 240,
    stop_room: int | None = None,
) -> dict[str, Any]:
    start = snapshot_env(env)
    frames = 0
    idle = 0
    prev = (start.link_x, start.link_y, start.room_base_id, start.submodule)
    events: list[dict[str, Any]] = []
    action = no_action() if not buttons or buttons == ("NONE",) else action_for(*buttons)
    while frames < max_frames:
        step_frames(env, action, 4)
        frames += 4
        snap = snapshot_env(env)
        cur = (snap.link_x, snap.link_y, snap.room_base_id, snap.submodule)
        if cur != prev:
            idle = 0
            events.append(
                {
                    "f": frames,
                    "room": f"0x{snap.room_base_id:02X}",
                    "mod": int(snap.game_mode),
                    "sub": int(snap.submodule),
                    "xy": [snap.link_x, snap.link_y],
                    "ctrl": bool(snap.has_control),
                    "in": bool(snap.indoors),
                }
            )
            if stop_room is not None and snap.room_base_id == stop_room and snap.has_control:
                break
            if snap.room_base_id != start.room_base_id and snap.has_control:
                break
            if snap.game_mode == 0x12:
                break
        elif snap.has_control:
            idle += 1
            if idle >= 16:
                break
        prev = cur
    settle_control(env, max_frames=180)
    return {
        "buttons": list(buttons),
        "frames": frames,
        "events": events[-24:],
        "end": glance(env),
    }


def scout_cardinals(env: object, out: Path) -> dict[str, Any]:
    start = glance(env)
    rec: dict[str, Any] = {"start": start, "holds": {}}
    for name, buttons in (
        ("UP", ("UP",)),
        ("DOWN", ("DOWN",)),
        ("LEFT", ("LEFT",)),
        ("RIGHT", ("RIGHT",)),
    ):
        env.reset()  # type: ignore[attr-defined]
        if not resync_custom_state(env, GAME_DIR, INTEGRATION, SAVE_NAME):
            # Fall back to freshly preloaded HyruleCastleGrounds pin.
            if not resync_custom_state(env, GAME_DIR, INTEGRATION, SOURCE_PIN):
                raise RuntimeError("failed to reload pin")
            apply_loadout(env)
            preload_thieves(env)
        settle_control(env)
        h = hold(env, buttons, max_frames=320)
        rec["holds"][name] = h
        save_png(env, out / f"hold_{name.lower()}.png", f"{name} {brief(h['end'])}")
        print(f"  {name} → {brief(h['end'])} events={len(h['events'])}")
    return rec


EXPLORE_WPS: tuple[tuple[str, int, int], ...] = (
    ("south_hall", 5880, 7040),
    ("mid_hall", 5880, 6960),
    ("right_of_platform", 5968, 6960),
    ("right_north", 5968, 6848),
    ("east_mid", 6080, 6848),
    ("east_boundary", 6160, 6848),
    ("se_quad", 6240, 6848),
    ("north_of_spawn", 5880, 6880),
    ("left_of_platform", 5760, 6960),
    ("west_north", 5760, 6720),
    ("nw_quad", 5760, 6600),
)


CHAIN_PATH: tuple[tuple[str, int, int], ...] = (
    ("east_south", 5952, 7104),
    ("east_up", 5952, 6968),
    ("conveyor", 5952, 6856),
    ("east_along", 6120, 6856),
    ("east_dc", 6400, 6856),
)


def chain_east(env: object, out: Path) -> dict[str, Any]:
    rec: dict[str, Any] = {"start": glance(env), "stops": []}
    for label, x, y in CHAIN_PATH:
        r = move_to(env, Waypoint(x, y, tolerance=12), max_frames=500)
        g = glance(env)
        rec["stops"].append(
            {"label": label, "want": [x, y], "reason": r.reason, "ok": r.ok, "end": g}
        )
        save_png(env, out / f"chain_{label}.png", f"{label} {brief(g)}")
        print(f"  chain {label} want=({x},{y}) → {brief(g)} {r.reason}")
        if int(g.get("module", 0)) == 0x12:
            rec["died"] = label
            return rec
        if int(g.get("room", HYP_ROOM)) != HYP_ROOM and g.get("has_control"):
            rec["hop"] = {"label": label, "end": g}
            return rec
    push = hold(env, ("RIGHT",), max_frames=400)
    rec["push"] = push
    save_png(env, out / "chain_push_right.png", f"push {brief(push['end'])}")
    print(f"  chain RIGHT → {brief(push['end'])}")
    rec["end"] = push["end"]
    if int(push["end"].get("room", HYP_ROOM)) != HYP_ROOM:
        rec["hop"] = {"label": "push_right", "end": push["end"]}
    return rec


def explore_room(env: object, out: Path) -> dict[str, Any]:
    rec: dict[str, Any] = {"start": glance(env), "stops": []}
    for label, x, y in EXPLORE_WPS:
        env.reset()  # type: ignore[attr-defined]
        if not resync_custom_state(env, GAME_DIR, INTEGRATION, SAVE_NAME):
            raise RuntimeError("failed to load ThievesTownEntry")
        settle_control(env)
        r = move_to(env, Waypoint(x, y, tolerance=10), max_frames=700)
        g = glance(env)
        rec["stops"].append(
            {
                "label": label,
                "want": [x, y],
                "reason": r.reason,
                "ok": r.ok,
                "end": g,
            }
        )
        save_png(env, out / f"ex_{label}.png", f"{label} {brief(g)}")
        print(f"  {label} want=({x},{y}) → {brief(g)} {r.reason}")
        if g.get("room") != HYP_ROOM and g.get("has_control"):
            rec["hop"] = {"label": label, "end": g}
            break
    return rec


def isolate_south(env: object, out: Path) -> dict[str, Any]:
    """DOWN from lobby spawn → Village of Outcasts (overworld)."""
    start = glance(env)
    push = hold(env, ("DOWN",), max_frames=360)
    end = push["end"]
    rec = {
        "start": start,
        "push": push,
        "end": end,
        "ok": (not end.get("indoors")) and end.get("has_control") and int(end.get("module", 0)) == 0x09,
    }
    save_png(env, out / "south_push.png", f"south {brief(end)}")
    print(f"  south isolate → {brief(end)} ok={rec['ok']}")
    return rec


def isolate_east(env: object, out: Path) -> dict[str, Any]:
    """Walk east from lobby spawn and push the first door."""
    start = glance(env)
    here = snapshot_env(env)
    room = int(here.room_base_id)
    mid = (here.link_x + 80, here.link_y)
    east = (here.link_x + 200, here.link_y)
    w1 = move_to(env, Waypoint(mid[0], mid[1], tolerance=8, room=room), max_frames=400)
    pose = glance(env)
    w2 = move_to(
        env,
        Waypoint(east[0], pose["link_y"], tolerance=8, room=room),
        max_frames=400,
    )
    approach = glance(env)
    push = hold(env, ("RIGHT",), max_frames=360)
    end = push["end"]
    rec = {
        "start": start,
        "walk": [w1.reason, w2.reason],
        "approach": approach,
        "push": push,
        "end": end,
        "ok": end.get("room_base_id") != start.get("room_base_id") and end.get("has_control"),
    }
    save_png(env, out / "east_push.png", f"east {brief(end)}")
    print(f"  east isolate → {brief(end)} ok={rec['ok']}")
    return rec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Thieves' Town entry pin")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--save", action="store_true", help="write ThievesTownEntry.state")
    parser.add_argument("--scout", action="store_true", help="cardinal holds from the pin")
    parser.add_argument("--explore", action="store_true", help="waypoint hunt for 0xDC/0xCB")
    parser.add_argument("--chain", action="store_true", help="chained east path from pin")
    parser.add_argument("--edge", choices=("east", "south"), default=None)
    parser.add_argument("--from-save", action="store_true", help="load ThievesTownEntry")
    parser.add_argument("--entrance", type=lambda s: int(s, 0), default=THIEVES_ENTRANCE)
    args = parser.parse_args(argv)
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    pin = SAVE_NAME if args.from_save else SOURCE_PIN
    env = build_boot_env(pin, render_mode="rgb_array")
    payload: dict[str, Any] = {
        "schema": "alttp_thieves_entry",
        "schemaVersion": 1,
        "measured": datetime.now(timezone.utc).isoformat(),
        "source": "state_load_dev",
        "sourcePin": SOURCE_PIN,
        "saveName": SAVE_NAME,
        "hypRoom": hex(HYP_ROOM),
        "pokes": [],
    }
    try:
        env.reset()
        if not resync_custom_state(env, GAME_DIR, INTEGRATION, pin):
            raise RuntimeError(f"failed to load {pin}")
        settle_control(env)
        payload["start"] = glance(env)
        save_png(env, out / "start.png", f"start {brief(payload['start'])}")
        print(f"START {brief(payload['start'])}")

        if not args.from_save:
            payload["pokes"] = apply_loadout(env)
            payload["afterPokes"] = glance(env)
            save_png(env, out / "after_pokes.png", f"pokes {brief(payload['afterPokes'])}")
            payload["preload"] = preload_thieves(env, args.entrance)
            payload["lobby"] = payload["preload"]["end"]
        else:
            payload["lobby"] = glance(env)
        save_png(env, out / "lobby.png", f"lobby {brief(payload['lobby'])}")
        print(f"LOBBY {brief(payload['lobby'])}")

        lobby = payload["lobby"]
        indoor_ok = (
            bool(lobby.get("indoors"))
            and int(lobby.get("module", 0)) == MODULE_DUNGEON
            and bool(lobby.get("has_control"))
            and int(lobby.get("room", lobby.get("room_base_id", 0))) == HYP_ROOM
        )
        payload["indoorOk"] = indoor_ok

        if args.save and indoor_ok:
            path = save_state(env, GAME_DIR, INTEGRATION, SAVE_NAME)
            payload["saved"] = str(path)
            print(f"SAVED {path}")

        if args.scout:
            payload["scout"] = scout_cardinals(env, out)
        if args.explore:
            payload["explore"] = explore_room(env, out)
        if args.chain:
            payload["chain"] = chain_east(env, out)
        if args.edge == "south":
            if args.from_save:
                env.reset()
                resync_custom_state(env, GAME_DIR, INTEGRATION, SAVE_NAME)
                settle_control(env)
            payload["south"] = isolate_south(env, out)
        if args.edge == "east":
            if args.from_save:
                env.reset()
                resync_custom_state(env, GAME_DIR, INTEGRATION, SAVE_NAME)
                settle_control(env)
            payload["east"] = isolate_east(env, out)

        (out / "leftover.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"WROTE {out / 'leftover.json'} indoorOk={indoor_ok}")
        return 0 if indoor_ok or args.scout or args.edge else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())

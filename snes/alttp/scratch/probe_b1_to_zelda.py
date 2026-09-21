"""Isolated 0x72 (CastleB1Guard) → Zelda cell 0x80 chase.

Usage:
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_b1_to_zelda.py
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from alttp.opening_route.room_engine import run_room_edge
from alttp.paths import GAME_DIR, INTEGRATION, RECORDINGS_DIR
from alttp.primitives import (
    SPRITE_SMALL_KEY,
    Waypoint,
    active_sprites,
    fight_nearby,
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
from alttp.room_sense import detect_edge
from alttp.startup import action_for, build_boot_env, no_action, snapshot_env, step_frames
from retro_harness.env import resync_custom_state

OUT_DIR = RECORDINGS_DIR / "probe_b1_to_zelda"
DIRS = ("UP", "DOWN", "LEFT", "RIGHT")


def leftover_bytes(env: object) -> dict[str, int]:
    ram = env.get_ram()  # type: ignore[attr-defined]
    return {
        "F359_sword": int(ram[wram_index(EQUIP_SWORD)]),
        "F34A_lamp": int(ram[wram_index(LINK_ITEM_LAMP)]),
        "F36F_keys": int(ram[wram_index(NUM_KEYS)]),
        "F3CC_follower": int(ram[wram_index(FOLLOWER)]),
        "F36D_hp": int(ram[wram_index(LINK_HP)]),
        "F36C_max_hp": int(ram[wram_index(LINK_MAX_HP)]),
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
        "sword": int(snap.sword_level),
        "lamp": int(snap.lamp_level),
        "keys": int(snap.num_keys),
        "follower": int(snap.follower),
        "has_control": bool(snap.has_control),
        "leftover": leftover_bytes(env),
        "sprites": sprites[:12],
    }


def hold_until_room(
    env: object, direction: str, dest: int | None, *, max_frames: int = 720
) -> dict[str, Any]:
    start = snapshot_env(env)
    frames = 0
    idle = 0
    prev = (start.link_x, start.link_y, start.room_base_id, start.submodule)
    events: list[dict[str, Any]] = []
    while frames < max_frames:
        step_frames(env, action_for(direction), 4)
        frames += 4
        snap = snapshot_env(env)
        cur = (snap.link_x, snap.link_y, snap.room_base_id, snap.submodule)
        if cur != prev:
            idle = 0
            events.append(
                {
                    "f": frames,
                    "room": f"0x{snap.room_base_id:02X}",
                    "sub": int(snap.submodule),
                    "xy": [snap.link_x, snap.link_y],
                    "ctrl": bool(snap.has_control),
                }
            )
            if dest is not None and snap.room_base_id == dest and snap.has_control:
                break
            if snap.room_base_id != start.room_base_id and snap.has_control:
                break
        elif snap.has_control:
            idle += 1
            if idle >= 10:
                break
        prev = cur
    return {"dir": direction, "frames": frames, "events": events[-12:], "end": glance(env)}


def walk(env: object, x: int, y: int, *, room: int | None = None, frames: int = 500) -> Any:
    return move_to(env, Waypoint(x, y, tolerance=12, room=room), max_frames=frames)


def axis_walk(env: object, x: int, y: int, *, room: int | None = None) -> Any:
    """Axis-aligned (x then y) so diagonals do not fall in the 0x72 pit."""
    here = snapshot_env(env)
    r = walk(env, x, here.link_y, room=room)
    if not r.ok:
        return r
    return walk(env, x, y, room=room)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="0x72 Guard → 0x80 chase")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--no-key", action="store_true")
    parser.add_argument("--ledge", action="store_true", help="Sweep north-ledge DOWN rays.")
    parser.add_argument("--lip", action="store_true", help="Walk the y=3776 lip looking for a pit gap.")
    parser.add_argument("--star", action="store_true", help="Step on the north-platform star tile.")
    parser.add_argument("--block", action="store_true", help="Push the south block on the north ledge.")
    parser.add_argument("--from-pit", action="store_true", help="Chase 0x72 south (CastleB1Pit) toward 0x80.")
    parser.add_argument("--cell", action="store_true", help="From 0x81 pins, hunt west door to 0x80.")
    args = parser.parse_args(argv)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    env = build_boot_env("CastleB1Guard", render_mode="rgb_array")
    log: list[dict[str, Any]] = []
    try:
        env.reset()
        pin = "CastleB1Pit" if args.from_pit else "CastleB1FarDoor" if args.cell else "CastleB1Guard"
        assert resync_custom_state(env, GAME_DIR, INTEGRATION, pin)
        settle_control(env)
        start = glance(env)
        print(
            f"START room={start['room_hex']} xy=({start['link_x']},{start['link_y']}) "
            f"keys={start['keys']} F3CC={start['follower']} "
            f"hp={start['leftover']['F36D_hp']}/{start['leftover']['F36C_max_hp']}"
        )
        if args.from_pit:
            fight_nearby(env, room=0x72, max_distance=160, max_cycles=200)
            settle_control(env)
            print(f"post-fight {glance(env)['link_x']},{glance(env)['link_y']} hp={leftover_bytes(env)['F36D_hp']}")
            axis_walk(env, 1184, 4080, room=0x72)
            south = hold_until_room(env, "DOWN", 0x82)
            print(f"south {south['end']['room_hex']} ({south['end']['link_x']},{south['end']['link_y']})")
            if snapshot_env(env).room_base_id == 0x82:
                fight_nearby(env, room=0x82, max_distance=80, max_cycles=80)
                axis_walk(env, 1010, 4496, room=0x82)
                west = hold_until_room(env, "LEFT", 0x81)
                print(f"west82 {west['end']['room_hex']} ({west['end']['link_x']},{west['end']['link_y']})")
            if snapshot_env(env).room_base_id == 0x81:
                fight_nearby(env, room=0x81, max_distance=140, max_cycles=160)
                axis_walk(env, 632, 4168, room=0x81)
                cell = hold_until_room(env, "LEFT", 0x80, max_frames=500)
                print(
                    f"cell {cell['end']['room_hex']} ({cell['end']['link_x']},{cell['end']['link_y']}) "
                    f"F3CC={cell['end']['follower']} keys={cell['end']['keys']}"
                )
            end = glance(env)
            (out / "from_pit.json").write_text(
                json.dumps({"start": start, "end": end}, indent=2) + "\n"
            )
            return 0
        if args.cell:
            origin = env.em.get_state()  # type: ignore[attr-defined]
            for name in ("CastleB1FarDoor", "CastleB1WestRoom", "CastleB1GreenRoom", "CastleB1Shutter"):
                if not resync_custom_state(env, GAME_DIR, INTEGRATION, name):
                    continue
                settle_control(env)
                s = glance(env)
                print(f"{name} {s['room_hex']} ({s['link_x']},{s['link_y']}) keys={s['keys']}")
                if s["room_base_id"] != 0x81:
                    continue
                axis_walk(env, 632, s["link_y"], room=0x81)
                axis_walk(env, 632, 4168, room=0x81)
                h = hold_until_room(env, "LEFT", 0x80, max_frames=400)
                e = h["end"]
                print(f"  LEFT → {e['room_hex']} ({e['link_x']},{e['link_y']}) F3CC={e['follower']}")
                if e["room_base_id"] == 0x80:
                    (out / "cell.json").write_text(json.dumps({"pin": name, "start": s, "end": e}, indent=2) + "\n")
                    return 0
            end = glance(env)
            (out / "cell.json").write_text(json.dumps({"start": start, "end": end}, indent=2) + "\n")
            return 0
        if args.ledge:
            origin = env.em.get_state()  # type: ignore[attr-defined]
            rays = []
            for x in range(1184, 1336, 16):
                env.em.set_state(origin)  # type: ignore[attr-defined]
                settle_control(env)
                axis_walk(env, x, start["link_y"], room=0x72)
                pose = snapshot_env(env)
                if pose.room_base_id != 0x72:
                    continue
                hold_until_room(env, "DOWN", None, max_frames=240)
                end = glance(env)
                rec = {
                    "x_target": x,
                    "pose": [pose.link_x, pose.link_y],
                    "end": [end["link_x"], end["link_y"]],
                    "room": end["room_hex"],
                    "mod": end["module"],
                    "hp": end["leftover"]["F36D_hp"],
                }
                rays.append(rec)
                print(f"ledge x={x} pose={rec['pose']} DOWN→{rec['end']} room={rec['room']} hp={rec['hp']}")
            (out / "ledge.json").write_text(json.dumps({"start": start, "rays": rays}, indent=2) + "\n")
            return 0
        if args.lip:
            from PIL import Image
            from alttp.room_map import load_room_map
            from alttp.room_sense import overlay_from_env

            m = load_room_map("room_72")
            img = overlay_from_env(env, include_all_sprites=True, points=m.points, title="Guard 0x72")
            Image.fromarray(img).save(out / "guard_overlay.png")
            origin = env.em.get_state()  # type: ignore[attr-defined]
            # Reach east lip without fighting, then slide west along y=3776.
            axis_walk(env, 1320, start["link_y"], room=0x72)
            hold_until_room(env, "DOWN", None, max_frames=200)
            lip0 = glance(env)
            print(f"east lip {lip0['link_x']},{lip0['link_y']} hp={lip0['leftover']['F36D_hp']}")
            events = []
            for _ in range(80):
                step_frames(env, action_for("LEFT"), 4)
                snap = snapshot_env(env)
                events.append(
                    {
                        "xy": [snap.link_x, snap.link_y],
                        "room": f"0x{snap.room_base_id:02X}",
                        "mod": snap.game_mode,
                        "sub": snap.submodule,
                        "hp": leftover_bytes(env)["F36D_hp"],
                        "ctrl": snap.has_control,
                    }
                )
                if snap.game_mode == 0x12 or snap.room_base_id != 0x72 or not snap.has_control:
                    print(f"lip change {events[-1]}")
                    if snap.game_mode == 0x12:
                        break
                    settle_control(env, max_frames=400)
                    break
            end = glance(env)
            print(f"lip end room={end['room_hex']} xy=({end['link_x']},{end['link_y']}) hp={end['leftover']['F36D_hp']} mod={end['module_hex']}/{end['submodule']}")
            (out / "lip.json").write_text(
                json.dumps({"start": start, "east_lip": lip0, "slide": events[::4], "end": end}, indent=2) + "\n"
            )
            return 0
        if args.star:
            origin = env.em.get_state()  # type: ignore[attr-defined]
            hits = []
            for x, y in (
                (1272, 3696),
                (1272, 3704),
                (1272, 3712),
                (1280, 3704),
                (1264, 3704),
                (1272, 3728),
                (1248, 3704),
                (1296, 3704),
            ):
                env.em.set_state(origin)  # type: ignore[attr-defined]
                settle_control(env)
                r = axis_walk(env, x, y, room=0x72)
                step_frames(env, no_action(), 30)
                after = glance(env)
                hits.append(
                    {
                        "target": [x, y],
                        "ok": r.ok,
                        "reason": r.reason,
                        "xy": [after["link_x"], after["link_y"]],
                        "room": after["room_hex"],
                        "sub": after["submodule"],
                    }
                )
                print(
                    f"star {x},{y} ok={r.ok} now=({after['link_x']},{after['link_y']}) "
                    f"room={after['room_hex']} sub={after['submodule']} {r.reason}"
                )
            (out / "star.json").write_text(json.dumps({"start": start, "hits": hits}, indent=2) + "\n")
            return 0
        if args.block:
            walk(env, 1273, 3688, room=0x72)
            step_frames(env, action_for("A"), 16)
            step_frames(env, action_for("A", "DOWN"), 180)
            b = glance(env)
            print(f"grab DOWN xy=({b['link_x']},{b['link_y']}) sprites={b['sprites']}")
            from PIL import Image
            from alttp.room_map import load_room_map
            from alttp.room_sense import overlay_from_env

            m = load_room_map("room_72")
            Image.fromarray(
                overlay_from_env(env, include_all_sprites=True, points=m.points, title="after grab")
            ).save(out / "after_grab.png")
            slide = hold_until_room(env, "LEFT", None, max_frames=280)
            print(f"slide L events={slide['events'][-6:]} end=({slide['end']['link_x']},{slide['end']['link_y']})")
            fight_nearby(env, room=0x72, max_distance=48, max_cycles=80)
            down = hold_until_room(env, "DOWN", None, max_frames=360)
            settle_control(env, max_frames=400)
            end = glance(env)
            print(
                f"end room={end['room_hex']} xy=({end['link_x']},{end['link_y']}) "
                f"hp={end['leftover']['F36D_hp']} F3CC={end['follower']}"
            )
            (out / "block.json").write_text(
                json.dumps({"start": start, "grab": b, "slide": slide, "down": down, "end": end}, indent=2)
                + "\n"
            )
            return 0
        keys_on_floor = sprites_of_type(env, (SPRITE_SMALL_KEY,), max_distance=200)
        print(f"keys_on_floor={[(s.x, s.y) for s in keys_on_floor]}")

        if not args.no_key and keys_on_floor:
            k = keys_on_floor[0]
            axis_walk(env, k.x, k.y, room=0x72)
            settle_control(env, max_frames=180)
            print(f"after key xy=({snapshot_env(env).link_x},{snapshot_env(env).link_y}) keys={snapshot_env(env).num_keys}")

        # Soldiers sit on the south lip of the north ledge (~y=3760). Fight first.
        combat = fight_nearby(env, room=0x72, max_distance=200, max_cycles=250)
        print(f"combat reason={combat.reason} frames={combat.frames} now={glance(env)['link_x']},{glance(env)['link_y']}")
        settle_control(env)
        for x, y in ((1320, 3656), (1320, 3872), (1320, 4032), (1272, 4032), (1184, 4080)):
            r = axis_walk(env, x, y, room=0x72)
            snap = snapshot_env(env)
            print(
                f"axis ({x},{y}) ok={r.ok} now=({snap.link_x},{snap.link_y}) "
                f"room=0x{snap.room_base_id:02X} {r.reason}"
            )
            if snap.game_mode == 0x12 or snap.room_base_id != 0x72:
                break
        stuck = snapshot_env(env)
        print(
            f"pre-fall xy=({stuck.link_x},{stuck.link_y}) hp={leftover_bytes(env)['F36D_hp']}/"
            f"{leftover_bytes(env)['F36C_max_hp']} room=0x{stuck.room_base_id:02X}"
        )
        fall = hold_until_room(env, "DOWN", None, max_frames=900)
        log.append({"step": "pit_fall", **fall})
        settle_control(env, max_frames=400)
        after_fall = glance(env)
        print(
            f"FALL room={after_fall['room_hex']} xy=({after_fall['link_x']},{after_fall['link_y']}) "
            f"mod={after_fall['module_hex']}/{after_fall['submodule']} "
            f"hp={after_fall['leftover']['F36D_hp']} keys={after_fall['keys']}"
        )
        south = {"end": after_fall}

        if snapshot_env(env).room_base_id == 0x82:
            fight_nearby(env, room=0x82, max_distance=70, max_cycles=60)
            walk(env, 1184, 4496, room=0x82, frames=700)
            walk(env, 1010, 4496, room=0x82, frames=500)
            west = hold_until_room(env, "LEFT", 0x81)
            log.append({"step": "west_to_0x81", **west})
            print(f"WEST82 end room={west['end']['room_hex']} xy=({west['end']['link_x']},{west['end']['link_y']})")

        if snapshot_env(env).room_base_id == 0x81:
            fight_nearby(env, room=0x81, max_distance=120, max_cycles=120)
            here = snapshot_env(env)
            # Sweep west along y~4168 (cell door band) and y~4496 (south seam).
            for y in (here.link_y, 4496, 4316, 4168, 4152):
                walk(env, here.link_x, y, room=0x81, frames=400)
                walk(env, 632, y, room=0x81, frames=600)
                snap = snapshot_env(env)
                print(f"81 y={y} now=({snap.link_x},{snap.link_y}) room=0x{snap.room_base_id:02X}")
                if snap.room_base_id != 0x81:
                    break
            if snapshot_env(env).room_base_id == 0x81:
                cell = hold_until_room(env, "LEFT", 0x80, max_frames=400)
                log.append({"step": "west_to_0x80", **cell})
                print(
                    f"CELL end room={cell['end']['room_hex']} xy=({cell['end']['link_x']},{cell['end']['link_y']}) "
                    f"F3CC={cell['end']['follower']}"
                )

        # Isolated south edge from Guard (map path as-is).
        if not resync_custom_state(env, GAME_DIR, INTEGRATION, "CastleB1Guard"):
            raise RuntimeError("reload Guard failed")
        settle_control(env)
        edge = run_room_edge(
            env, "room_72", "south_to_0x82", clear=True, source="state_load_dev"
        )
        print(f"room_engine south_to_0x82 ok={edge.ok} phase={edge.phase} dest={edge.snapshot.room_base_id:#x}")
        end = glance(env)
        payload = {
            "schema": "alttp_b1_to_zelda",
            "schemaVersion": 1,
            "measured": datetime.now(timezone.utc).isoformat(),
            "source": "state_load_dev",
            "poked": False,
            "start": start,
            "end": end,
            "engineSouth": {
                "ok": edge.ok,
                "phase": edge.phase,
                "room": f"0x{edge.snapshot.room_base_id:02X}",
                "xy": [edge.snapshot.link_x, edge.snapshot.link_y],
            },
            "log": log,
        }
        (out / "chase.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {out / 'chase.json'} F3CC={end['follower']} room={end['room_hex']}")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())

"""0x82 water maze: 0x72-south leftover → west door to 0x81.

Natural-entry question: does CastleB1PitGuardCleared south_to_0x82 chain
across the water/pit to west_to_0x81 without drowning or north-door respawn.

Usage:
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_82_water.py
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_82_water.py --hold LEFT
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

from alttp.opening_route.room_engine import run_room_edge
from alttp.paths import RECORDINGS_DIR
from alttp.primitives import Waypoint, fight_nearby, move_to, settle_control
from alttp.ram import (
    EQUIP_SWORD,
    FOLLOWER,
    LINK_HP,
    LINK_ITEM_LAMP,
    LINK_MAX_HP,
    NUM_KEYS,
    room_label,
    snapshot_to_diag,
    wram_index,
)
from alttp.room_sense import overlay_from_env
from alttp.startup import action_for, build_boot_env, snapshot_env, step_frames
from alttp.screen_glance import leftover_from_snapshot

OUT_DIR = RECORDINGS_DIR / "probe_82_water"
# 0x72-south lands in the north-door alcove (1198,4108). LEFT/RIGHT there
# are walls (x=1184 / x=1200). Floor is south; pit is west; east corridor
# is the south-east of this chamber (island 1275,4253). Halt at first miss.
CLAIMS: tuple[tuple[str, int, int], ...] = (
    ("floor_south", 1198, 4160),
    ("east_lane", 1272, 4160),
    ("island_y", 1272, 4253),
    ("corridor_x", 1255, 4253),
    ("bridge", 1255, 4393),
    ("east_hug", 1324, 4393),
    ("south_walk", 1324, 4492),
    ("east_b1", 1036, 4492),
    ("west_door", 1010, 4496),
)


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
    rec = leftover_from_snapshot(snap)
    rec.update(
        {
            "module_hex": f"0x{int(snap.game_mode):02X}",
            "room_hex": f"0x{int(snap.room_base_id):02X}",
            "room_label": room_label(snap.room_base_id),
            "has_control": bool(snap.has_control),
            "link_action": int(snap.link_action),
            "bytes": leftover_bytes(env),
            "diag": snapshot_to_diag(snap),
        }
    )
    return rec


def save_overlay(env: object, path: Path, title: str) -> None:
    from alttp.room_map import load_room_map

    m = load_room_map("room_82")
    img = overlay_from_env(
        env, include_all_sprites=True, points=m.points, title=title
    )
    Image.fromarray(img).save(path)


def pit_or_dead(snap: Any) -> str | None:
    if int(snap.game_mode) == 0x12:
        return "death_module_0x12"
    if int(snap.submodule) == 20:
        return "pit_submodule_20"
    if int(snap.submodule) not in (0, 1, 2) and int(snap.game_mode) == 0x07:
        if int(snap.submodule) in (9, 14, 16, 17, 18, 19, 20):
            return f"submodule_{int(snap.submodule)}"
    return None


def land_from_pitguard(env: object) -> dict[str, Any]:
    settle_control(env)
    start = glance(env)
    edge = run_room_edge(
        env,
        "room_72",
        "south_to_0x82",
        clear=False,
        source="state_load_dev",
    )
    settle_control(env, max_frames=180)
    end = glance(env)
    return {
        "ok": bool(edge.ok) and end["room"] == 0x82 and end["module"] == 0x07,
        "edge_ok": bool(edge.ok),
        "phase": edge.phase,
        "frames": int(edge.frames),
        "start": start,
        "end": end,
        "blocker": edge.blocker if hasattr(edge, "blocker") else "",
    }


def hold_dir(
    env: object, direction: str, *, max_frames: int = 240, origin_room: int = 0x82
) -> dict[str, Any]:
    """One claimed hold. Halt at pit/death/room-change/stuck."""
    start = glance(env)
    events: list[dict[str, Any]] = []
    frames = 0
    idle = 0
    prev = (start["x"], start["y"], start["room"], start["submodule"])
    miss = None
    while frames < max_frames:
        step_frames(env, action_for(direction), 4)
        frames += 4
        snap = snapshot_env(env)
        cur = (snap.link_x, snap.link_y, snap.room_base_id, snap.submodule)
        rec = {
            "f": frames,
            "xy": [snap.link_x, snap.link_y],
            "room": f"0x{snap.room_base_id:02X}",
            "sub": int(snap.submodule),
            "mod": int(snap.game_mode),
            "ctrl": bool(snap.has_control),
        }
        if cur != prev:
            idle = 0
            events.append(rec)
        elif snap.has_control:
            idle += 1
            if idle >= 8:
                miss = "stuck"
                events.append(rec)
                break
        hit = pit_or_dead(snap)
        if hit:
            miss = hit
            events.append(rec)
            break
        if snap.room_base_id != origin_room and snap.has_control:
            miss = f"left_room_0x{snap.room_base_id:02X}"
            events.append(rec)
            break
        prev = cur
    end = glance(env)
    return {
        "dir": direction,
        "frames": frames,
        "miss": miss,
        "start": start,
        "end": end,
        "events": events[-24:],
        "dx": end["x"] - start["x"],
        "dy": end["y"] - start["y"],
    }


def axis_claim(
    env: object, label: str, x: int, y: int, *, room: int = 0x82
) -> dict[str, Any]:
    """Axis-aligned claim (y then x). West opening is a y-band at ~4152."""
    before = glance(env)
    r1 = move_to(
        env, Waypoint(before["x"], y, tolerance=4, room=room, label=f"{label}_y"),
        max_frames=400,
    )
    mid = glance(env)
    hit = pit_or_dead(snapshot_env(env))
    if hit or not r1.ok or mid["room"] != room:
        return {
            "label": label,
            "target": [x, y],
            "ok": False,
            "miss": hit or r1.reason,
            "before": before,
            "end": mid,
            "phase": "y",
        }
    r2 = move_to(
        env, Waypoint(x, y, tolerance=4, room=room, label=label),
        max_frames=400,
    )
    end = glance(env)
    hit = pit_or_dead(snapshot_env(env))
    crossed = end["room"] == 0x81
    ok = crossed or (
        r2.ok
        and hit is None
        and end["room"] == room
        and abs(end["x"] - x) <= 8
        and abs(end["y"] - y) <= 8
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
        "crossed_to_0x81": crossed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="0x82 water maze from 0x72-south")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--hold", choices=("LEFT", "RIGHT", "UP", "DOWN"))
    parser.add_argument(
        "--holds",
        help="Comma-separated cardinals to hold in order (halt at first miss).",
    )
    parser.add_argument(
        "--nudge-south",
        type=int,
        default=0,
        help="Walk south this many px out of the north alcove before --hold.",
    )
    parser.add_argument(
        "--from-west",
        action="store_true",
        help="Load CastleB1West (north seam) instead of 0x72 south.",
    )
    parser.add_argument(
        "--state",
        help="Load this pin in 0x82 instead of chaining 0x72 south.",
    )
    parser.add_argument("--no-fight", action="store_true")
    parser.add_argument(
        "--west-scan",
        action="store_true",
        help="From landing, south-nudge 8px steps then LEFT; halt on first west progress.",
    )
    parser.add_argument(
        "--east-from-bridge",
        action="store_true",
        help="Follow claims through bridge, then EAST to south_area instead of south into pit.",
    )
    parser.add_argument(
        "--east-wall",
        action="store_true",
        help="Walk the east wall x=1312 from north floor toward CastleB1South.",
    )
    args = parser.parse_args(argv)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    pin = (
        args.state
        if args.state
        else ("CastleB1West" if args.from_west else "CastleB1PitGuardCleared")
    )
    env = build_boot_env(pin, render_mode="rgb_array")
    log: dict[str, Any] = {
        "schema": "alttp_82_water",
        "schemaVersion": 1,
        "measured": datetime.now(timezone.utc).isoformat(),
        "source": "state_load_dev",
        "poked": False,
        "pin": pin,
        "claims": [list(c) for c in CLAIMS],
    }
    try:
        env.reset()
        settle_control(env)
        if args.from_west or args.state:
            land = {"ok": True, "end": glance(env), "skipped": pin}
        else:
            land = land_from_pitguard(env)
        log["land"] = land
        print(
            f"LAND ok={land['ok']} room={land['end']['room_hex']} "
            f"xy=({land['end']['x']},{land['end']['y']}) "
            f"F3CC={land['end']['follower']} sub={land['end']['submodule']}"
        )
        save_overlay(env, out / "land.png", "0x82 after 0x72-south")
        Image.fromarray(env.render()).save(out / "land_raw.png")  # type: ignore[attr-defined]
        if not land["ok"] or land["end"]["room"] != 0x82:
            log["halt"] = "failed_to_land_0x82"
            (out / "probe.json").write_text(json.dumps(log, indent=2) + "\n")
            return 1

        if args.nudge_south:
            here = snapshot_env(env)
            rec = axis_claim(
                env, "nudge_south", here.link_x, here.link_y + args.nudge_south
            )
            log["nudge"] = rec
            print(
                f"NUDGE south {args.nudge_south} ok={rec['ok']} miss={rec['miss']} "
                f"now=({rec['end']['x']},{rec['end']['y']})"
            )
            if not rec["ok"]:
                log["halt"] = rec
                (out / "probe.json").write_text(json.dumps(log, indent=2) + "\n")
                return 2
        hold_seq = [args.hold] if args.hold else (
            [h.strip().upper() for h in args.holds.split(",") if h.strip()]
            if args.holds
            else []
        )
        if hold_seq:
            held_log = []
            miss = None
            for direction in hold_seq:
                held = hold_dir(env, direction)
                held_log.append(held)
                print(
                    f"HOLD {direction} miss={held['miss']} "
                    f"d=({held['dx']},{held['dy']}) "
                    f"end=({held['end']['x']},{held['end']['y']}) "
                    f"sub={held['end']['submodule']} room={held['end']['room_hex']}"
                )
                save_overlay(
                    env, out / f"hold_{direction.lower()}.png", f"hold {direction}"
                )
                if held["miss"] and held["miss"] != "stuck":
                    miss = held
                    break
                if held["miss"] == "stuck" and direction != hold_seq[-1]:
                    # stuck is OK between sequence steps; try the next cardinal
                    continue
                if held["miss"] == "stuck":
                    miss = held
            log["holds"] = held_log
            (out / "probe.json").write_text(json.dumps(log, indent=2) + "\n")
            return 0 if miss is None else 2

        if args.west_scan:
            origin = env.em.get_state()  # type: ignore[attr-defined]
            rows = []
            found = None
            for dy in range(0, 201, 8):
                env.em.set_state(origin)  # type: ignore[attr-defined]
                settle_control(env)
                here = snapshot_env(env)
                if dy:
                    rec = axis_claim(env, f"south_{dy}", here.link_x, here.link_y + dy)
                    if rec["miss"] and (
                        str(rec["miss"]).startswith("pit") or rec["end"]["module"] == 0x12
                    ):
                        rows.append({"dy": dy, "south": rec, "miss": rec["miss"]})
                        print(f"WESTSCAN dy={dy} south miss={rec['miss']}")
                        break
                pose = glance(env)
                held = hold_dir(env, "LEFT", max_frames=160)
                row = {
                    "dy": dy,
                    "after_south": pose,
                    "hold": {
                        "miss": held["miss"],
                        "dx": held["dx"],
                        "dy": held["dy"],
                        "end": held["end"],
                    },
                }
                rows.append(row)
                print(
                    f"WESTSCAN dy={dy} LEFT dx={held['dx']} miss={held['miss']} "
                    f"end=({held['end']['x']},{held['end']['y']}) sub={held['end']['submodule']}"
                )
                if held["dx"] <= -48 and held["end"]["room"] == 0x82 and not (
                    held["miss"] and str(held["miss"]).startswith("pit")
                ):
                    found = found or row
            log["west_scan"] = {"rows": rows, "found": found}
            (out / "probe.json").write_text(json.dumps(log, indent=2) + "\n")
            return 0 if found else 2

        claims = CLAIMS
        if args.east_wall:
            claims = (
                ("floor_south", 1198, 4160),
                ("east_wall", 1312, 4160),
                ("east_mid", 1312, 4336),
                ("south_area", 1312, 4456),
            )
            log["claims"] = [list(c) for c in claims]
        elif args.east_from_bridge:
            # Hug east wall (x>=1312) before dropping to the y=4492 south walkway.
            # x=1310 at y=4388 then south pits; CastleZeldaB1East RIGHT walks
            # y=4492 from x=1036 to x=1360.
            claims = (
                ("floor_south", 1198, 4160),
                ("east_lane", 1272, 4160),
                ("island_y", 1272, 4253),
                ("corridor_x", 1255, 4253),
                ("bridge", 1255, 4393),
                ("east_hug", 1324, 4393),
                ("south_walk", 1324, 4492),
                ("east_b1", 1036, 4492),
                ("west_door", 1010, 4496),
            )
            log["claims"] = [list(c) for c in claims]

        if not args.no_fight:
            fight = fight_nearby(env, room=0x82, max_distance=80, max_cycles=80)
            settle_control(env)
            log["fight"] = {
                "ok": fight.ok,
                "reason": fight.reason,
                "frames": fight.frames,
                "after": glance(env),
            }
            print(f"fight {fight.reason} now=({log['fight']['after']['x']},{log['fight']['after']['y']})")
            hit = pit_or_dead(snapshot_env(env))
            if hit or log["fight"]["after"]["room"] != 0x82:
                log["halt"] = hit or "left_room_during_fight"
                save_overlay(env, out / "fight_halt.png", "fight halt")
                (out / "probe.json").write_text(json.dumps(log, indent=2) + "\n")
                print(f"HALT fight {log['halt']}")
                return 2

        steps = []
        halted = None
        for label, x, y in claims:
            rec = axis_claim(env, label, x, y)
            steps.append(rec)
            print(
                f"CLAIM {label} ({x},{y}) ok={rec['ok']} miss={rec['miss']} "
                f"now=({rec['end']['x']},{rec['end']['y']}) "
                f"room={rec['end']['room_hex']} sub={rec['end']['submodule']}"
            )
            save_overlay(env, out / f"claim_{label}.png", f"{label} {x},{y}")
            if not rec["ok"]:
                halted = rec
                break
        log["steps"] = steps
        log["halt"] = None if halted is None else {
            "label": halted["label"],
            "miss": halted["miss"],
            "end": halted["end"],
        }

        snap = snapshot_env(env)
        if snap.room_base_id == 0x81:
            settle_control(env, max_frames=240)
            push = glance(env)
            log["west_push"] = {"ok": True, "via": "path_crossed", "end": push}
            log["halt"] = None
            log["final"] = push
            save_overlay(env, out / "west_push.png", "0x81 after 0x82 west door")
            print(
                f"WEST_PUSH ok=True room={push['room_hex']} "
                f"xy=({push['x']},{push['y']}) F3CC={push['follower']} "
                f"keys={push['keys']} hp={push['bytes']['F36D_hp']} "
                f"sub={push['submodule']}"
            )
            (out / "probe.json").write_text(json.dumps(log, indent=2) + "\n")
            (out / "leftover.json").write_text(
                json.dumps(
                    {
                        "ok": True,
                        "chained_0x72_south_to_0x81": True,
                        "leftover": push,
                    },
                    indent=2,
                )
                + "\n"
            )
            return 0 if push["room"] == 0x81 and push["module"] == 0x07 else 2
        if snap.room_base_id == 0x82 and snap.has_control and halted is None:
            # Push LEFT through west door.
            before = glance(env)
            frames = 0
            dest = None
            while frames < 180:
                step_frames(env, action_for("LEFT"), 4)
                frames += 4
                s = snapshot_env(env)
                if s.room_base_id == 0x81 and s.has_control:
                    dest = glance(env)
                    break
                hit = pit_or_dead(s)
                if hit:
                    dest = glance(env)
                    log["halt"] = {"label": "west_push", "miss": hit, "end": dest}
                    break
            settle_control(env, max_frames=180)
            push = glance(env)
            log["west_push"] = {
                "frames": frames,
                "before": before,
                "end": push,
                "ok": push["room"] == 0x81,
            }
            print(
                f"WEST_PUSH ok={push['room']==0x81} room={push['room_hex']} "
                f"xy=({push['x']},{push['y']}) F3CC={push['follower']} "
                f"keys={push['keys']} sub={push['submodule']}"
            )
            save_overlay(env, out / "west_push.png", "after west push")

        log["final"] = glance(env)
        (out / "probe.json").write_text(json.dumps(log, indent=2) + "\n")
        print(f"wrote {out / 'probe.json'} halt={log.get('halt')}")
        return 0 if log.get("halt") is None and log["final"]["room"] == 0x81 else 2
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())

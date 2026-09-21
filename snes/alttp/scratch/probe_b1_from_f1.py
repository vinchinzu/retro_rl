"""Probe F1 rooms 0x01/0x51/0x52/0x62 for a stair/door into B1 (0x70–0x82).

State-load only. Known map doors are recorded; new B1 edges halt the room.

Usage:
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_b1_from_f1.py
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_b1_from_f1.py --down01
    SDL_VIDEODRIVER=dummy uv run python snes/alttp/scratch/probe_b1_from_f1.py --down01 --from50
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

from alttp.opening_route.room_engine import clear_room
from alttp.paths import RECORDINGS_DIR
from alttp.primitives import Waypoint, fight_nearby, move_to, settle_control
from alttp.ram import EQUIP_SWORD, FOLLOWER, NUM_KEYS, snapshot_to_diag, wram_index
from alttp.room_sense import detect_edge, load_room_map, overlay_from_env
from alttp.startup import action_for, build_boot_env, no_action, snapshot_env, step_frames

OUT_DIR = RECORDINGS_DIR / "probe_b1_from_f1"
B1_LO = 0x70
B1_HI = 0x82
CARDINALS = ("UP", "DOWN", "LEFT", "RIGHT")
ROOMS: tuple[dict[str, Any], ...] = (
    {"map_id": "room_01", "state": "CastleRoom01", "hex": "0x01"},
    {"map_id": "room_51", "state": "CastleRoom51", "hex": "0x51"},
    {"map_id": "room_52", "state": "CastleRoom52", "hex": "0x52"},
    {"map_id": "room_62", "state": "CastleRoom62", "hex": "0x62"},
)
# First pass: south/east of measured 0x62 bbox. Follow-up: corridor x then DOWN
# onto the stairwell visible SE of CastleRoom62 spawn (not reachable by pure DOWN).
EXTRAS: dict[str, tuple[tuple[str, int, int], ...]] = {
    "room_62": (
        ("south_of_spawn", 1072, 3500),
        ("south_of_north_area", 1272, 3500),
        ("east_of_extent", 1456, 3320),
    ),
}
FOLLOWUP_62: tuple[tuple[str, int, int], ...] = (
    ("corr_1144", 1144, 3320),
    ("corr_1200", 1200, 3320),
    ("corr_1248", 1248, 3320),
    ("corr_1280", 1280, 3320),
    ("corr_1320", 1320, 3320),
    ("se_1200", 1200, 3400),
    ("se_1248", 1248, 3400),
    ("se_1280", 1280, 3400),
)
# East-wall stair graphic at CastleRoom62North; RIGHT ray ended (1352, 3120) stuck.
NORTH_62: tuple[tuple[str, int, int], ...] = (
    ("east_ray_end", 1352, 3120),
    ("stair_north", 1352, 3080),
    ("stair_south", 1352, 3168),
    ("far_east", 1440, 3120),
    ("south_1216", 1216, 3320),
)


def leftover(env: object) -> dict[str, Any]:
    snap = snapshot_env(env)
    ram = env.get_ram()  # type: ignore[attr-defined]
    diag = snapshot_to_diag(snap)
    return {
        "room_hex": diag["room_hex"],
        "module": int(snap.game_mode),
        "submodule": int(snap.submodule),
        "x": int(snap.link_x),
        "y": int(snap.link_y),
        "sword": int(snap.sword_level),
        "F359_sword": int(ram[wram_index(EQUIP_SWORD)]),
        "F3CC_follower": int(ram[wram_index(FOLLOWER)]),
        "F36F_keys": int(ram[wram_index(NUM_KEYS)]),
        "keys": int(snap.num_keys),
        "indoors": bool(snap.indoors),
        "has_control": bool(snap.has_control),
        "diag": diag,
    }


def is_b1(room: int) -> bool:
    return B1_LO <= int(room) <= B1_HI


def pin_state(env: object) -> bytes:
    return env.em.get_state()  # type: ignore[attr-defined]


def restore(env: object, blob: bytes) -> None:
    env.em.set_state(blob)  # type: ignore[attr-defined]
    step_frames(env, no_action(), 1)


def ray(env: object, room: int, direction: str, *, max_frames: int = 360) -> dict[str, Any]:
    start = snapshot_env(env)
    rec: dict[str, Any] = {
        "dir": direction,
        "fromXy": [start.link_x, start.link_y],
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
            expected_room=room,
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
                    "b1": (not edge.outdoors) and is_b1(edge.to_room),
                    "approachXy": list(edge.from_xy),
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


def origins_for(room_map: Any, spawn_xy: tuple[int, int], extras: tuple[tuple[str, int, int], ...]) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = [("spawn", spawn_xy[0], spawn_xy[1])]
    seen = {spawn_xy}
    for pt in room_map.points:
        xy = (pt.x, pt.y)
        if xy in seen:
            continue
        if pt.role in {"approach", "waypoint", "spawn"}:
            out.append((pt.label, pt.x, pt.y))
            seen.add(xy)
    for label, x, y in extras:
        if (x, y) not in seen:
            out.append((label, x, y))
            seen.add((x, y))
    return out[:8]


def save_overlay(env: object, path: Path, title: str, points: Any) -> None:
    img = overlay_from_env(env, include_enemies=True, points=points, title=title)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


def rays_from_here(
    env: object,
    room: int,
    origin: str,
    report: dict[str, Any],
    points: Any,
) -> bool:
    """Independent cardinals. True if a B1 edge was recorded."""
    origin_xy = (snapshot_env(env).link_x, snapshot_env(env).link_y)
    origin_pin = pin_state(env)
    for direction in CARDINALS:
        restore(env, origin_pin)
        hit = ray(env, room, direction)
        hit["origin"] = origin
        hit["originXy"] = list(origin_xy)
        report["rays"].append(hit)
        if hit.get("toRoom"):
            report["exits"].append(hit)
        if hit.get("b1"):
            report["b1"] = hit
            save_overlay(
                env,
                OUT_DIR / f"{report['mapId']}_b1_{direction}.png",
                f"{report['roomHex']} {direction} → {hit['toRoom']}",
                points,
            )
            report["b1Leftover"] = leftover(env)
            return True
    restore(env, origin_pin)
    return False


def probe_room(spec: dict[str, Any]) -> dict[str, Any]:
    room_map = load_room_map(spec["map_id"])
    room = room_map.room_base_id
    env = build_boot_env(spec["state"])
    report: dict[str, Any] = {
        "mapId": spec["map_id"],
        "state": spec["state"],
        "roomHex": spec["hex"],
        "knownDoors": [
            {
                "label": d.label,
                "dir": d.direction,
                "to": f"0x{d.to_room:02X}" if d.to_room is not None else "outdoors",
            }
            for d in room_map.doors
        ],
        "exits": [],
        "misses": [],
        "rays": [],
        "b1": None,
    }
    try:
        env.reset()  # type: ignore[attr-defined]
        settle = settle_control(env)
        report["spawnLeftover"] = leftover(env)
        save_overlay(
            env,
            OUT_DIR / f"{spec['map_id']}_spawn.png",
            f"{spec['hex']} spawn",
            room_map.points,
        )
        if not settle.ok or snapshot_env(env).room_base_id != room:
            report["blocker"] = "spawn did not settle in expected room"
            report["afterClearLeftover"] = report["spawnLeftover"]
            return report

        no_clear = bool(spec.get("no_clear"))
        skip_extra = bool(spec.get("skip_extra_fight"))
        if no_clear:
            report["clear"] = {"ok": True, "detail": "skipped", "frames": 0}
        else:
            cleared = clear_room(env, room_map)
            extra = fight_nearby(
                env,
                room=room,
                max_distance=80 if skip_extra else 220,
                max_cycles=80 if skip_extra else 250,
            )
            extra_reason, extra_frames = extra.reason, extra.frames
            report["clear"] = {
                "ok": cleared.ok,
                "detail": cleared.detail,
                "frames": cleared.frames,
                "extraFight": extra_reason,
                "extraFrames": extra_frames,
            }
            settle_control(env, max_frames=120)
        report["afterClearLeftover"] = leftover(env)
        after = snapshot_env(env)
        if after.game_mode == 0x12:
            report["blocker"] = "died during clear"
            return report
        if after.room_base_id != room:
            report["blocker"] = "left room during clear"
            return report
        if not no_clear and not report["clear"]["ok"] and report["clear"].get("extraFight") not in {
            "no nearby targets",
            "combat acceptance reached",
            "skipped",
        }:
            report["blocker"] = report["clear"]["detail"] or "clear failed"
            return report

        post = pin_state(env)
        spawn = snapshot_env(env)
        if spec.get("north62"):
            origin_list = [("spawn", spawn.link_x, spawn.link_y), *NORTH_62]
        elif spec.get("followup") and spec["map_id"] == "room_62":
            origin_list = [("spawn", spawn.link_x, spawn.link_y), *FOLLOWUP_62]
        else:
            extras = EXTRAS.get(spec["map_id"], ())
            origin_list = origins_for(room_map, (spawn.link_x, spawn.link_y), extras)
        for label, x, y in origin_list:
            restore(env, post)
            if abs(snapshot_env(env).link_x - x) > 8 or abs(snapshot_env(env).link_y - y) > 8:
                walk = move_to(
                    env,
                    Waypoint(x, y, tolerance=12, room=room, label=label),
                    max_frames=500,
                )
                if not walk.ok:
                    report["misses"].append(
                        {
                            "origin": label,
                            "target": [x, y],
                            "reason": walk.reason,
                            "xy": [walk.snapshot.link_x, walk.snapshot.link_y],
                            "room": f"0x{walk.snapshot.room_base_id:02X}",
                        }
                    )
                    continue
            if rays_from_here(env, room, label, report, room_map.points):
                return report
        report["finalLeftover"] = leftover(env)
        restore(env, post)
        save_overlay(
            env,
            OUT_DIR / f"{spec['map_id']}_cleared.png",
            f"{spec['hex']} cleared",
            room_map.points,
        )
    finally:
        env.close()  # type: ignore[attr-defined]
    return report


def probe_down01(*, from50: bool = False) -> dict[str, Any]:
    """CastleRoom01 (or 0x50 east) → (760,99) → DOWN → 0x72. Isolated only."""
    from alttp.opening_route.room_engine import run_room_edge
    from alttp.room_sense import load_room_map

    room_map = load_room_map("room_01")
    state = "CastleRoom50" if from50 else "CastleRoom01"
    env = build_boot_env(state)
    rec: dict[str, Any] = {
        "state": state,
        "from50": from50,
        "walks": [],
        "down": None,
        "b1": False,
    }
    try:
        env.reset()  # type: ignore[attr-defined]
        settle_control(env)
        rec["spawnLeftover"] = leftover(env)
        save_overlay(env, OUT_DIR / f"down01_{state}_spawn.png", "down01 spawn", room_map.points)
        if from50:
            edge = run_room_edge(
                env, "room_50", "east_to_0x01", clear=True, source="state_load_dev"
            )
            rec["eastTo01"] = {
                "ok": edge.ok,
                "phase": edge.phase,
                "frames": edge.frames,
                "blocker": edge.blocker,
                "leftover": leftover(env),
            }
            if snapshot_env(env).room_base_id != 1:
                rec["blocker"] = "0x50 east did not land in 0x01"
                return rec
        snap = snapshot_env(env)
        if snap.room_base_id != 1:
            rec["blocker"] = f"not in 0x01 (0x{snap.room_base_id:02X})"
            return rec
        # Corridor y=120 is open west through x=760; y=104 hits a wall at x=832.
        # Tight tolerance: ±8 stopped at (766,106), south of the (760,99) landing.
        for label, x, y, tol in (
            ("west_corridor", 760, 120, 4),
            ("stair_approach", 760, 99, 2),
        ):
            walk = move_to(
                env,
                Waypoint(x, y, tolerance=tol, room=1, label=label),
                max_frames=700,
            )
            rec["walks"].append(
                {
                    "label": label,
                    "ok": walk.ok,
                    "reason": walk.reason,
                    "frames": walk.frames,
                    "leftover": leftover(env),
                }
            )
            if snapshot_env(env).room_base_id != 1:
                rec["blocker"] = f"left 0x01 during {label}"
                return rec
            if not walk.ok:
                rec["blocker"] = walk.reason
                return rec
        save_overlay(
            env,
            OUT_DIR / f"down01_{state}_approach.png",
            "down01 approach",
            room_map.points,
        )
        rec["approachLeftover"] = leftover(env)
        approach = snapshot_env(env)
        approach_pin = pin_state(env)
        down = ray(env, 1, "DOWN", max_frames=480)
        rec["down"] = down
        hit = down if (down.get("b1") or down.get("toRoom") == "0x72") else None
        if hit is None:
            restore(env, approach_pin)
            up = ray(env, 1, "UP", max_frames=240)
            rec["up"] = up
            if up.get("b1") or up.get("toRoom") == "0x72":
                hit = up
        if hit is not None:
            rec["b1"] = True
            settle_control(env, max_frames=400)
            rec["destLeftover"] = leftover(env)
            save_overlay(
                env,
                OUT_DIR / f"down01_{state}_dest.png",
                f"down01 dest {rec['destLeftover']['room_hex']}",
                room_map.points,
            )
        else:
            rec["finalLeftover"] = leftover(env)
            rec["blocker"] = (
                f"DOWN/UP from ({approach.link_x},{approach.link_y}) "
                f"did not reach 0x72: down={down}"
            )
    finally:
        env.close()  # type: ignore[attr-defined]
    return rec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", default="", help="Limit to one hex, e.g. 0x62")
    parser.add_argument(
        "--followup",
        action="store_true",
        help="0x62 corridor-east then south + CastleRoom62North; 0x51 no-clear",
    )
    parser.add_argument(
        "--north62",
        action="store_true",
        help="CastleRoom62North fight + east-wall stair approaches",
    )
    parser.add_argument(
        "--down01",
        action="store_true",
        help="Walk 0x01 to (760,99) and hold DOWN toward 0x72",
    )
    parser.add_argument(
        "--from50",
        action="store_true",
        help="With --down01: start at CastleRoom50 east leftover",
    )
    args = parser.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.down01:
        rec = probe_down01(from50=bool(args.from50))
        tag = "down01_from50" if args.from50 else "down01_CastleRoom01"
        path = OUT_DIR / f"{tag}.json"
        path.write_text(json.dumps(rec, indent=2))
        print(f"Wrote {path}", flush=True)
        dest = rec.get("destLeftover") or rec.get("approachLeftover") or rec.get("spawnLeftover")
        print(f"b1={rec.get('b1')} leftover={dest} blocker={rec.get('blocker')!r}", flush=True)
        return 0 if rec.get("b1") else 1
    if args.north62:
        jobs: list[dict[str, Any]] = [
            {
                "map_id": "room_62",
                "state": "CastleRoom62North",
                "hex": "0x62",
                "skip_extra_fight": False,
                "north62": True,
            },
        ]
        summary_path = OUT_DIR / "north62.json"
    elif args.followup:
        jobs: list[dict[str, Any]] = [
            {
                "map_id": "room_62",
                "state": "CastleRoom62",
                "hex": "0x62",
                "skip_extra_fight": True,
                "followup": True,
            },
            {
                "map_id": "room_62",
                "state": "CastleRoom62North",
                "hex": "0x62",
                "skip_extra_fight": True,
                "followup": True,
            },
            {
                "map_id": "room_51",
                "state": "CastleRoom51",
                "hex": "0x51",
                "no_clear": True,
            },
        ]
        summary_path = OUT_DIR / "followup.json"
    else:
        wanted = args.room.lower()
        jobs = [r for r in ROOMS if not wanted or r["hex"].lower() == wanted]
        summary_path = OUT_DIR / "summary.json"
        if not jobs:
            print(f"unknown --room {args.room!r}", flush=True)
            return 2
    bundle: dict[str, Any] = {
        "when": datetime.now(timezone.utc).isoformat(),
        "followup": bool(args.followup),
        "rooms": [],
        "b1Found": False,
    }
    for spec in jobs:
        print(f"probing {spec['hex']} from {spec['state']}", flush=True)
        rec = probe_room(spec)
        bundle["rooms"].append(rec)
        if args.north62:
            tag = "CastleRoom62North_stairs"
        elif args.followup:
            tag = spec["state"]
        else:
            tag = spec["map_id"]
        path = OUT_DIR / f"{tag}.json"
        path.write_text(json.dumps(rec, indent=2))
        print(f"Wrote {path}", flush=True)
        if rec.get("b1"):
            bundle["b1Found"] = True
            print(f"B1 hit: {rec['b1']}", flush=True)
            break
        exits = rec.get("exits") or []
        print(
            f"  leftover {rec.get('afterClearLeftover')} "
            f"exits={len(exits)} misses={len(rec.get('misses') or [])}",
            flush=True,
        )
    summary_path.write_text(json.dumps(bundle, indent=2))
    print(f"Wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Probe Level 3 (Manji) overworld door + entry room (assisted recon).

Live (2026-08-06): door **0x74**, entry room **0x7c**. Source path via 0x67
is blocked. Prefer ``--from-state OW_66`` or ``Level3Entrance`` for stability.

Examples::

    uv run python nes/zelda_i/scripts/probe_level3_entry.py --infinite-life --save-state
    uv run python nes/zelda_i/scripts/probe_level3_entry.py --infinite-life \\
        --from-state OW_66 --tag l3_recon
    uv run python nes/zelda_i/scripts/probe_level3_entry.py --from-state Level3Entrance \\
        --map-only --tag l3_map
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NES = Path(__file__).resolve().parents[2]
for p in (_REPO_ROOT, _NES):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from retro_harness.env import make_env, save_state
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ids import object_name, room_item_name
from zelda_i.level3_overworld import (
    LEVEL3,
    LEVEL3_DOOR_HOPS_FROM_66,
    LEVEL3_PATH_HOPS,
    LEVEL3_PATH_SCREENS,
    SEGMENT_MAX_FRAMES,
    SCREEN_LEVEL3_ENTRANCE,
    SCREEN_LEVEL3_ENTRY_ROOM,
    OverworldToLevel3Controller,
    level3_entrance_success,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot, read_snapshot
from zelda_i.sword_cave import SEGMENT_MAX_FRAMES as SWORD_MAX, SwordCaveController

DOOR_RIGHT = 0x01
DOOR_LEFT = 0x02
DOOR_DOWN = 0x04
DOOR_UP = 0x08
PROBE_DIRS = ("LEFT", "UP", "RIGHT", "DOWN")


def _snapshot_dict(snap: ZeldaSnapshot) -> dict:
    objs = [
        {
            "slot": o.slot,
            "type": o.type_id,
            "type_name": object_name(o.type_id),
            "x": o.x,
            "y": o.y,
            "hp": o.hp,
        }
        for o in snap.objects
        if o.slot >= 1 and not (o.type_id in (0, 0xFF) and o.y == 0)
    ][:16]
    return {
        "mode": snap.mode,
        "level": snap.level,
        "screen": snap.screen,
        "x": snap.link_x,
        "y": snap.link_y,
        "health": snap.health,
        "hearts": f"{snap.filled_hearts}/{snap.heart_containers}",
        "sword": snap.sword,
        "bombs": snap.bombs,
        "keys": snap.keys,
        "triforce": snap.triforce,
        "room_item_id": snap.room_item_id,
        "room_item_name": room_item_name(snap.room_item_id),
        "room_all_dead": snap.room_all_dead,
        "cur_opened_doors": snap.cur_opened_doors,
        "cur_opened_doors_bits": {
            "R": bool(snap.cur_opened_doors & DOOR_RIGHT),
            "L": bool(snap.cur_opened_doors & DOOR_LEFT),
            "D": bool(snap.cur_opened_doors & DOOR_DOWN),
            "U": bool(snap.cur_opened_doors & DOOR_UP),
            "raw": snap.cur_opened_doors,
        },
        "open_doorway_mask": snap.open_doorway_mask,
        "objects": objs,
        "type_counts": dict(Counter(o["type"] for o in objs)),
    }


def _ensure_sword(env, obs, assist, tag: str):
    snap = read_snapshot(env.get_ram())
    if snap.has_sword:
        return obs, None
    ctrl = SwordCaveController()
    for f in range(SWORD_MAX):
        snap = read_snapshot(env.get_ram())
        act = ctrl.step(snap)
        obs, *_ = env.step(act.action)
        if assist is not None:
            assist.apply_env(env, frame=f + 1)
        if ctrl.success or ctrl.phase.name == "FAILED":
            break
    snap = read_snapshot(env.get_ram())
    return obs, {
        "success": bool(ctrl.success and snap.has_sword),
        "controller": ctrl.report(),
        "final": _snapshot_dict(snap),
    }


def _door_explore(env, obs, *, assist, max_frames: int, door_x: int, tag: str):
    """Tour 0x74 rock maze + UP hunt (first live enter used this style)."""
    trail: list[dict] = []
    wps = [
        (40, 140),
        (80, 140),
        (120, 140),
        (160, 140),
        (200, 140),
        (200, 120),
        (200, 100),
        (180, 100),
        (160, 100),
        (140, 100),
        (120, 100),
        (100, 100),
        (80, 100),
        (100, 160),
        (140, 160),
        (180, 160),
        (160, 80),
        (120, 80),
        (180, 80),
        (200, 80),
    ]
    for f in range(max_frames):
        snap = read_snapshot(env.get_ram())
        if snap.level == LEVEL3 and snap.mode == PLAY_MODE:
            trail.append({"f": f, "event": "entered", **_snapshot_dict(snap)})
            break
        if snap.mode == 17:
            break
        if snap.level == 0 and snap.screen != SCREEN_LEVEL3_ENTRANCE:
            trail.append({"f": f, "event": "left_door_screen", **_snapshot_dict(snap)})
            # try return
            btn = "RIGHT" if snap.screen == 0x73 else "LEFT"
            obs, *_ = env.step(nes_action(btn))
            if assist is not None:
                assist.apply_env(env, frame=f)
            continue
        if snap.transitioning or snap.mode not in (PLAY_MODE, 8, 11, 16, 2, 3, 4):
            obs, *_ = env.step(
                nes_idle_action() if snap.level == LEVEL3 else nes_action("UP")
            )
            if assist is not None:
                assist.apply_env(env, frame=f)
            continue
        if f % 2000 < 1200:
            wp = wps[(f // 80) % len(wps)]
            dx, dy = wp[0] - snap.link_x, wp[1] - snap.link_y
            if abs(dx) > 6 and abs(dx) >= abs(dy):
                btn = "RIGHT" if dx > 0 else "LEFT"
            elif abs(dy) > 6:
                btn = "DOWN" if dy > 0 else "UP"
            else:
                btn = "UP"
        else:
            xs = list(range(40, 220, 12))
            tx = xs[(f // 90) % len(xs)]
            ty = 100 + ((f // 90) % 3) * 20
            if abs(snap.link_y - ty) > 6:
                btn = "UP" if snap.link_y > ty else "DOWN"
            elif abs(snap.link_x - tx) > 4:
                btn = "LEFT" if snap.link_x > tx else "RIGHT"
            else:
                btn = "UP"
        act = nes_action(btn, "A") if f % 10 < 3 else nes_action(btn)
        obs, *_ = env.step(act)
        if assist is not None:
            assist.apply_env(env, frame=f)
        if f % 120 == 0:
            trail.append({"f": f, "event": "explore", **_snapshot_dict(snap)})
    snap = read_snapshot(env.get_ram())
    save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_after_explore.png")
    return obs, trail, {
        "entered": snap.level == LEVEL3 and snap.mode == PLAY_MODE,
        "final": _snapshot_dict(snap),
    }


def _map_rooms(env, obs, *, assist, entry_room: int, tag: str) -> tuple[object, list[dict]]:
    """Sample entry + try N/E/S/W without reopening the emulator (reload can SIGSEGV)."""
    probes: list[dict] = []
    idle_log: list[dict] = []
    for f in range(250):
        snap = read_snapshot(env.get_ram())
        if f in (0, 50, 100, 150, 249):
            idle_log.append({"f": f, **_snapshot_dict(snap)})
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=f)
    save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_entry_idle.png")
    probes.append(
        {
            "room": "entry",
            "screen": entry_room,
            "idle_samples": idle_log,
            "final": _snapshot_dict(read_snapshot(env.get_ram())),
            "visual": "Keese pack, west doorway open, south mouth (see screenshots)",
        }
    )

    for d in PROBE_DIRS:
        # Nudge back toward entry if we left (best-effort; no env reload).
        for rf in range(400):
            snap = read_snapshot(env.get_ram())
            if snap.level == LEVEL3 and snap.screen == entry_room:
                break
            if snap.level == 0:
                # re-enter from OW if we dropped south
                btn = "UP"
            else:
                btn = "DOWN"
            obs, *_ = env.step(nes_action(btn))
            if assist is not None:
                assist.apply_env(env, frame=rf)

        start = read_snapshot(env.get_ram()).screen
        arrived = None
        for f in range(500):
            snap = read_snapshot(env.get_ram())
            if (
                snap.level == LEVEL3
                and snap.mode == PLAY_MODE
                and snap.screen != start
                and not snap.transitioning
            ):
                for _ in range(220):
                    obs, *_ = env.step(nes_idle_action())
                    if assist is not None:
                        assist.apply_env(env, frame=f)
                snap = read_snapshot(env.get_ram())
                arrived = _snapshot_dict(snap)
                save_rgb_png(
                    obs, RECORDINGS_DIR / f"{tag}_{d}_0x{snap.screen:02x}.png"
                )
                break
            if d in ("UP", "DOWN") and abs(snap.link_x - 128) > 10:
                btn = "LEFT" if snap.link_x > 128 else "RIGHT"
            elif d in ("LEFT", "RIGHT") and abs(snap.link_y - 157) > 12:
                btn = "UP" if snap.link_y > 157 else "DOWN"
            else:
                btn = d
            if d in ("LEFT", "RIGHT") and snap.link_y > 190:
                btn = "UP"
            act = nes_action(btn, "A") if f % 10 < 3 else nes_action(btn)
            obs, *_ = env.step(act)
            if assist is not None:
                assist.apply_env(env, frame=f)
        probes.append(
            {
                "direction": d,
                "from_screen": start,
                "arrived": arrived,
                "ok": arrived is not None,
                "final": _snapshot_dict(read_snapshot(env.get_ram())),
            }
        )
    return obs, probes


def run_probe(
    *,
    start_state: str,
    infinite_life: bool,
    enter_dungeon: bool,
    door_only: bool,
    map_only: bool,
    save_checkpoint: bool,
    map_rooms: bool,
    max_frames: int,
    door_x: int,
    tag: str,
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    track = "assisted" if infinite_life else "clean"
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)

        entry = _snapshot_dict(read_snapshot(env.get_ram()))
        sword_report = None

        if map_only or (
            entry["level"] == LEVEL3 and entry["screen"] == SCREEN_LEVEL3_ENTRY_ROOM
        ):
            snap = read_snapshot(env.get_ram())
            if not (
                snap.level == LEVEL3
                and snap.mode == PLAY_MODE
                and snap.screen == SCREEN_LEVEL3_ENTRY_ROOM
            ):
                return {
                    "ok": False,
                    "track": track,
                    "error": "map_only requires Level3Entrance / room 0x7c",
                    "entry": entry,
                }
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_entry.png")
            obs, room_probes = _map_rooms(
                env, obs, assist=assist, entry_room=snap.screen, tag=tag
            )
            return {
                "ok": True,
                "track": track,
                "door_screen": f"0x{SCREEN_LEVEL3_ENTRANCE:02x}",
                "entry_room": f"0x{SCREEN_LEVEL3_ENTRY_ROOM:02x}",
                "entry": entry,
                "room_probes": room_probes,
                "assist": assist.report() if assist else None,
                "final": _snapshot_dict(read_snapshot(env.get_ram())),
            }

        obs, sword_report = _ensure_sword(env, obs, assist, tag)
        snap = read_snapshot(env.get_ram())
        if not snap.has_sword:
            return {
                "ok": False,
                "track": track,
                "stage": "sword",
                "entry": entry,
                "sword": sword_report,
                "final": _snapshot_dict(snap),
            }

        # Choose hop table by start screen
        if snap.screen == 0x66 or start_state == "OW_66":
            hops = LEVEL3_DOOR_HOPS_FROM_66
        else:
            hops = LEVEL3_PATH_HOPS

        trail: list[dict] = []
        nav = OverworldToLevel3Controller(
            hops=hops,
            require_level3_screen=door_only,
            require_dungeon=enter_dungeon and not door_only,
            door_x=door_x,
        )
        if door_only:
            nav.require_dungeon = False
            nav.require_level3_screen = True
        elif enter_dungeon or map_rooms or save_checkpoint:
            nav.require_dungeon = True

        frames = 0
        last_screen = snap.screen
        while frames < max_frames:
            snap = read_snapshot(env.get_ram())
            if snap.screen != last_screen:
                trail.append({"f": frames, **_snapshot_dict(snap)})
                last_screen = snap.screen
                save_rgb_png(
                    obs,
                    RECORDINGS_DIR / f"{tag}_sc{snap.level}_{snap.screen:02x}.png",
                )
            if snap.mode == 17:
                break
            if nav.success or nav.phase.name == "FAILED":
                break
            # If hops done on door screen but not entered, fall through to explore
            if (
                nav.hop_index >= len(nav.hops)
                and nav.require_dungeon
                and snap.level == 0
                and snap.screen == SCREEN_LEVEL3_ENTRANCE
                and nav.phase_frames > 500
            ):
                break
            act = nav.step(snap)
            obs, *_ = env.step(act.action)
            frames += 1
            if assist is not None:
                assist.apply_env(env, frame=frames)

        snap = read_snapshot(env.get_ram())
        nav_report = nav.report()
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_after_nav.png")

        explore_report = None
        if (enter_dungeon or map_rooms or save_checkpoint) and not (
            snap.level == LEVEL3 and snap.mode == PLAY_MODE
        ):
            if snap.level == 0 and snap.screen == SCREEN_LEVEL3_ENTRANCE:
                obs, explore_trail, explore_report = _door_explore(
                    env,
                    obs,
                    assist=assist,
                    max_frames=12000,
                    door_x=door_x,
                    tag=tag,
                )
                trail.extend(explore_trail)
                snap = read_snapshot(env.get_ram())

        entered = snap.level == LEVEL3 and snap.mode == PLAY_MODE
        entry_room = snap.screen if entered else None
        entry_fields = _snapshot_dict(snap) if entered else None

        state_path = None
        if entered and save_checkpoint:
            for sf in range(200):
                obs, *_ = env.step(nes_idle_action())
                if assist is not None:
                    assist.apply_env(env, frame=frames + sf)
            snap = read_snapshot(env.get_ram())
            entry_room = snap.screen
            entry_fields = _snapshot_dict(snap)
            path = save_state(env, GAME_DIR, GAME, "Level3Entrance")
            state_path = str(path)
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_entrance.png")

        room_probes: list[dict] = []
        if entered and map_rooms and entry_room is not None:
            obs, room_probes = _map_rooms(
                env, obs, assist=assist, entry_room=entry_room, tag=tag
            )

        door_reached = (
            snap.level == 0 and snap.screen == SCREEN_LEVEL3_ENTRANCE
        ) or entered
        ok = (
            entered
            if (enter_dungeon or map_rooms or save_checkpoint)
            else door_reached
        )
        if door_only:
            ok = door_reached

        return {
            "ok": ok,
            "track": track,
            "path_screens": [f"0x{s:02x}" for s in LEVEL3_PATH_SCREENS],
            "door_screen": f"0x{SCREEN_LEVEL3_ENTRANCE:02x}",
            "entry_room_expected": f"0x{SCREEN_LEVEL3_ENTRY_ROOM:02x}",
            "entry": entry,
            "sword": sword_report,
            "nav": nav_report,
            "explore": explore_report,
            "trail": trail[-50:],
            "entered_level3": entered,
            "entry_room": f"0x{entry_room:02x}" if entry_room is not None else None,
            "entry_snapshot": entry_fields,
            "state_path": state_path,
            "room_probes": room_probes,
            "assist": assist.report() if assist else None,
            "final": _snapshot_dict(read_snapshot(env.get_ram())),
        }
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-state",
        default="OW_66",
        help="Integration save (default OW_66 near door path; or Level3Entrance)",
    )
    parser.add_argument("--infinite-life", action="store_true")
    parser.add_argument("--enter-dungeon", action="store_true")
    parser.add_argument("--door-only", action="store_true")
    parser.add_argument(
        "--map-only",
        action="store_true",
        help="Only map rooms from Level3Entrance (no OW walk)",
    )
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--map-rooms", action="store_true")
    parser.add_argument("--max-frames", type=int, default=SEGMENT_MAX_FRAMES)
    parser.add_argument("--door-x", type=int, default=128)
    parser.add_argument("--tag", default="l3_recon")
    args = parser.parse_args(argv)

    enter = args.enter_dungeon or args.save_state or args.map_rooms
    if not args.door_only and not args.map_only and not enter:
        enter = True
        args.save_state = True
        args.map_rooms = True

    rep = run_probe(
        start_state=args.from_state,
        infinite_life=args.infinite_life or args.map_only,
        enter_dungeon=enter,
        door_only=args.door_only,
        map_only=args.map_only,
        save_checkpoint=args.save_state,
        map_rooms=args.map_rooms or args.save_state,
        max_frames=args.max_frames,
        door_x=args.door_x,
        tag=args.tag,
    )
    out = RECORDINGS_DIR / f"{args.tag}_report.json"
    write_json_report(out, rep)
    final = rep.get("final") or {}
    print(
        f"ok={rep['ok']} track={rep['track']} "
        f"entered={rep.get('entered_level3')} "
        f"entry_room={rep.get('entry_room')} "
        f"final_lv={final.get('level')} sc=0x{final.get('screen', 0):02x} "
        f"nav_phase={(rep.get('nav') or {}).get('phase')} "
        f"hops={(rep.get('nav') or {}).get('hop_index')}"
    )
    print(f"wrote {out}")
    if rep.get("state_path"):
        print(f"state {rep['state_path']}")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

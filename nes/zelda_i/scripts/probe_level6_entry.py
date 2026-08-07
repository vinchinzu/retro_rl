"""Assisted recon: Level 6 (Dragon) OW door + entry rooms.

Live geometry (2026-08-06)::

    OW door **0x22**, enter UP @ x≈24–56 → dungeon room **0x79**.
    RIGHT wall-first (y≈157 → y≈138) → **0x7a** (5× type 0x24 + key 0x19).

Examples::

    # From OW door fixture, enter dungeon, save Level6Entrance
    uv run python zelda_i/scripts/probe_level6_entry.py --infinite-life --save-state

    # From entry, also probe east key room
    uv run python zelda_i/scripts/probe_level6_entry.py --from-state Level6Entrance \\
        --infinite-life --probe-east --tag l6_recon

Not Clean STATUS — Survival assist only.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retro_harness.env import make_env, save_state
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ids import object_name, room_item_name
from zelda_i.level6_overworld import (
    LEVEL6,
    LEVEL6_DOOR_X,
    LEVEL6_DOOR_X_HI,
    LEVEL6_DOOR_X_LO,
    LEVEL6_EAST_KEY_ROOM,
    LEVEL6_ENTRY_ROOM,
    SCREEN_LEVEL6_ENTRANCE,
    Level6EntryRightController,
    OverworldToLevel6Controller,
    level6_entrance_success,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot

SEGMENT_MAX = 20000


def _objs(snap) -> list[dict]:
    out: list[dict] = []
    for o in snap.objects:
        if not (1 <= o.slot <= 10) or o.type_id in (0, 0xFF):
            continue
        out.append(
            {
                "slot": o.slot,
                "type": o.type_id,
                "type_name": object_name(o.type_id),
                "x": o.x,
                "y": o.y,
                "hp": o.hp,
            }
        )
    return out


def _snapshot_dict(snap) -> dict:
    objs = _objs(snap)
    return {
        "mode": snap.mode,
        "level": snap.level,
        "screen": snap.screen,
        "x": snap.link_x,
        "y": snap.link_y,
        "health": snap.health,
        "sword": snap.sword,
        "keys": snap.keys,
        "bombs": snap.bombs,
        "triforce": snap.triforce,
        "room_item_id": snap.room_item_id,
        "room_item_name": room_item_name(snap.room_item_id),
        "room_all_dead": snap.room_all_dead,
        "room_obj_count": snap.room_obj_count,
        "cur_opened_doors": snap.cur_opened_doors,
        "open_doorway_mask": snap.open_doorway_mask,
        "objects": objs,
        "type_counts": dict(Counter(o["type"] for o in objs)),
    }


def run_probe(
    *,
    start_state: str,
    infinite_life: bool,
    enter_dungeon: bool,
    probe_east: bool,
    save_checkpoint: bool,
    max_frames: int,
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
        trail: list[dict] = []
        total = 0
        reports: dict = {"entry": entry, "track": track}

        snap = read_snapshot(env.get_ram())

        # --- OW door hunt / enter ---
        if snap.level == 0 and (
            enter_dungeon
            or snap.screen == SCREEN_LEVEL6_ENTRANCE
            or start_state in ("L6Probe_22", "PostSwordStart")
        ):
            nav = OverworldToLevel6Controller(
                require_dungeon=enter_dungeon or save_checkpoint,
                require_level6_screen=not (enter_dungeon or save_checkpoint),
            )
            # If already on door screen with enter requested, door-hunt only.
            if snap.screen == SCREEN_LEVEL6_ENTRANCE:
                nav.hops = ()
                nav.require_dungeon = True
                nav.require_level6_screen = False

            last_sc = snap.screen
            while total < max_frames:
                snap = read_snapshot(env.get_ram())
                if snap.mode == 17:
                    reports["fail"] = "link_death"
                    break
                if level6_entrance_success(env.get_ram()) or nav.success:
                    break
                if nav.phase.name == "FAILED":
                    reports["fail"] = "nav_failed"
                    reports["nav"] = nav.report()
                    break
                act = nav.step(snap)
                obs, *_ = env.step(act.action)
                total += 1
                if assist is not None:
                    assist.apply_env(env, frame=total)
                if snap.screen != last_sc or snap.level != 0:
                    trail.append({"f": total, **_snapshot_dict(snap)})
                    last_sc = snap.screen

            snap = read_snapshot(env.get_ram())
            reports["nav"] = nav.report()
            reports["after_nav"] = _snapshot_dict(snap)
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_after_nav.png")

            # Finish dungeon settle if mid-enter (mode 16/2/3/4).
            for _ in range(400):
                snap = read_snapshot(env.get_ram())
                if level6_entrance_success(env.get_ram()):
                    break
                if snap.level != LEVEL6:
                    break
                obs, *_ = env.step(nes_idle_action())
                total += 1
                if assist is not None:
                    assist.apply_env(env, frame=total)

            if level6_entrance_success(env.get_ram()):
                for _ in range(120):
                    obs, *_ = env.step(nes_idle_action())
                    total += 1
                    if assist is not None:
                        assist.apply_env(env, frame=total)
                snap = read_snapshot(env.get_ram())
                reports["entry_room"] = _snapshot_dict(snap)
                save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_entry_0x79.png")
                if save_checkpoint:
                    path = save_state(env, GAME_DIR, GAME, "Level6Entrance")
                    reports["saved"] = str(path)

        # --- East key room ---
        if probe_east and read_snapshot(env.get_ram()).level == LEVEL6:
            east = Level6EntryRightController()
            while total < max_frames:
                snap = read_snapshot(env.get_ram())
                if east.success or east.phase.name == "FAILED":
                    break
                if snap.mode == 17:
                    break
                act = east.step(snap)
                obs, *_ = env.step(act.action)
                total += 1
                if assist is not None:
                    assist.apply_env(env, frame=total)
            for _ in range(200):
                obs, *_ = env.step(nes_idle_action())
                total += 1
                if assist is not None:
                    assist.apply_env(env, frame=total)
            snap = read_snapshot(env.get_ram())
            reports["east"] = {
                "controller": east.report(),
                "snap": _snapshot_dict(snap),
            }
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_east_0x{snap.screen:02x}.png")
            if snap.screen == LEVEL6_EAST_KEY_ROOM and save_checkpoint:
                path = save_state(env, GAME_DIR, GAME, "L6Room_7a")
                reports["saved_east"] = str(path)

        reports["ok"] = level6_entrance_success(env.get_ram()) or (
            reports.get("entry_room", {}).get("level") == LEVEL6
        )
        reports["frames"] = total
        reports["trail"] = trail
        reports["assist"] = assist.report() if assist else None
        reports["constants"] = {
            "door_screen": SCREEN_LEVEL6_ENTRANCE,
            "door_x": LEVEL6_DOOR_X,
            "door_x_band": [LEVEL6_DOOR_X_LO, LEVEL6_DOOR_X_HI],
            "entry_room": LEVEL6_ENTRY_ROOM,
            "east_key_room": LEVEL6_EAST_KEY_ROOM,
        }
        reports["final"] = _snapshot_dict(read_snapshot(env.get_ram()))

        out_path = RECORDINGS_DIR / f"{tag}_recon.json"
        write_json_report(out_path, reports)
        reports["report_path"] = str(out_path)
        return reports
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--from-state",
        default="L6Probe_22",
        help="Start state (default L6Probe_22 OW door; or Level6Entrance)",
    )
    p.add_argument("--infinite-life", action="store_true")
    p.add_argument(
        "--enter-dungeon",
        action="store_true",
        default=True,
        help="Door-hunt into room 0x79 (default on)",
    )
    p.add_argument("--no-enter", action="store_true", help="Stop on OW 0x22 only")
    p.add_argument("--probe-east", action="store_true", help="Also 0x79→0x7a")
    p.add_argument("--save-state", action="store_true")
    p.add_argument("--max-frames", type=int, default=SEGMENT_MAX)
    p.add_argument("--tag", default="l6_entry")
    args = p.parse_args()
    enter = False if args.no_enter else args.enter_dungeon
    report = run_probe(
        start_state=args.from_state,
        infinite_life=args.infinite_life,
        enter_dungeon=enter,
        probe_east=args.probe_east,
        save_checkpoint=args.save_state,
        max_frames=args.max_frames,
        tag=args.tag,
    )
    print(
        f"ok={report.get('ok')} track={report.get('track')} "
        f"final={report.get('final')} path={report.get('report_path')}"
    )
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()

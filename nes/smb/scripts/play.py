#!/usr/bin/env python3
"""Human free-record for Super Mario Bros. (32-exit default).

Product entry: ``./play smb`` (power-on / stage pin / resume). Same shape as
``snes/super_metroid/play``: F5 saves a stitchable tape, F6 mid-run pin,
reusing ``--name`` archives the prior take, stage seams write durable pins.

```bash
./play smb                         # power-on → all_exits_v1
./play smb 4-1 all_exits_v1        # continue from 4-1 pin
./play smb resume all_exits_v1     # last F5 / pin
./play smb --list
uv run python -m smb.scripts.play --from start --name all_exits_v1
```

Controls (PlaySession):
  F5 / F1     save recording + end state + pins, exit
  F6          mid-run pin (does not stop)
  ESC / Q     cancel without saving
  SELECT+R2   checkpoint save · SELECT+L2 load (tape truncates)
  [ ] TAB     speed / turbo
  Arrows      D-pad · Z=B · X=A
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from retro_harness.env import make_env, read_state_bytes
from retro_harness.nes import NES_ACTION_SIZE
from retro_harness.play_session import PlaySession
from retro_harness.play_spine import configure_display
from smb.paths import GAME_DIR, GAME_V0
from smb.play_record import (
    NES9_LAYOUT,
    ExitClock,
    archive_existing_take,
    fmt_time,
    load_pin_rta_offset,
    stage_label_of,
    trace_row,
    write_stage_pin,
)
from smb.policy import compress_nes9_rle
from smb.ram import read_snapshot
from smb.routes import get_route, list_routes
from smb.start_presets import (
    DEFAULT_ROUTE_ID,
    DEFAULT_TASK_NAME,
    HUMAN_DIR,
    POWER_ON_STARTS,
    ResolvedStart,
    list_start_presets,
    normalize_stage_id,
    resolve_start,
)


def _list_presets(task_name: str, route_id: str, out_dir: Path) -> int:
    route = get_route(route_id)
    print(f"Route: {route.route_id}  ({len(route.exits)} exits)  {route.display_name}")
    print(f"Task:  {task_name}  pins → {out_dir / (task_name + '_pins')}")
    print("\nStart presets:")
    print(f"  {'start':8s} [OK] power-on (title → 1-1). aliases: {', '.join(sorted(POWER_ON_STARTS))}")
    for key, mark, blurb in list_start_presets(
        task_name=task_name, route=route, out_dir=out_dir
    ):
        if key == "start":
            continue
        print(f"  {key:8s} [{mark}] {blurb}")
    print("\nRoutes:")
    for item in list_routes():
        print(f"  {item.route_id:<24} {len(item.exits):>2} exits  {item.display_name}")
    print("\nF5/F1=save  F6=pin  ESC=cancel  SELECT+R2=checkpoint")
    return 0


def record_session(
    *,
    start: ResolvedStart,
    task_name: str,
    route_id: str,
    out_dir: Path,
    scale: int,
    no_archive: bool,
    headless: bool,
) -> Path | None:
    route = get_route(route_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_path = out_dir / f"{task_name}.json"
    end_state_path = out_dir / f"{task_name}_end.state"

    if task_path.is_file() and not no_archive:
        archived = archive_existing_take(task_path)
        if archived is not None:
            print(f"[REC] archived previous take → {archived}", flush=True)

    power_on = start.kind == "power_on"
    state_bytes: bytes | None = None
    if not power_on:
        if start.path is None or not start.path.is_file():
            raise FileNotFoundError(f"start state missing: {start.path}")
        state_bytes = read_state_bytes(start.path)

    stage_key = normalize_stage_id(start.key)
    rta_offset = 0
    rta_zero_live: list[int | None] = [None]
    if power_on:
        rta_offset = 0
    elif stage_key:
        rta_offset = load_pin_rta_offset(task_name, stage_key, out_dir=out_dir)
    else:
        rta_offset = load_pin_rta_offset(task_name, "resume", out_dir=out_dir)

    clock = ExitClock(route, start_index=start.route_index)
    configure_display(headless=headless)
    env = make_env(GAME_V0, "NONE", GAME_DIR, render_mode="rgb_array")

    recorded: list[list[int]] = []
    trace: list[dict[str, Any]] = []
    saved = {"ok": False}
    live: dict[str, Any] = {
        "stage": "title" if power_on else start.label,
        "x": 0,
        "y": 0,
        "event": "",
    }

    def _rta_frames(local_fr: int) -> int:
        if power_on:
            if rta_zero_live[0] is None:
                return 0
            return max(0, int(local_fr) - int(rta_zero_live[0]))
        return max(0, int(rta_offset) + max(0, int(local_fr)))

    def _maybe_pin(snap, frame: int, *, kind: str, stage_id: str) -> None:
        try:
            blob = env.em.get_state()
        except Exception as exc:
            print(f"[PIN] capture failed: {exc}", flush=True)
            return
        path = write_stage_pin(
            task_name=task_name,
            stage_id=stage_id,
            state_bytes=blob,
            snap=snap,
            frame=frame,
            rta_frames=_rta_frames(frame),
            kind=kind,
            out_dir=out_dir,
        )
        print(
            f"[PIN] {stage_id}  t={fmt_time(_rta_frames(frame))} (f{frame})  "
            f"xy=({snap.player_x},{snap.player_y}) → {path}",
            flush=True,
        )

    def on_step(obs, reward, done, info) -> None:
        del obs, reward, done, info
        action = list(session.last_action_post_sanitize[:NES_ACTION_SIZE])
        if len(action) < NES_ACTION_SIZE:
            action.extend([0] * (NES_ACTION_SIZE - len(action)))
        rec_frame = len(recorded)
        snap = read_snapshot(env.get_ram(), frame=rec_frame)
        row = trace_row(snap, action, rec_frame=rec_frame)
        recorded.append(action)
        trace.append(row)
        live["stage"] = row["stage"]
        live["x"] = row["x"]
        live["y"] = row["y"]

        if power_on and rta_zero_live[0] is None and snap.playing and snap.player_state in (7, 8):
            if snap.world == 0 and snap.level == 0:
                rta_zero_live[0] = rec_frame
                print(
                    f"[RTA] 1-1 control zero @ local f{rec_frame} (32-exit clock starts here)",
                    flush=True,
                )

        event = clock.observe(snap, frame=rec_frame)
        if event == "entry":
            stage_id = clock.next_exit.exit_id if clock.next_exit else stage_label_of(snap)
            live["event"] = f"entry {stage_id}"
            _maybe_pin(snap, rec_frame, kind="control", stage_id=stage_id)
            print(
                f"[STAGE] {stage_id} control  t={fmt_time(_rta_frames(rec_frame))}  "
                f"exits={len(clock.completed)}/{len(route.exits) - start.route_index}",
                flush=True,
            )
        elif event == "exit":
            done_row = clock.completed[-1]
            live["event"] = f"exit {done_row['exit_id']}"
            print(
                f"[EXIT] {done_row['exit_id']} → {done_row['successor']}  "
                f"t={fmt_time(_rta_frames(rec_frame))}  "
                f"{len(clock.completed)}/{len(route.exits) - start.route_index}",
                flush=True,
            )
        elif event == "death":
            live["event"] = "death"
            death = clock.deaths[-1]
            print(
                f"[DEATH] {death['stage']} xy=({death['x']},{death['y']})  "
                f"lives={death['lives']}  t={fmt_time(_rta_frames(rec_frame))}",
                flush=True,
            )
        elif event == "off_route":
            warn = clock.off_route[-1]
            live["event"] = f"off-route {warn['stage']}"
            print(
                f"[WARN] off-route {warn['stage']} (expected {warn['expected']})  "
                f"t={fmt_time(_rta_frames(rec_frame))}",
                flush=True,
            )

    def on_hud(info) -> list[str]:
        del info
        n = len(recorded)
        remaining = len(route.exits) - start.route_index
        line = (
            f"t={fmt_time(_rta_frames(n))}  {live['stage']}  "
            f"xy=({live['x']},{live['y']})  "
            f"exits={len(clock.completed)}/{remaining}  deaths={len(clock.deaths)}"
        )
        if clock.complete:
            line += "  32-EXIT DONE"
        return [line]

    def on_key_down(key: int) -> bool:
        import pygame

        if key in (pygame.K_F5, pygame.K_F1):
            _finalize(save=True)
            session.running = False
            return True
        if key == pygame.K_F6:
            snap = read_snapshot(env.get_ram(), frame=max(0, len(recorded) - 1))
            _maybe_pin(
                snap,
                max(0, len(recorded) - 1),
                kind="manual",
                stage_id=stage_label_of(snap),
            )
            return True
        if key in (pygame.K_ESCAPE, pygame.K_q):
            n = len(recorded)
            if n > 0:
                print(
                    f"[REC] cancelled — dropping {n} frames. F5 to keep a stitchable tape.",
                    flush=True,
                )
            else:
                print("[REC] cancelled", flush=True)
            session.running = False
            return True
        return False

    def on_trigger_save(slot: int) -> None:
        frame = session.save_checkpoint(slot)
        snap = read_snapshot(env.get_ram(), frame=frame)
        print(
            f"[CP SAVE {slot}] {stage_label_of(snap)}  "
            f"t={fmt_time(_rta_frames(frame))} xy=({snap.player_x},{snap.player_y})",
            flush=True,
        )

    def on_trigger_load(slot: int) -> None:
        frame = session.load_checkpoint(slot)
        if frame is None:
            print(f"[CP LOAD {slot}] empty", flush=True)
            return
        if len(recorded) > frame:
            del recorded[frame:]
        if len(trace) > frame:
            del trace[frame:]
        clock.rewind(frame)
        snap = read_snapshot(env.get_ram(), frame=len(recorded))
        live["stage"] = stage_label_of(snap)
        live["x"] = snap.player_x
        live["y"] = snap.player_y
        print(
            f"[CP LOAD {slot}] {stage_label_of(snap)}  tape truncated to {len(recorded)}f",
            flush=True,
        )

    def _finalize(*, save: bool) -> None:
        if not save or saved["ok"]:
            return
        if not recorded:
            print("[REC] nothing recorded")
            return
        try:
            end_bytes = env.em.get_state()
            end_state_path.write_bytes(end_bytes)
            snap = read_snapshot(env.get_ram(), frame=max(0, len(recorded) - 1))
            write_stage_pin(
                task_name=task_name,
                stage_id=stage_label_of(snap),
                state_bytes=end_bytes,
                snap=snap,
                frame=max(0, len(recorded) - 1),
                rta_frames=_rta_frames(max(0, len(recorded) - 1)),
                kind="end",
                out_dir=out_dir,
            )
        except Exception as exc:
            print(f"[REC] end-state capture failed: {exc}")
            end_bytes = None

        report = clock.report()
        payload = {
            "format": "smb_human_nes9",
            "name": task_name,
            "recorded_at": datetime.now().isoformat(),
            "route": route.route_id,
            "handoff": start.key,
            "start_state": start.label,
            "power_on": power_on,
            "game_name": GAME_V0,
            "button_layout": list(NES9_LAYOUT),
            "num_frames": len(recorded),
            "num_human_frames": len(recorded),
            "frames": recorded,
            "human_frames": recorded,
            "trace": trace,
            "segments_rle": compress_nes9_rle(recorded),
            "human_segments_rle": compress_nes9_rle(recorded),
            "exits": report,
            "rta": {
                "zero": "1-1_first_control" if power_on else "pin_offset",
                "offset_frames": int(rta_offset),
                "local_frames": len(recorded),
                "rta_frames": _rta_frames(max(0, len(recorded) - 1)),
                "rta_label": fmt_time(_rta_frames(max(0, len(recorded) - 1))),
            },
            "end_state": str(end_state_path.relative_to(GAME_DIR))
            if end_state_path.exists()
            else None,
        }
        task_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        saved["ok"] = True
        print(
            f"[REC] saved {task_path} ({len(recorded)} frames, "
            f"exits={len(clock.completed)}/{len(route.exits) - start.route_index}, "
            f"t={payload['rta']['rta_label']})",
            flush=True,
        )
        if end_state_path.exists():
            print(f"[REC] end state → {end_state_path}", flush=True)
        print(
            f"[REC] parse: uv run python -m smb.scripts.parse_human_recording "
            f"{task_path} --export-skills",
            flush=True,
        )

    session = PlaySession(
        env,
        game_dir=str(GAME_DIR),
        game=GAME_V0,
        scale=scale,
        title=f"SMB REC: {task_name} [{route.route_id} / {start.label}]",
        action_size=NES_ACTION_SIZE,
        base_fps=60,
        resync_state_bytes=state_bytes,
    )
    session.quiet_checkpoints = True
    session.hud_minimal = True
    session.on_step = on_step
    session.on_hud = on_hud
    session.on_key_down = on_key_down
    session.on_trigger_save = on_trigger_save
    session.on_trigger_load = on_trigger_load

    print(
        f"[play] --from {start.key} --name {task_name} --route {route.route_id}",
        flush=True,
    )
    print(
        f"[REC] {start.blurb}  F5=save  F6=pin  ESC=cancel",
        flush=True,
    )
    try:
        session.run()
    finally:
        env.close()
    return task_path if saved["ok"] else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--from",
        dest="start",
        default="start",
        help="Start preset: power-on, 1-1…8-4 pin, resume, or a .state path",
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_TASK_NAME,
        help=f"Tape stem under recordings/human/ (default: {DEFAULT_TASK_NAME})",
    )
    parser.add_argument(
        "--route",
        default=DEFAULT_ROUTE_ID,
        help="Route id (default: all_exits / 32). Also: warp",
    )
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=HUMAN_DIR,
        help=f"Output dir (default: {HUMAN_DIR})",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Overwrite tasks/<name>.json without archiving the prior take",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List start presets + routes and exit",
    )
    args = parser.parse_args(argv)

    if args.list:
        return _list_presets(args.name, args.route, args.out_dir)

    try:
        start = resolve_start(
            args.start,
            task_name=args.name,
            route=get_route(args.route),
            out_dir=args.out_dir,
        )
    except (FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    path = record_session(
        start=start,
        task_name=args.name,
        route_id=args.route,
        out_dir=args.out_dir,
        scale=args.scale,
        no_archive=args.no_archive,
        headless=args.headless,
    )
    return 0 if path is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())

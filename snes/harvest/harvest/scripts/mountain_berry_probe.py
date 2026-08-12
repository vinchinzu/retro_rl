#!/usr/bin/env python3
"""Spring D2 → first mountain berry probe (reactive segments + frame splits).

Default: ``Y1_Inside_House`` (Spring D2 06:08) through ``MountainBerryTask``.
Does **not** replay ``get_berry.json``. Optional ``--mode replay`` is only a
discovery aid to locate the first held-item change on the human tape.

Examples:

    HEADLESS=1 uv run python -m harvest.scripts.mountain_berry_probe
    HEADLESS=1 uv run python -m harvest.scripts.mountain_berry_probe \\
      --video recordings/mountain_berry_d2.mp4
    HEADLESS=1 uv run python -m harvest.scripts.mountain_berry_probe \\
      --mode replay --state Y1_Inside_House
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from harvest.paths import PROJECT_DIR, ensure_monorepo_on_path

ensure_monorepo_on_path()

from retro_harness import TaskStatus, WorldState

from harvest.core.animal_status import read_held_item
from harvest.core.npc_catalog import game_objects
from harvest.core.ram_catalog import read_ram_value
from harvest.core.task_progress import task_progress_snapshot
from harvest.maps.map_config import get_map_name
from harvest.runtime.recording_trace import pressed_buttons
from harvest.runtime.retro_setup import make_harvest_env
from harvest.tasks.mountain_berry import (
    GRAPE_STAND_PX,
    MountainBerryTask,
    at_grape_stand,
    held_forage_name,
    is_mountain_forage,
    nearby_tile_scan,
)
from harvest.tasks.nav import get_pos_from_ram, make_action
from harvest.tasks.recorded_task import RecordedTask


def _configure_headless() -> None:
    os.environ.setdefault("HEADLESS", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.pop("INFINITE_STAMINA", None)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", default="Y1_Inside_House")
    p.add_argument(
        "--mode",
        choices=("reactive", "replay"),
        default="reactive",
        help="reactive=MountainBerryTask; replay=get_berry tape (discovery only)",
    )
    p.add_argument("--task", default="get_berry", help="Recording name for --mode replay")
    p.add_argument("--timeout", type=int, default=12_000)
    p.add_argument(
        "--pick",
        action="store_true",
        help="A-pick the ground grape and keep it (Don't eat). Default is stand only.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_DIR / "recordings" / "mountain_berry_probe.json",
    )
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--video-fps", type=int, default=60)
    p.add_argument("--video-scale", type=int, default=3)
    p.add_argument(
        "--screenshot",
        type=Path,
        default=PROJECT_DIR / "recordings" / "mountain_grape_stand.png",
    )
    return p.parse_args()


def _snap(ram, frame: int, *, phase: str = "", extra: dict | None = None) -> dict:
    pos = get_pos_from_ram(ram)
    tilemap = int(read_ram_value(ram, "tilemap"))
    held = int(read_held_item(ram))
    row = {
        "frame": frame,
        "phase": phase,
        "tilemap": tilemap,
        "tilemap_hex": f"0x{tilemap:02X}",
        "map": get_map_name(tilemap),
        "x": int(pos.x),
        "y": int(pos.y),
        "tx": int(pos.x) // 16,
        "ty": int(pos.y) // 16,
        "held_item": held,
        "held_hex": f"0x{held:02X}",
        "held_name": held_forage_name(held) or "",
        "hour": int(read_ram_value(ram, "hour")),
        "minute": int(read_ram_value(ram, "minute")),
        "input_lock": int(read_ram_value(ram, "input_lock")),
        "dialog_text_id": int(read_ram_value(ram, "dialog_text_id", raw=True)),
        "dialog_menu_cursor": int(read_ram_value(ram, "dialog_menu_cursor", raw=True)),
    }
    if extra:
        row.update(extra)
    return row


class _VideoRecorder:
    def __init__(self, path: Path, first_frame, *, fps: int, scale: int) -> None:
        frame = np.ascontiguousarray(np.asarray(first_frame, dtype=np.uint8)[:, :, :3])
        self.path = path
        self.fps = fps
        self.scale = scale
        self.width = int(frame.shape[1])
        self.height = int(frame.shape[0])
        self.frames = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self._process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{self.width}x{self.height}",
                "-framerate",
                str(fps),
                "-i",
                "-",
                "-an",
                "-vf",
                f"scale=iw*{scale}:ih*{scale}:flags=neighbor,setsar=7/6",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )
        self.write(frame)

    def write(self, frame) -> None:
        image = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8)[:, :, :3])
        if self._process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        self._process.stdin.write(image.tobytes())
        self.frames += 1

    def close(self) -> dict:
        if self._process.stdin is not None:
            self._process.stdin.close()
        code = self._process.wait()
        return {
            "path": str(self.path),
            "frames": self.frames,
            "fps": self.fps,
            "scale": self.scale,
            "ffmpeg_ok": code == 0,
        }


def _save_png(obs, path: Path) -> None:
    if obs is None or path is None:
        return
    from PIL import Image

    arr = np.asarray(obs)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr[..., :3].astype("uint8")).save(path)


def _run_reactive(env, args: argparse.Namespace, video: _VideoRecorder | None) -> dict:
    ram = env.get_ram()
    task = MountainBerryTask(
        timeout=args.timeout,
        approach_only=not args.pick,
        pick_attempts=3 if args.pick else 0,
    )
    world = WorldState(frame=0, ram=ram, info={}, obs=None)
    task.reset(world)
    start = _snap(ram, 0, phase=task.phase_text)
    log = [{"event": "start", **start}]
    splits: list[dict] = []
    last_phase = task.phase_text
    last_map = start["tilemap"]
    frame = 0
    status = TaskStatus.RUNNING
    reason = "start"
    obs = None

    print(f"[BERRY] start map={start['map']} pos=({start['x']},{start['y']}) phase={last_phase}")
    while frame < args.timeout and status == TaskStatus.RUNNING:
        ram = env.get_ram()
        world = WorldState(frame=frame, ram=ram, info={}, obs=obs)
        result = task.step(world)
        status = result.status
        reason = result.reason or status.value
        action = make_action()
        if result.action is not None:
            action = getattr(result.action, "action", result.action)
        step = env.step(action)
        obs = step[0]
        if video is not None:
            video.write(obs)
        frame += 1
        ram = env.get_ram()
        phase = task.phase_text
        tilemap = int(read_ram_value(ram, "tilemap"))
        if phase != last_phase or tilemap != last_map:
            row = _snap(ram, frame, phase=phase, extra={"reason": reason, "prev_phase": last_phase})
            splits.append(row)
            log.append({"event": "split", **row})
            print(
                f"[BERRY] f={frame} {last_phase}->{phase} "
                f"map={row['map']} pos=({row['x']},{row['y']}) held=0x{row['held_item']:02X}"
            )
            last_phase = phase
            last_map = tilemap
        if is_mountain_forage(int(read_held_item(ram))) and not args.pick:
            break
        if frame % 400 == 0:
            snap = _snap(ram, frame, phase=phase)
            prog = task_progress_snapshot(task)
            print(
                f"[BERRY] f={frame} phase={phase} map={snap['map']} "
                f"pos=({snap['x']},{snap['y']}) {reason}"
            )
            log.append({"event": "tick", **snap, "progress": asdict(prog) if prog else None})

    ram = env.get_ram()
    end = _snap(ram, frame, phase=task.phase_text, extra={"reason": reason})
    nearby = []
    for obj in game_objects(ram):
        if obj.is_player:
            continue
        nearby.append(
            {
                "label": obj.label,
                "kind": obj.kind,
                "sprite": f"0x{obj.sprite_table_idx:04X}",
                "pixel": list(obj.pixel),
                "tile": list(obj.tile),
            }
        )
    held = int(end["held_item"])
    lock = int(end.get("input_lock", 1))
    reached = at_grape_stand(ram) or (
        end["tilemap"] == 16
        and abs(int(end["x"]) - GRAPE_STAND_PX[0]) <= 24
        and abs(int(end["y"]) - GRAPE_STAND_PX[1]) <= 24
    )
    picked = is_mountain_forage(held)
    kept = picked and lock == 1
    if args.pick:
        success = kept
    else:
        success = reached
    print(
        f"[BERRY] end reached={reached} picked={picked} kept={kept} frames={frame} "
        f"held=0x{held:02X} ({end['held_name'] or 'none'}) lock={lock} map={end['map']} "
        f"pos=({end['x']},{end['y']})"
    )
    tiles = nearby_tile_scan(ram, radius=3)
    shot = None
    if args.screenshot is not None:
        try:
            frame_img = obs if obs is not None else env.get_screen()
        except Exception:
            frame_img = obs
        _save_png(frame_img, args.screenshot)
        shot = str(args.screenshot)
    log.append({"event": "end", **end, "objects": nearby[:12], "tiles": tiles})
    return {
        "mode": "reactive",
        "success": success,
        "reached_stand": reached,
        "picked": picked,
        "kept": kept,
        "pick_verified": bool(args.pick and kept),
        "frames": frame,
        "seconds": round(frame / 60.0, 3),
        "status": status.value,
        "reason": reason,
        "start": start,
        "end": end,
        "splits": splits,
        "nearby_objects": nearby[:12],
        "nearby_tiles": tiles,
        "screenshot": shot,
        "log": log,
    }


def _run_replay(env, args: argparse.Namespace, video: _VideoRecorder | None) -> dict:
    tape = RecordedTask.load(args.task)
    ram = env.get_ram()
    start = _snap(ram, 0, phase="replay")
    events: list[dict] = [{"event": "start", **start}]
    first_mountain = None
    first_forage = None
    last_map = start["tilemap"]
    frame = 0
    obs = None
    print(f"[REPLAY] {args.task} frames={len(tape.frames)} start map={start['map']}")
    world = WorldState(frame=0, ram=ram, info={}, obs=None)
    tape.reset(world)
    while frame < min(args.timeout, len(tape.frames)):
        ram = env.get_ram()
        world = WorldState(frame=frame, ram=ram, info={}, obs=obs)
        result = tape.step(world)
        action = make_action()
        if result.action is not None:
            action = getattr(result.action, "action", result.action)
        step = env.step(action)
        obs = step[0]
        if video is not None:
            video.write(obs)
        frame += 1
        ram = env.get_ram()
        tilemap = int(read_ram_value(ram, "tilemap"))
        held = int(read_held_item(ram))
        if tilemap != last_map:
            row = _snap(ram, frame, phase="replay", extra={"buttons": pressed_buttons(action)})
            events.append({"event": "map_change", **row})
            print(f"[REPLAY] f={frame} map={row['map']} pos=({row['x']},{row['y']})")
            if tilemap == 0x10 and first_mountain is None:
                first_mountain = row
            last_map = tilemap
        if first_forage is None and is_mountain_forage(held):
            first_forage = _snap(ram, frame, phase="replay")
            events.append({"event": "first_forage", **first_forage})
            print(
                f"[REPLAY] first forage f={frame} {first_forage['held_name']} "
                f"pos=({first_forage['x']},{first_forage['y']}) map={first_forage['map']}"
            )
            break
    ram = env.get_ram()
    end = _snap(ram, frame, phase="replay")
    return {
        "mode": "replay",
        "task": args.task,
        "success": first_forage is not None,
        "frames": frame,
        "seconds": round(frame / 60.0, 3),
        "start": start,
        "end": end,
        "first_mountain": first_mountain,
        "first_forage": first_forage,
        "events": events,
    }


def main() -> int:
    args = _parse_args()
    _configure_headless()
    wall0 = time.time()
    env = make_harvest_env(args.state)
    env.reset()
    video = None
    if args.video is not None:
        obs = env.get_screen() if hasattr(env, "get_screen") else None
        if obs is None:
            step = env.step(make_action())
            obs = step[0]
        video = _VideoRecorder(args.video, obs, fps=args.video_fps, scale=args.video_scale)

    try:
        if args.mode == "replay":
            report = _run_replay(env, args, video)
        else:
            report = _run_reactive(env, args, video)
    finally:
        video_result = video.close() if video is not None else None

    report["state"] = args.state
    report["timeout"] = args.timeout
    report["wall_seconds"] = round(time.time() - wall0, 1)
    report["video"] = video_result
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[BERRY] wrote {args.out}")
    if video_result:
        print(f"[BERRY] video {video_result['path']} frames={video_result['frames']}")
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())

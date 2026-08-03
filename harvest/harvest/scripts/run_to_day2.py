#!/usr/bin/env python3
"""Run Harvest Moon from a morning state through one or more overnights.

Default path: multi-day planner with ``target_days=1`` (return-home + sleep that
always finds the house). Optional ``--day-plan boot_to_day2`` runs the explicit
macro-chained phase sequence inside a single-day plan that ends with sleep.

Success: calendar goal reached, morning scene stable, no mid-run state load.

Examples:

    HEADLESS=1 uv run python -m harvest.scripts.run_to_day2
    HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 --state Y1_Inside_House
    # Bootstrap from a real power-on first (D1 town-gate → farm remains open):
    HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 --power-on --until-day 2
    HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 --days 2 --until-day 4
    # Full spring from pinned morning (sleeps through Spring 30 → Summer 1):
    HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 --end-of-spring \\
      --out recordings/run_spring_month.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from harvest.paths import GAME_DIR, PROJECT_DIR, ensure_monorepo_on_path

ensure_monorepo_on_path()

from retro_harness import TaskStatus, WorldState

from harvest.core.ram_catalog import read_ram_value
from harvest.core.scene import classify_scene_from_ram, morning_scene_ready
from harvest.planner.day_plan import DayPlanTask, MultiDayPlannerTask, PHASE_SEQUENCES
from harvest.runtime.power_on import PowerOnStartTask
from harvest.runtime.retro_setup import make_harvest_env
from harvest.tasks.farm_clearer import make_action


def _configure_headless() -> None:
    os.environ.setdefault("HEADLESS", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    # This runner is evidence-oriented: never inherit an optional play-session
    # assist from the caller's shell.
    os.environ.pop("INFINITE_STAMINA", None)


def _file_sha256(path: Path) -> str | None:
    """Return a source-state digest for a replay report, when available."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _VideoRecorder:
    """Stream raw emulator RGB frames to an H.264 MP4 without dropping frames."""

    def __init__(self, path: Path, first_frame, *, fps: int, scale: int) -> None:
        if fps <= 0:
            raise ValueError("video fps must be positive")
        if scale <= 0:
            raise ValueError("video scale must be positive")

        frame = self._normalize_frame(first_frame)
        self.path = path
        self.fps = fps
        self.scale = scale
        self.width = int(frame.shape[1])
        self.height = int(frame.shape[0])
        self.frames = 0
        self._closed = False
        self._result: dict[str, object] | None = None

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

    @staticmethod
    def _normalize_frame(frame) -> np.ndarray:
        image = np.asarray(frame, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"expected RGB emulator frame, got shape={image.shape}")
        return np.ascontiguousarray(image[:, :, :3])

    def write(self, frame) -> None:
        image = self._normalize_frame(frame)
        if image.shape != (self.height, self.width, 3):
            raise ValueError(
                f"emulator video size changed from {self.width}x{self.height} "
                f"to {image.shape[1]}x{image.shape[0]}"
            )
        if self._process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        self._process.stdin.write(image.tobytes())
        self.frames += 1

    def close(self) -> dict[str, object]:
        if self._result is not None:
            return self._result
        if self._process.stdin is not None:
            self._process.stdin.close()
        return_code = self._process.wait()
        self._closed = True
        self._result = {
            "path": str(self.path),
            "fps": self.fps,
            "scale": self.scale,
            "frames": self.frames,
            "duration_seconds": round(self.frames / self.fps, 3),
            "encoded": return_code == 0 and self.path.is_file(),
            "ffmpeg_return_code": return_code,
            "audio": False,
        }
        return self._result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", default="Y1_Inside_House")
    p.add_argument(
        "--power-on",
        action="store_true",
        help=(
            "Boot with no save state, create a new diary, and hand off from "
            "the controllable Spring day-1 opening. Overrides --state."
        ),
    )
    p.add_argument(
        "--day-plan",
        default=None,
        choices=list(PHASE_SEQUENCES.keys()),
        help="Named phase sequence; default uses multi-day auto plan",
    )
    p.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of successful overnights to complete (multi-day mode)",
    )
    p.add_argument(
        "--until-day",
        type=int,
        default=None,
        help="Stop once calendar day is >= this (inclusive morning target)",
    )
    p.add_argument(
        "--until-season",
        type=int,
        default=None,
        help="Season index for --until-day (default: start season)",
    )
    p.add_argument(
        "--end-of-spring",
        action="store_true",
        help="Run until Summer morning (MultiDay until Spring 30 exclusive end)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Hard frame budget (default scales with overnight count)",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=2000,
        help="Print progress every N frames (0 disables)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_DIR / "recordings" / "run_to_day2.json",
    )
    p.add_argument(
        "--save-end-state",
        type=str,
        default=None,
        help="Optional state name to write under custom_integrations when done",
    )
    p.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optional H.264 MP4 path. Captures every emulated frame at --video-fps.",
    )
    p.add_argument(
        "--video-fps",
        type=int,
        default=60,
        help="Frame rate for --video (default: 60, native emulation rate)",
    )
    p.add_argument(
        "--video-scale",
        type=int,
        default=3,
        help="Integer nearest-neighbor output scale for --video (default: 3)",
    )
    return p.parse_args()


def _world(env, frame: int = 0) -> WorldState:
    return WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None)


def _date_fields(ram) -> dict[str, int]:
    return {
        "season": int(read_ram_value(ram, "season")),
        "day": int(read_ram_value(ram, "day")),
        "hour": int(read_ram_value(ram, "hour")),
        "minute": int(read_ram_value(ram, "minute")),
        "tilemap": int(read_ram_value(ram, "tilemap")),
        "money": int(read_ram_value(ram, "money")),
        "stamina": int(read_ram_value(ram, "stamina")),
    }


def _build_task(args: argparse.Namespace, start_season: int) -> object:
    if args.day_plan:
        if (
            args.days is not None
            or args.until_day is not None
            or args.end_of_spring
        ):
            raise SystemExit(
                "--day-plan cannot be combined with --days/--until-day/--end-of-spring"
            )
        return DayPlanTask(
            phase_sequence=list(PHASE_SEQUENCES[args.day_plan]),
            state_name=args.state,
        )

    # MultiDayPlannerTask: target_days OR exclusive until_(season, day).
    # harvest_bot --end-of-spring uses until_season=0 until_day=30 so success
    # when date > Spring 30 (i.e. Summer 1 morning).
    if args.end_of_spring:
        return MultiDayPlannerTask(
            until_season=0,
            until_day=30,
            max_days=40,
        )

    target_days = args.days
    until_day = args.until_day
    until_season = args.until_season if args.until_season is not None else start_season

    if target_days is None and until_day is None:
        target_days = 1

    kwargs: dict = {"max_days": 40}
    if target_days is not None:
        kwargs["target_days"] = target_days
        kwargs["max_days"] = max(target_days + 1, 2)

    if until_day is not None:
        # Inclusive morning target → exclusive MultiDay bound (day-1).
        # Example: --until-day 4 → stop when date > Spring 3 → D4 morning.
        # Cross-season: --until-season 1 --until-day 1 → exclusive (1, 0).
        if until_day <= 1 and until_season > 0:
            kwargs["until_season"] = until_season - 1
            kwargs["until_day"] = 30
        else:
            kwargs["until_season"] = until_season
            kwargs["until_day"] = max(0, until_day - 1)
        if target_days is None:
            kwargs.pop("target_days", None)
            kwargs["max_days"] = max(kwargs.get("until_day", 30) + 5, 4)

    return MultiDayPlannerTask(**kwargs)


def _goal_reached(
    *,
    start: tuple[int, int],
    end: tuple[int, int],
    days_completed: int | None,
    args: argparse.Namespace,
) -> bool:
    if end <= start and not args.end_of_spring:
        # Still allow end-of-spring only if we somehow wrap? No.
        if end == start:
            return False
    if args.end_of_spring:
        # Summer (or later) morning after finishing Spring 30 overnight.
        return end > (0, 30)
    if end <= start:
        return False
    if args.until_day is not None:
        until_season = (
            args.until_season if args.until_season is not None else start[0]
        )
        return end >= (until_season, args.until_day)
    if args.days is not None:
        if days_completed is not None:
            return days_completed >= args.days
        return end[0] > start[0] or end[1] >= start[1] + args.days
    return end > start


def _summarize_journal(journal: list[dict]) -> dict:
    phase_success: dict[str, int] = {}
    phase_skip: dict[str, int] = {}
    for row in journal:
        for result in row.get("phase_results") or []:
            name = str(result.get("phase", "?"))
            status = str(result.get("status", ""))
            if status == "success":
                phase_success[name] = phase_success.get(name, 0) + 1
            elif status == "skipped":
                phase_skip[name] = phase_skip.get(name, 0) + 1
    return {
        "overnights": len(journal),
        "phase_success_counts": phase_success,
        "phase_skip_counts": phase_skip,
        "final_money": journal[-1].get("money") if journal else None,
    }


def main() -> int:
    args = _parse_args()
    _configure_headless()

    if args.end_of_spring:
        overnights_budget = 32
    elif args.days is not None:
        overnights_budget = args.days
    elif args.until_day is not None:
        overnights_budget = max(2, args.until_day + 2)
    else:
        overnights_budget = 1

    if args.max_frames is None:
        # ~25k frames/day upper bound with crop/clear work.
        args.max_frames = 30_000 * max(1, overnights_budget)

    source_state = None if args.power_on else GAME_DIR / f"{args.state}.state"
    source_state_sha256 = _file_sha256(source_state) if source_state else None
    env = make_harvest_env(state=None if args.power_on else args.state, render_mode="rgb_array")
    video: _VideoRecorder | None = None
    t0 = time.monotonic()
    try:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        if args.video is not None:
            video = _VideoRecorder(
                args.video,
                obs,
                fps=args.video_fps,
                scale=args.video_scale,
            )
        frames = 0
        power_on_task: PowerOnStartTask | None = None
        power_on_report: dict[str, object] | None = None
        boot_frames = 0
        if args.power_on:
            power_on_task = PowerOnStartTask()
            power_on_task.reset(_world(env, frames))
            print("[RUN] power-on bootstrap: title -> START -> new diary -> Spring D1", flush=True)
            while boot_frames < power_on_task.timeout:
                world = _world(env, frames)
                bootstrap = power_on_task.step(world)
                if bootstrap.status == TaskStatus.SUCCESS:
                    power_on_report = power_on_task.summary(world)
                    break
                if bootstrap.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
                    power_on_report = power_on_task.summary(world)
                    power_on_report["failure"] = bootstrap.reason or bootstrap.status.value
                    report = {
                        "state": None,
                        "power_on": power_on_report,
                        "success": False,
                        "reason": bootstrap.reason or bootstrap.status.value,
                        "mid_run_state_load": False,
                        "clean_run": {
                            "intervention_class": "Clean",
                            "initial_state_loads": 0,
                            "mid_run_state_loads": 0,
                            "ram_writes": 0,
                            "infinite_stamina": False,
                            "source_state": None,
                            "source_state_sha256": None,
                        },
                    }
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                    print(json.dumps(report, indent=2), flush=True)
                    return 2
                action = (
                    bootstrap.action.action
                    if bootstrap.action is not None
                    else make_action()
                )
                step = env.step(action)
                if video is not None:
                    video.write(step[0])
                frames += 1
                boot_frames += 1
            else:
                world = _world(env, frames)
                power_on_report = power_on_task.summary(world)
                power_on_report["failure"] = "power-on frame budget exhausted"
                report = {
                    "state": None,
                    "power_on": power_on_report,
                    "success": False,
                    "reason": "power-on frame budget exhausted",
                    "mid_run_state_load": False,
                    "clean_run": {
                        "intervention_class": "Clean",
                        "initial_state_loads": 0,
                        "mid_run_state_loads": 0,
                        "ram_writes": 0,
                        "infinite_stamina": False,
                        "source_state": None,
                        "source_state_sha256": None,
                    },
                }
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                print(json.dumps(report, indent=2), flush=True)
                return 2

            # Count the successful bootstrap frame in the evidence, but do not
            # advance the emulator after the task has already reached a stable
            # controllable scene.
            boot_frames = frames
            print(
                f"[RUN] power-on ready after {boot_frames} frames: "
                f"{power_on_report['scene']['mode']} "
                f"S{power_on_report['date']['season']}D{power_on_report['date']['day']}",
                flush=True,
            )

        world = _world(env, frames)
        start_fields = _date_fields(world.ram)
        start_season = start_fields["season"]
        start_day = start_fields["day"]
        start_key = (start_season, start_day)

        task = _build_task(args, start_season)
        task.reset(world)

        plan_label = args.day_plan or (
            "end_of_spring" if args.end_of_spring else "auto_multi_day"
        )
        print(
            f"[RUN] state={'power_on' if args.power_on else args.state} plan={plan_label} "
            f"start=S{start_season}D{start_day} "
            f"days={args.days} until=({args.until_season},{args.until_day}) "
            f"end_of_spring={args.end_of_spring} max_frames={args.max_frames}",
            flush=True,
        )

        reason = "budget"
        terminal = False
        last_logged_day = start_key
        planner_start_frame = frames
        while frames - planner_start_frame < args.max_frames:
            world = _world(env, frames)
            result = task.step(world)
            frames += 1

            season = int(read_ram_value(world.ram, "season"))
            day = int(read_ram_value(world.ram, "day"))
            current = (season, day)

            if current != last_logged_day:
                print(
                    f"[RUN] day change S{last_logged_day[0]}D{last_logged_day[1]} "
                    f"-> S{season}D{day} at frame={frames}",
                    flush=True,
                )
                last_logged_day = current

            if result.status == TaskStatus.SUCCESS:
                reason = result.reason or "success"
                terminal = True
                break
            if result.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
                reason = result.reason or result.status.value
                terminal = True
                break

            if args.day_plan and _goal_reached(
                start=start_key,
                end=current,
                days_completed=getattr(task, "_days_completed", None),
                args=args,
            ):
                reason = "goal reached"
                terminal = True
                break

            if args.progress_every and frames % args.progress_every == 0:
                phase = getattr(task, "phase_text", "?")
                progress = getattr(task, "progress_text", "")
                hour = int(read_ram_value(world.ram, "hour"))
                minute = int(read_ram_value(world.ram, "minute"))
                money = int(read_ram_value(world.ram, "money"))
                print(
                    f"[RUN] f={frames} date=S{season}D{day} "
                    f"{hour:02d}:{minute:02d} ${money} phase={phase} {progress}",
                    flush=True,
                )

            action = (
                result.action.action
                if result.action is not None
                else make_action()
            )
            step = env.step(action)
            if video is not None:
                video.write(step[0])

        world = _world(env, frames)
        scene = classify_scene_from_ram(world.ram)
        end_fields = _date_fields(world.ram)
        end_key = (end_fields["season"], end_fields["day"])
        days_completed = getattr(task, "_days_completed", None)
        journal = list(getattr(task, "day_journal", ()) or ())
        advanced = end_key > start_key
        goal = _goal_reached(
            start=start_key,
            end=end_key,
            days_completed=days_completed,
            args=args,
        )
        morning_ok = morning_scene_ready(scene, end_fields["hour"]) or (
            scene.is_normal_map and int(read_ram_value(world.ram, "input_lock")) == 1
        )
        success = bool(goal and advanced and morning_ok)

        # The ROM can expose its next-day RAM before its fade has produced a
        # visible frame.  These neutral frames do not alter inputs, RAM, or
        # the completed plan; they simply let the captured presentation catch
        # up to the already-verified morning state.
        presentation_settle_frames = 0
        if video is not None and success:
            for _ in range(360):
                presentation_step = env.step(make_action())
                video.write(presentation_step[0])
                presentation_settle_frames += 1

        video_result = video.close() if video is not None else None
        if video_result is not None:
            video_result["post_success_neutral_frames"] = presentation_settle_frames

        if args.save_end_state and success:
            try:
                import gzip

                state_bytes = env.em.get_state()
                # stable-retro expects gzip-compressed .state files (same as
                # play_session / retro_harness.recorder). Raw s9xsnp bytes fail
                # load with BadGzipFile.
                out_state = GAME_DIR / f"{args.save_end_state}.state"
                with gzip.open(out_state, "wb", compresslevel=9) as handle:
                    # Preserve original basename inside the gzip header.
                    handle.write(state_bytes)
                print(f"[RUN] Saved end state -> {out_state}", flush=True)
            except Exception as exc:
                print(f"[RUN] Could not save end state: {exc}", flush=True)

        report = {
            "state": None if args.power_on else args.state,
            "power_on": power_on_report,
            "boot_frames": boot_frames,
            "planner_frames": frames - planner_start_frame,
            "day_plan": plan_label,
            "days": args.days,
            "until_day": args.until_day,
            "until_season": args.until_season,
            "end_of_spring": bool(args.end_of_spring),
            "frames": frames,
            "wall_seconds": round(time.monotonic() - t0, 1),
            "start": start_fields,
            "end": end_fields,
            "scene": scene.summary(),
            "morning_ready": bool(morning_ok),
            "days_completed": days_completed,
            "day_failures": list(getattr(task, "day_failures", ()) or ()),
            "day_journal": journal,
            "journal_summary": _summarize_journal(journal),
            "success": success,
            "advanced": advanced,
            "goal_reached": goal,
            "reason": ("goal reached" if success else reason),
            "terminal": terminal,
            "mid_run_state_load": False,
            "clean_run": {
                "intervention_class": "Clean",
                "initial_state_loads": 0 if args.power_on else 1,
                "mid_run_state_loads": 0,
                "ram_writes": 0,
                "infinite_stamina": False,
                "source_state": str(source_state) if source_state else None,
                "source_state_sha256": source_state_sha256,
            },
            "video": video_result,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        return 0 if success else 1
    finally:
        if video is not None:
            video.close()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())

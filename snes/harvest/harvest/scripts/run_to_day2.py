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
import time
from pathlib import Path

import numpy as np

from harvest.paths import GAME_DIR, PROJECT_DIR, ensure_monorepo_on_path

ensure_monorepo_on_path()

from retro_harness import TaskStatus, WorldState
from retro_harness.video import VideoCaptureConfig, VideoRecorder

from harvest.core.ram_catalog import read_ram_value
from harvest.core.scene import classify_scene_from_ram, morning_scene_ready
from harvest.planner.day_plan import DayPlanTask, MultiDayPlannerTask, PHASE_SEQUENCES
from harvest.runtime.power_on import PowerOnStartTask
from harvest.runtime.retro_setup import make_harvest_env
from harvest.tasks.nav import make_action
from harvest.tasks.town_day1_handoff import TownDay1HandoffTask


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


def _rgb_frame(frame) -> np.ndarray:
    image = np.asarray(frame, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"expected RGB emulator frame, got shape={image.shape}")
    return np.ascontiguousarray(image[:, :, :3])


def _open_video(path: Path, first_frame, *, fps: int, scale: int) -> VideoRecorder:
    rgb = _rgb_frame(first_frame)
    recorder = VideoRecorder(
        path,
        width=int(rgb.shape[1]),
        height=int(rgb.shape[0]),
        config=VideoCaptureConfig(
            fps=fps,
            scale=scale,
            crf=18,
            preset="medium",
            audio=False,
            footer=False,
        ),
    )
    recorder.write(rgb)
    return recorder


def _video_report(recorder: VideoRecorder, **extra: object) -> dict[str, object]:
    fps = recorder.config.fps
    return {
        "path": str(recorder.path),
        "fps": fps,
        "scale": recorder.config.scale,
        "frames": recorder.frames,
        "duration_seconds": round(recorder.frames / fps, 3),
        "encoded": True,
        "audio": False,
        **extra,
    }


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
        "--d1-handoff",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "After power-on (or any Spring D1 town start), run TownDay1Handoff "
            "(talks+truck+shed+sleep→D2) before multi-day. Default: on for "
            "--power-on, off otherwise. Use --no-d1-handoff to disable."
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
        help="Hard frame budget (default scales with overnight count; 0 disables)",
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
        "--save-after-weeds",
        type=int,
        default=None,
        help="Save one debug checkpoint after CLEAR_BUSHES clears this many weeds",
    )
    p.add_argument(
        "--weed-checkpoint-state",
        default="Y1_D2_After_400_Weeds",
        help="State name used by --save-after-weeds",
    )
    p.add_argument(
        "--stop-after-d2-shipping",
        action="store_true",
        help=(
            "Stop on Spring D2 immediately after the farm 5pm ShippingScene, "
            "requiring a successful mountain grape route and potato seed "
            "purchase. Use with --power-on (continuous D1 handoff) or a D2 "
            "morning --state, plus --save-end-state."
        ),
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
    phase_no_work: dict[str, int] = {}
    phase_failure: dict[str, int] = {}
    water_deltas: list[dict] = []
    establish_deltas: list[dict] = []
    harvest_deltas: list[dict] = []
    total_shipped = 0
    total_harvested = 0
    total_planted = 0
    for row in journal:
        try:
            total_shipped += int(row.get("shipped_count") or 0)
            total_harvested += int(row.get("harvested_count") or 0)
            total_planted += int(row.get("establish_planted") or 0)
        except Exception:
            pass
        for result in row.get("phase_results") or []:
            name = str(result.get("phase", "?"))
            status = str(result.get("status", ""))
            reason = str(result.get("reason") or "")
            if status == "success":
                phase_success[name] = phase_success.get(name, 0) + 1
            elif status == "skipped":
                phase_skip[name] = phase_skip.get(name, 0) + 1
            elif status == "no_work":
                phase_no_work[name] = phase_no_work.get(name, 0) + 1
            elif status in {"failure", "blocked"}:
                phase_failure[name] = phase_failure.get(name, 0) + 1
            if name == "CROP_WATER" and "watered=" in reason:
                water_deltas.append(
                    {
                        "plan_day": row.get("plan_day"),
                        "status": status,
                        "reason": reason,
                    }
                )
            if name == "CROP_ESTABLISH" and "planted=" in reason:
                establish_deltas.append(
                    {
                        "plan_day": row.get("plan_day"),
                        "status": status,
                        "reason": reason,
                    }
                )
            if name == "HARVEST_ROUTE":
                harvest_deltas.append(
                    {
                        "plan_day": row.get("plan_day"),
                        "status": status,
                        "reason": reason,
                        "shipped_count": result.get("shipped_count"),
                        "harvested_count": result.get("harvested_count"),
                    }
                )
    final_money = journal[-1].get("money") if journal else None
    harvest_phases_present = bool(
        phase_success.get("HARVEST_ROUTE") or harvest_deltas
    )
    # Count real plant deltas (planted=N with N>0), not merely phase presence.
    if total_planted <= 0:
        for row in establish_deltas:
            reason = str(row.get("reason") or "")
            if "planted=" not in reason:
                continue
            try:
                n = int(reason.split("planted=")[1].split()[0].rstrip(","))
            except Exception:
                n = 0
            if n > 0:
                total_planted += n
    establish_nonzero = total_planted > 0
    # Gate A (rr-y8n): money growth + harvest phases on continuous soak.
    try:
        money_ok = final_money is not None and int(final_money) > 100
    except Exception:
        money_ok = False
    return {
        "overnights": len(journal),
        "phase_success_counts": phase_success,
        "phase_skip_counts": phase_skip,
        "phase_no_work_counts": phase_no_work,
        "phase_failure_counts": phase_failure,
        "crop_water_deltas": water_deltas,
        "crop_establish_deltas": establish_deltas,
        "harvest_deltas": harvest_deltas,
        "total_shipped": total_shipped,
        "total_harvested": total_harvested,
        "total_planted": total_planted,
        "final_money": final_money,
        "harvest_phases_present": harvest_phases_present,
        "crop_establish_nonzero": establish_nonzero,
        "gate_a_economy_ok": bool(money_ok and harvest_phases_present),
    }


def _crop_survival_report(ram: np.ndarray) -> dict:
    """Farm crop tile counts when the farm map is loaded; else map-not-farm note."""
    from harvest.planner.day_plan_status import is_farm_tilemap
    from harvest.tasks.crop_planter import count_crop_survival
    from harvest.core.tile_catalog import ADDR_TILEMAP

    tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else -1
    if not is_farm_tilemap(tilemap):
        return {
            "farm_map_loaded": False,
            "tilemap": tilemap,
            "note": "crop tiles only visible on farm metatile map; house end-state is inconclusive",
        }
    counts = count_crop_survival(ram)
    return {
        "farm_map_loaded": True,
        "tilemap": tilemap,
        **counts,
        "alive": int(counts.get("crop", 0)) > 0,
    }


def _d2_spine_checkpoint_evidence(task: object, world: WorldState) -> dict[str, object]:
    """Report whether the continuous power-on D2 spine reached its save gate."""
    fields = _date_fields(world.ram)
    phase_results = list(getattr(task, "_last_day_phase_results", ()) or ())
    successful = {
        str(row.get("phase"))
        for row in phase_results
        if str(row.get("status")) == "success"
    }
    potato_seeds = int(read_ram_value(world.ram, "potato_seeds"))
    ready = bool(
        fields["season"] == 0
        and fields["day"] == 2
        and fields["hour"] >= 17
        and getattr(task, "_phase", None) == "return_home"
        and "MOUNTAIN_BERRY" in successful
        and "BUY_SEEDS" in successful
        and "WAIT_FARM_SHIPPING" in successful
    )
    return {
        "ready": ready,
        "date": fields,
        "planner_phase": getattr(task, "_phase", None),
        "successful_phases": sorted(successful),
        "potato_seeds": potato_seeds,
        "mountain_grape_shipped": "MOUNTAIN_BERRY" in successful,
        # CROP_ESTABLISH may consume the newly purchased bag before the 5pm
        # shipper gate. BUY_SEEDS is closed by stock-up + wallet-down evidence.
        "potato_purchase_complete": "BUY_SEEDS" in successful,
        "shipping_dialogue_cleared": "WAIT_FARM_SHIPPING" in successful,
    }


def _active_farm_clear_task(task: object) -> object | None:
    """Find the active FarmClearTask through planner/orchestrator wrappers."""
    from harvest.tasks.farm_clear_task import FarmClearTask

    current: object | None = task
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, FarmClearTask):
            return current
        seen.add(id(current))
        current = getattr(current, "current_task", None)
    return None


def _save_emulator_state(env, state_name: str) -> Path:
    import gzip

    out_state = GAME_DIR / f"{state_name}.state"
    with gzip.open(out_state, "wb", compresslevel=9) as handle:
        handle.write(env.em.get_state())
    return out_state


def main() -> int:
    args = _parse_args()
    _configure_headless()

    if args.stop_after_d2_shipping and args.power_on is False and args.state is None:
        raise SystemExit("--stop-after-d2-shipping needs --power-on or a D2 morning --state")
    if args.stop_after_d2_shipping and args.day_plan:
        raise SystemExit("--stop-after-d2-shipping uses the live multi-day planner")

    if args.end_of_spring:
        overnights_budget = 32
    elif args.days is not None:
        overnights_budget = args.days
    elif args.until_day is not None:
        overnights_budget = max(2, args.until_day + 2)
    else:
        overnights_budget = 1

    if args.max_frames is None:
        # Campaign leftover (CLEAR_BUSHES / pond dumps) needs a day-sized
        # budget; halt earlier only if nav oscillates with no progress.
        args.max_frames = 200_000 * max(1, overnights_budget)

    source_state = None if args.power_on else GAME_DIR / f"{args.state}.state"
    source_state_sha256 = _file_sha256(source_state) if source_state else None
    env = make_harvest_env(state=None if args.power_on else args.state, render_mode="rgb_array")
    video: VideoRecorder | None = None
    t0 = time.monotonic()
    try:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        if args.video is not None:
            video = _open_video(
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
                    video.write(_rgb_frame(step[0]))
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

        # Gate B full / rr-5in: power-on lands Spring D1 town; multi-day alone
        # cannot do six talks + truck + shed + sleep. Run TownDay1Handoff first.
        if args.d1_handoff is None:
            do_d1_handoff = bool(
                args.power_on and start_season == 0 and start_day == 1
            )
        else:
            do_d1_handoff = bool(args.d1_handoff)

        d1_handoff_report: dict[str, object] | None = None
        d1_handoff_frames = 0
        if do_d1_handoff and not args.day_plan:
            handoff = TownDay1HandoffTask(
                include_sleep=True,
                pick_starter_tools=True,
                # auto: house_size==0 → require grass+can (power-on Gate B)
                require_starter_tools=None,
            )
            handoff.reset(world)
            print(
                "[RUN] D1 handoff: talks + truck + outdoor intro + shed + sleep → D2",
                flush=True,
            )
            handoff_start = frames
            handoff_budget = min(int(handoff.timeout), max(30_000, args.max_frames // 3))
            while frames - handoff_start < handoff_budget:
                world = _world(env, frames)
                hr = handoff.step(world)
                if hr.status == TaskStatus.SUCCESS:
                    d1_handoff_report = handoff.summary(world)
                    d1_handoff_report["status"] = "success"
                    d1_handoff_frames = frames - handoff_start
                    print(
                        f"[RUN] D1 handoff OK after {d1_handoff_frames} frames: "
                        f"S{d1_handoff_report.get('season')}D{d1_handoff_report.get('day')} "
                        f"grass={d1_handoff_report.get('has_grass_seeds')} "
                        f"can={d1_handoff_report.get('has_watering_can')}",
                        flush=True,
                    )
                    break
                if hr.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
                    d1_handoff_report = handoff.summary(world)
                    d1_handoff_report["status"] = hr.status.value
                    d1_handoff_report["failure"] = hr.reason or hr.status.value
                    d1_handoff_frames = frames - handoff_start
                    report = {
                        "state": None if args.power_on else args.state,
                        "power_on": power_on_report,
                        "d1_handoff": d1_handoff_report,
                        "d1_handoff_frames": d1_handoff_frames,
                        "boot_frames": boot_frames,
                        "success": False,
                        "reason": f"d1_handoff: {hr.reason or hr.status.value}",
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
                    }
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                    print(json.dumps(report, indent=2), flush=True)
                    return 2
                action = hr.action.action if hr.action is not None else make_action()
                step = env.step(action)
                if video is not None:
                    video.write(_rgb_frame(step[0]))
                frames += 1
            else:
                world = _world(env, frames)
                d1_handoff_report = handoff.summary(world)
                d1_handoff_report["status"] = "failure"
                d1_handoff_report["failure"] = "d1 handoff frame budget exhausted"
                d1_handoff_frames = frames - handoff_start
                report = {
                    "state": None if args.power_on else args.state,
                    "power_on": power_on_report,
                    "d1_handoff": d1_handoff_report,
                    "d1_handoff_frames": d1_handoff_frames,
                    "boot_frames": boot_frames,
                    "success": False,
                    "reason": "d1 handoff frame budget exhausted",
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
                }
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                print(json.dumps(report, indent=2), flush=True)
                return 2

            # Re-sample date after D1→D2 so multi-day plan builds from D2 morning.
            world = _world(env, frames)
            # Keep start_key as power-on D1 for continuous claim goal checks;
            # multi-day internal date is live from RAM on reset.

        task = _build_task(args, start_season)
        task.reset(world)

        plan_label = args.day_plan or (
            "end_of_spring" if args.end_of_spring else "auto_multi_day"
        )
        if do_d1_handoff:
            plan_label = f"d1_handoff+{plan_label}"
        print(
            f"[RUN] state={'power_on' if args.power_on else args.state} plan={plan_label} "
            f"start=S{start_season}D{start_day} "
            f"days={args.days} until=({args.until_season},{args.until_day}) "
            f"end_of_spring={args.end_of_spring} max_frames={args.max_frames}",
            flush=True,
        )

        reason = "budget"
        terminal = False
        d2_checkpoint_evidence: dict[str, object] | None = None
        weed_checkpoint_saved: Path | None = None
        # Multi-day budget is independent of handoff frames already spent.
        planner_start_frame = frames
        last_logged_day = (
            int(read_ram_value(world.ram, "season")),
            int(read_ram_value(world.ram, "day")),
        )
        while args.max_frames <= 0 or frames - planner_start_frame < args.max_frames:
            world = _world(env, frames)
            result = task.step(world)
            frames += 1

            season = int(read_ram_value(world.ram, "season"))
            day = int(read_ram_value(world.ram, "day"))
            current = (season, day)

            if args.save_after_weeds is not None and weed_checkpoint_saved is None:
                clear_task = _active_farm_clear_task(task)
                if clear_task is not None:
                    clearer = getattr(clear_task, "clearer", None)
                    priority = getattr(clear_task, "priority", None) or []
                    weed_only = len(priority) == 1 and getattr(priority[0], "name", "") == "WEED"
                    cleared = int(getattr(clearer, "cleared_count", 0) or 0)
                    if weed_only and cleared >= args.save_after_weeds:
                        weed_checkpoint_saved = _save_emulator_state(
                            env, args.weed_checkpoint_state
                        )
                        print(
                            f"[RUN] Saved weed checkpoint cleared={cleared} "
                            f"-> {weed_checkpoint_saved}",
                            flush=True,
                        )

            if args.stop_after_d2_shipping:
                candidate = _d2_spine_checkpoint_evidence(task, world)
                if bool(candidate["ready"]):
                    d2_checkpoint_evidence = candidate
                    reason = "day-2 grape/seed spine reached post-5pm checkpoint"
                    terminal = True
                    break

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
                video.write(_rgb_frame(step[0]))

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
        if args.stop_after_d2_shipping:
            d2_checkpoint_evidence = (
                d2_checkpoint_evidence
                or _d2_spine_checkpoint_evidence(task, world)
            )
            success = bool(terminal and d2_checkpoint_evidence["ready"])
        else:
            success = bool(goal and advanced and morning_ok)

        # The ROM can expose its next-day RAM before its fade has produced a
        # visible frame.  These neutral frames do not alter inputs, RAM, or
        # the completed plan; they simply let the captured presentation catch
        # up to the already-verified morning state.
        presentation_settle_frames = 0
        if video is not None and success:
            for _ in range(360):
                presentation_step = env.step(make_action())
                video.write(_rgb_frame(presentation_step[0]))
                presentation_settle_frames += 1

        video_result = None
        if video is not None:
            video.close()
            video_result = _video_report(
                video,
                post_success_neutral_frames=presentation_settle_frames,
            )

        if args.save_end_state and success:
            try:
                out_state = _save_emulator_state(env, args.save_end_state)
                print(f"[RUN] Saved end state -> {out_state}", flush=True)
            except Exception as exc:
                print(f"[RUN] Could not save end state: {exc}", flush=True)

        # Crop keep-alive evidence (rr-3v9): only valid on farm metatile maps.
        crop_survival = _crop_survival_report(world.ram)

        js = _summarize_journal(journal)
        final_money = end_fields.get("money")
        try:
            money_ok = final_money is not None and int(final_money) > 100
        except Exception:
            money_ok = False
        report = {
            "state": None if args.power_on else args.state,
            "power_on": power_on_report,
            "d1_handoff": d1_handoff_report,
            "d1_handoff_frames": d1_handoff_frames,
            "d2_spine_checkpoint": d2_checkpoint_evidence,
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
            "journal_summary": js,
            "crop_survival": crop_survival,
            "success": success,
            "advanced": advanced,
            "goal_reached": goal,
            "money_gt_100": bool(money_ok),
            "gate_b_full_ok": bool(
                success
                and args.power_on
                and money_ok
                and end_key > (0, 30)
            ),
            "reason": (reason if success and args.stop_after_d2_shipping else "goal reached")
            if success
            else reason,
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

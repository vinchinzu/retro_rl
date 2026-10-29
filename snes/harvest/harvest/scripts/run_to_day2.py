from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
from pathlib import Path

import numpy as np

from harvest.paths import GAME_DIR, PROJECT_DIR, ensure_monorepo_on_path

ensure_monorepo_on_path()

from retro_harness import AuditCapabilities, AuditedEnv, TaskStatus, WorldState
from retro_harness.video import VideoCaptureConfig, VideoRecorder

from harvest.core.ram_catalog import read_ram_value
from harvest.core.scene import classify_scene_from_ram, morning_scene_ready
from harvest.core.task_progress import RunProgressSidecar, format_run_progress_line
from harvest.planner.campaign_composer import season_pin_session
from harvest.planner.day_plan import DayPlanTask, MultiDayPlannerTask, PHASE_SEQUENCES
from harvest.planner.day_plan_status import format_world_date, read_world_ymd
from harvest.planner.tasks.ending import EndingTask, is_ending_date, read_ending_signals
from harvest.runtime.power_on import PowerOnStartTask
from harvest.runtime.retro_setup import make_harvest_env
from harvest.tasks.nav import make_action
from harvest.tasks.town_day1_handoff import TownDay1HandoffTask
from harvest.tools.pr999_report import (
    _date_fields,
    _summarize_journal,
    _year_report_fields,
    _gate_b_full_ok,
    _crop_survival_report,
    _d2_spine_checkpoint_evidence,
)

_STOP = {"sig": None}

def _request_stop(signum, _frame) -> None:
    _STOP["sig"] = int(signum)

def _install_stop_handler() -> None:
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

def _configure_headless() -> None:
    os.environ.setdefault("HEADLESS", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.pop("INFINITE_STAMINA", None)

def _file_sha256(path: Path) -> str | None:
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

def _open_video(path: Path, first_frame, *, fps: int, scale: int, segment_frames: int | None) -> VideoRecorder:
    rgb = _rgb_frame(first_frame)
    recorder = VideoRecorder(
        path,
        width=int(rgb.shape[1]),
        height=int(rgb.shape[0]),
        segment_frames=segment_frames,
        config=VideoCaptureConfig( fps=fps, scale=scale, crf=28, preset="ultrafast", audio=False, footer=False, layout="native", ),
    )
    recorder.write(rgb)
    return recorder

def _video_report(recorder: VideoRecorder, **extra: object) -> dict[str, object]:
    fps = recorder.config.fps
    return {
        "path": str(recorder.parts[0]) if recorder.parts else str(recorder.path),
        "parts": [str(part) for part in recorder.parts],
        "fps": fps,
        "scale": recorder.config.scale,
        "frames": recorder.frames,
        "duration_seconds": round(recorder.frames / fps, 3),
        "encoded": True,
        "audio": False,
        "segment_frames": recorder.segment_frames,
        **extra,
    }

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Harvest Moon through one or more overnights.")
    p.add_argument("--state", default="Y1_Inside_House")
    p.add_argument("--power-on", action="store_true")
    p.add_argument("--d1-handoff", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--day-plan", default=None, choices=list(PHASE_SEQUENCES.keys()))
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--until-day", type=int, default=None)
    p.add_argument("--until-season", type=int, default=None)
    p.add_argument("--until-year", type=int, default=None)
    p.add_argument("--until-ending", action="store_true")
    p.add_argument("--perfect-rancher", action="store_true",
                   help="Dispatch the Perfect Rancher milestone campaign with native receipts.")
    p.add_argument("--end-of-spring", action="store_true")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--progress-every", type=int, default=2000)
    p.add_argument("--progress-sidecar", type=Path, default=None)
    p.add_argument("--progress-sidecar-every", type=int, default=None)
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--checkpoint-on-progress", action="store_true")
    p.add_argument("--out", type=Path, default=PROJECT_DIR / "recordings" / "run_to_day2.json")
    p.add_argument("--save-end-state", type=str, default=None)
    p.add_argument("--save-after-weeds", type=int, default=None)
    p.add_argument("--weed-checkpoint-state", default="Y1_D2_After_400_Weeds")
    p.add_argument("--stop-after-d2-shipping", action="store_true")
    p.add_argument("--stop-after-d2-clear", action="store_true")
    p.add_argument("--video", type=Path, default=None,
                   help="Part stem. Parts are <stem>_partNNN.mp4, closed every --video-segment-frames.")
    p.add_argument("--video-fps", type=int, default=60)
    p.add_argument("--video-scale", type=int, default=3)
    p.add_argument("--video-segment-frames", type=int, default=216_000,
                   help="Frames per playable part at 60fps (216000 = 1 hour). 0 keeps one file.")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--save-season-pins", action="store_true")
    return p.parse_args(argv)

def _world(env, frame: int = 0) -> WorldState:
    return WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None)

def _clean_run_report(
    env: AuditedEnv,
    *,
    initial_state_loads: int,
    source_state: Path | None,
    source_state_sha256: str | None,
) -> dict[str, object]:
    audit = env.audit()
    return {
        "intervention_class": "Clean",
        "initial_state_loads": int(initial_state_loads),
        "mid_run_state_loads": audit.mid_run_loads,
        "ram_writes": audit.ram_writes,
        "assists": dict(audit.assists or {}),
        "audit_capabilities": ( audit.capabilities.to_record() if audit.capabilities is not None else None ),
        "infinite_stamina": False,
        "source_state": str(source_state) if source_state else None,
        "source_state_sha256": source_state_sha256,
    }

def _d2_clear_status(task: object, ram):
    from harvest.planner.d2_work import observe_d2_farm
    journal: list = list(getattr(task, "_last_day_phase_results", ()) or ())
    current: object | None = task
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        journal.extend(getattr(current, "journal", ()) or ())
        journal.extend(getattr(current, "phase_results", ()) or ())
        current = getattr(current, "current_task", None)
    return observe_d2_farm(ram, journal)

def _d2_counts(ram) -> dict:
    from harvest.maps.farm_pond import WEST_POCKET_PLANT_CENTER
    from harvest.tasks.crop_skills import count_ring_planted, count_ring_wet
    from harvest.tasks.farm_clear_quota import count_debris
    row = count_debris(ram).as_dict()
    row["planted"] = count_ring_planted(ram, WEST_POCKET_PLANT_CENTER)
    row["wet"] = count_ring_wet(ram, WEST_POCKET_PLANT_CENTER)
    return row

DAYS_PER_SEASON = 30

def _ymd_text(key: tuple[int, int, int]) -> str:
    return f"Y{key[0] + 1}S{key[1]}D{key[2]}"

SEASONS_PER_YEAR = 4

def _day_ordinal(key: tuple[int, int, int]) -> int:
    year, season, day = key
    return (year * SEASONS_PER_YEAR + season) * DAYS_PER_SEASON + day

def days_between(start: tuple[int, int, int], end: tuple[int, int, int]) -> int:
    return max(0, _day_ordinal(end) - _day_ordinal(start))

def _previous_day(key: tuple[int, int, int]) -> tuple[int, int, int]:
    year, season, day = key
    if day > 1:
        return (year, season, day - 1)
    if season > 0:
        return (year, season - 1, DAYS_PER_SEASON)
    if year > 0:
        return (year - 1, SEASONS_PER_YEAR - 1, DAYS_PER_SEASON)
    return (0, 0, 0)

def resolve_until_target( args: argparse.Namespace, start_key: tuple[int, int, int] ) -> tuple[int, int, int] | None:
    if args.until_day is None and args.until_year is None:
        return None
    year = start_key[0] if args.until_year is None else max(0, args.until_year - 1)
    if args.until_season is not None:
        season = args.until_season
    else:
        season = start_key[1] if args.until_year is None else 0
    day = args.until_day if args.until_day is not None else 1
    return (year, season, day)

def _build_task(args: argparse.Namespace, start_key: tuple[int, int, int]) -> object:
    extra: dict = {}
    if getattr(args, "perfect_rancher", False):
        extra["perfect_rancher"] = True
    if getattr(args, "stop_after_d2_clear", False):
        from harvest.planner.day_phase_types import DayPlannerPolicy
        extra["policy"] = DayPlannerPolicy(include_end_day=False)
    if args.day_plan:
        if (
            getattr(args, "perfect_rancher", False)
            or args.days is not None
            or args.until_day is not None
            or args.until_year is not None
            or args.until_ending
            or args.end_of_spring
        ):
            raise SystemExit("--day-plan cannot be combined with campaign or multi-day targets")
        return DayPlanTask( phase_sequence=list(PHASE_SEQUENCES[args.day_plan]), state_name=args.state, )
    if args.end_of_spring:
        return MultiDayPlannerTask( until_season=0, until_day=30, max_days=40, **extra, )
    target_days = args.days
    target = resolve_until_target(args, start_key)
    if target_days is None and target is None and not args.until_ending:
        target_days = 1
    kwargs: dict = {"max_days": 40}
    if target_days is not None:
        kwargs["target_days"] = target_days
        kwargs["max_days"] = max(target_days + 1, 2)
    if args.until_ending:
        kwargs["until_year"] = 99
        kwargs["max_days"] = 400
    elif target is not None:
        bound = _previous_day(target)
        kwargs["until_year"], kwargs["until_season"], kwargs["until_day"] = bound
        if target_days is None:
            kwargs.pop("target_days", None)
            kwargs["max_days"] = max( days_between(start_key, target) + 5, bound[2] + 5, 4 )
    kwargs.update(extra)
    return MultiDayPlannerTask(**kwargs)

def _goal_reached(
    *,
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    days_completed: int | None,
    args: argparse.Namespace,
) -> bool:
    if args.end_of_spring:
        return end > (0, 0, 30)
    if end <= start:
        return False
    target = resolve_until_target(args, start)
    if target is not None:
        return end >= target
    if args.days is not None:
        if days_completed is not None:
            return days_completed >= args.days
        return end[:2] > start[:2] or end[2] >= start[2] + args.days
    return end > start




def _active_farm_clear_task(task: object) -> object | None:
    from harvest.tasks.farm_clear_task import FarmClearTask
    current: object | None = task
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, FarmClearTask):
            return current
        seen.add(id(current))
        current = getattr(current, "current_task", None)
    return None

def _write_state(env, path: Path) -> Path:
    import gzip
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wb", compresslevel=9) as handle:
        handle.write(env.em.get_state())
    os.replace(tmp, path)
    return path

def _save_emulator_state(env, state_name: str) -> Path:
    return _write_state(env, GAME_DIR / f"{state_name}.state")

def _progress_sidecar_path(args: argparse.Namespace) -> Path | None:
    every = args.progress_sidecar_every
    if every is None:
        every = args.progress_every
    if every == 0:
        return None
    if args.progress_sidecar is not None:
        return args.progress_sidecar
    if args.stop_after_d2_clear or args.power_on:
        return args.out.with_name(args.out.stem + ".progress.json")
    return None

def _checkpoint_dir(args: argparse.Namespace) -> Path | None:
    if args.checkpoint_dir is not None:
        return args.checkpoint_dir
    if args.checkpoint_on_progress:
        return PROJECT_DIR / "recordings" / "d2_progress_checkpoints"
    return None

def main() -> int:
    args = _parse_args()
    _configure_headless()
    if args.stop_after_d2_shipping and args.power_on is False and args.state is None:
        raise SystemExit("--stop-after-d2-shipping needs --power-on or a D2 morning --state")
    if args.stop_after_d2_shipping and args.day_plan:
        raise SystemExit("--stop-after-d2-shipping uses the live multi-day planner")
    if args.stop_after_d2_clear and args.power_on is False and args.state is None:
        raise SystemExit("--stop-after-d2-clear needs --power-on or a D2 morning --state")
    if args.stop_after_d2_clear and args.day_plan:
        raise SystemExit("--stop-after-d2-clear uses the live multi-day planner")
    if args.end_of_spring:
        overnights_budget = 32
    elif args.days is not None:
        overnights_budget = args.days
    elif args.until_ending:
        overnights_budget = 300
    elif args.until_year is not None:
        overnights_budget = max(2, (args.until_year - 1) * 120 + (args.until_season or 0) * 30 + (args.until_day or 1) + 2)
    elif args.until_day is not None:
        overnights_budget = max(2, args.until_day + 2)
    else:
        overnights_budget = 1
    if args.max_frames is None:
        args.max_frames = 2_000_000 * max(1, overnights_budget)
    source_state = None if args.power_on else GAME_DIR / f"{args.state}.state"
    source_state_sha256 = _file_sha256(source_state) if source_state else None
    env = AuditedEnv(
        make_harvest_env(
            state=None if args.power_on else args.state,
            render_mode="rgb_array",
            lean_step=args.video is None
            and os.environ.get("HARVEST_LEAN_STEP", "1") != "0",
        ),
        capabilities=AuditCapabilities.all("harvest.run_to_day2"),
    )
    video: VideoRecorder | None = None
    video_sealed = False
    video_result: dict[str, object] | None = None

    def _seal_video(**extra: object) -> dict[str, object] | None:
        nonlocal video_sealed, video_result
        if video is None:
            return None
        if not video_sealed:
            video_sealed = True
            try:
                video.close()
            except Exception as exc:
                video_result = {
                    "error": str(exc),
                    "parts": [str(part) for part in video.parts],
                    "frames": video.frames,
                }
            else:
                video_result = _video_report(video)
        if extra and isinstance(video_result, dict) and "error" not in video_result:
            video_result = {**video_result, **extra}
        return video_result

    def _publish(report: dict, code: int) -> int:
        report["video"] = _seal_video()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        return code

    t0 = time.monotonic()
    sidecar_path = _progress_sidecar_path(args)
    checkpoint_dir = _checkpoint_dir(args)
    sidecar_every = ( args.progress_sidecar_every if args.progress_sidecar_every is not None else args.progress_every )
    sidecar = (
        RunProgressSidecar(
            sidecar_path,
            checkpoint_dir=checkpoint_dir,
            checkpoint_on_progress=bool(args.checkpoint_on_progress or checkpoint_dir),
        )
        if sidecar_path is not None
        else None
    )
    _install_stop_handler()
    try:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        if args.video is not None:
            video = _open_video(
                args.video, obs, fps=args.video_fps, scale=args.video_scale,
                segment_frames=args.video_segment_frames or None,
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
                        "clean_run": _clean_run_report( env, initial_state_loads=0, source_state=None, source_state_sha256=None, ),
                    }
                    return _publish(report, 2)
                action = ( bootstrap.action.action if bootstrap.action is not None else make_action() )
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
                    "clean_run": _clean_run_report( env, initial_state_loads=0, source_state=None, source_state_sha256=None, ),
                }
                return _publish(report, 2)
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
        start_key = (start_fields["year"], start_season, start_day)
        if args.d1_handoff is None:
            do_d1_handoff = bool( args.power_on and start_season == 0 and start_day == 1 )
        else:
            do_d1_handoff = bool(args.d1_handoff)
        d1_handoff_report: dict[str, object] | None = None
        d1_handoff_frames = 0
        if do_d1_handoff and not args.day_plan:
            handoff = TownDay1HandoffTask( include_sleep=True, pick_starter_tools=True, require_starter_tools=None, )
            handoff.reset(world)
            print( "[RUN] D1 handoff: talks + truck + outdoor intro + shed + sleep → D2", flush=True, )
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
                        "clean_run": _clean_run_report(
                            env,
                            initial_state_loads=0 if args.power_on else 1,
                            source_state=source_state,
                            source_state_sha256=source_state_sha256,
                        ),
                    }
                    if args.save_end_state:
                        _save_emulator_state(env, f"{args.save_end_state}_FAIL")
                    return _publish(report, 2)
                action = hr.action.action if hr.action is not None else make_action()
                step = env.step(action)
                if video is not None:
                    video.write(_rgb_frame(step[0]))
                frames += 1
                if sidecar is not None and sidecar_every and frames % sidecar_every == 0:
                    sidecar.write( frame=frames, planner_frames=0, wall_seconds=time.monotonic() - t0, ram=world.ram, task=handoff, )
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
                    "clean_run": _clean_run_report(
                        env,
                        initial_state_loads=0 if args.power_on else 1,
                        source_state=source_state,
                        source_state_sha256=source_state_sha256,
                    ),
                }
                return _publish(report, 2)
            world = _world(env, frames)
        task = _build_task(args, start_key)
        planner = task  # never overwritten; --until-ending steps ending_task
        task.reset(world)
        pins = season_pin_session(args, start_key, source_state_sha256)
        plan_label = args.day_plan or ( "end_of_spring" if args.end_of_spring else "auto_multi_day" )
        if do_d1_handoff:
            plan_label = f"d1_handoff+{plan_label}"
        print(
            f"[RUN] state={'power_on' if args.power_on else args.state} plan={plan_label} "
            f"start=Y{start_key[0] + 1}S{start_key[1]}D{start_key[2]} "
            f"days={args.days} until=({args.until_year},{args.until_season},{args.until_day}) "
            f"until_ending={args.until_ending} "
            f"end_of_spring={args.end_of_spring} max_frames={args.max_frames}",
            flush=True,
        )
        reason = "budget"
        terminal = False
        d2_checkpoint_evidence: dict[str, object] | None = None
        d2_prev_status = None
        d2_start_counts = _d2_counts(world.ram) if args.stop_after_d2_clear else None
        weed_checkpoint_saved: Path | None = None
        planner_start_frame = frames
        last_logged_day = read_world_ymd(world.ram)
        ending_task: EndingTask | None = None
        ending_report: dict[str, object] = {
            "started": False,
            "finished": False,
            "handoff_frame": None,
            "handoff_date": None,
            "peak_index": 0,
            "final": None,
        }
        stepped = planner
        def _start_ending(active_world, why: str) -> EndingTask:
            nonlocal ending_task
            ending_task = EndingTask()
            ending_task.reset(active_world)
            ending_report["handoff_frame"] = frames
            ending_report["handoff_date"] = format_world_date(active_world.ram)
            print( f"[RUN] ending handoff ({why}) date={format_world_date(active_world.ram)} " f"frame={frames}", flush=True, )
            return ending_task
        def _sidecar_write(active_task, active_world, *, d2_status=None):
            if sidecar is None:
                return None
            return sidecar.write(
                frame=frames,
                planner_frames=frames - planner_start_frame,
                wall_seconds=time.monotonic() - t0,
                ram=active_world.ram,
                task=active_task,
                d2_status=d2_status,
                save_state=( (lambda path: _write_state(env, path)) if sidecar.checkpoint_on_progress else None ),
            )
        result = None
        while args.max_frames <= 0 or frames - planner_start_frame < args.max_frames:
            world = _world(env, frames)
            if ( args.until_ending and ending_task is None and is_ending_date(*read_world_ymd(world.ram)) ):
                _start_ending(world, "ending date reached")
            if ending_task is not None:
                stepped = ending_task
                result = ending_task.step(world)
            else:
                if pins is not None: pins.before_step(planner, world.ram, frames, lambda path: _write_state(env, path), env)
                stepped = planner
                result = planner.step(world)
            frames += 1
            year, season, day = read_world_ymd(world.ram)
            current = (year, season, day)
            if ending_task is not None:
                sig = ending_task.signals
                ending_report["started"] = bool(sig is not None and sig.started) or bool( ending_report["started"] )
                ending_report["peak_index"] = max( int(ending_report["peak_index"]), ending_task.peak_index )
            if args.save_after_weeds is not None and weed_checkpoint_saved is None:
                clear_task = _active_farm_clear_task(task)
                if clear_task is not None:
                    clearer = getattr(clear_task, "clearer", None)
                    priority = getattr(clear_task, "priority", None) or []
                    weed_only = len(priority) == 1 and getattr(priority[0], "name", "") == "WEED"
                    cleared = int(getattr(clearer, "cleared_count", 0) or 0)
                    if weed_only and cleared >= args.save_after_weeds:
                        weed_checkpoint_saved = _save_emulator_state( env, args.weed_checkpoint_state )
                        print( f"[RUN] Saved weed checkpoint cleared={cleared} " f"-> {weed_checkpoint_saved}", flush=True, )
            if args.stop_after_d2_shipping:
                candidate = _d2_spine_checkpoint_evidence(task, world)
                if bool(candidate["ready"]):
                    d2_checkpoint_evidence = candidate
                    reason = "day-2 grape/seed spine reached post-5pm checkpoint"
                    terminal = True
                    break
            if args.stop_after_d2_clear:
                from harvest.planner.d2_work import confirm_d2_complete
                d2_status = _d2_clear_status(task, world.ram)
                if confirm_d2_complete(d2_prev_status, d2_status):
                    reason = "d2 farm clear complete"
                    terminal = True
                    break
                d2_prev_status = d2_status
            if current != last_logged_day:
                print( f"[RUN] day change {_ymd_text(last_logged_day)} " f"-> {_ymd_text(current)} at frame={frames}", flush=True, )
                last_logged_day = current
            if ( result.status == TaskStatus.SUCCESS and args.until_ending and ending_task is None and "ending" in (result.reason or "") ):
                _start_ending(world, result.reason or "ending reached")
                stepped = ending_task
                result = ending_task.step(world)
            if ( result.status == TaskStatus.SUCCESS and ending_task is not None and stepped is ending_task ):
                ending_report["finished"] = True
                reason = result.reason or "ending credits complete"
                terminal = True
                break
            elif result.status == TaskStatus.SUCCESS:
                if args.stop_after_d2_clear:
                    d2_status = _d2_clear_status(task, world.ram)
                    from harvest.planner.d2_work import confirm_d2_complete
                    if confirm_d2_complete(d2_prev_status, d2_status):
                        reason = "d2 farm clear complete"
                        terminal = True
                        break
                    d2_prev_status = d2_status
                else:
                    reason = result.reason or "success"
                    terminal = True
                    break
            if result.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
                reason = result.reason or result.status.value
                terminal = True
                break
            if args.day_plan and not args.stop_after_d2_clear and _goal_reached(
                start=start_key,
                end=current,
                days_completed=getattr(planner, "_days_completed", None),
                args=args,
            ):
                reason = "goal reached"
                terminal = True
                break
            stop_sig = _STOP["sig"]
            sample = bool(
                (args.progress_every and frames % args.progress_every == 0)
                or (sidecar is not None and sidecar_every and frames % sidecar_every == 0)
                or stop_sig is not None
            )
            if sample:
                record = _sidecar_write( stepped, world, d2_status=d2_prev_status if args.stop_after_d2_clear else None, )
                if args.progress_every and frames % args.progress_every == 0:
                    if record is not None:
                        print(format_run_progress_line(record), flush=True)
                    else:
                        phase = getattr(stepped, "phase_text", "?")
                        progress = getattr(stepped, "progress_text", "")
                        hour = int(read_ram_value(world.ram, "hour"))
                        minute = int(read_ram_value(world.ram, "minute"))
                        money = int(read_ram_value(world.ram, "money"))
                        cows = int(read_ram_value(world.ram, "num_cows"))
                        chickens = int(read_ram_value(world.ram, "num_chickens"))
                        tilemap = int(read_ram_value(world.ram, "tilemap"))
                        print(
                            f"[RUN] f={frames} date={_ymd_text(current) if year else f'S{season}D{day}'} "
                            f"{hour:02d}:{minute:02d} ${money} cows={cows} chickens={chickens} "
                            f"map={tilemap:#04x} phase={phase} {progress}",
                            flush=True,
                        )
            if stop_sig is not None:
                reason = f"interrupted by signal {stop_sig}"
                terminal = True
                break
            action = ( result.action.action if result.action is not None else make_action() )
            step = env.step(action)
            if pins is not None: pins.note(action, env.get_ram())
            if video is not None:
                video.write(_rgb_frame(step[0]))
        world = _world(env, frames)
        _sidecar_write( stepped, world, d2_status=d2_prev_status if args.stop_after_d2_clear else None, )
        scene = classify_scene_from_ram(world.ram)
        end_fields = _date_fields(world.ram)
        end_key = (end_fields["year"], end_fields["season"], end_fields["day"])
        if ending_task is not None and ending_task.signals is not None:
            ending_report["final"] = ending_task.signals.to_dict()
            ending_report["peak_index"] = max( int(ending_report["peak_index"]), ending_task.peak_index )
        year_fields = _year_report_fields(planner)
        days_completed = year_fields["days_completed"]
        journal = list(year_fields["day_journal"] or [])
        advanced = end_key > start_key
        goal = _goal_reached( start=start_key, end=end_key, days_completed=days_completed, args=args, )
        morning_ok = morning_scene_ready(scene, end_fields["hour"]) or (
            scene.is_normal_map and int(read_ram_value(world.ram, "input_lock")) == 1
        )
        d2_end_counts = None
        d2_final_status = None
        d2_two_settled_observations = False
        if args.stop_after_d2_shipping:
            d2_checkpoint_evidence = ( d2_checkpoint_evidence or _d2_spine_checkpoint_evidence(task, world) )
            success = bool(terminal and d2_checkpoint_evidence["ready"])
        elif args.stop_after_d2_clear:
            d2_status = d2_prev_status or _d2_clear_status(task, world.ram)
            d2_end_counts = _d2_counts(world.ram)
            d2_final_status = d2_status.to_record() if d2_status is not None else None
            from harvest.planner.d2_work import confirm_d2_complete
            d2_two_settled_observations = bool( confirm_d2_complete(d2_prev_status, _d2_clear_status(task, world.ram)) )
            success = bool( terminal and d2_status is not None and d2_status.is_complete and d2_two_settled_observations )
        elif args.until_ending:
            success = bool(ending_report["finished"])
            goal = success
        else:
            plan_finished = bool( args.day_plan and terminal and result is not None and result.status == TaskStatus.SUCCESS )
            success = bool(plan_finished or (goal and advanced and morning_ok))
        presentation_settle_frames = 0
        if video is not None and success:
            for _ in range(360):
                presentation_step = env.step(make_action())
                video.write(_rgb_frame(presentation_step[0]))
                presentation_settle_frames += 1
        video_result = _seal_video(post_success_neutral_frames=presentation_settle_frames)
        if args.save_end_state:
            state_name = args.save_end_state if success else f"{args.save_end_state}_FAIL"
            try:
                out_state = _save_emulator_state(env, state_name)
                print(f"[RUN] Saved end state -> {out_state}", flush=True)
            except Exception as exc:
                print(f"[RUN] Could not save end state: {exc}", flush=True)
        crop_survival = _crop_survival_report(world.ram)
        js = year_fields["journal_summary"]
        if not isinstance(js, dict):
            js = _summarize_journal(journal)
        final_money = end_fields.get("money")
        try:
            money_ok = final_money is not None and int(final_money) > 100
        except Exception:
            money_ok = False
        clean_run = _clean_run_report(
            env,
            initial_state_loads=0 if args.power_on else 1,
            source_state=source_state,
            source_state_sha256=source_state_sha256,
        )
        try:
            mid_run_loads: int | None = int(clean_run["mid_run_state_loads"])
        except (KeyError, TypeError, ValueError):
            mid_run_loads = None
        try:
            total_shipped = int(js.get("total_shipped") or 0)
            total_harvested = int(js.get("total_harvested") or 0)
        except (TypeError, ValueError):
            total_shipped = 0
            total_harvested = 0
        gate_b = _gate_b_full_ok(
            power_on=bool(args.power_on),
            until_ending=bool(args.until_ending),
            end_key=end_key,
            total_shipped=total_shipped,
            total_harvested=total_harvested,
            mid_run_state_loads=mid_run_loads,
        )
        pin_report = None if pins is None else pins.finish()
        report = {
            "state": None if args.power_on else args.state,
            "season_pins": pin_report,
            "power_on": power_on_report,
            "d1_handoff": d1_handoff_report,
            "d1_handoff_frames": d1_handoff_frames,
            "d2_spine_checkpoint": d2_checkpoint_evidence,
            "boot_frames": boot_frames,
            "planner_frames": frames - planner_start_frame,
            "day_plan": plan_label,
            "phase_results": [ dict(row) for row in (getattr(planner, "phase_results", ()) or ()) ],
            "days": args.days,
            "until_day": args.until_day,
            "until_season": args.until_season,
            "until_year": args.until_year,
            "until_ending": bool(args.until_ending),
            "ending": ending_report if args.until_ending else None,
            "end_date": _ymd_text(end_key),
            "end_of_spring": bool(args.end_of_spring),
            "frames": frames,
            "wall_seconds": round(time.monotonic() - t0, 1),
            "start": start_fields,
            "end": end_fields,
            "scene": scene.summary(),
            "morning_ready": bool(morning_ok),
            "days_completed": days_completed,
            "day_failures": year_fields["day_failures"],
            "day_journal": journal,
            "journal_summary": js,
            "floor_report": year_fields["floor_report"],
            "crop_survival": crop_survival,
            "success": success,
            "advanced": advanced,
            "goal_reached": goal,
            "money_gt_100": bool(money_ok),
            "gate_b_full_ok": gate_b,
            "d2_farm": (
                {
                    "start": d2_start_counts,
                    "end": d2_end_counts,
                    "final_status": d2_final_status,
                    "two_consecutive_settled_observations": d2_two_settled_observations,
                }
                if args.stop_after_d2_clear
                else None
            ),
            "reason": (
                reason
                if success
                and (args.stop_after_d2_shipping or args.stop_after_d2_clear or args.until_ending)
                else "goal reached"
            )
            if success
            else reason,
            "terminal": terminal,
            "progress_sidecar": str(sidecar_path) if sidecar_path is not None else None,
            "mid_run_state_load": False,
            "clean_run": clean_run,
            "video": video_result,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        return 0 if success else 1
    finally:
        if video is not None and not video_sealed:
            try:
                video.close()
            except Exception as exc:
                print(f"[RUN] video close failed: {exc}", flush=True)
        env.close()

if __name__ == "__main__":
    raise SystemExit(main())

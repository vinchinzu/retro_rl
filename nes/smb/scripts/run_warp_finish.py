"""Run the policy-driven eight-exit SMB warp route to the 8-4 ending.

Modes:

- ``poweron`` (preferred / M7): hard reset → fixed boot script frames → idle
  phase-align → one controller seed through all eight exits. **Clean** —
  zero emulator-state loads after ``env.reset()``.
- ``continuous``: published ``Level1_1`` + idle phase-align + same seed
  (no mid-attempt load; not power-on).
- ``suffix``: published ``Level1_2_WarpMid`` + ending suffix only.
- ``chain``: power-on natural 1-1 + one disclosed mid-1-2 splice + suffix.

```bash
# Clean power-on → 8-4 ending
uv run python -m smb.scripts.run_warp_finish --mode poweron --trials 3

# Record MP4
uv run python -m smb.scripts.run_warp_finish --mode poweron --record
```
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from retro_harness.env import make_env, read_state_bytes, reset_obs
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from retro_harness.youtube_intro import DEFAULT_INTRO_FRAMES, project_intro_lines
from smb.menus import boot_to_level1_script, boot_to_ready, idle_n
from smb.paths import (
    FULLGAME_REPLAYS_DIR,
    GAME_DIR,
    GAME_V0,
    INTEGRATION_V0_DIR,
    RECORDINGS_DIR,
)
from smb.policy import (
    CONTINUOUS_SETTLE_FRAMES,
    DEFAULT_1_1_SEED,
    DEFAULT_CONTINUOUS_SEED,
    DEFAULT_MAX_FRAMES_11,
    DEFAULT_WARP_SUFFIX_SEED,
    ENDING_PEACH_HOLD_FRAMES,
    ENDING_SETTLE_FRAMES,
    NATURAL_SETTLE_FRAMES,
    POWERON_BOOT_FRAMES,
    POWERON_SETTLE_FRAMES,
    Level11ReplayPolicy,
    Nes9ReplayPolicy,
)
from smb.ram import read_snapshot, reached_ending, segment_1_1_success
from smb.reactive_route import RouteProgressTracker
from smb.routes import ROUTE_WARP_ANY_PERCENT
from smb.rta_panel import VideoWriter, env_audio_rate, write_video
from smb.timing import build_timing_block, summarize_comparisons

WARP_MID_STATE = INTEGRATION_V0_DIR / "Level1_2_WarpMid.state"
LEVEL1_1_STATE = INTEGRATION_V0_DIR / "Level1_1.state"
DEFAULT_MAX_SUFFIX_FRAMES = 22_000
DEFAULT_MAX_CONTINUOUS_FRAMES = 25_000
_VideoWriter = VideoWriter
_write_video = write_video
_env_audio_rate = env_audio_rate


def _snapshot_dict(snap) -> dict[str, int]:
    return {
        "player_x": snap.player_x,
        "player_y": snap.player_y,
        "world": snap.world,
        "level": snap.level,
        "level_id": snap.level_id,
        "area_pointer": snap.area_pointer,
        "lives": snap.lives,
        "player_state": snap.player_state,
        "oper_mode": snap.oper_mode,
    }


def _run_policy_to_ending(
    env,
    *,
    seed_path: Path,
    max_frames: int,
    route_start_index: int,
    ending_settle_frames: int = ENDING_SETTLE_FRAMES,
    peach_hold_frames: int = 0,
    video: _VideoWriter | None = None,
    label: str = "",
) -> tuple[dict[str, Any], object | None]:
    """Replay an RLE seed until ending / death / timeout; optional video.

    ``ending_settle_frames`` is the success gate (stable oper_mode=2).
    ``peach_hold_frames`` is total post-ending idle for capture (bridge →
    Peach courtyard → full thank-you text). When larger than the settle
    gate, extra frames are held after success with label ``peach``.
    """
    policy = Nes9ReplayPolicy(
        seed_path=seed_path,
        action_size=int(env.action_space.shape[0]),
    )
    start = read_snapshot(env.get_ram())
    start_lives = start.lives
    progress = RouteProgressTracker(
        ROUTE_WARP_ANY_PERCENT,
        start_lives=start.lives,
        start_index=route_start_index,
    )
    max_x_by_level: dict[str, int] = {}
    outcome = "timeout"
    obs = None
    frame = 0

    while frame < max_frames:
        if policy.remaining == 0:
            outcome = "seed_exhausted"
            break
        tick = policy.step()
        obs, *_ = env.step(tick.action)
        frame += 1
        snap = read_snapshot(env.get_ram(), frame=frame)
        _write_video(
            video,
            obs,
            env=env,
            action=tick.action,
            label=label,
            snap=snap,
        )
        level_key = f"{snap.world + 1}-{snap.level + 1}"
        max_x_by_level[level_key] = max(
            max_x_by_level.get(level_key, 0), snap.player_x
        )

        progress.observe(snap, frame=frame)
        if snap.lives < start_lives or snap.dying:
            outcome = "death"
            break

        if reached_ending(env.get_ram(), start_lives=start_lives):
            outcome = "ending"
            break

    stable = 0
    peach_held = 0
    total_hold = max(ending_settle_frames, int(peach_hold_frames or 0))
    if outcome == "ending":
        idle = np.zeros(int(env.action_space.shape[0]), dtype=np.int8)
        for i in range(total_hold):
            obs, *_ = env.step(idle)
            hold_frame = frame + i + 1
            snap = read_snapshot(env.get_ram(), frame=hold_frame)
            phase = "ending" if i < ending_settle_frames else "peach"
            _write_video(
                video,
                obs,
                env=env,
                action=idle,
                label=phase,
                snap=snap,
            )
            if reached_ending(env.get_ram(), start_lives=start_lives):
                if i < ending_settle_frames:
                    stable += 1
                else:
                    peach_held += 1

    final = read_snapshot(env.get_ram(), frame=frame + total_hold)
    success = (
        outcome == "ending"
        and progress.complete
        and stable == ending_settle_frames
    )
    return (
        {
            "success": success,
            "outcome": outcome,
            "label": label,
            "policy_frames": frame,
            "ending_settle_frames": stable,
            "peach_hold_frames": peach_held,
            "ending_total_hold_frames": total_hold if outcome == "ending" else 0,
            "start": _snapshot_dict(start),
            "final": _snapshot_dict(final),
            "milestones": progress.completed,
            "route_progress": progress.report(),
            "max_x_by_level": max_x_by_level,
            "policy": policy.report(),
            "state_loads_during_policy": 0,
        },
        obs,
    )


def run_suffix_policy(
    env,
    *,
    seed_path: Path = DEFAULT_WARP_SUFFIX_SEED,
    max_frames: int = DEFAULT_MAX_SUFFIX_FRAMES,
    ending_settle_frames: int = ENDING_SETTLE_FRAMES,
    peach_hold_frames: int = 0,
    video: _VideoWriter | None = None,
) -> tuple[dict[str, Any], object | None]:
    """Run the no-reload mid-1-2-to-ending policy from the current state."""
    report, obs = _run_policy_to_ending(
        env,
        seed_path=seed_path,
        max_frames=max_frames,
        route_start_index=1,
        ending_settle_frames=ending_settle_frames,
        peach_hold_frames=peach_hold_frames,
        video=video,
        label="continuous_suffix",
    )
    report["state_loads_during_suffix"] = 0
    return report, obs


def _run_natural_1_1(
    env,
    *,
    seed_path: Path,
    max_frames: int,
    video: _VideoWriter | None = None,
) -> tuple[dict[str, Any], object | None]:
    obs, boot_frames = boot_to_ready(env)
    if obs is None:
        return {"success": False, "outcome": "boot_fail"}, obs
    if video is not None:
        # Boot frames were not recorded; capture from settle onward.
        pass
    obs = idle_n(env, NATURAL_SETTLE_FRAMES)
    policy = Level11ReplayPolicy(
        seed_path=seed_path,
        action_size=int(env.action_space.shape[0]),
    )
    start = read_snapshot(env.get_ram())
    max_x = start.player_x
    outcome = "timeout"
    frame = 0
    for frame in range(1, max_frames + 1):
        tick = policy.step()
        obs, *_ = env.step(tick.action)
        snap = read_snapshot(env.get_ram(), frame=frame)
        _write_video(
            video,
            obs,
            env=env,
            action=tick.action,
            label="1-1",
            snap=snap,
        )
        max_x = max(max_x, snap.player_x)
        if snap.lives < start.lives or snap.dying:
            outcome = "death"
            break
        if segment_1_1_success(
            env.get_ram(),
            start_lives=start.lives,
            max_player_x=max_x,
        ):
            outcome = "success"
            break
    return (
        {
            "success": outcome == "success",
            "outcome": outcome,
            "frames": frame,
            "boot_frames": boot_frames,
            "settle_frames": NATURAL_SETTLE_FRAMES,
            "max_player_x": max_x,
            "policy": policy.report(),
        },
        obs,
    )


def _fixed_boot(
    env,
    n_frames: int,
    *,
    video: _VideoWriter | None = None,
) -> tuple[object | None, int]:
    """Run exactly ``n_frames`` of the title boot script (pad with idle)."""
    idle = np.zeros(int(env.action_space.shape[0]), dtype=np.int8)
    obs = None
    frames = 0
    for scripted in boot_to_level1_script():
        obs, *_ = env.step(scripted.action)
        frames += 1
        snap = read_snapshot(env.get_ram(), frame=frames)
        _write_video(
            video,
            obs,
            env=env,
            action=scripted.action,
            label="boot",
            snap=snap,
        )
        if frames >= n_frames:
            return obs, frames
    while frames < n_frames:
        obs, *_ = env.step(idle)
        frames += 1
        snap = read_snapshot(env.get_ram(), frame=frames)
        _write_video(
            video,
            obs,
            env=env,
            action=idle,
            label="boot",
            snap=snap,
        )
    return obs, frames


def run_warp_finish(
    *,
    mode: str = "poweron",
    seed_11: Path = DEFAULT_1_1_SEED,
    seed_suffix: Path = DEFAULT_WARP_SUFFIX_SEED,
    seed_continuous: Path = DEFAULT_CONTINUOUS_SEED,
    settle_continuous: int = CONTINUOUS_SETTLE_FRAMES,
    boot_poweron: int = POWERON_BOOT_FRAMES,
    settle_poweron: int = POWERON_SETTLE_FRAMES,
    max_frames_11: int = DEFAULT_MAX_FRAMES_11,
    max_suffix_frames: int = DEFAULT_MAX_SUFFIX_FRAMES,
    max_continuous_frames: int = DEFAULT_MAX_CONTINUOUS_FRAMES,
    out_dir: Path | None = None,
    tag: str = "warp_finish",
    record_path: Path | None = None,
    record_scale: int = 3,
    record_hud: bool = True,
    record_audio: bool = True,
    intro_frames: int = DEFAULT_INTRO_FRAMES,
    intro_enabled: bool = True,
    peach_hold_frames: int | None = None,
) -> dict[str, Any]:
    """Run poweron / continuous / suffix / legacy chain finish attempt.

    When recording, defaults to ``ENDING_PEACH_HOLD_FRAMES`` so the MP4 holds
    through Peach + full thank-you text (not just Bowser drop / bridge walk).
    """
    configure_headless()
    out = out_dir or (RECORDINGS_DIR / "warp_finish")
    out.mkdir(parents=True, exist_ok=True)

    if mode in ("poweron", "continuous") and not seed_continuous.exists():
        raise SystemExit(
            f"missing continuous seed: {seed_continuous} "
            "(run: uv run python -m smb.scripts.fold_continuous_policy)"
        )

    if mode == "poweron":
        intervention = {
            "class": "Clean",
            "start": "power_on",
            "boot_frames": boot_poweron,
            "settle_frames": settle_poweron,
            "mid_attempt_state_loads": 0,
            "note": (
                "env.reset power-on; fixed boot script frames + idle "
                "phase-align; controller-only through 8-4 ending"
            ),
        }
        benchmark_eligible = True
    elif mode == "continuous":
        if not LEVEL1_1_STATE.exists():
            raise SystemExit(f"missing Level1_1 state: {LEVEL1_1_STATE}")
        intervention = {
            "class": "Clean",
            "initial_state": str(LEVEL1_1_STATE.relative_to(GAME_DIR.parent)),
            "settle_frames": settle_continuous,
            "mid_attempt_state_loads": 0,
            "note": (
                "Level1_1 start + fixed idle phase-align; controller-only "
                "through 8-4 ending"
            ),
        }
        benchmark_eligible = False
    elif mode == "suffix":
        if not WARP_MID_STATE.exists():
            raise SystemExit(f"missing suffix start state: {WARP_MID_STATE}")
        intervention = {
            "class": "Clean",
            "initial_state": str(WARP_MID_STATE.relative_to(GAME_DIR.parent)),
            "mid_attempt_state_loads": 0,
            "note": "published development start state; controller-only afterward",
        }
        benchmark_eligible = False
    elif mode == "chain":
        intervention = {
            "class": "development splice (not benchmark eligible)",
            "mid_attempt_state_loads": 1,
            "note": (
                "one mid-1-2 state splice follows natural 1-1; "
                "prefer --mode poweron"
            ),
        }
        benchmark_eligible = False
    else:
        raise SystemExit(f"unknown mode {mode!r}")

    env = make_env(GAME_V0, "NONE", GAME_DIR, render_mode="rgb_array")
    report: dict[str, Any] = {
        "mode": mode,
        "success": False,
        "route_id": "smb_warp_any_percent",
        "runtime_observation": "Bronze",
        "benchmark_eligible": benchmark_eligible,
        "intervention": intervention,
        "stages": {},
    }
    video: _VideoWriter | None = None
    obs = None
    # Recordings hold through Peach + thank-you text; dry runs keep the 120f gate.
    if peach_hold_frames is None:
        resolved_peach_hold = (
            ENDING_PEACH_HOLD_FRAMES if record_path is not None else 0
        )
    else:
        resolved_peach_hold = max(0, int(peach_hold_frames))
    try:
        obs, _ = reset_obs(env)

        if record_path is not None:
            if obs is None:
                obs = env.render()
            h, w = int(obs.shape[0]), int(obs.shape[1])
            audio_rate = _env_audio_rate(env) if record_audio else None
            video = _VideoWriter(
                record_path,
                width=w,
                height=h,
                scale=record_scale,
                audio_rate=audio_rate,
                hud=record_hud,
                route_label="SMB any%",
            )
            # Generic project intro (pre-roll). Gameplay still one continuous
            # power-on session — not a stitch of segment clips.
            if intro_enabled and intro_frames > 0:
                mode_summary = {
                    "poweron": "Clean power-on any% warp to 8-4 ending",
                    "continuous": "Level1_1 continuous any% warp to 8-4",
                    "chain": "Dev chain (1-1 + mid-1-2 splice) to ending",
                    "suffix": "Warp-mid suffix to World 8-4 ending",
                }.get(mode, f"SMB warp finish ({mode})")
                intro_lines = project_intro_lines(
                    game_title="Super Mario Bros. (NES)",
                    run_summary=mode_summary,
                    extra_lines=(
                        "HUD: frame timer, level/lives, NES buttons",
                    ),
                )
                video.write_intro(intro_lines, hold_frames=intro_frames)
            if obs is not None:
                _write_video(
                    video,
                    obs,
                    env=env,
                    action=None,
                    label="reset",
                )
            report["recording"] = {
                "path": str(record_path),
                "scale": record_scale,
                "hud": record_hud,
                "audio": audio_rate is not None,
                "audio_rate": audio_rate,
                "intro_frames": video.intro_frames,
                "continuous_run": True,
                "stitched_segments": False,
                "peach_hold_frames": resolved_peach_hold,
            }

        if mode == "poweron":
            obs, boot_frames = _fixed_boot(
                env, boot_poweron, video=video
            )
            idle = np.zeros(int(env.action_space.shape[0]), dtype=np.int8)
            for i in range(settle_poweron):
                obs, *_ = env.step(idle)
                snap = read_snapshot(env.get_ram(), frame=boot_frames + i + 1)
                _write_video(
                    video,
                    obs,
                    env=env,
                    action=idle,
                    label="settle",
                    snap=snap,
                )
            report["stages"]["boot"] = {
                "frames": boot_frames,
                "method": "fixed_boot_script",
                "settle_frames": settle_poweron,
            }
            policy_report, obs = _run_policy_to_ending(
                env,
                seed_path=seed_continuous,
                max_frames=max_continuous_frames,
                route_start_index=0,
                peach_hold_frames=resolved_peach_hold,
                video=video,
                label="poweron_to_ending",
            )
            report["stages"]["continuous"] = policy_report
            report["success"] = bool(policy_report["success"])
            report["outcome"] = policy_report["outcome"]
            report["exits_completed"] = len(policy_report["milestones"])
            report["state_loads_during_attempt"] = 0

        elif mode == "continuous":
            env.em.set_state(read_state_bytes(LEVEL1_1_STATE))
            idle = np.zeros(int(env.action_space.shape[0]), dtype=np.int8)
            if video is not None:
                frame0 = env.render()
                if frame0 is not None:
                    _write_video(
                        video,
                        frame0,
                        env=env,
                        action=None,
                        label="Level1_1",
                    )
            for i in range(settle_continuous):
                obs, *_ = env.step(idle)
                snap = read_snapshot(env.get_ram(), frame=i + 1)
                _write_video(
                    video,
                    obs,
                    env=env,
                    action=idle,
                    label="settle",
                    snap=snap,
                )
            report["stages"]["settle"] = {
                "frames": settle_continuous,
                "start_state": "Level1_1",
            }
            policy_report, obs = _run_policy_to_ending(
                env,
                seed_path=seed_continuous,
                max_frames=max_continuous_frames,
                route_start_index=0,
                peach_hold_frames=resolved_peach_hold,
                video=video,
                label="continuous_1_1_to_ending",
            )
            report["stages"]["continuous"] = policy_report
            report["success"] = bool(policy_report["success"])
            report["outcome"] = policy_report["outcome"]
            report["exits_completed"] = len(policy_report["milestones"])
            report["state_loads_during_attempt"] = 0

        elif mode == "chain":
            stage_11, obs = _run_natural_1_1(
                env,
                seed_path=seed_11,
                max_frames=max_frames_11,
                video=video,
            )
            report["stages"]["1-1"] = stage_11
            if not stage_11["success"]:
                report["outcome"] = f"1-1_{stage_11['outcome']}"
                return report
            if not WARP_MID_STATE.exists():
                raise SystemExit(f"missing suffix start state: {WARP_MID_STATE}")
            env.em.set_state(read_state_bytes(WARP_MID_STATE))
            report["suffix_entry"] = {
                "state": str(WARP_MID_STATE),
                "start_exit": "1-2",
            }
            suffix, obs = run_suffix_policy(
                env,
                seed_path=seed_suffix,
                max_frames=max_suffix_frames,
                peach_hold_frames=resolved_peach_hold,
                video=video,
            )
            report["stages"]["continuous_suffix"] = suffix
            report["success"] = bool(suffix["success"])
            report["outcome"] = suffix["outcome"]
            report["exits_completed"] = 1 + len(suffix["milestones"])
            report["state_loads_during_attempt"] = 1

        elif mode == "suffix":
            env.em.set_state(read_state_bytes(WARP_MID_STATE))
            report["suffix_entry"] = {
                "state": str(WARP_MID_STATE),
                "start_exit": "1-2",
            }
            suffix, obs = run_suffix_policy(
                env,
                seed_path=seed_suffix,
                max_frames=max_suffix_frames,
                peach_hold_frames=resolved_peach_hold,
                video=video,
            )
            report["stages"]["continuous_suffix"] = suffix
            report["success"] = bool(suffix["success"])
            report["outcome"] = suffix["outcome"]
            report["exits_completed"] = len(suffix["milestones"])
            report["state_loads_during_attempt"] = 0
        else:
            raise SystemExit(f"unknown mode {mode!r}")

        if obs is not None:
            suffix_name = "ending" if report["success"] else "fail"
            png = save_rgb_png(obs, out / f"{tag}_{suffix_name}.png")
            report["screenshot"] = str(png)
        if video is not None:
            report["video"] = str(record_path)
            report["video_frames"] = video.frames
            report["video_timer_frames"] = video.timer_frames
            report["video_intro_frames"] = video.intro_frames
            report["video_audio_samples"] = video.audio_samples

        # Attach named TAS/RTA timing contracts when we have a continuous path.
        continuous = report.get("stages", {}).get("continuous")
        if continuous and continuous.get("policy_frames") is not None:
            boot = report.get("stages", {}).get("boot") or {}
            settle = report.get("stages", {}).get("settle") or {}
            boot_frames = boot.get("frames") if mode == "poweron" else None
            settle_frames = (
                boot.get("settle_frames")
                if mode == "poweron"
                else settle.get("frames")
            )
            report["timing"] = build_timing_block(
                mode=mode,
                boot_frames=boot_frames,
                settle_frames=settle_frames,
                policy_frames_to_ending=int(continuous["policy_frames"]),
                milestones=continuous.get("milestones") or [],
            )
        return report
    finally:
        if video is not None:
            try:
                video.close()
            except Exception as exc:  # noqa: BLE001 — still write report
                report["video_error"] = str(exc)
        report_path = out / f"{tag}_report.json"
        write_json_report(report_path, report)
        stage = (
            report.get("stages", {}).get("continuous")
            or report.get("stages", {}).get("continuous_suffix")
            or {}
        )
        timing_note = ""
        if report.get("timing"):
            timing_note = " | " + summarize_comparisons(report["timing"])
        print(
            f"warp_finish mode={mode} outcome={report.get('outcome')} "
            f"success={report.get('success')} "
            f"exits={report.get('exits_completed', 0)} "
            f"policy_frames={stage.get('policy_frames', 0)} "
            f"report={report_path}"
            + (f" video={record_path}" if record_path else "")
            + timing_note,
            flush=True,
        )
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("poweron", "continuous", "chain", "suffix"),
        default="poweron",
        help="poweron=Clean reset-to-ending (default); continuous=Level1_1",
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed-11", type=Path, default=DEFAULT_1_1_SEED)
    parser.add_argument(
        "--seed-suffix",
        type=Path,
        default=DEFAULT_WARP_SUFFIX_SEED,
    )
    parser.add_argument(
        "--seed-continuous",
        type=Path,
        default=DEFAULT_CONTINUOUS_SEED,
    )
    parser.add_argument(
        "--settle-continuous",
        type=int,
        default=CONTINUOUS_SETTLE_FRAMES,
    )
    parser.add_argument(
        "--boot-poweron",
        type=int,
        default=POWERON_BOOT_FRAMES,
    )
    parser.add_argument(
        "--settle-poweron",
        type=int,
        default=POWERON_SETTLE_FRAMES,
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--record",
        action="store_true",
        help="Write MP4 under recordings/fullgame_replays/",
    )
    parser.add_argument(
        "--record-path",
        type=Path,
        default=None,
        help="Explicit MP4 path (implies recording)",
    )
    parser.add_argument("--record-scale", type=int, default=3)
    parser.add_argument(
        "--no-record-hud",
        action="store_true",
        help="Disable button / timestamp footer overlay on recordings",
    )
    parser.add_argument(
        "--no-record-audio",
        action="store_true",
        help="Disable native emulator audio mux into the MP4",
    )
    parser.add_argument(
        "--intro-frames",
        type=int,
        default=DEFAULT_INTRO_FRAMES,
        help=(
            "YouTube project intro hold frames at 60fps "
            f"(default {DEFAULT_INTRO_FRAMES}; 0 disables)"
        ),
    )
    parser.add_argument(
        "--no-intro",
        action="store_true",
        help="Skip the generic retro_rl YouTube intro slide",
    )
    parser.add_argument(
        "--peach-hold-frames",
        type=int,
        default=None,
        help=(
            "Post-ending idle frames for Peach/thank-you hold "
            f"(default {ENDING_PEACH_HOLD_FRAMES} when recording, 0 otherwise; "
            "success gate remains 120f)"
        ),
    )
    args = parser.parse_args()

    successes = 0
    trial_reports: list[dict[str, Any]] = []
    for trial in range(1, args.trials + 1):
        tag = (
            f"warp_finish_{args.mode}_t{trial}"
            if args.trials > 1
            else f"warp_finish_{args.mode}"
        )
        record_path = args.record_path
        if args.record and record_path is None:
            record_path = (
                FULLGAME_REPLAYS_DIR
                / f"smb_warp_any_percent_{args.mode}"
                f"{'' if args.trials == 1 else f'_t{trial}'}.mp4"
            )
        elif args.trials > 1 and record_path is not None:
            record_path = record_path.with_name(
                f"{record_path.stem}_t{trial}{record_path.suffix}"
            )

        report = run_warp_finish(
            mode=args.mode,
            seed_11=args.seed_11,
            seed_suffix=args.seed_suffix,
            seed_continuous=args.seed_continuous,
            settle_continuous=args.settle_continuous,
            boot_poweron=args.boot_poweron,
            settle_poweron=args.settle_poweron,
            out_dir=args.out_dir,
            tag=tag,
            record_path=record_path,
            record_scale=args.record_scale,
            record_hud=not args.no_record_hud,
            record_audio=not args.no_record_audio,
            intro_frames=0 if args.no_intro else max(0, args.intro_frames),
            intro_enabled=not args.no_intro and args.intro_frames != 0,
            peach_hold_frames=args.peach_hold_frames,
        )
        trial_reports.append(report)
        successes += int(bool(report.get("success")))

    if args.trials > 1:
        stage_key = (
            "continuous"
            if args.mode in ("poweron", "continuous")
            else "continuous_suffix"
        )
        first_stage = (
            trial_reports[0].get("stages", {}).get(stage_key, {})
            if trial_reports
            else {}
        )
        seed_path = (
            args.seed_continuous
            if args.mode in ("poweron", "continuous")
            else args.seed_suffix
        ).resolve()
        try:
            seed_label = str(seed_path.relative_to(GAME_DIR.parent.resolve()))
        except ValueError:
            seed_label = str(seed_path)
        summary = {
            "route_id": "smb_warp_any_percent",
            "mode": args.mode,
            "runtime_observation": "Bronze",
            "benchmark_eligible": bool(
                trial_reports[0].get("benchmark_eligible") if trial_reports else False
            ),
            "intervention": (
                trial_reports[0].get("intervention") if trial_reports else None
            ),
            "trials": args.trials,
            "successes": successes,
            "success_rate": successes / args.trials,
            "outcomes": [report.get("outcome") for report in trial_reports],
            "exits_completed": [
                report.get("exits_completed") for report in trial_reports
            ],
            "state_loads_during_attempt": [
                report.get("state_loads_during_attempt") for report in trial_reports
            ],
            "policy_seed": seed_label,
            "policy_frames": first_stage.get("policy_frames"),
            "ending_settle_frames": first_stage.get("ending_settle_frames"),
            "milestones": first_stage.get("milestones"),
            "final": first_stage.get("final"),
        }
        if args.mode == "chain":
            summary["prelude"] = {
                "exit_id": "1-1",
                "outcomes": [
                    report.get("stages", {}).get("1-1", {}).get("outcome")
                    for report in trial_reports
                ],
                "frames": [
                    report.get("stages", {}).get("1-1", {}).get("frames")
                    for report in trial_reports
                ],
            }
        if args.mode == "continuous":
            summary["settle_frames"] = args.settle_continuous
        if args.mode == "poweron":
            summary["boot_frames"] = args.boot_poweron
            summary["settle_frames"] = args.settle_poweron
        summary_dir = args.out_dir or (RECORDINGS_DIR / "warp_finish")
        write_json_report(
            summary_dir / f"warp_finish_{args.mode}_trials_report.json",
            summary,
        )
        print(f"trials {successes}/{args.trials} success", flush=True)
    raise SystemExit(0 if successes == args.trials else 1)


if __name__ == "__main__":
    main()

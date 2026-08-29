"""Record verified HappyLee warps slices as MP4 (HUD + audio).

Full raw FM2 power-on desyncs on fceumm. This replays the **adapted**
control-relative chain we already verify:

  Level1_1 → HL 1-1 → surface → HL 1-2 → W4
           → 4-1 control → HL 4-1 → 4-2 control → HL 4-2 → W8
           → 8-1 → 8-2 → hybrid 8-3 (natural) + 8-4 (flamexx) → axe

Uses the same ``_VideoWriter`` path as ``run_warp_finish --record`` (footer
timer, NES buttons, optional audio). Timer is gameplay-only (intro excluded).

```bash
# Full hybrid clear (~5:00.02; HL→8-2 + natural 8-3 + flamexx 8-4)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.record_happylee --to ending

# HL chain to World 8 (~2:05)
uv run python -m smb.scripts.record_happylee --to w8

# Stop at W4 only / isolated 1-1
uv run python -m smb.scripts.record_happylee --to w4
uv run python -m smb.scripts.record_happylee --to 1-1
```

Output default: ``recordings/tas_import/happylee_<target>.mp4``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless, write_json_report
from retro_harness.youtube_intro import DEFAULT_INTRO_FRAMES, project_intro_lines
from smb.paths import GAME_DIR, GAME_V0, MODELS_DIR, RECORDINGS_DIR
from smb.policy import expand_nes9_rle, load_nes9_rle_seed
from smb.ram import (
    PLAYER_STATE_DYING,
    WORLD_INDEX_8,
    read_snapshot,
    reached_ending,
    reached_world_4,
)
from smb.reactive_12 import is_surface_control
from smb.policy import ENDING_PEACH_HOLD_FRAMES
from smb.rta_panel import VideoWriter as _VideoWriter
from smb.rta_panel import env_audio_rate as _env_audio_rate
from smb.rta_panel import write_video as _write_video
from smb.tas.replay import IDLE, to_action9
from smb.tas.slice import (
    HL_1_1_SETTLE,
    is_4_1_control,
    is_4_2_control,
    is_8_1_control,
    is_8_2_control,
    is_8_3_control,
)
from smb.timing import NTSC_FPS, format_time

DEFAULT_OUT_DIR = RECORDINGS_DIR / "tas_import"
SEED_1_1 = MODELS_DIR / "smb_1_1_happylee_slice.json"
SEED_1_2 = MODELS_DIR / "smb_1_2_happylee_slice.json"
SEED_4_1 = MODELS_DIR / "smb_4_1_happylee_slice.json"
SEED_4_2 = MODELS_DIR / "smb_4_2_happylee_slice.json"
SEED_8_1 = MODELS_DIR / "smb_8_1_happylee_slice.json"
SEED_8_2 = MODELS_DIR / "smb_8_2_happylee_slice.json"
SEED_HYBRID_ENDING = MODELS_DIR / "smb_happylee_hybrid_v2_fx84.json"
SEED_HYBRID_V1 = MODELS_DIR / "smb_happylee_hybrid_ending.json"  # 18769f natural 8-4

# Brief hold after target so the last milestone is visible on video.
DEFAULT_TAIL_HOLD = 120

TARGETS = ("1-1", "w4", "w8", "ending")


def _step_video(
    env,
    action: np.ndarray,
    video: _VideoWriter | None,
    *,
    label: str,
    frame_i: int,
) -> tuple[Any, Any]:
    obs, *_ = env.step(action)
    snap = read_snapshot(env.get_ram(), frame=frame_i)
    _write_video(video, obs, env=env, action=action, label=label, snap=snap)
    return obs, snap


def _play_body(
    env,
    frames: list[list[int]],
    video: _VideoWriter | None,
    *,
    label: str,
    start_frame: int,
    start_lives: int,
    stop: Callable[[Any, Any], bool] | None = None,
) -> dict[str, Any]:
    """Play nes9 body; optional early stop predicate(snap, ram)."""
    death: int | None = None
    stop_at: int | None = None
    max_x = 0
    last_snap = read_snapshot(env.get_ram(), frame=start_frame)
    for i, fr in enumerate(frames):
        fnum = start_frame + i + 1
        obs, snap = _step_video(env, to_action9(fr), video, label=label, frame_i=fnum)
        last_snap = snap
        px = int(snap.player_x)
        if 0 < px < 20_000:
            max_x = max(max_x, px)
        ram = env.get_ram()
        if int(snap.lives) < start_lives or int(snap.player_state) == PLAYER_STATE_DYING:
            death = fnum
            break
        if stop is not None and stop(snap, ram):
            stop_at = fnum
            break
    played = (stop_at or death or (start_frame + len(frames))) - start_frame
    return {
        "label": label,
        "played": played,
        "body_len": len(frames),
        "death": death,
        "stop_at": stop_at,
        "max_x": max_x,
        "end_frame": start_frame + played,
        "end_snap": {
            "world": int(last_snap.world) + 1,
            "level": int(last_snap.level) + 1,
            "player_x": int(last_snap.player_x),
            "lives": int(last_snap.lives),
            "timer": int(last_snap.timer),
            "player_state": int(last_snap.player_state),
        },
    }


def _idle_until(
    env,
    video: _VideoWriter | None,
    *,
    pred: Callable[[Any], bool],
    label: str,
    start_frame: int,
    max_wait: int = 800,
) -> tuple[int, Any]:
    """Video-aware idle wait (records each frame; not the plain replay helper)."""
    wait = 0
    snap = read_snapshot(env.get_ram(), frame=start_frame)
    for _ in range(max_wait):
        snap = read_snapshot(env.get_ram(), frame=start_frame + wait)
        if pred(snap):
            return wait, snap
        _step_video(env, IDLE, video, label=label, frame_i=start_frame + wait + 1)
        wait += 1
    return wait, snap


def record_happylee(
    *,
    target: str = "w8",
    record_path: Path | None = None,
    seed_1_1: Path = SEED_1_1,
    seed_1_2: Path = SEED_1_2,
    seed_4_1: Path = SEED_4_1,
    seed_4_2: Path = SEED_4_2,
    settle: int = HL_1_1_SETTLE,
    record_scale: int = 3,
    record_hud: bool = True,
    record_audio: bool = True,
    intro_frames: int = DEFAULT_INTRO_FRAMES,
    intro_enabled: bool = True,
    tail_hold: int = DEFAULT_TAIL_HOLD,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Level1_1 HappyLee chain recording. Returns report dict."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")

    out_dir = out_dir or DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if record_path is None:
        record_path = out_dir / f"happylee_{target.replace('-', '_')}.mp4"

    bodies: dict[str, list[list[int]]] = {
        "1-1": expand_nes9_rle(load_nes9_rle_seed(seed_1_1)),
    }
    if target in ("w4", "w8"):
        bodies["1-2"] = expand_nes9_rle(load_nes9_rle_seed(seed_1_2))
    if target == "w8":
        bodies["4-1"] = expand_nes9_rle(load_nes9_rle_seed(seed_4_1))
        bodies["4-2"] = expand_nes9_rle(load_nes9_rle_seed(seed_4_2))

    configure_headless()
    env = make_env(GAME_V0, "Level1_1", GAME_DIR, render_mode="rgb_array")
    obs, _ = reset_obs(env)

    video: _VideoWriter | None = None
    audio_rate = _env_audio_rate(env) if record_audio else None
    if obs is None:
        obs = env.render()
    h, w = int(obs.shape[0]), int(obs.shape[1])
    video = _VideoWriter(
        record_path,
        width=w,
        height=h,
        scale=record_scale,
        audio_rate=audio_rate,
        hud=record_hud,
        route_label="SMB HL warps",
    )

    target_label = {
        "1-1": "HappyLee 1-1 (Level1_1 isolated)",
        "w4": "HappyLee chain → World 4 warp",
        "w8": "HappyLee chain → World 8 warp",
    }[target]
    if intro_enabled and intro_frames > 0:
        lines = project_intro_lines(
            game_title="Super Mario Bros. (NES)",
            run_summary=target_label,
            extra_lines=(
                "Adapted HappyLee #1715 bodies (fceumm control-relative)",
                "HUD: frame timer · level/lives · NES buttons",
            ),
        )
        video.write_intro(lines, hold_frames=intro_frames)

    _write_video(video, obs, env=env, action=None, label="reset")

    stages: dict[str, Any] = {}
    frame = 0
    success = False
    outcome = "incomplete"
    start_lives: int | None = None

    try:
        # Settle after Level1_1 (isolated HL recipe uses settle=2).
        for i in range(settle):
            frame += 1
            obs, snap = _step_video(
                env, IDLE, video, label="settle", frame_i=frame
            )
            if start_lives is None and 0 <= int(snap.lives) <= 8:
                start_lives = int(snap.lives)
        stages["settle"] = {"frames": settle}

        if start_lives is None:
            start_lives = int(read_snapshot(env.get_ram(), 0).lives)

        # --- 1-1 ---
        st = _play_body(
            env,
            bodies["1-1"],
            video,
            label="hl_1_1",
            start_frame=frame,
            start_lives=start_lives,
        )
        frame = st["end_frame"]
        stages["1_1"] = st
        if st["death"] is not None:
            outcome = "death_1_1"
            raise _Stop()

        if target == "1-1":
            # Flag / castle walk may still be finishing; short idle hold.
            for i in range(tail_hold):
                frame += 1
                _step_video(env, IDLE, video, label="tail", frame_i=frame)
            stages["tail_hold"] = tail_hold
            success = st["death"] is None and st["max_x"] >= 2500
            outcome = "clear_1_1" if success else "1_1_incomplete"
            raise _Stop()

        # Surface control gate
        wait12, ctrl = _idle_until(
            env,
            video,
            pred=is_surface_control,
            label="wait_1_2",
            start_frame=frame,
        )
        frame += wait12
        stages["ctrl_wait_1_2"] = wait12
        if not is_surface_control(ctrl):
            outcome = "surface_control_timeout"
            raise _Stop()

        # --- 1-2 → W4 ---
        st12 = _play_body(
            env,
            bodies["1-2"],
            video,
            label="hl_1_2",
            start_frame=frame,
            start_lives=int(ctrl.lives),
            stop=lambda _s, ram: reached_world_4(ram),
        )
        frame = st12["end_frame"]
        stages["1_2"] = st12
        if st12["death"] is not None:
            outcome = "death_1_2"
            raise _Stop()
        if not reached_world_4(env.get_ram()):
            outcome = "missed_w4"
            raise _Stop()

        if target == "w4":
            for i in range(tail_hold):
                frame += 1
                _step_video(env, IDLE, video, label="tail_w4", frame_i=frame)
            stages["tail_hold"] = tail_hold
            success = True
            outcome = "w4"
            raise _Stop()

        # 4-1 control
        wait41, ctrl41 = _idle_until(
            env,
            video,
            pred=is_4_1_control,
            label="wait_4_1",
            start_frame=frame,
        )
        frame += wait41
        stages["ctrl_wait_4_1"] = wait41
        if not is_4_1_control(ctrl41):
            outcome = "4_1_control_timeout"
            raise _Stop()

        # --- 4-1 ---
        def _left_to_4_2(snap, _ram) -> bool:
            return int(snap.world) == 3 and int(snap.level) == 1

        st41 = _play_body(
            env,
            bodies["4-1"],
            video,
            label="hl_4_1",
            start_frame=frame,
            start_lives=int(ctrl41.lives),
            stop=_left_to_4_2,
        )
        frame = st41["end_frame"]
        stages["4_1"] = st41
        if st41["death"] is not None:
            outcome = "death_4_1"
            raise _Stop()

        # 4-2 control
        wait42, ctrl42 = _idle_until(
            env,
            video,
            pred=is_4_2_control,
            label="wait_4_2",
            start_frame=frame,
        )
        frame += wait42
        stages["ctrl_wait_4_2"] = wait42
        if not is_4_2_control(ctrl42):
            outcome = "4_2_control_timeout"
            raise _Stop()

        # --- 4-2 → W8 ---
        def _to_w8(snap, _ram) -> bool:
            return int(snap.world) == WORLD_INDEX_8

        st42 = _play_body(
            env,
            bodies["4-2"],
            video,
            label="hl_4_2",
            start_frame=frame,
            start_lives=int(ctrl42.lives),
            stop=_to_w8,
        )
        frame = st42["end_frame"]
        stages["4_2"] = st42
        if st42["death"] is not None:
            outcome = "death_4_2"
            raise _Stop()
        if int(read_snapshot(env.get_ram(), frame).world) != WORLD_INDEX_8:
            outcome = "missed_w8"
            raise _Stop()

        for i in range(tail_hold):
            frame += 1
            _step_video(env, IDLE, video, label="tail_w8", frame_i=frame)
        stages["tail_hold"] = tail_hold
        success = True
        outcome = "w8"

    except _Stop:
        pass
    finally:
        end_snap = read_snapshot(env.get_ram(), frame=frame)
        timer_frames = video.timer_frames if video is not None else frame
        intro_n = video.intro_frames if video is not None else 0
        if video is not None:
            try:
                video.close()
            except Exception as exc:  # noqa: BLE001
                stages["video_close_error"] = str(exc)
        env.close()

    # Gameplay frame count excludes intro + tail for the "chain" clock.
    tail = int(stages.get("tail_hold") or 0)
    chain_frames = max(0, frame - tail)
    report: dict[str, Any] = {
        "success": success,
        "outcome": outcome,
        "target": target,
        "start_state": "Level1_1",
        "settle": settle,
        "total_gameplay_frames": frame,
        "chain_frames_to_target": chain_frames,
        "chain_time_ntsc": format_time(chain_frames, NTSC_FPS),
        "timer_frames_hud": timer_frames,
        "intro_frames": intro_n,
        "tail_hold": tail,
        "stages": {
            k: (
                {sk: sv for sk, sv in v.items() if sk != "end_snap"}
                if isinstance(v, dict)
                else v
            )
            for k, v in stages.items()
        },
        "end_snapshot": {
            "world": int(end_snap.world) + 1,
            "level": int(end_snap.level) + 1,
            "player_x": int(end_snap.player_x),
            "lives": int(end_snap.lives),
            "timer": int(end_snap.timer),
            "oper_mode": int(end_snap.oper_mode),
            "player_state": int(end_snap.player_state),
        },
        "seeds": {
            "1_1": str(seed_1_1),
            "1_2": str(seed_1_2) if "1-2" in bodies else None,
            "4_1": str(seed_4_1) if "4-1" in bodies else None,
            "4_2": str(seed_4_2) if "4-2" in bodies else None,
        },
        "recording": {
            "path": str(record_path),
            "scale": record_scale,
            "hud": record_hud,
            "audio": audio_rate is not None,
            "audio_rate": audio_rate,
            "exists": record_path.exists(),
            "bytes": record_path.stat().st_size if record_path.exists() else 0,
        },
        "note": (
            "Control-relative HappyLee bodies on fceumm (not raw #1715 power-on). "
            "L+R preserved. Full axe clear needs W8 fold (open)."
        ),
        "vs_natural_82": {
            "w4": 3884,
            "w8_as_8_1_entry": 12628,
        },
    }
    if success and target == "w4":
        report["delta_vs_natural_82_w4"] = 3884 - chain_frames
    if success and target == "w8":
        report["delta_vs_natural_82_8_1_entry"] = 12628 - chain_frames
        report["approx_vs_hl_probe_w8"] = 7512
        report["delta_vs_hl_probe_w8"] = chain_frames - 7512

    rep_path = record_path.with_suffix(".json")
    write_json_report(rep_path, report)
    report["report_path"] = str(rep_path)
    return report


class _Stop(Exception):
    """Control-flow for early success / failure exit from the try body."""


def record_hybrid_ending(
    *,
    seed_path: Path = SEED_HYBRID_ENDING,
    record_path: Path | None = None,
    record_scale: int = 3,
    record_hud: bool = True,
    record_audio: bool = True,
    intro_frames: int = DEFAULT_INTRO_FRAMES,
    intro_enabled: bool = True,
    peach_hold: int = ENDING_PEACH_HOLD_FRAMES,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Record continuous hybrid seed (HL→8-2 + natural 8-3/8-4) to axe + Peach.

    Seed already embeds Level1_1 settle=2 and fixed control waits. Do not add
    extra settle before playback.
    """
    out_dir = out_dir or DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if record_path is None:
        record_path = out_dir / "happylee_ending.mp4"

    frames = expand_nes9_rle(load_nes9_rle_seed(seed_path))
    configure_headless()
    env = make_env(GAME_V0, "Level1_1", GAME_DIR, render_mode="rgb_array")
    obs, _ = reset_obs(env)
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
        route_label="SMB HL hybrid",
    )
    if intro_enabled and intro_frames > 0:
        lines = project_intro_lines(
            game_title="Super Mario Bros. (NES)",
            run_summary="HappyLee hybrid any% → 8-4 axe",
            extra_lines=(
                "HL→8-2 + natural 8-3 + flamexx 8-4 (~5:00.02)",
                "fceumm control-relative (not raw #1715 power-on)",
            ),
        )
        video.write_intro(lines, hold_frames=intro_frames)
    _write_video(video, obs, env=env, action=None, label="reset")

    start_lives: int | None = None
    ending_frame: int | None = None
    death_frame: int | None = None
    frame = 0
    stages: dict[str, Any] = {"seed": str(seed_path), "seed_frames": len(frames)}

    try:
        for i, fr in enumerate(frames):
            frame = i + 1
            obs, snap = _step_video(
                env, to_action9(fr), video, label="hybrid", frame_i=frame
            )
            if start_lives is None and int(snap.oper_mode) == 1 and 0 <= int(snap.lives) <= 8:
                if int(snap.player_state) in (0, 7, 8) and 0 < int(snap.player_x) < 200:
                    start_lives = int(snap.lives)
            if start_lives is None:
                continue
            ram = env.get_ram()
            if reached_ending(ram, start_lives=start_lives):
                ending_frame = frame
                break
            if int(snap.lives) < start_lives or int(snap.player_state) == PLAYER_STATE_DYING:
                death_frame = frame
                break

        peach = 0
        if ending_frame is not None and peach_hold > 0:
            for j in range(peach_hold):
                frame += 1
                _step_video(env, IDLE, video, label="peach", frame_i=frame)
                peach += 1
        stages["ending_frame"] = ending_frame
        stages["death_frame"] = death_frame
        stages["peach_hold"] = peach
    finally:
        end_snap = read_snapshot(env.get_ram(), frame=frame)
        timer_frames = video.timer_frames
        intro_n = video.intro_frames
        try:
            video.close()
        except Exception as exc:  # noqa: BLE001
            stages["video_close_error"] = str(exc)
        env.close()

    success = ending_frame is not None and death_frame is None
    chain = ending_frame or frame
    splits_report = video.splits.report() if video.splits is not None else None
    report: dict[str, Any] = {
        "success": success,
        "outcome": "ending" if success else ("death" if death_frame else "incomplete"),
        "target": "ending",
        "start_state": "Level1_1",
        "settle": 0,  # baked into hybrid seed
        "total_gameplay_frames": frame,
        "chain_frames_to_target": chain,
        "chain_time_ntsc": format_time(chain, NTSC_FPS),
        "timer_frames_hud": timer_frames,
        "timer_frozen": bool(getattr(video, "_timer_frozen", False)),
        "intro_frames": intro_n,
        "rta_splits": splits_report,
        "stages": stages,
        "end_snapshot": {
            "world": int(end_snap.world) + 1,
            "level": int(end_snap.level) + 1,
            "player_x": int(end_snap.player_x),
            "lives": int(end_snap.lives),
            "timer": int(end_snap.timer),
            "oper_mode": int(end_snap.oper_mode),
            "player_state": int(end_snap.player_state),
        },
        "seeds": {"hybrid": str(seed_path)},
        "recording": {
            "path": str(record_path),
            "scale": record_scale,
            "hud": record_hud,
            "audio": audio_rate is not None,
            "audio_rate": audio_rate,
            "exists": record_path.exists(),
            "bytes": record_path.stat().st_size if record_path.exists() else 0,
        },
        "note": (
            "Hybrid v2: HappyLee through 8-2 + natural_82@15933 8-3 bridge + "
            "flamexx 8-4@15210 (2661f). Level1_1 continuous (not Clean power-on). "
            "18031f / ~5:00.02; pure HL 8-3 still open for true sub-5 / WR class."
        ),
        "vs_natural_82_ending": 21559,
        "delta_vs_natural_82_ending": 21559 - chain if success else None,
        "sub_5_min_budget_frames": 18_030,
        "delta_vs_sub_5": chain - 18_030 if success else None,
    }
    rep_path = record_path.with_suffix(".json")
    write_json_report(rep_path, report)
    report["report_path"] = str(rep_path)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--to",
        choices=TARGETS,
        default="ending",
        help="recording target milestone (default: ending)",
    )
    p.add_argument(
        "--record-path",
        type=Path,
        default=None,
        help="MP4 output path (default: recordings/tas_import/happylee_<to>.mp4)",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--settle", type=int, default=HL_1_1_SETTLE)
    p.add_argument("--record-scale", type=int, default=3)
    p.add_argument("--no-record-hud", action="store_true")
    p.add_argument("--no-record-audio", action="store_true")
    p.add_argument(
        "--intro-frames",
        type=int,
        default=DEFAULT_INTRO_FRAMES,
        help=f"intro hold frames (default {DEFAULT_INTRO_FRAMES}; 0 disables)",
    )
    p.add_argument("--no-intro", action="store_true")
    p.add_argument(
        "--tail-hold",
        type=int,
        default=DEFAULT_TAIL_HOLD,
        help=f"idle frames after non-ending targets (default {DEFAULT_TAIL_HOLD})",
    )
    p.add_argument(
        "--peach-hold",
        type=int,
        default=ENDING_PEACH_HOLD_FRAMES,
        help=f"post-ending Peach hold (default {ENDING_PEACH_HOLD_FRAMES})",
    )
    p.add_argument("--seed-1-1", type=Path, default=SEED_1_1)
    p.add_argument("--seed-1-2", type=Path, default=SEED_1_2)
    p.add_argument("--seed-4-1", type=Path, default=SEED_4_1)
    p.add_argument("--seed-4-2", type=Path, default=SEED_4_2)
    p.add_argument("--seed-hybrid", type=Path, default=SEED_HYBRID_ENDING)
    args = p.parse_args(argv)

    if args.to == "ending":
        report = record_hybrid_ending(
            seed_path=args.seed_hybrid,
            record_path=args.record_path,
            record_scale=args.record_scale,
            record_hud=not args.no_record_hud,
            record_audio=not args.no_record_audio,
            intro_frames=0 if args.no_intro else max(0, args.intro_frames),
            intro_enabled=not args.no_intro and args.intro_frames != 0,
            peach_hold=max(0, args.peach_hold),
            out_dir=args.out_dir,
        )
    else:
        report = record_happylee(
            target=args.to,
            record_path=args.record_path,
            seed_1_1=args.seed_1_1,
            seed_1_2=args.seed_1_2,
            seed_4_1=args.seed_4_1,
            seed_4_2=args.seed_4_2,
            settle=args.settle,
            record_scale=args.record_scale,
            record_hud=not args.no_record_hud,
            record_audio=not args.no_record_audio,
            intro_frames=0 if args.no_intro else max(0, args.intro_frames),
            intro_enabled=not args.no_intro and args.intro_frames != 0,
            tail_hold=max(0, args.tail_hold),
            out_dir=args.out_dir,
        )
    # Compact stdout for agents; full detail in JSON.
    summary = {
        "success": report["success"],
        "outcome": report["outcome"],
        "target": report["target"],
        "chain_frames_to_target": report["chain_frames_to_target"],
        "chain_time_ntsc": report["chain_time_ntsc"],
        "recording": report["recording"]["path"],
        "bytes": report["recording"]["bytes"],
        "report": report.get("report_path"),
    }
    for k in (
        "delta_vs_natural_82_8_1_entry",
        "delta_vs_hl_probe_w8",
        "delta_vs_natural_82_ending",
        "delta_vs_sub_5",
    ):
        if k in report:
            summary[k] = report[k]
    print(json.dumps(summary, indent=2))
    if not report["success"]:
        print(json.dumps(report.get("stages", {}), indent=2), file=sys.stderr)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

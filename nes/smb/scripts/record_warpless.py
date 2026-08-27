"""Record / play HappyLee & Mars608 warpless #3728M through a named leave.

Uses **only** exported cuts from ``happylee_mars608_warpless_3728M.fm2``.
``play_warpless_to`` is table-driven (``WARPLESS_LEGS``). Do not load #1715M
warp seeds or the hand-built ``smb_1_2_flag`` body.

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.record_warpless --to 1-4
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.record_warpless --to 1-4 --record
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless, write_json_report
from retro_harness.youtube_intro import DEFAULT_INTRO_FRAMES, project_intro_lines
from smb.paths import GAME_DIR, GAME_V0, RECORDINGS_DIR
from smb.ram import read_snapshot
from smb.scripts.run_warp_finish import (
    _VideoWriter,
    _env_audio_rate,
    _write_video,
)
from smb.tas.replay import IDLE
from smb.tas.warpless import (
    WARPLESS_LEGS,
    WL_1_1_SETTLE,
    play_warpless_to,
    slices_present,
)
from smb.timing import NTSC_FPS, format_time

DEFAULT_OUT_DIR = RECORDINGS_DIR / "tas_import" / "warpless_3728M"
DEFAULT_TAIL_HOLD = 120
_STAGE_IDS = tuple(leg.id for leg in WARPLESS_LEGS)

_TARGET_LABELS = {
    "1-1": "Warpless #3728M 1-1 (Level1_1)",
    "1-2": "Warpless #3728M 1-1 → 1-2 flag → 1-3",
    "1-3": "Warpless #3728M 1-1 → 1-2 flag → 1-3 → 1-4",
    "1-4": "Warpless #3728M 1-1 → 1-4 castle → 2-1",
    "2-1": "Warpless #3728M 1-1 → 2-1 → 2-2",
    "2-2": "Warpless #3728M 1-1 → 2-2 flag pipe → 2-3",
}


def record_warpless(
    *,
    target: str = "1-3",
    record: bool = False,
    record_path: Path | None = None,
    settle: int = WL_1_1_SETTLE,
    record_scale: int = 3,
    record_hud: bool = True,
    record_audio: bool = True,
    intro_frames: int = DEFAULT_INTRO_FRAMES,
    intro_enabled: bool = True,
    tail_hold: int = DEFAULT_TAIL_HOLD,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Play (and optionally record) the #3728M chain through *target*."""
    if target not in _STAGE_IDS:
        raise ValueError(f"target must be one of 1-1…8-4, got {target!r}")
    if not slices_present(target):
        raise FileNotFoundError(
            f"missing warpless seeds through {target}; "
            f"export with: uv run python -m smb.scripts.annotate_fm2 --search {target} --export"
        )

    out_dir = out_dir or DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if record and record_path is None:
        record_path = out_dir / f"warpless_{target.replace('-', '_')}.mp4"

    configure_headless()
    env = make_env(GAME_V0, "Level1_1", GAME_DIR, render_mode="rgb_array")
    obs, _ = reset_obs(env)
    if obs is None:
        obs = env.render()

    video: _VideoWriter | None = None
    audio_rate: int | None = None
    if record:
        assert record_path is not None
        h, w = int(obs.shape[0]), int(obs.shape[1])
        audio_rate = _env_audio_rate(env) if record_audio else None
        video = _VideoWriter(
            record_path,
            width=w,
            height=h,
            scale=record_scale,
            audio_rate=audio_rate,
            hud=record_hud,
            route_label="SMB 32-exit #3728M",
        )
        if intro_enabled and intro_frames > 0:
            lines = project_intro_lines(
                game_title="Super Mario Bros. (NES)",
                run_summary=_TARGET_LABELS.get(
                    target, f"Warpless #3728M 1-1 → {target}"
                ),
                extra_lines=(
                    "HappyLee & Mars608 warpless #3728M (same-file cuts)",
                    "HUD: frame timer · level/lives · NES buttons",
                ),
            )
            video.write_intro(lines, hold_frames=intro_frames)
        _write_video(video, obs, env=env, action=None, label="reset")

    def on_step(step_obs: Any, action: Any, label: str, frame_i: int) -> None:
        if video is None:
            return
        snap = read_snapshot(env.get_ram(), frame=frame_i)
        _write_video(video, step_obs, env=env, action=action, label=label, snap=snap)

    report: dict[str, Any] = {}
    end_snap = None
    try:
        report = play_warpless_to(
            env,
            to=target,
            settle=settle,
            on_step=on_step if video is not None else None,
        )
        frame = int(report.get("frame") or 0)
        if report.get("ok") and tail_hold > 0:
            for _ in range(tail_hold):
                frame += 1
                obs, *_ = env.step(IDLE)
                if video is not None:
                    snap = read_snapshot(env.get_ram(), frame=frame)
                    _write_video(
                        video, obs, env=env, action=IDLE, label="tail", snap=snap
                    )
            report["tail_hold"] = tail_hold
            report["frame"] = frame
        else:
            report["tail_hold"] = 0
        end_snap = read_snapshot(env.get_ram(), frame=int(report.get("frame") or 0))
    finally:
        if end_snap is None:
            try:
                end_snap = read_snapshot(env.get_ram(), 0)
            except Exception:
                end_snap = None
        if video is not None:
            try:
                video.close()
            except Exception as exc:  # noqa: BLE001
                report.setdefault("stages", {})["video_close_error"] = str(exc)
        env.close()
    if end_snap is None:
        raise RuntimeError("warpless chain closed before a snapshot was read")

    chain = max(0, int(report.get("frame") or 0) - int(report.get("tail_hold") or 0))
    rec_path = record_path if record else None
    report.update(
        {
            "chain_frames_to_target": chain,
            "chain_time_ntsc": format_time(chain, NTSC_FPS),
            "end_snapshot": {
                "world": int(end_snap.world) + 1,
                "dash_level": int(end_snap.dash_level) + 1,
                "player_x": int(end_snap.player_x),
                "player_y": int(end_snap.player_y),
                "lives": int(end_snap.lives),
                "timer": int(end_snap.timer),
                "player_state": int(end_snap.player_state),
            },
            "recording": {
                "enabled": bool(record),
                "path": str(rec_path) if rec_path is not None else None,
                "scale": record_scale if record else None,
                "hud": record_hud if record else None,
                "audio": audio_rate is not None if record else False,
                "audio_rate": audio_rate,
                "exists": bool(rec_path is not None and rec_path.exists()),
                "bytes": rec_path.stat().st_size if rec_path is not None and rec_path.exists() else 0,
            },
            "note": (
                "Control-relative #3728M bodies on fceumm. Same movie for every "
                "leg. L+R preserved. Isolated Level1_3 is a different phase."
            ),
        }
    )
    if rec_path is not None:
        rep_path = rec_path.with_suffix(".json")
    else:
        rep_path = out_dir / f"warpless_{target.replace('-', '_')}_play.json"
    write_json_report(rep_path, report)
    report["report_path"] = str(rep_path)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--to",
        choices=_STAGE_IDS,
        default="1-3",
        help="leave milestone (default: 1-3 → 1-4 control)",
    )
    p.add_argument(
        "--record",
        action="store_true",
        help="write HUD+audio MP4 under recordings/tas_import/warpless_3728M/",
    )
    p.add_argument("--record-path", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--settle", type=int, default=WL_1_1_SETTLE)
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
        help=f"idle frames after the leave (default {DEFAULT_TAIL_HOLD})",
    )
    args = p.parse_args(argv)

    try:
        report = record_warpless(
            target=args.to,
            record=args.record,
            record_path=args.record_path,
            settle=args.settle,
            record_scale=args.record_scale,
            record_hud=not args.no_record_hud,
            record_audio=not args.no_record_audio,
            intro_frames=0 if args.no_intro else max(0, args.intro_frames),
            intro_enabled=not args.no_intro and args.intro_frames != 0,
            tail_hold=max(0, args.tail_hold),
            out_dir=args.out_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = {
        "success": report["success"],
        "outcome": report["outcome"],
        "target": report["target"],
        "chain_frames_to_target": report["chain_frames_to_target"],
        "chain_time_ntsc": report["chain_time_ntsc"],
        "end_snapshot": report.get("end_snapshot"),
        "recording": report["recording"]["path"],
        "bytes": report["recording"]["bytes"],
        "report": report.get("report_path"),
    }
    print(json.dumps(summary, indent=2))
    if not report["success"]:
        print(json.dumps(report.get("stages", {}), indent=2), file=sys.stderr)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

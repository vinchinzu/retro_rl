#!/usr/bin/env python3
"""Post-Torizo Morph+Bombs → Gauntlet Entrance (side quest, not continuous).

Pin: ``scratch/post_torizo_parlor_continuous.state`` (Flyway door in Parlor).
Climb Parlor's right shaft to Landing, long IBJ the cliff, bomb Obstacle A,
walk through the top-left door.

```bash
# Pin RAM + screenshot
uv run python snes/super_metroid/scripts/probe/gauntlet_bombs.py dump

# One hop from the pin (optional --from parlor|landing|ship|ibj|obstacle|cave)
# --stop-at flyway|parlor_top|landing|cave_exit|ship|ibj_high|ledge|lip|wall
uv run python snes/super_metroid/scripts/probe/gauntlet_bombs.py run --from ship --stop-at ibj_high --no-video

# Proof video (MP4 under recordings/)
uv run python snes/super_metroid/scripts/probe/gauntlet_bombs.py record

# Two runs, overwrite scratch/gauntlet_bombs_dual.json
uv run python snes/super_metroid/scripts/probe/gauntlet_bombs.py dual --no-video
```

Leave proof is RAM + JSON; ``record`` additionally writes an MP4 because the
user asked for video on this side quest. Do not STATUS-promote.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

from retro_harness.actions import idle_action
from retro_harness.video import VideoCaptureConfig, VideoRecorder
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.combat.probe import open_state_env, write_json_report
from super_metroid.dev.common import save_dev_state
from super_metroid.hop_glance import final_from_state, grade_final
from super_metroid.paths import GAME_DIR, RECORDINGS_DIR, SCRATCH_STATE_DIR
from super_metroid.ram import parse_env_state, probe_pin
from super_metroid.room_timer import format_segment_time
from super_metroid.routes.kpdr.gauntlet import (
    play_landing_to_gauntlet,
    play_parlor_to_gauntlet,
)
from super_metroid.routes.kpdr.gauntlet.geometry import PARLOR_TO_GAUNTLET
from super_metroid.routes.kpdr.room_ids import ROOM_GAUNTLET_ENTRANCE
from super_metroid.routes.skills.geometry import PhaseStop

DEFAULT_SOURCE = SCRATCH_STATE_DIR / "post_torizo_parlor_continuous.state"
DEFAULT_OUT = SCRATCH_STATE_DIR / "post_gauntlet_entrance.state"
DEFAULT_REPORT = GAME_DIR / "scratch" / "gauntlet_bombs.json"
DEFAULT_DUAL = GAME_DIR / "scratch" / "gauntlet_bombs_dual.json"
DEFAULT_VIDEO = RECORDINGS_DIR / "gauntlet_bombs.mp4"
DEFAULT_SHOT = GAME_DIR / "scratch" / "gauntlet_bombs_pin.png"
HOP = "parlor_to_gauntlet"
BOOT_SETTLE = 5

STOP_CHOICES = (
    "flyway",
    "parlor_top",
    "landing",
    "cave_exit",
    "ship",
    "ibj_high",
    "ledge",
    "lip",
    "wall",
)
FROM_CHOICES = ("parlor", "landing", "ship", "ibj", "lip", "obstacle", "cave")
NAMED_SOURCES = {
    "parlor": DEFAULT_SOURCE,
    "post-torizo": DEFAULT_SOURCE,
    "landing": SCRATCH_STATE_DIR / "gauntlet_landing_floor.state",
    "ship": SCRATCH_STATE_DIR / "gauntlet_ship_floor.state",
    "ibj": SCRATCH_STATE_DIR / "gauntlet_ibj_peak2.state",
    "lip": SCRATCH_STATE_DIR / "gauntlet_lip.state",
    "obstacle": SCRATCH_STATE_DIR / "gauntlet_a_up8_L12.state",
    "cave": SCRATCH_STATE_DIR / "gauntlet_ba_L_bomb.state",
    "wj2": SCRATCH_STATE_DIR / "gauntlet_wj2.state",
}


class _Sess:
    """ControllerSession + optional video writer (every frame)."""

    def __init__(
        self,
        env: Any,
        assist: UnlimitedResourcesAssist | None,
        writer: VideoRecorder | None = None,
    ) -> None:
        self.env = env
        self.assist = assist
        self.writer = writer
        self.frame = 0
        self.state = parse_env_state(env, mode="nav")
        self.min_y = int(self.state.samus_y)
        self.min_y_xy = [int(self.state.samus_x), int(self.state.samus_y)]
        self.last_action = None

    def step(self, action, reason: str = ""):
        del reason
        obs, *_ = self.env.step(action)
        self.frame += 1
        st = parse_env_state(self.env, frame=self.frame, mode="nav")
        if self.assist is not None:
            try:
                self.assist.apply(self.env.data, st)
            except Exception:  # noqa: BLE001
                try:
                    self.assist.apply(self.env, st)
                except Exception:  # noqa: BLE001
                    pass
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        self.last_action = action
        y = int(self.state.samus_y)
        if y < self.min_y:
            self.min_y = y
            self.min_y_xy = [int(self.state.samus_x), y]
        if self.writer is not None:
            self.writer.write_from_env(
                self.env,
                obs,
                action=action,
                frame_index=self.frame,
                room_id=int(self.state.room_id),
            )
        return self.state


def _snap(st: Any, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out = final_from_state(st)
    out.update(
        {
            "items": f"0x{int(st.collected_items):04X}",
            "beams": f"0x{int(st.collected_beams):04X}",
            "bombs": bool(getattr(st, "bombs", False)),
            "morph": bool(int(st.collected_items) & 0x0004),
            "pose": int(st.pose),
            "facing": int(st.facing),
            "vx": int(st.velocity_x),
            "vy": int(st.velocity_y),
            "probePin": probe_pin(st),
        }
    )
    if extra:
        out.update(extra)
    return out


def _save_shot(env: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(env.render()).save(path)
    return path


def _open(source: Path, *, settle: int = BOOT_SETTLE):
    return open_state_env(
        source,
        settle=settle,
        missing_hint="Need post_torizo_parlor_continuous.state",
    )


def dump_pin(source: Path, shot: Path, frames: int = 0) -> dict[str, Any]:
    env, resolved = _open(source)
    try:
        for _ in range(max(0, frames)):
            env.step(idle_action())
        st = parse_env_state(env, mode="nav")
        png = _save_shot(env, shot)
        report = {
            "kind": "gauntlet_bombs_dump",
            "source": resolved,
            "shot": str(png),
            "final": _snap(st),
        }
        write_json_report(report, None)
        return report
    finally:
        env.close()


def _play(session: Any, *, from_kind: str, stop_at: str | None) -> None:
    """Parlor is the full hop; landing/ship/lip skip the Alcatraz prefix."""
    if from_kind == "cave":
        play_landing_to_gauntlet(session, stop_at=stop_at, start_at="cave")
        return
    if from_kind == "obstacle":
        play_landing_to_gauntlet(session, stop_at=stop_at, start_at="obstacle_a")
        return
    if from_kind == "lip":
        play_landing_to_gauntlet(session, stop_at=stop_at, start_at="lip")
        return
    if from_kind == "ibj":
        play_landing_to_gauntlet(session, stop_at=stop_at, start_at="ibj_high")
        return
    if from_kind in ("landing", "ship"):
        play_landing_to_gauntlet(session, stop_at=stop_at)
        return
    play_parlor_to_gauntlet(session, stop_at=stop_at)


def _resolve_source(source: Path | str, from_kind: str) -> Path:
    token = str(source)
    if token in NAMED_SOURCES:
        return NAMED_SOURCES[token]
    path = Path(token)
    if path == DEFAULT_SOURCE and from_kind in NAMED_SOURCES and from_kind != "parlor":
        return NAMED_SOURCES[from_kind]
    return path


def run_once(
    source: Path,
    *,
    from_kind: str = "parlor",
    stop_at: str | None = None,
    video_path: Path | None = None,
    out_pin: Path | None = None,
    audio: bool = False,
    scale: int = 2,
    crf: int = 20,
) -> dict[str, Any]:
    settle = 2 if from_kind in ("ibj", "obstacle", "cave") else BOOT_SETTLE
    env, resolved = _open(source, settle=settle)
    assist = UnlimitedResourcesAssist()
    writer: VideoRecorder | None = None
    error: str | None = None
    phase_stop: str | None = None
    t0 = time.perf_counter()
    try:
        if video_path is not None:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            obs = env.render()
            audio_rate = None
            if audio:
                audio_rate = int(env.em.get_audio_rate())  # type: ignore[attr-defined]
            writer = VideoRecorder(
                video_path,
                width=int(obs.shape[1]),
                height=int(obs.shape[0]),
                config=VideoCaptureConfig(
                    fps=60, scale=scale, crf=crf, audio=audio, footer=True
                ),
                audio_rate=audio_rate,
            )
            writer.write_from_env(
                env, obs, action=None, frame_index=0, room_id=int(
                    parse_env_state(env, mode="nav").room_id
                ),
            )
        session = _Sess(env, assist, writer)
        start = _snap(session.state)
        try:
            _play(session, from_kind=from_kind, stop_at=stop_at)
        except PhaseStop as exc:
            phase_stop = exc.phase
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        final = _snap(
            session.state,
            extra={
                "minY": session.min_y,
                "minYxy": session.min_y_xy,
                "frames": session.frame,
            },
        )
        if out_pin is not None and error is None:
            save_dev_state(env, out_pin)
        png = None
        if error is not None or phase_stop is not None:
            png = GAME_DIR / "scratch" / "gauntlet_bombs_red.png"
            _save_shot(env, png)
        elif int(session.state.room_id) == ROOM_GAUNTLET_ENTRANCE:
            png = GAME_DIR / "scratch" / "gauntlet_bombs_leave.png"
            _save_shot(env, png)
    finally:
        encoded = 0
        if writer is not None:
            encoded = writer.frames
            writer.close()
        env.close()

    elapsed = time.perf_counter() - t0
    misses = list(grade_final(final, PARLOR_TO_GAUNTLET)) if error is None else ["error"]
    success = (
        error is None
        and phase_stop is None
        and int(session.state.room_id) == ROOM_GAUNTLET_ENTRANCE
        and not misses
    )
    timing = format_segment_time(session.frame)
    report = {
        "kind": "gauntlet_bombs_probe",
        "hop": HOP,
        "source": resolved,
        "success": success,
        "phaseStop": phase_stop,
        "error": error,
        "misses": misses,
        "frames": session.frame,
        "seconds": timing["seconds"],
        "clock": timing["clock"],
        "timing": timing,
        "encodedFrames": encoded,
        "elapsedSec": round(elapsed, 2),
        "video": str(video_path) if video_path is not None else None,
        "outPin": str(out_pin) if out_pin is not None else None,
        "redPng": str(png) if png is not None else None,
        "start": start,
        "final": final,
        "note": (
            "Side quest from post-Torizo parlor pin. Not continuous evidence. "
            "Do not STATUS-promote."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("cmd", choices=("dump", "run", "record", "dual"))
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE))
    parser.add_argument("--from", dest="from_kind", choices=FROM_CHOICES, default="parlor")
    parser.add_argument("--stop-at", choices=STOP_CHOICES, default=None)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--shot", type=Path, default=DEFAULT_SHOT)
    parser.add_argument("--frames", type=int, default=0, help="dump: extra idle")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--audio", action="store_true")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--crf", type=int, default=20)
    args = parser.parse_args()

    source = _resolve_source(args.source, args.from_kind)
    out_pin = args.out
    if out_pin is None and args.cmd in ("run", "record", "dual") and args.stop_at is None:
        out_pin = DEFAULT_OUT

    if args.cmd == "dump":
        dump_pin(source, args.shot, frames=args.frames)
        return

    video = None
    if args.cmd == "record" or (args.cmd in ("run", "dual") and not args.no_video):
        if args.cmd == "record" or not args.no_video:
            video = args.video if args.cmd == "record" else None
    if args.cmd == "record":
        video = args.video

    if args.cmd == "dual":
        rows = []
        for i in range(2):
            print(f"[gauntlet] dual row {i + 1}/2", flush=True)
            row = run_once(
                source,
                from_kind=args.from_kind,
                stop_at=args.stop_at,
                video_path=None,
                out_pin=out_pin if i == 0 else None,
                audio=False,
            )
            rows.append(row)
            print(json.dumps({k: row[k] for k in (
                "success", "phaseStop", "error", "frames", "misses", "final"
            )}, indent=2), flush=True)
        match = (
            rows[0]["success"]
            and rows[1]["success"]
            and rows[0]["frames"] == rows[1]["frames"]
        )
        dual = {
            "kind": "gauntlet_bombs_dual",
            "match": match,
            "rows": rows,
            "note": "Pin dual only — not continuous evidence.",
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        dual_path = DEFAULT_DUAL if args.report == DEFAULT_REPORT else args.report
        dual_path.parent.mkdir(parents=True, exist_ok=True)
        dual_path.write_text(json.dumps(dual, indent=2) + "\n")
        print(json.dumps({"match": match, "frames": [r["frames"] for r in rows]}, indent=2))
        print(f"[gauntlet] wrote {dual_path}", flush=True)
        if not rows[0]["success"]:
            raise SystemExit(1)
        return

    print(
        f"[gauntlet] cmd={args.cmd} from={args.from_kind} source={source} "
        f"stop={args.stop_at}",
        flush=True,
    )
    report = run_once(
        source,
        from_kind=args.from_kind,
        stop_at=args.stop_at,
        video_path=video,
        out_pin=out_pin,
        audio=args.audio,
        scale=args.scale,
        crf=args.crf,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    slim = {k: report[k] for k in (
        "kind", "success", "phaseStop", "error", "misses", "frames",
        "clock", "elapsedSec", "video", "final", "redPng", "note",
    ) if k in report}
    print(json.dumps(slim, indent=2), flush=True)
    print(f"[gauntlet] wrote {args.report}", flush=True)
    if video is not None:
        print(f"[gauntlet] video {video}", flush=True)
    if not report["success"] and args.stop_at is None:
        raise SystemExit(1)
    if args.stop_at and report.get("error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

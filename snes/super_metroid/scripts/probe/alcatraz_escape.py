#!/usr/bin/env python3
"""Post-Torizo Alcatraz left-chimney WJ + instant-morph roll-out.

Pin: ``scratch/post_torizo_parlor_continuous.state`` (Flyway door in Parlor).
Goal: land the shaft lip (y<=210) and roll left through the morph opening.

```bash
uv run python snes/super_metroid/scripts/probe/alcatraz_escape.py run --no-video
uv run python snes/super_metroid/scripts/probe/alcatraz_escape.py dual --no-video
uv run python snes/super_metroid/scripts/probe/alcatraz_escape.py record
```

Overwrite ``scratch/alcatraz_escape_dual.json``. Do not STATUS-promote.
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
from super_metroid.hop_glance import final_from_state
from super_metroid.paths import GAME_DIR, RECORDINGS_DIR, SCRATCH_STATE_DIR
from super_metroid.ram import parse_env_state, probe_pin
from super_metroid.room_timer import format_segment_time
from super_metroid.routes.kpdr.alcatraz_escape import (
    SHAFT_LIP_Y,
    at_alcatraz_rollout,
    play_alcatraz_escape,
)

DEFAULT_SOURCE = SCRATCH_STATE_DIR / "post_torizo_parlor_continuous.state"
DEFAULT_OUT = SCRATCH_STATE_DIR / "post_alcatraz_escape.state"
DEFAULT_REPORT = GAME_DIR / "scratch" / "alcatraz_escape.json"
DEFAULT_DUAL = GAME_DIR / "scratch" / "alcatraz_escape_dual.json"
DEFAULT_VIDEO = RECORDINGS_DIR / "alcatraz_escape.mp4"
DEFAULT_SHOT = GAME_DIR / "scratch" / "alcatraz_escape_leave.png"
BOOT_SETTLE = 5
HOP = "alcatraz_escape"


class _Sess:
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
        self.trace: list[dict[str, Any]] = []

    def step(self, action, reason: str = ""):
        obs, *_ = self.env.step(action)
        self.frame += 1
        if self.assist is not None:
            st = parse_env_state(self.env, frame=self.frame, mode="nav")
            try:
                self.assist.apply(self.env.data, st)
            except Exception:  # noqa: BLE001
                try:
                    self.assist.apply(self.env, st)
                except Exception:  # noqa: BLE001
                    pass
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        y = int(self.state.samus_y)
        if y < self.min_y:
            self.min_y = y
            self.min_y_xy = [int(self.state.samus_x), y]
        if self.frame % 6 == 0 or int(self.state.pose) in (131, 132) or "morph" in reason:
            self.trace.append(
                {
                    "f": self.frame,
                    "x": int(self.state.samus_x),
                    "y": y,
                    "p": int(self.state.pose),
                    "r": reason,
                }
            )
        if self.writer is not None:
            self.writer.write_from_env(
                self.env,
                obs,
                action=action,
                frame_index=self.frame,
                room_id=int(self.state.room_id),
            )
        return self.state


def _save_shot(env: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(env.render()).save(path)
    return path


def _snap(st: Any, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out = final_from_state(st)
    out.update(
        {
            "items": f"0x{int(st.collected_items):04X}",
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


def _open(source: Path, *, settle: int = BOOT_SETTLE):
    return open_state_env(
        source,
        settle=settle,
        missing_hint="Need post_torizo_parlor_continuous.state",
    )


def run_once(
    source: Path,
    *,
    video_path: Path | None = None,
    out_pin: Path | None = None,
    scale: int = 2,
    crf: int = 20,
) -> dict[str, Any]:
    env, resolved = _open(source)
    assist = UnlimitedResourcesAssist()
    writer: VideoRecorder | None = None
    error: str | None = None
    t0 = time.perf_counter()
    session: _Sess | None = None
    try:
        if video_path is not None:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            obs = env.render()
            writer = VideoRecorder(
                video_path,
                width=int(obs.shape[1]),
                height=int(obs.shape[0]),
                config=VideoCaptureConfig(
                    fps=60, scale=scale, crf=crf, audio=False, footer=True
                ),
                audio_rate=None,
            )
            writer.write_from_env(
                env,
                obs,
                action=None,
                frame_index=0,
                room_id=int(parse_env_state(env, mode="nav").room_id),
            )
        session = _Sess(env, assist, writer)
        start = _snap(session.state)
        try:
            evidence = play_alcatraz_escape(session)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            evidence = None
        final = _snap(
            session.state,
            extra={
                "minY": session.min_y,
                "minYxy": session.min_y_xy,
                "frames": session.frame,
            },
        )
        png = DEFAULT_SHOT if error is None else GAME_DIR / "scratch" / "alcatraz_escape_red.png"
        _save_shot(env, png)
        if out_pin is not None and error is None:
            save_dev_state(env, out_pin)
    finally:
        encoded = 0
        if writer is not None:
            encoded = writer.frames
            writer.close()
        env.close()

    assert session is not None
    rolled = at_alcatraz_rollout(session.state)
    lip = session.min_y <= SHAFT_LIP_Y
    success = error is None and rolled
    timing = format_segment_time(session.frame)
    report = {
        "kind": HOP,
        "source": resolved,
        "success": success,
        "error": error,
        "frames": session.frame,
        "seconds": timing["seconds"],
        "clock": timing["clock"],
        "minY": session.min_y,
        "minYxy": session.min_y_xy,
        "lipClass": lip,
        "rolledOut": rolled,
        "start": start,
        "final": final,
        "evidence": None if evidence is None else evidence.to_dict(),
        "video": None if video_path is None else str(video_path),
        "encodedFrames": encoded,
        "elapsedSec": round(time.perf_counter() - t0, 2),
        "shot": str(png),
        "trace": session.trace[-80:],
    }
    write_json_report(report, None)
    return report


def dual(source: Path, *, video: bool) -> dict[str, Any]:
    rows = []
    for i in range(2):
        video_path = None
        if video:
            video_path = RECORDINGS_DIR / f"alcatraz_escape_dual{i}.mp4"
        row = run_once(source, video_path=video_path)
        rows.append(row)
        if not row["success"]:
            break
    match = len(rows) == 2 and all(r["success"] for r in rows)
    payload = {
        "kind": "alcatraz_escape_dual",
        "match": match,
        "rows": [{k: v for k, v in r.items() if k != "trace"} for r in rows],
    }
    DEFAULT_DUAL.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_DUAL.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=("run", "dual", "record"))
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    video = None
    if args.cmd == "record" or (args.cmd == "run" and not args.no_video):
        video = DEFAULT_VIDEO
    if args.cmd == "dual":
        payload = dual(args.source, video=not args.no_video)
        print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))
        for i, row in enumerate(payload["rows"]):
            slim = {k: v for k, v in row.items() if k not in ("trace", "start")}
            print(f"row{i}: {json.dumps(slim, indent=2)}")
        raise SystemExit(0 if payload["match"] else 1)

    out_pin = args.out if args.cmd != "record" else None
    report = run_once(args.source, video_path=video, out_pin=out_pin)
    DEFAULT_REPORT.write_text(json.dumps({k: v for k, v in report.items() if k != "trace"}, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("trace", "start")}, indent=2))
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()

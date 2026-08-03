#!/usr/bin/env python3
"""Bubble diagnostic: thrash path map + clean save-runway seat dump.

Not hop GREEN / continuous evidence.

1) Trace current pure controller (capped) → xy path map PNG + JSON.
2) Replay human ``bubble_jump_try`` lower → seat on save runway → dump
   scratch state (+ short video of the clean approach only).

```bash
uv run python super_metroid/scripts/probe/bubble_trace_and_seat.py
```
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retro_harness.actions import idle_action  # noqa: E402
from retro_harness.video import VideoCaptureConfig, VideoRecorder  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.dev.common import (  # noqa: E402
    boot_from_state,
    make_dev_env,
    save_dev_state,
)
from super_metroid.paths import GAME_DIR, RECORDINGS_DIR  # noqa: E402
from super_metroid.ram import parse_env_state  # noqa: E402
from super_metroid.routes.kpdr.bubble_mountain import (  # noqa: E402
    BubblePhaseStop,
    BubbleTrack,
    bubble_lower_to_mid_pin,
    bubble_on_save_runway,
    play_bubble_to_bat_cave,
)

SCRATCH = GAME_DIR / "custom_integrations" / "SuperMetroid-Snes" / "scratch"
DEFAULT_SOURCE = SCRATCH / "post_rising_tide_to_bubble_pure.state"
HUMAN_TASK = GAME_DIR / "tasks" / "bubble_jump_try.json"
DEBUG = GAME_DIR / "debug"

ROOM_BUBBLE = 0xACB3
ROOM_BUBBLE_SAVE = 0xB0DD


class TraceSession:
    """ControllerSession-compatible probe that records (frame,x,y,pose)."""

    def __init__(
        self,
        env: Any,
        assist: UnlimitedResourcesAssist,
        *,
        sample_every: int = 1,
        writer: VideoRecorder | None = None,
        max_frames: int | None = None,
    ) -> None:
        self.env = env
        self.assist = assist
        self.sample_every = max(1, sample_every)
        self.writer = writer
        self.max_frames = max_frames
        self.frame = 0
        self.state = parse_env_state(env, mode="nav")
        self.last_action = None
        self.segment_marks: list[dict[str, object]] = []
        self.trace: list[dict[str, int]] = []
        self._sample()

    def mark(self, name: str) -> None:
        st = self.state
        self.segment_marks.append(
            {
                "segment": name,
                "frame": self.frame,
                "roomIdHex": f"0x{int(st.room_id):04X}",
                "samusX": int(st.samus_x),
                "samusY": int(st.samus_y),
                "pose": int(st.pose),
            }
        )

    def _sample(self) -> None:
        if self.frame % self.sample_every != 0 and self.frame != 0:
            return
        st = self.state
        self.trace.append(
            {
                "f": self.frame,
                "x": int(st.samus_x),
                "y": int(st.samus_y),
                "p": int(st.pose),
                "r": int(st.room_id),
                "vx": int(st.velocity_x),
                "vy": int(st.velocity_y),
            }
        )

    def step(self, action, reason: str = ""):
        del reason
        if self.max_frames is not None and self.frame >= self.max_frames:
            raise TimeoutError(
                f"trace_cap frame={self.frame} xy=({self.state.samus_x},{self.state.samus_y})"
            )
        obs, *_ = self.env.step(action)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        self.assist.apply(self.env.data, self.state)
        self.last_action = action
        self._sample()
        if self.writer is not None:
            self.writer.write_from_env(
                self.env,
                obs,
                action=action,
                frame_index=self.frame,
                room_id=int(self.state.room_id),
            )
        return self.state


def _plot_trace(
    points: list[dict[str, int]],
    out_path: Path,
    *,
    title: str,
    marks: list[tuple[int, int, str]] | None = None,
) -> None:
    """Write a simple world-space path PNG (y down as in game)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"Pillow required for path map: {exc}") from exc

    if not points:
        raise ValueError("empty trace")

    xs = [p["x"] for p in points if p["r"] == ROOM_BUBBLE]
    ys = [p["y"] for p in points if p["r"] == ROOM_BUBBLE]
    if not xs:
        xs = [p["x"] for p in points]
        ys = [p["y"] for p in points]

    pad = 40
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    # Bubble is 2×4 screens; clamp to room-ish bounds for readability.
    x0, x1 = max(0, x0), min(1024, x1)
    y0, y1 = max(0, y0), min(1024, y1)
    w_world = max(1, x1 - x0)
    h_world = max(1, y1 - y0)

    # Fit into ~900px wide canvas, keep aspect.
    out_w = 900
    scale = out_w / w_world
    out_h = max(200, int(h_world * scale) + 80)
    img = Image.new("RGB", (out_w, out_h), (18, 18, 22))
    draw = ImageDraw.Draw(img)

    def to_px(x: int, y: int) -> tuple[int, int]:
        px = int((x - x0) * scale)
        py = 50 + int((y - y0) * scale)  # y increases downward (game coords)
        return px, py

    # Grid every 64 px world.
    grid = (40, 40, 48)
    gx = (x0 // 64) * 64
    while gx <= x1:
        p1 = to_px(gx, y0)
        p2 = to_px(gx, y1)
        draw.line([p1, p2], fill=grid, width=1)
        gx += 64
    gy = (y0 // 64) * 64
    while gy <= y1:
        p1 = to_px(x0, gy)
        p2 = to_px(x1, gy)
        draw.line([p1, p2], fill=grid, width=1)
        gy += 64

    # Save runway band highlight.
    run = [
        to_px(25, 380),
        to_px(90, 380),
        to_px(90, 430),
        to_px(25, 430),
    ]
    draw.polygon(run, outline=(80, 160, 255), fill=(30, 50, 80))

    n = max(1, len(points) - 1)
    for i in range(1, len(points)):
        a, b = points[i - 1], points[i]
        if a["r"] != ROOM_BUBBLE or b["r"] != ROOM_BUBBLE:
            continue
        t = i / n
        # early=green → late=red (shows thrash duration)
        color = (
            int(40 + 200 * t),
            int(200 * (1 - t)),
            int(60 + 40 * (1 - t)),
        )
        draw.line([to_px(a["x"], a["y"]), to_px(b["x"], b["y"])], fill=color, width=2)

    # Start / end dots.
    s, e = points[0], points[-1]
    draw.ellipse(
        [to_px(s["x"], s["y"])[0] - 5, to_px(s["x"], s["y"])[1] - 5,
         to_px(s["x"], s["y"])[0] + 5, to_px(s["x"], s["y"])[1] + 5],
        fill=(80, 255, 120),
    )
    draw.ellipse(
        [to_px(e["x"], e["y"])[0] - 5, to_px(e["x"], e["y"])[1] - 5,
         to_px(e["x"], e["y"])[0] + 5, to_px(e["x"], e["y"])[1] + 5],
        fill=(255, 80, 80),
    )

    if marks:
        for mx, my, label in marks:
            px, py = to_px(mx, my)
            draw.ellipse([px - 4, py - 4, px + 4, py + 4], outline=(255, 220, 80), width=2)
            draw.text((px + 6, py - 8), label, fill=(255, 220, 80))

    min_y = min(ys)
    max_x = max(xs)
    draw.text(
        (8, 8),
        f"{title}  n={len(points)}  min_y={min_y} max_x={max_x}  "
        f"start=({s['x']},{s['y']}) end=({e['x']},{e['y']})",
        fill=(220, 220, 230),
    )
    draw.text(
        (8, 28),
        "green→red = time · blue box = save runway · y down",
        fill=(140, 140, 150),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def run_pure_thrash_trace(
    *,
    source: Path,
    max_frames: int,
    sample_every: int,
    map_path: Path,
    json_path: Path,
) -> dict[str, object]:
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    t0 = time.perf_counter()
    error: str | None = None
    metrics: dict[str, object] = {}
    try:
        boot_from_state(env, source)
        session = TraceSession(
            env, assist, sample_every=sample_every, max_frames=max_frames
        )
        session.mark("_boot")
        try:
            play_bubble_to_bat_cave(session)  # type: ignore[arg-type]
        except BubblePhaseStop as stop:
            error = str(stop)
            metrics = dict(stop.metrics)
        except TimeoutError as exc:
            error = str(exc)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        final = session.state
        # Annotate first time on save runway / launch height class.
        marks: list[tuple[int, int, str]] = []
        saw_runway = False
        saw_high = False
        for p in session.trace:
            if (
                not saw_runway
                and p["r"] == ROOM_BUBBLE
                and 25 <= p["x"] <= 90
                and 380 <= p["y"] <= 430
            ):
                marks.append((p["x"], p["y"], f"runway@f{p['f']}"))
                saw_runway = True
            if not saw_high and p["r"] == ROOM_BUBBLE and p["y"] <= 240:
                marks.append((p["x"], p["y"], f"high@f{p['f']}"))
                saw_high = True
        _plot_trace(
            session.trace,
            map_path,
            title="PURE thrash (current controller)",
            marks=marks,
        )
        report = {
            "kind": "bubble_pure_thrash_trace",
            "source": str(source),
            "frames": session.frame,
            "sampleEvery": sample_every,
            "error": error,
            "metrics": metrics,
            "final": {
                "room": f"0x{int(final.room_id):04X}",
                "x": int(final.samus_x),
                "y": int(final.samus_y),
                "pose": int(final.pose),
            },
            "tracePoints": len(session.trace),
            "map": str(map_path),
            "elapsedSec": round(time.perf_counter() - t0, 2),
            "minY": min((p["y"] for p in session.trace), default=None),
            "maxX": max((p["x"] for p in session.trace), default=None),
            "trace": session.trace,
        }
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        env.close()


def run_human_seat_dump(
    *,
    source: Path,
    task_path: Path,
    state_out: Path,
    video_out: Path | None,
    max_task_frames: int,
    open_save_door: bool,
) -> dict[str, object]:
    """Replay human lower inputs until save runway seat; dump scratch state."""
    task = json.loads(task_path.read_text())
    frames: list[list[int]] = task["frames"]
    if not frames:
        raise ValueError(f"no frames in {task_path}")

    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    writer: VideoRecorder | None = None
    t0 = time.perf_counter()
    try:
        boot_from_state(env, source)
        for _ in range(5):
            env.step(idle_action())
            assist.apply(env.data, parse_env_state(env, mode="nav"))

        obs = env.render()
        if obs is None:
            obs, *_ = env.step(idle_action())
            assist.apply(env.data, parse_env_state(env, mode="nav"))
            obs = env.render()
        assert obs is not None

        if video_out is not None:
            video_out.parent.mkdir(parents=True, exist_ok=True)
            writer = VideoRecorder(
                video_out,
                width=int(obs.shape[1]),
                height=int(obs.shape[0]),
                config=VideoCaptureConfig(
                    fps=60, scale=2, crf=22, preset="veryfast", audio=False, footer=True
                ),
                audio_rate=None,
            )
            writer.write_from_env(env, obs, action=None, frame_index=0, room_id=ROOM_BUBBLE)

        session = TraceSession(env, assist, sample_every=1, writer=writer)
        seated_frame: int | None = None
        limit = min(len(frames), max_task_frames)
        for i in range(limit):
            action = list(frames[i])
            # Pad / trim to 12 SNES buttons if needed.
            if len(action) < 12:
                action = action + [0] * (12 - len(action))
            action = action[:12]
            st = session.step(action)
            if (
                seated_frame is None
                and int(st.room_id) == ROOM_BUBBLE
                and bubble_on_save_runway(st)
                and abs(int(st.velocity_x)) <= 1
                and abs(int(st.velocity_y)) <= 1
                and int(st.pose) in (1, 2, 9, 10)
                and 25 <= int(st.samus_x) <= 55
            ):
                # Prefer max-left fire window, wait a few grounded frames.
                grounded = 0
                for _ in range(20):
                    st = session.step(idle_action())
                    if bubble_on_save_runway(st) and abs(int(st.velocity_y)) <= 1:
                        grounded += 1
                    if grounded >= 8:
                        seated_frame = session.frame
                        break
                if seated_frame is not None:
                    break

        st = session.state
        # Dump runway seat FIRST (before any door detour).
        for _ in range(25):
            st = session.step(idle_action())
            if (
                bubble_on_save_runway(st)
                and abs(int(st.velocity_y)) <= 1
                and int(st.pose) in (1, 2, 9, 10)
            ):
                break
        pre_door = {
            "room": f"0x{int(st.room_id):04X}",
            "x": int(st.samus_x),
            "y": int(st.samus_y),
            "pose": int(st.pose),
            "vx": int(st.velocity_x),
            "vy": int(st.velocity_y),
            "onSaveRunway": bubble_on_save_runway(st),
            "frame": session.frame,
        }
        if pre_door["onSaveRunway"] and pre_door["room"] == "0xACB3":
            save_dev_state(env, state_out)

        # Optional: open Save blue door, peek in, walk back out to runway.
        opened = False
        save_room_seen = False
        post_door_state: Path | None = None
        if open_save_door and int(st.room_id) == ROOM_BUBBLE and bubble_on_save_runway(st):
            from retro_harness.actions import buttons

            for _ in range(8):
                st = session.step(buttons("LEFT"))
            for _ in range(20):
                st = session.step(buttons("LEFT", "B"))
            for _ in range(50):
                st = session.step(buttons("LEFT", "X"))
                if int(st.room_id) == ROOM_BUBBLE_SAVE:
                    save_room_seen = True
                    opened = True
                    break
            if save_room_seen:
                for _ in range(15):
                    st = session.step(idle_action())
                # Exit Save: try RIGHT first (door you entered), then LEFT.
                for dir_h in ("RIGHT", "LEFT"):
                    for _ in range(100):
                        if int(st.room_id) != ROOM_BUBBLE_SAVE:
                            break
                        st = session.step(buttons(dir_h, "B"))
                    if int(st.room_id) == ROOM_BUBBLE:
                        break
                for _ in range(80):
                    if int(st.room_id) != ROOM_BUBBLE:
                        break
                    # After re-entry, door deposits at left shell; walk right onto fire.
                    st = session.step(buttons("RIGHT", "B"))
                    if bubble_on_save_runway(st) and 30 <= int(st.samus_x) <= 90:
                        opened = True
                        break
                for _ in range(20):
                    st = session.step(idle_action())
            else:
                for _ in range(14):
                    st = session.step(buttons("RIGHT"))
                for _ in range(20):
                    st = session.step(idle_action())
                opened = bubble_on_save_runway(st)

            if bubble_on_save_runway(st) and int(st.room_id) == ROOM_BUBBLE:
                post_door_state = state_out.with_name(
                    state_out.stem + "_door_open" + state_out.suffix
                )
                save_dev_state(env, post_door_state)
                save_dev_state(env, state_out)

        # If human seat never landed, leave state_out to caller fallback.
        map_path = DEBUG / "bubble_clean_seat_path.png"
        _plot_trace(
            session.trace,
            map_path,
            title="CLEAN human lower → save runway",
            marks=[(int(st.samus_x), int(st.samus_y), "end")],
        )
        report = {
            "kind": "bubble_clean_save_runway_seat",
            "source": str(source),
            "humanTask": str(task_path),
            "seatedFrame": seated_frame,
            "frames": session.frame,
            "stateOut": str(state_out),
            "postDoorState": str(post_door_state) if post_door_state else None,
            "video": str(video_out) if video_out else None,
            "map": str(map_path),
            "openedSaveDoor": opened,
            "saveRoomSeen": save_room_seen,
            "preDoor": pre_door,
            "final": {
                "room": f"0x{int(st.room_id):04X}",
                "x": int(st.samus_x),
                "y": int(st.samus_y),
                "pose": int(st.pose),
                "vx": int(st.velocity_x),
                "vy": int(st.velocity_y),
                "onSaveRunway": bubble_on_save_runway(st),
            },
            "elapsedSec": round(time.perf_counter() - t0, 2),
            "note": (
                "Debug seat only. Next: rewrite climb as run+double-WJ from this pin. "
                "Not pure-controller GREEN proof."
            ),
        }
        (DEBUG / "bubble_clean_seat.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        if writer is not None:
            writer.close()
        env.close()


def run_pure_lower_only_seat(
    *,
    source: Path,
    state_out: Path,
    max_frames: int = 4000,
) -> dict[str, object]:
    """Run pure lower shelves only; dump if fire runway seats."""
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        boot_from_state(env, source)
        session = TraceSession(env, assist, sample_every=2, max_frames=max_frames)
        track = BubbleTrack(label="bubble_lower_only")
        try:
            bubble_lower_to_mid_pin(session, track)  # type: ignore[arg-type]
        except TimeoutError as exc:
            return {"ok": False, "error": str(exc), "frames": session.frame}
        st = session.state
        ok = bubble_on_save_runway(st)
        if ok:
            # Settle idle.
            for _ in range(20):
                session.step(idle_action())
            st = session.state
            save_dev_state(env, state_out)
        map_path = DEBUG / "bubble_pure_lower_path.png"
        _plot_trace(session.trace, map_path, title="PURE lower only")
        return {
            "ok": ok,
            "frames": session.frame,
            "mid_reached": track.mid_reached,
            "stateOut": str(state_out) if ok else None,
            "map": str(map_path),
            "final": {
                "room": f"0x{int(st.room_id):04X}",
                "x": int(st.samus_x),
                "y": int(st.samus_y),
                "pose": int(st.pose),
                "onSaveRunway": bubble_on_save_runway(st),
            },
            "minY": min((p["y"] for p in session.trace), default=None),
            "maxX": max((p["x"] for p in session.trace), default=None),
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--skip-thrash",
        action="store_true",
        help="Skip pure thrash trace (seat dump only)",
    )
    parser.add_argument(
        "--thrash-frames",
        type=int,
        default=6000,
        help="Cap pure thrash frames (default 6000 ≈1.5min game; full timeout is 28k)",
    )
    parser.add_argument("--sample-every", type=int, default=3)
    parser.add_argument(
        "--no-open-door",
        action="store_true",
        help="Do not attempt to open/enter save door",
    )
    parser.add_argument(
        "--state-out",
        type=Path,
        default=SCRATCH / "post_bubble_save_runway_pure.state",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=RECORDINGS_DIR / "bubble_clean_save_runway.mp4",
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--human-frames",
        type=int,
        default=700,
        help="Max human task frames to replay for lower seat",
    )
    args = parser.parse_args()

    DEBUG.mkdir(parents=True, exist_ok=True)

    if not args.skip_thrash:
        print("[1/2] pure thrash path map…", flush=True)
        thrash = run_pure_thrash_trace(
            source=args.source,
            max_frames=args.thrash_frames,
            sample_every=args.sample_every,
            map_path=DEBUG / "bubble_pure_thrash_path.png",
            json_path=DEBUG / "bubble_pure_thrash_trace.json",
        )
        print(
            f"  frames={thrash['frames']} min_y={thrash['minY']} max_x={thrash['maxX']}",
            flush=True,
        )
        print(f"  map={thrash['map']}", flush=True)
        print(f"  json={DEBUG / 'bubble_pure_thrash_trace.json'}", flush=True)
        if thrash.get("error"):
            print(f"  stop={thrash['error'][:160]}", flush=True)

    print("[2/2] clean human lower → save runway seat + state dump…", flush=True)
    if not HUMAN_TASK.is_file():
        raise SystemExit(f"missing human task: {HUMAN_TASK}")
    seat = run_human_seat_dump(
        source=args.source,
        task_path=HUMAN_TASK,
        state_out=args.state_out,
        video_out=None if args.no_video else args.video,
        max_task_frames=args.human_frames,
        open_save_door=not args.no_open_door,
    )
    print(json.dumps(seat, indent=2), flush=True)
    fin = seat["final"]  # type: ignore[index]
    pre = seat.get("preDoor") or {}
    seated_ok = bool(
        (fin.get("onSaveRunway") and fin.get("room") == "0xACB3")
        or (pre.get("onSaveRunway") and pre.get("room") == "0xACB3")
    )
    if not seated_ok:
        print("[warn] seat dump not cleanly on Bubble save runway", flush=True)
        print("[fallback] pure lower only…", flush=True)
        fb = run_pure_lower_only_seat(source=args.source, state_out=args.state_out)
        print(json.dumps(fb, indent=2), flush=True)
        sys.exit(0 if fb.get("ok") else 1)
    if pre.get("onSaveRunway") and not fin.get("onSaveRunway"):
        print(
            "[ok] pre-door runway seat dumped "
            f"(door detour ended room={fin.get('room')}; working pin is pre-door)",
            flush=True,
        )
    print(f"[ok] state → {args.state_out}", flush=True)
    if seat.get("postDoorState"):
        print(f"[ok] door-open pin → {seat['postDoorState']}", flush=True)
    if seat.get("video"):
        print(f"[ok] video → {seat['video']}", flush=True)
    print(f"[ok] map → {seat['map']}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()

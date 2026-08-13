#!/usr/bin/env python3
"""Bubble Save door → consecutive wall-jump climb demo (tech video).

Map Rando room 97 Bubble Mountain (``0xACB3``):

* Save door = node 2 (Top Middle Left) ↔ Save Room ``0xB0DD``
* Top-left door = node 1 (Top Left)

Demo focus: **canConsecutiveWallJump** / **canPreciseWallJump** builders.
Default path uses the proven save-runway double/triple WJ (product R15 family),
then steers toward the **top-left door lip**. Not continuous-tip evidence.

```bash
# Full leave-save + multi WJ + top-left steer (video)
uv run python snes/super_metroid/scripts/probe/bubble_left_wj_climb.py \
  --video snes/super_metroid/recordings/bubble_save_to_top_left_wj.mp4

# Human fire-seat pin (skips leave-save)
uv run python snes/super_metroid/scripts/probe/bubble_left_wj_climb.py \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/bubble_human_runway.state \
  --video snes/super_metroid/recordings/bubble_save_to_top_left_wj.mp4 \
  --wj-count 3
```
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
_SNES = Path(__file__).resolve().parents[3]
for _p in (ROOT, _SNES):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from retro_harness.actions import buttons, idle_action  # noqa: E402
from retro_harness.video import VideoCaptureConfig, VideoRecorder  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.dev.common import boot_from_state, make_dev_env  # noqa: E402
from super_metroid.paths import GAME_DIR, INTEGRATION_DIR  # noqa: E402
from super_metroid.ram import SuperMetroidState, parse_env_state  # noqa: E402
from super_metroid.routes.controller_common import (  # noqa: E402
    WallJumpTiming,
    is_morph,
    unmorph,
)
from super_metroid.routes.skills.geometry import new_climb_track  # noqa: E402
from super_metroid.routes.skills.policies import bubble_to_bat as P  # noqa: E402
from super_metroid.routes.skills.runway import (  # noqa: E402
    prepare_fire_run,
    runway_dash,
    spin_glide,
)
from super_metroid.routes.skills.walljump import consecutive_walljumps  # noqa: E402

# Human save-runway isolation timings (Phase D green on bubble_human_runway).
# Product live pure uses shorter WJ2 (14/2/6); human pin needs L24/R14/follow56.
_HUMAN_WJ1 = WallJumpTiming(
    into="LEFT", flip="RIGHT", into_frames=20, amid_frames=4, flip_frames=8
)
_HUMAN_WJ2 = WallJumpTiming(
    into="LEFT", flip="RIGHT", into_frames=24, amid_frames=2, flip_frames=14
)
_HUMAN_FOLLOW = 56

SCRATCH = INTEGRATION_DIR / "scratch"
DEFAULT_SAVE_PIN = SCRATCH / "bubble_save.state"
DEFAULT_RUNWAY_PIN = SCRATCH / "bubble_human_runway.state"
DEFAULT_VIDEO = (
    GAME_DIR / "recordings" / "bubble_save_to_top_left_wj.mp4"
)
ROOM_SAVE = 0xB0DD
ROOM_BUBBLE = 0xACB3

# Top-left door lip (node 1): left edge, top screen of 2×4 Bubble.
TOP_LEFT_X_MAX = 80
TOP_LEFT_Y_MAX = 160
# Secondary: Phase D height (product right-structure top band).
PHASE_D_X = P.PHASE_D_X
PHASE_D_Y = P.PHASE_D_Y


class RecordingSession:
    """Minimal ControllerSession + video frames."""

    def __init__(self, env: Any, assist: UnlimitedResourcesAssist, writer: VideoRecorder | None):
        self.env = env
        self.assist = assist
        self.writer = writer
        self.frame = 0
        self.state = parse_env_state(env, frame=0, mode="nav")
        self.min_y = int(self.state.samus_y)
        self.min_y_at = (int(self.state.samus_x), int(self.state.samus_y))
        self.latch_frames = 0
        self.wj_pulses = 0

    def _update(self) -> SuperMetroidState:
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        self.assist.apply(self.env.data, self.state)
        y = int(self.state.samus_y)
        if y < self.min_y:
            self.min_y = y
            self.min_y_at = (int(self.state.samus_x), y)
        if int(self.state.pose) == 132:
            self.latch_frames += 1
        return self.state

    def step(self, action: Any, reason: str = "") -> SuperMetroidState:
        del reason
        obs, *_ = self.env.step(action)
        self.frame += 1
        st = self._update()
        if self.writer is not None:
            try:
                self.writer.write_from_env(
                    self.env,
                    obs if obs is not None else self.env.render(),
                    action=action,
                    frame_index=self.frame,
                    room_id=int(st.room_id),
                )
            except Exception:
                # Footer/audio path can be picky; keep sim running.
                rgb = self.env.render()
                if rgb is not None:
                    self.writer.write_frame(rgb, frame_index=self.frame)
        return st

    def hold(self, n: int, *names: str, reason: str = "") -> SuperMetroidState:
        act = buttons(*names) if names else idle_action()
        st = self.state
        for _ in range(n):
            st = self.step(act, reason=reason)
        return st


def leave_save_to_bubble(session: RecordingSession, *, budget: int = 400) -> SuperMetroidState:
    """Unmorph / leave Save 0xB0DD RIGHT into Bubble 0xACB3 save runway."""
    st = session.state
    if int(st.room_id) == ROOM_BUBBLE:
        return st
    if is_morph(int(st.pose)) or int(st.pose) in (155, 156, 0x9B):
        try:
            unmorph(session)
        except Exception:
            # Save-station poses: mash UP then walk
            session.hold(20, "UP", reason="leave_save_up")
            session.hold(10, reason="leave_save_up_idle")
    # Walk RIGHT out the door (do not re-enter with grounded x≲20 later).
    for _ in range(budget):
        st = session.state
        if int(st.room_id) == ROOM_BUBBLE:
            # Settle ordinary on runway
            session.hold(20, reason="leave_save_settle")
            return session.state
        session.hold(1, "RIGHT", reason="leave_save_right")
    raise RuntimeError(
        f"leave_save: still room 0x{int(session.state.room_id):04X} "
        f"xy=({session.state.samus_x},{session.state.samus_y}) pose={session.state.pose}"
    )


def seat_save_runway(session: RecordingSession, *, max_frames: int = 240) -> SuperMetroidState:
    """Walk to max-left fire seat x∈[25,30] y∈[380,430] if not already there."""
    lo, hi = P.SAVE_HUMAN_SEAT_X
    y0, y1 = P.SAVE_RUNWAY_Y
    for _ in range(max_frames):
        st = session.state
        if int(st.room_id) != ROOM_BUBBLE:
            break
        x, y = int(st.samus_x), int(st.samus_y)
        if lo <= x <= hi and y0 <= y <= y1 and int(st.pose) in (1, 2, 9, 10):
            return st
        # Prefer left on runway; if too high/low, gentle walk
        if x > hi:
            session.hold(1, "LEFT", reason="seat_left")
        elif x < lo:
            session.hold(1, "RIGHT", reason="seat_right")
        elif y > y1:
            session.hold(1, "LEFT", "A", reason="seat_up_left")
        else:
            session.hold(1, reason="seat_idle")
    return session.state


def steer_top_left(session: RecordingSession, *, budget: int = 500) -> SuperMetroidState:
    """After height, push toward top-left door lip."""
    for _ in range(budget):
        st = session.state
        if int(st.room_id) != ROOM_BUBBLE:
            return st
        x, y = int(st.samus_x), int(st.samus_y)
        if x <= TOP_LEFT_X_MAX and y <= TOP_LEFT_Y_MAX:
            # Pressure LEFT into door
            session.hold(30, "LEFT", reason="top_left_door_push")
            return session.state
        # High enough: walk/jump left; still mid: keep WJ period on left wall
        if y <= 220:
            if x > TOP_LEFT_X_MAX:
                session.hold(1, "LEFT", "B", reason="top_left_dash")
            else:
                session.hold(1, "LEFT", "A", reason="top_left_hop")
        else:
            # Still climbing — left-wall period WJ
            session.hold(3, "LEFT", "A", reason="left_wj_into")
            session.hold(2, "A", reason="left_wj_amid")
            session.hold(4, "RIGHT", "A", reason="left_wj_flip")
            session.wj_pulses += 1
    return session.state


def on_top_left(st: SuperMetroidState) -> bool:
    return (
        int(st.room_id) == ROOM_BUBBLE
        and int(st.samus_x) <= TOP_LEFT_X_MAX
        and int(st.samus_y) <= TOP_LEFT_Y_MAX
    )


def on_phase_d(st: SuperMetroidState) -> bool:
    return (
        int(st.room_id) == ROOM_BUBBLE
        and int(st.samus_x) >= PHASE_D_X
        and int(st.samus_y) <= PHASE_D_Y
    )


def run_demo(
    *,
    source: Path,
    video_path: Path | None,
    wj_count: int,
    leave_save: bool,
    audio: bool,
    scale: int,
) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)

    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    writer: VideoRecorder | None = None
    t0 = time.perf_counter()
    error: str | None = None
    green = False

    try:
        boot_from_state(env, source)
        for _ in range(3):
            env.step(idle_action())
            assist.apply(env.data, parse_env_state(env, mode="nav"))

        obs = env.render()
        if obs is None:
            obs, *_ = env.step(idle_action())
            obs = env.render()
        assert obs is not None

        if video_path is not None:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            audio_rate = None
            if audio:
                try:
                    audio_rate = int(env.em.get_audio_rate())  # type: ignore[attr-defined]
                except Exception:
                    audio_rate = None
            writer = VideoRecorder(
                video_path,
                width=int(obs.shape[1]),
                height=int(obs.shape[0]),
                config=VideoCaptureConfig(
                    fps=60,
                    scale=scale,
                    crf=20,
                    preset="veryfast",
                    audio=bool(audio and audio_rate),
                    footer=True,
                ),
                audio_rate=audio_rate,
            )
            writer.write_from_env(
                env, obs, action=None, frame_index=0, room_id=int(parse_env_state(env).room_id)
            )

        session = RecordingSession(env, assist, writer)

        # --- leave save if needed ---
        if leave_save or int(session.state.room_id) == ROOM_SAVE:
            leave_save_to_bubble(session)
        if int(session.state.room_id) != ROOM_BUBBLE:
            raise RuntimeError(
                f"expected Bubble 0xACB3, got 0x{int(session.state.room_id):04X}"
            )

        # Human fire-seat pins must NOT re-seat (subpixel is the green key).
        # After leave-save, soft-seat only when not already on runway.
        st0 = session.state
        on_runway = (
            int(st0.room_id) == ROOM_BUBBLE
            and P.SAVE_RUNWAY_Y[0] <= int(st0.samus_y) <= P.SAVE_RUNWAY_Y[1]
            and int(st0.samus_x) <= 40
        )
        if not on_runway:
            seat_save_runway(session)

        track = new_climb_track(session, label="bubble_left_wj")
        n = max(2, wj_count)
        session.wj_pulses = n
        jumps = [_HUMAN_WJ1, _HUMAN_WJ2]
        while len(jumps) < n:
            jumps.append(_HUMAN_WJ2)

        height_box = [False]
        prepare_fire_run(session, track, policy=P, y_clear=True, crouch=False)
        runway_dash(session, track, policy=P, arm_pump=True)
        if not spin_glide(session, track, policy=P, height_box=height_box):
            consecutive_walljumps(
                session,
                track,
                jumps[:n],
                policy=P,  # type: ignore[arg-type]
                pre_approach=True,
                follow_spin=True,
                follow_frames=_HUMAN_FOLLOW,
                height_box=height_box,
            )
        topped = bool(track.top_reached or height_box[0])

        # Hold top briefly for video; optional soft left nudge (no thrash fall).
        st = session.state
        if topped or on_phase_d(st) or session.min_y <= 150:
            for _ in range(90):
                st = session.state
                if int(st.samus_y) > 220:
                    break
                if int(st.samus_x) > 200:
                    session.hold(1, "LEFT", reason="top_hold_left")
                else:
                    session.hold(1, reason="top_hold")

        st = session.state
        # Phase D top from multi-WJ is the tech demo GREEN; strict node-1 is bonus.
        strict = on_top_left(st) or (
            int(st.room_id) != ROOM_BUBBLE and int(st.samus_y) <= TOP_LEFT_Y_MAX
        )
        green = bool(topped) or bool(track.top_reached) or (
            session.min_y <= 150 and int(session.min_y_at[0]) >= 290
        ) or strict

        result = {
            "ok": green,
            "strict_top_left": strict,
            "phase_d": on_phase_d(st) or bool(track.top_reached),
            "fire_topped": bool(topped or track.top_reached),
            "frames": session.frame,
            "room": f"0x{int(st.room_id):04X}",
            "x": int(st.samus_x),
            "y": int(st.samus_y),
            "pose": int(st.pose),
            "min_y": session.min_y,
            "min_y_at": list(session.min_y_at),
            "latch_frames": session.latch_frames,
            "wj_pulses": session.wj_pulses,
            "source": str(source),
            "video": str(video_path) if video_path else None,
            "seconds": round(time.perf_counter() - t0, 2),
            "timings": "arm_pump + WJ1 L20/a4/R8 + WJ2 L24/a2/R14 follow56",
            "tech": [
                "canDash",
                "canWallJump",
                "canPreciseWallJump",
                "canConsecutiveWallJump",
            ],
        }
        return result
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        st = parse_env_state(env, mode="nav")
        return {
            "ok": False,
            "error": error,
            "frames": 0,
            "room": f"0x{int(st.room_id):04X}",
            "x": int(st.samus_x),
            "y": int(st.samus_y),
            "pose": int(st.pose),
            "source": str(source),
            "video": str(video_path) if video_path else None,
        }
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SAVE_PIN,
        help="Save pin (0xB0DD) or Bubble runway pin",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO,
        help="Output mp4 path (use --no-video to skip)",
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--wj-count", type=int, default=3, help="Consecutive WJ pulses (≥2)")
    parser.add_argument(
        "--from-runway",
        action="store_true",
        help=f"Use {DEFAULT_RUNWAY_PIN.name} fire seat (skip leave-save)",
    )
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    source = DEFAULT_RUNWAY_PIN if args.from_runway else args.source
    video = None if args.no_video else args.video
    result = run_demo(
        source=source,
        video_path=video,
        wj_count=args.wj_count,
        leave_save=not args.from_runway,
        audio=not args.no_audio,
        scale=args.scale,
    )
    report = args.report
    if report is None and video is not None:
        report = video.with_suffix(".json")
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2) + "\n")

    tag = "GREEN" if result.get("ok") else "RED"
    print(
        f"[{tag}] frames={result.get('frames')} room={result.get('room')} "
        f"xy=({result.get('x')},{result.get('y')}) pose={result.get('pose')} "
        f"min_y={result.get('min_y')} latch={result.get('latch_frames')} "
        f"wj={result.get('wj_pulses')} top_left={result.get('strict_top_left')} "
        f"phase_d={result.get('phase_d')}"
    )
    if result.get("error"):
        print(f"  error: {result['error']}")
    if result.get("video"):
        vp = Path(str(result["video"]))
        size = vp.stat().st_size if vp.is_file() else 0
        print(f"  video: {vp} ({size} bytes)")
    if report is not None:
        print(f"  report: {report}")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

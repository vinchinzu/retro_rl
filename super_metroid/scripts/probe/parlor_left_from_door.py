#!/usr/bin/env python3
"""Demo: post-Torizo Flyway door → Parlor → **left** Alcatraz wall-jump climb.

Not continuous evidence. Purpose: show the LEFT shaft path (true Alcatraz
wall-jump chimney), not the mid-room platform hop near the Flyway door that
the product Terminator route uses.

```bash
# Default: post-Torizo Parlor at Flyway door (x~968 y651) → LEFT only + MP4
uv run python super_metroid/scripts/probe/parlor_left_from_door.py

# Optional: try editor Flyway state + left door cross (often door-blocked)
uv run python super_metroid/scripts/probe/parlor_left_from_door.py --from-flyway

# Longer climb budget / more WJ pulses
uv run python super_metroid/scripts/probe/parlor_left_from_door.py --wj-count 16 --budget 2400

# Custom source / paths
uv run python super_metroid/scripts/probe/parlor_left_from_door.py \\
  --state super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_torizo_parlor_continuous.state \\
  --video super_metroid/recordings/parlor_left_from_door.mp4
```

Outputs: ``recordings/parlor_left_from_door.mp4`` + ``debug/spore/parlor_left/*.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from retro_harness.actions import buttons, idle_action  # noqa: E402
from retro_harness.video import VideoCaptureConfig, VideoRecorder  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.dev.common import boot_from_state, make_dev_env  # noqa: E402
from super_metroid.paths import GAME_DIR, RECORDINGS_DIR  # noqa: E402
from super_metroid.ram import parse_env_state  # noqa: E402
from super_metroid.routes.controller_common import (  # noqa: E402
    WallJumpTiming,
    consecutive_walljumps,
    is_wall_latch,
    settle_hold,
    walljump_once,
)

ROOM_PARLOR = 0x92FD
ROOM_FLYWAY = 0x9879

INTEGRATION = GAME_DIR / "custom_integrations" / "SuperMetroid-Snes"
SCRATCH = INTEGRATION / "scratch"
DEFAULT_FLYWAY = INTEGRATION / "Flyway [from Bomb Torizo Room].state"
DEFAULT_PARLOR = SCRATCH / "post_torizo_parlor_continuous.state"
DEBUG_DIR = GAME_DIR / "debug" / "spore" / "parlor_left"

# Pre-HiJump alternating shaft WJ (left wall first, then right, …).
# into+A holds then flip+A — same pulse shape as Bubble, slower for early game.
_WJ_LEFT_WALL = WallJumpTiming(
    into="LEFT",
    flip="RIGHT",
    into_frames=14,
    amid_frames=3,
    flip_frames=10,
    delay_into_frames=0,
)
_WJ_RIGHT_WALL = WallJumpTiming(
    into="RIGHT",
    flip="LEFT",
    into_frames=14,
    amid_frames=3,
    flip_frames=10,
    delay_into_frames=0,
)


class RecSession:
    """Minimal ControllerSession + video + trace."""

    def __init__(self, env, assist: UnlimitedResourcesAssist, writer: VideoRecorder | None):
        self.env = env
        self.assist = assist
        self.writer = writer
        self.frame = 0
        self.state = parse_env_state(env, mode="nav")
        self.last_action = None
        self.trace: list[dict[str, object]] = []
        self._log_every = 15

    def step(self, action, reason: str = ""):
        obs, *_ = self.env.step(action)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        self.assist.apply(self.env.data, self.state)
        self.last_action = action
        if self.writer is not None:
            self.writer.write_from_env(
                self.env,
                obs,
                action=action,
                frame_index=self.frame,
                room_id=int(self.state.room_id),
            )
        if self.frame % self._log_every == 0 or reason.endswith("_mark"):
            self._record(reason)
        return self.state

    def hold(self, frames: int, *names: str, reason: str = "hold"):
        action = buttons(*names) if names else idle_action()
        st = self.state
        for _ in range(frames):
            st = self.step(action, reason)
        return st

    def _record(self, reason: str) -> None:
        st = self.state
        self.trace.append(
            {
                "f": self.frame,
                "room": f"0x{int(st.room_id):04X}",
                "x": int(st.samus_x),
                "y": int(st.samus_y),
                "pose": int(st.pose),
                "vx": int(st.velocity_x),
                "vy": int(st.velocity_y),
                "latch": is_wall_latch(st),
                "reason": reason,
            }
        )

    def snapshot(self, label: str) -> Path:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        obs = self.env.render()
        if obs is None:
            obs, *_ = self.env.step(idle_action())
            self.assist.apply(self.env.data, parse_env_state(self.env, mode="nav"))
            obs = self.env.render()
        path = DEBUG_DIR / f"{label}.png"
        if obs is not None:
            cv2.imwrite(str(path), cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        print(
            f"[snap] {label} f={self.frame} "
            f"0x{self.state.room_id:04X} ({self.state.samus_x},{self.state.samus_y}) "
            f"p={self.state.pose} → {path}",
            flush=True,
        )
        return path

    def log(self, msg: str) -> None:
        st = self.state
        print(
            f"[left] {msg} f={self.frame} room=0x{st.room_id:04X} "
            f"xy=({st.samus_x},{st.samus_y}) pose={st.pose} "
            f"latch={is_wall_latch(st)}",
            flush=True,
        )


def cross_flyway_to_parlor(session: RecSession, *, timeout: int = 900) -> None:
    """From Flyway, run LEFT through the blue door into Parlor."""
    if session.state.room_id == ROOM_PARLOR:
        session.log("already in parlor")
        return
    if session.state.room_id != ROOM_FLYWAY:
        raise RuntimeError(
            f"expected Flyway 0x{ROOM_FLYWAY:04X} or Parlor 0x{ROOM_PARLOR:04X}, "
            f"got 0x{session.state.room_id:04X}"
        )
    session.log("flyway → parlor (hold LEFT)")
    session.snapshot("00_flyway_start")
    for i in range(timeout):
        if session.state.room_id == ROOM_PARLOR:
            session.log(f"entered parlor after {i}f")
            session.snapshot("01_parlor_door_entry")
            # settle past door transition
            session.hold(40, reason="parlor_door_settle")
            session.snapshot("02_parlor_door_settled")
            return
        # door is left of Flyway spawn; run + occasional jump for the short tunnel
        if i % 40 < 28:
            session.hold(1, "LEFT", "B", reason="flyway_left_run")
        else:
            session.hold(1, "LEFT", "A", "B", reason="flyway_left_jump")
    raise TimeoutError(f"never left Flyway: {session.state}")


def approach_left_shaft(session: RecSession, *, target_x: int = 420) -> None:
    """From Flyway-side parlor (~x960), run/jump LEFT toward Alcatraz shaft."""
    session.log(f"approach left shaft (target x≤{target_x})")
    session.snapshot("03_approach_start")
    # Clear any post-door right bias; face left.
    session.hold(8, "LEFT", reason="face_left")
    for i in range(500):
        st = session.state
        if st.room_id != ROOM_PARLOR:
            session.log(f"left room during approach → 0x{st.room_id:04X}")
            return
        if st.samus_x <= target_x:
            session.log(f"reached left band x={st.samus_x}")
            session.snapshot("04_left_shaft_band")
            session.hold(20, reason="left_band_settle")
            return
        # platforms: dash left + hop
        if i % 36 < 22:
            session.hold(1, "LEFT", "B", "X", reason="left_run")
        elif i % 36 < 30:
            session.hold(1, "LEFT", "A", "B", reason="left_hop")
        else:
            session.hold(1, "LEFT", reason="left_brake")
    session.snapshot("04_left_approach_timeout")
    session.log(f"approach timeout at x={session.state.samus_x}")


def left_walljump_climb(
    session: RecSession,
    *,
    wj_count: int = 12,
    budget: int = 1800,
    y_goal: int = 200,
) -> None:
    """Open-loop alternating WJ up the left shaft; stop on height / leave room."""
    session.log(f"LEFT shaft WJ climb count={wj_count} y_goal≤{y_goal}")
    session.snapshot("05_wj_start")

    # Small hop into left wall to arm the first latch/contact.
    session.hold(6, "LEFT", reason="pre_wj_face")
    session.hold(18, "LEFT", "A", reason="pre_wj_jump")
    session.hold(4, "LEFT", reason="pre_wj_into_wall")

    jumps: list[WallJumpTiming] = []
    for i in range(wj_count):
        jumps.append(_WJ_LEFT_WALL if i % 2 == 0 else _WJ_RIGHT_WALL)

    min_y = int(session.state.samus_y)
    start_f = session.frame

    def _stop(st) -> bool:
        nonlocal min_y
        if st.room_id != ROOM_PARLOR:
            return True
        min_y = min(min_y, int(st.samus_y))
        if int(st.samus_y) <= y_goal:
            return True
        if session.frame - start_f >= budget:
            return True
        return False

    consecutive_walljumps(
        session,
        jumps,
        reason="alcatraz_left_wj",
        gap_frames=2,
        stop_when=_stop,
    )

    # Extra period farm if still low and in room.
    while (
        session.state.room_id == ROOM_PARLOR
        and session.state.samus_y > y_goal
        and session.frame - start_f < budget
    ):
        # period-ish: into left, flip right
        walljump_once(
            session,
            _WJ_LEFT_WALL,
            reason="alcatraz_left_extra",
            stop_when=_stop,
        )
        if _stop(session.state):
            break
        walljump_once(
            session,
            _WJ_RIGHT_WALL,
            reason="alcatraz_right_extra",
            stop_when=_stop,
        )
        if _stop(session.state):
            break
        settle_hold(session, 4, reason="alcatraz_extra_gap")

    session.log(f"climb end min_y={min_y}")
    session.snapshot("06_wj_end")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--from-flyway",
        action="store_true",
        help="Boot editor Flyway state and try LEFT door into Parlor",
    )
    parser.add_argument(
        "--from-parlor",
        action="store_true",
        help="(default) Boot post-Torizo Parlor settle at Flyway door",
    )
    parser.add_argument("--state", type=Path, default=None, help="Override source .state")
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--wj-count", type=int, default=12)
    parser.add_argument("--budget", type=int, default=1800, help="Max climb frames")
    parser.add_argument("--y-goal", type=int, default=200)
    parser.add_argument("--target-x", type=int, default=420, help="Left approach band")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()

    if args.state is not None:
        source = args.state
    elif args.from_flyway:
        source = DEFAULT_FLYWAY
    else:
        # Default = continuous post-BT Parlor pin at the Flyway door (x~968,y651).
        source = DEFAULT_PARLOR
    if not source.is_file():
        raise SystemExit(
            f"missing source state: {source}\n"
            "Dump one with continuous bombs → parlor, or pass --state PATH."
        )

    video_path = args.video or (RECORDINGS_DIR / "parlor_left_from_door.mp4")
    report_path = args.report or (RECORDINGS_DIR / "parlor_left_from_door.json")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[left] source={source}", flush=True)
    print(f"[left] video={video_path}", flush=True)

    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    writer: VideoRecorder | None = None
    session: RecSession | None = None
    t0 = time.perf_counter()
    error: str | None = None

    try:
        boot_from_state(env, source, settle_frames=6)
        for _ in range(4):
            env.step(idle_action())
            assist.apply(env.data, parse_env_state(env, mode="nav"))

        obs = env.render()
        if obs is None:
            obs, *_ = env.step(idle_action())
            assist.apply(env.data, parse_env_state(env, mode="nav"))
            obs = env.render()
        assert obs is not None

        config = VideoCaptureConfig(
            fps=60,
            scale=args.scale,
            crf=20,
            preset="veryfast",
            audio=not args.no_audio,
            footer=True,
        )
        audio_rate = None
        if not args.no_audio:
            audio_rate = int(env.em.get_audio_rate())  # type: ignore[attr-defined]
        writer = VideoRecorder(
            video_path,
            width=int(obs.shape[1]),
            height=int(obs.shape[0]),
            config=config,
            audio_rate=audio_rate,
        )
        session = RecSession(env, assist, writer)
        writer.write_from_env(
            env,
            obs,
            action=None,
            frame_index=0,
            room_id=int(session.state.room_id),
        )
        session._record("boot")
        session.log("boot")

        if args.from_flyway and session.state.room_id == ROOM_FLYWAY:
            cross_flyway_to_parlor(session)
        else:
            session.snapshot("01_parlor_door")

        approach_left_shaft(session, target_x=args.target_x)
        left_walljump_climb(
            session,
            wj_count=args.wj_count,
            budget=args.budget,
            y_goal=args.y_goal,
        )
        # brief idle so the end pin is visible
        session.hold(60, reason="end_hold")
        session.snapshot("07_final")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"[left] FAIL {error}", flush=True)
        if session is not None:
            try:
                session.snapshot("99_fail")
            except Exception:
                pass
    finally:
        encoded = writer.frames if writer is not None else 0
        if writer is not None:
            writer.close()
        env.close()

    elapsed = time.perf_counter() - t0
    st = session.state if session is not None else None
    report = {
        "kind": "parlor_left_from_door_demo",
        "source": str(source.resolve()),
        "video": str(video_path.resolve()),
        "success": error is None,
        "error": error,
        "frames": session.frame if session is not None else 0,
        "encodedFrames": encoded,
        "elapsedSec": round(elapsed, 2),
        "final": (
            {
                "roomIdHex": f"0x{int(st.room_id):04X}",
                "samusX": int(st.samus_x),
                "samusY": int(st.samus_y),
                "pose": int(st.pose),
                "minY": min((t["y"] for t in session.trace), default=None),
            }
            if st is not None and session is not None
            else None
        ),
        "trace": session.trace if session is not None else [],
        "note": (
            "Demo only: Flyway door → left Alcatraz WJ shaft. "
            "Not the product Terminator platform path."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "trace"}, indent=2))
    print(f"[left] wrote {video_path}", flush=True)
    print(f"[left] wrote {report_path}", flush=True)
    print(f"[left] snaps under {DEBUG_DIR}", flush=True)
    sys.exit(0 if error is None else 1)


if __name__ == "__main__":
    main()

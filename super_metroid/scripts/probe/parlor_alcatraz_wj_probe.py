#!/usr/bin/env python3
"""Probe Alcatraz shaft WJ from post-BT Parlor door (human-demo timings).

Uses shared ``walljump_once`` / ``WallJumpTiming``. Open-loop or human2
bit-exact prefix from ``scratch/post_torizo_parlor_continuous.state``.
Not continuous evidence.

Human sources: ``tasks/parlor_left_human.json`` (3244f),
``tasks/parlor_left_human2.json`` (630f). See
``docs/tasks/PARLOR_ALCATRAZ_HUMAN.md``.

```bash
# Human2 replay → left wall, then open-loop spin + mid-rise WJ chain
uv run python super_metroid/scripts/probe/parlor_alcatraz_wj_probe.py --mode h2-midrise

# Bit-exact human2 full climb (control)
uv run python super_metroid/scripts/probe/parlor_alcatraz_wj_probe.py --mode h2-full

# Human1 true-WJ recipe: mid-ledge → right wall → LEFT+A rise
uv run python super_metroid/scripts/probe/parlor_alcatraz_wj_probe.py --mode h1-true

# Short alternating chimney pulses (classic chain attempt)
uv run python super_metroid/scripts/probe/parlor_alcatraz_wj_probe.py --mode short-chain --wj-count 4

# Open-loop approach + spin (legacy)
uv run python super_metroid/scripts/probe/parlor_alcatraz_wj_probe.py --mode openloop --wj-count 2

# Sweep mid-rise cut frames (writes one report, best video)
uv run python super_metroid/scripts/probe/parlor_alcatraz_wj_probe.py --mode sweep
```
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

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
SCRATCH = GAME_DIR / "custom_integrations" / "SuperMetroid-Snes" / "scratch"
DEFAULT_STATE = SCRATCH / "post_torizo_parlor_continuous.state"
DEBUG_DIR = GAME_DIR / "debug" / "spore" / "parlor_alcatraz_probe"
TASKS = GAME_DIR / "tasks"
HUMAN2_TASK = TASKS / "parlor_left_human2.json"
HUMAN1_TASK = TASKS / "parlor_left_human.json"

# Human2: at f450 left wall ~(805,355) facing RIGHT; spin starts f452.
HUMAN2_LEFT_WALL_CUT = 450  # exclusive end of replay prefix
HUMAN2_SPIN_END = 492  # after peak pose-131 coast

# Human1 true WJ (best cluster f2441–2478): mid-rise p132 LEFT+A y356→258.
# Entry: mid-ledge → RIGHT+B+A into right wall → face LEFT → LEFT+A.
WJ1_RIGHT_WALL = WallJumpTiming(
    into="LEFT",
    flip="RIGHT",
    into_frames=30,
    amid_frames=3,
    flip_frames=12,
    delay_into_frames=0,
)
WJ2_LEFT_WALL = WallJumpTiming(
    into="RIGHT",
    flip="LEFT",
    into_frames=28,
    amid_frames=3,
    flip_frames=12,
    delay_into_frames=2,
)

# Classic short alternating chimney (try real bounce chain).
_SHORT_L = WallJumpTiming(
    into="LEFT", flip="RIGHT", into_frames=8, amid_frames=2, flip_frames=10
)
_SHORT_R = WallJumpTiming(
    into="RIGHT", flip="LEFT", into_frames=8, amid_frames=2, flip_frames=10
)


class Sess:
    def __init__(self, env, assist, writer):
        self.env = env
        self.assist = assist
        self.writer = writer
        self.frame = 0
        self.state = parse_env_state(env, mode="nav")
        self.trace: list[dict] = []
        self.min_y = int(self.state.samus_y)
        self.latch_frames = 0
        self.p132_rise_frames = 0  # pose 132 with y decreasing
        self._prev_y = int(self.state.samus_y)
        self._prev_pose = int(self.state.pose)
        self.true_wj_events: list[dict] = []  # rising p132 clusters

    def step(self, action, reason: str = ""):
        obs, *_ = self.env.step(action)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        self.assist.apply(self.env.data, self.state)
        y = int(self.state.samus_y)
        pose = int(self.state.pose)
        self.min_y = min(self.min_y, y)
        if is_wall_latch(self.state):
            self.latch_frames += 1
            if y < self._prev_y:
                self.p132_rise_frames += 1
                if self._prev_pose != 132 or self._prev_y <= y:
                    # start / continue tracked in aggregate only
                    pass
        if self.writer is not None:
            self.writer.write_from_env(
                self.env,
                obs,
                action=action,
                frame_index=self.frame,
                room_id=int(self.state.room_id),
            )
        if self.frame % 8 == 0 or is_wall_latch(self.state) or pose in (131, 132):
            self.trace.append(
                {
                    "f": self.frame,
                    "x": int(self.state.samus_x),
                    "y": y,
                    "pose": pose,
                    "latch": is_wall_latch(self.state),
                    "rising132": pose == 132 and y < self._prev_y,
                    "reason": reason,
                }
            )
        self._prev_y = y
        self._prev_pose = pose
        return self.state

    def hold(self, n: int, *names: str, reason: str = "hold"):
        act = buttons(*names) if names else idle_action()
        st = self.state
        for _ in range(n):
            st = self.step(act, reason)
        return st

    def step_raw(self, action, reason: str = "replay"):
        """Step a raw 12-button action vector (human task frames)."""
        return self.step(action, reason)

    def log(self, msg: str) -> None:
        st = self.state
        print(
            f"[alc] {msg} f={self.frame} xy=({st.samus_x},{st.samus_y}) "
            f"p={st.pose} min_y={self.min_y} latch_f={self.latch_frames} "
            f"rise132={self.p132_rise_frames}",
            flush=True,
        )

    def snap(self, label: str) -> None:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        obs = self.env.render()
        if obs is None:
            return
        path = DEBUG_DIR / f"{label}.png"
        cv2.imwrite(str(path), cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        self.log(f"snap {label} → {path.name}")


def load_task_frames(path: Path) -> list:
    data = json.loads(path.read_text())
    return list(data["frames"])


def replay_frames(s: Sess, frames: list, *, start: int = 0, end: int | None = None) -> None:
    """Bit-exact replay of task action vectors [start, end)."""
    end = len(frames) if end is None else end
    s.log(f"replay frames[{start}:{end}] ({end - start}f)")
    for i in range(start, end):
        act = frames[i]
        if not isinstance(act, (list, tuple, np.ndarray)):
            raise TypeError(f"frame {i}: expected action vector, got {type(act)}")
        s.step_raw(list(act), reason=f"h_f{i}")
    s.log(f"replay done @ cut={end}")


def approach_mid_ledge(s: Sess) -> None:
    """Human2: door → mid platform y539 → mid ledge y459."""
    s.log("approach from flyway door")
    s.snap("00_boot")
    s.hold(6, "LEFT", reason="face_left")
    s.hold(24, "LEFT", "B", reason="run_left")
    s.hold(22, "RIGHT", "A", reason="hop_mid_plat")
    s.hold(12, "RIGHT", reason="land_mid")
    settle_hold(s, 20, reason="mid_plat_settle")  # type: ignore[arg-type]
    s.log("mid platform")
    s.hold(25, "LEFT", "B", "A", reason="to_ledge")
    s.hold(8, "LEFT", reason="ledge_brake")
    settle_hold(s, 16, reason="ledge_settle")  # type: ignore[arg-type]
    s.snap("01_mid_ledge")
    s.log("mid ledge")


def spin_up_right_wall(s: Sess) -> None:
    """Human2 f411–491: LEFT into left wall then RIGHT+A spin (p131)."""
    s.log("spin-up right wall")
    s.hold(30, "LEFT", "B", "A", reason="into_left_wall")
    s.hold(6, "B", "A", reason="coast")
    s.hold(2, "RIGHT", reason="face_right")
    s.hold(38, "RIGHT", "A", reason="spin_right_up")
    s.hold(5, "A", reason="spin_coast")
    s.snap("02_spin_peak")
    s.log("spin peak")


def run_wj_chain(s: Sess, *, count: int) -> None:
    """WJ1 right-wall (LEFT into), optional WJ2 left-wall (RIGHT into)."""
    s.log(f"WJ chain count={count}")

    def _stop(st) -> bool:
        return int(st.room_id) != ROOM_PARLOR or int(st.samus_y) <= 180

    walljump_once(s, WJ1_RIGHT_WALL, reason="alc_wj1", stop_when=_stop)  # type: ignore[arg-type]
    s.snap("03_after_wj1")
    s.log("after WJ1")
    if count < 2 or _stop(s.state):
        return
    s.hold(4, "LEFT", reason="wj2_drift")
    walljump_once(s, WJ2_LEFT_WALL, reason="alc_wj2", stop_when=_stop)  # type: ignore[arg-type]
    s.snap("04_after_wj2")
    s.log("after WJ2")


def recipe_h2_midrise(
    s: Sess,
    frames: list,
    *,
    spin_frames: int = 18,
    wj2_into: int = 28,
    wj3: bool = True,
) -> None:
    """Replay human2 to left wall, short spin (WJ1), mid-rise LEFT+A (WJ2).

    Human2 full spin is ~40f RIGHT+A to peak y256 at right wall — too late for
    a second bounce. Cut spin early so we still have rise budget when we face
    the right wall and fire LEFT+A (true mid-rise p132 pattern from human1).
    """
    replay_frames(s, frames, end=HUMAN2_LEFT_WALL_CUT)
    s.snap("01_left_wall")
    # Face + short first bounce off left wall (human2 uses RIGHT+A → p131).
    s.hold(2, "RIGHT", reason="face_right")
    s.hold(spin_frames, "RIGHT", "A", reason="wj1_spin_short")
    s.log(f"after short spin ({spin_frames}f)")
    s.snap("02_after_wj1_spin")
    # Mid-rise WJ2 on right wall: face LEFT briefly then LEFT+A (human1 BEST).
    s.hold(2, "LEFT", reason="face_left_wall")
    s.hold(wj2_into, "LEFT", "A", reason="wj2_midrise")
    s.hold(3, "A", reason="wj2_amid")
    s.hold(8, "RIGHT", "A", reason="wj2_flip")
    s.log("after WJ2 mid-rise")
    s.snap("03_after_wj2")
    if not wj3:
        return
    # Experimental WJ3: bounce off left wall again.
    s.hold(2, "RIGHT", reason="face_right_wj3")
    s.hold(20, "RIGHT", "A", reason="wj3")
    s.hold(3, "A", reason="wj3_amid")
    s.hold(8, "LEFT", "A", reason="wj3_flip")
    s.log("after WJ3")
    s.snap("04_after_wj3")


def recipe_h2_inject_at_y(
    s: Sess,
    frames: list,
    *,
    inject_below_y: int = 320,
    wj2_into: int = 30,
) -> None:
    """Replay human2 through spin until y <= inject_below_y, then LEFT+A WJ2.

    Tries to catch the human2 spin mid-rise (y still high number = lower on
    screen? No: lower y = higher in room). inject_below_y=320 means once Samus
    has risen to y<=320, fire mid-rise latch/WJ.
    """
    replay_frames(s, frames, end=HUMAN2_LEFT_WALL_CUT)
    s.snap("01_left_wall")
    s.hold(2, "RIGHT", reason="face_right")
    # Spin until height gate or budget
    budget = 50
    for i in range(budget):
        if int(s.state.samus_y) <= inject_below_y:
            s.log(f"inject gate y<={inject_below_y} at spin_i={i}")
            break
        s.hold(1, "RIGHT", "A", reason="spin_to_gate")
    else:
        s.log("spin budget exhausted before inject gate")
    s.snap("02_inject")
    s.hold(2, "LEFT", reason="face_left")
    s.hold(wj2_into, "LEFT", "A", reason="wj2_inject")
    s.hold(4, "A", reason="wj2_amid")
    s.hold(10, "RIGHT", "A", reason="wj2_flip")
    # try WJ3
    s.hold(2, "RIGHT", reason="face_r")
    s.hold(18, "RIGHT", "A", reason="wj3")
    s.hold(8, "LEFT", "A", reason="wj3_flip")
    s.log("inject recipe done")
    s.snap("03_done")


def recipe_h1_true(s: Sess, frames: list | None = None) -> None:
    """Human1 best true-WJ open-loop (or human2 to mid-ledge then recipe).

    From mid-ledge y459: RIGHT+B+A into right wall y~355, face LEFT, LEFT+A
    rise (p132 y356→258). Then attempt WJ2 toward left wall.
    """
    if frames is not None:
        # Human2 reaches mid-ledge ~f180–400; cut at f400 (on ledge, facing leftish)
        replay_frames(s, frames, end=400)
        s.snap("01_mid_ledge_h2")
        # Settle facing right for the approach
        s.hold(6, "RIGHT", reason="face_right_ledge")
        settle_hold(s, 8, reason="ledge_settle")  # type: ignore[arg-type]
    else:
        approach_mid_ledge(s)
        s.hold(4, "RIGHT", reason="face_right")

    s.log("h1-true: RIGHT into right wall")
    # Human1 BEST: ~28f RIGHT+B+A then ~7f B+A into wall at (875,355)
    s.hold(28, "RIGHT", "B", "A", reason="h1_approach")
    s.hold(7, "B", "A", reason="h1_into_wall")
    s.hold(2, reason="h1_release")
    s.hold(3, "LEFT", reason="h1_face_into")
    s.snap("02_right_wall")
    # True WJ1: LEFT+A ~36f rising
    s.hold(36, "LEFT", "A", reason="h1_wj1_true")
    s.hold(2, "A", reason="h1_wj1_end")
    s.log("after h1 true WJ1")
    s.snap("03_after_wj1")
    # Attempt WJ2: need left wall — keep LEFT then flip RIGHT+A
    s.hold(4, "LEFT", "A", reason="h1_to_left_wall")
    s.hold(2, "RIGHT", reason="h1_face_right")
    s.hold(24, "RIGHT", "A", reason="h1_wj2")
    s.hold(3, "A", reason="h1_wj2_amid")
    s.hold(10, "LEFT", "A", reason="h1_wj2_flip")
    s.log("after h1 WJ2 attempt")
    s.snap("04_after_wj2")


def recipe_short_chain(s: Sess, frames: list, *, count: int = 4) -> None:
    """Human2 to left wall, then short alternating walljump pulses."""
    replay_frames(s, frames, end=HUMAN2_LEFT_WALL_CUT)
    s.snap("01_left_wall")
    # First contact is left wall — WJ off left means into=LEFT? 
    # Human2 bounces with RIGHT+A (away from left wall). So into wall = LEFT
    # briefly then flip RIGHT+A — but human just does RIGHT+A from contact.
    # Match human: short RIGHT+A, then LEFT+A, then RIGHT+A...
    def _stop(st) -> bool:
        return int(st.room_id) != ROOM_PARLOR or int(st.samus_y) <= 180

    # Seed: face right + first bounce
    s.hold(2, "RIGHT", reason="face_r")
    s.hold(6, "RIGHT", "A", reason="seed_bounce")

    jumps: list[WallJumpTiming] = []
    for i in range(count):
        # odd: bounce off right wall (into LEFT); even: off left (into RIGHT)
        if i % 2 == 0:
            jumps.append(
                WallJumpTiming(
                    into="LEFT",
                    flip="RIGHT",
                    into_frames=10,
                    amid_frames=2,
                    flip_frames=12,
                )
            )
        else:
            jumps.append(
                WallJumpTiming(
                    into="RIGHT",
                    flip="LEFT",
                    into_frames=10,
                    amid_frames=2,
                    flip_frames=12,
                )
            )
    consecutive_walljumps(
        s,  # type: ignore[arg-type]
        jumps,
        reason="short_chain",
        gap_frames=1,
        stop_when=_stop,
    )
    s.log(f"short-chain {count} done")
    s.snap("02_chain_end")


def recipe_h2_full(s: Sess, frames: list) -> None:
    """Bit-exact full human2 climb (control / video compare)."""
    replay_frames(s, frames, end=len(frames))
    s.snap("01_h2_end")
    s.log("human2 full replay done")


def recipe_openloop(s: Sess, *, wj_count: int) -> None:
    approach_mid_ledge(s)
    spin_up_right_wall(s)
    run_wj_chain(s, count=wj_count)


def _make_writer(env, video: Path, *, scale: int, audio: bool):
    obs = env.render()
    if obs is None:
        obs, *_ = env.step(idle_action())
        obs = env.render()
    assert obs is not None
    cfg = VideoCaptureConfig(
        fps=60,
        scale=scale,
        crf=20,
        preset="veryfast",
        audio=audio,
        footer=True,
    )
    ar = None if not audio else int(env.em.get_audio_rate())  # type: ignore[attr-defined]
    writer = VideoRecorder(
        video,
        width=int(obs.shape[1]),
        height=int(obs.shape[0]),
        config=cfg,
        audio_rate=ar,
    )
    return writer, obs


def run_once(
    *,
    mode: str,
    state: Path,
    video: Path,
    wj_count: int,
    spin_frames: int,
    inject_y: int,
    scale: int,
    audio: bool,
    h2_frames: list | None,
) -> dict:
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    writer = None
    s: Sess | None = None
    err = None
    t0 = time.perf_counter()
    try:
        boot_from_state(env, state, settle_frames=6)
        for _ in range(4):
            env.step(idle_action())
            assist.apply(env.data, parse_env_state(env, mode="nav"))
        writer, obs = _make_writer(env, video, scale=scale, audio=audio)
        s = Sess(env, assist, writer)
        writer.write_from_env(
            env, obs, action=None, frame_index=0, room_id=int(s.state.room_id)
        )
        s.log(f"mode={mode}")
        if mode == "openloop":
            recipe_openloop(s, wj_count=wj_count)
        elif mode == "h2-full":
            assert h2_frames is not None
            recipe_h2_full(s, h2_frames)
        elif mode == "h2-midrise":
            assert h2_frames is not None
            recipe_h2_midrise(
                s, h2_frames, spin_frames=spin_frames, wj3=wj_count >= 3
            )
        elif mode == "h2-inject":
            assert h2_frames is not None
            recipe_h2_inject_at_y(s, h2_frames, inject_below_y=inject_y)
        elif mode == "h1-true":
            recipe_h1_true(s, h2_frames)
        elif mode == "short-chain":
            assert h2_frames is not None
            recipe_short_chain(s, h2_frames, count=wj_count)
        else:
            raise ValueError(f"unknown mode {mode}")
        s.hold(60, reason="end_hold")
        s.snap("99_final")
        s.log("done")
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        print(f"[alc] FAIL {err}", flush=True)
        if s is not None:
            try:
                s.snap("99_fail")
            except Exception:
                pass
    finally:
        enc = writer.frames if writer is not None else 0
        if writer is not None:
            writer.close()
        env.close()

    return {
        "kind": "parlor_alcatraz_wj_probe",
        "mode": mode,
        "source": str(state.resolve()),
        "video": str(video.resolve()),
        "wjCount": wj_count,
        "spinFrames": spin_frames,
        "injectY": inject_y,
        "success": err is None,
        "error": err,
        "frames": s.frame if s else 0,
        "encodedFrames": enc,
        "minY": s.min_y if s else None,
        "latchFrames": s.latch_frames if s else 0,
        "p132RiseFrames": s.p132_rise_frames if s else 0,
        "elapsedSec": round(time.perf_counter() - t0, 2),
        "final": (
            {
                "x": int(s.state.samus_x),
                "y": int(s.state.samus_y),
                "pose": int(s.state.pose),
            }
            if s
            else None
        ),
        "trace": s.trace if s else [],
        "humanTasks": [
            "super_metroid/tasks/parlor_left_human.json",
            "super_metroid/tasks/parlor_left_human2.json",
        ],
        "notesDoc": "super_metroid/docs/tasks/PARLOR_ALCATRAZ_HUMAN.md",
    }


def run_sweep(
    *,
    state: Path,
    scale: int,
    audio: bool,
    h2_frames: list,
) -> dict:
    """Try several mid-rise cut lengths; keep best min_y video."""
    # spin_frames: how long RIGHT+A after left wall before mid-rise LEFT+A
    candidates = [
        ("h2-midrise", {"spin_frames": 10, "wj_count": 3}),
        ("h2-midrise", {"spin_frames": 14, "wj_count": 3}),
        ("h2-midrise", {"spin_frames": 18, "wj_count": 3}),
        ("h2-midrise", {"spin_frames": 22, "wj_count": 3}),
        ("h2-midrise", {"spin_frames": 28, "wj_count": 3}),
        ("h2-inject", {"inject_y": 340, "wj_count": 2}),
        ("h2-inject", {"inject_y": 320, "wj_count": 2}),
        ("h2-inject", {"inject_y": 300, "wj_count": 2}),
        ("h2-inject", {"inject_y": 280, "wj_count": 2}),
        ("h1-true", {"wj_count": 2}),
        ("short-chain", {"wj_count": 4}),
        ("h2-full", {"wj_count": 1}),
    ]
    results = []
    best = None
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    for i, (mode, kw) in enumerate(candidates):
        tag = f"{mode}_" + "_".join(f"{k}{v}" for k, v in sorted(kw.items()))
        video = RECORDINGS_DIR / f"parlor_alc_sweep_{i:02d}_{tag}.mp4"
        print(f"\n[sweep] {i+1}/{len(candidates)} {tag}", flush=True)
        payload = run_once(
            mode=mode,
            state=state,
            video=video,
            wj_count=int(kw.get("wj_count", 2)),
            spin_frames=int(kw.get("spin_frames", 18)),
            inject_y=int(kw.get("inject_y", 320)),
            scale=scale,
            audio=audio,
            h2_frames=h2_frames,
        )
        summary = {
            "tag": tag,
            "mode": mode,
            "minY": payload["minY"],
            "latchFrames": payload["latchFrames"],
            "p132RiseFrames": payload["p132RiseFrames"],
            "final": payload["final"],
            "video": str(video),
            "success": payload["success"],
            "error": payload["error"],
        }
        results.append(summary)
        print(
            f"[sweep] → min_y={summary['minY']} latch={summary['latchFrames']} "
            f"rise132={summary['p132RiseFrames']} final={summary['final']}",
            flush=True,
        )
        if summary["minY"] is not None and (
            best is None or summary["minY"] < best["minY"]
        ):
            best = summary

    report = {
        "kind": "parlor_alcatraz_wj_sweep",
        "results": results,
        "best": best,
        "notesDoc": "super_metroid/docs/tasks/PARLOR_ALCATRAZ_HUMAN.md",
    }
    out = RECORDINGS_DIR / "parlor_alcatraz_wj_sweep.json"
    # Drop per-run traces from disk report already not included
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n[sweep] best={best}", flush=True)
    print(f"[sweep] wrote {out}", flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", type=Path, default=DEFAULT_STATE)
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument(
        "--mode",
        default="h2-midrise",
        choices=(
            "openloop",
            "h2-full",
            "h2-midrise",
            "h2-inject",
            "h1-true",
            "short-chain",
            "sweep",
        ),
    )
    ap.add_argument("--wj-count", type=int, default=3)
    ap.add_argument(
        "--spin-frames",
        type=int,
        default=18,
        help="h2-midrise: RIGHT+A frames after left wall before WJ2",
    )
    ap.add_argument(
        "--inject-y",
        type=int,
        default=320,
        help="h2-inject: fire WJ2 once samus_y <= this",
    )
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--scale", type=int, default=2)
    args = ap.parse_args()

    if not args.state.is_file():
        raise SystemExit(f"missing state: {args.state}")

    h2_frames = None
    if args.mode != "openloop" or True:
        if HUMAN2_TASK.is_file():
            h2_frames = load_task_frames(HUMAN2_TASK)
        elif args.mode not in ("openloop",):
            raise SystemExit(f"missing human2 task: {HUMAN2_TASK}")

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "sweep":
        assert h2_frames is not None
        report = run_sweep(
            state=args.state,
            scale=args.scale,
            audio=not args.no_audio,
            h2_frames=h2_frames,
        )
        print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
        # print compact results table
        for r in report["results"]:
            print(
                f"  {r['tag']:40s} min_y={r['minY']} "
                f"latch={r['latchFrames']} rise132={r['p132RiseFrames']}"
            )
        sys.exit(0)

    video = args.video or (
        RECORDINGS_DIR / f"parlor_alcatraz_{args.mode}_wj{args.wj_count}.mp4"
    )
    report_path = args.report or (
        RECORDINGS_DIR / f"parlor_alcatraz_{args.mode}_wj{args.wj_count}.json"
    )
    print(f"[alc] state={args.state}", flush=True)
    print(
        f"[alc] mode={args.mode} video={video} wj={args.wj_count} "
        f"spin={args.spin_frames} inject_y={args.inject_y}",
        flush=True,
    )

    payload = run_once(
        mode=args.mode,
        state=args.state,
        video=video,
        wj_count=args.wj_count,
        spin_frames=args.spin_frames,
        inject_y=args.inject_y,
        scale=args.scale,
        audio=not args.no_audio,
        h2_frames=h2_frames,
    )
    report_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "trace"}, indent=2))
    print(f"[alc] wrote {video}", flush=True)
    sys.exit(0 if payload["success"] else 1)


if __name__ == "__main__":
    main()

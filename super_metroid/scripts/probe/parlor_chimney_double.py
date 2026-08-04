#!/usr/bin/env python3
"""Open-loop Alcatraz left-chimney double wall-jump (no human replay).

Derives climb from geometry + live RAM. Proven recipe (2026-08-03):

1. Door (968,651) → left-wall base (805,545) via hop ladder
2. Base → mid ledge ~(828,459) via spin_full 40/28 ×3
3. Chimney: into left wall, alternate RIGHT+A spin / LEFT+A latch

Best open-loop class: **chim min_y ≈ 243** with chained rises
y459→363 (+96) then y363→266 (+97). Shaft lip goal y≤210 still open.

```bash
uv run python super_metroid/scripts/probe/parlor_chimney_double.py
uv run python super_metroid/scripts/probe/parlor_chimney_double.py --recipe ext_40
uv run python super_metroid/scripts/probe/parlor_chimney_double.py --recipe baseline
```

Artifacts: ``recordings/parlor_chimney_double_best.mp4`` (+ matching .json).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
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

ROOM_PARLOR = 0x92FD
STATE = (
    GAME_DIR
    / "custom_integrations"
    / "SuperMetroid-Snes"
    / "scratch"
    / "post_torizo_parlor_continuous.state"
)
DEBUG = GAME_DIR / "debug" / "spore" / "parlor_chimney_double"

# Geometry pins (world) from live recon.
LEFT_WALL_X = 805
BASE_Y = 545
MID_LEDGE_Y = 459
SHAFT_LIP_Y = 210

POSE_SPIN_WJ = frozenset({25, 26, 129, 130, 131, 132})
GROUNDED = frozenset({1, 2, 9, 10, 5, 6, 7, 8})


@dataclass
class Sess:
    env: object
    assist: object
    writer: object = None
    frame: int = 0
    min_y: int = 9999
    chim_min: int = 9999
    chim_arm: int = 0
    rise: int = 0
    bounces: list = field(default_factory=list)
    trace: list = field(default_factory=list)
    _py: int = 0
    _rising: bool = False
    _y0: int = 0
    _x0: int = 0
    _f0: int = 0

    def __post_init__(self) -> None:
        self.state = parse_env_state(self.env, mode="nav")
        self.min_y = int(self.state.samus_y)
        self._py = self.min_y

    def step(self, *names: str, reason: str = "") -> object:
        act = buttons(*names) if names else idle_action()
        obs, *_ = self.env.step(act)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        self.assist.apply(self.env.data, self.state)
        x = int(self.state.samus_x)
        y = int(self.state.samus_y)
        p = int(self.state.pose)
        self.min_y = min(self.min_y, y)
        if self.frame >= self.chim_arm > 0:
            self.chim_min = min(self.chim_min, y)
        rising = p in POSE_SPIN_WJ and y < self._py
        if rising and self.frame >= self.chim_arm > 0:
            self.rise += 1
            if not self._rising:
                self._rising = True
                self._y0, self._x0, self._f0 = self._py, x, self.frame
        elif self._rising:
            gain = self._y0 - self._py
            if gain >= 30 and self.frame >= self.chim_arm:
                self.bounces.append(
                    {
                        "f0": self._f0,
                        "y0": self._y0,
                        "y1": self._py,
                        "gain": gain,
                        "x0": self._x0,
                        "x1": x,
                    }
                )
            self._rising = False
        if self.writer is not None:
            self.writer.write_from_env(
                self.env,
                obs,
                action=act,
                frame_index=self.frame,
                room_id=int(self.state.room_id),
            )
        if self.frame % 6 == 0 or p in (131, 132) or reason.startswith("M"):
            self.trace.append(
                {"f": self.frame, "x": x, "y": y, "p": p, "r": reason}
            )
        self._py = y
        return self.state

    def hold(self, n: int, *names: str, reason: str = "") -> object:
        for _ in range(n):
            self.step(*names, reason=reason)
        return self.state

    def log(self, msg: str) -> None:
        st = self.state
        print(
            f"[chim] {msg} f={self.frame} ({st.samus_x},{st.samus_y}) p={st.pose} "
            f"min_y={self.min_y} chim={self.chim_min} bounces={len(self.bounces)}",
            flush=True,
        )

    def snap(self, label: str) -> None:
        DEBUG.mkdir(parents=True, exist_ok=True)
        obs = self.env.render()
        if obs is None:
            return
        path = DEBUG / f"{label}.png"
        cv2.imwrite(str(path), cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        self.log(f"snap {label}")

    def unmorph_if_needed(self) -> None:
        p = int(self.state.pose)
        if p in (29, 30, 31, 32, 49, 50, 65, 66, 165, 166, 167):
            self.hold(8, "UP", reason="unmorph")
            self.hold(6, reason="unmorph_set")


def boot(video: Path | None, *, scale: int = 2) -> Sess:
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    boot_from_state(env, STATE, settle_frames=6)
    for _ in range(4):
        env.step(idle_action())
        assist.apply(env.data, parse_env_state(env, mode="nav"))
    obs = env.render()
    if obs is None:
        obs, *_ = env.step(idle_action())
        obs = env.render()
    assert obs is not None
    writer = None
    if video is not None:
        video.parent.mkdir(parents=True, exist_ok=True)
        cfg = VideoCaptureConfig(
            fps=60,
            scale=scale,
            crf=18,
            preset="veryfast",
            audio=False,
            footer=True,
        )
        writer = VideoRecorder(
            video,
            width=int(obs.shape[1]),
            height=int(obs.shape[0]),
            config=cfg,
            audio_rate=None,
        )
        st0 = parse_env_state(env, mode="nav")
        writer.write_from_env(
            env, obs, action=None, frame_index=0, room_id=int(st0.room_id)
        )
    return Sess(env, assist, writer)


def close(s: Sess) -> None:
    if s.writer is not None:
        s.writer.close()
    s.env.close()


# ---------------------------------------------------------------------------
# Phase A/B: door → base (805,545) → mid ledge (~828,459)
# ---------------------------------------------------------------------------


def land_base(s: Sess) -> bool:
    s.log("phase A: base")
    s.snap("00_boot")
    s.hold(4, "LEFT", reason="face")
    for i in range(18):
        s.hold(1, "LEFT", "Y" if i % 6 == 0 else "B", reason="run")
    for jlen in (10, 12, 16):
        s.hold(2, "LEFT", reason="face")
        s.hold(jlen, "LEFT", "A", reason=f"hop{jlen}")
        s.hold(18, reason="land")
        s.unmorph_if_needed()
        y, x, p = int(s.state.samus_y), int(s.state.samus_x), int(s.state.pose)
        if y <= 550 and x <= 820 and p in GROUNDED:
            s.log("base landed")
            s.snap("01_base")
            return True
    s.hold(2, "LEFT")
    s.hold(18, "LEFT", "A")
    s.hold(20)
    s.unmorph_if_needed()
    s.snap("01_base")
    s.log("base attempt end")
    return int(s.state.samus_y) <= 560


def to_mid_ledge(s: Sess) -> bool:
    """spin_full 40/28 ×3 — proven base→mid-ledge open-loop."""
    s.log("phase B: mid ledge")
    s.hold(2, "RIGHT", reason="face_r")
    for i in range(3):
        s.hold(40, "RIGHT", "A", reason=f"mid_spin{i}")
        s.hold(2, "LEFT", reason="mid_face")
        s.hold(28, "LEFT", "A", reason=f"mid_latch{i}")
        s.log(f"mid stage {i}")
        if int(s.state.samus_y) <= 470 and int(s.state.pose) in GROUNDED:
            break
    s.hold(25, reason="ledge_settle")
    s.snap("02_ledge")
    s.log("mid ledge done")
    return int(s.state.samus_y) <= 480


def arm_chimney(s: Sess) -> None:
    s.chim_arm = s.frame + 1
    s.chim_min = int(s.state.samus_y)
    s.bounces.clear()
    s.rise = 0
    s.log("phase C: chimney armed")


# ---------------------------------------------------------------------------
# Phase C: chimney wall-jump chain
# ---------------------------------------------------------------------------


def _into_left_wall(s: Sess) -> None:
    s.hold(3, "LEFT", reason="face_l")
    s.hold(22, "LEFT", "B", "A", reason="to_Lwall")
    s.hold(4, "LEFT", reason="press_L")
    s.snap("03_left_wall")


def recipe_baseline(s: Sess) -> None:
    """left_spin_38_32 — chim min_y ≈ 252, double rise 459→359→263."""
    _into_left_wall(s)
    for i in range(3):
        s.hold(2, "RIGHT", reason="face_r")
        s.hold(38, "RIGHT", "A", reason=f"spin{i}")
        s.log(f"spin{i}")
        s.hold(2, "LEFT", reason="face_l")
        s.hold(32, "LEFT", "A", reason=f"latch{i}")
        s.log(f"latch{i}")
        s.snap(f"04_b{i}")
        if int(s.state.samus_y) <= SHAFT_LIP_Y:
            break


def recipe_ext_40(s: Sess) -> None:
    """ext_40_36_32 — chim min_y ≈ 243, up to 5 bounce clusters."""
    _into_left_wall(s)
    pairs = ((40, 30), (36, 28), (32, 26), (28, 24))
    for i, (spin, latch) in enumerate(pairs):
        s.hold(2, "RIGHT", reason="face_r")
        s.hold(spin, "RIGHT", "A", reason=f"spin{i}")
        s.log(f"spin{i}")
        s.hold(2, "LEFT", reason="face_l")
        s.hold(latch, "LEFT", "A", reason=f"latch{i}")
        s.log(f"latch{i}")
        s.snap(f"04_b{i}")
        if int(s.state.samus_y) <= SHAFT_LIP_Y:
            break


def recipe_midrise_260(s: Sess) -> None:
    """Cut second spin at y≤260 then keep chaining — chim ≈ 243."""
    _into_left_wall(s)
    s.hold(2, "RIGHT")
    s.hold(38, "RIGHT", "A", reason="wj1_spin")
    s.hold(2, "LEFT")
    s.hold(28, "LEFT", "A", reason="wj1_latch")
    s.log("wj1 done")
    s.snap("04_wj1")
    s.hold(2, "RIGHT")
    for i in range(50):
        if int(s.state.samus_y) <= 260:
            s.log(f"mid-rise cut @ spin_i={i}")
            break
        s.hold(1, "RIGHT", "A", reason="wj2_spin")
    s.hold(2, "LEFT")
    s.hold(24, "LEFT", "A", reason="wj3_latch")
    s.hold(2, "RIGHT")
    s.hold(30, "RIGHT", "A", reason="wj3_spin")
    s.hold(2, "LEFT")
    s.hold(20, "LEFT", "A", reason="wj4")
    s.snap("04_wj3")


RECIPES = {
    "baseline": recipe_baseline,
    "ext_40": recipe_ext_40,
    "midrise_260": recipe_midrise_260,
}


def run(recipe: str, video: Path, *, scale: int = 2) -> dict:
    s = boot(video, scale=scale)
    err = None
    t0 = time.perf_counter()
    try:
        land_base(s)
        to_mid_ledge(s)
        arm_chimney(s)
        RECIPES[recipe](s)
        s.hold(50, reason="end")
        s.snap("99_final")
        s.log("done")
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        print(f"[chim] FAIL {err}", flush=True)
        try:
            s.snap("99_fail")
        except Exception:
            pass
    finally:
        close(s)

    lip = s.chim_min <= SHAFT_LIP_Y
    double_ok = len(s.bounces) >= 2 and s.chim_min <= 280
    out = {
        "kind": "parlor_chimney_double",
        "recipe": recipe,
        "source": str(STATE.resolve()),
        "video": str(video.resolve()),
        "success": err is None,
        "error": err,
        "frames": s.frame,
        "minY": s.min_y,
        "chimMinY": s.chim_min if s.chim_min < 9999 else None,
        "riseFrames": s.rise,
        "bounceCount": len(s.bounces),
        "bounces": s.bounces,
        "doubleClass": double_ok,  # ≥2 rises, peak ≤280
        "lipClass": lip,  # y≤210
        "final": {
            "x": int(s.state.samus_x),
            "y": int(s.state.samus_y),
            "pose": int(s.state.pose),
            "room": f"0x{int(s.state.room_id):04X}",
        },
        "elapsedSec": round(time.perf_counter() - t0, 2),
        "trace": s.trace,
        "notes": (
            "Open-loop only. Double class: chained rises from mid-ledge "
            "(e.g. 459→363 and 363→266). Lip y≤210 not yet closed."
        ),
    }
    print(
        f"[chim] RESULT recipe={recipe} chim_min={out['chimMinY']} "
        f"bounces={out['bounceCount']} double={double_ok} lip={lip}",
        flush=True,
    )
    for b in s.bounces:
        print(
            f"       y{b['y0']}→{b['y1']} +{b['gain']} x{b['x0']}→{b['x1']}",
            flush=True,
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--recipe",
        default="ext_40",
        choices=tuple(RECIPES),
        help="Chimney pulse recipe (default: best open-loop)",
    )
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--scale", type=int, default=2)
    args = ap.parse_args()

    if not STATE.is_file():
        raise SystemExit(f"missing state: {STATE}")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG.mkdir(parents=True, exist_ok=True)

    video = args.video or (
        RECORDINGS_DIR / f"parlor_chimney_double_{args.recipe}.mp4"
    )
    report_path = args.report or (
        RECORDINGS_DIR / f"parlor_chimney_double_{args.recipe}.json"
    )
    print(f"[chim] recipe={args.recipe} video={video}", flush=True)
    payload = run(args.recipe, video, scale=args.scale)
    report_path.write_text(json.dumps(payload, indent=2) + "\n")
    # Also write/update best pointer when double class hits
    if payload.get("doubleClass"):
        best_vid = RECORDINGS_DIR / "parlor_chimney_double_best.mp4"
        best_json = RECORDINGS_DIR / "parlor_chimney_double_best.json"
        if video.resolve() != best_vid.resolve():
            best_vid.write_bytes(video.read_bytes())
        best_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"[chim] promoted best → {best_vid}", flush=True)
    print(json.dumps({k: v for k, v in payload.items() if k != "trace"}, indent=2))
    sys.exit(0 if payload["success"] else 1)


if __name__ == "__main__":
    main()

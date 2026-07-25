"""Pan Scene1 panorama then settle-gated click hunt for +1500.

AfterFind: move cursor to right/left edge, hold to scroll camera, snapshot,
then grid-click without P2-A. Also supports waldo_first from Scene1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from PIL import Image

from great_waldo_search.paths import GAME, GAME_DIR, RECORDINGS_DIR
from great_waldo_search.targets import (
    CURSOR_X_ADDR,
    CURSOR_Y_ADDR,
    FOUND_FLAG_ADDR,
    SCORE_HI_ADDR,
    SCORE_LO_ADDR,
    score_u16,
)
from retro_harness.env import make_env, save_state
from snes_oneshot.actions import buttons_multi, idle_action_multi
from snes_oneshot.cursor import CursorPose, CursorTarget, step_toward_target


def _configure_headless() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("SDL_SOFTWARE_RENDERER", "1")


def _idle(env: object, frames: int) -> np.ndarray:
    obs = None
    for _ in range(max(frames, 1)):
        obs, *_rest = env.step(idle_action_multi(players=2))  # type: ignore[attr-defined]
    assert obs is not None
    return np.asarray(obs)


def _drive(env: object, target: CursorTarget, frames: int = 700) -> CursorPose:
    for _ in range(frames):
        ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
        pose = CursorPose(int(ram[CURSOR_X_ADDR]), int(ram[CURSOR_Y_ADDR]))
        action = step_toward_target(pose, target, fast_button="Y")
        if action.reason == "confirm_at_target":
            return pose
        multi = idle_action_multi(players=2)
        multi[:12] = list(action.action)
        env.step(multi)  # type: ignore[attr-defined]
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    return CursorPose(int(ram[CURSOR_X_ADDR]), int(ram[CURSOR_Y_ADDR]))


def _score_flag(env: object) -> tuple[int, int, int, int]:
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    return (
        score_u16(ram[SCORE_LO_ADDR], ram[SCORE_HI_ADDR]),
        int(ram[FOUND_FLAG_ADDR]),
        int(ram[CURSOR_X_ADDR]),
        int(ram[CURSOR_Y_ADDR]),
    )


def _settle(env: object) -> tuple[int, int, bool, np.ndarray]:
    obs = _idle(env, 100)
    samples: list[tuple[int, int]] = []
    for _ in range(4):
        s, f, _x, _y = _score_flag(env)
        samples.append((s, f))
        obs = _idle(env, 8)
    return samples[-1][0], samples[-1][1], len(set(samples)) == 1, obs


def _pan(env: object, direction: str, frames: int) -> None:
    edge_x = 240 if direction == "RIGHT" else 16
    _drive(env, CursorTarget(x=edge_x, y=100, deadzone=3))
    btn = (direction, "Y")
    for _ in range(frames):
        env.step(buttons_multi(p1=btn))  # type: ignore[attr-defined]
    _idle(env, 8)


def run_pan_hunt(
    *,
    mode: str,
    direction: str,
    pan_frames: int,
    step: int,
) -> dict:
    """Pan then coarse-grid hunt one viewport."""
    _configure_headless()
    out = RECORDINGS_DIR / f"pan_hunt_{mode}_{direction}_{pan_frames}"
    out.mkdir(parents=True, exist_ok=True)
    state = "Scene1_AfterFind1000" if mode == "after_find" else "Scene1"
    want = 2500 if mode == "after_find" else 1500

    env = make_env(
        game=GAME,
        state=state,
        game_dir=GAME_DIR,
        render_mode="rgb_array",
        players=2,
    )
    rows: list[dict] = []
    hit = None
    try:
        env.reset()
        _idle(env, 8)
        _pan(env, direction, pan_frames)
        # Capture in-memory state after pan via save/load for each click.
        pan_state = env.em.get_state()  # type: ignore[attr-defined]
        obs = _idle(env, 2)
        Image.fromarray(obs).save(out / "viewport.png")
        s0, f0, x0, y0 = _score_flag(env)
        print(
            f"[pan] {mode} {direction}x{pan_frames} "
            f"score={s0} found={f0} cursor=({x0},{y0})"
        )

        points = [
            (x, y)
            for y in range(28, 165, step)
            for x in range(12, 248, step)
        ]
        for idx, (x, y) in enumerate(points):
            env.em.set_state(pan_state)  # type: ignore[attr-defined]
            _idle(env, 2)
            pose = _drive(env, CursorTarget(x=x, y=y, deadzone=1))
            sb, fb, _cx, _cy = _score_flag(env)
            for _ in range(6):
                env.step(buttons_multi(p1=("A",)))
            sa, fa, stable, obs = _settle(env)
            delta = sa - sb
            row = {
                "x": x,
                "y": y,
                "pose": {"x": pose.x, "y": pose.y},
                "delta": delta,
                "score_after": sa,
                "found_after": fa,
                "stable": stable,
            }
            if delta != 0 or fa != fb:
                Image.fromarray(obs).save(
                    out / f"hit_{x}_{y}_d{delta}.png"
                )
                print(
                    f"[pan] HIT ({x},{y}) d={delta} total={sa} "
                    f"flag={fa} stable={stable}"
                )
                rows.append(row)
            if stable and (sa >= want or delta >= 1500):
                hit = row
                path = save_state(env, GAME_DIR, GAME, "Scene1_Cleared")
                Image.fromarray(obs).save(out / "SUCCESS.png")
                print(f"[pan] SUCCESS {row} -> {path}")
                break
            if idx % 40 == 0:
                print(f"[pan] {idx}/{len(points)}")
    finally:
        env.close()

    report = {
        "mode": mode,
        "direction": direction,
        "pan_frames": pan_frames,
        "hit": hit,
        "hits": rows,
    }
    path = out / "report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[pan] wrote {path} hits={len(rows)}")
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI for panorama Waldo hunt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("after_find", "waldo_first"), default="after_find")
    parser.add_argument("--direction", choices=("RIGHT", "LEFT"), default="RIGHT")
    parser.add_argument("--pan-frames", type=int, default=120)
    parser.add_argument("--step", type=int, default=10)
    args = parser.parse_args(argv)
    run_pan_hunt(
        mode=args.mode,
        direction=args.direction,
        pan_frames=args.pan_frames,
        step=args.step,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Dense settle-gated click hunt for +1500 Waldo.

Modes:
- after_find: load Scene1_AfterFind1000, NO second P2-A, drive+click grid
- waldo_first: load Scene1, skip scroll assist, drive+click grid
Success: settled score >= 2500 (after_find) or == 1500 (waldo_first),
or found-flag change with stable score.
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
from retro_harness.actions import buttons_multi, idle_action_multi
from retro_harness.cursor import CursorPose, CursorTarget, step_toward_target


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


def _drive(env: object, target: CursorTarget, frames: int = 600) -> CursorPose:
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


def _score_flag(env: object) -> tuple[int, int]:
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    return (
        score_u16(ram[SCORE_LO_ADDR], ram[SCORE_HI_ADDR]),
        int(ram[FOUND_FLAG_ADDR]),
    )


def _settle(env: object, settle: int = 100) -> tuple[int, int, bool, np.ndarray]:
    obs = _idle(env, settle)
    samples: list[tuple[int, int]] = []
    for _ in range(4):
        samples.append(_score_flag(env))
        obs = _idle(env, 8)
    stable = len(set(samples)) == 1
    score, found = samples[-1]
    return score, found, stable, obs


def _grid(x0: int, x1: int, y0: int, y1: int, step: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for y in range(y0, y1 + 1, step):
        for x in range(x0, x1 + 1, step):
            pts.append((x, y))
    return pts


def run_hunt(
    *,
    mode: str,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    step: int,
) -> dict:
    """Run a dense grid hunt in the chosen mode."""
    _configure_headless()
    out = RECORDINGS_DIR / f"waldo_hunt_{mode}"
    out.mkdir(parents=True, exist_ok=True)
    state = "Scene1_AfterFind1000" if mode == "after_find" else "Scene1"
    want_total = 2500 if mode == "after_find" else 1500
    rows: list[dict] = []
    hit: dict | None = None

    points = _grid(x0, x1, y0, y1, step)
    print(f"[hunt] mode={mode} state={state} points={len(points)}")

    env = make_env(
        game=GAME,
        state=state,
        game_dir=GAME_DIR,
        render_mode="rgb_array",
        players=2,
    )
    try:
        for idx, (x, y) in enumerate(points):
            env.close()
            env = make_env(
                game=GAME,
                state=state,
                game_dir=GAME_DIR,
                render_mode="rgb_array",
                players=2,
            )
            env.reset()
            _idle(env, 10)
            pose = _drive(env, CursorTarget(x=x, y=y, deadzone=1))
            sb, fb = _score_flag(env)
            for _ in range(6):
                env.step(buttons_multi(p1=("A",)))
            sa, fa, stable, obs = _settle(env)
            delta = sa - sb
            row = {
                "x": x,
                "y": y,
                "pose": {"x": pose.x, "y": pose.y},
                "score_before": sb,
                "score_after": sa,
                "delta": delta,
                "found_before": fb,
                "found_after": fa,
                "stable": stable,
            }
            if delta != 0 or fa != fb:
                Image.fromarray(obs).save(
                    out / f"hit_{x}_{y}_d{delta}_f{fa}.png"
                )
                print(
                    f"[hunt] HIT ({x},{y}) {sb}->{sa} d={delta} "
                    f"flag {fb}->{fa} stable={stable}"
                )
            rows.append(row)
            if stable and (sa >= want_total or delta >= 1500):
                hit = row
                path = save_state(
                    env,
                    GAME_DIR,
                    GAME,
                    "Scene1_Cleared" if sa >= 2500 else "Scene1_Waldo1500",
                )
                Image.fromarray(obs).save(out / "SUCCESS.png")
                print(f"[hunt] SUCCESS {row} saved {path}")
                break
            if idx % 25 == 0:
                print(f"[hunt] progress {idx}/{len(points)}")
    finally:
        env.close()

    report = {
        "mode": mode,
        "state": state,
        "bbox": [x0, y0, x1, y1],
        "step": step,
        "hit": hit,
        "nonzero": [r for r in rows if r["delta"] != 0 or r["found_after"] != r["found_before"]],
        "rows": rows,
    }
    path = out / "report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[hunt] wrote {path} nonzero={len(report['nonzero'])}")
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI for Waldo grid hunt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("after_find", "waldo_first"),
        default="after_find",
    )
    parser.add_argument("--x0", type=int, default=180)
    parser.add_argument("--x1", type=int, default=240)
    parser.add_argument("--y0", type=int, default=30)
    parser.add_argument("--y1", type=int, default=130)
    parser.add_argument("--step", type=int, default=4)
    args = parser.parse_args(argv)
    run_hunt(
        mode=args.mode,
        x0=args.x0,
        x1=args.x1,
        y0=args.y0,
        y1=args.y1,
        step=args.step,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

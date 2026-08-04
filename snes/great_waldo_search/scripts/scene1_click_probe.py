"""Systematic Scene1 click probe: P2-A assist + candidate A-press RAM diffs.

Loads Scene1.state with players=2, holds controller-2 A to seek an objective,
optionally nudges to candidate coords, presses P1 A, and records score /
found-flag deltas. Writes JSON + PNGs under recordings/click_probe_out/.
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
    SCENE1_TARGETS,
    SCORE_HI_ADDR,
    SCORE_LO_ADDR,
    score_u16,
)
from retro_harness.env import make_env
from retro_harness.actions import buttons_multi, idle_action_multi
from retro_harness.cursor import CursorPose, CursorTarget, step_toward_target
from retro_harness.ram_state import diff_changed, snapshot


def _configure_headless() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("SDL_SOFTWARE_RENDERER", "1")


def _idle(env: object, frames: int) -> np.ndarray:
    obs = None
    for _ in range(frames):
        obs, *_rest = env.step(idle_action_multi(players=2))  # type: ignore[attr-defined]
    assert obs is not None
    return obs


def _hold_p2a(env: object, frames: int) -> CursorPose:
    for _ in range(frames):
        env.step(buttons_multi(p2=("A",)))  # type: ignore[attr-defined]
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    return CursorPose(int(ram[CURSOR_X_ADDR]), int(ram[CURSOR_Y_ADDR]))


def _drive(
    env: object,
    target: CursorTarget,
    *,
    frames: int = 400,
) -> CursorPose:
    for _ in range(frames):
        ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
        pose = CursorPose(int(ram[CURSOR_X_ADDR]), int(ram[CURSOR_Y_ADDR]))
        action = step_toward_target(pose, target, fast_button="Y")
        if action.reason == "confirm_at_target":
            return pose
        # Expand 12-button action into 24-button multi vector.
        multi = idle_action_multi(players=2)
        multi[:12] = list(action.action)
        env.step(multi)  # type: ignore[attr-defined]
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    return CursorPose(int(ram[CURSOR_X_ADDR]), int(ram[CURSOR_Y_ADDR]))


def _click_and_diff(env: object, *, settle: int = 90) -> dict:
    before = snapshot(np.asarray(env.get_ram(), dtype=np.uint8))  # type: ignore[attr-defined]
    for _ in range(6):
        env.step(buttons_multi(p1=("A",)))  # type: ignore[attr-defined]
    obs = _idle(env, settle)
    after = snapshot(np.asarray(env.get_ram(), dtype=np.uint8))  # type: ignore[attr-defined]
    deltas = [
        {
            "address": d.address,
            "before": d.before,
            "after": d.after,
            "delta": d.delta,
        }
        for d in diff_changed(before, after, limit=None)
        if d.address < 0x300
    ]
    return {
        "score_before": score_u16(before[SCORE_LO_ADDR], before[SCORE_HI_ADDR]),
        "score_after": score_u16(after[SCORE_LO_ADDR], after[SCORE_HI_ADDR]),
        "found_before": int(before[FOUND_FLAG_ADDR]),
        "found_after": int(after[FOUND_FLAG_ADDR]),
        "cursor": {
            "x": int(after[CURSOR_X_ADDR]),
            "y": int(after[CURSOR_Y_ADDR]),
        },
        "low_deltas": deltas[:80],
        "obs": obs,
    }


def run_probe(
    *,
    p2a_frames: int = 300,
    click_assist_landing: bool = True,
    also_candidates: bool = True,
) -> dict:
    """Run P2-A assist probe and optional candidate clicks from Scene1."""
    _configure_headless()
    out_dir = RECORDINGS_DIR / "click_probe_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    env = make_env(
        game=GAME,
        state="Scene1",
        game_dir=GAME_DIR,
        render_mode="rgb_array",
        players=2,
    )
    try:
        obs, _info = env.reset()
        _idle(env, 12)
        Image.fromarray(obs).save(out_dir / "00_start.png")

        pose = _hold_p2a(env, p2a_frames)
        Image.fromarray(_idle(env, 2)).save(out_dir / "01_assist.png")
        print(f"[probe] P2-A landing pose=({pose.x},{pose.y})")

        if click_assist_landing:
            result = _click_and_diff(env)
            Image.fromarray(result.pop("obs")).save(out_dir / "02_assist_click.png")
            row = {
                "label": "p2a_landing",
                "target": {"x": pose.x, "y": pose.y},
                **result,
            }
            rows.append(row)
            print(
                f"[probe] assist click score "
                f"{row['score_before']}->{row['score_after']} "
                f"found {row['found_before']}->{row['found_after']}"
            )

        if also_candidates:
            for target in SCENE1_TARGETS:
                # Fresh load so candidates are independent.
                env.close()
                env = make_env(
                    game=GAME,
                    state="Scene1",
                    game_dir=GAME_DIR,
                    render_mode="rgb_array",
                    players=2,
                )
                env.reset()
                _idle(env, 10)
                _hold_p2a(env, p2a_frames)
                final = _drive(env, target)
                result = _click_and_diff(env)
                Image.fromarray(result.pop("obs")).save(
                    out_dir / f"cand_{target.label}.png"
                )
                row = {
                    "label": target.label,
                    "target": {"x": target.x, "y": target.y},
                    "final_pose": {"x": final.x, "y": final.y},
                    **result,
                }
                rows.append(row)
                print(
                    f"[probe] {target.label} score "
                    f"{row['score_before']}->{row['score_after']} "
                    f"found {row['found_before']}->{row['found_after']}"
                )
    finally:
        env.close()

    report = {
        "state": "Scene1",
        "p2a_frames": p2a_frames,
        "rows": rows,
        "notes": [
            "P2-A seeks an objective; landing (32,100) yields +1000.",
            "Score bytes 0x0047/0x0048 are noisy mid-animation.",
            "0x01BD becomes 2 after the confirmed +1000 find.",
            "Second objective (Waldo +1500) not yet confirmed.",
        ],
    }
    path = out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[probe] wrote {path}")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2a-frames", type=int, default=300)
    parser.add_argument("--no-candidates", action="store_true")
    parser.add_argument("--assist-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry for the Scene1 click probe."""
    args = _build_parser().parse_args(argv)
    run_probe(
        p2a_frames=args.p2a_frames,
        click_assist_landing=True,
        also_candidates=not args.no_candidates and not args.assist_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

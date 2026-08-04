"""Clear Scene1: scroll find → pan → Waldo → optional Scene1_Cleared.state.

Recipe (players=2, settle-gated scores):
1. Scene1 + P2-A → click (32,100) → +1000 (scroll), 0x01BD=2
2. From AfterFind: drive to right edge, hold RIGHT+Y ~80f (panorama)
3. Click Waldo ~ (36,28) → settled total >= 2500 → congrats / next scene

Do **not** re-hold P2-A after the first find before the Waldo click; assist
re-seek blocks useful P1-A scoring on this port.
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
    SCENE1_CLEAR_SCORE,
    SCORE_HI_ADDR,
    SCORE_LO_ADDR,
    WALDO_POINTS,
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
        obs, *_rest = env.step(
            idle_action_multi(players=2)
        )  # type: ignore[attr-defined]
    assert obs is not None
    return np.asarray(obs)


def _drive(env: object, target: CursorTarget, frames: int = 800) -> CursorPose:
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


def _metrics(env: object) -> dict[str, int]:
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    return {
        "score": score_u16(ram[SCORE_LO_ADDR], ram[SCORE_HI_ADDR]),
        "found": int(ram[FOUND_FLAG_ADDR]),
        "x": int(ram[CURSOR_X_ADDR]),
        "y": int(ram[CURSOR_Y_ADDR]),
        "c3": int(ram[0x00C3]),
    }


def _settle(env: object) -> tuple[dict[str, int], bool, np.ndarray]:
    obs = _idle(env, 100)
    samples: list[int] = []
    last = _metrics(env)
    for _ in range(5):
        last = _metrics(env)
        samples.append(last["score"])
        obs = _idle(env, 8)
    return last, len(set(samples)) == 1, obs


def _click_a(env: object, hold: int = 6) -> None:
    for _ in range(hold):
        env.step(buttons_multi(p1=("A",)))  # type: ignore[attr-defined]


def clear_scene1(
    *,
    p2a_frames: int = 300,
    pan_frames: int = 80,
    waldo_x: int = 36,
    waldo_y: int = 28,
    save_states: bool = True,
) -> dict:
    """Run the Scene1 clear recipe; return summary dict."""
    _configure_headless()
    out = RECORDINGS_DIR / "scene1_clear_run"
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "pan_frames": pan_frames,
        "waldo": {"x": waldo_x, "y": waldo_y},
    }

    env = make_env(
        game=GAME,
        state="Scene1",
        game_dir=GAME_DIR,
        render_mode="rgb_array",
        players=2,
    )
    try:
        env.reset()
        _idle(env, 10)
        for _ in range(p2a_frames):
            env.step(buttons_multi(p2=("A",)))
        _idle(env, 4)
        land = _metrics(env)
        _click_a(env)
        first, first_ok, obs = _settle(env)
        Image.fromarray(obs).save(out / "01_after_scroll.png")
        summary["scroll_landing"] = {"x": land["x"], "y": land["y"]}
        summary["after_scroll"] = first
        summary["scroll_stable"] = first_ok
        if not (first_ok and first["score"] >= 1000 and first["found"] == 2):
            raise RuntimeError(f"scroll find failed: {first}")

        if save_states:
            summary["after_find_state"] = str(
                save_state(env, GAME_DIR, GAME, "Scene1_AfterFind1000")
            )

        # Panorama pan — required before Waldo is clickable.
        _drive(env, CursorTarget(x=240, y=100, deadzone=3))
        for _ in range(pan_frames):
            env.step(buttons_multi(p1=("RIGHT", "Y")))
        _idle(env, 6)
        Image.fromarray(_idle(env, 2)).save(out / "02_after_pan.png")
        summary["after_pan"] = _metrics(env)

        _drive(env, CursorTarget(x=waldo_x, y=waldo_y, deadzone=2))
        before_waldo = _metrics(env)
        _click_a(env)
        after, waldo_ok, obs = _settle(env)
        Image.fromarray(obs).save(out / "03_after_waldo.png")
        delta = after["score"] - before_waldo["score"]
        summary["before_waldo"] = before_waldo
        summary["after_waldo"] = after
        summary["waldo_delta"] = delta
        summary["waldo_stable"] = waldo_ok
        summary["cleared"] = bool(
            waldo_ok
            and (
                after["score"] >= SCENE1_CLEAR_SCORE
                or delta >= WALDO_POINTS
            )
        )
        if summary["cleared"] and save_states:
            summary["cleared_state"] = str(
                save_state(env, GAME_DIR, GAME, "Scene1_Cleared")
            )
            Image.fromarray(obs).save(out / "SUCCESS.png")
    finally:
        env.close()

    path = out / "report.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2a-frames", type=int, default=300)
    parser.add_argument("--pan-frames", type=int, default=80)
    parser.add_argument("--waldo-x", type=int, default=36)
    parser.add_argument("--waldo-y", type=int, default=28)
    parser.add_argument("--no-save", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI for Scene1 clear recipe."""
    args = _build_parser().parse_args(argv)
    summary = clear_scene1(
        p2a_frames=args.p2a_frames,
        pan_frames=args.pan_frames,
        waldo_x=args.waldo_x,
        waldo_y=args.waldo_y,
        save_states=not args.no_save,
    )
    return 0 if summary.get("cleared") else 1


if __name__ == "__main__":
    raise SystemExit(main())

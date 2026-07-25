"""Clear Scene5 (Land of Waldos): P2-A scroll → P2-A Waldo → ending.

Recipe (players=2, long settle-gated scores, from Scene5.state HUD):
1. Hold P2-A ~300f (lands ~(32, 100)) → P1-A → +1000/+3000
   (~11650–13650), found=2
2. Hold P2-A ~500f (cursor ~(204, 100)) → P1-A at ~(180, 60)
   → settled total ≥15150 (often ~18575–18725) → five-scrolls ending

Scene5 is the final search scene (five scrolls). Scene5.state must be
Land of Waldos + HUD with favorable layout RNG. Rebuild from
Scene4_Cleared: idle ~5f, then ~7× (A hold 6 + idle 60). Soft layouts
send P2-A to ~(206,100) instead of left scroll.
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
    SCENE5_CLEAR_SCORE,
    SCENE5_SCROLL_SCORE,
    SCORE_HI_ADDR,
    SCORE_LO_ADDR,
    WALDO_POINTS,
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


def _drive(env: object, target: CursorTarget, frames: int = 900) -> CursorPose:
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


def _settle(
    env: object,
    *,
    warm: int = 220,
    samples: int = 8,
    gap: int = 12,
) -> tuple[dict[str, int], bool, np.ndarray]:
    """Long settle: Scene5 Waldo bonus animates longer than Scenes 1–4."""
    obs = _idle(env, warm)
    scores: list[int] = []
    last = _metrics(env)
    for _ in range(samples):
        last = _metrics(env)
        scores.append(last["score"])
        obs = _idle(env, gap)
    return last, len(set(scores)) == 1, obs


def _click_a(env: object, hold: int = 6) -> None:
    for _ in range(hold):
        env.step(buttons_multi(p1=("A",)))  # type: ignore[attr-defined]


def clear_scene5(
    *,
    scroll_p2a: int = 300,
    scroll_x: int = 32,
    scroll_y: int = 100,
    waldo_p2a: int = 500,
    waldo_x: int = 180,
    waldo_y: int = 60,
    save_states: bool = True,
    state: str = "Scene5",
) -> dict:
    """Run the Scene5 clear recipe; return summary dict."""
    _configure_headless()
    out = RECORDINGS_DIR / "scene5_clear_run"
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "state": state,
        "scroll_p2a": scroll_p2a,
        "waldo_p2a": waldo_p2a,
        "scroll": {"x": scroll_x, "y": scroll_y},
        "waldo": {"x": waldo_x, "y": waldo_y},
    }

    env = make_env(
        game=GAME,
        state=state,
        game_dir=GAME_DIR,
        render_mode="rgb_array",
        players=2,
    )
    try:
        env.reset()
        _idle(env, 10)
        summary["load"] = _metrics(env)

        for _ in range(scroll_p2a):
            env.step(buttons_multi(p2=("A",)))  # type: ignore[attr-defined]
        _idle(env, 6)
        land = _metrics(env)
        summary["scroll_land"] = {"x": land["x"], "y": land["y"]}
        _drive(env, CursorTarget(x=scroll_x, y=scroll_y, deadzone=2))
        _click_a(env)
        first, first_ok, obs = _settle(env, warm=120)
        Image.fromarray(obs).save(out / "01_after_scroll.png")
        summary["after_scroll"] = first
        summary["scroll_stable"] = first_ok
        if not (
            first_ok
            and first["score"] >= SCENE5_SCROLL_SCORE
            and first["found"] == 2
        ):
            raise RuntimeError(f"scroll find failed: {first}")

        if save_states:
            summary["after_find_state"] = str(
                save_state(env, GAME_DIR, GAME, "Scene5_AfterFind1000")
            )

        for _ in range(waldo_p2a):
            env.step(buttons_multi(p2=("A",)))  # type: ignore[attr-defined]
        _idle(env, 6)
        summary["after_waldo_p2a"] = _metrics(env)
        Image.fromarray(_idle(env, 2)).save(out / "02_after_waldo_p2a.png")

        before_waldo = _metrics(env)
        _drive(env, CursorTarget(x=waldo_x, y=waldo_y, deadzone=2))
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
                after["score"] >= SCENE5_CLEAR_SCORE
                or delta >= WALDO_POINTS
            )
        )
        summary["ending"] = bool(
            summary["cleared"] and after["score"] >= SCENE5_CLEAR_SCORE
        )
        if summary["cleared"] and save_states:
            summary["cleared_state"] = str(
                save_state(env, GAME_DIR, GAME, "Scene5_Cleared")
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
    parser.add_argument("--state", default="Scene5")
    parser.add_argument("--scroll-p2a", type=int, default=300)
    parser.add_argument("--scroll-x", type=int, default=32)
    parser.add_argument("--scroll-y", type=int, default=100)
    parser.add_argument("--waldo-p2a", type=int, default=500)
    parser.add_argument("--waldo-x", type=int, default=180)
    parser.add_argument("--waldo-y", type=int, default=60)
    parser.add_argument("--no-save", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI for Scene5 clear recipe."""
    args = _build_parser().parse_args(argv)
    summary = clear_scene5(
        scroll_p2a=args.scroll_p2a,
        scroll_x=args.scroll_x,
        scroll_y=args.scroll_y,
        waldo_p2a=args.waldo_p2a,
        waldo_x=args.waldo_x,
        waldo_y=args.waldo_y,
        save_states=not args.no_save,
        state=args.state,
    )
    return 0 if summary.get("cleared") else 1


if __name__ == "__main__":
    raise SystemExit(main())

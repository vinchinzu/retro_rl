"""Try advancing Scene1 after the +1000 find via START / Whitebeard.

Hypothesis: Normal mode may clear with one objective (scroll), and START
(or an automatic timer) advances through Whitebeard into the next scene.
Also probes whether a second find is still required before START works.
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


def _metrics(env: object) -> dict:
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    return {
        "score": score_u16(ram[SCORE_LO_ADDR], ram[SCORE_HI_ADDR]),
        "found": int(ram[FOUND_FLAG_ADDR]),
        "x": int(ram[CURSOR_X_ADDR]),
        "y": int(ram[CURSOR_Y_ADDR]),
        "c3": int(ram[0x00C3]),
        "b0": int(ram[0x00B0]),
        "b1": int(ram[0x00B1]),
        "1bc": int(ram[0x01BC]),
        "1bd": int(ram[0x01BD]),
        "1be": int(ram[0x01BE]),
        "1bf": int(ram[0x01BF]),
    }


def _sig(obs: np.ndarray) -> dict:
    top = obs[:180]
    return {
        "mean_rgb": [float(x) for x in top.reshape(-1, 3).mean(axis=0)],
        "std": float(top.std()),
    }


def _pulse(env: object, *names: str, frames: int = 4) -> None:
    for _ in range(frames):
        env.step(buttons_multi(p1=names))  # type: ignore[attr-defined]
    _idle(env, 8)


def run_advance(*, recipe: str) -> dict:
    """Run one advance recipe from Scene1_AfterFind1000."""
    _configure_headless()
    out = RECORDINGS_DIR / "scene_advance"
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(
        game=GAME,
        state="Scene1_AfterFind1000",
        game_dir=GAME_DIR,
        render_mode="rgb_array",
        players=2,
    )
    frames: list[dict] = []
    try:
        obs, _info = env.reset()
        Image.fromarray(obs).save(out / f"{recipe}_00.png")
        frames.append({"t": 0, "m": _metrics(env), "sig": _sig(obs)})

        if recipe == "start_mash":
            # START into Whitebeard, then mash A/START through dialogue.
            _pulse(env, "START", frames=2)
            for i in range(1, 25):
                obs = _idle(env, 20)
                if i % 3 == 0:
                    _pulse(env, "A", frames=2)
                if i % 5 == 0:
                    _pulse(env, "START", frames=2)
                if i % 2 == 0:
                    Image.fromarray(obs).save(out / f"{recipe}_{i:02d}.png")
                    frames.append(
                        {"t": i, "m": _metrics(env), "sig": _sig(obs)}
                    )
        elif recipe == "idle_long":
            for i in range(1, 40):
                obs = _idle(env, 30)
                if i % 2 == 0:
                    Image.fromarray(obs).save(out / f"{recipe}_{i:02d}.png")
                    frames.append(
                        {"t": i, "m": _metrics(env), "sig": _sig(obs)}
                    )
        elif recipe == "start_then_a_hold":
            _pulse(env, "START", frames=2)
            for i in range(1, 30):
                for _ in range(10):
                    env.step(buttons_multi(p1=("A",)))
                obs = _idle(env, 10)
                if i % 2 == 0:
                    Image.fromarray(obs).save(out / f"{recipe}_{i:02d}.png")
                    frames.append(
                        {"t": i, "m": _metrics(env), "sig": _sig(obs)}
                    )
        elif recipe == "second_assist_then_start":
            for _ in range(300):
                env.step(buttons_multi(p2=("A",)))
            _idle(env, 4)
            for _ in range(6):
                env.step(buttons_multi(p1=("A",)))
            obs = _idle(env, 100)
            Image.fromarray(obs).save(out / f"{recipe}_prestart.png")
            frames.append({"t": "pre", "m": _metrics(env), "sig": _sig(obs)})
            _pulse(env, "START", frames=2)
            for i in range(1, 30):
                for _ in range(8):
                    env.step(buttons_multi(p1=("A",)))
                obs = _idle(env, 12)
                if i % 2 == 0:
                    Image.fromarray(obs).save(out / f"{recipe}_{i:02d}.png")
                    frames.append(
                        {"t": i, "m": _metrics(env), "sig": _sig(obs)}
                    )
        else:
            raise ValueError(recipe)

        final = _metrics(env)
        final_obs = _idle(env, 2)
        Image.fromarray(final_obs).save(out / f"{recipe}_final.png")
        # Detect non-carpet / non-black-ish playfield
        mean = np.asarray(final_obs[:180], dtype=np.float32).mean(axis=(0, 1))
        cleared_path = None
        # Heuristic: left carpet-flyers mean blue-ish; cave is darker gray-brown;
        # whitebeard is near-black with face colors.
        if float(mean.std()) > 15 and float(mean.mean()) > 30:
            # Likely a play scene (not black Whitebeard).
            if abs(float(mean[2] - mean[0])) < 40:  # not strongly blue sky
                cleared_path = str(
                    save_state(env, GAME_DIR, GAME, "Scene1_Cleared")
                )
                print(f"[adv] saved Scene1_Cleared -> {cleared_path}")

        report = {
            "recipe": recipe,
            "final": final,
            "final_mean_rgb": [float(x) for x in mean],
            "cleared_state": cleared_path,
            "frames": frames,
        }
        path = out / f"{recipe}_report.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[adv] {recipe} final={final} mean={mean} -> {path}")
        return report
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    """CLI for scene-advance experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        default="all",
        choices=[
            "all",
            "start_mash",
            "idle_long",
            "start_then_a_hold",
            "second_assist_then_start",
        ],
    )
    args = parser.parse_args(argv)
    recipes = (
        [
            "start_mash",
            "start_then_a_hold",
            "second_assist_then_start",
            "idle_long",
        ]
        if args.recipe == "all"
        else [args.recipe]
    )
    for recipe in recipes:
        run_advance(recipe=recipe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

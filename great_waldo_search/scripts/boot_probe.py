"""Headless boot / menu probe for The Great Waldo Search.

Boots from NONE, advances title → difficulty → first search scene using a
TAS-aligned START window, dumps screenshots under recordings/, and writes a
development Scene1.state once a search HUD is detected.
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

from great_waldo_search.paths import GAME, GAME_DIR, INTEGRATION_DIR, RECORDINGS_DIR
from retro_harness.env import make_env, save_state
from snes_oneshot.actions import buttons, idle_action


def _configure_headless() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("SDL_SOFTWARE_RENDERER", "1")


def _save_png(obs: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(obs).save(path)


def _mean_rgb(obs: np.ndarray) -> tuple[float, float, float]:
    return (
        float(np.mean(obs[:, :, 0])),
        float(np.mean(obs[:, :, 1])),
        float(np.mean(obs[:, :, 2])),
    )


def _rgb_std(obs: np.ndarray) -> float:
    return float(np.std(obs.astype(np.float32)))


def _black_fraction(obs: np.ndarray) -> float:
    return float(
        np.mean(
            (obs[:, :, 0] < 8) & (obs[:, :, 1] < 8) & (obs[:, :, 2] < 8)
        )
    )


def looks_like_search_scene(obs: np.ndarray) -> bool:
    """Detect a search illustration with bottom HUD (not logo/menu orbs)."""
    std = _rgb_std(obs)
    black = _black_fraction(obs)
    hud = obs[-32:, :, :]
    hud_mean = float(np.mean(hud))
    # Search scenes are busy, mostly non-black, with a mid-gray status bar.
    return black < 0.2 and std > 50.0 and 50.0 < hud_mean < 170.0


def build_boot_script() -> list[list[int]]:
    """Frame script: logos → title START → NORMAL → START into Scene1.

    Title START must land in ~540–599 (TAS window). Confirming NORMAL with A,
    then pulsing START, reaches the first Flying Carpets search.
    """
    frames: list[list[int]] = []
    frames.extend([idle_action()] * 560)
    frames.extend([buttons("START")] * 6)
    frames.extend([idle_action()] * 220)
    frames.extend([buttons("A")] * 4)
    frames.extend([idle_action()] * 120)
    # Pulse START a few times with gaps to clear previews into search.
    for _ in range(4):
        frames.extend([buttons("START")] * 8)
        frames.extend([idle_action()] * 80)
    frames.extend([idle_action()] * 60)
    return frames


def run_boot_probe(
    *,
    frames: int = 1400,
    save_scene1: bool = True,
    out_dir: Path | None = None,
) -> dict:
    """Boot Waldo headlessly and dump recordings + optional Scene1.state."""
    _configure_headless()
    out = out_dir or RECORDINGS_DIR
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(
        game=GAME,
        state="NONE",
        game_dir=GAME_DIR,
        render_mode="rgb_array",
    )
    script = build_boot_script()
    total = max(frames, len(script) + 1)
    checkpoints = {0, 120, 240, 480, 560, 700, 800, 1000, 1200, total - 1}
    snapshots: list[dict] = []
    scene1_path: str | None = None
    title_path: str | None = None

    try:
        obs, info = env.reset()
        _save_png(obs, out / "boot_0000.png")
        save_state(env, GAME_DIR, GAME, "Boot")

        for frame_i in range(total):
            if frame_i < len(script):
                action = script[frame_i]
            else:
                action = idle_action()

            obs, _reward, terminated, truncated, info = env.step(action)

            if frame_i in checkpoints:
                name = f"boot_{frame_i:04d}"
                _save_png(obs, out / f"{name}.png")
                r, g, b = _mean_rgb(obs)
                snapshots.append(
                    {
                        "frame": frame_i,
                        "mean_rgb": [r, g, b],
                        "rgb_std": _rgb_std(obs),
                        "black_frac": _black_fraction(obs),
                        "search_like": looks_like_search_scene(obs),
                        "info_keys": sorted(str(k) for k in info.keys()),
                    }
                )

            # Title: busy yellow Waldo wallpaper before menus (no HUD).
            if (
                title_path is None
                and 480 <= frame_i <= 600
                and _rgb_std(obs) > 80
                and _black_fraction(obs) < 0.05
            ):
                title_path = str(save_state(env, GAME_DIR, GAME, "Title"))
                _save_png(obs, out / "title.png")

            if (
                save_scene1
                and scene1_path is None
                and frame_i > 900
                and looks_like_search_scene(obs)
            ):
                path = save_state(env, GAME_DIR, GAME, "Scene1")
                scene1_path = str(path)
                _save_png(obs, out / "scene1_candidate.png")

            if terminated or truncated:
                obs, info = env.reset()

        report = {
            "frames": total,
            "script_len": len(script),
            "title_state": title_path,
            "scene1_state": scene1_path,
            "integration_dir": str(INTEGRATION_DIR),
            "recordings": str(out),
            "snapshots": snapshots,
        }
        report_path = out / "boot_probe_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[boot_probe] report={report_path}")
        print(f"[boot_probe] title={title_path}")
        print(f"[boot_probe] scene1={scene1_path}")
        return report
    finally:
        env.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=1400)
    parser.add_argument("--no-scene1", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry for the headless boot probe."""
    args = _build_parser().parse_args(argv)
    run_boot_probe(frames=args.frames, save_scene1=not args.no_scene1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

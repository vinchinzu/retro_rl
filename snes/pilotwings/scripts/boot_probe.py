"""Boot Pilotwings from reset and save an airborne Lesson 1 checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pilotwings.menus import boot_to_lesson1_plane_script
from pilotwings.paths import GAME, GAME_DIR, RECORDINGS_DIR
from pilotwings.ram import parse_game_state
from retro_harness.env import make_env, save_state
from retro_harness.ram_state import GameMode
from retro_harness.segment_runner import configure_headless, save_rgb_png


def run_probe(*, save_lesson: bool = True) -> int:
    """Reach the first light-plane lesson, verify flight, and save it."""
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        frame = 0
        for scripted in boot_to_lesson1_plane_script():
            obs, *_ = env.step(scripted.action)
            frame += 1

        state = parse_game_state(env.get_ram(), frame=frame)
        png = save_rgb_png(obs, RECORDINGS_DIR / "boot_lesson1_plane.png")
        print(
            f"LESSON1_PLANE frame={frame} mode={state.mode.name} "
            f"altitude={state.extras['altitude']} screenshot={png}"
        )
        if save_lesson:
            path = save_state(env, GAME_DIR, GAME, "Lesson1Plane")
            print(f"saved {path}")
        return 0 if state.mode is GameMode.PLAYING else 1
    finally:
        env.close()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run_probe(save_lesson=not args.no_save))


if __name__ == "__main__":
    main()

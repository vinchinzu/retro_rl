"""Boot The Magical Quest from reset and save Stage1.state."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from magical_quest.menus import boot_to_stage1_script
from magical_quest.paths import GAME, GAME_DIR, RECORDINGS_DIR
from magical_quest.ram import parse_game_state
from retro_harness.env import make_env, save_state
from retro_harness.ram_state import GameMode
from retro_harness.segment_runner import configure_headless, save_rgb_png


def run_probe(*, save_stage1: bool = True) -> int:
    """Reach the first controllable room and optionally save it."""
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        frame = 0
        for scripted in boot_to_stage1_script():
            obs, *_ = env.step(scripted.action)
            frame += 1
        state = parse_game_state(env.get_ram(), frame=frame)
        png = save_rgb_png(obs, RECORDINGS_DIR / "boot_stage1.png")
        print(
            f"LEVEL_READY frame={frame} mode={state.mode.name} "
            f"x={state.player_x} progress={state.camera_x} screenshot={png}"
        )
        if state.mode is not GameMode.PLAYING:
            return 1
        if save_stage1:
            path = save_state(env, GAME_DIR, GAME, "Stage1")
            print(f"saved {path}")
        return 0
    finally:
        env.close()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run_probe(save_stage1=not args.no_save))


if __name__ == "__main__":
    main()


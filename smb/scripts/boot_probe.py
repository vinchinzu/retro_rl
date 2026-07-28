"""Boot Super Mario Bros. (NES) from reset and save a controllable Level1 state."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from smb.menus import boot_to_level1_script
from smb.paths import GAME, GAME_DIR, RECORDINGS_DIR
from smb.ram import is_level1_ready, parse_game_state
from retro_harness.env import make_env, save_state
from retro_harness.nes import nes_action
from snes_oneshot.game_state import GameMode
from snes_oneshot.segment_runner import configure_headless, save_rgb_png


def run_probe(*, save_level1: bool = True, walk_frames: int = 40) -> int:
    """Reach Level 1, verify readiness, optionally walk, and save checkpoint."""
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        frame = 0
        for scripted in boot_to_level1_script():
            obs, *_ = env.step(scripted.action)
            frame += 1

            if is_level1_ready(env.get_ram(), obs_mean=float(obs.mean())):
                break

        for _ in range(walk_frames):
            obs, *_ = env.step(nes_action("RIGHT"))
            frame += 1

        state = parse_game_state(env.get_ram(), frame=frame)
        ready = is_level1_ready(env.get_ram(), obs_mean=float(obs.mean()))
        png = save_rgb_png(obs, RECORDINGS_DIR / "boot_level1.png")
        print(
            f"LEVEL1 frame={frame} mode={state.mode.name} ready={ready} "
            f"mean={float(obs.mean()):.1f} screenshot={png}"
        )
        if save_level1 and ready:
            path = save_state(env, GAME_DIR, GAME, "Level1")
            print(f"saved {path}")
        return 0 if ready and state.mode is GameMode.PLAYING else 1
    finally:
        env.close()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--walk-frames", type=int, default=40)
    args = parser.parse_args()
    raise SystemExit(run_probe(save_level1=not args.no_save, walk_frames=args.walk_frames))


if __name__ == "__main__":
    main()

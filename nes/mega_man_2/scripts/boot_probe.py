"""Boot Mega Man 2 (NES) from reset and save a controllable Level1 state."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mega_man_2.menus import boot_to_level1_script
from mega_man_2.paths import GAME, GAME_DIR, RECORDINGS_DIR
from mega_man_2.ram import is_level1_ready, parse_game_state
from retro_harness.env import make_env, save_state
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png

STABLE_FRAMES = 20
MIN_FRAME = 400
POST_READY_WAIT = 150


def run_probe(*, save_level1: bool = True, walk_frames: int = 40) -> int:
    """Reach Air Man stage, verify readiness past READY, save checkpoint."""
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        frame = 0
        stable = 0
        for scripted in boot_to_level1_script():
            obs, *_ = env.step(scripted.action)
            frame += 1
            mean = float(obs.mean())
            if frame >= MIN_FRAME and is_level1_ready(env.get_ram(), obs_mean=mean):
                stable += 1
            else:
                stable = 0
            if stable >= STABLE_FRAMES:
                break
        else:
            png = save_rgb_png(obs, RECORDINGS_DIR / "boot_level1.png")
            print(f"LEVEL1 frame={frame} ready=False mean={float(obs.mean()):.1f} screenshot={png}")
            return 1

        # Wait through READY card so Mega Man is actually controllable.
        for _ in range(POST_READY_WAIT):
            obs, *_ = env.step(nes_idle_action())
            frame += 1

        before = env.get_ram().copy()
        for _ in range(walk_frames):
            obs, *_ = env.step(nes_action("RIGHT"))
            frame += 1
        after = env.get_ram()
        changed = int((before != after).sum())
        mean = float(obs.mean())
        ready = is_level1_ready(after, obs_mean=mean) and changed >= 3
        state = parse_game_state(after, frame=frame, obs_mean=mean)
        png = save_rgb_png(obs, RECORDINGS_DIR / "boot_level1.png")
        print(
            f"LEVEL1 frame={frame} mode={state.mode.name} ready={ready} "
            f"changed={changed} mean={mean:.1f} screenshot={png}"
        )
        if save_level1 and ready:
            path = save_state(env, GAME_DIR, GAME, "Level1")
            print(f"saved {path}")
        return 0 if ready else 1
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

"""Boot Rival Turf! from reset and save a fight-ready Stage1.state."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retro_harness.env import make_env, save_state
from rival_turf.menus import boot_to_stage1_script
from rival_turf.paths import GAME, GAME_DIR, RECORDINGS_DIR
from rival_turf.ram import parse_game_state
from snes_oneshot.actions import buttons
from snes_oneshot.game_state import GameMode
from snes_oneshot.segment_runner import configure_headless, save_rgb_png


def run_probe(*, approach_frames: int = 360, save_stage1: bool = True) -> int:
    """Reach Stage 1, approach its opening combat lock, and save it."""
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        frame = 0
        for scripted in boot_to_stage1_script():
            obs, *_ = env.step(scripted.action)
            frame += 1

        stage_open = parse_game_state(env.get_ram(), frame=frame)
        if stage_open.mode is not GameMode.PLAYING:
            print(
                "boot script did not reach active Stage 1: "
                f"run_state={stage_open.extras['run_state']} "
                f"active={stage_open.extras['player_active']} "
                f"pos=({stage_open.player_x},{stage_open.player_y})"
            )
            return 1

        for _ in range(approach_frames):
            obs, *_ = env.step(buttons("RIGHT"))
            frame += 1

        state = parse_game_state(env.get_ram(), frame=frame)
        png = save_rgb_png(obs, RECORDINGS_DIR / "boot_stage1.png")
        print(
            f"FIGHT_READY frame={frame} mode={state.mode.name} "
            f"pos=({state.player_x},{state.player_y}) screenshot={png}"
        )
        if save_stage1:
            path = save_state(env, GAME_DIR, GAME, "Stage1")
            print(f"saved {path}")
        return 0 if state.mode is GameMode.PLAYING else 1
    finally:
        env.close()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approach-frames", type=int, default=360)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        run_probe(
            approach_frames=args.approach_frames,
            save_stage1=not args.no_save,
        )
    )


if __name__ == "__main__":
    main()


"""Headless boot probe: reach Mission 1 and save fight-ready Stage1.state."""

from __future__ import annotations

import argparse

from retro_harness.env import make_env, save_state
from retro_harness.actions import idle_action
from retro_harness.ram_state import GameMode
from retro_harness.segment_runner import configure_headless, save_rgb_png
from super_double_dragon.menus import boot_to_stage1_script
from super_double_dragon.paths import GAME, GAME_DIR, RECORDINGS_DIR
from super_double_dragon.ram import parse_game_state

def run_probe(*, max_frames: int = 3000, save_stage1: bool = True) -> int:
    """Drive the default 1P menu and optionally save live Mission 1."""
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        env.reset()
        script = iter(boot_to_stage1_script())
        entered_at: int | None = None
        for frame in range(max_frames):
            state = parse_game_state(env.get_ram(), frame=frame)
            if state.mode is GameMode.PLAYING and entered_at is None:
                entered_at = frame
                print(f"ENTER_MISSION_1 frame={frame}")
            if entered_at is not None and state.living_enemies:
                path = None
                if save_stage1:
                    path = save_state(env, GAME_DIR, GAME, "Stage1")
                png = save_rgb_png(
                    env.render(), RECORDINGS_DIR / "boot_stage1.png"
                )
                print(
                    f"FIGHT_READY frame={frame} hp={state.health} "
                    f"lives={state.lives} enemies={len(state.living_enemies)}"
                )
                if path is not None:
                    print(f"saved {path}")
                print(f"screenshot {png}")
                return 0
            if entered_at is not None:
                action = idle_action()
            else:
                try:
                    action = next(script).action
                except StopIteration:
                    action = idle_action()
            env.step(action)
        print("probe finished without fight-ready Mission 1")
        return 1
    finally:
        env.close()

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-frames", type=int, default=3000)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        run_probe(max_frames=args.max_frames, save_stage1=not args.no_save)
    )

if __name__ == "__main__":
    main()

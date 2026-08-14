"""Replay controlled Stage 1 inputs and print confirmed player deltas."""

from __future__ import annotations

import argparse

from retro_harness.env import get_available_states, make_env
from rival_turf.paths import GAME, GAME_DIR, STAGE1_STATE
from rival_turf.ram import parse_game_state
from retro_harness.actions import buttons
from retro_harness.segment_runner import configure_headless

def run_probe(*, frames: int = 30) -> int:
    """Print coordinate deltas for four directional probes."""
    configure_headless()
    if STAGE1_STATE not in get_available_states(GAME, GAME_DIR):
        print("Stage1.state is missing; run scripts/boot_probe.py first")
        return 1
    for name in ("RIGHT", "LEFT", "UP", "DOWN", "Y"):
        env = make_env(GAME, STAGE1_STATE, GAME_DIR, render_mode="rgb_array")
        try:
            env.reset()
            before = parse_game_state(env.get_ram())
            for _ in range(frames):
                env.step(buttons(name))
            after = parse_game_state(env.get_ram(), frame=frames)
            print(
                f"{name:>5}: ({before.player_x},{before.player_y}) -> "
                f"({after.player_x},{after.player_y}) "
                f"mode={after.mode.name}"
            )
        finally:
            env.close()
    return 0

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=30)
    args = parser.parse_args()
    raise SystemExit(run_probe(frames=args.frames))

if __name__ == "__main__":
    main()


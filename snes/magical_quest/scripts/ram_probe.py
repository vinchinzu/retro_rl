"""Replay controlled Stage 1 movement and print X/progress deltas."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from magical_quest.paths import GAME, GAME_DIR, STAGE1_STATE
from magical_quest.ram import parse_game_state
from retro_harness.env import get_available_states, make_env
from retro_harness.actions import buttons, idle_action
from retro_harness.segment_runner import configure_headless


def run_probe(*, warmup: int = 180, frames: int = 60) -> int:
    """Print LEFT/RIGHT and jump state responses."""
    configure_headless()
    if STAGE1_STATE not in get_available_states(GAME, GAME_DIR):
        print("Stage1.state is missing; run scripts/boot_probe.py first")
        return 1
    for label, keys in (("RIGHT", ("RIGHT",)), ("LEFT", ("LEFT",)), ("JUMP", ("B",))):
        env = make_env(GAME, STAGE1_STATE, GAME_DIR, render_mode="rgb_array")
        try:
            env.reset()
            for _ in range(warmup):
                env.step(idle_action())
            before = parse_game_state(env.get_ram())
            for _ in range(frames):
                env.step(buttons(*keys))
            after = parse_game_state(env.get_ram(), frame=frames)
            print(
                f"{label:>5}: x={before.player_x}->{after.player_x} "
                f"progress={before.camera_x}->{after.camera_x}"
            )
        finally:
            env.close()
    return 0


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=180)
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()
    raise SystemExit(run_probe(warmup=args.warmup, frames=args.frames))


if __name__ == "__main__":
    main()


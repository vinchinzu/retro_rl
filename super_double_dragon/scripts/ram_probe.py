"""Print the actor pool before and after controlled Mission 1 actions."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retro_harness.env import get_available_states, make_env
from snes_oneshot.actions import buttons
from snes_oneshot.segment_runner import configure_headless
from super_double_dragon.paths import GAME, GAME_DIR, STAGE1_STATE
from super_double_dragon.ram import parse_game_state


def run_probe(*, frames: int = 30) -> int:
    """Replay short directional probes and print normalized state deltas."""
    configure_headless()
    available = get_available_states(GAME, GAME_DIR)
    chosen = STAGE1_STATE if STAGE1_STATE in available else "NONE"
    for name in ("RIGHT", "LEFT", "UP", "DOWN", "Y", "A"):
        env = make_env(GAME, chosen, GAME_DIR, render_mode="rgb_array")
        try:
            env.reset()
            before = parse_game_state(env.get_ram())
            for _ in range(frames):
                env.step(buttons(name))
            after = parse_game_state(env.get_ram(), frame=frames)
            print(
                f"{name:>5}: p=({before.player_x},{before.player_y})"
                f"->({after.player_x},{after.player_y}) "
                f"hp={before.health}->{after.health} "
                "enemies="
                f"{[(e.slot, e.x, e.y, e.health) for e in after.living_enemies]}"
            )
        finally:
            env.close()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=30)
    args = parser.parse_args()
    raise SystemExit(run_probe(frames=args.frames))


if __name__ == "__main__":
    main()

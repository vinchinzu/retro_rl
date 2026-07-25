"""Probe horizontal movement from the Joe & Mac Stage 1 checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from joe_and_mac.paths import GAME, GAME_DIR, STAGE1_STATE
from joe_and_mac.ram import parse_game_state
from retro_harness.env import get_available_states, make_env
from snes_oneshot.actions import buttons, idle_action
from snes_oneshot.segment_runner import configure_headless


def _run(keys: tuple[str, ...], *, frames: int) -> tuple[int, int, int]:
    env = make_env(GAME, STAGE1_STATE, GAME_DIR, render_mode="rgb_array")
    try:
        env.reset()
        before = parse_game_state(env.get_ram())
        action = buttons(*keys) if keys else idle_action()
        for _ in range(frames):
            env.step(action)
        after = parse_game_state(env.get_ram(), frame=frames)
        return (
            int(before.extras["horizontal_progress"]),
            int(after.extras["horizontal_progress"]),
            int(after.extras["actor_state"]),
        )
    finally:
        env.close()


def run_probe(*, frames: int = 180) -> int:
    """Print idle/left/right progress responses."""
    configure_headless()
    if STAGE1_STATE not in get_available_states(GAME, GAME_DIR):
        print("Stage1.state is missing; run scripts/boot_probe.py first")
        return 1
    for label, keys in (
        ("IDLE", ()),
        ("LEFT", ("LEFT",)),
        ("RIGHT", ("RIGHT",)),
        ("RIGHT+B", ("RIGHT", "B")),
    ):
        progress0, progress1, actor = _run(keys, frames=frames)
        print(
            f"{label:>7}: progress={progress0}->{progress1} "
            f"actor_state={actor}"
        )
    return 0


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=180)
    args = parser.parse_args()
    raise SystemExit(run_probe(frames=args.frames))


if __name__ == "__main__":
    main()

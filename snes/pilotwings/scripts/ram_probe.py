"""Probe pitch and heading controls from the Lesson 1 plane checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pilotwings.paths import GAME, GAME_DIR, LESSON1_PLANE_STATE
from pilotwings.ram import parse_game_state
from retro_harness.env import get_available_states, make_env
from retro_harness.actions import buttons, idle_action
from retro_harness.segment_runner import configure_headless


def _run(keys: tuple[str, ...], *, frames: int) -> tuple[int, int, int, int]:
    env = make_env(GAME, LESSON1_PLANE_STATE, GAME_DIR, render_mode="rgb_array")
    try:
        env.reset()
        before = parse_game_state(env.get_ram())
        action = buttons(*keys) if keys else idle_action()
        for _ in range(frames):
            env.step(action)
        after = parse_game_state(env.get_ram(), frame=frames)
        return (
            int(before.extras["altitude"]),
            int(after.extras["altitude"]),
            int(after.extras["pitch_control"]),
            int(after.extras["heading_raw"]),
        )
    finally:
        env.close()


def run_probe(*, frames: int = 90) -> int:
    """Print altitude, pitch, and turn response for held directions."""
    configure_headless()
    if LESSON1_PLANE_STATE not in get_available_states(GAME, GAME_DIR):
        print("Lesson1Plane.state is missing; run scripts/boot_probe.py first")
        return 1
    for label, keys in (
        ("IDLE", ()),
        ("LEFT", ("LEFT",)),
        ("RIGHT", ("RIGHT",)),
        ("UP", ("UP",)),
        ("DOWN", ("DOWN",)),
    ):
        altitude0, altitude1, pitch, heading = _run(keys, frames=frames)
        print(
            f"{label:>5}: altitude={altitude0}->{altitude1} "
            f"pitch={pitch} heading_raw={heading}"
        )
    return 0


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=90)
    args = parser.parse_args()
    raise SystemExit(run_probe(frames=args.frames))


if __name__ == "__main__":
    main()

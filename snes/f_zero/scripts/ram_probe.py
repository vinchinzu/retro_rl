"""Probe acceleration and steering fields from the Mute City checkpoint."""

from __future__ import annotations

import argparse

from f_zero.paths import GAME, GAME_DIR, MUTE_CITY_STATE
from f_zero.ram import parse_game_state
from retro_harness.env import get_available_states, make_env
from retro_harness.actions import buttons
from retro_harness.segment_runner import configure_headless

def _run(keys: tuple[str, ...], *, warmup: int, frames: int) -> tuple[int, int, int]:
    env = make_env(GAME, MUTE_CITY_STATE, GAME_DIR, render_mode="rgb_array")
    try:
        env.reset()
        for _ in range(warmup):
            env.step(buttons("B"))
        before = parse_game_state(env.get_ram())
        for _ in range(frames):
            env.step(buttons(*keys))
        after = parse_game_state(env.get_ram(), frame=frames)
        return (
            int(before.extras["speed_raw"]),
            int(after.extras["speed_raw"]),
            int(after.extras["lateral"]),
        )
    finally:
        env.close()

def run_probe(*, warmup: int = 240, frames: int = 45) -> int:
    """Print straight/left/right speed and lateral responses."""
    configure_headless()
    if MUTE_CITY_STATE not in get_available_states(GAME, GAME_DIR):
        print("MuteCity.state is missing; run scripts/boot_probe.py first")
        return 1
    for label, keys in (
        ("STRAIGHT", ("B",)),
        ("LEFT", ("B", "LEFT")),
        ("RIGHT", ("B", "RIGHT")),
    ):
        speed0, speed1, lateral = _run(keys, warmup=warmup, frames=frames)
        print(
            f"{label:>8}: speed_raw={speed0}->{speed1} "
            f"lateral={lateral}"
        )
    return 0

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=240)
    parser.add_argument("--frames", type=int, default=45)
    args = parser.parse_args()
    raise SystemExit(run_probe(warmup=args.warmup, frames=args.frames))

if __name__ == "__main__":
    main()


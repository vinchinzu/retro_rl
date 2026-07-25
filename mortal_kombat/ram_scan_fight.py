#!/usr/bin/env python3
"""
RAM discovery helper for in-fight MK1 states.

Diffs get_ram() while walking, blocking, and jumping to find position proxies
and sub-pixel state candidates. Complements cheat_extractor.py --scan (match
counter between stages).

Usage:
    cd mortal_kombat
    uv run python ram_scan_fight.py --state Fight_LiuKang
    uv run python ram_scan_fight.py --state Practice_MortalKombat --players 2
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import stable_retro as retro

from fighters_common.game_configs import get_game_config

_B, _Y, _SELECT, _START, _UP, _DOWN, _LEFT, _RIGHT, _A, _X, _L, _R = range(12)


@dataclass(frozen=True)
class InputPhase:
    """Named controller input held for a scan segment."""

    name: str
    buttons: dict[int, int]
    frames: int = 60


def _buttons_array(buttons: dict[int, int], players: int) -> np.ndarray:
    size = 12 * players
    arr = np.zeros(size, dtype=np.int8)
    for btn, val in buttons.items():
        arr[btn] = val
    return arr


def _settle(env, frames: int = 60, players: int = 1) -> None:
    noop = _buttons_array({}, players)
    for _ in range(frames):
        env.step(noop)


def _run_phase(env, phase: InputPhase, players: int) -> tuple[np.ndarray, np.ndarray]:
    before = env.unwrapped.get_ram().copy()
    action = _buttons_array(phase.buttons, players)
    for _ in range(phase.frames):
        env.step(action)
    after = env.unwrapped.get_ram().copy()
    return before, after


def _diff_summary(
    before: np.ndarray,
    after: np.ndarray,
    *,
    min_delta: int = 3,
    top_n: int = 15,
) -> list[tuple[int, int, int, int]]:
    rows: list[tuple[int, int, int, int]] = []
    for addr in range(len(before)):
        delta = int(after[addr]) - int(before[addr])
        if abs(delta) >= min_delta:
            rows.append((addr, int(before[addr]), int(after[addr]), delta))
    rows.sort(key=lambda row: abs(row[3]), reverse=True)
    return rows[:top_n]


def _print_phase(name: str, rows: list[tuple[int, int, int, int]]) -> None:
    print(f"\n{name}:")
    if not rows:
        print("  (no significant deltas)")
        return
    for addr, old, new, delta in rows:
        print(f"  0x{addr:04X} ({addr:5d}): {old:3d} -> {new:3d} (delta {delta:+4d})")


def _read_known(info_keys: dict[str, int], ram: np.ndarray) -> dict[str, int]:
    return {key: int(ram[addr]) for key, addr in info_keys.items() if addr < len(ram)}


def scan_fight(state: str, players: int = 1) -> None:
    """Run movement/block/jump RAM diffs on a fight save state."""
    config = get_game_config("mk1")
    game_dir = ROOT_DIR / config.game_dir_name
    retro.data.Integrations.add_custom_path(str(game_dir / "custom_integrations"))

    retro_kwargs = dict(
        game=config.game_id,
        state=state,
        render_mode="rgb_array",
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        use_restricted_actions=retro.Actions.ALL,
    )
    if players == 2:
        retro_kwargs["players"] = 2

    env = retro.make(**retro_kwargs)
    env.reset()
    _settle(env, frames=90, players=players)

    known_addrs = {
        "health": 1209,
        "enemy_health": 1211,
        "timer": 290,
        "p1_character": 6514,
        "p1_rounds": 6510,
        "p2_rounds": 1207,
        "p2_character": 36,
        "match_counter": 10,
    }
    baseline = env.unwrapped.get_ram()
    print("Known RAM snapshot:")
    for key, val in _read_known(known_addrs, baseline).items():
        print(f"  {key}: {val}")

    phases = [
        InputPhase("walk_right", {_RIGHT: 1}),
        InputPhase("walk_left", {_LEFT: 1}),
        InputPhase("jump", {_UP: 1}, frames=30),
        InputPhase("crouch", {_DOWN: 1}, frames=30),
        InputPhase("block", {_X: 1}, frames=45),
    ]

    bidirectional_x: set[int] = set()
    right_rows = _diff_summary(*_run_phase(env, phases[0], players))
    _print_phase("Walk right candidates", right_rows)
    left_rows = _diff_summary(*_run_phase(env, phases[1], players))
    _print_phase("Walk left candidates", left_rows)

    right_addrs = {row[0] for row in right_rows}
    for addr, _, _, delta in left_rows:
        if addr in right_addrs and delta < 0:
            bidirectional_x.add(addr)

    if bidirectional_x:
        print("\nBidirectional X proxies (right +, left -):")
        for addr in sorted(bidirectional_x):
            print(f"  0x{addr:04X} ({addr})")

    for phase in phases[2:]:
        rows = _diff_summary(*_run_phase(env, phase, players))
        _print_phase(phase.name, rows)

    env.close()
    print(
        "\nNext: validate candidates across opponents/states; add confirmed "
        "addresses to data.json + MK1_RAM_FEATURES."
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Scan MK1 fight RAM for new features")
    parser.add_argument(
        "--state",
        default="Fight_LiuKang",
        help="Save state to load (default: Fight_LiuKang)",
    )
    parser.add_argument(
        "--players",
        type=int,
        default=1,
        help="Player count (2 for Practice_* states)",
    )
    args = parser.parse_args()
    scan_fight(args.state, players=args.players)


if __name__ == "__main__":
    main()

"""Differential RAM probe: walk/attack and print known + candidate addrs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from retro_harness.env import get_available_states, make_env
from snes_oneshot.actions import buttons
from snes_oneshot.ram_diff import (
    candidates_decreasing,
    candidates_increasing,
    diff_changed,
    snapshot,
)
from snes_oneshot.segment_runner import configure_headless
from tmnt_iv.paths import GAME, GAME_DIR, STAGE1_STATE
from tmnt_iv.ram import (
    ADDR_CAMERA_X,
    ADDR_MENU,
    ENEMY_BASES,
    OFF_HP,
    OFF_X,
    OFF_Y,
    PLAYER_BASE,
    parse_game_state,
    read_u8,
    read_u16le,
)


def _print_known(ram: np.ndarray) -> None:
    state = parse_game_state(ram)
    print(
        f"menu=0x{read_u8(ram, ADDR_MENU):02X} "
        f"mode={state.mode.name} "
        f"player=({state.player_x},{state.player_y}) "
        f"hp={state.health} lives={state.lives} "
        f"cam={state.camera_x}"
    )
    for i, base in enumerate(ENEMY_BASES):
        print(
            f"  enemy{i}: xy=({read_u16le(ram, base + OFF_X)},"
            f"{read_u16le(ram, base + OFF_Y)}) "
            f"hp={read_u8(ram, base + OFF_HP)}"
        )


def run_probe(*, frames_walk: int = 45, frames_attack: int = 30) -> int:
    """Load Stage1 when available; sample walk/attack differentials."""
    configure_headless()
    states = get_available_states(GAME, GAME_DIR)
    start = STAGE1_STATE if STAGE1_STATE in states else "NONE"
    env = make_env(GAME, start, GAME_DIR, render_mode="rgb_array")
    try:
        env.reset()
        before = snapshot(env.get_ram())
        print(f"start_state={start}")
        _print_known(before)
        for _ in range(frames_walk):
            env.step(buttons("RIGHT"))
        after_walk = snapshot(env.get_ram())
        walk_deltas = diff_changed(before, after_walk, limit=64)
        print(
            f"\nwalk RIGHT {frames_walk}f — "
            f"{len(walk_deltas)} changes (cap 64)"
        )
        for d in candidates_increasing(walk_deltas)[:12]:
            print(f"  + 0x{d.address:04X}: {d.before} -> {d.after}")
        print(
            f"  known player_x 0x{PLAYER_BASE + OFF_X:04X}: "
            f"{read_u16le(before, PLAYER_BASE + OFF_X)} -> "
            f"{read_u16le(after_walk, PLAYER_BASE + OFF_X)}"
        )
        print(
            f"  known camera_x 0x{ADDR_CAMERA_X:04X}: "
            f"{read_u16le(before, ADDR_CAMERA_X)} -> "
            f"{read_u16le(after_walk, ADDR_CAMERA_X)}"
        )
        mid = after_walk
        for _ in range(frames_attack):
            env.step(buttons("Y"))
        after_atk = snapshot(env.get_ram())
        atk_deltas = diff_changed(mid, after_atk, limit=64)
        print(f"\nattack Y {frames_attack}f — decreasing sample")
        for d in candidates_decreasing(atk_deltas)[:12]:
            print(f"  - 0x{d.address:04X}: {d.before} -> {d.after}")
        _print_known(after_atk)
        return 0
    finally:
        env.close()


def main() -> None:
    """CLI entry for the RAM probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walk-frames", type=int, default=45)
    parser.add_argument("--attack-frames", type=int, default=30)
    args = parser.parse_args()
    raise SystemExit(
        run_probe(
            frames_walk=args.walk_frames,
            frames_attack=args.attack_frames,
        )
    )


if __name__ == "__main__":
    main()

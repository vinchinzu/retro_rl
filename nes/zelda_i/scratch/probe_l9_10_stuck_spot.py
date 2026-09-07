"""Teleport Link to (128,93) inside room 0x10 (via the L9Room10EntryReal pin,
no combat) and hold DOWN, to see whether that spot is genuinely blocked or
whether the room10_silver_arrows_controller stall was caused by something
else (residual object, scroll edge, etc.)
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless
from zelda_i.level9.stair_run import _assign
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import ADDR_LINK_X, ADDR_LINK_Y, read_snapshot

STATE_NAME = "L9Room10EntryReal"


def main() -> int:
    configure_headless()
    env = make_env(GAME, STATE_NAME, GAME_DIR, render_mode="rgb_array")
    reset_obs(env)
    _assign(env, ADDR_LINK_X, 128)
    _assign(env, ADDR_LINK_Y, 93)
    env.step(nes_action())
    snap = read_snapshot(env.get_ram())
    print(f"placed xy=({snap.link_x},{snap.link_y}) tile={snap.colliding_tile:#x} screen={snap.screen:#x}")
    for i in range(200):
        env.step(nes_action("DOWN"))
        if i % 10 == 0:
            snap = read_snapshot(env.get_ram())
            print(f"f{i} xy=({snap.link_x},{snap.link_y}) tile={snap.colliding_tile:#x} mode={snap.mode} screen={snap.screen:#x}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

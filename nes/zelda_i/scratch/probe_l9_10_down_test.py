"""Isolated test: from L9Room10EntryReal, force DOWN for many frames and see
if Link's y actually changes, independent of the room10 controller logic.
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

STATE_NAME = "L9Room10EntryReal"


def main() -> int:
    configure_headless()
    env = make_env(GAME, STATE_NAME, GAME_DIR, render_mode="rgb_array")
    reset_obs(env)
    assist = UnlimitedHealthAssist(enabled=True)
    for i in range(600):
        env.step(nes_action("DOWN"))
        assist.apply_env(env, frame=i)
        if i % 20 == 0:
            snap = read_snapshot(env.get_ram())
            print(f"f{i} xy=({snap.link_x},{snap.link_y}) tile={snap.colliding_tile:#x} mode={snap.mode}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

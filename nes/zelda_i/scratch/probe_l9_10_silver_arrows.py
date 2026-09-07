"""Drive Level9Room10SilverArrowsController from the real power-on pin
L9Room10EntryReal and verify it collects the Silver Arrows (ADDR_ARROWS==2)
and returns to room 0x10.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l9_10_silver_arrows.py
"""
from __future__ import annotations

from pathlib import Path

from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level9.prefix import make_room10_silver_arrows_controller
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

STATE_NAME = "L9Room10EntryReal"


def main() -> int:
    configure_headless()
    env = make_env(GAME, STATE_NAME, GAME_DIR, render_mode="rgb_array")
    obs, _ = reset_obs(env)
    assist = UnlimitedHealthAssist(enabled=True)
    ctl = make_room10_silver_arrows_controller()
    frame = 0
    while not ctl.success and not ctl.failed:
        snap = read_snapshot(env.get_ram())
        act = ctl.step(snap)
        obs, *_ = env.step(act.action)
        frame += 1
        assist.apply_env(env, frame=frame)
        if frame % 500 == 0:
            print(f"f{frame} mode={snap.mode} screen=0x{snap.screen:02x} "
                  f"xy=({snap.link_x},{snap.link_y}) arrows={snap.arrows} "
                  f"pushed={ctl._pushed} floor={ctl._on_floor} note={act.reason}",
                  flush=True)
    snap = read_snapshot(env.get_ram())
    print(f"DONE success={ctl.success} failed={ctl.failed} frames={ctl.frames}")
    print(f"final mode={snap.mode} screen=0x{snap.screen:02x} "
          f"xy=({snap.link_x},{snap.link_y}) arrows={snap.arrows} notes={ctl.notes}")
    save_rgb_png(obs, Path("nes/zelda_i/recordings/l9_10_silver_arrows_probe_final.png"))
    env.close()
    return 0 if ctl.success else 1


if __name__ == "__main__":
    raise SystemExit(main())

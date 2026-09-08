"""Pin the real power-on state at the start of the silver-arrows chapter
(Level 9 room 0x76), so the 0x14 stall introduced downstream of the White
Sword detour can be iterated in seconds.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/pin_l9_entry.py
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.spine.survival import run_survival_spine

STATE_NAME = "L9EntryReal"
saved = {"done": False}


def on_frame(env, obs, action, frame):
    if saved["done"]:
        return
    snap = read_snapshot(env.get_ram())
    if snap.level == 9 and snap.screen == 0x76 and snap.mode == 5 and not snap.transitioning:
        print(f"f{frame}: L9 0x76 ({snap.link_x},{snap.link_y}) sword={snap.sword} "
              f"bombs={snap.bombs} keys={snap.keys} tf=0x{snap.triforce:02x} "
              f"containers={snap.heart_containers}", flush=True)
        save_state(env, GAME_DIR, GAME, STATE_NAME)
        saved["done"] = True


def main() -> int:
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    obs, _ = reset_obs(env)
    try:
        run = run_survival_spine(env, obs, assist=UnlimitedHealthAssist(enabled=True),
                                 through="level9-entry", on_frame=on_frame)
    finally:
        env.close()
    rep = run.report()
    print(f"pinned={saved['done']} ok={rep.get('ok')} "
          f"failed_stage={rep.get('failed_stage')}", flush=True)
    return 0 if saved["done"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

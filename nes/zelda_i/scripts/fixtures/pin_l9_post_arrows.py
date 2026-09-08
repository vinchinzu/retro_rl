"""Pin a REAL power-on savestate the first frame Link is back in room 0x10
holding the Silver Arrows (ADDR_ARROWS == 2), then let the run continue so
the same sitting also reports the next failing stage.

The pin is the start state for the Patra join, so downstream fixes can be
iterated in seconds instead of a ~6 minute power-on run.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/pin_l9_post_arrows.py
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.spine.survival import run_survival_spine

STATE_NAME = "L9PostArrowsReal"

saved = {"done": False}


def on_frame(env, obs, action, frame):
    if saved["done"]:
        return
    snap = read_snapshot(env.get_ram())
    if (
        snap.level == 9
        and snap.screen == 0x10
        and snap.mode == 5
        and not snap.transitioning
        and snap.arrows >= 2
    ):
        print(
            f"f{frame}: post-arrows 0x10 xy=({snap.link_x},{snap.link_y}) "
            f"arrows={snap.arrows} tf=0x{snap.triforce:02x} bombs={snap.bombs}",
            flush=True,
        )
        save_state(env, GAME_DIR, GAME, STATE_NAME)
        saved["done"] = True


def main() -> int:
    configure_headless()
    assist = UnlimitedHealthAssist(enabled=True)
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    obs, _ = reset_obs(env)
    try:
        run = run_survival_spine(
            env, obs, assist=assist, through="level9-credits", on_frame=on_frame
        )
    finally:
        env.close()
    rep = run.report()
    print(f"pinned={saved['done']} state={STATE_NAME}")
    print(f"ok={rep.get('ok')} failed_stage={rep.get('failed_stage')} "
          f"end_frame={rep.get('end_frame')}", flush=True)
    return 0 if saved["done"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

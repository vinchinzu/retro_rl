"""Pin a REAL power-on savestate at the moment Link enters L9 room 0x05
(the level9_stairs_05 hop), for fast iteration on the block-push deadlock
found in the full --through level9-credits run (rr-sz8.6).

Runs the actual production run_survival_spine() pipeline (byte-identical
to the real failing run) with an on_frame hook that saves state and aborts
as soon as snap.level==9 and snap.screen==0x05 and mode==PLAY_MODE, instead
of replaying the whole ~5min spine on every iteration afterward.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/pin_l9_room05_entry.py
"""

from __future__ import annotations

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.spine.survival import run_survival_spine

STATE_NAME = "L9Room05EntryReal"


class _StopEarly(Exception):
    pass


def main() -> int:
    configure_headless()
    assist = UnlimitedHealthAssist(enabled=True)
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    obs, _ = reset_obs(env)

    saved = {"done": False, "stable": 0}

    def on_frame(env, obs, action, frame):
        if saved["done"]:
            return
        snap = read_snapshot(env.get_ram())
        if snap.level == 9 and snap.screen == 0x05 and snap.mode == 5 and not snap.transitioning:
            saved["stable"] += 1
        else:
            saved["stable"] = 0
        # Require 60 consecutive stable frames -- the entry scroll can dip
        # back to mode==5 briefly mid-transition before truly settling
        # (found via contaminated experiment runs off a too-early pin).
        if saved["stable"] == 60:
            print(f"f{frame}: settled level9 room 0x05 xy=({snap.link_x},{snap.link_y}) "
                  f"keys={snap.keys} bombs={snap.bombs}")
            save_state(env, GAME_DIR, GAME, STATE_NAME)
            saved["done"] = True
            raise _StopEarly()

    try:
        run_survival_spine(env, obs, assist=assist, through="level9-credits", on_frame=on_frame)
    except _StopEarly:
        pass
    finally:
        env.close()

    if not saved["done"]:
        print("never reached level9 room 0x05 -- check run failed earlier")
        return 1
    print(f"saved state: {STATE_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

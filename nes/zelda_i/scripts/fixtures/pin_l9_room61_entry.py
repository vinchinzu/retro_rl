"""Pin a REAL power-on savestate at the moment Link enters L9 room 0x61
(the level9_stairs_61 hop -- other Patra fight + block push -> cellar 0x75),
for fast iteration on whatever's stalling it (rr-sz8.6/.7 continuous run
died here at frame 316954, final xy=(64,93), well off the expected south-
aisle/push path -- looks like the same class of naive-combat bug fixed for
stairs_05's Wizzrobes: patra_slash mashes A on a period without checking
the sword hitbox).

Runs the actual production run_survival_spine() pipeline (byte-identical to
the real failing run) with an on_frame hook that saves state and aborts as
soon as snap.level==9 and snap.screen==0x61 and mode==PLAY_MODE, instead of
replaying the whole spine on every iteration afterward.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/pin_l9_room61_entry.py
"""

from __future__ import annotations

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.spine.survival import run_survival_spine

STATE_NAME = "L9Room61EntryReal"


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
        if snap.level == 9 and snap.screen == 0x61 and snap.mode == 5 and not snap.transitioning:
            saved["stable"] += 1
        else:
            saved["stable"] = 0
        # stable==1 (not 60): the chained hop controller hands off to the
        # next hop's policy() on the exact snapshot where the predecessor's
        # own arrived()/success check first sees mode==PLAY_MODE,
        # screen==dest, not transitioning -- with no settle delay. A 60-frame
        # wait here let Patra's orbiting eyes keep moving well past that
        # point, producing a pin that diverged from the real hop-transition
        # geometry (found via two live power-on runs both failing
        # byte-identically at frame 332954 in this room, while this same
        # over-settled pin kept succeeding in isolation).
        if saved["stable"] == 1:
            print(f"f{frame}: settled level9 room 0x61 xy=({snap.link_x},{snap.link_y}) "
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
        print("never reached level9 room 0x61 -- check run failed earlier")
        return 1
    print(f"saved state: {STATE_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

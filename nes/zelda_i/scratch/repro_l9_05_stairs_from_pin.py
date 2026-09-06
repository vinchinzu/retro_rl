"""Fast repro of the level9_stairs_05 block-push deadlock from the real
power-on pin L9Room05EntryReal (see pin_l9_room05_entry.py).

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/repro_l9_05_stairs_from_pin.py
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level9.prefix import make_stairs_05_controller
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

STATE_NAME = "L9Room05EntryReal"


def main() -> int:
    configure_headless()
    assist = UnlimitedHealthAssist(enabled=True)
    env = make_env(GAME, STATE_NAME, GAME_DIR, render_mode="rgb_array")
    reset_obs(env)

    ctrl = make_stairs_05_controller()
    last_key = None
    for i in range(ctrl.max_frames):
        snap = read_snapshot(env.get_ram())
        act = ctrl.step(snap)
        key = (act.reason, snap.link_x, snap.link_y)
        if key != last_key:
            print(f"f{i}: xy=({snap.link_x},{snap.link_y}) act={act.reason}")
            last_key = key
        env.step(act.action)
        assist.apply_env(env, frame=i)
        if ctrl.failed or ctrl.success:
            print(f"DONE f{i}: failed={ctrl.failed} success={ctrl.success} notes={ctrl.notes}")
            break
    else:
        print("ran out of frames without flag")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

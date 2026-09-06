"""Fast repro of the level9_stairs_61 stall from the real power-on pin
L9Room61EntryReal (see pin_l9_room61_entry.py).

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/repro_l9_61_stairs_from_pin.py
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level9.prefix import make_stairs_61_controller
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

STATE_NAME = "L9Room61EntryReal"


def main() -> int:
    configure_headless()
    assist = UnlimitedHealthAssist(enabled=True)
    env = make_env(GAME, STATE_NAME, GAME_DIR, render_mode="rgb_array")
    reset_obs(env)

    ctrl = make_stairs_61_controller()
    last_key = None
    for i in range(ctrl.max_frames):
        snap = read_snapshot(env.get_ram())
        act = ctrl.step(snap)
        key = (act.reason, snap.link_x, snap.link_y)
        if key != last_key:
            eyes = [o for o in snap.objects if o.type_id == 0x25 and o.hp > 0]
            body = next((o for o in snap.objects if o.type_id == 0x47 and o.hp > 0), None)
            print(
                f"f{i}: xy=({snap.link_x},{snap.link_y}) act={act.reason} "
                f"n_eyes={len(eyes)} body={'alive' if body else None}"
            )
            last_key = key
        env.step(act.action)
        assist.apply_env(env, frame=i)
        if ctrl.failed or ctrl.success:
            print(f"DONE f{i}: failed={ctrl.failed} success={ctrl.success} notes={ctrl.notes}")
            break
    else:
        snap = read_snapshot(env.get_ram())
        eyes = [o for o in snap.objects if o.type_id == 0x25 and o.hp > 0]
        body = next((o for o in snap.objects if o.type_id == 0x47 and o.hp > 0), None)
        print(f"ran out of frames: xy=({snap.link_x},{snap.link_y}) n_eyes={len(eyes)} body={'alive' if body else None}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

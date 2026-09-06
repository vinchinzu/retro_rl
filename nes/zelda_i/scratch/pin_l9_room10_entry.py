"""Pin a REAL power-on savestate at the moment Link enters L9 room 0x10
(the Silver Arrows room), captured on the exact first frame (stable==1,
not a delayed settle -- see pin_l9_room61_entry.py's postmortem for why a
late capture diverges from the true hop-transition snapshot).

Live power-on run (rr-sz8.6, 2026-09-06) reaches this room with
arrows==1 (wooden) and NaturalSilverArrowsController already reporting
success -- nothing walks onto the Silver Arrows floor item, so the
level9_silver_arrows stop predicate (ADDR_ARROWS==2) never fires. This pin
is for probing the room's real item-pickup geometry.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/pin_l9_room10_entry.py
"""

from __future__ import annotations

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.spine.survival import run_survival_spine

STATE_NAME = "L9Room10EntryReal"


class _StopEarly(Exception):
    pass


def main() -> int:
    configure_headless()
    assist = UnlimitedHealthAssist(enabled=True)
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    obs, _ = reset_obs(env)

    saved = {"done": False}

    def on_frame(env, obs, action, frame):
        if saved["done"]:
            return
        snap = read_snapshot(env.get_ram())
        if snap.level == 9 and snap.screen == 0x10 and snap.mode == 5 and not snap.transitioning:
            print(
                f"f{frame}: settled level9 room 0x10 xy=({snap.link_x},{snap.link_y}) "
                f"arrows={snap.arrows} room_item_id={snap.room_item_id} "
                f"objs={[(o.slot, o.type_id, o.x, o.y, o.hp) for o in snap.objects]}"
            )
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
        print("never reached level9 room 0x10 -- check run failed earlier")
        return 1
    print(f"saved state: {STATE_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

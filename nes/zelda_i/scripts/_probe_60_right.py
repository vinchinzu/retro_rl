
"""From 0x60 north settle (120,77), hold RIGHT toward the visible stairs."""
from retro_harness.env import make_env
from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.level9_stairs import dest_report, stair_transition_modes
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot
from zelda_i.scripts.run_level9_ganon import FIXTURE_SOURCE, _step
from zelda_i.scripts.run_level9_stairs import materialize_stair_room, _exit_cellar

def main():
    configure_headless()
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    total = [0]
    try:
        obs, loader, loaded = materialize_stair_room(env, 0x60, total=total)
        snap = read_snapshot(env.get_ram())
        print("SETTLE", snap.link_x, snap.link_y, hex(snap.colliding_tile))
        for i in range(300):
            snap = read_snapshot(env.get_ram())
            if i % 20 == 0:
                print(f"f{i} xy=({snap.link_x},{snap.link_y}) tile=0x{snap.colliding_tile:02x} room=0x{snap.screen:02x} mode={snap.mode}")
            if stair_transition_modes(snap.mode) or snap.screen != 0x60:
                print("LEFT", dest_report(snap))
                break
            obs = _step(env, nes_action("RIGHT"), assist=None, total=total)
        snap = read_snapshot(env.get_ram())
        save_rgb_png(obs, RECORDINGS_DIR / "l9_stair60_hold_right.png")
        if stair_transition_modes(snap.mode) or snap.mode in (9, 10, 11, 16):
            obs, snap = _exit_cellar(env, total=total, side="left")
            print("CELLAR", dest_report(snap))
            save_rgb_png(obs, RECORDINGS_DIR / "l9_stair60_hold_right_exit.png")
        else:
            print("END", dest_report(snap)["link"], hex(snap.screen), snap.mode)
    finally:
        env.close()

if __name__ == "__main__":
    main()

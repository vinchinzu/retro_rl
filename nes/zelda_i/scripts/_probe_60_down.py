
"""From 0x60 north mouth, walk onto the center tiles and record dest."""
from retro_harness.env import make_env
from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.level9_stairs import dest_report, stair_transition_modes, walk_to_step
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot
from zelda_i.scripts.run_level9_ganon import FIXTURE_SOURCE, _idle, _step
from zelda_i.scripts.run_level9_stairs import materialize_stair_room, _exit_cellar

def main():
    configure_headless()
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    total = [0]
    try:
        obs, loader, loaded = materialize_stair_room(env, 0x60, total=total)
        snap = read_snapshot(env.get_ram())
        print("SETTLE", loader.label, dest_report(snap)["link"], "tile", hex(snap.colliding_tile))
        save_rgb_png(obs, RECORDINGS_DIR / "l9_stair60_north_settle.png")
        # Walk a column down the center, then sweep x on the mid band.
        targets = [
            (120, 96), (120, 112), (120, 128), (120, 141), (120, 157),
            (96, 141), (144, 141), (80, 141), (160, 141),
            (96, 128), (144, 128), (80, 112), (160, 112),
            (208, 96), (208, 141),  # block-reveal stairs pos is (208, 96)
        ]
        for tx, ty in targets:
            for _ in range(250):
                snap = read_snapshot(env.get_ram())
                if stair_transition_modes(snap.mode) or (snap.screen != 0x60 and snap.mode not in (6, 7)):
                    print(f"LEFT at ({tx},{ty}) room=0x{snap.screen:02x} mode={snap.mode} xy=({snap.link_x},{snap.link_y})")
                    break
                frame = walk_to_step(snap, tx, ty, y_first=True)
                if frame.reason == "walk_arrived":
                    break
                obs = _step(env, frame.action, assist=None, total=total)
            else:
                snap = read_snapshot(env.get_ram())
            print(f"target ({tx},{ty}) now=({snap.link_x},{snap.link_y}) tile=0x{snap.colliding_tile:02x} room=0x{snap.screen:02x} mode={snap.mode}")
            if stair_transition_modes(snap.mode) or snap.screen != 0x60:
                break
        snap = read_snapshot(env.get_ram())
        if stair_transition_modes(snap.mode) or snap.mode in (9, 10, 11, 16):
            save_rgb_png(obs, RECORDINGS_DIR / "l9_stair60_passage.png")
            print("PASSAGE", dest_report(snap))
            for side in ("left", "right"):
                env2_needed = False
                obs, snap = _exit_cellar(env, total=total, side=side)
                print("EXIT", side, dest_report(snap))
                save_rgb_png(obs, RECORDINGS_DIR / f"l9_stair60_exit_{side}.png")
                if snap.mode == 5 and snap.screen != 0x60:
                    break
        else:
            obs = _idle(env, 10, assist=None, total=total)
            snap = read_snapshot(env.get_ram())
            save_rgb_png(obs, RECORDINGS_DIR / "l9_stair60_north_after.png")
            print("STILL", dest_report(snap))
        write_json_report(RECORDINGS_DIR / "l9_stair60_north.json", dest_report(snap))
    finally:
        env.close()

if __name__ == "__main__":
    main()

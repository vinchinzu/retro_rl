
"""Hold UP from 0x60 south door onto the visible center stairs."""
from retro_harness.env import make_env
from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.level9_stairs import dest_report, stair_transition_modes
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
        print("SETTLE", dest_report(snap))
        save_rgb_png(obs, RECORDINGS_DIR / "l9_stair60_holdup_start.png")
        log = []
        for i in range(400):
            snap = read_snapshot(env.get_ram())
            if i % 30 == 0 or stair_transition_modes(snap.mode) or snap.screen != 0x60:
                log.append({
                    "i": i,
                    "room": snap.screen,
                    "mode": snap.mode,
                    "x": snap.link_x,
                    "y": snap.link_y,
                    "tile": snap.colliding_tile,
                })
                print(f"f{i} room=0x{snap.screen:02x} mode={snap.mode} xy=({snap.link_x},{snap.link_y}) tile=0x{snap.colliding_tile:02x}")
            if stair_transition_modes(snap.mode) or (snap.screen != 0x60 and snap.mode not in (6, 7)):
                break
            obs = _step(env, nes_action("UP"), assist=None, total=total)
        snap = read_snapshot(env.get_ram())
        if stair_transition_modes(snap.mode) or snap.mode in (9, 10, 11, 16):
            print("IN PASSAGE", dest_report(snap))
            save_rgb_png(obs, RECORDINGS_DIR / "l9_stair60_holdup_passage.png")
            obs, snap = _exit_cellar(env, total=total, side="left")
            print("LEFT EXIT", dest_report(snap))
            save_rgb_png(obs, RECORDINGS_DIR / "l9_stair60_holdup_left.png")
        else:
            obs = _idle(env, 15, assist=None, total=total)
            snap = read_snapshot(env.get_ram())
            print("AFTER UP", dest_report(snap))
            save_rgb_png(obs, RECORDINGS_DIR / "l9_stair60_holdup_after.png")
        write_json_report(RECORDINGS_DIR / "l9_stair60_holdup.json", {"log": log, "final": dest_report(snap)})
    finally:
        env.close()

if __name__ == "__main__":
    main()

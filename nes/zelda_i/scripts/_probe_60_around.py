
"""Leave 0x60 south mouth and try to step on the center stair/block tiles."""
from retro_harness.env import make_env
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.level9_stairs import dest_report, stair_transition_modes, walk_to_step
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot
from zelda_i.scripts.run_level9_ganon import FIXTURE_SOURCE, _idle, _step
from zelda_i.scripts.run_level9_stairs import materialize_stair_room, _exit_cellar

def hold(env, total, buttons, frames, tag):
    obs = None
    for i in range(frames):
        snap = read_snapshot(env.get_ram())
        if stair_transition_modes(snap.mode) or (snap.screen != 0x60 and snap.mode not in (6, 7)):
            print(f"{tag} f{i} LEFT room=0x{snap.screen:02x} mode={snap.mode} xy=({snap.link_x},{snap.link_y})")
            return obs, snap, True
        obs = _step(env, nes_action(*buttons) if buttons else nes_idle_action(), assist=None, total=total)
    snap = read_snapshot(env.get_ram())
    print(f"{tag} end xy=({snap.link_x},{snap.link_y}) tile=0x{snap.colliding_tile:02x} room=0x{snap.screen:02x} mode={snap.mode}")
    return obs, snap, False

def walk_pts(env, total, pts):
    obs = None
    for x, y in pts:
        for _ in range(300):
            snap = read_snapshot(env.get_ram())
            if stair_transition_modes(snap.mode) or (snap.screen != 0x60 and snap.mode not in (6, 7)):
                return obs, snap, True
            frame = walk_to_step(snap, x, y, y_first=False)
            if frame.reason == "walk_arrived":
                break
            obs = _step(env, frame.action, assist=None, total=total)
        snap = read_snapshot(env.get_ram())
        print(f"  at target ({x},{y}) now=({snap.link_x},{snap.link_y}) tile=0x{snap.colliding_tile:02x}")
    return obs, read_snapshot(env.get_ram()), False

def main():
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    paths = [
        ("left_up", (("LEFT",), 40), (("UP",), 80), (("RIGHT",), 80)),
        ("right_up", (("RIGHT",), 40), (("UP",), 80), (("LEFT",), 80)),
        ("diag_lu", (("LEFT", "UP"), 120),),
        ("diag_ru", (("RIGHT", "UP"), 120),),
        ("xfirst_left", "waypoints", ((96, 189), (64, 189), (64, 141), (96, 141), (120, 141))),
        ("xfirst_right", "waypoints", ((144, 189), (176, 189), (176, 141), (144, 141), (120, 141))),
        ("push_from_left", "waypoints", ((64, 189), (64, 144), (100, 144))),
    ]
    for name, *spec in paths:
        env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
        total = [0]
        try:
            obs, loader, loaded = materialize_stair_room(env, 0x60, total=total)
            print("====", name, "loaded", loaded, dest_report(read_snapshot(env.get_ram()))["link"])
            left = False
            snap = read_snapshot(env.get_ram())
            if spec and spec[0] == "waypoints":
                obs, snap, left = walk_pts(env, total, spec[1])
            else:
                for buttons, frames in spec:
                    obs, snap, left = hold(env, total, buttons, frames, name)
                    if left:
                        break
            if left or stair_transition_modes(snap.mode):
                print("TRANSITION", dest_report(snap))
                if snap.mode in (9, 10, 11, 16) or stair_transition_modes(snap.mode):
                    obs, snap = _exit_cellar(env, total=total, side="left")
                    print("CELLAR LEFT", dest_report(snap))
            png = RECORDINGS_DIR / f"l9_stair60_{name}.png"
            save_rgb_png(obs if obs is not None else env.render(), png)
            results.append({"name": name, "dest": dest_report(snap), "png": str(png)})
            print("RESULT", name, "room", hex(snap.screen), "mode", snap.mode, "xy", snap.link_x, snap.link_y)
        finally:
            env.close()
    write_json_report(RECORDINGS_DIR / "l9_stair60_around.json", {"attempts": results})

if __name__ == "__main__":
    main()

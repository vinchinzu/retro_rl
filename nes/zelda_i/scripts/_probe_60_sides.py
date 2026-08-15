
"""Load 0x60 from east and west; try to walk onto center tiles."""
from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.level9_ganon import LEVEL9
from zelda_i.level9_stairs import StairLoader, dest_report, stair_transition_modes, walk_to_step
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_CUR_OPENED_DOORS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_NEXT_SCREEN,
    ADDR_OPEN_DOORWAY_MASK,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.scripts.run_level9_ganon import FIXTURE_SOURCE, FULL_LOADOUT, _assign, _idle, _step
from zelda_i.scripts.run_level9_stairs import _apply_loader, _hold_until_room, _exit_cellar

def try_loader(from_room, direction, lx, ly, tag):
    loader = StairLoader(0x60, from_room, direction, lx, ly, tag)
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    total = [0]
    try:
        reset_obs(env)
        _apply_loader(env, loader)
        obs, loaded = _hold_until_room(env, loader, total=total)
        if loaded:
            obs = _idle(env, 20, assist=None, total=total)
        snap = read_snapshot(env.get_ram())
        print(f"==== {tag} loaded={loaded} room=0x{snap.screen:02x} mode={snap.mode} xy=({snap.link_x},{snap.link_y}) tile=0x{snap.colliding_tile:02x}")
        save_rgb_png(obs if obs is not None else env.render(), RECORDINGS_DIR / f"l9_stair60_{tag}_settle.png")
        if not loaded:
            return {"tag": tag, "loaded": False, "dest": dest_report(snap)}
        targets = [(snap.link_x, 141), (120, 141), (96, 141), (144, 141), (208, 96), (80, 141)]
        for tx, ty in targets:
            for _ in range(280):
                snap = read_snapshot(env.get_ram())
                if stair_transition_modes(snap.mode) or (snap.screen != 0x60 and snap.mode not in (6, 7)):
                    print(f"  LEFT {tag} -> 0x{snap.screen:02x} mode={snap.mode} xy=({snap.link_x},{snap.link_y})")
                    break
                frame = walk_to_step(snap, tx, ty, y_first=False)
                if frame.reason == "walk_arrived":
                    break
                obs = _step(env, frame.action, assist=None, total=total)
            snap = read_snapshot(env.get_ram())
            print(f"  {tag} target ({tx},{ty}) now=({snap.link_x},{snap.link_y}) tile=0x{snap.colliding_tile:02x} room=0x{snap.screen:02x} mode={snap.mode}")
            if stair_transition_modes(snap.mode) or snap.screen != 0x60:
                break
        if stair_transition_modes(snap.mode) or snap.mode in (9, 10, 11, 16):
            obs, snap = _exit_cellar(env, total=total, side="left")
            print(f"  {tag} cellar left", dest_report(snap)["screen"], dest_report(snap)["mode"])
        save_rgb_png(obs if obs is not None else env.render(), RECORDINGS_DIR / f"l9_stair60_{tag}_after.png")
        return {"tag": tag, "loaded": True, "dest": dest_report(snap)}
    finally:
        env.close()

def main():
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        try_loader(0x61, "LEFT", 0x20, 0xBD, "from61_left"),
        try_loader(0x5F, "RIGHT", 0xD0, 0xBD, "from5f_right"),
        try_loader(0x61, "LEFT", 0x20, 0x8D, "from61_left_midy"),
        try_loader(0x50, "DOWN", 0x40, 0xDD, "from50_down_x40"),
        try_loader(0x50, "DOWN", 0xC0, 0xDD, "from50_down_xc0"),
    ]
    write_json_report(RECORDINGS_DIR / "l9_stair60_sides.json", {"attempts": rows})

if __name__ == "__main__":
    main()

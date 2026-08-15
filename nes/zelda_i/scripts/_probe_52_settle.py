
"""Confirm 0x77 left cellar dest settles live Patra in 0x52."""
from retro_harness.env import make_env
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.level9_stairs import dest_report, landed_final_patra
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_LINK_X, ADDR_LINK_Y, ADDR_MODE, PLAY_MODE, read_snapshot
from zelda_i.scripts.run_level9_ganon import FIXTURE_SOURCE, _assign, _idle, _step
from zelda_i.scripts.run_level9_stairs import materialize_stair_room

ADDR_UPDATING = 0x0011
ADDR_UW_EXIT = 0x005A


def main():
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    total = [0]
    try:
        obs, loader, loaded = materialize_stair_room(env, 0x77, total=total)
        _assign(env, ADDR_MODE, 9)
        _assign(env, 0x0013, 0)
        _assign(env, ADDR_UPDATING, 0)
        _assign(env, ADDR_UW_EXIT, 0)
        for i in range(400):
            obs = _step(env, nes_idle_action(), assist=None, total=total)
            snap = read_snapshot(env.get_ram())
            if snap.mode == 9 and int(env.get_ram()[ADDR_UPDATING]) != 0 and i > 20:
                break
        _assign(env, ADDR_LINK_X, 0x50)
        _assign(env, ADDR_LINK_Y, 0x3D)
        obs = _idle(env, 4, assist=None, total=total)
        for i in range(80):
            obs = _step(env, nes_action("UP"), assist=None, total=total)
            snap = read_snapshot(env.get_ram())
            if snap.screen == 0x52:
                break
        # Let the game finish entering 0x52.
        for i in range(400):
            snap = read_snapshot(env.get_ram())
            if snap.mode == PLAY_MODE and snap.screen == 0x52 and not snap.transitioning:
                obs = _idle(env, 24, assist=None, total=total)
                snap = read_snapshot(env.get_ram())
                break
            obs = _step(env, nes_idle_action(), assist=None, total=total)
        dest = dest_report(snap)
        save_rgb_png(obs, RECORDINGS_DIR / "l9_77_left_0x52_settled.png")
        print(
            f"SETTLED 0x{dest['screen']:02X} mode={dest['mode']} "
            f"patra={dest['final_patra_live']} eyes={dest['patra_eyes']} "
            f"landed={dest['landed_final_patra']} "
            f"doors={dest['doors']} "
            f"objs={[o['type_name'] for o in dest['objects'][:10]]}"
        )
        write_json_report(
            RECORDINGS_DIR / "l9_77_left_0x52_settled.json",
            {
                "ok": bool(dest["landed_final_patra"]),
                "loader": loader.label,
                "loaded": loaded,
                "dest": dest,
            },
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()

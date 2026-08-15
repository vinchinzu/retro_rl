
"""Hypothesis: cellar 0x77 left mouth (AttrsA=0x52) drops into Patra.

Also try standing on 0x67 / 0x77 stair-looking tiles with controller.
"""
from retro_harness.env import make_env
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.level9_stairs import dest_report, landed_final_patra, stair_transition_modes
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_LINK_X, ADDR_LINK_Y, ADDR_MODE, PLAY_MODE, read_snapshot
from zelda_i.scripts.run_level9_ganon import FIXTURE_SOURCE, _assign, _idle, _step
from zelda_i.scripts.run_level9_stairs import materialize_stair_room, _exit_cellar

CELLAR_MODE = 9


def place_and_nudge(env, total, x, y, nudges=40):
    _assign(env, ADDR_LINK_X, x)
    _assign(env, ADDR_LINK_Y, y)
    obs = _idle(env, 4, assist=None, total=total)
    for direction in ("DOWN", "UP", "LEFT", "RIGHT"):
        for _ in range(nudges):
            snap = read_snapshot(env.get_ram())
            if stair_transition_modes(snap.mode) or snap.mode in (9, 10, 11, 16):
                return obs, snap, True
            if snap.screen not in (0x67, 0x77) and snap.mode not in (4, 6, 7):
                return obs, snap, True
            obs = _step(env, nes_action(direction), assist=None, total=total)
    return obs, read_snapshot(env.get_ram()), False


def try_natural(room, stands):
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    total = [0]
    try:
        obs, loader, loaded = materialize_stair_room(env, room, total=total)
        print(f"NATURAL 0x{room:02X} loaded={loaded} {loader.label}")
        if not loaded:
            return {"room": room, "loaded": False}
        for x, y in stands:
            obs, snap, hit = place_and_nudge(env, total, x, y)
            print(f"  stand ({x},{y}) room=0x{snap.screen:02x} mode={snap.mode} xy=({snap.link_x},{snap.link_y}) tile=0x{snap.colliding_tile:02x} hit={hit}")
            if hit:
                save_rgb_png(obs, RECORDINGS_DIR / f"l9_stair_0x{room:02x}_hit.png")
                if snap.mode in (9, 10, 11, 16) or stair_transition_modes(snap.mode):
                    obs, snap = _exit_cellar(env, total=total, side="left")
                    print("  cellar-left", dest_report(snap))
                    save_rgb_png(obs, RECORDINGS_DIR / f"l9_stair_0x{room:02x}_left.png")
                    if not landed_final_patra(snap):
                        # one more env for right
                        pass
                return {"room": room, "hit": True, "dest": dest_report(snap), "patra": landed_final_patra(snap)}
        save_rgb_png(obs, RECORDINGS_DIR / f"l9_stair_0x{room:02x}_nohit.png")
        return {"room": room, "hit": False, "final": dest_report(read_snapshot(env.get_ram()))}
    finally:
        env.close()


def try_force_cellar_77(side):
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    total = [0]
    try:
        obs, loader, loaded = materialize_stair_room(env, 0x77, total=total)
        print("FORCE cellar 0x77 loaded", loaded)
        _assign(env, ADDR_MODE, CELLAR_MODE)
        _assign(env, ADDR_LINK_X, 0x50 if side == "left" else 0xB0)
        _assign(env, ADDR_LINK_Y, 0x80)
        obs = _idle(env, 30, assist=None, total=total)
        snap = read_snapshot(env.get_ram())
        print(f"  after mode9 poke side={side} room=0x{snap.screen:02x} mode={snap.mode} xy=({snap.link_x},{snap.link_y})")
        save_rgb_png(obs, RECORDINGS_DIR / f"l9_77_mode9_{side}_start.png")
        obs, snap = _exit_cellar(env, total=total, side=side)
        print(f"  exit {side}", dest_report(snap))
        save_rgb_png(obs, RECORDINGS_DIR / f"l9_77_mode9_{side}_exit.png")
        return {"side": side, "dest": dest_report(snap), "patra": landed_final_patra(snap)}
    finally:
        env.close()


def main():
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "natural_67": try_natural(0x67, (
            (0x80, 0x8D), (0xB0, 0x90), (0xA0, 0x80), (0xC0, 0x80),
            (0x74, 0x85), (0x84, 0x85), (0xD0, 0x60), (0x78, 0x90),
        )),
        "natural_77": try_natural(0x77, (
            (0xB0, 0x90), (0xA0, 0x90), (0xC0, 0x90), (0x80, 0x8D),
            (0xD0, 0x60), (0x90, 0x88),
        )),
        "force_77_left": try_force_cellar_77("left"),
        "force_77_right": try_force_cellar_77("right"),
    }
    write_json_report(RECORDINGS_DIR / "l9_77_cellar_probe.json", report)
    print("DONE", {k: (v.get("patra") if isinstance(v, dict) else v) for k, v in report.items()})


if __name__ == "__main__":
    main()

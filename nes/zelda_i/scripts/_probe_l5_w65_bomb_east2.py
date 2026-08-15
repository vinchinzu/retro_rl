"""Retry 0x65 bomb-east; wait through scroll to 0x66, then north."""
from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import goto, idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_path import select_b_item_menu, walk_axis, _step
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Whistle65"


def dump(env):
    s = read_snapshot(env.get_ram())
    return {
        "sc": f"0x{s.screen:02x}",
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "keys": int(s.keys),
        "bombs": int(s.bombs),
        "doors": int(s.cur_opened_doors),
        "mask": int(s.open_doorway_mask),
    }


def wait_play(env, assist, total, n=300):
    for _ in range(n):
        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and not s.transitioning:
            idle(env, assist, total, 10)
            return
        _step(env, assist, total, nes_action("RIGHT"))


def main():
    configure_headless()
    env = None
    total = [1]
    log = []
    try:
        env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
        assist = UnlimitedHealthAssist(enabled=True)
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 12)
        for axis, tgt in (("y", 109), ("x", 208), ("y", 141), ("x", 216)):
            walk_axis(env, assist, total, axis, tgt, max_f=400)
        goto(env, assist, total, 216, 141, tol=3, max_f=300)
        for _ in range(6):
            _step(env, assist, total, nes_action("RIGHT"))
        idle(env, assist, total, 6)
        select_b_item_menu(env, assist, total, 1)
        room0 = read_snapshot(env.get_ram()).screen
        _step(env, assist, total, nes_action("RIGHT", "B"))
        for _ in range(14):
            _step(env, assist, total, nes_action("LEFT"))
        idle(env, assist, total, 110)
        for _ in range(280):
            s = read_snapshot(env.get_ram())
            if s.screen != room0 or s.mode in (6, 7):
                break
            _step(env, assist, total, nes_action("RIGHT"))
        wait_play(env, assist, total)
        rec = {"tag": "after_bomb", **dump(env)}
        log.append(rec)
        print("AFTER_BOMB", rec, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w65_east2.png")

        if read_snapshot(env.get_ram()).screen == 0x66:
            # leave west mouth, north door
            walk_axis(env, assist, total, "x", 200, max_f=300)
            walk_axis(env, assist, total, "y", 141, max_f=300)
            walk_axis(env, assist, total, "x", 120, max_f=400)
            walk_axis(env, assist, total, "y", 93, max_f=400)
            push_dir(env, assist, total, "UP", frames=280)
            wait_play(env, assist, total)
            rec = {"tag": "north66", **dump(env)}
            log.append(rec)
            print("NORTH66", rec, flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w66_north2.png")

        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and s.screen in (0x66, 0x56, 0x76):
            name = f"Level5Whistle{s.screen:02X}"
            write_state_bytes(state_path(GAME_DIR, GAME, name), env.em.get_state())
            write_state_provenance(
                GAME_DIR / "custom_integrations" / GAME / f"{name}.state",
                source_state_path=GAME_DIR / "custom_integrations" / GAME / "Level5Whistle65.state",
                request={"segment": name, "via": "0x65 bomb-east settle", "key_poke": False, "door_poke": False},
                selected_trial={"success": True, "room": int(s.screen), "whistle_0x065C": 1},
                natural_entry=False,
            )
            print("SAVED", name, flush=True)

        body = {"ok": read_snapshot(env.get_ram()).screen in (0x66, 0x56), "log": log, "final": dump(env), "pokes": False}
        write_json_report(RECORDINGS_DIR / "l5_w65_east2.json", body)
        print("FINAL", body["final"], "OK", body["ok"], flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

"""From Level5Whistle76: south OW, re-enter L5, UP 0x76→0x66→0x56."""
from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_path import walk_axis
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Whistle76"


def dump(env):
    s = read_snapshot(env.get_ram())
    return {
        "L": s.level,
        "sc": f"0x{s.screen:02x}",
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "keys": int(s.keys),
        "doors": int(s.cur_opened_doors),
        "mask": int(s.open_doorway_mask),
    }


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
        print("START", dump(env), flush=True)
        # 0x76 south to OW
        walk_axis(env, assist, total, "y", 141, max_f=300)
        walk_axis(env, assist, total, "x", 120, max_f=300)
        walk_axis(env, assist, total, "y", 205, max_f=400)
        push_dir(env, assist, total, "DOWN", frames=280)
        idle(env, assist, total, 20)
        for _ in range(400):
            s = read_snapshot(env.get_ram())
            if s.level == 0 and s.mode == PLAY_MODE:
                break
            env.step(nes_action("DOWN"))
            total[0] += 1
            assist.apply_env(env, frame=total[0])
        idle(env, assist, total, 20)
        log.append({"tag": "ow", **dump(env)})
        print("OW", log[-1], flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w76_ow.png")

        # Re-enter: align x≈112 and UP
        walk_axis(env, assist, total, "y", 141, max_f=300)
        walk_axis(env, assist, total, "x", 112, max_f=400)
        push_dir(env, assist, total, "UP", frames=400)
        idle(env, assist, total, 20)
        for _ in range(400):
            s = read_snapshot(env.get_ram())
            if s.level == 5 and s.mode == PLAY_MODE:
                break
            env.step(nes_action("UP"))
            total[0] += 1
            assist.apply_env(env, frame=total[0])
        idle(env, assist, total, 20)
        log.append({"tag": "reenter", **dump(env)})
        print("REENTER", log[-1], flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w76_reenter.png")

        if read_snapshot(env.get_ram()).level == 5:
            walk_axis(env, assist, total, "y", 141, max_f=300)
            walk_axis(env, assist, total, "x", 120, max_f=300)
            for i in range(900):
                s = read_snapshot(env.get_ram())
                if s.mode == PLAY_MODE and s.screen == 0x56:
                    break
                if s.mode == PLAY_MODE and s.screen == 0x66:
                    if s.link_y < 141 and s.link_x < 80:
                        env.step(nes_action("DOWN"))
                    elif abs(s.link_x - 120) > 4:
                        env.step(nes_action("LEFT" if s.link_x > 120 else "RIGHT"))
                    else:
                        env.step(nes_action("UP"))
                else:
                    env.step(nes_action("UP"))
                total[0] += 1
                assist.apply_env(env, frame=total[0])
                if i % 100 == 0:
                    print("F", i, dump(env), flush=True)
            idle(env, assist, total, 16)
            log.append({"tag": "after_up", **dump(env)})
            print("AFTER", log[-1], flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w76_reenter_up.png")

        s = read_snapshot(env.get_ram())
        if s.level == 5 and s.screen == 0x56:
            write_state_bytes(state_path(GAME_DIR, GAME, "Level5Whistle56"), env.em.get_state())
            write_state_provenance(
                GAME_DIR / "custom_integrations" / GAME / "Level5Whistle56.state",
                source_state_path=GAME_DIR / "custom_integrations" / GAME / "Level5Whistle76.state",
                request={"segment": "Level5Whistle56", "via": "OW reenter then 0x76 UP 0x66 0x56", "key_poke": False, "door_poke": False},
                selected_trial={"success": True, "room": 0x56, "whistle_0x065C": 1},
                natural_entry=False,
            )
            print("SAVED Level5Whistle56", flush=True)
        body = {"ok": s.level == 5 and s.screen == 0x56, "log": log, "final": dump(env), "pokes": False}
        write_json_report(RECORDINGS_DIR / "l5_w76_reenter.json", body)
        print("FINAL", body["final"], "OK", body["ok"], flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

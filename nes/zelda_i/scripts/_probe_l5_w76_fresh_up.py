"""Save 0x76 from Whistle66, reload fresh, hold UP x=120 through 0x66 to 0x56."""
from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_path import walk_axis
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

def dump(env):
    s = read_snapshot(env.get_ram())
    return {
        "sc": f"0x{s.screen:02x}",
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "keys": int(s.keys),
        "doors": int(s.cur_opened_doors),
        "mask": int(s.open_doorway_mask),
    }


def to_76():
    configure_headless()
    env = make_env(GAME, "Level5Whistle66", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    total = [1]
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 10)
        walk_axis(env, assist, total, "y", 141, max_f=300)
        walk_axis(env, assist, total, "x", 120, max_f=400)
        walk_axis(env, assist, total, "y", 205, max_f=400)
        push_dir(env, assist, total, "DOWN", frames=240)
        idle(env, assist, total, 20)
        for _ in range(180):
            s = read_snapshot(env.get_ram())
            if s.mode == PLAY_MODE and s.screen == 0x76:
                break
            env.step(nes_action("DOWN"))
            total[0] += 1
            assist.apply_env(env, frame=total[0])
        idle(env, assist, total, 16)
        print("AT76", dump(env), flush=True)
        if read_snapshot(env.get_ram()).screen == 0x76:
            write_state_bytes(state_path(GAME_DIR, GAME, "Level5Whistle76"), env.em.get_state())
            write_state_provenance(
                GAME_DIR / "custom_integrations" / GAME / "Level5Whistle76.state",
                source_state_path=GAME_DIR / "custom_integrations" / GAME / "Level5Whistle66.state",
                request={"segment": "Level5Whistle76", "via": "0x66 south open", "key_poke": False, "door_poke": False},
                selected_trial={"success": True, "room": 0x76, "whistle_0x065C": 1},
                natural_entry=False,
            )
            print("SAVED Level5Whistle76", flush=True)
            return True
        return False
    finally:
        env.close()


def from_76():
    configure_headless()
    env = make_env(GAME, "Level5Whistle76", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    total = [1]
    log = []
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 16)
        log.append({"tag": "fresh76", **dump(env)})
        print("FRESH76", log[-1], flush=True)
        # Leave south if needed, align north, hold UP through 0x66.
        walk_axis(env, assist, total, "y", 141, max_f=300)
        walk_axis(env, assist, total, "x", 120, max_f=300)
        push_dir(env, assist, total, "UP", frames=400)
        idle(env, assist, total, 12)
        # If in 0x66, keep holding UP at x=120, finish ladder if needed.
        for i in range(800):
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
            if i % 80 == 0:
                print("F", i, dump(env), flush=True)
        idle(env, assist, total, 16)
        log.append({"tag": "after_up", **dump(env)})
        print("AFTER", log[-1], flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w76_fresh_up.png")
        s = read_snapshot(env.get_ram())
        if s.screen == 0x56:
            write_state_bytes(state_path(GAME_DIR, GAME, "Level5Whistle56"), env.em.get_state())
            write_state_provenance(
                GAME_DIR / "custom_integrations" / GAME / "Level5Whistle56.state",
                source_state_path=GAME_DIR / "custom_integrations" / GAME / "Level5Whistle76.state",
                request={"segment": "Level5Whistle56", "via": "fresh 0x76 UP through 0x66", "key_poke": False, "door_poke": False},
                selected_trial={"success": True, "room": 0x56, "whistle_0x065C": 1},
                natural_entry=False,
            )
            print("SAVED Level5Whistle56", flush=True)
        body = {"ok": s.screen == 0x56, "log": log, "final": dump(env), "pokes": False}
        write_json_report(RECORDINGS_DIR / "l5_w76_fresh_up.json", body)
        print("FINAL", body["final"], "OK", body["ok"], flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    if to_76():
        from_76()

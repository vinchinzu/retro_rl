"""Level5Whistle66 south 0x76, then free-UP through 0x66 to 0x56."""
from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_path import Level5West65Controller, walk_axis
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Whistle66"


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

        # Leave west mouth, south door (open).
        walk_axis(env, assist, total, "y", 141, max_f=300)
        walk_axis(env, assist, total, "x", 120, max_f=400)
        walk_axis(env, assist, total, "y", 205, max_f=400)
        push_dir(env, assist, total, "DOWN", frames=240)
        idle(env, assist, total, 16)
        for _ in range(200):
            s = read_snapshot(env.get_ram())
            if s.mode == PLAY_MODE and s.screen == 0x76:
                break
            env.step(nes_action("DOWN"))
            total[0] += 1
            assist.apply_env(env, frame=total[0])
        idle(env, assist, total, 12)
        rec = {"tag": "south76", **dump(env)}
        log.append(rec)
        print("SOUTH76", rec, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w66_76.png")

        if read_snapshot(env.get_ram()).screen == 0x76:
            ctl = Level5West65Controller()
            for _ in range(ctl.max_frames):
                snap = read_snapshot(env.get_ram())
                action = ctl.step(snap)
                env.step(action.action)
                total[0] += 1
                assist.apply_env(env, frame=total[0])
                if ctl.success or ctl.failed:
                    break
            rec = {"tag": "west65_ctl", **ctl.report(), **dump(env)}
            log.append(rec)
            print("CTL", rec, flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w76_to_56.png")

        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and s.screen == 0x56:
            write_state_bytes(state_path(GAME_DIR, GAME, "Level5Whistle56"), env.em.get_state())
            write_state_provenance(
                GAME_DIR / "custom_integrations" / GAME / "Level5Whistle56.state",
                source_state_path=GAME_DIR / "custom_integrations" / GAME / "Level5Whistle66.state",
                request={"segment": "Level5Whistle56", "via": "0x66 south 0x76 free-UP 0x56", "key_poke": False, "door_poke": False},
                selected_trial={"success": True, "room": 0x56, "whistle_0x065C": 1},
                natural_entry=False,
            )
            print("SAVED Level5Whistle56", flush=True)

        body = {"ok": read_snapshot(env.get_ram()).screen == 0x56, "log": log, "final": dump(env), "pokes": False}
        write_json_report(RECORDINGS_DIR / "l5_w66_via76.json", body)
        print("FINAL", body["final"], "OK", body["ok"], flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

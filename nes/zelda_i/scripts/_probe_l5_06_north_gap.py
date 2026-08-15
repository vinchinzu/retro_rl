"""0x06: north-wall around diamond, then center stairs. Never x=120 while y>170."""
from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_path import cellar_other_mouth, walk_axis, walk_east_from_05, _step
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Whistle05"
CELLAR = (9, 10, 11, 16)


def dump(env):
    s = read_snapshot(env.get_ram())
    return {
        "sc": f"0x{s.screen:02x}",
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "tile": int(s.colliding_tile),
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "keys": int(s.keys),
        "doors": int(s.cur_opened_doors),
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
        east = walk_east_from_05(env, assist, total)
        print("EAST", east, flush=True)
        log.append({"tag": "east", **east})

        # North-wall gap, then center. Never approach south door.
        for axis, tgt in (("y", 93), ("x", 80), ("y", 141), ("x", 120), ("y", 141)):
            ok = walk_axis(env, assist, total, axis, tgt, max_f=400)
            rec = {"tag": f"{axis}:{tgt}", "ok": ok, **dump(env)}
            log.append(rec)
            print("NAV", rec, flush=True)
            s = read_snapshot(env.get_ram())
            if s.mode in CELLAR or s.screen == 0x07:
                break

        idle(env, assist, total, 12)
        if read_snapshot(env.get_ram()).mode not in CELLAR:
            for d in ("UP", "DOWN", "RIGHT", "LEFT"):
                for _ in range(18):
                    s = read_snapshot(env.get_ram())
                    if s.mode in CELLAR or s.screen != 0x06:
                        break
                    # refuse south-door x-band while low
                    if d == "DOWN" and s.link_y > 170 and abs(s.link_x - 120) < 20:
                        break
                    _step(env, assist, total, nes_action(d))
                idle(env, assist, total, 6)
                if read_snapshot(env.get_ram()).mode in CELLAR:
                    break

        log.append({"tag": "after_nudge", **dump(env)})
        print("AFTER", log[-1], flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_north_gap.png")

        s = read_snapshot(env.get_ram())
        if s.mode in CELLAR or s.screen == 0x07:
            cellar = cellar_other_mouth(env, assist, total)
            log.append({"tag": "cellar", **{k: cellar[k] for k in cellar if k != "start"}})
            print("CELLAR", log[-1], flush=True)
            idle(env, assist, total, 16)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_north_64.png")
            s = read_snapshot(env.get_ram())
            if s.mode == PLAY_MODE and s.screen == 0x64:
                path = write_state_bytes(state_path(GAME_DIR, GAME, "Level5Whistle64"), env.em.get_state())
                write_state_provenance(
                    path,
                    source_state_path=GAME_DIR / "custom_integrations" / GAME / "Level5Whistle05.state",
                    request={
                        "segment": "Level5Whistle64",
                        "via": "0x06 north-wall diamond stairs 0x07 left",
                        "key_poke": False,
                        "door_poke": False,
                    },
                    selected_trial={"success": True, "room": 0x64, "whistle_0x065C": 1},
                    natural_entry=False,
                )
                print("SAVED Level5Whistle64", flush=True)

        body = {"ok": read_snapshot(env.get_ram()).screen == 0x64, "log": log, "final": dump(env), "pokes": False}
        write_json_report(RECORDINGS_DIR / "l5_06_north_gap.json", body)
        print("FINAL", body["final"], "OK", body["ok"], flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

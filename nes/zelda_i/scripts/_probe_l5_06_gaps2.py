"""Leave west mouth first, then north or south around 0x06 diamond to center stairs."""
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
        "keys": int(s.keys),
        "doors": int(s.cur_opened_doors),
    }


def run_path(env, assist, total, steps, name):
    log = []
    for axis, tgt in steps:
        ok = walk_axis(env, assist, total, axis, tgt, max_f=400)
        rec = {"name": name, "step": f"{axis}:{tgt}", "ok": ok, **dump(env)}
        log.append(rec)
        print("NAV", rec, flush=True)
        s = read_snapshot(env.get_ram())
        if s.mode in CELLAR or s.screen != 0x06:
            return True, log
    idle(env, assist, total, 10)
    for d in ("UP", "RIGHT", "LEFT", "DOWN"):
        for _ in range(14):
            s = read_snapshot(env.get_ram())
            if s.mode in CELLAR or s.screen != 0x06:
                return True, log
            if d == "DOWN" and s.link_y > 175 and 100 <= s.link_x <= 140:
                break
            _step(env, assist, total, nes_action(d))
        idle(env, assist, total, 4)
        if read_snapshot(env.get_ram()).mode in CELLAR:
            return True, log
    return False, log


def main():
    configure_headless()
    paths = (
        ("leave_north", (("x", 40), ("y", 93), ("x", 80), ("y", 141), ("x", 120))),
        ("leave_south80", (("x", 40), ("y", 189), ("x", 80), ("y", 141), ("x", 120))),
        ("leave_south64", (("x", 40), ("y", 189), ("x", 64), ("y", 141), ("x", 120))),
        ("diag_n", (("x", 40), ("y", 109), ("x", 80), ("y", 125), ("x", 120), ("y", 141))),
        ("diag_s", (("x", 40), ("y", 173), ("x", 80), ("y", 157), ("x", 120), ("y", 141))),
    )
    env = None
    total = [1]
    all_log = []
    try:
        env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
        assist = UnlimitedHealthAssist(enabled=True)
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 12)
        east = walk_east_from_05(env, assist, total)
        print("EAST", east, flush=True)
        all_log.append({"tag": "east", **east})
        found = False
        used = None
        for name, steps in paths:
            s = read_snapshot(env.get_ram())
            if s.screen != 0x06:
                break
            ok, log = run_path(env, assist, total, steps, name)
            all_log.extend(log)
            if ok:
                found = True
                used = name
                break
        print("FOUND", found, used, dump(env), flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_gaps2.png")
        s = read_snapshot(env.get_ram())
        if s.mode in CELLAR or s.screen == 0x07:
            cellar = cellar_other_mouth(env, assist, total)
            all_log.append({"tag": "cellar", **{k: cellar[k] for k in cellar if k != "start"}})
            print("CELLAR", all_log[-1], flush=True)
            idle(env, assist, total, 16)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_gaps2_64.png")
            s = read_snapshot(env.get_ram())
            if s.mode == PLAY_MODE and s.screen == 0x64:
                path = write_state_bytes(state_path(GAME_DIR, GAME, "Level5Whistle64"), env.em.get_state())
                write_state_provenance(
                    path,
                    source_state_path=GAME_DIR / "custom_integrations" / GAME / "Level5Whistle05.state",
                    request={"segment": "Level5Whistle64", "via": f"0x06 {used} stairs 0x07 left", "key_poke": False, "door_poke": False},
                    selected_trial={"success": True, "room": 0x64, "whistle_0x065C": 1},
                    natural_entry=False,
                )
                print("SAVED Level5Whistle64", flush=True)
        body = {"ok": read_snapshot(env.get_ram()).screen == 0x64, "used": used, "log": all_log, "final": dump(env), "pokes": False}
        write_json_report(RECORDINGS_DIR / "l5_06_gaps2.json", body)
        print("FINAL", body["final"], "OK", body["ok"], flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

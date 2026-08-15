"""Level5Whistle05 → east 0x06 diamond-gap stairs → 0x07 left → 0x64."""
from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_path import cellar_other_mouth, take_center_stairs_06, walk_east_from_05
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Whistle05"


def dump(env):
    s = read_snapshot(env.get_ram())
    return {
        "sc": f"0x{s.screen:02x}",
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "keys": int(s.keys),
        "doors": int(s.cur_opened_doors),
    }


def main():
    configure_headless()
    env = None
    total = [1]
    hops = []
    try:
        env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
        assist = UnlimitedHealthAssist(enabled=True)
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 12)
        print("START", dump(env), flush=True)
        east = walk_east_from_05(env, assist, total)
        hops.append({"hop": "east_05", **{k: east[k] for k in east}})
        print("EAST", hops[-1], flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_gap06_arrive.png")
        stairs = take_center_stairs_06(env, assist, total)
        hops.append({"hop": "stairs_06", **{k: stairs[k] for k in stairs if k != "log"}, "steps": stairs.get("log")})
        print("STAIRS", {k: hops[-1][k] for k in hops[-1] if k != "steps"}, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_gap06_stairs.png")
        cellar = None
        if stairs.get("success") or read_snapshot(env.get_ram()).mode in (9, 10, 11, 16) or read_snapshot(env.get_ram()).screen == 0x07:
            cellar = cellar_other_mouth(env, assist, total)
            hops.append({"hop": "cellar", **{k: cellar[k] for k in cellar if k != "start"}})
            print("CELLAR", hops[-1], flush=True)
            idle(env, assist, total, 16)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_gap06_64.png")
            s = read_snapshot(env.get_ram())
            if s.mode == PLAY_MODE and s.screen == 0x64:
                path = write_state_bytes(state_path(GAME_DIR, GAME, "Level5Whistle64"), env.em.get_state())
                write_state_provenance(
                    path,
                    source_state_path=GAME_DIR / "custom_integrations" / GAME / "Level5Whistle05.state",
                    request={
                        "segment": "Level5Whistle64",
                        "predecessor_entry": True,
                        "start_state": STATE,
                        "via": "0x05 east 0x06 diamond stairs 0x07 left",
                        "key_poke": False,
                        "door_poke": False,
                    },
                    selected_trial={"success": True, "room": 0x64, "whistle_0x065C": 1},
                    natural_entry=False,
                )
                print("SAVED Level5Whistle64", flush=True)
        body = {"ok": read_snapshot(env.get_ram()).screen == 0x64, "hops": hops, "final": dump(env), "pokes": False}
        write_json_report(RECORDINGS_DIR / "l5_06_gap_stairs.json", body)
        print("FINAL", body["final"], "OK", body["ok"], flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

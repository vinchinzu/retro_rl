"""Level5Whistle05 → east 0x06, north-around push 0x68, idle (120,141) → 0x07 → 0x64."""
from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_path import cellar_to_64, take_center_stairs_06, walk_east_from_05
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Whistle05"


def dump(env):
    s = read_snapshot(env.get_ram())
    return {
        "sc": f"0x{s.screen:02x}",
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "tile": int(s.colliding_tile),
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "keys": int(s.keys),
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
        east = walk_east_from_05(env, assist, total)
        log.append({"tag": "east", **{k: east[k] for k in east}})
        print("EAST", log[-1], flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_push_arrive.png")

        stairs = take_center_stairs_06(env, assist, total)
        log.append({"tag": "stairs", **{k: stairs[k] for k in stairs if k != "log"}, "steps": stairs.get("log")})
        print("STAIRS", stairs.get("success"), "dest", hex(stairs.get("dest", 0)), "mode", stairs.get("mode"), "xy", stairs.get("xy"), "tile", stairs.get("tile"), flush=True)
        for row in stairs.get("log") or []:
            print("  ", row, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_push_stairs.png")

        s = read_snapshot(env.get_ram())
        if s.mode in (9, 10, 11, 16) or s.screen == 0x07:
            write_state_bytes(state_path(GAME_DIR, GAME, "Level5Whistle07"), env.em.get_state())
            write_state_provenance(
                state_path(GAME_DIR, GAME, "Level5Whistle07"),
                source_state_path=GAME_DIR / "custom_integrations" / GAME / f"{STATE}.state",
                request={
                    "segment": "Level5Whistle07",
                    "predecessor_entry": True,
                    "start_state": STATE,
                    "via": "0x06 north-around push 0x68 idle 0x71",
                    "key_poke": False,
                    "door_poke": False,
                },
                selected_trial={"success": True, "room": int(s.screen), "mode": int(s.mode), "whistle_0x065C": 1, "xy": [s.link_x, s.link_y]},
                natural_entry=False,
            )
            print("SAVED Level5Whistle07", dump(env), flush=True)
            cellar = cellar_to_64(env, assist, total)
            log.append({"tag": "cellar", **{k: cellar[k] for k in cellar if k != "start"}})
            print("CELLAR", {k: cellar[k] for k in cellar if k != "log"}, flush=True)
            for row in cellar.get("log") or []:
                print("  ", row, flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_push_64.png")
            s = read_snapshot(env.get_ram())
            if s.mode == PLAY_MODE and s.screen == 0x64 and int(read_u8(env.get_ram(), ADDR_WHISTLE)) == 1:
                write_state_bytes(state_path(GAME_DIR, GAME, "Level5Whistle64"), env.em.get_state())
                write_state_provenance(
                    state_path(GAME_DIR, GAME, "Level5Whistle64"),
                    source_state_path=GAME_DIR / "custom_integrations" / GAME / f"{STATE}.state",
                    request={
                        "segment": "Level5Whistle64",
                        "predecessor_entry": True,
                        "start_state": STATE,
                        "via": "0x06 north-around push 0x68, idle 120,141, cellar left 0x64",
                        "key_poke": False,
                        "door_poke": False,
                    },
                    selected_trial={"success": True, "room": 0x64, "whistle_0x065C": 1, "xy": [s.link_x, s.link_y]},
                    natural_entry=False,
                )
                print("SAVED Level5Whistle64", dump(env), flush=True)

        body = {"ok": read_snapshot(env.get_ram()).screen == 0x64, "log": log, "final": dump(env), "pokes": False, "status_claim": None}
        write_json_report(RECORDINGS_DIR / "l5_06_push_stairs.json", body)
        print("FINAL", body["final"], "OK", body["ok"], flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

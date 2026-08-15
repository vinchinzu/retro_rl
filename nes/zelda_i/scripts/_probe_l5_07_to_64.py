"""From Level5Whistle07: raw-step pit drop, left mouth → 0x64. No pokes."""
from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_path import cellar_to_64
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8


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
    env = make_env(GAME, "Level5Whistle07", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    total = [1]
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 12)
        start = dump(env)
        print("START", start, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_07_to64_start.png")
        rec = cellar_to_64(env, assist, total)
        print("CELLAR", rec, flush=True)
        for row in rec.get("log") or []:
            print(" ", row, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_07_to64_final.png")
        end = dump(env)
        ok = end["sc"] == "0x64" and end["mode"] == PLAY_MODE and end["whistle"] == 1
        if ok:
            write_state_bytes(state_path(GAME_DIR, GAME, "Level5Whistle64"), env.em.get_state())
            write_state_provenance(
                state_path(GAME_DIR, GAME, "Level5Whistle64"),
                source_state_path=GAME_DIR / "custom_integrations" / GAME / "Level5Whistle07.state",
                request={
                    "segment": "Level5Whistle64",
                    "predecessor_entry": True,
                    "start_state": "Level5Whistle07",
                    "via": "raw x192 pit189 x48 y61 UP",
                    "key_poke": False,
                    "door_poke": False,
                },
                selected_trial={"success": True, **end},
                natural_entry=False,
            )
            print("SAVED Level5Whistle64", end, flush=True)
        body = {"ok": ok, "start": start, "cellar": {k: rec[k] for k in rec if k != "start"}, "final": end, "pokes": False, "status_claim": None}
        write_json_report(RECORDINGS_DIR / "l5_07_to64.json", body)
        print("FINAL", end, "OK", ok, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()

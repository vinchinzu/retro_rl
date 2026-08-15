"""Map walkable dirs from 0x04 item chamber, then take right ladder down and left ladder out."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Whistle"


def step(env, assist, total, a):
    env.step(a)
    total[0] += 1
    assist.apply_env(env, frame=total[0])


def dump(env):
    s = read_snapshot(env.get_ram())
    return {
        "sc": f"0x{s.screen:02x}",
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "tile": int(s.colliding_tile),
    }


def hold(env, assist, total, d, n):
    trail = []
    last = None
    for i in range(n):
        step(env, assist, total, nes_action(d))
        s = read_snapshot(env.get_ram())
        key = (s.screen, s.mode, s.link_x, s.link_y)
        if key != last:
            trail.append({"i": i, **dump(env)})
            last = key
        if s.mode == PLAY_MODE and s.screen != 0x04:
            break
    return trail


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
        idle(env, assist, total, 80)
        # unstick fanfare
        for _ in range(20):
            step(env, assist, total, nes_action("LEFT"))
        idle(env, assist, total, 10)
        log.append({"tag": "ready", **dump(env)})
        print("READY", log[-1], flush=True)

        for d in ("RIGHT", "DOWN", "UP", "LEFT"):
            # reload-less: from current, try dir, then we'll just keep going if useful
            t = hold(env, assist, total, d, 80)
            rec = {"dir": d, "end": dump(env), "n": len(t), "trail": t[:8]}
            log.append(rec)
            print("DIR", d, "end", rec["end"], "steps", t[:6], flush=True)

        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_04_chamber.png")

        # From wherever we are: seek right ladder (x high) then DOWN, then LEFT to 48, then UP
        s = read_snapshot(env.get_ram())
        print("SEEK from", dump(env), flush=True)
        t1 = hold(env, assist, total, "RIGHT", 200)
        log.append({"tag": "seek_right", "end": dump(env), "trail": t1[:10]})
        print("RIGHT", dump(env), flush=True)
        t2 = hold(env, assist, total, "DOWN", 200)
        log.append({"tag": "seek_down", "end": dump(env), "trail": t2[:10]})
        print("DOWN", dump(env), flush=True)
        t3 = hold(env, assist, total, "LEFT", 400)
        log.append({"tag": "seek_left", "end": dump(env), "trail": t3[:10]})
        print("LEFT", dump(env), flush=True)
        t4 = hold(env, assist, total, "UP", 400)
        log.append({"tag": "seek_up", "end": dump(env), "trail": t4[:12]})
        print("UP", dump(env), flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_04_chamber_end.png")

        s = read_snapshot(env.get_ram())
        if s.screen != 0x04 or s.mode == PLAY_MODE:
            write_state_bytes(state_path(GAME_DIR, GAME, "Level5WhistleOut"), env.em.get_state())
            write_state_provenance(
                state_path(GAME_DIR, GAME, "Level5WhistleOut"),
                source_state_path=GAME_DIR / "custom_integrations" / GAME / "Level5Whistle.state",
                request={
                    "segment": "Level5WhistleOut",
                    "predecessor_entry": True,
                    "start_state": STATE,
                    "via": "0x04 chamber ladders",
                    "key_poke": False,
                    "door_poke": False,
                    "bomb_count_poke": False,
                    "selected_item_poke": False,
                },
                selected_trial={"success": True, "room": int(s.screen), "whistle_0x065C": 1},
                natural_entry=False,
            )
            print("SAVED", dump(env), flush=True)

        write_json_report(RECORDINGS_DIR / "l5_04_chamber.json", {"log": log, "final": dump(env), "pokes": False})
        print("FINAL", dump(env), flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

"""0x06: west door -> around diamond -> center stairs -> 0x07 -> 0x64."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import exit_door, idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_path import cellar_other_mouth, walk_axis
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
    }


def wait_play(env, assist, total, max_f=240):
    for _ in range(max_f):
        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and not s.transitioning:
            return
        env.step(nes_idle_action())
        total[0] += 1
        assist.apply_env(env, frame=total[0])


def main():
    configure_headless()
    env = make_env(GAME, "Level5Whistle05", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    total = [1]
    log = []
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 12)
        exit_door(env, assist, total, "RIGHT", push=200)
        wait_play(env, assist, total)
        print("ARRIVE", dump(env), flush=True)

        paths = [
            (("x", 96), ("y", 93), ("x", 120), ("y", 141)),
            (("x", 96), ("y", 189), ("x", 120), ("y", 141)),
            (("x", 80), ("y", 93), ("x", 120), ("y", 125)),
            (("x", 80), ("y", 189), ("x", 120), ("y", 157)),
        ]
        done = False
        for i, steps in enumerate(paths):
            s = read_snapshot(env.get_ram())
            if s.mode in (9, 10, 11, 16) or s.screen == 0x07:
                done = True
                break
            print("PATH", i, flush=True)
            for axis, tgt in steps:
                ok = walk_axis(env, assist, total, axis, tgt, max_f=500)
                rec = {"path": i, "step": f"{axis}{tgt}", "ok": ok, **dump(env)}
                log.append(rec)
                print("  ", rec, flush=True)
                s = read_snapshot(env.get_ram())
                if s.mode in (9, 10, 11, 16) or s.screen != 0x06:
                    done = True
                    break
            if done:
                break
            # nudge onto tile
            for d in ("UP", "DOWN", "RIGHT", "LEFT"):
                push_dir(env, assist, total, d, frames=24)
                rec = {"path": i, "nudge": d, **dump(env)}
                log.append(rec)
                print("  NUDGE", rec, flush=True)
                s = read_snapshot(env.get_ram())
                if s.mode in (9, 10, 11, 16) or s.screen != 0x06:
                    done = True
                    break
            if done:
                break

        wait_play(env, assist, total, max_f=80)
        after = dump(env)
        print("AFTER", after, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_center.png")

        dest = after
        if after["mode"] in (9, 10, 11, 16) or after["sc"] == "0x07":
            cellar = cellar_other_mouth(env, assist, total)
            print("CELLAR", {k: cellar[k] for k in cellar if k != "start"}, flush=True)
            wait_play(env, assist, total)
            dest = dump(env)
            print("DEST", dest, flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_64_from06.png")
            if dest["sc"] != "0x64":
                # force opposite mouth
                walk_axis(env, assist, total, "y", 189, max_f=400)
                sx = read_snapshot(env.get_ram()).link_x
                tx = 48 if sx > 128 else 192
                walk_axis(env, assist, total, "x", tx, max_f=500)
                push_dir(env, assist, total, "UP", frames=260)
                wait_play(env, assist, total)
                dest = dump(env)
                print("DEST2", dest, flush=True)
            if dest["sc"] == "0x64" and dest["mode"] == 5:
                write_state_bytes(state_path(GAME_DIR, GAME, "Level5Whistle64"), env.em.get_state())
                write_state_provenance(
                    state_path(GAME_DIR, GAME, "Level5Whistle64"),
                    source_state_path=GAME_DIR / "custom_integrations" / GAME / "Level5Whistle05.state",
                    request={
                        "segment": "Level5Whistle64",
                        "predecessor_entry": True,
                        "start_state": "Level5Whistle05",
                        "via": "0x06 diamond center stairs, 0x07 other mouth",
                        "key_poke": False,
                        "door_poke": False,
                        "bomb_count_poke": False,
                        "selected_item_poke": False,
                    },
                    selected_trial={"success": True, "room": 0x64, "whistle_0x065C": 1},
                    natural_entry=False,
                )
                print("SAVED Level5Whistle64", dest, flush=True)

        write_json_report(RECORDINGS_DIR / "l5_06_center.json", {"log": log, "final": dest, "pokes": False})
        print("FINAL", dest, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()

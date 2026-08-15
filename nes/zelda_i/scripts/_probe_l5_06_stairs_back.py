"""From Level5Whistle05: east 0x06, hunt stairs to 0x07, left mouth to 0x64."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_path import cellar_other_mouth, walk_axis
from zelda_i.level9_stairs import on_stair_tile, on_warp_tile
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Whistle05"
CELLAR = (9, 10, 11, 16)


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
        "tile": int(s.colliding_tile),
        "stair": on_stair_tile(s),
        "warp": on_warp_tile(s),
        "doors": int(s.cur_opened_doors),
        "next": int(s.next_screen),
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "keys": int(s.keys),
        "blocks": [
            {"x": o.x, "y": o.y, "st": o.state}
            for o in s.objects
            if 1 <= o.slot <= 12 and o.type_id == 0x68
        ],
    }


def wait_leave(env, assist, total, room, n=200):
    for _ in range(n):
        s = read_snapshot(env.get_ram())
        if s.screen != room or s.mode in CELLAR:
            return True
        step(env, assist, total, nes_idle_action())
    return False


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
        log.append({"tag": "start", **dump(env)})
        print("START", log[-1], flush=True)

        # East into 0x06
        walk_axis(env, assist, total, "y", 141, max_f=300)
        walk_axis(env, assist, total, "x", 224, max_f=400)
        push_dir(env, assist, total, "RIGHT", frames=220)
        idle(env, assist, total, 16)
        log.append({"tag": "at06", **dump(env)})
        print("AT06", log[-1], flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w06_arrive.png")

        stands = (
            (96, 157),
            (96, 141),
            (96, 133),
            (96, 128),
            (80, 141),
            (112, 141),
            (120, 141),
            (96, 165),
            (96, 173),
            (64, 141),
            (128, 141),
            (96, 117),
            (96, 109),
            (208, 208),  # L9 block-stairs
            (120, 125),
            (104, 149),
        )
        found = False
        for tx, ty in stands:
            s = read_snapshot(env.get_ram())
            if s.mode in CELLAR or s.screen != 0x06:
                found = True
                break
            walk_axis(env, assist, total, "y", ty, max_f=280)
            walk_axis(env, assist, total, "x", tx, max_f=280)
            idle(env, assist, total, 10)
            rec = {"tag": "stand", "tgt": [tx, ty], **dump(env)}
            log.append(rec)
            print("STAND", rec, flush=True)
            if rec["stair"] or rec["warp"] or rec["mode"] in CELLAR or rec["sc"] != "0x06":
                for d in ("UP", "DOWN", "LEFT", "RIGHT"):
                    s = read_snapshot(env.get_ram())
                    if s.mode in CELLAR or s.screen != 0x06:
                        found = True
                        break
                    for _ in range(20):
                        step(env, assist, total, nes_action(d))
                        s = read_snapshot(env.get_ram())
                        if s.mode in CELLAR or s.screen != 0x06:
                            found = True
                            break
                    if found:
                        break
            if found:
                break
            # nudge on tile
            for d in ("UP", "DOWN"):
                for _ in range(12):
                    step(env, assist, total, nes_action(d))
                    s = read_snapshot(env.get_ram())
                    if s.mode in CELLAR or s.screen != 0x06:
                        found = True
                        break
                if found:
                    break
            if found:
                break

        idle(env, assist, total, 20)
        wait_leave(env, assist, total, 0x06)
        log.append({"tag": "after_hunt", **dump(env)})
        print("AFTER_HUNT", log[-1], flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w06_stairs.png")

        s = read_snapshot(env.get_ram())
        if s.mode in CELLAR or s.screen == 0x07:
            cellar = cellar_other_mouth(env, assist, total)
            log.append({"tag": "cellar", **{k: cellar[k] for k in cellar if k != "start"}})
            print("CELLAR", log[-1], flush=True)
            idle(env, assist, total, 16)
            log.append({"tag": "after_cellar", **dump(env)})
            print("AFTER_CELLAR", log[-1], flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w06_64.png")
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
                        "via": "0x05 east 0x06 stairs 0x07 left mouth",
                        "key_poke": False,
                        "door_poke": False,
                    },
                    selected_trial={"success": True, "room": 0x64, "whistle_0x065C": 1},
                    natural_entry=False,
                )
                print("SAVED Level5Whistle64", flush=True)

        body = {"ok": read_snapshot(env.get_ram()).screen == 0x64, "log": log, "final": dump(env), "pokes": False}
        write_json_report(RECORDINGS_DIR / "l5_06_stairs_back.json", body)
        print("FINAL", body["final"], "OK", body["ok"], flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

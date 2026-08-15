"""From Level5Whistle05: east into 0x06, north-band around diamond, take stairs."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import exit_door, idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_path import cellar_other_mouth, walk_axis
from zelda_i.level9_stairs import on_stair_tile
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8


def step(env, assist, total, a):
    env.step(a)
    total[0] += 1
    assist.apply_env(env, frame=total[0])


def dump(env):
    s = read_snapshot(env.get_ram())
    return {
        "sc": f"0x{s.screen:02x}",
        "next": f"0x{s.next_screen:02x}",
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "tile": int(s.colliding_tile),
        "stair": on_stair_tile(s),
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "doors": s.cur_opened_doors,
        "item": s.room_item_id,
        "objs": [
            {"t": o.type_id, "x": o.x, "y": o.y, "hp": o.hp}
            for o in s.objects
            if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF)
        ],
    }


def wait_play(env, assist, total, max_f=300):
    for _ in range(max_f):
        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and not s.transitioning:
            return dump(env)
        step(env, assist, total, nes_idle_action())
    return dump(env)


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
        idle(env, assist, total, 16)
        print("START", dump(env), flush=True)
        rec = exit_door(env, assist, total, "RIGHT", push=200)
        log.append({"tag": "east", **dump(env)})
        print("EAST", dump(env), flush=True)
        settled = wait_play(env, assist, total)
        log.append({"tag": "settle", **settled})
        print("SETTLE", settled, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_settle.png")

        # North band around diamond, then to stairs spawn (96,133)
        for axis, tgt in (("y", 93), ("x", 96), ("y", 133), ("x", 96)):
            ok = walk_axis(env, assist, total, axis, tgt, max_f=500)
            rec = {"tag": f"{axis}{tgt}", "ok": ok, **dump(env)}
            log.append(rec)
            print("WALK", rec, flush=True)
            s = read_snapshot(env.get_ram())
            if s.mode in (9, 10, 11, 16) or s.screen == 0x07:
                break

        # Push block / stand on stairs
        if read_snapshot(env.get_ram()).mode == PLAY_MODE and read_snapshot(env.get_ram()).screen == 0x06:
            for d in ("UP", "LEFT", "DOWN", "RIGHT"):
                push_dir(env, assist, total, d, frames=40)
                rec = {"tag": f"push_{d}", **dump(env)}
                log.append(rec)
                print("PUSH", rec, flush=True)
                s = read_snapshot(env.get_ram())
                if s.mode in (9, 10, 11, 16) or s.screen != 0x06:
                    break
            if read_snapshot(env.get_ram()).screen == 0x06:
                for tx, ty in ((96, 128), (96, 141), (112, 144), (80, 141), (120, 141), (96, 144)):
                    walk_axis(env, assist, total, "y", ty, max_f=300)
                    walk_axis(env, assist, total, "x", tx, max_f=300)
                    idle(env, assist, total, 10)
                    rec = {"tag": f"stand_{tx}_{ty}", **dump(env)}
                    log.append(rec)
                    print("STAND", rec, flush=True)
                    s = read_snapshot(env.get_ram())
                    if s.mode in (9, 10, 11, 16) or s.screen != 0x06:
                        break

        wait_play(env, assist, total, max_f=120)
        after = dump(env)
        print("AFTER_STAIRS", after, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_after_stairs.png")

        if after["mode"] in (9, 10, 11, 16) or after["sc"] == "0x07":
            cellar = cellar_other_mouth(env, assist, total)
            print("CELLAR", cellar, flush=True)
            wait_play(env, assist, total)
            dest = dump(env)
            print("DEST", dest, flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_cellar_dest.png")
            if dest["sc"] == "0x64" and dest["mode"] == 5:
                write_state_bytes(state_path(GAME_DIR, GAME, "Level5Whistle64"), env.em.get_state())
                write_state_provenance(
                    state_path(GAME_DIR, GAME, "Level5Whistle64"),
                    source_state_path=GAME_DIR / "custom_integrations" / GAME / "Level5Whistle05.state",
                    request={
                        "segment": "Level5Whistle64",
                        "predecessor_entry": True,
                        "start_state": "Level5Whistle05",
                        "via": "0x05 east, 0x06 north-band stairs, 0x07 other mouth",
                        "key_poke": False,
                        "door_poke": False,
                        "bomb_count_poke": False,
                        "selected_item_poke": False,
                    },
                    selected_trial={"success": True, "room": 0x64, "whistle_0x065C": 1},
                    natural_entry=False,
                )
                print("SAVED Level5Whistle64", dest, flush=True)
            after = dest

        write_json_report(RECORDINGS_DIR / "l5_06_around.json", {"log": log, "final": after, "pokes": False})
        print("FINAL", after, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()

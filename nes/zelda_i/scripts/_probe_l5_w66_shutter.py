"""Probe 0x66 north shutter from Level5Whistle64. No pokes."""
from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.level5_path import bomb_east_from_65, walk_axis, walk_east_from_64
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot, read_u8, ADDR_WHISTLE

def dump(env):
    s = read_snapshot(env.get_ram())
    objs = [
        {"t": f"0x{o.type_id:02x}", "hp": o.hp, "xy": [o.x, o.y], "st": o.state}
        for o in s.objects
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF)
    ]
    return {
        "sc": f"0x{s.screen:02x}",
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "doors": int(s.cur_opened_doors),
        "mask": int(s.open_doorway_mask),
        "all_dead": int(s.room_all_dead),
        "item": int(s.room_item_id),
        "count": int(s.room_obj_count),
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "keys": int(s.keys),
        "objs": objs,
    }

def main():
    configure_headless()
    env = make_env(GAME, "Level5Whistle64", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    n = [1]
    log = []
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, n, 8)
        print("START", dump(env), flush=True)
        print("E64", walk_east_from_64(env, assist, n), flush=True)
        print("B65", {k: v for k, v in bomb_east_from_65(env, assist, n).items() if k != "menu"}, flush=True)
        rec = dump(env)
        log.append({"tag": "in66", **rec})
        print("IN66", rec, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w66_shutter.png")

        # Idle in case shutter is delayed.
        idle(env, assist, n, 90)
        rec = dump(env)
        log.append({"tag": "idle90", **rec})
        print("IDLE90", rec, flush=True)

        # South to 0x76 then back north, then try UP again.
        walk_axis(env, assist, n, "x", 120, max_f=400)
        walk_axis(env, assist, n, "y", 189, max_f=400)
        push_dir(env, assist, n, "DOWN", frames=220)
        idle(env, assist, n, 16)
        rec = dump(env)
        log.append({"tag": "south", **rec})
        print("SOUTH", rec, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w66_south.png")

        if rec["sc"] == "0x76":
            walk_axis(env, assist, n, "x", 120, max_f=400)
            walk_axis(env, assist, n, "y", 93, max_f=400)
            push_dir(env, assist, n, "UP", frames=240)
            idle(env, assist, n, 16)
            rec = dump(env)
            log.append({"tag": "reenter66", **rec})
            print("REENTER66", rec, flush=True)
            walk_axis(env, assist, n, "x", 120, max_f=300)
            walk_axis(env, assist, n, "y", 93, max_f=300)
            push_dir(env, assist, n, "UP", frames=260)
            idle(env, assist, n, 16)
            rec = dump(env)
            log.append({"tag": "north2", **rec})
            print("NORTH2", rec, flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w66_north2.png")

        write_json_report(RECORDINGS_DIR / "l5_w66_shutter.json", {"log": log, "final": rec})
        print("FINAL", rec, flush=True)
    finally:
        env.close()

if __name__ == "__main__":
    main()

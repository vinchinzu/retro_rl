"""Replay to 0x65 via whistle return; dump doors/objs; try N/E/S."""
from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.level5_path import cellar_07_to_64, exit_whistle_04, take_block_stairs_06, walk_axis, walk_east_from_05
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

def dump(env):
    s = read_snapshot(env.get_ram())
    objs = [{"t": f"0x{o.type_id:02x}", "hp": o.hp, "xy": [o.x, o.y]} for o in s.objects if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF)]
    return {
        "sc": f"0x{s.screen:02x}", "mode": s.mode, "xy": [s.link_x, s.link_y],
        "doors": int(s.cur_opened_doors), "mask": int(s.open_doorway_mask),
        "keys": int(s.keys), "bombs": int(s.bombs),
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "alldead": int(s.room_all_dead), "item": int(s.room_item_id),
        "objs": objs,
    }

def try_door(env, assist, n, d, ax, ay):
    room0 = read_snapshot(env.get_ram()).screen
    walk_axis(env, assist, n, "y", ay, max_f=300)
    walk_axis(env, assist, n, "x", ax, max_f=300)
    push_dir(env, assist, n, d, frames=200)
    idle(env, assist, n, 12)
    s = read_snapshot(env.get_ram())
    return {"dir": d, "changed": s.screen != room0, **dump(env)}

def main():
    configure_headless()
    env = make_env(GAME, "Level5Whistle", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    n = [1]
    log = []
    try:
        reset_obs(env); env.step(nes_idle_action()); assist.apply_env(env, frame=0)
        idle(env, assist, n, 8)
        exit_whistle_04(env, assist, n)
        walk_east_from_05(env, assist, n)
        take_block_stairs_06(env, assist, n)
        cellar_07_to_64(env, assist, n)
        log.append({"tag": "at64", **dump(env)})
        print("AT64", log[-1], flush=True)
        # east to 65
        walk_axis(env, assist, n, "y", 189, max_f=400)
        walk_axis(env, assist, n, "x", 208, max_f=400)
        walk_axis(env, assist, n, "y", 141, max_f=300)
        walk_axis(env, assist, n, "x", 224, max_f=200)
        push_dir(env, assist, n, "RIGHT", frames=240)
        idle(env, assist, n, 16)
        log.append({"tag": "at65", **dump(env)})
        print("AT65", log[-1], flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_65_from64.png")
        for d, ax, ay in (("UP", 120, 93), ("RIGHT", 224, 141), ("DOWN", 120, 205), ("LEFT", 32, 141)):
            rec = try_door(env, assist, n, d, ax, ay)
            log.append(rec)
            print("DOOR", rec, flush=True)
            if rec["changed"]:
                # go back if we can
                back = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}[d]
                bx, by = {"UP": (120, 205), "DOWN": (120, 93), "LEFT": (224, 141), "RIGHT": (32, 141)}[d]
                rec2 = try_door(env, assist, n, back, bx, by)
                log.append({"tag": "back", **rec2})
                print("BACK", rec2, flush=True)
        write_json_report(RECORDINGS_DIR / "l5_65_from64.json", {"log": log, "pokes": False})
    finally:
        env.close()

if __name__ == "__main__":
    main()

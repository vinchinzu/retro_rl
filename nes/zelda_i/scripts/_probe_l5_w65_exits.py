"""From Level5Whistle64: east 0x65, dump, try north (shutter/bomb/block), else east 0x66."""
from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.level5_path import select_b_item_menu, walk_axis, walk_east_from_64
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8
from zelda_i.scripts._probe_l5_whistle_path import rom_room

STATE = "Level5Whistle64"


def step(env, assist, total, a):
    env.step(a)
    total[0] += 1
    assist.apply_env(env, frame=total[0])


def dump(env):
    s = read_snapshot(env.get_ram())
    objs = [{"t": o.type_id, "th": f"0x{o.type_id:02x}", "hp": o.hp, "x": o.x, "y": o.y}
            for o in s.objects if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF)]
    return {
        "sc": f"0x{s.screen:02x}",
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "tile": int(s.colliding_tile),
        "tile_h": f"0x{int(s.colliding_tile):02x}",
        "doors": int(s.cur_opened_doors),
        "mask": int(s.open_doorway_mask),
        "item": int(s.room_item_id),
        "all_dead": int(s.room_all_dead),
        "keys": int(s.keys),
        "bombs": int(s.bombs),
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "objs": objs,
    }


def main():
    configure_headless()
    env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    total = [1]
    log = []
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 12)
        print("START", dump(env), flush=True)
        print("ROM65", rom_room(0x65), flush=True)
        print("ROM55", rom_room(0x55), flush=True)
        print("ROM66", rom_room(0x66), flush=True)
        print("ROM56", rom_room(0x56), flush=True)
        east = walk_east_from_64(env, assist, total)
        print("EAST64", east, dump(env), flush=True)
        log.append({"tag": "arrive65", **dump(env)})
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w65_arrive.png")

        # North approaches
        for name, steps in (
            ("y109_x120_up", (("y", 109), ("x", 120), ("y", 93))),
            ("y189_x120_up", (("y", 189), ("x", 120), ("y", 93))),
        ):
            if read_snapshot(env.get_ram()).screen != 0x65:
                break
            for axis, tgt in steps:
                walk_axis(env, assist, total, axis, tgt, max_f=400)
            rec = {"tag": name, **dump(env)}
            log.append(rec)
            print("NAV", rec, flush=True)
            room0 = read_snapshot(env.get_ram()).screen
            push_dir(env, assist, total, "UP", frames=200)
            idle(env, assist, total, 12)
            rec = {"tag": f"{name}_push", **dump(env)}
            log.append(rec)
            print("PUSH_UP", rec, flush=True)
            if read_snapshot(env.get_ram()).screen != room0:
                break

        # Push any 0x68
        s = read_snapshot(env.get_ram())
        blocks = [o for o in s.objects if 1 <= o.slot <= 12 and o.type_id == 0x68]
        print("BLOCKS", [{"x": o.x, "y": o.y} for o in blocks], flush=True)
        if blocks and read_snapshot(env.get_ram()).screen == 0x65:
            b = blocks[0]
            walk_axis(env, assist, total, "y", b.y + 16, max_f=300)
            walk_axis(env, assist, total, "x", b.x, max_f=300)
            for d in ("UP", "LEFT", "RIGHT", "DOWN"):
                push_dir(env, assist, total, d, frames=80)
                rec = {"tag": f"push68_{d}", **dump(env)}
                log.append(rec)
                print("PUSH68", rec, flush=True)
                if read_snapshot(env.get_ram()).screen != 0x65:
                    break

        # Bomb north (natural, not a poke)
        if read_snapshot(env.get_ram()).screen == 0x65:
            walk_axis(env, assist, total, "y", 109, max_f=300)
            walk_axis(env, assist, total, "x", 120, max_f=300)
            walk_axis(env, assist, total, "y", 93, max_f=200)
            menu = select_b_item_menu(env, assist, total, 1)
            print("MENU", menu, dump(env), flush=True)
            bombs0 = read_snapshot(env.get_ram()).bombs
            step(env, assist, total, nes_action("UP", "B"))
            for _ in range(20):
                step(env, assist, total, nes_action("DOWN"))
            idle(env, assist, total, 120)
            rec = {"tag": "bomb_north", "bombs_in": bombs0, **dump(env)}
            log.append(rec)
            print("BOMB_N", rec, flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w65_bomb_n.png")
            push_dir(env, assist, total, "UP", frames=240)
            idle(env, assist, total, 16)
            rec = {"tag": "after_bomb_n", **dump(env)}
            log.append(rec)
            print("AFTER_BOMB_N", rec, flush=True)

        # East bomb hole to 0x66 (already opened on this lineage)
        if read_snapshot(env.get_ram()).screen == 0x65:
            walk_axis(env, assist, total, "y", 141, max_f=400)
            walk_axis(env, assist, total, "x", 224, max_f=500)
            push_dir(env, assist, total, "RIGHT", frames=240)
            idle(env, assist, total, 16)
            rec = {"tag": "east66", **dump(env)}
            log.append(rec)
            print("EAST66", rec, flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w65_east66.png")

        body = {"pokes": False, "status_claim": None, "rom65": rom_room(0x65), "rom55": rom_room(0x55), "log": log, "final": dump(env)}
        write_json_report(RECORDINGS_DIR / "l5_w65_exits.json", body)
        print("FINAL", body["final"], flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()

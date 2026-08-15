"""From Level5Whistle24: walk mid, play recorder once, dump shrink, sword leftovers."""
from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_path import select_b_item_menu, walk_axis
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_HEALTH, ADDR_SELECTED_ITEM, ADDR_TRIFORCE, ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Whistle24"


def step(env, assist, total, a):
    env.step(a)
    total[0] += 1
    assist.apply_env(env, frame=total[0])


def objs(snap):
    return [
        {"t": hex(o.type_id), "hp": o.hp, "xy": [o.x, o.y], "st": o.state}
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF)
    ]


def dump(env):
    s = read_snapshot(env.get_ram())
    ram = env.get_ram()
    return {
        "sc": hex(s.screen),
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "sel": int(read_u8(ram, ADDR_SELECTED_ITEM)),
        "whistle": int(read_u8(ram, ADDR_WHISTLE)),
        "tf": int(read_u8(ram, ADDR_TRIFORCE)),
        "item": s.room_item_id,
        "doors": int(s.cur_opened_doors),
        "health": int(s.health),
        "objs": objs(s),
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
        idle(env, assist, total, 16)
        log.append({"tag": "start", **dump(env)})
        print("START", log[-1], flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_24w_start.png")

        walk_axis(env, assist, total, "x", 160, max_f=300)
        walk_axis(env, assist, total, "y", 141, max_f=200)
        walk_axis(env, assist, total, "x", 120, max_f=300)
        idle(env, assist, total, 40)
        log.append({"tag": "mid", **dump(env)})
        print("MID", log[-1], flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_24w_mid.png")

        menu = select_b_item_menu(env, assist, total, 5)
        print("MENU", menu, flush=True)
        idle(env, assist, total, 20)
        step(env, assist, total, nes_action("B"))
        for i in range(12):
            idle(env, assist, total, 20)
            rec = {"tag": f"song_{i}", **dump(env)}
            log.append(rec)
            print("SONG", i, rec["objs"], "sel", rec["sel"], flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_24w_after_song.png")

        # Sword spam toward live foes
        for n in range(4000):
            s = read_snapshot(env.get_ram())
            live = [o for o in s.objects if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF, 0x55, 0x4E, 0x40, 0x68) and o.hp > 0]
            if not live:
                print("CLEAR", n, dump(env), flush=True)
                break
            tgt = live[0]
            if abs(s.link_x - tgt.x) > 6:
                step(env, assist, total, nes_action("RIGHT" if s.link_x < tgt.x else "LEFT"))
            elif abs(s.link_y - tgt.y) > 6:
                step(env, assist, total, nes_action("DOWN" if s.link_y < tgt.y else "UP"))
            else:
                step(env, assist, total, nes_action("A"))
            if n % 200 == 0:
                print("FIGHT", n, [(hex(o.type_id), o.hp) for o in live], [s.link_x, s.link_y], flush=True)
        idle(env, assist, total, 20)
        log.append({"tag": "after_fight", **dump(env)})
        print("AFTER_FIGHT", log[-1], flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_24w_after_fight.png")

        write_json_report(RECORDINGS_DIR / "l5_24w_fight.json", {"log": log, "final": dump(env), "pokes": False})
        print("FINAL", dump(env), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()

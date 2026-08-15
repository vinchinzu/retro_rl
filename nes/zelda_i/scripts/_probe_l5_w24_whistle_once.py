"""Play Recorder ONCE from center, wait full song, dump Digdogger."""
from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.level5_path import select_b_item_menu, walk_axis
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_SELECTED_ITEM, ADDR_WHISTLE, read_snapshot, read_u8


def dump_boss(env, tag):
    s = read_snapshot(env.get_ram())
    objs = [
        {"t": f"0x{o.type_id:02x}", "hp": o.hp, "x": o.x, "y": o.y, "st": o.state}
        for o in s.objects
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF)
    ]
    rec = {
        "tag": tag,
        "sc": f"0x{s.screen:02x}",
        "xy": [s.link_x, s.link_y],
        "sel": int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM)),
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "mode": s.mode,
        "item": s.room_item_id,
        "all_dead": s.room_all_dead,
        "objs": objs,
    }
    print(tag, rec, flush=True)
    return rec


def main():
    configure_headless()
    env = make_env(GAME, "Level5Whistle24", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    total = [1]
    log = []
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 16)
        log.append(dump_boss(env, "start"))
        menu = select_b_item_menu(env, assist, total, 5)
        print("MENU", menu, flush=True)
        idle(env, assist, total, 20)
        walk_axis(env, assist, total, "y", 141, max_f=300)
        walk_axis(env, assist, total, "x", 120, max_f=400)
        idle(env, assist, total, 16)
        log.append(dump_boss(env, "center"))
        # Hold B 16 frames to start the song.
        for _ in range(16):
            env.step(nes_action("B"))
            total[0] += 1
            assist.apply_env(env, frame=total[0])
        log.append(dump_boss(env, "held_b"))
        # Full recorder song ~2s; wait 240f and sample every 60.
        for i in range(5):
            idle(env, assist, total, 60)
            log.append(dump_boss(env, f"song_{i}"))
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w24_song_done.png")
        write_json_report(RECORDINGS_DIR / "l5_w24_whistle_once.json", {"menu": menu, "log": log})
    finally:
        env.close()


if __name__ == "__main__":
    main()

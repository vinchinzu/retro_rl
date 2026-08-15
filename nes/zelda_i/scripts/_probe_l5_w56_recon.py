"""Recon Level5Whistle56: objects + east hop to 0x57."""
from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.level5_path import walk_axis
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8


def dump(env):
    s = read_snapshot(env.get_ram())
    objs = [
        {"slot": o.slot, "t": f"0x{o.type_id:02x}", "hp": o.hp, "x": o.x, "y": o.y}
        for o in s.objects
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF)
    ]
    return {
        "sc": f"0x{s.screen:02x}",
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "doors": int(s.cur_opened_doors),
        "mask": int(s.open_doorway_mask),
        "all_dead": s.room_all_dead,
        "item": s.room_item_id,
        "keys": int(s.keys),
        "bombs": int(s.bombs),
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "objs": objs,
    }


def main():
    configure_headless()
    env = make_env(GAME, "Level5Whistle56", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    total = [1]
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 16)
        start = dump(env)
        print("START", start, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w56_start.png")
        # diamond-safe east
        for axis, tgt in (("y", 141), ("x", 208), ("y", 141), ("x", 224)):
            ok = walk_axis(env, assist, total, axis, tgt, max_f=400)
            s = read_snapshot(env.get_ram())
            print(f"WALK {axis}={tgt} ok={ok} xy=({s.link_x},{s.link_y}) sc=0x{s.screen:02x}", flush=True)
        push_dir(env, assist, total, "RIGHT", frames=280)
        idle(env, assist, total, 20)
        end = dump(env)
        print("EAST", end, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w56_east.png")
        write_json_report(RECORDINGS_DIR / "l5_w56_recon.json", {"start": start, "end": end})
    finally:
        env.close()


if __name__ == "__main__":
    main()

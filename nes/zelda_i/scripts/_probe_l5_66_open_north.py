"""Dump 0x66 from Whistle76; try block pushes and other north stands."""
from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.level5_path import walk_axis, _step
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Whistle76"


def dump(env):
    s = read_snapshot(env.get_ram())
    return {
        "sc": f"0x{s.screen:02x}",
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "doors": int(s.cur_opened_doors),
        "mask": int(s.open_doorway_mask),
        "item": s.room_item_id,
        "all_dead": s.room_all_dead,
        "blocks": [
            {"x": o.x, "y": o.y, "st": o.state, "t": o.type_id}
            for o in s.objects
            if 1 <= o.slot <= 12 and o.type_id in (0x68, 0x70, 0x71, 0x72, 0x73)
        ],
        "objs": [
            {"t": o.type_id, "hp": o.hp, "x": o.x, "y": o.y}
            for o in s.objects
            if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF)
        ],
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
        walk_axis(env, assist, total, "x", 120, max_f=200)
        push_dir(env, assist, total, "UP", frames=200)
        idle(env, assist, total, 20)
        log.append({"tag": "in66", **dump(env)})
        print("IN66", log[-1], flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w66_dump.png")

        # Push any 0x68
        snap = read_snapshot(env.get_ram())
        blocks = [o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id == 0x68]
        for b in blocks:
            walk_axis(env, assist, total, "y", b.y + 16, max_f=300)
            walk_axis(env, assist, total, "x", b.x, max_f=300)
            push_dir(env, assist, total, "UP", frames=100)
            idle(env, assist, total, 8)
            walk_axis(env, assist, total, "y", b.y, max_f=200)
            walk_axis(env, assist, total, "x", b.x + 16, max_f=200)
            push_dir(env, assist, total, "LEFT", frames=100)
            idle(env, assist, total, 8)
            rec = {"tag": "pushed", "at": [b.x, b.y], **dump(env)}
            log.append(rec)
            print("PUSH", rec, flush=True)

        # Try north at several x
        for tx in (56, 80, 96, 120, 144, 160, 176):
            if read_snapshot(env.get_ram()).screen != 0x66:
                break
            walk_axis(env, assist, total, "y", 109, max_f=300)
            walk_axis(env, assist, total, "x", tx, max_f=300)
            walk_axis(env, assist, total, "y", 93, max_f=200)
            room0 = read_snapshot(env.get_ram()).screen
            push_dir(env, assist, total, "UP", frames=160)
            idle(env, assist, total, 10)
            s = read_snapshot(env.get_ram())
            rec = {"tag": "north", "tx": tx, "changed": s.screen != room0, **dump(env)}
            log.append(rec)
            print("NORTH", rec, flush=True)
            if s.screen != room0:
                break

        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w66_north_try.png")
        write_json_report(RECORDINGS_DIR / "l5_w66_open_north.json", {"log": log, "final": dump(env), "pokes": False})
        print("FINAL", dump(env), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()

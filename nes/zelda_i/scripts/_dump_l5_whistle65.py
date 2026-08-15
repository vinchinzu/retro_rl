"""Dump Level5Whistle65 RAM pin. No pokes. No route."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.dungeon_ids import object_name
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_TRIFORCE, ADDR_WHISTLE, read_snapshot, read_u8

STATE = "Level5Whistle65"


def dump(env):
    ram = env.get_ram()
    s = read_snapshot(ram)
    objs = []
    for o in s.objects:
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF):
            objs.append({
                "slot": o.slot,
                "type": o.type_id,
                "type_hex": f"0x{o.type_id:02x}",
                "name": object_name(o.type_id),
                "hp": o.hp,
                "x": o.x,
                "y": o.y,
                "state": o.state,
            })
    return {
        "state": STATE,
        "mode": s.mode,
        "level": s.level,
        "screen": s.screen,
        "screen_hex": f"0x{s.screen:02x}",
        "x": s.link_x,
        "y": s.link_y,
        "keys": s.keys,
        "bombs": s.bombs,
        "doors": int(s.cur_opened_doors),
        "doors_hex": hex(int(s.cur_opened_doors)),
        "mask": int(s.open_doorway_mask),
        "mask_hex": hex(int(s.open_doorway_mask)),
        "all_dead": int(s.room_all_dead),
        "room_item": int(s.room_item_id),
        "room_obj_count": int(s.room_obj_count),
        "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
        "triforce_0x0671": int(read_u8(ram, ADDR_TRIFORCE)),
        "tf_l5_bit": bool(int(read_u8(ram, ADDR_TRIFORCE)) & 0x10),
        "objects": objs,
    }


def main():
    configure_headless()
    env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        rec = dump(env)
        png = RECORDINGS_DIR / "l5_whistle65_dump.png"
        save_rgb_png(env.step(nes_idle_action())[0], png)
        rec["png"] = str(png)
        path = RECORDINGS_DIR / "l5_whistle65_dump.json"
        write_json_report(path, rec)
        print("DUMP65", rec, flush=True)
        print("wrote", path, flush=True)
        return 0 if rec["screen"] == 0x65 and rec["whistle_0x065C"] == 1 else 2
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())

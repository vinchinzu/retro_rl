"""Dump 0x06 from Level5Whistle05 (after east) and Level5Entered06 spawn."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import exit_door, idle
from zelda_i.level9_stairs import dest_report, on_stair_tile, on_warp_tile
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot, read_u8, ADDR_WHISTLE


def step(env, assist, total, a):
    env.step(a)
    total[0] += 1
    assist.apply_env(env, frame=total[0])


def dump(env):
    s = read_snapshot(env.get_ram())
    objs = []
    for o in s.objects:
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF):
            objs.append({"slot": o.slot, "type": o.type_id, "hp": o.hp, "x": o.x, "y": o.y})
    return {
        "sc": f"0x{s.screen:02x}",
        "next": f"0x{s.next_screen:02x}",
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "tile": int(s.colliding_tile),
        "stair": on_stair_tile(s),
        "warp": on_warp_tile(s),
        "item": s.room_item_id,
        "doors": s.cur_opened_doors,
        "mask": s.open_doorway_mask,
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "keys": s.keys,
        "objs": objs,
        "dest": dest_report(s),
    }


def main():
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    # Spawn reference
    env = make_env(GAME, "Level5Entered06", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    total = [1]
    reset_obs(env)
    env.step(nes_idle_action())
    assist.apply_env(env, frame=0)
    idle(env, assist, total, 12)
    entered = dump(env)
    print("ENTERED06", entered, flush=True)
    save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_entered_ref.png")
    env.close()

    env = make_env(GAME, "Level5Whistle05", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    total = [1]
    reset_obs(env)
    env.step(nes_idle_action())
    assist.apply_env(env, frame=0)
    idle(env, assist, total, 12)
    print("START05", dump(env), flush=True)
    rec = exit_door(env, assist, total, "RIGHT")
    idle(env, assist, total, 16)
    arrive = dump(env)
    print("ARRIVE06", arrive, flush=True)
    save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_from05.png")

    log = []
    # Sweep likely stair / gap tiles
    stands = [
        (120, 141), (120, 125), (120, 157), (80, 141), (160, 141),
        (64, 141), (176, 141), (120, 109), (120, 173), (120, 189),
        (120, 93), (40, 141), (200, 141), (96, 141), (144, 141),
        (80, 109), (160, 109), (80, 173), (160, 173), (64, 189),
        (176, 189), (120, 80), (88, 144), (152, 144),
    ]
    for tx, ty in stands:
        # walk
        for _ in range(400):
            s = read_snapshot(env.get_ram())
            if s.mode != PLAY_MODE or s.screen != 0x06:
                break
            if abs(s.link_x - tx) <= 2 and abs(s.link_y - ty) <= 2:
                break
            if abs(s.link_x - tx) > 2:
                step(env, assist, total, nes_action("RIGHT" if s.link_x < tx else "LEFT"))
            else:
                step(env, assist, total, nes_action("DOWN" if s.link_y < ty else "UP"))
        idle(env, assist, total, 4)
        d = dump(env)
        rec = {"stand": [tx, ty], **d}
        log.append(rec)
        print("STAND", rec["stand"], "xy", rec["xy"], "tile", rec["tile"], "stair", rec["stair"], "mode", rec["mode"], "sc", rec["sc"], flush=True)
        if d["mode"] != 5 or d["sc"] != "0x06":
            break

    # Nudge on last few
    final = dump(env)
    save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_stairs_hunt.png")
    write_json_report(RECORDINGS_DIR / "l5_06_stairs.json", {
        "entered06_ref": entered,
        "arrive": arrive,
        "log": log,
        "final": final,
        "pokes": False,
    })
    print("FINAL", final, flush=True)
    env.close()


if __name__ == "__main__":
    main()

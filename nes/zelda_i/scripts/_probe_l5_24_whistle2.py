"""Play recorder from doorway stands; hold B; do not move during the song."""
from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle
from zelda_i.level5_path import select_b_item_menu, walk_axis
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_SELECTED_ITEM, ADDR_WHISTLE, read_snapshot, read_u8

STATE = "Level5Whistle24"


def step(env, assist, total, a):
    env.step(a)
    total[0] += 1
    assist.apply_env(env, frame=total[0])


def live_boss(snap):
    return [
        {"t": hex(o.type_id), "hp": o.hp, "xy": [o.x, o.y]}
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id in (0x38, 0x39, 0x18) and (o.hp > 0 or o.type_id == 0x38)
    ]


def all_foes(snap):
    return [
        {"t": hex(o.type_id), "hp": o.hp, "xy": [o.x, o.y]}
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF, 0x55, 0x4E, 0x40, 0x68)
    ]


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
        menu = select_b_item_menu(env, assist, total, 5)
        print("MENU", menu, "sel", int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM)), flush=True)
        stands = ((224, 141), (120, 189), (32, 141), (120, 93), (160, 189), (80, 189))
        for tx, ty in stands:
            walk_axis(env, assist, total, "y", ty, max_f=300)
            walk_axis(env, assist, total, "x", tx, max_f=300)
            idle(env, assist, total, 8)
            s = read_snapshot(env.get_ram())
            before = {"stand": [tx, ty], "xy": [s.link_x, s.link_y], "boss": live_boss(s), "foes": all_foes(s)}
            print("STAND", before, flush=True)
            # Hold B 12 frames, then freeze.
            for _ in range(12):
                step(env, assist, total, nes_action("B"))
            for i in range(16):
                idle(env, assist, total, 12)
                s = read_snapshot(env.get_ram())
                foes = all_foes(s)
                n38 = sum(1 for o in foes if o["t"] == "0x38")
                print("  t", i, "xy", [s.link_x, s.link_y], "n38", n38, "foes", foes, flush=True)
                if n38 != 1 or any(o["hp"] not in (0, 240) for o in foes if o["t"] == "0x38") or len(foes) > 1:
                    print("SHRINK?", foes, flush=True)
                    save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / f"l5_24w_shrink_{tx}_{ty}.png")
                    log.append({"stand": [tx, ty], "shrink": True, "foes": foes})
                    write_json_report(RECORDINGS_DIR / "l5_24w_whistle2.json", {"log": log, "menu": menu})
                    return
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / f"l5_24w_try_{tx}_{ty}.png")
            log.append({"stand": [tx, ty], "shrink": False, "foes": all_foes(read_snapshot(env.get_ram()))})
        write_json_report(RECORDINGS_DIR / "l5_24w_whistle2.json", {"log": log, "menu": menu, "pokes": False})
        print("NO_SHRINK", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()

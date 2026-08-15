"""Level5Whistle24: whistle-shrink 0x38→0x18, sword, heart, north TF 0x10."""
from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_path import select_b_item_menu, walk_axis
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_HEALTH, ADDR_TRIFORCE, ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Whistle24"
TF_BIT = 0x10


def step(env, assist, total, a):
    env.step(a)
    total[0] += 1
    assist.apply_env(env, frame=total[0])


def foes(snap):
    return [
        o
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF, 0x55, 0x4E, 0x40, 0x68) and o.hp > 0
    ]


def main():
    configure_headless()
    env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    total = [1]
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 12)
        walk_axis(env, assist, total, "y", 141, max_f=200)
        walk_axis(env, assist, total, "x", 120, max_f=400)
        idle(env, assist, total, 8)
        menu = select_b_item_menu(env, assist, total, 5)
        print("MENU", menu, flush=True)
        shrunk = False
        for attempt in range(4):
            for _ in range(12):
                step(env, assist, total, nes_action("B"))
            for i in range(20):
                idle(env, assist, total, 12)
                s = read_snapshot(env.get_ram())
                live = foes(s)
                types = [(hex(o.type_id), o.hp) for o in live]
                print("SONG", attempt, i, types, flush=True)
                if any(o.type_id == 0x18 for o in live) or (live and all(o.type_id != 0x38 for o in live)):
                    shrunk = True
                    break
            if shrunk:
                break
        print("SHRUNK", shrunk, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_24_shrunk.png")

        for n in range(8000):
            s = read_snapshot(env.get_ram())
            live = foes(s)
            if not live:
                print("DEAD", n, flush=True)
                break
            tgt = min(live, key=lambda o: abs(o.x - s.link_x) + abs(o.y - s.link_y))
            dx, dy = tgt.x - s.link_x, tgt.y - s.link_y
            if abs(dx) > 10:
                step(env, assist, total, nes_action("RIGHT" if dx > 0 else "LEFT"))
            elif abs(dy) > 10:
                step(env, assist, total, nes_action("DOWN" if dy > 0 else "UP"))
            else:
                step(env, assist, total, nes_action("A"))
            if n % 250 == 0:
                print("FIGHT", n, [(hex(o.type_id), o.hp, o.x, o.y) for o in live], [s.link_x, s.link_y], flush=True)
        idle(env, assist, total, 30)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_24_dead.png")

        hc0 = ((int(read_u8(env.get_ram(), ADDR_HEALTH)) >> 4) & 0x0F) + 1
        for tx, ty in ((120, 141), (120, 125), (96, 141), (144, 141), (120, 157), (80, 141), (160, 141), (224, 141)):
            walk_axis(env, assist, total, "y", ty, max_f=200)
            walk_axis(env, assist, total, "x", tx, max_f=200)
            idle(env, assist, total, 8)
            hc1 = ((int(read_u8(env.get_ram(), ADDR_HEALTH)) >> 4) & 0x0F) + 1
            if hc1 > hc0:
                print("HEART", hc0, "->", hc1, flush=True)
                break
        hc1 = ((int(read_u8(env.get_ram(), ADDR_HEALTH)) >> 4) & 0x0F) + 1
        print("HC", hc0, hc1, flush=True)

        walk_axis(env, assist, total, "y", 141, max_f=300)
        walk_axis(env, assist, total, "x", 120, max_f=300)
        walk_axis(env, assist, total, "y", 93, max_f=400)
        push_dir(env, assist, total, "UP", frames=260)
        idle(env, assist, total, 20)
        for _ in range(200):
            s = read_snapshot(env.get_ram())
            if s.mode == PLAY_MODE and s.screen == 0x14:
                break
            step(env, assist, total, nes_idle_action())
        s = read_snapshot(env.get_ram())
        print("NORTH", hex(s.screen), [s.link_x, s.link_y], "item", s.room_item_id, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_14_arrive.png")

        tf0 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
        for tx, ty in ((120, 141), (120, 125), (120, 157), (96, 141), (144, 141), (120, 109), (80, 141), (160, 141)):
            walk_axis(env, assist, total, "y", ty, max_f=200)
            walk_axis(env, assist, total, "x", tx, max_f=200)
            idle(env, assist, total, 10)
            if int(read_u8(env.get_ram(), ADDR_TRIFORCE)) > tf0:
                break
        for _ in range(500):
            if int(read_u8(env.get_ram(), ADDR_TRIFORCE)) & TF_BIT:
                break
            step(env, assist, total, nes_idle_action())
        ram = env.get_ram()
        s = read_snapshot(ram)
        tf1 = int(read_u8(ram, ADDR_TRIFORCE))
        rec = {
            "ok": bool(tf1 & TF_BIT),
            "shrunk": shrunk,
            "hc_in": hc0,
            "hc_out": hc1,
            "tf_in": tf0,
            "tf_out": tf1,
            "tf_l5": bool(tf1 & TF_BIT),
            "room": s.screen,
            "xy": [s.link_x, s.link_y],
            "whistle": int(read_u8(ram, ADDR_WHISTLE)),
            "item": s.room_item_id,
            "pokes": False,
            "status_claim": None,
        }
        print("TF", rec, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_tf_final.png")
        if rec["ok"]:
            write_state_bytes(state_path(GAME_DIR, GAME, "Level5Complete"), env.em.get_state())
            write_state_provenance(
                state_path(GAME_DIR, GAME, "Level5Complete"),
                source_state_path=GAME_DIR / "custom_integrations" / GAME / f"{STATE}.state",
                request={
                    "segment": "Level5Complete",
                    "predecessor_entry": True,
                    "start_state": STATE,
                    "via": "whistle shrink 0x38->0x18, sword, heart, north TF 0x10",
                    "key_poke": False,
                    "door_poke": False,
                },
                selected_trial={"success": True, **rec},
                natural_entry=False,
            )
            print("PINNED Level5Complete", flush=True)
        write_json_report(RECORDINGS_DIR / "l5_24_whistle_boss.json", rec)
    finally:
        env.close()


if __name__ == "__main__":
    main()

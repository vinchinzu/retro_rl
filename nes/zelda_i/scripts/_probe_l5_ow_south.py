"""Leave L5 0x0B at x=112, then walk 0x1B south / east with long pushes."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ids import object_name
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_CANDLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5EntranceFromL4"


def open_env():
    env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    reset_obs(env)
    env.step(nes_idle_action())
    assist.apply_env(env, frame=0)
    return env, assist


def step(env, assist, total, action):
    env.step(action)
    total[0] += 1
    assist.apply_env(env, frame=total[0])


def walk_axis(env, assist, total, axis, target, max_f=400):
    last = None
    stall = 0
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        if axis == "x":
            if abs(snap.link_x - target) <= 2:
                return True
            step(env, assist, total, nes_action("RIGHT" if snap.link_x < target else "LEFT"))
        else:
            if abs(snap.link_y - target) <= 2:
                return True
            step(env, assist, total, nes_action("DOWN" if snap.link_y < target else "UP"))
        pos = (snap.link_x, snap.link_y)
        stall = stall + 1 if pos == last else 0
        last = pos
        if stall >= 45:
            return False
    return False


def wait_play(env, assist, total, max_f=200):
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE and not snap.transitioning:
            idle(env, assist, total, 10)
            return True
        step(env, assist, total, nes_idle_action())
    return False


def rec(env, tag):
    snap = read_snapshot(env.get_ram())
    return {
        "tag": tag,
        "L": snap.level,
        "sc": f"0x{snap.screen:02x}",
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "R": snap.rupees,
        "candle": int(read_u8(env.get_ram(), ADDR_CANDLE)),
        "objs": [
            {
                "t": f"0x{o.type_id:02x}",
                "n": object_name(o.type_id),
                "xy": [o.x, o.y],
                "hp": o.hp,
            }
            for o in snap.objects
            if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF)
        ],
    }


def main():
    configure_headless()
    env = None
    total = [1]
    log = []
    try:
        env, assist = open_env()
        idle(env, assist, total, 12)
        walk_axis(env, assist, total, "x", 112)
        walk_axis(env, assist, total, "y", 205)
        push_dir(env, assist, total, "DOWN", frames=300)
        wait_play(env, assist, total)
        log.append(rec(env, "after_l5_exit"))
        print(log[-1], flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_ow_s_0b.png")

        # 0x0B south stairs x=112
        walk_axis(env, assist, total, "x", 112)
        walk_axis(env, assist, total, "y", 205)
        sc0 = read_snapshot(env.get_ram()).screen
        push_dir(env, assist, total, "DOWN", frames=280)
        wait_play(env, assist, total)
        log.append(rec(env, "0b_south"))
        print(log[-1], flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_ow_s_1b.png")

        snap = read_snapshot(env.get_ram())
        if snap.screen == 0x1B:
            # try south first (skip east pocket)
            walk_axis(env, assist, total, "x", 112)
            walk_axis(env, assist, total, "y", 205)
            push_dir(env, assist, total, "DOWN", frames=280)
            wait_play(env, assist, total)
            log.append(rec(env, "1b_south_x112"))
            print(log[-1], flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_ow_s_1b_south.png")

        snap = read_snapshot(env.get_ram())
        if snap.screen == 0x1B:
            for y in (172, 189, 141, 109):
                walk_axis(env, assist, total, "y", y)
                walk_axis(env, assist, total, "x", 224)
                push_dir(env, assist, total, "RIGHT", frames=260)
                wait_play(env, assist, total)
                log.append(rec(env, f"1b_east_y{y}"))
                print(log[-1], flush=True)
                save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / f"l5_ow_s_1b_e{y}.png")
                if read_snapshot(env.get_ram()).screen != 0x1B:
                    break

        snap = read_snapshot(env.get_ram())
        if snap.screen == 0x1B:
            walk_axis(env, assist, total, "x", 48)
            walk_axis(env, assist, total, "y", 205)
            push_dir(env, assist, total, "DOWN", frames=280)
            wait_play(env, assist, total)
            log.append(rec(env, "1b_south_x48"))
            print(log[-1], flush=True)

        body = {"log": log, "pokes": False, "status_claim": None, "final": rec(env, "final")}
        write_json_report(RECORDINGS_DIR / "l5_ow_south.json", body)
        print("FINAL", body["final"], flush=True)
        return body
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

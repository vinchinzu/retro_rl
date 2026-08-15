"""Probe 0x0B Armos farm + 0x1B exits after L5 leave. No pokes."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ids import object_name
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.nav_common import swing_action
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_CANDLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5EntranceFromL4"


def open_env():
    env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    obs, _ = reset_obs(env)
    obs, *_ = env.step(nes_idle_action())
    assist.apply_env(env, frame=0)
    return env, assist


def step(env, assist, total, action):
    obs, *_ = env.step(action)
    total[0] += 1
    assist.apply_env(env, frame=total[0])
    return obs


def objs(snap):
    out = []
    for o in snap.objects:
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF):
            out.append(
                {
                    "slot": o.slot,
                    "t": f"0x{o.type_id:02x}",
                    "name": object_name(o.type_id),
                    "xy": [o.x, o.y],
                    "hp": o.hp,
                    "st": o.state,
                }
            )
    return out


def walk_axis(env, assist, total, axis, target, max_f=300):
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
        if stall >= 40:
            return False
    return False


def main():
    configure_headless()
    env = None
    total = [1]
    try:
        env, assist = open_env()
        idle(env, assist, total, 16)
        # leave L5
        walk_axis(env, assist, total, "x", 120)
        walk_axis(env, assist, total, "y", 205)
        push_dir(env, assist, total, "DOWN", frames=280)
        for _ in range(240):
            snap = read_snapshot(env.get_ram())
            if snap.level == 0 and snap.mode == PLAY_MODE and not snap.transitioning:
                idle(env, assist, total, 16)
                break
            step(env, assist, total, nes_idle_action())
        snap = read_snapshot(env.get_ram())
        print("ON_OW", hex(snap.screen), snap.mode, [snap.link_x, snap.link_y], "R", snap.rupees, flush=True)
        print("OBJS0", objs(snap), flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_ow_0b_armos.png")

        # Touch each Armos-like statue: walk onto common stands
        stands = [
            (64, 109), (112, 109), (160, 109), (208, 109),
            (64, 157), (112, 157), (160, 157), (208, 157),
            (80, 125), (144, 125), (80, 173), (144, 173),
        ]
        rupee_log = []
        r0 = snap.rupees
        for tx, ty in stands:
            walk_axis(env, assist, total, "y", ty, max_f=180)
            walk_axis(env, assist, total, "x", tx, max_f=180)
            for _ in range(40):
                act = swing_action(total[0], "UP", "wake", period=8, hold=3)
                step(env, assist, total, act.action)
            snap = read_snapshot(env.get_ram())
            rec = {
                "stand": [tx, ty],
                "xy": [snap.link_x, snap.link_y],
                "R": snap.rupees,
                "objs": objs(snap),
                "tile": int(snap.colliding_tile),
                "mode": snap.mode,
                "sc": f"0x{snap.screen:02x}",
            }
            rupee_log.append(rec)
            print("ARMOS", rec["stand"], "R", rec["R"], "nobj", len(rec["objs"]), rec["objs"][:4], flush=True)
            if snap.rupees > r0:
                print("RUPEE_GAIN", r0, "->", snap.rupees, flush=True)
                r0 = snap.rupees

        # chase anything alive for a bit
        for i in range(800):
            snap = read_snapshot(env.get_ram())
            if snap.rupees >= 60:
                break
            live = [
                o
                for o in snap.objects
                if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF) and 20 < o.y < 220
            ]
            if not live:
                break
            t = min(live, key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y))
            dx, dy = t.x - snap.link_x, t.y - snap.link_y
            d = "RIGHT" if abs(dx) >= abs(dy) and dx >= 0 else "LEFT" if abs(dx) >= abs(dy) else "DOWN" if dy >= 0 else "UP"
            act = swing_action(total[0], d, "chase", period=8, hold=3)
            step(env, assist, total, act.action)
            if i % 80 == 0:
                print("CHASE", i, "R", snap.rupees, "n", len(live), "t", hex(t.type_id), flush=True)
        snap = read_snapshot(env.get_ram())
        print("AFTER_ARMOS R", snap.rupees, "sc", hex(snap.screen), objs(snap), flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_ow_0b_after_armos.png")

        # Go south to 0x1B
        walk_axis(env, assist, total, "x", 120, max_f=250)
        walk_axis(env, assist, total, "y", 205, max_f=250)
        push_dir(env, assist, total, "DOWN", frames=220)
        for _ in range(120):
            snap = read_snapshot(env.get_ram())
            if snap.screen != 0x0B and snap.mode == PLAY_MODE and not snap.transitioning:
                idle(env, assist, total, 12)
                break
            step(env, assist, total, nes_idle_action())
        snap = read_snapshot(env.get_ram())
        print("AT_1B?", hex(snap.screen), [snap.link_x, snap.link_y], "R", snap.rupees, objs(snap), flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_ow_1b.png")

        # From 0x1B try exits at several y
        exits = []
        sc0 = snap.screen
        for direction, y, x in (
            ("RIGHT", 165, 220),
            ("RIGHT", 141, 220),
            ("RIGHT", 189, 220),
            ("DOWN", 205, 120),
            ("LEFT", 165, 32),
            ("LEFT", 141, 32),
        ):
            # reload-ish: if we left, we can't easily return; skip if left
            snap = read_snapshot(env.get_ram())
            if snap.screen != sc0:
                print("LEFT_1B already", hex(snap.screen), flush=True)
                break
            walk_axis(env, assist, total, "y", y, max_f=250)
            walk_axis(env, assist, total, "x", x, max_f=250)
            room0 = snap.screen
            push_dir(env, assist, total, direction, frames=200)
            idle(env, assist, total, 10)
            for _ in range(80):
                snap = read_snapshot(env.get_ram())
                if not snap.transitioning and snap.mode == PLAY_MODE:
                    break
                step(env, assist, total, nes_idle_action())
            snap = read_snapshot(env.get_ram())
            rec = {
                "try": f"{direction}_y{y}_x{x}",
                "dest": f"0x{snap.screen:02x}",
                "xy": [snap.link_x, snap.link_y],
                "changed": snap.screen != sc0,
            }
            exits.append(rec)
            print("EXIT1B", rec, flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / f"l5_ow_1b_{direction}_{y}.png")
            if snap.screen != sc0:
                break

        body = {
            "rupees": int(read_snapshot(env.get_ram()).rupees),
            "screen": f"0x{read_snapshot(env.get_ram()).screen:02x}",
            "candle": int(read_u8(env.get_ram(), ADDR_CANDLE)),
            "exits": exits,
            "pokes": False,
            "status_claim": None,
        }
        write_json_report(RECORDINGS_DIR / "l5_ow_nav.json", body)
        print("DONE", body, flush=True)
        return body
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

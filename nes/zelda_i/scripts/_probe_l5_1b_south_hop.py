"""Hop-controller SOUTH 0x1B -> 0x2B after leaving L5. Hold through scroll."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle
from zelda_i.overworld import ScreenHop
from zelda_i.ow_path import OverworldPathController
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot

STATE = "Level5EntranceFromL4"


def open_env():
    env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    reset_obs(env)
    env.step(nes_idle_action())
    assist.apply_env(env, frame=0)
    return env, assist


def step(env, assist, total, a):
    env.step(a)
    total[0] += 1
    assist.apply_env(env, frame=total[0])


def run_hops(env, assist, total, hops, label, max_f=6000):
    nav = OverworldPathController(hops=hops, require_sword=False, max_frames=max_f)
    trail = []
    last = None
    for i in range(max_f):
        snap = read_snapshot(env.get_ram())
        if snap.level == 5:
            step(env, assist, total, nes_action("DOWN"))
            continue
        key = (snap.level, snap.screen, snap.mode, nav.hop_index if hasattr(nav, "hop_index") else -1)
        if key != last:
            rec = {
                "i": i,
                "L": snap.level,
                "sc": f"0x{snap.screen:02x}",
                "mode": snap.mode,
                "xy": [snap.link_x, snap.link_y],
                "hop": nav.hop_index,
                "ok": nav.success,
            }
            trail.append(rec)
            print(label, rec, flush=True)
            last = key
        if nav.success or (hasattr(nav.phase, "name") and nav.phase.name == "FAILED"):
            break
        act = nav.step(snap)
        step(env, assist, total, act.action)
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE and not snap.transitioning and nav.success:
            break
    return trail, nav.report()


def main():
    configure_headless()
    env = None
    total = [1]
    try:
        env, assist = open_env()
        idle(env, assist, total, 10)
        t1, r1 = run_hops(env, assist, total, (ScreenHop(0x1B, "DOWN", align_x=112),), "TO1B")
        s = read_snapshot(env.get_ram())
        print("AT1B", hex(s.screen), [s.link_x, s.link_y], r1, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_1b_south_arrive.png")

        t2, r2 = run_hops(env, assist, total, (ScreenHop(0x2B, "DOWN", align_x=112),), "TO2B")
        s = read_snapshot(env.get_ram())
        print("AT2B", hex(s.screen), [s.link_x, s.link_y], r2, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_1b_south_2b.png")

        # If still 0x1B, try x=48 then x=80
        extra = []
        if s.screen == 0x1B:
            t3, r3 = run_hops(env, assist, total, (ScreenHop(0x2B, "DOWN", align_x=48),), "TO2B_x48")
            s = read_snapshot(env.get_ram())
            extra.append({"align": 48, "sc": hex(s.screen), "xy": [s.link_x, s.link_y], "nav": r3})
            print("AT2B48", extra[-1], flush=True)
        if s.screen == 0x1B:
            t4, r4 = run_hops(env, assist, total, (ScreenHop(0x2B, "DOWN", align_x=80),), "TO2B_x80")
            s = read_snapshot(env.get_ram())
            extra.append({"align": 80, "sc": hex(s.screen), "xy": [s.link_x, s.link_y], "nav": r4})
            print("AT2B80", extra[-1], flush=True)

        body = {
            "to_1b": r1,
            "to_2b": r2,
            "extra": extra,
            "final": {"sc": f"0x{s.screen:02x}", "xy": [s.link_x, s.link_y], "mode": s.mode, "R": s.rupees},
            "pokes": False,
        }
        write_json_report(RECORDINGS_DIR / "l5_1b_south_hop.json", body)
        print("FINAL", body["final"], flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

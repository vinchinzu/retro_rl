"""Stand on 0x1B east ledge (240,141) and hold RIGHT through scroll."""
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
WPS = (
    (112, 109), (112, 141), (144, 141), (176, 141), (176, 157), (176, 173), (192, 173), (208, 173), (208, 157), (208, 141), (224, 141), (240, 141),
)


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


def to_1b(env, assist, total):
    idle(env, assist, total, 10)
    nav = OverworldPathController(hops=(ScreenHop(0x1B, "DOWN", align_x=112),), require_sword=False, max_frames=4000)
    for _ in range(4000):
        snap = read_snapshot(env.get_ram())
        if snap.level == 5:
            step(env, assist, total, nes_action("DOWN"))
            continue
        act = nav.step(snap)
        step(env, assist, total, act.action)
        if read_snapshot(env.get_ram()).screen == 0x1B and nav.success:
            break
        if hasattr(nav.phase, "name") and nav.phase.name == "FAILED":
            break


def toward(env, assist, total, tx, ty, n=100):
    for _ in range(n):
        snap = read_snapshot(env.get_ram())
        if abs(snap.link_x - tx) <= 1 and abs(snap.link_y - ty) <= 1:
            return True
        if abs(snap.link_x - tx) > 1:
            step(env, assist, total, nes_action("RIGHT" if snap.link_x < tx else "LEFT"))
        else:
            step(env, assist, total, nes_action("DOWN" if snap.link_y < ty else "UP"))
    return False


def main():
    configure_headless()
    env = None
    total = [1]
    trail = []
    try:
        env, assist = open_env()
        to_1b(env, assist, total)
        for tx, ty in WPS:
            toward(env, assist, total, tx, ty)
            s = read_snapshot(env.get_ram())
            print("WP", [tx, ty], "at", [s.link_x, s.link_y], flush=True)
        s = read_snapshot(env.get_ram())
        print("LEDGE", [s.link_x, s.link_y], hex(s.screen), flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_1b_ledge.png")
        last = None
        for i in range(400):
            snap = read_snapshot(env.get_ram())
            key = (snap.level, snap.screen, snap.mode)
            if last != key:
                rec = {"i": i, "L": snap.level, "sc": f"0x{snap.screen:02x}", "mode": snap.mode, "xy": [snap.link_x, snap.link_y]}
                trail.append(rec)
                print("R", rec, flush=True)
                last = key
            step(env, assist, total, nes_action("RIGHT"))
        for _ in range(80):
            snap = read_snapshot(env.get_ram())
            if snap.mode == PLAY_MODE and not snap.transitioning:
                break
            step(env, assist, total, nes_action("RIGHT"))
        s = read_snapshot(env.get_ram())
        body = {"trail": trail, "final": {"sc": f"0x{s.screen:02x}", "xy": [s.link_x, s.link_y], "mode": s.mode, "R": s.rupees}, "pokes": False}
        write_json_report(RECORDINGS_DIR / "l5_1b_ledge.json", body)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_1b_ledge_after.png")
        print("FINAL", body["final"], flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

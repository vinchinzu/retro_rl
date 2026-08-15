"""Weave 0x1B rocks to the east-ledge gap (y≈172) then leave to 0x1C."""
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

# Dense waypoints through the 3x3 rock field toward the SE gap then east ledge.
WPS = (
    (112, 85),
    (112, 109),
    (112, 125),
    (80, 141),
    (112, 141),
    (144, 141),
    (176, 141),
    (176, 157),
    (176, 173),
    (144, 173),
    (112, 173),
    (80, 173),
    (48, 173),
    (48, 189),
    (80, 189),
    (112, 189),
    (144, 189),
    (176, 189),
    (192, 173),
    (208, 173),
    (208, 157),
    (208, 141),
    (224, 141),
    (224, 157),
    (224, 173),
    (224, 189),
    (240, 173),
    (240, 141),
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
    nav = OverworldPathController(
        hops=(ScreenHop(0x1B, "DOWN", align_x=112),),
        require_sword=False,
        max_frames=4000,
    )
    for _ in range(4000):
        snap = read_snapshot(env.get_ram())
        if snap.level == 5:
            step(env, assist, total, nes_action("DOWN"))
            continue
        act = nav.step(snap)
        step(env, assist, total, act.action)
        snap = read_snapshot(env.get_ram())
        if snap.screen == 0x1B and snap.mode == PLAY_MODE and nav.success:
            break
        if hasattr(nav.phase, "name") and nav.phase.name == "FAILED":
            break
    return read_snapshot(env.get_ram()).screen == 0x1B


def toward(env, assist, total, tx, ty, n=80):
    s0 = read_snapshot(env.get_ram())
    for _ in range(n):
        snap = read_snapshot(env.get_ram())
        if snap.screen != s0.screen:
            return snap
        if abs(snap.link_x - tx) <= 2 and abs(snap.link_y - ty) <= 2:
            return snap
        if abs(snap.link_x - tx) > 2:
            step(env, assist, total, nes_action("RIGHT" if snap.link_x < tx else "LEFT"))
        else:
            step(env, assist, total, nes_action("DOWN" if snap.link_y < ty else "UP"))
    return read_snapshot(env.get_ram())


def main():
    configure_headless()
    env = None
    total = [1]
    log = []
    try:
        env, assist = open_env()
        to_1b(env, assist, total)
        s = read_snapshot(env.get_ram())
        print("START", hex(s.screen), [s.link_x, s.link_y], flush=True)
        reached = []
        for tx, ty in WPS:
            snap = toward(env, assist, total, tx, ty, n=90)
            rec = {
                "want": [tx, ty],
                "xy": [snap.link_x, snap.link_y],
                "sc": f"0x{snap.screen:02x}",
                "hit": abs(snap.link_x - tx) <= 4 and abs(snap.link_y - ty) <= 4,
            }
            reached.append(rec)
            print("WP", rec, flush=True)
            if snap.screen != 0x1B:
                print("LEFT", rec, flush=True)
                break
        # from wherever we are, try RIGHT at current y and nearby
        s = read_snapshot(env.get_ram())
        if s.screen == 0x1B:
            for y in (s.link_y, 173, 141, 157, 189, 125):
                toward(env, assist, total, s.link_x, y, n=60)
                toward(env, assist, total, 240, y, n=80)
                for _ in range(200):
                    snap = read_snapshot(env.get_ram())
                    if snap.screen != 0x1B:
                        break
                    step(env, assist, total, nes_action("RIGHT"))
                idle(env, assist, total, 16)
                for _ in range(50):
                    snap = read_snapshot(env.get_ram())
                    if snap.mode == PLAY_MODE and not snap.transitioning:
                        break
                    step(env, assist, total, nes_idle_action())
                snap = read_snapshot(env.get_ram())
                rec = {"try_e_y": y, "sc": f"0x{snap.screen:02x}", "xy": [snap.link_x, snap.link_y]}
                log.append(rec)
                print("EAST", rec, flush=True)
                save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / f"l5_1b_gap_e{y}.png")
                if snap.screen != 0x1B:
                    break
        body = {
            "wps": reached,
            "east": log,
            "final": {
                "sc": f"0x{read_snapshot(env.get_ram()).screen:02x}",
                "xy": [
                    read_snapshot(env.get_ram()).link_x,
                    read_snapshot(env.get_ram()).link_y,
                ],
                "R": read_snapshot(env.get_ram()).rupees,
            },
            "pokes": False,
        }
        write_json_report(RECORDINGS_DIR / "l5_1b_gap.json", body)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_1b_gap_final.png")
        print("FINAL", body["final"], flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

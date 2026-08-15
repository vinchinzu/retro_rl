"""Escape Lost Hills 0x1B: west to 0x1A, or reverse-pocket east to 0x1C."""
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


def to_1b(env, assist, total):
    idle(env, assist, total, 12)
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
        if nav.success or (hasattr(nav.phase, "name") and nav.phase.name == "FAILED"):
            break
        if snap.level == 0 and snap.screen == 0x1B and snap.mode == PLAY_MODE:
            # still let controller finish settle
            pass
        act = nav.step(snap)
        step(env, assist, total, act.action)
        snap = read_snapshot(env.get_ram())
        if snap.screen == 0x1B and snap.mode == PLAY_MODE and not snap.transitioning and nav.success:
            break
    # if still 0x0B, force down via controller success anyway
    snap = read_snapshot(env.get_ram())
    print("AT", hex(snap.screen), [snap.link_x, snap.link_y], "nav", nav.report(), flush=True)
    return snap.screen == 0x1B


def hold(env, assist, total, d, n):
    for _ in range(n):
        step(env, assist, total, nes_action(d))


def go(env, assist, total, axis, tgt, n=300):
    for _ in range(n):
        snap = read_snapshot(env.get_ram())
        if axis == "x":
            if abs(snap.link_x - tgt) <= 2:
                return True
            step(env, assist, total, nes_action("RIGHT" if snap.link_x < tgt else "LEFT"))
        else:
            if abs(snap.link_y - tgt) <= 2:
                return True
            step(env, assist, total, nes_action("DOWN" if snap.link_y < tgt else "UP"))
    return False


def rec(env, tag):
    s = read_snapshot(env.get_ram())
    return {"tag": tag, "sc": f"0x{s.screen:02x}", "xy": [s.link_x, s.link_y], "mode": s.mode, "R": s.rupees}


def main():
    configure_headless()
    env = None
    total = [1]
    log = []
    try:
        env, assist = open_env()
        ok = to_1b(env, assist, total)
        log.append(rec(env, "arrive"))
        print(log[-1], flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_1b_arrive.png")
        if not ok and read_snapshot(env.get_ram()).screen != 0x1B:
            write_json_report(RECORDINGS_DIR / "l5_1b_escape.json", {"ok": False, "log": log})
            return

        # Plan 1: west at y=141, 165, 189, 109
        sc0 = 0x1B
        for y in (141, 165, 189, 109, 172):
            if read_snapshot(env.get_ram()).screen != sc0:
                break
            go(env, assist, total, "y", y)
            go(env, assist, total, "x", 24)
            hold(env, assist, total, "LEFT", 280)
            idle(env, assist, total, 20)
            for _ in range(60):
                s = read_snapshot(env.get_ram())
                if s.mode == PLAY_MODE and not s.transitioning:
                    break
                step(env, assist, total, nes_idle_action())
            log.append(rec(env, f"west_y{y}"))
            print(log[-1], flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / f"l5_1b_w{y}.png")
            if read_snapshot(env.get_ram()).screen != sc0:
                break

        # Plan 2: reverse pocket — south y=172, east, y=140, east
        if read_snapshot(env.get_ram()).screen == sc0:
            go(env, assist, total, "x", 100)
            go(env, assist, total, "y", 172)
            log.append(rec(env, "pocket_south"))
            print(log[-1], flush=True)
            go(env, assist, total, "x", 224)
            log.append(rec(env, "pocket_east"))
            print(log[-1], flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_1b_pocket.png")
            go(env, assist, total, "y", 140)
            hold(env, assist, total, "RIGHT", 360)
            idle(env, assist, total, 20)
            for _ in range(80):
                s = read_snapshot(env.get_ram())
                if s.mode == PLAY_MODE and not s.transitioning:
                    break
                step(env, assist, total, nes_idle_action())
            log.append(rec(env, "pocket_right"))
            print(log[-1], flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_1b_pocket_e.png")

        # Plan 3: from east mouth y=149 hold RIGHT longer
        if read_snapshot(env.get_ram()).screen == sc0:
            go(env, assist, total, "y", 149)
            go(env, assist, total, "x", 224)
            hold(env, assist, total, "RIGHT", 500)
            idle(env, assist, total, 24)
            for _ in range(80):
                s = read_snapshot(env.get_ram())
                if s.mode == PLAY_MODE and not s.transitioning:
                    break
                step(env, assist, total, nes_idle_action())
            log.append(rec(env, "east_y149_long"))
            print(log[-1], flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_1b_e149.png")

        body = {"log": log, "final": rec(env, "final"), "pokes": False, "status_claim": None}
        write_json_report(RECORDINGS_DIR / "l5_1b_escape.json", body)
        print("FINAL", body["final"], flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

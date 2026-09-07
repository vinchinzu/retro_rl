"""Pin down the deterministic 0x1A -> 0x0A step.

probe_ow_1a_hills_ups.py got through on the 8th UP at x=176, but that count
is not a maze secret: in most attempts Link never reached the north edge at
all (he stopped at y=133, a wall), so the climb only worked once he happened
to drift onto the right column. Sweep the columns at 8px and screenshot the
screen to find the actual opening.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_ow_1a_north.py
"""
from __future__ import annotations

from pathlib import Path

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

PLAY_MODE = 5
RECORDINGS = Path(__file__).resolve().parents[1] / "recordings"
COLS = tuple(range(32, 225, 8))
START_Y = 133


def walk_to(env, assist, *, tx=None, ty=None, limit=500):
    origin = read_snapshot(env.get_ram())
    for i in range(limit):
        snap = read_snapshot(env.get_ram())
        if snap.screen != origin.screen or snap.mode != PLAY_MODE:
            return False
        if ty is not None and abs(int(snap.link_y) - ty) > 2:
            d = "DOWN" if int(snap.link_y) < ty else "UP"
        elif tx is not None and abs(int(snap.link_x) - tx) > 2:
            d = "RIGHT" if int(snap.link_x) < tx else "LEFT"
        else:
            return True
        env.step(nes_action(d))
        assist.apply_env(env, frame=i)
    return False


def main() -> int:
    configure_headless()
    env = make_env(GAME, "OW_1A_Real", GAME_DIR, render_mode="rgb_array")
    obs, _ = reset_obs(env)
    assist = UnlimitedHealthAssist(enabled=True)
    snap = read_snapshot(env.get_ram())
    print(f"0x{snap.screen:02x} at ({snap.link_x},{snap.link_y})", flush=True)
    for _ in range(8):
        obs, *_ = env.step(nes_action(""))
    save_rgb_png(obs, RECORDINGS / "ow_1a_lost_hills_screen.png")
    print("screenshot -> recordings/ow_1a_lost_hills_screen.png", flush=True)
    pin = env.em.get_state()

    for col in COLS:
        env.em.set_state(pin)
        if not walk_to(env, assist, ty=START_Y):
            print(f"x={col:3d}: cannot reach y={START_Y}", flush=True)
            continue
        if not walk_to(env, assist, tx=col):
            s = read_snapshot(env.get_ram())
            print(f"x={col:3d}: unreachable (stalled at "
                  f"({s.link_x},{s.link_y}))", flush=True)
            continue
        s0 = read_snapshot(env.get_ram())
        outcome = "blocked"
        for i in range(500):
            snap = read_snapshot(env.get_ram())
            if snap.screen != s0.screen or snap.mode != PLAY_MODE:
                outcome = f"-> 0x{snap.screen:02x} mode={snap.mode}"
                break
            env.step(nes_action("UP"))
            assist.apply_env(env, frame=i)
        end = read_snapshot(env.get_ram())
        print(f"x={col:3d}: from ({s0.link_x},{s0.link_y}) UP {outcome} "
              f"at ({end.link_x},{end.link_y})", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

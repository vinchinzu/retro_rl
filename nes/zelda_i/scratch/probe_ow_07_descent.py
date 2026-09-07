"""Find the lane on screen 0x07 that connects the y=141 arrival band (coming
west-to-east from 0x06) down to the bottom band where the 0x17 exit is.

WhiteSwordDetourController stalls here: it aligns x=32 at y=141 and holds
DOWN forever. The live hop 0x07 -> 0x17 was originally measured from the
OW_07_Row0Real pin, where Link already stood at (64,221) on the bottom band,
so the descent inside 0x07 was never actually exercised.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_ow_07_descent.py
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
    env = make_env(GAME, "OW_07_Row0Real", GAME_DIR, render_mode="rgb_array")
    obs, _ = reset_obs(env)
    assist = UnlimitedHealthAssist(enabled=True)
    for _ in range(8):
        obs, *_ = env.step(nes_action(""))
    save_rgb_png(obs, RECORDINGS / "ow_07_screen.png")
    print("screenshot -> recordings/ow_07_screen.png", flush=True)

    # Recreate the controller's situation: stand on the y=141 arrival band.
    walk_to(env, assist, ty=141)
    snap = read_snapshot(env.get_ram())
    print(f"on band: ({snap.link_x},{snap.link_y})", flush=True)
    pin = env.em.get_state()

    for col in COLS:
        env.em.set_state(pin)
        if not walk_to(env, assist, tx=col):
            s = read_snapshot(env.get_ram())
            print(f"x={col:3d}: unreachable at y=141 (stalled ({s.link_x},{s.link_y}))",
                  flush=True)
            continue
        s0 = read_snapshot(env.get_ram())
        outcome = "blocked"
        for i in range(500):
            snap = read_snapshot(env.get_ram())
            if snap.screen != s0.screen or snap.mode != PLAY_MODE:
                outcome = f"-> 0x{snap.screen:02x}"
                break
            if snap.link_y >= 215:
                outcome = "reached bottom band"
                break
            env.step(nes_action("DOWN"))
            assist.apply_env(env, frame=i)
        end = read_snapshot(env.get_ram())
        print(f"x={col:3d}: from ({s0.link_x},{s0.link_y}) DOWN {outcome} "
              f"at ({end.link_x},{end.link_y})", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

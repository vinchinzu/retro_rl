"""Take whatever the Old Man in the 0x0A cave is offering, and confirm it is
the White Sword (ADDR_SWORD 1 -> 2).

Entered live from OW_0A_CaveReal (probe_ow_0a_mouth.py: climb the x=208 sand
corridor to the top band, walk to x~34, UP). Link carries 10 heart containers
here, comfortably over the 5-container Old Man gate.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_ow_0a_take_sword.py
"""
from __future__ import annotations

from pathlib import Path

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

RECORDINGS = Path(__file__).resolve().parents[1] / "recordings"


def main() -> int:
    configure_headless()
    env = make_env(GAME, "OW_0A_CaveReal", GAME_DIR, render_mode="rgb_array")
    obs, _ = reset_obs(env)
    assist = UnlimitedHealthAssist(enabled=True)

    for i in range(300):  # let the Old Man's text finish
        obs, *_ = env.step(nes_idle_action())
        assist.apply_env(env, frame=i)
    snap = read_snapshot(env.get_ram())
    print(f"in cave: mode={snap.mode} xy=({snap.link_x},{snap.link_y}) "
          f"sword={snap.sword} containers={snap.heart_containers} "
          f"rupees={snap.rupees}", flush=True)
    for o in snap.objects:
        if o.type_id:
            print(f"  obj slot={o.slot} type=0x{o.type_id:02x} xy=({o.x},{o.y})", flush=True)
    save_rgb_png(obs, RECORDINGS / "ow_0a_cave_interior.png")
    print("screenshot -> recordings/ow_0a_cave_interior.png", flush=True)
    pin = env.em.get_state()

    for tx in (88, 96, 104, 112, 120, 128, 136, 144, 152, 160):
        env.em.set_state(pin)
        for i in range(200):
            snap = read_snapshot(env.get_ram())
            if abs(int(snap.link_x) - tx) <= 2:
                break
            env.step(nes_action("RIGHT" if int(snap.link_x) < tx else "LEFT"))
            assist.apply_env(env, frame=i)
        for i in range(150):
            env.step(nes_action("UP"))
            assist.apply_env(env, frame=i)
        snap = read_snapshot(env.get_ram())
        print(f"  UP at x={tx}: sword={snap.sword} xy=({snap.link_x},{snap.link_y})",
              flush=True)
        if snap.sword >= 2:
            print(f"  *** sword {snap.sword} -- White Sword confirmed at x={tx} ***",
                  flush=True)
            save_state(env, GAME_DIR, GAME, "OW_0A_WhiteSwordTakenReal")
            print("  pinned OW_0A_WhiteSwordTakenReal", flush=True)
            env.close()
            return 0
    print("no position upgraded the sword", flush=True)
    env.close()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

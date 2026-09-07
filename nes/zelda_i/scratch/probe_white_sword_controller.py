"""Run WhiteSwordDetourController from the real power-on pin OW_05_Row0Real
(Level 9 approach screen 0x05, sword=1, 10 heart containers, TF 0xff).

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_white_sword_controller.py
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.overworld.white_sword import make_white_sword_detour_controller
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot


def main() -> int:
    configure_headless()
    env = make_env(GAME, "OW_05_Row0Real", GAME_DIR, render_mode="rgb_array")
    reset_obs(env)
    assist = UnlimitedHealthAssist(enabled=True)
    ctl = make_white_sword_detour_controller()
    snap = read_snapshot(env.get_ram())
    print(f"start 0x{snap.screen:02x} ({snap.link_x},{snap.link_y}) sword={snap.sword} "
          f"containers={snap.heart_containers} tf=0x{snap.triforce:02x}", flush=True)
    phase = None
    frame = 0
    while not ctl.success and not ctl.failed and frame < 25000:
        snap = read_snapshot(env.get_ram())
        if ctl.phase is not phase:
            phase = ctl.phase
            print(f"f{frame} -> {phase.name} 0x{snap.screen:02x} "
                  f"({snap.link_x},{snap.link_y}) sword={snap.sword}", flush=True)
        if frame % 4000 == 0 and frame:
            print(f"  ..f{frame} {ctl.phase.name} leg={ctl.leg_i} 0x{snap.screen:02x} "
                  f"({snap.link_x},{snap.link_y})", flush=True)
        env.step(ctl.step(snap).action)
        frame += 1
        assist.apply_env(env, frame=frame)
    snap = read_snapshot(env.get_ram())
    print(f"DONE success={ctl.success} failed={ctl.failed} notes={ctl.notes} "
          f"frames={ctl.frames} 0x{snap.screen:02x} ({snap.link_x},{snap.link_y}) "
          f"sword={snap.sword} tf=0x{snap.triforce:02x} bombs={snap.bombs}", flush=True)
    if ctl.success:
        save_state(env, GAME_DIR, GAME, "OW_05_PostWhiteSwordReal")
        print("pinned OW_05_PostWhiteSwordReal", flush=True)
    env.close()
    return 0 if ctl.success else 1


if __name__ == "__main__":
    raise SystemExit(main())

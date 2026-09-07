"""Run NaturalSilverArrowsController from the real power-on pin L9EntryReal
(Level 9 room 0x76, White Sword in hand, TF 0xff).

Iteration loop for the 0x14 stall the White Sword detour's RNG shift exposed:
east_14 walks x-first with no recovery, so a Like Like bump a few pixels off
the y=93 lane pinned Link at (176,101) holding LEFT into stone.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l9_arrows_from_entry.py
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level9.dungeon import MEASURED_POST_L8_HANDOFF
from zelda_i.level9.natural_path import make_natural_silver_arrows_controller
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot


def main() -> int:
    configure_headless()
    env = make_env(GAME, "L9EntryReal", GAME_DIR, render_mode="rgb_array")
    reset_obs(env)
    assist = UnlimitedHealthAssist(enabled=True)
    ctl = make_natural_silver_arrows_controller(MEASURED_POST_L8_HANDOFF)
    snap = read_snapshot(env.get_ram())
    print(f"start L{snap.level} 0x{snap.screen:02x} ({snap.link_x},{snap.link_y}) "
          f"sword={snap.sword} arrows={snap.arrows} bombs={snap.bombs}", flush=True)
    hop = -1
    frame = 0
    while not ctl.success and not ctl.failed and frame < 50000:
        snap = read_snapshot(env.get_ram())
        if ctl.hop_i != hop:
            hop = ctl.hop_i
            spec = getattr(ctl._hops[hop], "spec_id", f"hop{hop}") if hop < len(ctl._hops) else "done"
            print(f"f{frame} hop{hop} {spec} 0x{snap.screen:02x} "
                  f"({snap.link_x},{snap.link_y}) arrows={snap.arrows}", flush=True)
        if frame % 5000 == 0 and frame:
            print(f"  ..f{frame} hop{ctl.hop_i} 0x{snap.screen:02x} "
                  f"({snap.link_x},{snap.link_y})", flush=True)
        env.step(ctl.step(snap).action)
        frame += 1
        assist.apply_env(env, frame=frame)
    snap = read_snapshot(env.get_ram())
    print(f"DONE success={ctl.success} failed={ctl.failed} notes={ctl.notes} "
          f"frames={ctl.frames} hop={ctl.hop_i} 0x{snap.screen:02x} "
          f"({snap.link_x},{snap.link_y}) arrows={snap.arrows} sword={snap.sword}",
          flush=True)
    if ctl.success:
        save_state(env, GAME_DIR, GAME, "L9PostArrowsWhiteSwordReal")
        print("pinned L9PostArrowsWhiteSwordReal", flush=True)
    env.close()
    return 0 if ctl.success else 1


if __name__ == "__main__":
    raise SystemExit(main())

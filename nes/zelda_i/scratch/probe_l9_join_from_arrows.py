"""Run `NaturalPatraJoinController` from the real power-on pin L9PostArrowsReal
(room 0x10, arrows=2, TF 0xff) and trace every phase change.

This is the fast iteration loop for the last unfixed L9 stage,
`level9_natural_patra_join`, which timed out live at 24000 frames because
CLEAR_20 chased a Wizzrobe back north through the bomb hole into 0x10.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l9_join_from_arrows.py
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level9.natural_path import make_natural_patra_join_controller
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

STATE_NAME = "L9PostArrowsReal"
MAX_FRAMES = 30000


def main() -> int:
    configure_headless()
    env = make_env(GAME, STATE_NAME, GAME_DIR, render_mode="rgb_array")
    obs, _ = reset_obs(env)
    assist = UnlimitedHealthAssist(enabled=True)
    ctl = make_natural_patra_join_controller()
    snap = read_snapshot(env.get_ram())
    print(
        f"start room=0x{snap.screen:02x} xy=({snap.link_x},{snap.link_y}) "
        f"arrows={snap.arrows} tf=0x{snap.triforce:02x} bombs={snap.bombs}",
        flush=True,
    )
    phase = None
    frame = 0
    while not ctl.success and not ctl.failed and frame < MAX_FRAMES:
        snap = read_snapshot(env.get_ram())
        if ctl.phase is not phase:
            phase = ctl.phase
            print(
                f"f{frame} -> {phase.name} room=0x{snap.screen:02x} "
                f"xy=({snap.link_x},{snap.link_y}) reentries={ctl.reentries_10}",
                flush=True,
            )
        if frame % 4000 == 0 and frame:
            print(
                f"  ..f{frame} {ctl.phase.name} room=0x{snap.screen:02x} "
                f"xy=({snap.link_x},{snap.link_y})",
                flush=True,
            )
        act = ctl.step(snap)
        obs, *_ = env.step(act.action)
        frame += 1
        assist.apply_env(env, frame=frame)
    snap = read_snapshot(env.get_ram())
    print(
        f"DONE success={ctl.success} failed={ctl.failed} notes={ctl.notes} "
        f"frames={ctl.frames} phase={ctl.phase.name} room=0x{snap.screen:02x} "
        f"xy=({snap.link_x},{snap.link_y}) reentries={ctl.reentries_10}",
        flush=True,
    )
    env.close()
    return 0 if ctl.success else 1


if __name__ == "__main__":
    raise SystemExit(main())

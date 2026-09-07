"""Pin real power-on savestates on the row-0 screens the Level 9 approach
walks (0x07, 0x06, 0x05), so the White Sword detour can be developed from a
valid live position instead of hand-navigated.

Hand-driving 0x27 -> 0x17 -> 0x07 with a bare OverworldPathController stalls:
Link lands on 0x17 at (112,204) and cannot reach the x=64 column from there
(0x17 needs the "drop to y~133 then west" handling the Level 9 controller
has). The spine already does this correctly on its way to Level 9, so pin
from the spine rather than reimplement it.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/pin_ow_row0.py
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.spine.survival import run_survival_spine

WANT = {0x07: "OW_07_Row0Real", 0x06: "OW_06_Row0Real", 0x05: "OW_05_Row0Real"}
done: dict[int, bool] = {}


def make_on_frame(env_holder):
    def on_frame(env, obs, action, frame):
        snap = read_snapshot(env.get_ram())
        if snap.level != 0 or snap.mode != 5 or snap.transitioning:
            return
        name = WANT.get(snap.screen)
        if name is None or done.get(snap.screen):
            return
        done[snap.screen] = True
        print(f"f{frame}: 0x{snap.screen:02x} at ({snap.link_x},{snap.link_y}) "
              f"containers={snap.heart_containers} sword={snap.sword} "
              f"tf=0x{snap.triforce:02x}", flush=True)
        save_state(env, GAME_DIR, GAME, name)
    return on_frame


def main() -> int:
    configure_headless()
    assist = UnlimitedHealthAssist(enabled=True)
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    obs, _ = reset_obs(env)
    try:
        run = run_survival_spine(
            env, obs, assist=assist, through="level9-entry",
            on_frame=make_on_frame(env),
        )
    finally:
        env.close()
    rep = run.report()
    print(f"pinned={sorted(hex(k) for k in done)}")
    print(f"ok={rep.get('ok')} failed_stage={rep.get('failed_stage')} "
          f"end_frame={rep.get('end_frame')}", flush=True)
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())

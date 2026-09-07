"""Re-test west off the L5 door screen 0x0B toward the White Sword cave 0x0A.

The first sweep (probe_ow_0a_sweep.py) reported "sealed", but its three
northern bands never actually ran: the OW_0B_L5Door pin puts Link on x=112,
which is the Level 5 door column, so walking UP to align y entered the
dungeon and aborted the trial. This version steps Link off the door column
first, then aligns y, then holds LEFT.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_ow_0b_west_retest.py
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

PLAY_MODE = 5
L5_DOOR_X = 112
SAFE_X = 48
BANDS = (61, 77, 93, 109, 125, 141, 157, 173, 189)


def step_to(env, assist, *, tx=None, ty=None, limit=300):
    origin = read_snapshot(env.get_ram()).screen
    for i in range(limit):
        snap = read_snapshot(env.get_ram())
        if snap.screen != origin or snap.mode != PLAY_MODE:
            return False
        if tx is not None and abs(int(snap.link_x) - tx) > 2:
            d = "RIGHT" if int(snap.link_x) < tx else "LEFT"
        elif ty is not None and abs(int(snap.link_y) - ty) > 2:
            d = "DOWN" if int(snap.link_y) < ty else "UP"
        else:
            return True
        env.step(nes_action(d))
        assist.apply_env(env, frame=i)
    return False


def main() -> int:
    configure_headless()
    env = make_env(GAME, "OW_0B_L5Door", GAME_DIR, render_mode="rgb_array")
    reset_obs(env)
    assist = UnlimitedHealthAssist(enabled=True)
    pin = env.em.get_state()
    snap = read_snapshot(env.get_ram())
    print(f"pin: screen=0x{snap.screen:02x} xy=({snap.link_x},{snap.link_y}) "
          f"containers={snap.heart_containers} sword={snap.sword}", flush=True)

    for band in BANDS:
        env.em.set_state(pin)
        # off the Level 5 door column first, THEN align y, then go west.
        if not step_to(env, assist, tx=SAFE_X):
            print(f"  band={band:3d} could not clear the door column", flush=True)
            continue
        if not step_to(env, assist, ty=band):
            print(f"  band={band:3d} could not reach the band", flush=True)
            continue
        start = read_snapshot(env.get_ram())
        for i in range(700):
            snap = read_snapshot(env.get_ram())
            if snap.screen != start.screen or snap.mode not in (PLAY_MODE, 6, 7):
                break
            env.step(nes_action("LEFT"))
            assist.apply_env(env, frame=i)
        for i in range(180):
            snap = read_snapshot(env.get_ram())
            if snap.mode == PLAY_MODE:
                break
            env.step(nes_action("LEFT"))
            assist.apply_env(env, frame=i)
        end = read_snapshot(env.get_ram())
        tag = "SAME" if end.screen == start.screen else f"-> 0x{end.screen:02x}"
        print(f"  band={band:3d} from 0x{start.screen:02x} at ({start.link_x},{start.link_y}) "
              f"{tag} mode={end.mode} xy=({end.link_x},{end.link_y})", flush=True)
        if end.screen == 0x0A and end.mode == PLAY_MODE:
            save_state(env, GAME_DIR, GAME, "OW_0A_WhiteSword")
            print("  *** reached 0x0A -- pinned OW_0A_WhiteSword ***", flush=True)
            break
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

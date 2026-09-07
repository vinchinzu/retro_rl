"""Walk row 0 eastward from the Level 9 entrance screen 0x05 toward the White
Sword cave 0x0A, discovering the alignment band each hop needs.

0x0A is marked residual in route/item_gate_hops.py, and both of the routes
that module guessed at are now falsified live: west off the L5 door 0x0B is
sealed at every band, and so is west off Lost Hills 0x1B. Row 0 is the
remaining approach, and it is also where the Level 9 route already walks
(0x17 -> 0x07 -> 0x06 -> 0x05), so a detour here is nearly free.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_ow_row0_east.py
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

PLAY_MODE = 5
BANDS = (61, 77, 93, 109, 125, 141, 157, 173, 189)
TARGET = 0x0A


def align_then_walk(env, assist, direction, band, limit=700):
    horizontal = direction in ("LEFT", "RIGHT")
    origin = read_snapshot(env.get_ram()).screen
    for i in range(300):
        snap = read_snapshot(env.get_ram())
        if snap.screen != origin or snap.mode != PLAY_MODE:
            return None, snap
        cur = int(snap.link_y) if horizontal else int(snap.link_x)
        if abs(cur - band) <= 2:
            break
        d = ("DOWN" if cur < band else "UP") if horizontal else ("RIGHT" if cur < band else "LEFT")
        env.step(nes_action(d))
        assist.apply_env(env, frame=i)
    start = read_snapshot(env.get_ram())
    for i in range(limit):
        snap = read_snapshot(env.get_ram())
        if snap.screen != start.screen or snap.mode not in (PLAY_MODE, 6, 7):
            break
        env.step(nes_action(direction))
        assist.apply_env(env, frame=i)
    for i in range(180):
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE:
            break
        env.step(nes_action(direction))
        assist.apply_env(env, frame=i)
    return start, read_snapshot(env.get_ram())


def main() -> int:
    configure_headless()
    env = make_env(GAME, "OW_05_SpectacleRock", GAME_DIR, render_mode="rgb_array")
    reset_obs(env)
    assist = UnlimitedHealthAssist(enabled=True)
    snap = read_snapshot(env.get_ram())
    print(f"start screen=0x{snap.screen:02x} xy=({snap.link_x},{snap.link_y}) "
          f"containers={snap.heart_containers} sword={snap.sword}", flush=True)

    hops: list[tuple[int, str, int, int]] = []
    state = env.em.get_state()
    for hop in range(8):
        cur = read_snapshot(env.get_ram()).screen
        if cur == TARGET:
            break
        advanced = False
        for band in BANDS:
            env.em.set_state(state)
            start, end = align_then_walk(env, assist, "RIGHT", band)
            if start is None:
                continue
            if end.screen != start.screen and end.mode == PLAY_MODE:
                print(f"hop{hop}: 0x{start.screen:02x} RIGHT band={band} "
                      f"-> 0x{end.screen:02x} at ({end.link_x},{end.link_y})", flush=True)
                hops.append((start.screen, "RIGHT", band, end.screen))
                state = env.em.get_state()
                advanced = True
                break
        if not advanced:
            print(f"hop{hop}: 0x{cur:02x} RIGHT is sealed at every band", flush=True)
            break

    env.em.set_state(state)
    snap = read_snapshot(env.get_ram())
    print(f"\nended on 0x{snap.screen:02x} xy=({snap.link_x},{snap.link_y})", flush=True)
    print("hop table:", [(hex(a), d, b, hex(c)) for a, d, b, c in hops], flush=True)
    if snap.screen == TARGET:
        save_state(env, GAME_DIR, GAME, "OW_0A_WhiteSword")
        print("pinned OW_0A_WhiteSword", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

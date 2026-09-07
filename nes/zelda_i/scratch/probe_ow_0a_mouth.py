"""Find the cave mouth on screen 0x0A.

The tile-dump route is a dead end here: `dump_room_tiles` reads
`colliding_tile`, a dungeon-only field, so on the overworld it returns a
uniform 0x24 for every cell (0x0A dumped as one solid block). So find the
mouth by movement, and take a screenshot to look at.

Sweeps every 8px column, walking Link to the bottom of it and holding UP,
and reports what happened for each -- including columns he could not reach,
which the previous sweep silently skipped.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_ow_0a_mouth.py
"""
from __future__ import annotations

from pathlib import Path

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

PLAY_MODE = 5
CAVE_MODE = 11
RECORDINGS = Path(__file__).resolve().parents[1] / "recordings"
COLS = tuple(range(32, 225, 8))


def walk_to(env, assist, *, tx=None, ty=None, limit=500):
    origin = read_snapshot(env.get_ram())
    for i in range(limit):
        snap = read_snapshot(env.get_ram())
        if snap.screen != origin.screen or snap.mode != PLAY_MODE:
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
    env = make_env(GAME, "OW_0A_WhiteSwordReal", GAME_DIR, render_mode="rgb_array")
    obs, _ = reset_obs(env)
    assist = UnlimitedHealthAssist(enabled=True)
    snap = read_snapshot(env.get_ram())
    print(f"0x{snap.screen:02x} at ({snap.link_x},{snap.link_y}) mode={snap.mode}", flush=True)
    for _ in range(8):
        obs, *_ = env.step(nes_action(""))
    save_rgb_png(obs, RECORDINGS / "ow_0a_white_sword_screen.png")
    print("screenshot -> recordings/ow_0a_white_sword_screen.png", flush=True)
    # Link lands in a narrow sand corridor at x=208 and cannot move sideways
    # at y=221 -- a lake fills the middle of the screen and the cave mouth is
    # in the upper-left (recordings/ow_0a_white_sword_screen.png). Climb the
    # corridor to the top band first, then sweep across it.
    for i in range(500):
        snap = read_snapshot(env.get_ram())
        if snap.link_y <= 87 or snap.mode != PLAY_MODE:
            break
        env.step(nes_action("UP"))
        assist.apply_env(env, frame=i)
    snap = read_snapshot(env.get_ram())
    print(f"climbed to ({snap.link_x},{snap.link_y})", flush=True)
    pin = env.em.get_state()

    for col in COLS:
        env.em.set_state(pin)
        reached = walk_to(env, assist, tx=col)
        s0 = read_snapshot(env.get_ram())
        if not reached:
            print(f"x={col:3d}: unreachable, stalled at ({s0.link_x},{s0.link_y})", flush=True)
            continue
        outcome = "no change"
        for i in range(500):
            snap = read_snapshot(env.get_ram())
            if snap.mode == CAVE_MODE or snap.level != 0:
                outcome = f"CAVE mode={snap.mode} level={snap.level}"
                save_state(env, GAME_DIR, GAME, "OW_0A_CaveReal")
                break
            if snap.screen != 0x0A and snap.mode == PLAY_MODE:
                outcome = f"left to 0x{snap.screen:02x}"
                break
            env.step(nes_action("UP"))
            assist.apply_env(env, frame=i)
        end = read_snapshot(env.get_ram())
        print(f"x={col:3d}: from ({s0.link_x},{s0.link_y}) UP -> {outcome} "
              f"at ({end.link_x},{end.link_y})", flush=True)
        if outcome.startswith("CAVE"):
            print("*** pinned OW_0A_CaveReal ***", flush=True)
            break
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

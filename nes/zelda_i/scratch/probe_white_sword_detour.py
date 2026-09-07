"""End-to-end live check of the White Sword detour, from the real power-on
pin OW_07_Row0Real (row 0 on the Level 9 approach, TF 0xff, 10 containers)
out to the cave and back to the Level 9 entrance screen 0x05.

Route established live this session:

    0x07 --DOWN  x=32 --> 0x17
    0x17 --RIGHT y=141--> 0x18
    0x18 --RIGHT y=141--> 0x19
    0x19 --RIGHT y=141--> 0x1A
    0x1A --UP from (208,157)--> 0x0A      (single opening; NOT a maze count)
    0x0A  climb x=208 to y~85, LEFT to x~34, UP --> cave
    cave  walk to x=120, UP --> White Sword (ADDR_SWORD 1 -> 2)
    then reverse out to 0x05.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_white_sword_detour.py
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

PLAY_MODE = 5
CAVE_MODE = 11
OUT = (("DOWN", 32, 0x17), ("RIGHT", 141, 0x18), ("RIGHT", 141, 0x19),
       ("RIGHT", 141, 0x1A))
BACK = (("LEFT", 141, 0x19), ("LEFT", 141, 0x18), ("LEFT", 141, 0x17),
        ("UP", 64, 0x07), ("LEFT", 141, 0x06), ("LEFT", 141, 0x05))
frames = [0]


def step(env, assist, action):
    env.step(action)
    frames[0] += 1
    assist.apply_env(env, frame=frames[0])


def walk_to(env, assist, *, tx=None, ty=None, limit=600):
    origin = read_snapshot(env.get_ram())
    for _ in range(limit):
        snap = read_snapshot(env.get_ram())
        if snap.screen != origin.screen or snap.mode != PLAY_MODE:
            return False
        if ty is not None and abs(int(snap.link_y) - ty) > 2:
            d = "DOWN" if int(snap.link_y) < ty else "UP"
        elif tx is not None and abs(int(snap.link_x) - tx) > 2:
            d = "RIGHT" if int(snap.link_x) < tx else "LEFT"
        else:
            return True
        step(env, assist, nes_action(d))
    return False


def hop(env, assist, direction, band, want):
    if direction in ("LEFT", "RIGHT"):
        walk_to(env, assist, ty=band)
    else:
        walk_to(env, assist, tx=band)
    start = read_snapshot(env.get_ram())
    for _ in range(700):
        snap = read_snapshot(env.get_ram())
        if snap.screen != start.screen and snap.mode == PLAY_MODE:
            break
        step(env, assist, nes_action(direction))
    for _ in range(200):
        if read_snapshot(env.get_ram()).mode == PLAY_MODE:
            break
        step(env, assist, nes_action(direction))
    end = read_snapshot(env.get_ram())
    ok = end.screen == want
    print(f"  0x{start.screen:02x} {direction:5s} band={band:3d} -> "
          f"0x{end.screen:02x} at ({end.link_x},{end.link_y}) "
          f"{'ok' if ok else f'WANTED 0x{want:02x}'}", flush=True)
    return ok


def main() -> int:
    configure_headless()
    env = make_env(GAME, "OW_07_Row0Real", GAME_DIR, render_mode="rgb_array")
    reset_obs(env)
    assist = UnlimitedHealthAssist(enabled=True)
    snap = read_snapshot(env.get_ram())
    print(f"start 0x{snap.screen:02x} ({snap.link_x},{snap.link_y}) "
          f"sword={snap.sword} containers={snap.heart_containers} "
          f"tf=0x{snap.triforce:02x}", flush=True)

    print("out:", flush=True)
    for d, b, want in OUT:
        if not hop(env, assist, d, b, want):
            env.close()
            return 1

    # 0x1A -> 0x0A: the single north opening at x=208.
    walk_to(env, assist, ty=157)
    if not hop(env, assist, "UP", 208, 0x0A):
        env.close()
        return 1

    # 0x0A: climb the x=208 sand corridor, cross the top band west, enter.
    for _ in range(600):
        snap = read_snapshot(env.get_ram())
        if snap.link_y <= 87:
            break
        step(env, assist, nes_action("UP"))
    walk_to(env, assist, tx=34)
    for _ in range(400):
        snap = read_snapshot(env.get_ram())
        if snap.mode == CAVE_MODE:
            break
        step(env, assist, nes_action("UP"))
    snap = read_snapshot(env.get_ram())
    print(f"  cave: mode={snap.mode} at ({snap.link_x},{snap.link_y})", flush=True)
    if snap.mode != CAVE_MODE:
        env.close()
        return 1

    for _ in range(300):  # Old Man text
        step(env, assist, nes_idle_action())
    for _ in range(200):
        snap = read_snapshot(env.get_ram())
        if abs(int(snap.link_x) - 120) <= 2:
            break
        step(env, assist, nes_action("RIGHT" if int(snap.link_x) < 120 else "LEFT"))
    for _ in range(150):
        if read_snapshot(env.get_ram()).sword >= 2:
            break
        step(env, assist, nes_action("UP"))
    snap = read_snapshot(env.get_ram())
    print(f"  sword={snap.sword} after pickup", flush=True)
    if snap.sword < 2:
        env.close()
        return 1

    for _ in range(500):  # back out to 0x0A
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE and snap.screen == 0x0A:
            break
        step(env, assist, nes_action("DOWN"))
    snap = read_snapshot(env.get_ram())
    print(f"  exited to 0x{snap.screen:02x} at ({snap.link_x},{snap.link_y})", flush=True)

    print("back:", flush=True)
    # The cave spits Link out at the top-LEFT (32,77). The lake fills the
    # middle of 0x0A, so the only way south is back along the top band and
    # down the x=208 sand corridor -- going DOWN from the exit just walks him
    # into the west shore and stalls at (32,189).
    walk_to(env, assist, ty=85)
    walk_to(env, assist, tx=208)
    walk_to(env, assist, ty=213)
    if not hop(env, assist, "DOWN", 208, 0x1A):
        env.close()
        return 1
    for d, b, want in BACK:
        if not hop(env, assist, d, b, want):
            env.close()
            return 1

    snap = read_snapshot(env.get_ram())
    print(f"\nDONE 0x{snap.screen:02x} at ({snap.link_x},{snap.link_y}) "
          f"sword={snap.sword} tf=0x{snap.triforce:02x} bombs={snap.bombs} "
          f"in {frames[0]} frames", flush=True)
    save_state(env, GAME_DIR, GAME, "OW_05_WhiteSwordReal")
    print("pinned OW_05_WhiteSwordReal", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Enter the cave on screen 0x0A and find out what the Old Man gives.

0x0A is finally reachable live (probe_ow_1a_hills_ups.py: 0x1A is a Lost
Hills maze screen, so a single UP wraps and only a repeated climb near
x=176 breaks through). ROM says its cave id 18 is unique, sitting between
the wooden sword cave 0x77 (id 16) and the Magical Sword grave 0x21 (id 19).
This checks the premise directly: walk in and read the inventory.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_ow_0a_cave.py
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level9.stair_run import dump_room_tiles
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

PLAY_MODE = 5
CAVE_MODE = 11
COLS = tuple(range(32, 225, 8))


def walk_to(env, assist, *, tx=None, ty=None, limit=600):
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
    reset_obs(env)
    assist = UnlimitedHealthAssist(enabled=True)
    snap = read_snapshot(env.get_ram())
    print(f"0x{snap.screen:02x} at ({snap.link_x},{snap.link_y}) mode={snap.mode} "
          f"sword={snap.sword} containers={snap.heart_containers}", flush=True)
    pin = env.em.get_state()

    dump = dump_room_tiles(env, total=[0])
    ox, oy = dump["grid_origin"]
    print("tiles " + " ".join(f"{ox + 8 * i:3d}" for i in range(len(dump["grid"][0]))), flush=True)
    for ri, row in enumerate(dump["grid"]):
        print(f"y={oy + 8 * ri:3d} " + " ".join(f" {t:02x}" for t in row), flush=True)

    for col in COLS:
        env.em.set_state(pin)
        if not walk_to(env, assist, tx=col):
            continue
        entered = False
        for i in range(400):
            snap = read_snapshot(env.get_ram())
            if snap.mode == CAVE_MODE or snap.level != 0:
                entered = True
                break
            if snap.screen != 0x0A and snap.mode == PLAY_MODE:
                break
            env.step(nes_action("UP"))
            assist.apply_env(env, frame=i)
        if not entered:
            continue
        print(f"\n*** cave entered from x={col} ***", flush=True)
        save_state(env, GAME_DIR, GAME, "OW_0A_CaveReal")
        for i in range(400):  # let the old man talk
            env.step(nes_idle_action())
            assist.apply_env(env, frame=i)
        snap = read_snapshot(env.get_ram())
        print(f"in cave: mode={snap.mode} xy=({snap.link_x},{snap.link_y}) "
              f"sword={snap.sword} rupees={snap.rupees}", flush=True)
        for o in snap.objects:
            if o.type_id:
                print(f"  obj slot={o.slot} type=0x{o.type_id:02x} "
                      f"xy=({o.x},{o.y})", flush=True)
        # sweep left-to-right across the cave floor onto whatever is there
        for tx in (96, 112, 128, 144, 160):
            walk_to(env, assist, tx=tx, limit=200)
            for i in range(90):
                env.step(nes_action("UP"))
                assist.apply_env(env, frame=i)
            snap = read_snapshot(env.get_ram())
            print(f"  after UP at x={tx}: sword={snap.sword} "
                  f"xy=({snap.link_x},{snap.link_y})", flush=True)
            if snap.sword >= 2:
                print(f"  *** SWORD UPGRADED to {snap.sword} ***", flush=True)
                env.close()
                return 0
        break
    else:
        print("\nno cave mouth found along the south face", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

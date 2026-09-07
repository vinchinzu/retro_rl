"""Dump the walkable tilemap of overworld screen 0x0B (Level 5 door / White
Sword region), so the west edge can be read directly instead of guessed at
band by band.

Band sweeps from the OW_0B_L5Door pin keep aborting: Link spawns at (112,221)
on the Level 5 door column and cannot reach the west side of the screen from
there. This shows what is actually in the way.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/dump_ow_0b_tiles.py
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless
from zelda_i.level9.stair_run import dump_room_tiles
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

STATE = "OW_0B_L5Door"


def main() -> int:
    configure_headless()
    env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
    reset_obs(env)
    snap = read_snapshot(env.get_ram())
    print(f"screen=0x{snap.screen:02x} xy=({snap.link_x},{snap.link_y}) mode={snap.mode}")
    dump = dump_room_tiles(env, total=[0])
    print("tile_counts:", dump["tile_counts"])
    ox, oy = dump["grid_origin"]
    print("      " + " ".join(f"{ox + 8 * i:3d}" for i in range(len(dump["grid"][0]))))
    for ri, row in enumerate(dump["grid"]):
        print(f"y={oy + 8 * ri:3d} " + " ".join(f" {t:02x}" for t in row))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

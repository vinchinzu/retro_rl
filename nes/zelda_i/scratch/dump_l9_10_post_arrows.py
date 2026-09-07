"""Dump L9 room 0x10's tilemap + objects from the REAL post-arrows pin
(L9PostArrowsReal: Link back in 0x10 at (96,157) holding the Silver Arrows).

The Patra join's SOUTH_10 phase aligns to x=120 and holds DOWN forever from
here -- Link never moves. This dump shows what is actually south of him and
where the real walkable lane to 0x20's bomb hole is.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/dump_l9_10_post_arrows.py
"""
from __future__ import annotations

import json

from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless
from zelda_i.level9.stair_run import dump_room_tiles
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

STATE_NAME = "L9PostArrowsReal"


def main() -> int:
    configure_headless()
    env = make_env(GAME, STATE_NAME, GAME_DIR, render_mode="rgb_array")
    reset_obs(env)
    total = [0]
    snap = read_snapshot(env.get_ram())
    print(f"entry: room=0x{snap.screen:02x} xy=({snap.link_x},{snap.link_y}) "
          f"item=0x{snap.room_item_id:02x} arrows={snap.arrows}")
    print("objects:")
    for o in snap.objects:
        if o.type_id:
            print(f"  slot={o.slot} type=0x{o.type_id:02x} xy=({o.x},{o.y}) hp={o.hp} st={o.state}")
    dump = dump_room_tiles(env, total=total)
    print("tile_counts:", dump["tile_counts"])
    print("stair_hits:", dump["stair_hits"])
    print("mouth_hits:", dump["mouth_hits"])
    ox, oy = dump["grid_origin"]
    print(f"grid origin=({ox},{oy}) step={dump['grid_step']}")
    print("      " + " ".join(f"{ox + 8 * i:3d}" for i in range(len(dump["grid"][0]))))
    for ri, row in enumerate(dump["grid"]):
        print(f"y={oy + 8 * ri:3d} " + " ".join(f" {t:02x}" for t in row))
    with open("nes/zelda_i/recordings/l9_room10_post_arrows_tiles.json", "w") as f:
        json.dump(dump, f, indent=2)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

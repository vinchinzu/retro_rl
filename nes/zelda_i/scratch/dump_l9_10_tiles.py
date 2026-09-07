"""Dump the L9 room 0x10 (Silver Arrows) tilemap + live objects from the real
power-on pin, to find the safe thread through the statue-diamond grid.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/dump_l9_10_tiles.py
"""
from __future__ import annotations

import json

from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless
from zelda_i.level9.stair_run import dump_room_tiles, _step
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

STATE_NAME = "L9Room10EntryReal"


def main() -> int:
    configure_headless()
    env = make_env(GAME, STATE_NAME, GAME_DIR, render_mode="rgb_array")
    reset_obs(env)
    total = [0]
    snap = read_snapshot(env.get_ram())
    print("entry snap:", snap.link_x, snap.link_y, "room_item_id", snap.room_item_id)
    print("objects:")
    for o in snap.objects:
        print(" ", o.slot, hex(o.type_id), o.x, o.y, o.hp, o.state)
    dump = dump_room_tiles(env, total=total)
    print("tile_counts:", dump["tile_counts"])
    print("grid_origin:", dump["grid_origin"], "step:", dump["grid_step"])
    for row in dump["grid"]:
        print("".join(f"{t:02x} " for t in row))
    with open("nes/zelda_i/recordings/l9_room10_tiles.json", "w") as f:
        json.dump(dump, f, indent=2)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

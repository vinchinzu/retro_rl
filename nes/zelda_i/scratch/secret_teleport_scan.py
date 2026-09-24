"""Measurement only: load every overworld screen by a RAM 'teleport' and dump it.

For each screen T the base pin gets ``$EB = T+1`` (or ``T-1``), Link on that
screen's west (east) edge, and LEFT (RIGHT) held until the scroll finishes on
T.  The scroll runs the ROM's own ``LayoutRoomOW``, so the ``$6530`` tile map,
the slot-11 tile object and the lattice are the real ones.  This is a what-if
RAM write (never a route result): use it to read walls, secrets and edges.

    uv run python nes/zelda_i/scratch/secret_teleport_scan.py OW_78 out.json [screens...]
"""
from __future__ import annotations

import json
import sys

import numpy as np

from retro_harness.env import make_env, read_state_bytes, state_path
from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon.tilemap import ow_walkable_nodes, read_room_tiles
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot, read_u8

LEFT = np.zeros(9, dtype=np.int8)
LEFT[6] = 1
RIGHT = np.zeros(9, dtype=np.int8)
RIGHT[7] = 1
DOWN = np.zeros(9, dtype=np.int8)
DOWN[5] = 1


def load_screen(env, base: bytes, target: int, y: int = 141, via: str = "side") -> dict | None:
    env.em.set_state(base)
    mem = env.unwrapped.data.memory
    if via == "above" and target >= 0x10:
        # Enter from the screen above (the 0x61 forest maze only exits east).
        start, x0, y, press = target - 16, 120, 221, DOWN
    elif target & 0x0F != 0x0F:
        start, x0, press = target + 1, 0, LEFT
    else:
        start, x0, press = target - 1, 240, RIGHT
    mem.assign(0x00EB, "|u1", start)
    mem.assign(0x0070, "|u1", x0)
    mem.assign(0x0084, "|u1", y)
    left_start = False
    for f in range(400):
        env.step(press)
        s = read_snapshot(env.get_ram())
        if s.screen != start:
            left_start = True
        if left_start and s.mode == 5 and s.screen == target:
            break
    else:
        return None
    ram = env.get_ram()
    s = read_snapshot(ram)
    o11 = [(o.type_id, o.x, o.y, o.state) for o in s.objects if o.slot == 11]
    nodes = sorted(ow_walkable_nodes(ram))
    return dict(
        screen=target,
        frames=f + 1,
        flag=read_u8(ram, 0x067F + target),
        o11=o11,
        nodes=nodes,
        tiles=read_room_tiles(ram).tolist(),
        link=(s.link_x, s.link_y),
    )


def main(argv: list[str]) -> int:
    base_name, out = argv[0], argv[1]
    screens = [int(a, 16) for a in argv[2:]] or list(range(128))
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    env.reset()
    base = read_state_bytes(state_path(GAME_DIR, GAME, base_name))
    rows = {}
    for t in screens:
        r = load_screen(env, base, t)
        if r is None:
            r = load_screen(env, base, t, via="above")
        if r is None:
            print(f"{t:02X} FAILED")
            continue
        rows[f"{t:02X}"] = r
        o = [(hex(a), b, c) for a, b, c, _ in r["o11"] if a]
        print(f"{t:02X} flag={r['flag']:02X} o11={o} nodes={len(r['nodes'])} f={r['frames']}")
    json.dump(rows, open(out, "w"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

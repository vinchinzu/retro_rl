"""Look inside one save state: Link, objects, room tiles, item flags, presses.

The first question at every stall is "what is actually in this room", and
each sitting used to answer it with a throwaway script. One load, no ROM
policy, no writes except ``--fixture``::

    uv run python nes/zelda_i/scripts/pin_probe.py BlueRingFull3_fail
    uv run python nes/zelda_i/scripts/pin_probe.py BlueRingFull3_fail --tiles
    uv run python nes/zelda_i/scripts/pin_probe.py BlueRingFull3_fail --press DOWN:40 LEFT:12
    uv run python nes/zelda_i/scripts/pin_probe.py BlueRingFull3_fail --items
    uv run python nes/zelda_i/scripts/pin_probe.py BlueRingFull3_fail --fixture --note "why"

``--press DIR:N`` holds DIR (or ``NONE``) for N frames and reprints Link and
the objects after each; the pose after a press is what the ROM allows there.
``--fixture`` writes ``tests/fixtures/room_tiles_l<level>_0x<room>.json`` for
``tests/ram_helpers.room_tile_env``. One emulator per process.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from retro_harness.actions import nes_action
from retro_harness.env import make_env, read_state_bytes, state_path
from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon.ids import item_code_name, object_name
from zelda_i.dungeon.tilemap import (
    ADDR_FIRST_UNWALKABLE,
    WRAM_BASE,
    WRAM_RAM_OFFSET,
    ADDR_ROOM_TILE_MAP,
    TILE_COLS,
    TILE_ROWS,
    has_room_tile_map,
    ow_walkable_nodes,
    read_room_tiles,
)
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import (
    ADDR_CUR_OPENED_DOORS,
    WORLD_FLAG_ITEM,
    read_snapshot,
    read_u8,
    room_flag,
    hearts_held,
    room_flags_addr,
    room_item_xy,
)

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def describe(ram) -> list[str]:
    s = read_snapshot(ram)
    containers = (int(s.health) >> 4) + 1
    flag = room_flag(ram, s.level, s.screen)
    lines = [
        f"L{s.level} room=0x{s.screen:02x} mode={s.mode} link=({s.link_x},{s.link_y}) "
        f"facing=0x{s.facing:02x} hearts={hearts_held(s):.2f}/{containers} "
        f"sword={s.sword} bombs={s.bombs} keys={s.keys} rupees={s.rupees} "
        f"tf=0x{s.triforce:02x} ring={s.ring} ladder={s.ladder}",
        f"  room item=0x{s.room_item_id:02x} {item_code_name(s.room_item_id)} at {room_item_xy(ram)} "
        f"flag=0x{flag:02x} taken={bool(flag & WORLD_FLAG_ITEM)} "
        f"doors=0x{read_u8(ram, ADDR_CUR_OPENED_DOORS):02x}",
    ]
    for o in s.objects:
        if not (o.type_id or o.hp or o.state):
            continue
        lines.append(
            f"  slot {o.slot:2d} 0x{o.type_id:02x} "
            f"{'link' if o.slot == 0 else object_name(o.type_id):28s} "
            f"({o.x},{o.y}) dir=0x{o.facing:02x} hp={o.hp} state=0x{o.state:02x}"
        )
    return lines


def tile_lines(ram, overworld: bool) -> list[str]:
    tiles = read_room_tiles(ram)
    lines = [f"first_unwalkable=0x{read_u8(ram, ADDR_FIRST_UNWALKABLE):02x}"]
    for r in range(tiles.shape[0]):
        lines.append(f"{r:2d} y={64 + 8 * r:3d} " + " ".join(f"{v:02x}" for v in tiles[r]))
    nodes = ow_walkable_nodes(ram, overworld=overworld)
    xs = sorted({x for x, _ in nodes})
    ys = sorted({y for _, y in nodes})
    lines.append("walkable lattice nodes (# = Link can stand):")
    lines.append("      " + "".join(f"{x:4d}" for x in xs))
    for y in ys:
        lines.append(f"{y:5d} " + "".join("   #" if (x, y) in nodes else "   ." for x in xs))
    return lines


def item_lines(ram) -> list[str]:
    lines = []
    for level, tag in ((0, "overworld"), (1, "L1-6"), (7, "L7-9")):
        base = room_flags_addr(level)
        rooms = [f"{r:02x}" for r in range(128) if read_u8(ram, base + r) & WORLD_FLAG_ITEM]
        lines.append(f"item-taken {tag} (${base:04X}): {' '.join(rooms) or '-'}")
    return lines


def write_fixture(ram, note: str) -> Path:
    s = read_snapshot(ram)
    start = WRAM_RAM_OFFSET + ADDR_ROOM_TILE_MAP - WRAM_BASE
    tiles = [int(v) for v in ram[start : start + TILE_COLS * TILE_ROWS]]
    path = FIXTURES / f"room_tiles_l{s.level}_0x{s.screen:02x}.json"
    path.write_text(
        json.dumps(
            {
                "room": f"0x{s.screen:02x}",
                "level": int(s.level),
                "cur_opened_doors": int(read_u8(ram, ADDR_CUR_OPENED_DOORS)),
                "note": note,
                "tiles": tiles,
            }
        )
        + "\n"
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("state", help="save-state name (BlueRingFull3_fail) or a .state path")
    parser.add_argument("--tiles", action="store_true", help="print $6530 and the walk lattice")
    parser.add_argument("--items", action="store_true", help="print item-taken world flags")
    parser.add_argument("--press", nargs="*", default=(), metavar="DIR:N")
    parser.add_argument("--fixture", action="store_true", help="write the room tile fixture")
    parser.add_argument("--note", default="", help="fixture note")
    args = parser.parse_args(argv)

    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    env.reset()
    path = Path(args.state)
    if not (path.suffix == ".state" and path.exists()):
        path = state_path(GAME_DIR, GAME, args.state)
    env.em.set_state(read_state_bytes(path))
    ram = env.get_ram()
    print("\n".join(describe(ram)))
    overworld = int(read_snapshot(ram).level) == 0
    if args.tiles and has_room_tile_map(ram):
        print("\n".join(tile_lines(ram, overworld)))
    if args.items:
        print("\n".join(item_lines(ram)))
    if args.fixture:
        print(f"wrote {write_fixture(ram, args.note)}")
    for press in args.press:
        name, _, count = press.partition(":")
        action = nes_action() if name.upper() == "NONE" else nes_action(name.upper())
        for _ in range(int(count or 1)):
            env.step(action)
        print(f"-- after {press}")
        print("\n".join(describe(env.get_ram())))
    return 0


if __name__ == "__main__":
    sys.exit(main())

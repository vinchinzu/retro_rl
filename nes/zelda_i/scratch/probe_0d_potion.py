"""Walk the candle hops from a pin onto 0x0D and print its secret rock.

    uv run python nes/zelda_i/scratch/probe_0d_potion.py G1_candle

Saves ``<pin>_on_0d`` for ``pin_probe.py``. A measurement, not a route.
"""

from __future__ import annotations

import sys

from retro_harness.env import make_env, read_state_bytes, save_state, state_path
from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon.tilemap import ow_walkable_nodes
from zelda_i.overworld.gather_segments import CANDLE_HOPS, HopWalkController
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.route.chain import run_controller_stage



def main() -> int:
    pin = sys.argv[1] if len(sys.argv) > 1 else "G1_candle"
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    env.reset()
    env.em.set_state(read_state_bytes(state_path(GAME_DIR, GAME, pin)))
    ctl = HopWalkController(hops=CANDLE_HOPS[:3], waypoints={})
    run_controller_stage(env, None, name="to_0d", controller=ctl, max_frames=6000)
    snap = read_snapshot(env.get_ram())
    print(f"0x{snap.screen:02x} mode={snap.mode} link=({snap.link_x},{snap.link_y}) bombs={snap.bombs} rupees={snap.rupees}")
    for o in snap.objects:
        if o.type_id or o.hp or o.state:
            print(f"  slot {o.slot:2d} type=0x{o.type_id:02x} ({o.x},{o.y}) hp={o.hp} state=0x{o.state:02x}")
    nodes = sorted(ow_walkable_nodes(env.get_ram(), overworld=True))
    rows: dict[int, list[int]] = {}
    for x, y in nodes:
        rows.setdefault(y, []).append(x)
    for y in sorted(rows):
        print(f"  y={y:3d} x={rows[y]}")
    save_state(env, GAME_DIR, GAME, f"{pin}_on_0d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

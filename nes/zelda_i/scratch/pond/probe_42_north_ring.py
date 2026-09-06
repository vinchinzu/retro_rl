"""rr-6o7.1 recon: occupancy-walk the refilled 0x42 north strip around the pond.

Loads Level7Entrance (same north-strip geometry as MEASURED_POST_L7_EXIT),
walks DOWN to OW 0x42, dumps $6530, then OccupancyWalker toward the south
gap at x=112. Miss -> block cell -> replan; no path -> stand.

Not a route claim. Do not STATUS.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/pond/probe_42_north_ring.py --tag l8_42ring
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.tilemap import LINK_FOOT_OFFSET, read_room_tiles, tile_at_screen
from zelda_i.overworld.common import (
    EDGE_EAST_X,
    EDGE_NORTH_Y,
    EDGE_SOUTH_Y,
    EDGE_WEST_X,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.runner import add_common_args, make_assist
from zelda_i.walk.physics import OccupancyGrid, OccupancyWalker

POND_SCREEN = 0x42
SOUTH_GAP = (112, EDGE_SOUTH_Y)
EXIT_MAX = 800
WALK_MAX = 8_000
# $6530 on refilled 0x42: sand is 0x26. Water is 0x8d-0x98. Rocks/trees
# are d8-db / c4-c7 / etc. Link's 16x16 body needs the sand ring.
SAND_TILE = 0x26


def _dump_tiles(ram) -> None:
    grid = read_room_tiles(ram)
    print(f"tile map 0x{POND_SCREEN:02x} {grid.shape[0]}x{grid.shape[1]}")
    for row_i, row in enumerate(grid):
        print(f"row{row_i:02d} " + " ".join(f"{v:02x}" for v in row))


def _seed_nonsand(ram) -> set[tuple[int, int]]:
    """Block stored (x,y) whose *feet* are not on sand.

    Link's RAM y sits ``LINK_FOOT_OFFSET`` above the colliding row, so the
    leftover (96,93) is a rock tile in $6530 and sand under his feet.
    """
    blocked: set[tuple[int, int]] = set()
    for y in range(EDGE_NORTH_Y, EDGE_SOUTH_Y + 1):
        for x in range(EDGE_WEST_X, EDGE_EAST_X + 1):
            try:
                tile = tile_at_screen(ram, x, y + LINK_FOOT_OFFSET)
            except IndexError:
                blocked.add((x, y))
                continue
            if tile != SAND_TILE:
                blocked.add((x, y))
    return blocked


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state="Level7Entrance", default_tag="l8_42ring")
    args = parser.parse_args()
    configure_headless()
    assist = make_assist(args.infinite_life)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    obs_box: list = [None]
    total = [0]
    samples: list[dict] = []

    def step(btn: str | None):
        act = nes_idle_action() if not btn else nes_action(btn)
        obs, *_ = env.step(act)
        obs_box[0] = obs
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
        return read_snapshot(env.get_ram())

    def shot(label: str, snap) -> str:
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        path = RECORDINGS_DIR / (
            f"{args.tag}_{label}_f{total[0]}_L{snap.level}_s{snap.screen:02x}_m{snap.mode}.png"
        )
        save_rgb_png(obs_box[0], path)
        return str(path)

    try:
        obs, _ = reset_obs(env)
        obs_box[0] = obs
        snap = step(None)
        print(
            f"start L{snap.level} s=0x{snap.screen:02x} ({snap.link_x},{snap.link_y}) "
            f"mode={snap.mode}"
        )
        for _ in range(EXIT_MAX):
            snap = read_snapshot(env.get_ram())
            if (
                snap.level == 0
                and snap.mode == PLAY_MODE
                and snap.screen == POND_SCREEN
                and not snap.transitioning
            ):
                break
            snap = step("DOWN")
        else:
            print("FAIL: never settled OW 0x42")
            shot("fail_exit", snap)
            return

        ram = env.get_ram()
        print(
            f"settled 0x42 ({snap.link_x},{snap.link_y}) tile="
            f"{tile_at_screen(ram, snap.link_x, snap.link_y):02x} f={total[0]}"
        )
        shot("settled", snap)
        _dump_tiles(ram)
        blocked = _seed_nonsand(ram)
        print(f"seeded non-sand cells={len(blocked)} goal={SOUTH_GAP}")
        start_xy = (int(snap.link_x), int(snap.link_y))
        grid = OccupancyGrid(
            blocked=set(blocked),
            xmin=EDGE_WEST_X,
            xmax=EDGE_EAST_X,
            ymin=EDGE_NORTH_Y,
            ymax=EDGE_SOUTH_Y,
        )
        path0 = grid.shortest_path(start_xy, SOUTH_GAP)
        print(
            f"offline path from {start_xy}: "
            f"{'none' if path0 is None else f'len={len(path0)} first={path0[:8]} last={path0[-4:]}'}"
        )

        walker = OccupancyWalker(grid=grid, goal=SOUTH_GAP)
        reached = False
        last_screen = snap.screen
        print(f"live start dir={walker.next_dir(start_xy)} last_dir={walker.last_dir}")
        walker.last_dir = None
        walker.last_xy = None
        walker.path = None
        for i in range(WALK_MAX):
            snap = read_snapshot(env.get_ram())
            if snap.screen != last_screen or snap.level != 0:
                shot("left_screen", snap)
                print(
                    f"LEFT_SCREEN f={total[0]} L{snap.level} "
                    f"s=0x{snap.screen:02x} ({snap.link_x},{snap.link_y}) mode={snap.mode}"
                )
                break
            xy = (int(snap.link_x), int(snap.link_y))
            if abs(xy[0] - SOUTH_GAP[0]) <= 4 and xy[1] >= EDGE_SOUTH_Y - 4:
                reached = True
                shot("south_gap", snap)
                print(f"REACHED south gap {xy} f={total[0]} misses={walker.misses}")
                break
            walker.observe(xy)
            direction = walker.next_dir(xy)
            if direction is None:
                shot("no_path", snap)
                print(
                    f"NO_PATH {xy} f={total[0]} misses={walker.misses} "
                    f"blocked={len(walker.grid.blocked)}"
                )
                break
            # Do not slash on the first ring walk: a miss-grade from a swing
            # stall is not occupancy evidence (octorok sits on the west sand).
            obs, *_ = env.step(nes_action(direction))
            obs_box[0] = obs
            total[0] += 1
            if assist is not None:
                assist.apply_env(env, frame=total[0])
            if i % 250 == 0:
                samples.append(
                    {
                        "f": total[0],
                        "xy": xy,
                        "dir": direction,
                        "misses": walker.misses,
                    }
                )
                shot(f"sample_{i:04d}", snap)
                print(
                    f"sample f={total[0]} {xy} dir={direction} misses={walker.misses}"
                )

        snap = read_snapshot(env.get_ram())
        shot("final", snap)
        payload = {
            "bead": "rr-6o7.1",
            "from_state": args.from_state,
            "settled_xy": [int(snap.link_x), int(snap.link_y)],
            "screen": int(snap.screen),
            "mode": int(snap.mode),
            "reached_south_gap": reached,
            "misses": walker.misses,
            "blocked": len(walker.grid.blocked),
            "nonsand_seeded": len(blocked),
            "frames": total[0],
            "samples": samples,
            "writes": 0,
            "route_eligible": False,
        }
        out = RECORDINGS_DIR / f"{args.tag}.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print("report", json.dumps(payload, indent=2))
        print("wrote", out)
    finally:
        env.close()


if __name__ == "__main__":
    main()

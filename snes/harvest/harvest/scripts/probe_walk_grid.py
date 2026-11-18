#!/usr/bin/env python3
"""Probe-walk a map into an 8-pixel lattice of verified hops.

Some maps (festival grounds) have no walkability data in ``map_config``. This
loads a pin, then walks every direction from every reachable spot using
emulator save/restore, and writes which 8-pixel hops move the farmer. NPC
bodies show up as blocked cells, so run it where NPCs stand as in play.

    HEADLESS=1 uv run python -m harvest.scripts.probe_walk_grid \\
        --state <pin> --setup "up:25,left:80,wait:300" --tilemap 8 \\
        --out harvest/data/flower_festival_walk.json

``--setup`` is a comma list of ``dir:frames`` / ``wait:frames`` steps run before
the probe (the festival entry walks off the town edge). The lattice origin is
the position after setup; cell ``(i, j)`` is ``origin + 8 * (i, j)``. The mask
bits are 1 up, 2 down, 4 left, 8 right.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

STEP = 8
DIRS = {"up": (0, -1, 1), "down": (0, 1, 2), "left": (-1, 0, 4), "right": (1, 0, 8)}
TOLERANCE = 3  # the first frames after a stop walk a little short


def probe_hop(em, step, pos, tilemap, state: bytes, name: str, expect_tilemap: int, *, exact: bool = False):
    """Restore ``state``, walk one hop, report (moved, state_after)."""
    dx, dy, _ = DIRS[name]
    em.set_state(state)
    step()
    x0, y0 = pos()
    goal = (x0 + dx * STEP, y0 + dy * STEP)
    for _ in range(STEP * 3 if exact else STEP):
        step(**{name: True})
        if exact and pos() == goal:
            break
    x1, y1 = pos()
    moved = (
        abs(x1 - goal[0]) <= (0 if exact else TOLERANCE)
        and abs(y1 - goal[1]) <= (0 if exact else TOLERANCE)
        and tilemap() == expect_tilemap
    )
    return moved, (em.get_state() if moved else None)


def explore(em, step, pos, tilemap, expect_tilemap: int, limit: int = 4000, *, exact: bool = False) -> Dict[Tuple[int, int], int]:
    """BFS over lattice cells, every cell probing all four hops. Returns cell -> mask."""
    em.set_state(em.get_state())
    states = {(0, 0): em.get_state()}
    masks: Dict[Tuple[int, int], int] = {}
    queue = collections.deque([(0, 0)])
    probes = 0
    while queue and probes < limit:
        cell = queue.popleft()
        mask = 0
        for name, (dx, dy, bit) in DIRS.items():
            moved, after = probe_hop(em, step, pos, tilemap, states[cell], name, expect_tilemap, exact=exact)
            probes += 1
            if not moved:
                continue
            mask |= bit
            nxt = (cell[0] + dx, cell[1] + dy)
            if nxt not in states:
                states[nxt] = after
                queue.append(nxt)
        masks[cell] = mask
    return masks


def run_setup(step, setup: str) -> None:
    for token in filter(None, setup.split(",")):
        name, _, count = token.partition(":")
        for _ in range(int(count)):
            step() if name == "wait" else step(**{name: True})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", required=True)
    ap.add_argument("--setup", default="")
    ap.add_argument("--tilemap", type=lambda s: int(s, 0), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exact", action="store_true", help="Require exact 8-pixel endpoints so hop drift cannot accumulate.")
    args = ap.parse_args()
    os.environ.setdefault("HEADLESS", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    from harvest.paths import ensure_monorepo_on_path

    ensure_monorepo_on_path()
    from harvest.core.ram_catalog import read_ram_value
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.tasks.nav import get_pos_from_ram, make_action

    env = make_harvest_env(args.state)
    env.reset()
    ram = {"w": env.get_ram()}

    def step(**buttons):
        env.step(make_action(**buttons))
        ram["w"] = env.get_ram()

    def pos() -> Tuple[int, int]:
        p = get_pos_from_ram(ram["w"])
        return p.x, p.y

    def tilemap() -> int:
        return int(read_ram_value(ram["w"], "tilemap", raw=True))

    em = env.unwrapped.em
    run_setup(step, args.setup)
    origin = pos()
    masks = explore(em, step, pos, tilemap, args.tilemap, exact=args.exact)
    cells: List[List[int]] = sorted([i, j, m] for (i, j), m in masks.items() if m)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"tilemap": args.tilemap, "origin": list(origin), "step": STEP, "cells": cells},
                   separators=(",", ":"))
    )
    print(f"origin={origin} cells={len(cells)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

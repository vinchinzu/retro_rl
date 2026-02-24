#!/usr/bin/env python3
"""Map discovery tool — replay recordings or play interactively to discover
tilemap IDs, walkable tiles, and transition positions.

Usage:
    # Replay a recorded task, logging map transitions and tile frequencies
    uv run python utils/map_discovery.py --state Y1_Spring_D1_Dawn --task ship_berry

    # Interactive mode: human plays, HUD shows tilemap/position/tile-under-player
    uv run python utils/map_discovery.py --state Y1_Spring_D1_Dawn --interactive

    # Replay and dump tile frequency tables per tilemap
    uv run python utils/map_discovery.py --state Y1_Spring_D1_Farm --task buy_potato_seeds --dump-tiles
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HARVEST_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(HARVEST_DIR)
for d in (HARVEST_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import numpy as np

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import stable_retro as retro

from farm_clearer import (
    ADDR_MAP,
    ADDR_TILEMAP,
    ADDR_X,
    ADDR_Y,
    MAP_WIDTH,
    TILE_SIZE,
    get_tile_at,
)

INTEGRATION_PATH = os.path.join(HARVEST_DIR, "custom_integrations")
STATES_DIR = os.path.join(INTEGRATION_PATH, "HarvestMoon-Snes")
TASKS_DIR = os.path.join(HARVEST_DIR, "tasks")
retro.data.Integrations.add_custom_path(INTEGRATION_PATH)


def make_env(state: Optional[str] = None):
    kwargs = {
        "game": "HarvestMoon-Snes",
        "inttype": retro.data.Integrations.ALL,
        "use_restricted_actions": retro.Actions.ALL,
        "render_mode": "rgb_array",
    }
    if state:
        kwargs["state"] = state
    return retro.make(**kwargs)


def get_pos(ram: np.ndarray) -> Tuple[int, int]:
    x = int(ram[ADDR_X]) + (int(ram[ADDR_X + 1]) << 8)
    y = int(ram[ADDR_Y]) + (int(ram[ADDR_Y + 1]) << 8)
    return (x, y)


def get_tilemap(ram: np.ndarray) -> int:
    return int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0


def get_tile_under_player(ram: np.ndarray) -> int:
    x, y = get_pos(ram)
    tx, ty = x // TILE_SIZE, y // TILE_SIZE
    return get_tile_at(ram, tx, ty)


def load_task_frames(task_name: str) -> List[List[int]]:
    path = os.path.join(TASKS_DIR, f"{task_name}.json")
    with open(path) as f:
        data = json.load(f)
    return data.get("frames", [])


def replay_task(state: str, task_name: str, dump_tiles: bool = False,
                analyze: bool = False):
    """Replay a recorded task, logging tilemap transitions and tile stats."""
    frames = load_task_frames(task_name)
    print(f"Replaying {task_name}: {len(frames)} frames from state {state}")

    env = make_env(state)
    env.reset()
    ram = env.get_ram()

    prev_tilemap = get_tilemap(ram)
    prev_pos = get_pos(ram)
    transitions: List[Dict] = []
    tile_counts: Dict[int, Counter] = defaultdict(Counter)
    a_presses: List[Tuple] = []
    all_positions: List[Tuple] = []

    print(f"Start: tilemap=0x{prev_tilemap:02X} pos={prev_pos}")

    for i, frame in enumerate(frames):
        action = np.array(frame, dtype=np.int32)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"Game ended at frame {i}")
            break

        ram = env.get_ram()
        tilemap = get_tilemap(ram)
        pos = get_pos(ram)
        tile = get_tile_under_player(ram)

        # Track tile frequencies per map
        tile_counts[tilemap][tile] += 1
        all_positions.append((i, pos[0], pos[1], tilemap))

        # Track A-button presses
        if action[8] == 1:
            a_presses.append((i, pos[0], pos[1], tilemap))

        # Detect map transitions
        if tilemap != prev_tilemap:
            transition = {
                "frame": i,
                "from_map": f"0x{prev_tilemap:02X}",
                "to_map": f"0x{tilemap:02X}",
                "exit_pos": prev_pos,
                "entry_pos": pos,
            }
            transitions.append(transition)
            print(f"  Frame {i:5d}: 0x{prev_tilemap:02X} → 0x{tilemap:02X}"
                  f"  exit=({prev_pos[0]:3d},{prev_pos[1]:3d})"
                  f"  entry=({pos[0]:3d},{pos[1]:3d})")

        prev_tilemap = tilemap
        prev_pos = pos

    env.close()

    # Summary
    print(f"\n{'='*60}")
    print(f"Transitions: {len(transitions)}")
    for t in transitions:
        print(f"  {t['from_map']} → {t['to_map']} @ frame {t['frame']}"
              f"  exit={t['exit_pos']} entry={t['entry_pos']}")

    print(f"\nMaps visited: {sorted(f'0x{m:02X}' for m in tile_counts.keys())}")

    if analyze and all_positions:
        xs = [p[1] for p in all_positions]
        ys = [p[2] for p in all_positions]
        print(f"\nPosition range: x=[{min(xs)},{max(xs)}] y=[{min(ys)},{max(ys)}]")
        print(f"Tile range: tx=[{min(xs)//TILE_SIZE},{max(xs)//TILE_SIZE}]"
              f" ty=[{min(ys)//TILE_SIZE},{max(ys)//TILE_SIZE}]")
        print(f"\nA-button presses ({len(a_presses)}):")
        for frame, x, y, tm in a_presses[:30]:
            tx, ty = x // TILE_SIZE, y // TILE_SIZE
            print(f"  frame={frame:5d} px=({x:3d},{y:3d}) tile=({tx:2d},{ty:2d})"
                  f" tilemap=0x{tm:02X}")
        if len(a_presses) > 30:
            print(f"  ... {len(a_presses) - 30} more")

    if dump_tiles:
        print(f"\n{'='*60}")
        print("Tile frequency tables (top 20 per map):")
        for tilemap_id in sorted(tile_counts.keys()):
            counts = tile_counts[tilemap_id]
            total = sum(counts.values())
            print(f"\n  Tilemap 0x{tilemap_id:02X} ({total} samples):")
            for tile_id, count in counts.most_common(20):
                pct = 100 * count / total
                print(f"    0x{tile_id:02X}: {count:6d} ({pct:5.1f}%)")


def interactive_mode(state: str):
    """Interactive mode: human plays, HUD shows tilemap/position/tile info."""
    import pygame

    env = make_env(state)
    obs, info = env.reset()

    h, w = obs.shape[:2]
    scale = 3
    pygame.init()
    screen = pygame.display.set_mode((w * scale, h * scale))
    pygame.display.set_caption("Map Discovery - Interactive")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 12)

    from farm_clearer import make_action
    from retro_harness import keyboard_action

    prev_tilemap = None
    tile_log: Counter = Counter()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        action = np.zeros(12, dtype=np.int32)
        keyboard_action(keys, action, pygame)

        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

        ram = env.get_ram()
        tilemap = get_tilemap(ram)
        pos = get_pos(ram)
        tx, ty = pos[0] // TILE_SIZE, pos[1] // TILE_SIZE
        tile = get_tile_under_player(ram)

        tile_log[tile] += 1

        if prev_tilemap is not None and tilemap != prev_tilemap:
            print(f"MAP CHANGE: 0x{prev_tilemap:02X} → 0x{tilemap:02X}"
                  f" pos=({pos[0]},{pos[1]}) tile=({tx},{ty})")
        prev_tilemap = tilemap

        # Render
        surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
        scaled = pygame.transform.scale(surf, (w * scale, h * scale))
        screen.blit(scaled, (0, 0))

        lines = [
            f"Tilemap: 0x{tilemap:02X}",
            f"Pos: ({pos[0]},{pos[1]}) tile=({tx},{ty})",
            f"Tile under: 0x{tile:02X}",
            f"Unique tiles walked: {len(tile_log)}",
        ]
        for i, line in enumerate(lines):
            text = font.render(line, True, (255, 255, 0))
            screen.blit(text, (5, 5 + i * 15))

        pygame.display.flip()

        fast = keys[pygame.K_TAB]
        if not fast:
            clock.tick(60)

    # Dump walked-on tiles
    print(f"\nTiles walked on (by frequency):")
    for tile_id, count in tile_log.most_common():
        print(f"  0x{tile_id:02X}: {count}")

    env.close()
    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Map Discovery Tool")
    parser.add_argument("--state", required=True, help="Save state name")
    parser.add_argument("--task", type=str, default=None, help="Task recording to replay")
    parser.add_argument("--interactive", action="store_true", help="Interactive play mode")
    parser.add_argument("--dump-tiles", action="store_true", help="Dump tile frequency tables")
    parser.add_argument("--analyze", action="store_true", help="Show A-presses and position range")
    args = parser.parse_args()

    if args.interactive:
        interactive_mode(args.state)
    elif args.task:
        replay_task(args.state, args.task, dump_tiles=args.dump_tiles,
                    analyze=args.analyze)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

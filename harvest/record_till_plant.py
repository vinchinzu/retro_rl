#!/usr/bin/env python3
"""
Record a tilling + planting session for grass_planter discovery.

Loads Y1_Spring_Day01_06h00m, injects 99 grass seeds, and lets you play.
Logs tile changes every frame so we can see exactly what IDs correspond to
tilled soil, planted grass, etc.

Usage:
    .venv/bin/python record_till_plant.py

Controls:
    Arrows: Move | X(key): Use tool (Y btn) | V(key): Cycle tool (X btn)
    C(key): Confirm (A) | Z(key): Cancel (B) | A/S(key): L/R shoulder
    F1: Save recording + quit | ESC: Quit without saving
    TAB: Fast-forward | [ / ]: Speed

Instructions:
    1. Cycle to HOE (press V until you see the hoe)
    2. Walk into the farm area (the big open field)
    3. Face a clear tile and press X to hoe it (watch the log for tile change)
    4. Cycle to GRASS SEEDS (press V - item 0x0C)
    5. Face the tilled tile and press X to plant
    6. Press F1 when done
"""

import os
import sys
import json
import gzip

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import pygame
import stable_retro as retro

from retro_harness import (
    init_controller as _init_controller,
    controller_action,
    keyboard_action,
    sanitize_action,
)

INTEGRATION_PATH = os.path.join(SCRIPT_DIR, "custom_integrations")
STATES_DIR = os.path.join(INTEGRATION_PATH, "HarvestMoon-Snes")
TASKS_DIR = os.path.join(SCRIPT_DIR, "tasks")
retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

ADDR_X, ADDR_Y = 0x00D6, 0x00D8
ADDR_TOOL = 0x0921
ADDR_MAP = 0x09B6
MAP_WIDTH = 64
TILE_SIZE = 16

# Known tile meanings (will be updated by this session)
TILE_NAMES = {
    0x00: "empty",
    0x01: "untilled",
    0x02: "tilled?",
    0x07: "freshly_hoed",
    0x08: "cleared",
    0x70: "planted_grass?",
    0xA0: "path", 0xA1: "border", 0xA6: "pond_edge", 0xA8: "fence_area",
}

TOOL_NAMES = {
    0x00: "Empty", 0x01: "Sickle", 0x02: "Hoe", 0x03: "Hammer",
    0x04: "Axe", 0x0C: "GrassSeeds", 0x10: "WateringCan",
}


def init_controller():
    return _init_controller(pygame)


def get_player_tile(ram):
    x = int(ram[ADDR_X]) + (int(ram[ADDR_X + 1]) << 8)
    y = int(ram[ADDR_Y]) + (int(ram[ADDR_Y + 1]) << 8)
    return x // TILE_SIZE, y // TILE_SIZE, x, y


def snapshot_farm(ram, x_min=3, y_min=25, x_max=45, y_max=60):
    """Return dict of (tx,ty) -> tile_id for the farm area."""
    tiles = {}
    for ty in range(y_min, y_max + 1):
        for tx in range(x_min, x_max + 1):
            addr = ADDR_MAP + ty * MAP_WIDTH + tx
            if addr < len(ram):
                tiles[(tx, ty)] = int(ram[addr])
    return tiles


def main():
    state = "Y1_Spring_Day01_06h00m"
    scale = 3

    pygame.init()
    env = retro.make(
        game="HarvestMoon-Snes",
        inttype=retro.data.Integrations.ALL,
        use_restricted_actions=retro.Actions.ALL,
        render_mode="rgb_array",
        state=state,
    )
    obs, info = env.reset()

    # Inject grass seeds
    env.data.set_value("grass_seeds", 99)
    env.data.set_value("stamina", 100)
    obs, _, _, _, info = env.step(np.zeros(12, dtype=np.int32))

    h, w = obs.shape[:2]
    screen = pygame.display.set_mode((w * scale, h * scale))
    pygame.display.set_caption("Till & Plant Recorder")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 11)
    joystick = init_controller()

    print("=" * 60)
    print("TILL & PLANT RECORDER")
    print("=" * 60)
    print("Grass seeds injected: 99")
    print("Cycle tools with V key until you see Hoe or GrassSeeds")
    print("Use X key to use tool on tile you're facing")
    print("Tile changes will be logged automatically")
    print("F1 = save & quit | ESC = quit")
    print("=" * 60)

    ram = env.get_ram()
    prev_snapshot = snapshot_farm(ram)
    prev_tool = int(ram[ADDR_TOOL])
    frames_recorded = []
    frame_count = 0
    tile_transitions = []  # list of (frame, tx, ty, old_id, new_id)

    speed_levels = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    speed_idx = 2
    running = True
    save = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F1:
                    save = True
                    running = False
                elif event.key == pygame.K_LEFTBRACKET:
                    speed_idx = max(0, speed_idx - 1)
                elif event.key == pygame.K_RIGHTBRACKET:
                    speed_idx = min(len(speed_levels) - 1, speed_idx + 1)

        keys = pygame.key.get_pressed()
        action = np.zeros(12, dtype=np.int32)
        keyboard_action(keys, action, pygame)
        controller_action(joystick, action)
        sanitize_action(action)
        frames_recorded.append(action.tolist())

        obs, _, _, _, info = env.step(action)

        # Keep stamina and seeds topped up
        env.data.set_value("stamina", 100)
        if frame_count % 60 == 0:
            env.data.set_value("grass_seeds", 99)

        ram = env.get_ram()
        frame_count += 1

        # Detect tile changes
        cur_snapshot = snapshot_farm(ram)
        for pos, new_id in cur_snapshot.items():
            old_id = prev_snapshot.get(pos, new_id)
            if old_id != new_id:
                tx, ty = pos
                old_name = TILE_NAMES.get(old_id, f"0x{old_id:02X}")
                new_name = TILE_NAMES.get(new_id, f"0x{new_id:02X}")
                print(f"  [TILE] frame={frame_count} ({tx},{ty}): 0x{old_id:02X}({old_name}) -> 0x{new_id:02X}({new_name})")
                tile_transitions.append((frame_count, tx, ty, old_id, new_id))
        prev_snapshot = cur_snapshot

        # Detect tool changes
        cur_tool = int(ram[ADDR_TOOL])
        if cur_tool != prev_tool:
            name = TOOL_NAMES.get(cur_tool, f"0x{cur_tool:02X}")
            print(f"  [TOOL] frame={frame_count}: 0x{prev_tool:02X} -> 0x{cur_tool:02X} ({name})")
            prev_tool = cur_tool

        # Render
        fast_forward = keys[pygame.K_TAB]
        if not fast_forward:
            surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
            scaled = pygame.transform.scale(surf, (w * scale, h * scale))
            screen.blit(scaled, (0, 0))

            ptx, pty, px, py = get_player_tile(ram)
            tool_name = TOOL_NAMES.get(cur_tool, f"0x{cur_tool:02X}")
            seeds = int(ram[0x0927]) if 0x0927 < len(ram) else 0
            lines = [
                f"Frame: {frame_count} | Tool: {tool_name} | Seeds: {seeds}",
                f"Tile: ({ptx},{pty}) Px: ({px},{py})",
                f"Transitions: {len(tile_transitions)} | F1=Save ESC=Quit",
            ]
            for i, line in enumerate(lines):
                text = font.render(line, True, (255, 255, 255))
                screen.blit(text, (5, 5 + i * 14))

            pygame.display.flip()
            clock.tick(int(60 * speed_levels[speed_idx]))
        else:
            if frame_count % 60 == 0:
                pygame.event.pump()

    # Summary
    print()
    print("=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    print(f"Frames: {frame_count}")
    print(f"Tile transitions: {len(tile_transitions)}")
    for f, tx, ty, old, new in tile_transitions:
        old_name = TILE_NAMES.get(old, "?")
        new_name = TILE_NAMES.get(new, "?")
        print(f"  frame={f:5d} ({tx},{ty}): 0x{old:02X}({old_name}) -> 0x{new:02X}({new_name})")

    if save and frames_recorded:
        # Save as a task
        task_data = {
            "name": "till_and_plant_grass",
            "frames": frames_recorded,
            "start_state": state,
            "metadata": {
                "frame_count": len(frames_recorded),
                "tile_transitions": [
                    {"frame": f, "x": tx, "y": ty, "old": old, "new": new}
                    for f, tx, ty, old, new in tile_transitions
                ],
            },
        }
        path = os.path.join(TASKS_DIR, "till_and_plant_grass.json")
        with open(path, "w") as fp:
            json.dump(task_data, fp, indent=2)
        print(f"\nSaved: {path}")

        # Save end state
        end_state = env.em.get_state()
        state_path = os.path.join(STATES_DIR, "Y1_After_Till_Plant.state")
        with gzip.open(state_path, "wb") as fp:
            fp.write(end_state)
        print(f"Saved end state: {state_path}")

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()

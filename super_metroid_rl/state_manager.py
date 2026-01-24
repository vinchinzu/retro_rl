#!/usr/bin/env python3
"""
State Manager for Super Metroid RL

Create, rename, and manage save states with proper naming conventions.

Naming Convention:
    {Phase}_{Room}_{Direction}_{Items}.state

    Phase: Descent, Return, Boss
    Room: Short room name
    Direction: toX (where X is target)
    Items: withMorph, withMissiles, etc.

Examples:
    Descent_Parlor_toClimb.state
    Return_Parlor_toFlyway_withMorph.state
    Return_Climb_toParlor_withMorph.state

Usage:
    # List all states
    python state_manager.py list

    # Create a new state interactively
    python state_manager.py create

    # Copy and rename a state
    python state_manager.py rename "Parlor and Alcatraz [from Climb]" "Return_Parlor_toFlyway_withMorph"

    # Play and save a new state
    python state_manager.py record --name "Return_Parlor_toFlyway_withMorph"
"""

import os
import sys
import json
import shutil
import argparse
from typing import Dict, Optional, List

# Import shared controls (sets SDL_VIDEODRIVER)
from controls import (
    init_controller, get_controller_action, get_keyboard_action,
    sanitize_action, print_controls, KEYBOARD_MAP
)

import numpy as np
import pygame
import stable_retro as retro

# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTEGRATION_PATH = os.path.join(SCRIPT_DIR, "custom_integrations")
STATES_DIR = os.path.join(INTEGRATION_PATH, "SuperMetroid-Snes")
WORLD_MAP_PATH = os.path.join(SCRIPT_DIR, "world_map.json")

# Register custom integration path BEFORE any retro operations
retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

def get_available_states():
    """Get list of available state names."""
    import glob
    pattern = os.path.join(STATES_DIR, "*.state")
    states = glob.glob(pattern)
    return [os.path.basename(s).replace(".state", "") for s in states]

def find_state(name: str) -> Optional[str]:
    """Find a state by partial name match."""
    # Clean up the name (remove newlines, extra spaces)
    name = ' '.join(name.split())

    available = get_available_states()
    # Exact match
    if name in available:
        return name
    # Partial match
    matches = [s for s in available if name.lower() in s.lower()]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"Multiple matches for '{name}':")
        for m in matches[:5]:
            print(f"  {m}")
        return None
    return None

# =============================================================================
# WORLD MAP
# =============================================================================
def load_world_map() -> Dict[str, str]:
    if os.path.exists(WORLD_MAP_PATH):
        with open(WORLD_MAP_PATH, 'r') as f:
            return json.load(f)
    return {}

WORLD_MAP = load_world_map()

def get_room_name(room_id: int) -> str:
    """Get room name from ID."""
    room_hex = f"0x{room_id:x}"
    for name, rid in WORLD_MAP.items():
        if rid.lower() == room_hex.lower():
            return name
    return f"Unknown_0x{room_id:04X}"

# =============================================================================
# ITEM BITS
# =============================================================================
ITEM_BITS = {
    0x0001: "Morph",
    0x0002: "Bombs",
    0x0004: "SpringBall",
    0x0008: "HiJump",
    0x0020: "Varia",
    0x0100: "Gravity",
    0x0200: "SpeedBooster",
    0x1000: "SpaceJump",
    0x2000: "ScrewAttack",
}

def get_items_string(items: int) -> str:
    """Get short items string from bitmask."""
    parts = []
    for bit, name in ITEM_BITS.items():
        if items & bit:
            parts.append(name)
    return "_".join(parts) if parts else "NoItems"

# =============================================================================
# KEY MAPPING
# =============================================================================
KEY_MAP = {
    pygame.K_z: 0,      # B (run)
    pygame.K_x: 1,      # Y (item cancel)
    pygame.K_TAB: 2,    # Select
    pygame.K_RETURN: 3, # Start
    pygame.K_UP: 4,     # Up
    pygame.K_DOWN: 5,   # Down
    pygame.K_LEFT: 6,   # Left
    pygame.K_RIGHT: 7,  # Right
    pygame.K_c: 8,      # A (jump)
    pygame.K_v: 9,      # X (shoot)
    pygame.K_a: 10,     # L (aim up)
    pygame.K_s: 11,     # R (aim down)
}

# =============================================================================
# LIST STATES
# =============================================================================
def list_states(filter_str: Optional[str] = None):
    """List all available states."""
    import glob

    pattern = os.path.join(STATES_DIR, "*.state")
    states = glob.glob(pattern)

    print("\n" + "="*60)
    print("AVAILABLE STATES")
    print("="*60)

    # Group by naming style
    old_style = []  # "Room [from X].state"
    new_style = []  # "Phase_Room_Direction.state"
    numbered = []   # "RoomN.state"

    for state_path in sorted(states):
        name = os.path.basename(state_path).replace(".state", "")

        if filter_str and filter_str.lower() not in name.lower():
            continue

        if name.startswith("Room") and name[4:].isdigit():
            numbered.append(name)
        elif "[from" in name:
            old_style.append(name)
        else:
            new_style.append(name)

    if new_style:
        print("\n--- Route States (new naming) ---")
        for name in new_style:
            print(f"  {name}")

    if old_style:
        print("\n--- Room States (old naming) ---")
        for name in old_style:
            print(f"  {name}")

    if numbered:
        print("\n--- Numbered States ---")
        for name in numbered:
            print(f"  {name}")

    total = len(old_style) + len(new_style) + len(numbered)
    print(f"\nTotal: {total} states")
    print("="*60)


# =============================================================================
# RENAME STATE
# =============================================================================
def rename_state(old_name: str, new_name: str):
    """Copy a state file with a new name."""
    # Handle various input formats
    if not old_name.endswith(".state"):
        old_name_file = old_name + ".state"
    else:
        old_name_file = old_name

    if not new_name.endswith(".state"):
        new_name_file = new_name + ".state"
    else:
        new_name_file = new_name

    old_path = os.path.join(STATES_DIR, old_name_file)
    new_path = os.path.join(STATES_DIR, new_name_file)

    if not os.path.exists(old_path):
        print(f"Error: State not found: {old_path}")
        print("\nTry one of these:")
        list_states(old_name.split()[0] if " " in old_name else old_name[:10])
        return

    if os.path.exists(new_path):
        print(f"Warning: {new_name_file} already exists. Overwrite? [y/N]")
        if input().lower() != 'y':
            print("Cancelled.")
            return

    shutil.copy2(old_path, new_path)
    print(f"Created: {new_name_file}")
    print(f"(Copy of {old_name_file})")


# =============================================================================
# RECORD NEW STATE
# =============================================================================
def record_state(
    start_state: str = "ZebesStart",
    target_name: Optional[str] = None,
    scale: int = 2
):
    """
    Play the game and save a new state with F5.

    Controls:
        F5: Save current position as new state
        F1: Exit without saving
        ESC: Exit without saving
    """
    # Verify state exists
    matched_state = find_state(start_state)
    if matched_state is None:
        print(f"Error: State '{start_state}' not found.")
        print("\nAvailable states containing that name:")
        available = get_available_states()
        matches = [s for s in available if start_state.split()[0].lower() in s.lower()]
        for m in matches[:10]:
            print(f"  {m}")
        return None

    if matched_state != start_state:
        print(f"Using matched state: {matched_state}")
        start_state = matched_state

    pygame.init()

    try:
        env = retro.make(
            game="SuperMetroid-Snes",
            state=start_state,
            inttype=retro.data.Integrations.ALL,
            use_restricted_actions=retro.Actions.ALL,
            render_mode='rgb_array'
        )
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("\nTry using the exact state name from:")
        list_states(start_state.split()[0] if " " in start_state else None)
        return None

    obs, info = env.reset()
    h, w = obs.shape[0], obs.shape[1]

    screen = pygame.display.set_mode((w * scale, h * scale))
    pygame.display.set_caption("State Manager - F5 to save state")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('monospace', 12)

    print("\n" + "="*60)
    print("STATE RECORDING MODE")
    print("="*60)
    print(f"Starting from: {start_state}")

    # Initialize controller
    joystick = init_controller()
    print_controls(joystick)
    print("  F5/SELECT: Save state | ESC/START+SELECT: Exit")
    print("="*60)

    running = True
    saved_states = []
    save_counter = 0

    def save_current_state():
        nonlocal save_counter
        state_data = env.em.get_state()
        room_id = info.get('room_id', 0)
        room_name = get_room_name(room_id)
        items = info.get('collected_items', 0) or 0
        items_str = get_items_string(items)

        short_room = room_name.replace(" ", "").replace("and", "")[:20]
        if target_name:
            state_name = f"{target_name}_{save_counter}" if save_counter > 0 else target_name
        else:
            state_name = f"{short_room}_{items_str}"

        state_path = os.path.join(STATES_DIR, f"{state_name}.state")
        with open(state_path, 'wb') as f:
            f.write(state_data)

        save_counter += 1
        saved_states.append(state_name)
        print(f"\n[SAVED] {state_name}.state")
        print(f"  Room: {room_name} | Items: {items_str} | Total: {save_counter}")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F5:
                    save_current_state()
            elif event.type == pygame.JOYBUTTONDOWN:
                # SELECT (6) = save state
                if event.button == 6:
                    save_current_state()
                # START (7) + SELECT (6) = exit
                if joystick and joystick.get_button(7) and joystick.get_button(6):
                    running = False

        # Build action
        keys = pygame.key.get_pressed()
        action = np.zeros(12, dtype=np.int32)
        get_keyboard_action(keys, action)
        get_controller_action(joystick, action)
        sanitize_action(action)

        # Step
        obs, reward, terminated, truncated, info = env.step(action)

        # Render
        surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
        scaled = pygame.transform.scale(surf, (w * scale, h * scale))
        screen.blit(scaled, (0, 0))

        # HUD
        room_id = info.get('room_id', 0)
        room_name = get_room_name(room_id)
        hp = info.get('health', 0)
        missiles = info.get('missiles', 0)
        items = info.get('collected_items', 0) or 0

        hud_lines = [
            f"Room: {room_name}",
            f"HP: {hp} | Missiles: {missiles}",
            f"Items: {get_items_string(items)}",
            f"F5=QuickSave | F6=NamedSave | Saves:{save_counter}"
        ]

        for i, line in enumerate(hud_lines):
            text = font.render(line, True, (0, 255, 0))
            screen.blit(text, (5, 5 + i * 14))

        pygame.display.flip()
        clock.tick(60)

        if terminated:
            print("Episode ended. Restarting...")
            obs, info = env.reset()

    env.close()
    pygame.quit()

    # Summary
    if saved_states:
        print(f"\n{'='*60}")
        print(f"SESSION COMPLETE - {len(saved_states)} states saved:")
        for name in saved_states:
            print(f"  {name}.state")
        print(f"{'='*60}")
        return saved_states
    else:
        print("\nNo states saved.")
        return None


# =============================================================================
# SUGGEST NAMES
# =============================================================================
def suggest_names():
    """Print suggested state names for the Zebes -> Torizo route."""
    print("\n" + "="*60)
    print("SUGGESTED STATE NAMES FOR ROUTE")
    print("="*60)

    print("\n--- DESCENT PHASE (getting morph ball) ---")
    descent = [
        ("Descent_Landing_toParlor", "ZebesStart, heading left to Parlor"),
        ("Descent_Parlor_toClimb", "After entering Parlor from Landing, heading down-left to Climb"),
        ("Descent_Climb_toPitRoom", "After entering Climb from Parlor, heading down"),
        ("Descent_PitRoom_toElevator", "After entering Pit Room from Climb, heading down"),
        ("Descent_Elevator_toMorphRoom", "After entering Elevator from Pit Room, heading down"),
        ("Descent_MorphRoom_collectItem", "In Morph Ball Room, ready to collect"),
    ]
    for name, desc in descent:
        print(f"  {name}")
        print(f"    -> {desc}")

    print("\n--- RETURN PHASE (after morph ball, going to Torizo) ---")
    return_trip = [
        ("Return_MorphRoom_toElevator_withMorph", "After getting morph, heading up to Elevator"),
        ("Return_Elevator_toPitRoom_withMorph", "Coming from Morph Room, heading up"),
        ("Return_PitRoom_toClimb_withMorph", "Coming from Elevator, heading up"),
        ("Return_Climb_toParlor_withMorph", "Coming from Pit Room, heading up"),
        ("Return_Parlor_toFlyway_withMorph", "Coming from Climb, heading right to Flyway"),
        ("Return_Flyway_toTorizo_withMorph", "Heading right to Bomb Torizo Room"),
    ]
    for name, desc in return_trip:
        print(f"  {name}")
        print(f"    -> {desc}")

    print("\n--- BOSS PHASE ---")
    boss = [
        ("Boss_Torizo_fight_withMorph", "In Bomb Torizo Room, ready to fight"),
    ]
    for name, desc in boss:
        print(f"  {name}")
        print(f"    -> {desc}")

    print("\n" + "="*60)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="State Manager for Super Metroid RL")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # List states
    ls = subparsers.add_parser('list', help='List all states')
    ls.add_argument('--filter', type=str, help='Filter by name substring')

    # Rename state
    ren = subparsers.add_parser('rename', help='Copy and rename a state')
    ren.add_argument('old_name', type=str, help='Current state name')
    ren.add_argument('new_name', type=str, help='New state name')

    # Record new state
    rec = subparsers.add_parser('record', help='Play and save a new state')
    rec.add_argument('--start', type=str, default='ZebesStart', help='Starting state')
    rec.add_argument('--name', type=str, help='Target state name')
    rec.add_argument('--scale', type=int, default=2, help='Display scale')

    # Suggest names
    subparsers.add_parser('suggest', help='Show suggested state names')

    args = parser.parse_args()

    if args.command == 'list':
        list_states(filter_str=args.filter)
    elif args.command == 'rename':
        rename_state(args.old_name, args.new_name)
    elif args.command == 'record':
        record_state(start_state=args.start, target_name=args.name, scale=args.scale)
    elif args.command == 'suggest':
        suggest_names()
    else:
        parser.print_help()


if __name__ == "__main__":
    import time
    main()

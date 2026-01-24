#!/usr/bin/env python3
"""
Identify and rename generic Room*.state files to their proper room names.
Loads each state file and checks the initial room ID to determine the actual name.
"""

import os
import sys
import gzip
import struct
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stable_retro as retro
from recording.world import WorldMap

def get_room_id_from_state(state_path):
    """Load a state and extract the initial room ID."""
    env = retro.make(
        game="SuperMetroid-Snes",
        state=state_path,
        use_restricted_actions=retro.Actions.ALL
    )
    _, info = env.reset()
    room_id = info.get('room_id', 0)
    env.close()
    return room_id

def main():
    state_dir = "custom_integrations/SuperMetroid-Snes"
    world = WorldMap("world_map.json")

    # Find all Room*.state files
    room_files = []
    for f in os.listdir(state_dir):
        if f.startswith("Room") and f.endswith(".state"):
            # Extract number
            try:
                num = int(f.replace("Room", "").replace(".state", ""))
                room_files.append((num, f))
            except ValueError:
                continue

    if not room_files:
        print("No generic Room*.state files found to rename.")
        return

    room_files.sort()

    print("Analyzing generic Room files...")
    print("=" * 70)

    renames = []
    for num, filename in room_files:
        state_path = os.path.join(state_dir, filename)

        try:
            print(f"\nChecking {filename}...")
            room_id = get_room_id_from_state(state_path)
            room_name = world.get_room_name(room_id)

            if room_name:
                print(f"  Room ID: {hex(room_id)} -> '{room_name}'")
                new_filename = f"{room_name}.state"
                new_path = os.path.join(state_dir, new_filename)

                if os.path.exists(new_path):
                    print(f"  ⚠ '{new_filename}' already exists - keeping as '{filename}'")
                else:
                    renames.append((state_path, new_path, filename, new_filename))
            else:
                print(f"  Room ID: {hex(room_id)} -> Unknown (not in world map)")
        except Exception as e:
            print(f"  Error: {e}")

    if not renames:
        print("\n✓ No renames needed")
        return

    print("\n" + "=" * 70)
    print("Proposed renames:")
    for _, _, old_name, new_name in renames:
        print(f"  {old_name:<25} -> {new_name}")

    response = input("\nApply these renames? (y/n): ").strip().lower()
    if response == 'y':
        for old_path, new_path, old_name, new_name in renames:
            shutil.move(old_path, new_path)
            print(f"  ✓ Renamed {old_name} -> {new_name}")

        # Also copy to retro state dir
        retro_state_dir = os.path.dirname(retro.data.get_file_path("SuperMetroid-Snes", "rom.sha"))
        print(f"\nCopying to retro state directory: {retro_state_dir}")
        for _, new_path, _, new_name in renames:
            retro_dest = os.path.join(retro_state_dir, new_name)
            shutil.copy(new_path, retro_dest)
            print(f"  ✓ Copied {new_name}")

        print("\n✓ All renames complete!")
    else:
        print("Cancelled.")

if __name__ == "__main__":
    main()

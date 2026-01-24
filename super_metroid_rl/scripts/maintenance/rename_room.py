
import json
import os
import argparse
import sys

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from recording.world import WorldMap

def main():
    parser = argparse.ArgumentParser(description="Rename rooms in the World Map.")
    parser.add_argument("old_name", help="Current name of the room (e.g. Room15)")
    parser.add_argument("new_name", help="New descriptive name (e.g. Parlor)")
    args = parser.parse_args()

    base_dir = project_root
    map_path = os.path.join(base_dir, "world_map.json")
    
    world = WorldMap(map_path)
    
    # Check if old name exists
    room_id = world.get_room_id(args.old_name)
    if not room_id:
        # Maybe user provided the ID directly? like 0x96ba
        try:
            if args.old_name.startswith("0x"):
                room_id = int(args.old_name, 16)
                real_old_name = world.get_room_name(room_id)
                if not real_old_name:
                    print(f"ID {args.old_name} not found in map.")
                    return
                print(f"Found ID {args.old_name} mapped to '{real_old_name}'.")
                args.old_name = real_old_name
        except ValueError:
            print(f"Room '{args.old_name}' not found in map.")
            return

    # Check if new name is taken
    if world.get_room_id(args.new_name):
        print(f"Error: Name '{args.new_name}' is already used.")
        return

    # Rename
    # WorldMap doesn't have a rename method, we have to manipulate data directly or add one.
    # Let's verify WorldMap structure.
    # It stores self.data = {Name: ID}
    
    world.data[args.new_name] = room_id
    del world.data[args.old_name]
    world.save()
    
    print(f"Success! Renamed '{args.old_name}' -> '{args.new_name}' (ID: {hex(room_id)})")
    
    # Also update best_times.json to reflect the rename?
    stats_path = os.path.join(base_dir, "best_times.json")
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        if args.old_name in stats:
            stats[args.new_name] = stats[args.old_name]
            del stats[args.old_name]
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=4, sort_keys=True)
            print("Updated best_times.json history as well.")

if __name__ == "__main__":
    main()

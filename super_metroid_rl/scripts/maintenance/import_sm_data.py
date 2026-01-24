import json
import os
import argparse
import glob
import sys

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from recording.world import WorldMap

def main():
    parser = argparse.ArgumentParser(description="Import room names from sm-json-data.")
    parser.add_argument("repo_path", help="Path to the cloned sm-json-data repository.")
    args = parser.parse_args()

    base_dir = project_root
    map_path = os.path.join(base_dir, "world_map.json")
    world = WorldMap(map_path)
    
    count = 0
    # Search for all json files in region/ folder
    region_dir = os.path.join(args.repo_path, "region")
    
    print(f"Scanning {region_dir}...")
    
    for root, dirs, files in os.walk(region_dir):
        for file in files:
            if file.endswith(".json"):
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, 'r') as f:
                        data = json.load(f)
                        
                    # Check for roomAddress
                    if "roomAddress" in data and "name" in data:
                        raw_addr = data["roomAddress"] # e.g. "0x791F8"
                        room_name = data["name"]
                        
                        # Convert address to our 16-bit int ID
                        # Assuming it's a hex string
                        if isinstance(raw_addr, str) and raw_addr.startswith("0x"):
                            val = int(raw_addr, 16)
                            short_id = val & 0xFFFF # Mask to lower 16 bits
                            
                            # Add to WorldMap
                            # Check if already exists with a generic name
                            existing_name = world.get_room_name(short_id)
                            
                            if existing_name:
                                if existing_name != room_name and existing_name.startswith("Room"):
                                    # Overwrite generic name
                                    print(f"Updating {hex(short_id)}: {existing_name} -> {room_name}")
                                    del world.data[existing_name]
                                    world.data[room_name] = short_id
                                    count += 1
                                elif existing_name == room_name:
                                    pass # Already correct
                                else:
                                    # Conflict? Keep existing for now unless force
                                    pass 
                            else:
                                # New room
                                print(f"Adding {hex(short_id)}: {room_name}")
                                world.data[room_name] = short_id
                                count += 1

                except Exception as e:
                    print(f"Error parsing {file}: {e}")

    world.save()
    print(f"Import complete. Updated/Added {count} rooms.")
    
    # Also update best_times.json?
    # This is trickier because we need to map old generic names to new names in the stats file.
    # But since we just deleted the old generic names from WorldMap, we can try to do a best-effort migration.
    # For now, let's keep it simple.

if __name__ == "__main__":
    main()

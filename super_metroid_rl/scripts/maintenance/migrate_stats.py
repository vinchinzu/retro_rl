import json
import os
import sys

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from recording.world import WorldMap

def main():
    base_dir = project_root
    stats_path = os.path.join(base_dir, "best_times.json")
    map_path = os.path.join(base_dir, "world_map.json")
    
    if not os.path.exists(stats_path):
        print("No stats found.")
        return

    world = WorldMap(map_path)
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)

    new_stats = {}
    renamed_count = 0
    merged_count = 0

    # Snapshot keys to avoid runtime modification issues
    old_keys = list(stats.keys())
    
    for key in old_keys:
        # Key could be "0x91f8" or "Room15"
        # We want to see if this corresponds to a Name in WorldMap
        
        real_name = None
        
        # 1. Try treating key as ID (if it looks like hex)
        if key.startswith("0x"):
            try:
                # world_map now stores mapping Name -> HexString
                # we need ID -> Name
                # use world.get_room_name which handles strings now
                real_name = world.get_room_name(key)
            except: pass
        
        # 2. If valid name found, but it differs from key
        if real_name and real_name != key:
            print(f"Migrating '{key}' -> '{real_name}'")
            
            if real_name not in new_stats:
                new_stats[real_name] = stats[key]
                renamed_count += 1
            else:
                # Merge!
                print(f"  Merging data into existing '{real_name}'")
                target = new_stats[real_name]
                source = stats[key]
                
                # Update bests
                if source["best_seconds"] < target["best_seconds"]:
                    target["best_seconds"] = source["best_seconds"]
                    target["best_frames"] = source["best_frames"]
                
                # Merge history
                target["history"].extend(source.get("history", []))
                # Sort history by timestamp just in case
                target["history"].sort(key=lambda x: x.get("timestamp", 0))
                
                merged_count += 1
        else:
            # No rename, just keep (or if it was already migrated? check)
            if key not in new_stats:
                new_stats[key] = stats[key]
            else:
                # This case shouldn't happen unless we process same key twice
                pass

    # Save
    with open(stats_path, 'w') as f:
        json.dump(new_stats, f, indent=4, sort_keys=True)
        
    print(f"Migration Complete. Renamed: {renamed_count}, Merged: {merged_count}")

if __name__ == "__main__":
    main()

import sys
import os
import shutil

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from recording.world import WorldMap

def main():
    base_dir = project_root
    state_dir = os.path.join(base_dir, "custom_integrations", "SuperMetroid-Snes")
    map_path = os.path.join(base_dir, "world_map.json")
    
    world = WorldMap(map_path)
    
    print(f"Scanning {state_dir} for rename candidates...")
    
    count = 0
    for filename in os.listdir(state_dir):
        if not filename.endswith(".state"):
            continue
            
        name_part = os.path.splitext(filename)[0]
        
        # Candidate 1: Unknown_Room_0xXXXX
        new_name = None
        
        if name_part.startswith("Unknown_Room_0x"):
            hex_id = name_part.replace("Unknown_Room_", "")
            # hex_id is "0x9879" string
            
            real_name = world.get_room_name(hex_id)
            if real_name and real_name != name_part:
                new_name = real_name
        
        # Candidate 2: RoomXX - Harder, we need to check if we can resolve it.
        # But for now, let's stick to the IDs we know.
        
        if new_name:
            # Check if target exists
            src = os.path.join(state_dir, filename)
            dst = os.path.join(state_dir, f"{new_name}.state")
            
            if os.path.exists(dst):
                print(f"Target '{new_name}.state' already exists. Skipping {filename}.")
                # Optional: Compare sizes/dates?
            else:
                shutil.move(src, dst)
                print(f"Renamed: {filename} -> {new_name}.state")
                count += 1
                
    print(f"Renamed {count} state files.")

if __name__ == "__main__":
    main()

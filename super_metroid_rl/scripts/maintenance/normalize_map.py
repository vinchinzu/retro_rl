import sys
import os

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from recording.world import WorldMap

base_dir = project_root
map_path = os.path.join(base_dir, "world_map.json")

world = WorldMap(map_path)
print("Normalizing map data...")

keys = list(world.data.keys())
for key in keys:
    val = world.data[key]
    if isinstance(val, int):
        world.data[key] = hex(val)
        
world.save()
print("Done.")

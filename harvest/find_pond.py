import os
import stable_retro as retro
import numpy as np

def main():
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INTEGRATION_PATH = os.path.join(SCRIPT_DIR, "custom_integrations")
    retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

    env = retro.make(game="HarvestMoon-Snes", state="Y1_Spring_Day01_06h00m", inttype=retro.data.Integrations.ALL)
    ram = env.get_ram()
    
    ADDR_MAP = 0x09B6
    MAP_WIDTH = 64
    MAP_HEIGHT = 64
    
    map_data = ram[ADDR_MAP : ADDR_MAP + MAP_WIDTH * MAP_HEIGHT]
    grid = map_data.reshape((MAP_HEIGHT, MAP_WIDTH))
    
    pond_tiles = []
    water_ids = {0xA6, 0xF0}
    
    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            if grid[y, x] in water_ids:
                pond_tiles.append((x, y))
    
    if pond_tiles:
        print(f"Pond tiles found at: {pond_tiles}")
        min_x = min(t[0] for t in pond_tiles)
        max_x = max(t[0] for t in pond_tiles)
        min_y = min(t[1] for t in pond_tiles)
        max_y = max(t[1] for t in pond_tiles)
        print(f"Pond bounding box: ({min_x}, {min_y}) to ({max_x}, {max_y})")
        
        # Look at 2-tile radius around pond
        print("\nSurrounding area:")
        for y in range(min_y - 2, max_y + 3):
            line = ""
            for x in range(min_x - 2, max_x + 3):
                tid = grid[y, x]
                if (x, y) in pond_tiles:
                    line += f"[{tid:02X}] "
                else:
                    line += f" {tid:02X}  "
            print(f"Y={y:2d}: {line}")
            
    else:
        print("Pond tiles not found in map data at 0x09B6.")

    env.close()

if __name__ == "__main__":
    main()

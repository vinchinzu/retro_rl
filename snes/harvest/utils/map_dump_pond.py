import os
import stable_retro as retro
import numpy as np

def main():
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    
    # Use existing integration path
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INTEGRATION_PATH = os.path.join(SCRIPT_DIR, "custom_integrations")
    retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

    try:
        env = retro.make(game="HarvestMoon-Snes", state="Y1_Near_Pond", inttype=retro.data.Integrations.ALL)
        ram = env.get_ram()
        
        ADDR_MAP = 0x09B6
        MAP_WIDTH = 64
        
        map_data = ram[ADDR_MAP : ADDR_MAP + MAP_WIDTH * MAP_WIDTH]
        grid = map_data.reshape((MAP_WIDTH, MAP_WIDTH))
        
        # Area around pond
        min_x, max_x = 28, 34
        min_y, max_y = 28, 34
        
        print(f"Map dump around pond ({min_x}-{max_x}, {min_y}-{max_y}):")
        for y in range(min_y, max_y + 1):
            line = f"Y={y:2d}: "
            for x in range(min_x, max_x + 1):
                tid = grid[y, x]
                line += f"{tid:02X} "
            print(line)
        
        env.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

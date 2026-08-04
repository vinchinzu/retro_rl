import os
import stable_retro as retro
import numpy as np

def main():
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INTEGRATION_PATH = os.path.join(SCRIPT_DIR, "custom_integrations")
    retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

    # Try different states if one fails
    states = ["Y1_Spring_Day01_06h00m", "Y1_Mid_Field", "Y1_Near_Pond"]
    for state in states:
        print(f"\n--- Checking state: {state} ---")
        try:
            env = retro.make(game="HarvestMoon-Snes", state=state, inttype=retro.data.Integrations.ALL)
        except Exception as e:
            print(f"Failed to load state {state}: {e}")
            continue
            
        ram = env.get_ram()
        ADDR_MAP = 0x09B6
        MAP_WIDTH = 64
        
        map_data = ram[ADDR_MAP : ADDR_MAP + MAP_WIDTH * MAP_WIDTH]
        grid = map_data.reshape((MAP_WIDTH, MAP_WIDTH))
        
        # Center of pond area
        cx, cy = 31, 31
        radius = 5
        
        print(f"Map dump around ({cx}, {cy}):")
        for y in range(cy - radius, cy + radius + 1):
            line = f"Y={y:2d}: "
            for x in range(cx - radius, cx + radius + 1):
                tid = grid[y, x]
                line += f"{tid:02X} "
            print(line)
            
        env.close()

if __name__ == "__main__":
    main()

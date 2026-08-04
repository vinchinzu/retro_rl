import os
import stable_retro as retro
import numpy as np

def main():
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INTEGRATION_PATH = os.path.join(SCRIPT_DIR, "custom_integrations")
    retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

    env = retro.make(game="HarvestMoon-Snes", state="Y1_Near_Pond", inttype=retro.data.Integrations.ALL)
    ram = env.get_ram()
    
    # Coordinates in Harvest Moon are usually at 0xD6 (X) and 0xD8 (Y)
    # But let's check what harvest_bot uses
    x = ram[0x00D6]
    y = ram[0x00D8]
    
    # Also check pixel positions (16-bit?)
    # Usually X is 0xD6, 0xD7 and Y is 0xD8, 0xD9
    px = ram[0x00D6] + (ram[0x00D7] << 8)
    py = ram[0x00D8] + (ram[0x00D9] << 8)
    
    print(f"Player at: ({x}, {y})")
    print(f"Pixel POS: ({px}, {py})")
    print(f"Tile POS: ({px//16}, {py//16})")
    
    env.close()

if __name__ == "__main__":
    main()

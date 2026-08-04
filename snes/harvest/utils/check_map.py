import os
import stable_retro as retro
import numpy as np
from collections import Counter

def main():
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    
    # Use existing integration path
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INTEGRATION_PATH = os.path.join(SCRIPT_DIR, "custom_integrations")
    retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

    env = retro.make(game="HarvestMoon-Snes", state="Y1_Mid_Field", inttype=retro.data.Integrations.ALL)
    ram = env.get_ram()
    
    ADDR_MAP = 0x09B6
    MAP_SIZE = 64 * 64
    
    print(f"Reading {MAP_SIZE} bytes starting at 0x{ADDR_MAP:04X}...")
    map_data = ram[ADDR_MAP : ADDR_MAP + MAP_SIZE]
    
    counts = Counter(map_data)
    print("\nTile ID counts (Top 20):")
    for tid, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  0x{tid:02X}: {count}")

    # Search for characteristic tiles in the WHOLE RAM
    print("\nSearching for characteristic tiles (0xA6, 0xF0) in WHOLE RAM...")
    for tid in [0xA6, 0xF0]:
        addrs = np.where(ram == tid)[0]
        if len(addrs) > 0:
            print(f"  Tile 0x{tid:02X} found at {len(addrs)} addresses. First 5: {[hex(a) for a in addrs[:5]]}")
        else:
            print(f"  Tile 0x{tid:02X} not found.")

    env.close()

if __name__ == "__main__":
    main()

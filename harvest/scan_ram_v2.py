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
    
    print(f"RAM size: {len(ram)} bytes")
    
    # Search for characteristic tiles
    # Water: 0xA6, 0xF0
    # Fence: 0x05
    # Grass: 0x00, 0x01
    
    target_ids = [0x05, 0xA6, 0xF0]
    for tid in target_ids:
        addrs = np.where(ram == tid)[0]
        if len(addrs) > 0:
            print(f"Tile 0x{tid:02X} found at {len(addrs)} addresses. First 10: {[hex(a) for a in addrs[:10]]}")
        else:
            print(f"Tile 0x{tid:02X} not found.")

    # Search for TILEMAP (around 0x0022 as suggested by decomp)
    print(f"RAM[0x0020:0x0060]: {' '.join(f'{b:02x}' for b in ram[0x0020:0x0060])}")
    
    # Search for MAP (around 0x09B6)
    print(f"RAM[0x09B0:0x0A00]: {' '.join(f'{b:02x}' for b in ram[0x09B0:0x0A00])}")

    env.close()

if __name__ == "__main__":
    main()

import os
import stable_retro as retro
import numpy as np
from collections import Counter

def main():
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INTEGRATION_PATH = os.path.join(SCRIPT_DIR, "custom_integrations")
    retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

    env = retro.make(game="HarvestMoon-Snes", state="Y1_Near_Pond", inttype=retro.data.Integrations.ALL)
    ram = env.get_ram()
    
    print(f"RAM size: {len(ram)} bytes")
    
    debris_ids = {0x03, 0x04, 0x05, 0x06, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14}
    
    # Scan for clusters of debris
    window_size = 64
    matches = np.isin(ram, list(debris_ids))
    
    # Use a rolling sum to find high density areas
    kernel = np.ones(window_size)
    density = np.convolve(matches, kernel, mode='valid')
    
    top_indices = np.argsort(density)[::-1][:10]
    
    print("\nTop Debug Debris Densities:")
    for idx in top_indices:
        if density[idx] > 5:
            print(f"  Addr 0x{idx:04x}: Density {density[idx]} in next {window_size} bytes")
            # Print a snippet
            snippet = ram[idx:idx+16]
            print(f"    Snippet: {' '.join(f'{b:02x}' for b in snippet)}")

    # Check specifically for 0xA6 and 0xF0 again
    for tid in [0xA6, 0xF0]:
        addrs = np.where(ram == tid)[0]
        if len(addrs) > 0:
            print(f"\nTile 0x{tid:02X} found at {len(addrs)} addresses.")
            print(f"  First 10: {[hex(a) for a in addrs[:10]]}")
        else:
            print(f"\nTile 0x{tid:02X} not found.")

    env.close()

if __name__ == "__main__":
    main()

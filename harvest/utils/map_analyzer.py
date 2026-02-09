import os
import stable_retro as retro
import numpy as np

def main():
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INTEGRATION_PATH = os.path.join(SCRIPT_DIR, "custom_integrations")
    retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

    state_name = "Y1_Spring_Day01_06h00m"
    print(f"Loading state: {state_name}")
    
    try:
        env = retro.make(game="HarvestMoon-Snes", state=state_name, inttype=retro.data.Integrations.ALL)
        env.reset()
        # Step a few frames just in case
        for _ in range(5):
            env.step(np.zeros(12))
            
        ram = env.get_ram()
        
        # Check potential map addresses
        potential_addrs = {
            "MAP (0x09B6)": 0x09B6,
            "TILEMAP (0x0022)": 0x0022,
            "ALT_MAP (0x19B6)": 0x19B6,
        }
        
        for name, addr in potential_addrs.items():
            data = ram[addr : addr + 64*64]
            unique, counts = np.unique(data, return_counts=True)
            non_zero = len(data) - (counts[np.where(unique == 0)] if 0 in unique else 0)
            print(f"{name}: {non_zero}/{len(data)} non-zero bytes. Unique IDs: {unique.tolist()[:10]}")
            
            if non_zero > 100:
                grid = data.reshape((64, 64))
                output_file = f"farm_map_{name.split()[0].lower()}.txt"
                with open(output_file, "w") as f:
                    f.write(f"--- {name} Dump ---\n")
                    for y in range(64):
                        row = [f"{grid[y, x]:02x}" for x in range(64)]
                        f.write(f"Y={y:02d}: " + " ".join(row) + "\n")
                print(f"  Dumped to {output_file}")
                
                # Print stats for this map
                unique, counts = np.unique(data, return_counts=True)
                stats = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)
                print(f"  Stats for {name}:")
                for tid, count in stats:
                    print(f"    0x{tid:02x}: {count}")

        # Search entire RAM for debris pattern
        # Debris like fences (0x05) or weeds (0x03)
        for target in [0x05, 0x03]:
            addrs = np.where(ram == target)[0]
            if len(addrs) > 10:
                print(f"Tile 0x{target:02X} found at {len(addrs)} addresses.")
            
        env.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

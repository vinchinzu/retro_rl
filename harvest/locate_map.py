import os
import stable_retro as retro
import numpy as np
from collections import Counter

def main():
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    
    # Correct path setup
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INTEGRATION_PATH = os.path.join(SCRIPT_DIR, "custom_integrations")
    retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

    # Use the state that the bot is actually using
    state_name = "Y1_Spring_Day01_06h00m"
    print(f"Loading state: {state_name}")
    
    try:
        env = retro.make(game="HarvestMoon-Snes", state=state_name, inttype=retro.data.Integrations.ALL)
        ram = env.get_ram()
        
        print(f"RAM size: {len(ram)} bytes")
        
        # Searching for fence ID (0x05)
        # We expect a cluster of them if there's a map
        target = 0x05
        addrs = np.where(ram == target)[0]
        print(f"Found {len(addrs)} instances of 0x{target:02x}")
        
        if len(addrs) > 10:
            # Find clusters
            clusters = []
            if len(addrs) > 0:
                current_cluster = [addrs[0]]
                for i in range(1, len(addrs)):
                    if addrs[i] - addrs[i-1] < 64: # Nearby
                        current_cluster.append(addrs[i])
                    else:
                        if len(current_cluster) > 5:
                            clusters.append(current_cluster)
                        current_cluster = [addrs[i]]
                if len(current_cluster) > 5:
                    clusters.append(current_cluster)
            
            print(f"Found {len(clusters)} debris clusters.")
            for cluster in clusters:
                print(f"  Cluster at 0x{cluster[0]:04X}, size {len(cluster)}")
                snippet = ram[cluster[0]:cluster[0]+32]
                print(f"    Snippet: {' '.join(f'{b:02x}' for b in snippet)}")
        
        env.close()
    except Exception as e:
        print(f"Error loading state: {e}")

if __name__ == "__main__":
    main()

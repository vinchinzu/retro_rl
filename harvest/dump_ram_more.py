import os
import stable_retro as retro
import numpy as np

def main():
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INTEGRATION_PATH = os.path.join(SCRIPT_DIR, "custom_integrations")
    retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

    for state in ["Y1_Near_Pond", "Y1_Front_House", "Recorded"]:
        print(f"\n--- State: {state} ---")
        try:
            env = retro.make(game="HarvestMoon-Snes", state=state, inttype=retro.data.Integrations.ALL)
            ram = env.get_ram()
            
            x = ram[0x00D6]
            y = ram[0x00D8]
            px = ram[0x00D6] + (ram[0x00D7] << 8)
            py = ram[0x00D8] + (ram[0x00D9] << 8)
            
            print(f"X/Y (0xD6/0xD8): ({x}, {y})")
            print(f"Pixel POS: ({px}, {py})")
            
            # Print non-zero bytes in first 0x200
            print("Non-zero bytes in 0x00-0xFF:")
            for i in range(0x100):
                if ram[i] != 0:
                    print(f"  0x{i:02x}: 0x{ram[i]:02x}")
            
            env.close()
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

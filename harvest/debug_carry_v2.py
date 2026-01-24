import os
import stable_retro as retro
import numpy as np

def main():
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INTEGRATION_PATH = os.path.join(SCRIPT_DIR, "custom_integrations")
    retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

    env = retro.make(game="HarvestMoon-Snes", state="Y1_Liftable_Nearby", inttype=retro.data.Integrations.ALL)
    env.reset()
    
    # 0xDA is player direction (0:down, 1:up, 2:left, 3:right)
    # We want to face down (0)
    
    print("Frame | 0xD2 | 0xD3 | 0xD4 | Item (0x91D)")
    print("-" * 40)
    
    for frame in range(120):
        action = np.zeros(12, dtype=np.int32)
        if 10 <= frame < 30:
            action[8] = 1 # A button
            action[5] = 1 # Down
        
        obs, reward, terminated, truncated, info = env.step(action)
        ram = env.get_ram()
        
        if frame % 5 == 0 or (10 <= frame < 40):
            print(f"{frame:5d} | {ram[0xD2]:04b} | {ram[0xD3]:04b} | {ram[0xD4]:02x} | {ram[0x91D]:02x}")

    env.close()

if __name__ == "__main__":
    main()

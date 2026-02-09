import os
import stable_retro as retro
import numpy as np
import time

def main():
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    
    # Use existing integration path
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INTEGRATION_PATH = os.path.join(SCRIPT_DIR, "custom_integrations")
    retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

    # State near a fence
    env = retro.make(game="HarvestMoon-Snes", state="Y1_Liftable_Nearby", inttype=retro.data.Integrations.ALL)
    obs, info = env.reset()
    
    def get_ram_snapshot():
        return env.get_ram().copy()

    print("Monitoring RAM for carry bit...")
    
    # Before lift
    ram_before = get_ram_snapshot()
    
    # Sequence to lift (Face down and press A)
    # 1. Face down
    for _ in range(10):
        env.step(np.array([0,0,0,0,0,1,0,0,0,0,0,0], dtype=np.int32))
    
    # 2. Press A
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(np.array([0,0,0,0,0,1,0,0,1,0,0,0], dtype=np.int32))
    
    # 3. Wait for animation
    for _ in range(30):
        env.step(np.zeros(12, dtype=np.int32))
        
    ram_after_lift = get_ram_snapshot()
    
    # Compare
    diff = np.where(ram_before != ram_after_lift)[0]
    print(f"RAM addresses changed after lift: {[hex(a) for a in diff]}")
    
    for addr in [0xD2, 0xD3, 0xD4]:
        print(f"Address {hex(addr)}: Before=0x{ram_before[addr]:02x}, After=0x{ram_after_lift[addr]:02x}")

    env.close()

if __name__ == "__main__":
    main()

import stable_retro as retro
import numpy as np
import time

def scan_boss_movement():
    env = retro.make(game="SuperMetroid-Snes", state="BossTorizo")
    env.reset()
    
    print("Scanning for moving values in RAM...")
    
    # Stay right to avoid getting hit immediately, but allow boss to move
    action = np.zeros(12)
    action[7] = 1 # Right
    action[9] = 1 # Shoot
    
    initial_ram = env.get_ram().copy()
    
    # Run for 200 frames to let boss wake up and move
    for f in range(300):
        env.step(action)
        if f == 150: # Check point
            mid_ram = env.get_ram().copy()
        if f == 299: # End point
            final_ram = env.get_ram().copy()
            
    # Find values that changed between mid and final
    # and were near the boss start (left side, X < 100)
    print(f"{'Addr':<8} | {'Initial':<8} | {'Mid':<8} | {'Final':<8}")
    print("-" * 40)
    
    for i in range(0x0F00, 0x1100, 1): # Check enemy slots neighborhood
        if initial_ram[i] != mid_ram[i] or mid_ram[i] != final_ram[i]:
            # Possible moving value
            # Filter for likely X positions (not too huge)
            if final_ram[i] < 255:
                 print(f"0x{i:04X} ({i:<4}) | {initial_ram[i]:<8} | {mid_ram[i]:<8} | {final_ram[i]:<8}")

    env.close()

if __name__ == "__main__":
    scan_boss_movement()

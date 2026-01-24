import retro
import numpy as np

def main():
    game = "SuperMetroid-Snes"
    state = "BossTorizo.state"
    # Use the local integration directory
    env = retro.make(game=game, state="BossTorizo", 
                     data_dirs=["./custom_integrations"])
    
    env.reset()
    
    print(f"{'Frame':<6} | {'Slot0 X':<8} | {'Slot0 HP':<8} | {'Slot1 X':<8} | {'Slot1 HP':<8} | {'Slot2 X':<8} | {'Slot2 HP':<8}")
    print("-" * 80)
    
    # Aggressive attack pattern to wake boss up
    # 7 is move-right-and-shoot-up in our play script
    action = [0] * 12
    action[7] = 1 # Right
    action[9] = 1 # Shoot
    action[4] = 1 # Up
    
    for frame in range(600):
        # Apply action every step
        obs, reward, term, trunc, info = env.step(action)
        
        if frame % 30 == 0:
            s0x = info.get('enemy0_x', 0)
            s0h = info.get('enemy0_hp', 0)
            s1x = info.get('enemy1_x', 0)
            s1h = info.get('enemy1_hp', 0)
            s2x = info.get('enemy2_x', 0)
            s2h = info.get('enemy2_hp', 0)
            
            print(f"{frame:<6} | {s0x:<8} | {s0h:<8} | {s1x:<8} | {s1h:<8} | {s2x:<8} | {s2h:<8}")
            
        if term or trunc:
            break
            
    env.close()

if __name__ == "__main__":
    main()

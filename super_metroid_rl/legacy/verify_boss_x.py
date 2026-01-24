import stable_retro as retro
import numpy as np

def verify_info_keys():
    env = retro.make(game="SuperMetroid-Snes", state="BossTorizo")
    env.reset()
    
    # Wake up and move
    action = np.zeros(12)
    action[7] = 1 # Right
    action[9] = 1 # Shoot
    
    print(f"{'Frame':<6} | {'BossX':<8} | {'BossHP':<8} | {'Orb1HP':<8}")
    print("-" * 40)
    
    for f in range(200):
        obs, reward, term, trunc, info = env.step(action)
        if f % 30 == 0:
            bx = info.get('boss_x', 'MISSING')
            bh = info.get('boss_hp', 'MISSING')
            o1h = info.get('enemy1_hp', 'MISSING')
            print(f"{f:<6} | {bx:<8} | {bh:<8} | {o1h:<8}")
            
    env.close()

if __name__ == "__main__":
    verify_info_keys()

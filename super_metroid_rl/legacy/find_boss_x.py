import stable_retro as retro
import numpy as np
import time

def find_boss_x():
    env = retro.make(game="SuperMetroid-Snes", state="BossTorizo")
    env.reset()
    
    # Run long enough for the boss to be fully active
    print("Running environment to activate boss...")
    for _ in range(300):
        env.step(env.action_space.sample())
        
    data = env.unwrapped.data
    samus_x = data.lookup_value('samus_x')
    print(f"Samus X: {samus_x}")
    
    # Iterate through possible enemy slots and find non-zero X/HP
    # $0F7A is start of enemy data
    for slot in range(12):
        try:
            x = data.lookup_value(f'enemy{slot}_x')
            hp = data.lookup_value(f'enemy{slot}_hp')
            print(f"Slot {slot} -> X: {x}, HP: {hp}")
        except Exception as e:
            # Maybe the slot names aren't in data.json
            pass
            
    # Manually check some specific ranges if names aren't there
    # But we want to use lookup_value if possible.
    
    env.close()

if __name__ == "__main__":
    find_boss_x()

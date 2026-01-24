import stable_retro as retro
import pygame
import os

def main():
    # Use the fixed state which allows items to be used (even if selection input is glitchy)
    state_path = os.path.abspath('super_metroid_rl/custom_integrations/SuperMetroid-Snes/BossTorizo_Fixed.state')
    env = retro.make(game='SuperMetroid-Snes', state=state_path)
    obs = env.reset()

    print("Loaded BossTorizo_Fixed.")
    print("Press SPACE to toggle Missiles (via RAM override).")
    print("Press ESC to quit.")

    # Initialize Pygame for human input (if running locally) or just loop
    # Since we are in headless agent, we just demonstrate logic.
    # But for user, they will run this.
    
    # We will simulate a loop where we force missiles ON.
    
    print("Forcing Missiles Selected...")
    env.data.set_value('selected_item', 1) 
    
    done = False
    while not done:
        # Just idle or random actions
        # For the user, they would hook this up to their agent or controller
        action = [0]*12
        obs, reward, terminated, truncated, info = env.step(action)
        
        if info.get('selected_item') == 0:
            # Force it back if it resets
             env.data.set_value('selected_item', 1)
             
        if terminated or truncated:
            obs = env.reset()
            env.data.set_value('selected_item', 1)

    env.close()

if __name__ == "__main__":
    main()

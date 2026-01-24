import stable_retro as retro
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTEGRATION_PATH = os.path.join(SCRIPT_DIR, "custom_integrations")
retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

def test_missile_select():
    env = retro.make(game="SuperMetroid-Snes", state="BossTorizo")
    env.reset()
    
    # Introspect available variables
    print(f"Available variables: {env.data.vars}")
    
    print("\nInitial State:")
    missiles = env.data.lookup_value('missiles')
    selected = env.data.lookup_value('selected_item')
    print(f"Missiles: {missiles}, Selected: {selected}")
    
    print("\nForcing Missile Select (selected_item = 1)...")
    env.data.set_value('selected_item', 1)
    
    # Check immediately after setting
    selected = env.data.lookup_value('selected_item')
    print(f"Selected immediately after set_value: {selected}")
    
    print("\nStepping environment (No-op)...")
    # Action for SELECT button is usually index 2 but we want to see if RAM sticks
    obs, rew, term, trunc, info = env.step([0]*12)
    selected_info = info.get('selected_item')
    selected_mem = env.data.lookup_value('selected_item')
    print(f"Selected (info): {selected_info}")
    print(f"Selected (mem): {selected_mem}")
    
    print("\nStepping 10 more times without forcing...")
    for i in range(10):
        _, _, _, _, info = env.step([0]*12)
        print(f"  Step {i+1}: Selected={info.get('selected_item')}")
        
    env.close()

if __name__ == "__main__":
    test_missile_select()

#!/usr/bin/env python3
import stable_retro as retro
import os
import time

# Force software rendering for headless
os.environ['SDL_VIDEODRIVER'] = 'dummy'

def main():
    state_path = os.path.abspath('super_metroid_rl/custom_integrations/SuperMetroid-Snes/BossTorizo_Fixed.state')
    print(f"Loading SuperMetroid-Snes with state {state_path}...")
    try:
        env = retro.make(game='SuperMetroid-Snes', state=state_path)
        obs = env.reset()
    except Exception as e:
        print(f"Failed to load env: {e}")
        return

    # Check initial values
    obs, reward, terminated, truncated, info = env.step([0]*12)
    
    missiles = info.get('missiles', -1)
    max_missiles = info.get('max_missiles', -1)
    selected_item = info.get('selected_item', -1)
    game_state = info.get('game_state', -1)
    config_select = info.get('config_item_select', -1)
    config_cancel = info.get('config_item_cancel', -1)
    equipped = info.get('equipped_items', -1)
    collected = info.get('collected_items', -1)
    timer_type = info.get('timer_type', -1)
    auto_cancel = info.get('auto_cancel_item', -1)
    
    print(f"Initial Missiles: {missiles}")
    print(f"Max Missiles: {max_missiles}")
    print(f"Initial Selected Item: {selected_item}")  # 0 usually means nothing/beam
    print(f"Initial Game State: {hex(game_state) if isinstance(game_state, int) else game_state}")
    print(f"Config Item Select: {hex(config_select) if isinstance(config_select, int) else config_select}")
    print(f"Config Item Cancel: {hex(config_cancel) if isinstance(config_cancel, int) else config_cancel}")
    print(f"Equipped Items: {hex(equipped) if isinstance(equipped, int) else equipped}")
    print(f"Collected Items: {hex(collected) if isinstance(collected, int) else collected}")
    print(f"Timer Type: {timer_type}")
    print(f"Auto Cancel: {auto_cancel}")
    
    print("Testing ALL buttons for Item Selection (Using Fixed State)...")
    try:
        # Test buttons again
        test_buttons = [0, 1, 2, 3, 8, 9, 10, 11] 
        
        for btn_idx in test_buttons:
            print(f"Testing Button index {btn_idx} ({env.buttons[btn_idx]})...")
            action = [0]*12
            action[btn_idx] = 1
            
            # Press and Hold
            for _ in range(10):
                obs, _, _, _, info = env.step(action)
            
            inp = info.get('input_p1', 0)
            print(f"  Input P1: {hex(inp) if isinstance(inp, int) else inp}")
            
            # Release and wait
            for i in range(10):
                 obs, _, _, _, info = env.step([0]*12)
                 
            sel = info.get('selected_item')
            print(f"  Result: Selected={sel}")
            if sel != 0:
                print(f"FOUND IT! Button {btn_idx} works when Items are Equipped!")
                break
                
    except Exception as e:
        print(f"set_value failed: {e}")
    
    env.close()

if __name__ == "__main__":
    main()

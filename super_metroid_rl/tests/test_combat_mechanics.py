#!/usr/bin/env python3
"""Consolidated combat mechanics verification for Super Metroid RL."""

import os
import numpy as np
import stable_retro as retro
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
INTEGRATION_PATH = os.path.join(PROJECT_DIR, "custom_integrations")
retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

B, Y, SELECT, START, UP, DOWN, LEFT, RIGHT, A, X, L, R = range(12)

def save_frame(obs, frame_num, prefix="frame"):
    debug_dir = os.path.join(PROJECT_DIR, "debug_frames")
    os.makedirs(debug_dir, exist_ok=True)
    img = Image.fromarray(obs)
    img.save(f"{debug_dir}/{prefix}_{frame_num:04d}.png")

def test_shooting_mechanics(env, state_name):
    print(f"\n=== Testing Shooting in State: {state_name} ===")
    obs, info = env.reset()
    
    # Force capabilities
    env.data.set_value("equipped_beams", 0x1000)
    env.data.set_value("collected_beams", 0x1000)
    env.data.set_value("max_missiles", 20)
    env.data.set_value("missiles", 20)
    
    # Try shooting (Y button)
    action = np.zeros(12, dtype=np.int8)
    action[Y] = 1
    
    initial_missiles = info.get('missiles', 0)
    initial_boss_hp = info.get('boss_hp', 0)
    
    for frame in range(100):
        # Force missile select occasionally to ensure it stays selected
        env.data.set_value("selected_item", 1) 
        
        obs, reward, term, trunc, info = env.step(action)
        
        current_missiles = info.get('missiles', 0)
        current_boss_hp = info.get('boss_hp', 0)
        proj = info.get('projectile0_type', 0)
        
        if current_missiles != initial_missiles:
            print(f"  Frame {frame}: *** MISSILE FIRED! *** (Remaining: {current_missiles})")
            initial_missiles = current_missiles
        
        if current_boss_hp != initial_boss_hp and initial_boss_hp > 0:
            print(f"  Frame {frame}: *** BOSS DAMAGED! *** (HP: {current_boss_hp})")
            initial_boss_hp = current_boss_hp
            
        if frame % 20 == 0 and proj != 0:
            print(f"  Frame {frame}: Proj0 active (type: {proj})")

def main():
    states = ["BossTorizo", "ZebesStart"]
    for state in states:
        try:
            env = retro.make(
                game="SuperMetroid-Snes",
                state=state,
                use_restricted_actions=retro.Actions.ALL,
                render_mode=None
            )
            test_shooting_mechanics(env, state)
            env.close()
        except Exception as e:
            print(f"Could not test state {state}: {e}")

if __name__ == "__main__":
    main()

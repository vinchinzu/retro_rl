#!/usr/bin/env python3
"""Consolidated environment diagnostics for Super Metroid RL."""

import os
import numpy as np
import stable_retro as retro

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: Going up one level because this is now in tests/
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
INTEGRATION_PATH = os.path.join(PROJECT_DIR, "custom_integrations")
retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

B, Y, SELECT, START, UP, DOWN, LEFT, RIGHT, A, X, L, R = range(12)

def test_action_formats(env):
    print("\n=== Testing Action Formats ===")
    formats = [
        ("np.int8", np.zeros(12, dtype=np.int8)),
        ("np.uint8", np.zeros(12, dtype=np.uint8)),
        ("np.int32", np.zeros(12, dtype=np.int32)),
        ("list of int", [0] * 12),
        ("np.array default", np.zeros(12)),
    ]

    for name, action_template in formats:
        print(f"Testing format: {name}")
        obs, info = env.reset()
        action = action_template.copy() if hasattr(action_template, 'copy') else list(action_template)
        action[RIGHT] = 1
        
        start_x = info.get('samus_x', 0)
        for _ in range(30):
            obs, reward, term, trunc, info = env.step(action)
        
        end_x = info.get('samus_x', 0)
        moved = end_x - start_x
        print(f"  Result: moved {moved} pixels (RIGHT)")

def test_all_buttons(env):
    print("\n=== Testing All Buttons ===")
    buttons = env.buttons
    for btn_idx, btn_name in enumerate(buttons):
        obs, info = env.reset()
        
        # Force some state
        env.data.set_value("equipped_beams", 0x1000)
        env.data.set_value("collected_beams", 0x1000)
        
        initial_x = info.get('samus_x', 0)
        initial_y = info.get('samus_y', 0)

        action = np.zeros(12, dtype=np.int8)
        action[btn_idx] = 1
        for _ in range(30):
            obs, reward, term, trunc, info = env.step(action)

        proj = info.get('projectile0_type', 0)
        x = info.get('samus_x', 0)
        y = info.get('samus_y', 0)

        changes = []
        if proj != 0: changes.append(f"PROJ={proj}")
        if abs(x - initial_x) > 2: changes.append(f"X move:{x-initial_x}")
        if abs(y - initial_y) > 2: changes.append(f"Y move:{y-initial_y}")

        change_str = ", ".join(changes) if changes else "(no major change)"
        print(f"  {btn_idx:2d}: {btn_name:8s} -> {change_str}")

def main():
    print("Starting Environment Diagnostics...")
    
    # Test with custom integration
    try:
        env = retro.make(
            game="SuperMetroid-Snes",
            state="BossTorizo",
            use_restricted_actions=retro.Actions.ALL,
            render_mode=None
        )
        print("Successfully loaded custom integration.")
    except Exception as e:
        print(f"Failed to load custom integration: {e}")
        return

    test_action_formats(env)
    test_all_buttons(env)
    
    env.close()

if __name__ == "__main__":
    main()

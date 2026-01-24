#!/usr/bin/env python3
"""
Script to run Super Metroid using stable_retro.
Based on beat_level_1_1_naive.py but adapted for Super Metroid.
"""

import stable_retro as retro
import numpy as np
import time
import os
import sys

# Fix SDL renderer issues - force software rendering
os.environ['SDL_VIDEODRIVER'] = 'x11'

def run_level():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    args = parser.parse_args()

    print("Initializing Super Metroid Agent...")
    
    # Init Pygame
    pygame = None
    if not args.headless:
        try:
            import pygame
            pygame.init()
            print("✓ Pygame initialized for video output")
        except ImportError:
            print("Warning: pygame not found, video output disabled")

    record_dir = "./super_metroid_recordings"
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    # Use the newly created 'Start' state
    try:
        env = retro.make(game="SuperMetroid-Snes", state="Start", render_mode='rgb_array', record=record_dir)
        print("✓ Environment created successfully with state='Start'")
    except Exception as e:
        print(f"Error creating environment: {e}")
        return

    obs = env.reset()
    if isinstance(obs, tuple):
        obs, _ = obs

    print(f"Buttons: {env.buttons}")
        
    # Setup Window
    screen = None
    if pygame:
        screen = pygame.display.set_mode(obs.shape[1::-1], pygame.SWSURFACE)
        pygame.display.set_caption("Super Metroid Naive Agent")
        
    done = False
    frames = 0
    max_frames = 5000 
    clock = pygame.time.Clock() if pygame else None
    
    print(f"Starting run (Recording to {record_dir})...")
    
    try:
        while not done and frames < max_frames:
            # Handle window events
            if pygame:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                        done = True
            
            # Naive Strategy: Run Right and Jump randomly
            # SNES Button Map: [B, Y, Select, Start, Up, Down, Left, Right, A, X, L, R]
            action = [0] * 12
            action[7] = 1 # Right
            action[0] = 1 # B (Dash)
            action[9] = 1 # X (Shoot - just in case)
            
            # Random jumping
            # 0x0770 in RAM is Game State (10-12 = Gameplay usually)
            # Just jump periodically
            if (frames // 20) % 2 == 0:
                 action[8] = 1 # A (Jump)

            step_result = env.step(action)
            
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            
            # Render to Pygame window
            if screen:
                surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
                screen.blit(pygame.transform.scale(surf, screen.get_size()), (0, 0))
                pygame.display.flip()
                clock.tick(60) 
                
            frames += 1
            if frames % 60 == 0:  # Print every second at 60fps
                x = info.get('samus_x', 0)
                y = info.get('samus_y', 0)
                room = info.get('room_id', 0)
                hp = info.get('health', 0)
                vx = info.get('velocity_x', 0)
                vy = info.get('velocity_y', 0)
                game_state = info.get('game_state', 0)
                print(f"F{frames:4d} | Pos:({x:4d},{y:4d}) Room:{room:#06x} HP:{hp:3d} Vel:({vx:+4d},{vy:+4d}) State:{game_state}")

        print("Run finished.")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        if pygame:
            pygame.quit()

if __name__ == "__main__":
    run_level()

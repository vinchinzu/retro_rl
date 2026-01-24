#!/usr/bin/env python3
"""
Human Play Script for Stable-Retro
This allows you to play games with keyboard controls to test your setup
"""

import stable_retro as retro
import numpy as np
import sys
import os

# Fix SDL renderer issues - force software rendering for Wayland
os.environ['SDL_VIDEODRIVER'] = 'x11'
# os.environ['SDL_RENDER_DRIVER'] = 'software'

# Keyboard mapping (requires pygame or similar)
# This is a simplified version - in practice, you'd use pygame.event

def play_game(game_name="SuperMarioBros-Nes-v0", state="Level1-1"):
    """
    Play a game with keyboard controls
    Note: Requires pygame for keyboard input
    """
    try:
        import pygame
    except ImportError:
        print("Error: pygame not installed")
        print("Install with: retro_env/bin/pip install pygame")
        sys.exit(1)

    pygame.init()

    print(f"Loading {game_name} from state: {state}")

    if state == "NONE":
        state = retro.State.NONE

    try:
        env = retro.make(game=game_name, state=state, render_mode='rgb_array')
        print("✓ Game loaded!")
        print("\nKeyboard Controls:")
        print("  Arrow Keys: D-Pad")
        print("  Z: B button")
        print("  X: A button")
        print("  Enter: Start")
        print("  Shift: Select")
        print("  Q: Quit")

        obs, info = env.reset()

        # Create display with software rendering (no hardware acceleration)
        screen = pygame.display.set_mode(obs.shape[1::-1], pygame.SWSURFACE)
        pygame.display.set_caption(f"Stable-Retro: {game_name}")

        clock = pygame.time.Clock()
        running = True
        total_reward = 0

        while running:
            # Get keyboard input
            keys = pygame.key.get_pressed()

            # Map keyboard to action
            # SNES Button Map: [B, Y, Select, Start, Up, Down, Left, Right, A, X, L, R]
            # NES Button Map:  [B, NULL, Select, Start, Up, Down, Left, Right, A]
            
            num_buttons = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n
            action = [0] * num_buttons

            # Universal Controls (Arrow Keys + WASD/ZX)
            # Arrow Keys: D-Pad
            if keys[pygame.K_RIGHT]:
                action[7] = 1
            if keys[pygame.K_LEFT]:
                action[6] = 1
            if keys[pygame.K_DOWN]:
                action[5] = 1
            if keys[pygame.K_UP]:
                action[4] = 1
            
            # Start/Select
            if keys[pygame.K_RETURN]:
                action[3] = 1
            if keys[pygame.K_RSHIFT] or keys[pygame.K_LSHIFT]:
                action[2] = 1
                
            # Buttons
            if keys[pygame.K_z]: # B
                action[0] = 1 
            if keys[pygame.K_x]: # A
                action[8] = 1
            
            # SNES Specific
            if num_buttons > 9:
                if keys[pygame.K_a]: # Y
                    action[1] = 1
                if keys[pygame.K_s]: # X
                    action[9] = 1
                if keys[pygame.K_q]: # L
                    action[10] = 1
                if keys[pygame.K_w]: # R
                    action[11] = 1

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    
                    # Save State Feature (Press S)
                    # Use K_m for save (M for Memory/Save) since S is mapped to X button above
                    if event.key == pygame.K_m:
                        print("\nSaving state to 'ManualSave.state'...")
                        state_data = env.em.get_state()
                        with open("ManualSave.state", "wb") as f:
                            f.write(state_data)
                        print("Saved!")
                        
                        # Also attempt to save to game directory
                        try:
                             # Try to save to the custom integration folder if it exists
                             save_path = "super_metroid_rl/custom_integrations/SuperMetroid-Snes/Start.state"
                             if os.path.exists(os.path.dirname(save_path)):
                                 with open(save_path, "wb") as f:
                                     f.write(state_data)
                                 print(f"Also saved to {save_path}")
                        except:
                            pass

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Render to pygame surface
            surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
            screen.blit(pygame.transform.scale(surf, screen.get_size()), (0, 0))
            pygame.display.flip()

            # Cap at 60 FPS
            clock.tick(60)

            if done:
                print(f"\nEpisode ended! Total reward: {total_reward}")
                pygame.time.wait(2000)
                obs, info = env.reset()
                total_reward = 0

        env.close()
        pygame.quit()

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        if 'pygame' in locals():
            pygame.quit()

def list_games_for_play():
    """List games that are ready to play (ROM imported)"""
    print("Checking which games are ready to play...\n")
    print(f"All games: {retro.data.list_games()}")
    return []

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Play Stable-Retro games with keyboard")
    parser.add_argument("--game", type=str, default="SuperMetroid-Snes",
                       help="Game to play (e.g., SuperMetroid-Snes)")
    parser.add_argument("--state", type=str, default="Start",
                       help="Save state to load (e.g., Start)")
    parser.add_argument("--list", action="store_true",
                       help="List playable games")

    args = parser.parse_args()

    if args.list:
        playable = list_games_for_play()
        if playable:
            print(f"\n{len(playable)} games ready to play!")
            print("Use: retro_env/bin/python play_human.py --game <game_name>")
    else:
        play_game(args.game, args.state)

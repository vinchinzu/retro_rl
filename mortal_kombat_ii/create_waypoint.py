#!/usr/bin/env python3
"""
Create a waypoint save at character select screen.
This makes creating character states MUCH faster!
"""

import os
import sys
import gzip
from pathlib import Path

if not os.environ.get("SDL_VIDEODRIVER"):
    os.environ["SDL_VIDEODRIVER"] = "x11"

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pygame
import stable_retro as retro
from fighters_common.game_configs import get_game_config

def main():
    config = get_game_config("mk2")
    game_dir = ROOT_DIR / config.game_dir_name
    integrations = game_dir / "custom_integrations"
    retro.data.Integrations.add_custom_path(str(integrations))

    env = retro.make(
        game=config.game_id,
        state=retro.State.NONE,  # Start from ROM boot
        render_mode="rgb_array",
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        use_restricted_actions=retro.Actions.ALL,
    )

    # Pygame setup
    pygame.init()
    SCALE = 3
    width, height = 256 * SCALE, 224 * SCALE
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("MK2 Waypoint Creator")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 18)

    # Controller mapping
    button_map = {
        pygame.K_x: 0,      # B
        pygame.K_z: 1,      # Y
        pygame.K_RSHIFT: 2, # SELECT
        pygame.K_RETURN: 3, # START
        pygame.K_UP: 4,
        pygame.K_DOWN: 5,
        pygame.K_LEFT: 6,
        pygame.K_RIGHT: 7,
        pygame.K_s: 8,      # A
        pygame.K_a: 9,      # X
        pygame.K_q: 10,     # L
        pygame.K_w: 11,     # R
    }

    obs, info = env.reset()
    running = True

    print("\n" + "="*60)
    print("MK2 WAYPOINT CREATOR")
    print("="*60)
    print("\nNavigate to the CHARACTER SELECT screen, then press F1 to save.")
    print("\nControls:")
    print("  Arrow Keys: Move")
    print("  ENTER: Start/Confirm")
    print("  TAB: Turbo (hold for 10x speed through intros!)")
    print("  F1: Save waypoint at character select")
    print("  ESC: Quit")
    print("="*60)
    print("\nTIP: Hold TAB during intro screens and menus")
    print("     Save RIGHT when you see the character select screen")
    print()

    while running:
        action = np.zeros(12, dtype=np.int8)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F1:
                    save_waypoint(env, config, game_dir)

        # Read current keys
        keys = pygame.key.get_pressed()
        for key, button_idx in button_map.items():
            if keys[key]:
                action[button_idx] = 1

        # Turbo mode
        turbo_speed = 10 if keys[pygame.K_TAB] else 1

        # Step environment
        for _ in range(turbo_speed):
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break

        # Render
        frame = env.render()
        if frame is not None:
            surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            screen.blit(pygame.transform.scale(surf, (width, height)), (0, 0))

        # Show instructions
        y_offset = 10
        turbo_active = keys[pygame.K_TAB] if 'keys' in locals() else False
        turbo_indicator = " [TURBO x10]" if turbo_active else ""
        lines = [
            f"F1: Save Waypoint | TAB: Turbo{turbo_indicator}",
            "Navigate to CHARACTER SELECT screen",
        ]
        for line in lines:
            text = font.render(line, True, (0, 255, 0))
            text_bg = pygame.Surface((text.get_width() + 10, text.get_height() + 4))
            text_bg.fill((0, 0, 0))
            text_bg.set_alpha(180)
            screen.blit(text_bg, (5, y_offset))
            screen.blit(text, (10, y_offset + 2))
            y_offset += text.get_height() + 6

        pygame.display.flip()
        clock.tick(60)

    env.close()
    pygame.quit()

def save_waypoint(env, config, game_dir):
    """Save waypoint at character select screen."""
    state_name = "CharSelect_MortalKombatII"
    save_path = game_dir / "custom_integrations" / config.game_id / f"{state_name}.state"

    try:
        state_data = env.em.get_state()
        with gzip.open(save_path, "wb") as f:
            f.write(state_data)
        print(f"\n✓✓✓ WAYPOINT SAVED: {state_name}.state")
        print(f"    Now you can load this waypoint to skip intros!")
        print(f"    Location: {save_path}")
    except Exception as e:
        print(f"\n✗ Failed to save waypoint: {e}")

if __name__ == "__main__":
    main()

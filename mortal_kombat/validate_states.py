#!/usr/bin/env python3
"""
Validate save states by showing the first few seconds of each one.
Press SPACE to move to next state, Q to quit.
"""

import os
import sys
from pathlib import Path

# X11 for Wayland compat
if not os.environ.get("SDL_VIDEODRIVER"):
    os.environ["SDL_VIDEODRIVER"] = "x11"

import numpy as np
import pygame
import time

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

import stable_retro as retro
from fighters_common.game_configs import get_game_config


def validate_state(game_id, state_name, integrations_path):
    """Show the first few seconds of a save state."""

    # Create environment
    env = retro.make(
        game=game_id,
        state=state_name,
        render_mode="rgb_array",
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        use_restricted_actions=retro.Actions.ALL,
    )

    # Pygame setup
    pygame.init()
    SCALE = 3
    width, height = 256 * SCALE, 224 * SCALE
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"Validating: {state_name}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 24)

    # Reset to load the state
    obs = env.reset()

    # Create no-op action (all buttons off)
    noop = np.zeros(env.action_space.shape, dtype=np.int8)

    print(f"\nShowing: {state_name}")
    print("  Press SPACE to continue to next state")
    print("  Press Q or ESC to quit")

    frame_count = 0
    running = True
    next_state = False

    # Hold for 3 seconds to let user see it
    while running and not next_state and frame_count < 180:  # 3 seconds at 60fps
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    next_state = True

        # Take no-op action
        obs, reward, terminated, truncated, info = env.step(noop)
        frame_count += 1

        # Render
        frame = env.render()
        if frame is not None:
            surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            screen.blit(pygame.transform.scale(surf, (width, height)), (0, 0))

        # Show state name overlay
        text = font.render(state_name, True, (255, 255, 0))
        text_bg = pygame.Surface((text.get_width() + 20, text.get_height() + 10))
        text_bg.fill((0, 0, 0))
        text_bg.set_alpha(200)
        screen.blit(text_bg, (10, 10))
        screen.blit(text, (20, 15))

        # Show instructions
        inst_font = pygame.font.SysFont("monospace", 16)
        inst_text = inst_font.render("SPACE: Next | Q: Quit", True, (255, 255, 255))
        inst_bg = pygame.Surface((inst_text.get_width() + 20, inst_text.get_height() + 10))
        inst_bg.fill((0, 0, 0))
        inst_bg.set_alpha(200)
        screen.blit(inst_bg, (10, height - 40))
        screen.blit(inst_text, (20, height - 35))

        pygame.display.flip()
        clock.tick(60)

    env.close()
    pygame.quit()

    return running  # False if user wants to quit


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate MK save states visually")
    parser.add_argument("--game", default="mk1", help="Game alias (default: mk1)")
    args = parser.parse_args()

    config = get_game_config(args.game)
    game_dir = ROOT_DIR / config.game_dir_name
    integrations = game_dir / "custom_integrations"
    retro.data.Integrations.add_custom_path(str(integrations))

    states_dir = integrations / config.game_id

    # Get all Fight states
    state_files = sorted(states_dir.glob("Fight*.state"))

    if not state_files:
        print(f"No states found in {states_dir}")
        return

    print(f"Found {len(state_files)} states to validate")
    print("=" * 60)

    for i, state_path in enumerate(state_files, 1):
        state_name = state_path.stem  # Remove .state extension

        print(f"\n[{i}/{len(state_files)}] Validating: {state_name}")

        if not validate_state(config.game_id, state_name, integrations):
            print("\nValidation stopped by user.")
            break

    print("\n" + "=" * 60)
    print("Validation complete!")


if __name__ == "__main__":
    main()

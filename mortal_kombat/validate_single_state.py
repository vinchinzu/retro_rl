#!/usr/bin/env python3
"""
Validate a single state by showing it.
"""

import os
import sys
from pathlib import Path

if not os.environ.get("SDL_VIDEODRIVER"):
    os.environ["SDL_VIDEODRIVER"] = "x11"

import numpy as np
import pygame

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

import stable_retro as retro
from fighters_common.game_configs import get_game_config


def validate_state(game_id, state_name, integrations_path, duration=10):
    """Show a save state for specified duration."""
    env = retro.make(
        game=game_id,
        state=state_name,
        render_mode="rgb_array",
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        use_restricted_actions=retro.Actions.ALL,
    )

    pygame.init()
    SCALE = 3
    width, height = 256 * SCALE, 224 * SCALE
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"Validating: {state_name}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 24)

    obs = env.reset()
    noop = np.zeros(env.action_space.shape, dtype=np.int8)

    print(f"\nShowing: {state_name} for {duration} seconds")
    print("  Press SPACE to finish early, Q/ESC to quit")

    frame_count = 0
    running = True
    max_frames = duration * 60

    while running and frame_count < max_frames:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    print("Finished early")
                    running = False

        obs, reward, terminated, truncated, info = env.step(noop)
        frame_count += 1

        frame = env.render()
        if frame is not None:
            surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            screen.blit(pygame.transform.scale(surf, (width, height)), (0, 0))

        text = font.render(state_name, True, (255, 255, 0))
        text_bg = pygame.Surface((text.get_width() + 20, text.get_height() + 10))
        text_bg.fill((0, 0, 0))
        text_bg.set_alpha(200)
        screen.blit(text_bg, (10, 10))
        screen.blit(text, (20, 15))

        pygame.display.flip()
        clock.tick(60)

    env.close()
    pygame.quit()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate a single MK state")
    parser.add_argument("state", help="State name (without .state extension)")
    parser.add_argument("--duration", type=int, default=10, help="Seconds to show")
    parser.add_argument("--game", default="mk1", help="Game alias")
    args = parser.parse_args()

    config = get_game_config(args.game)
    game_dir = ROOT_DIR / config.game_dir_name
    integrations = game_dir / "custom_integrations"
    retro.data.Integrations.add_custom_path(str(integrations))

    validate_state(config.game_id, args.state, integrations, args.duration)


if __name__ == "__main__":
    main()

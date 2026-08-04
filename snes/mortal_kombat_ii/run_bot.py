#!/usr/bin/env python3
"""Mortal Kombat II (SNES): Interactive play and training."""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from retro_harness import (
    add_custom_integrations, make_env, keyboard_action,
    controller_action, sanitize_action, init_controller,
    save_state, get_available_states,
)

GAME = "MortalKombatII-Snes"


def play(args):
    import pygame

    add_custom_integrations(SCRIPT_DIR)
    states = get_available_states(GAME, SCRIPT_DIR)
    state = args.state if args.state in states else (states[0] if states else "NONE")

    env = make_env(game=GAME, state=state, game_dir=SCRIPT_DIR)
    obs, info = env.reset()

    pygame.init()
    screen = pygame.display.set_mode((obs.shape[1] * 3, obs.shape[0] * 3))
    pygame.display.set_caption(f"Mortal Kombat II - {state}")
    joystick = init_controller(pygame)
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_F5:
                save_state(env, SCRIPT_DIR, GAME, "QuickSave")

        keys = pygame.key.get_pressed()
        action = [0] * 12
        keyboard_action(keys, action, pygame)
        controller_action(joystick, action)
        sanitize_action(action)

        obs, reward, terminated, truncated, info = env.step(action)
        surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
        screen.blit(pygame.transform.scale(surf, screen.get_size()), (0, 0))

        font = pygame.font.SysFont("monospace", 16)
        hud = font.render(f"P1: {info.get('health', '?')}  P2: {info.get('enemy_health', '?')}", True, (255, 255, 0))
        screen.blit(hud, (10, 10))
        pygame.display.flip()
        clock.tick(60)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    pygame.quit()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    play_p = sub.add_parser("play")
    play_p.add_argument("--state", default="Start")
    args = parser.parse_args()
    if args.command == "play":
        play(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

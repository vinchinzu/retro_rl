#!/usr/bin/env python3
"""
Street Fighter II Turbo - Hyper Fighting
Interactive play and training entry point.
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from retro_harness import (
    add_custom_integrations,
    make_env,
    keyboard_action,
    controller_action,
    sanitize_action,
    init_controller,
    save_state,
    get_available_states,
)

GAME = "StreetFighterIITurbo-Snes"


def play(args):
    """Interactive play mode with keyboard/controller."""
    import pygame

    add_custom_integrations(SCRIPT_DIR)
    states = get_available_states(GAME, SCRIPT_DIR)
    state = args.state
    if state not in states and states:
        print(f"State '{state}' not found. Available: {states}")
        state = states[0] if states else "NONE"
        print(f"Using: {state}")

    env = make_env(game=GAME, state=state, game_dir=SCRIPT_DIR, render_mode="rgb_array")
    obs, info = env.reset()

    pygame.init()
    screen_w, screen_h = obs.shape[1] * 3, obs.shape[0] * 3
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption(f"SF2 Turbo - {state}")
    joystick = init_controller(pygame)
    clock = pygame.time.Clock()
    running = True
    frame = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F5:
                    save_state(env, SCRIPT_DIR, GAME, f"QuickSave")
                    print("State saved!")

        keys = pygame.key.get_pressed()
        action = [0] * 12
        keyboard_action(keys, action, pygame)
        controller_action(joystick, action)
        sanitize_action(action)

        obs, reward, terminated, truncated, info = env.step(action)
        frame += 1

        # Display health info
        health = info.get("health", "?")
        enemy_health = info.get("enemy_health", "?")

        surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
        screen.blit(pygame.transform.scale(surf, (screen_w, screen_h)), (0, 0))

        # HUD overlay
        font = pygame.font.SysFont("monospace", 16)
        hud = font.render(f"P1: {health}  P2: {enemy_health}  F: {frame}", True, (255, 255, 0))
        screen.blit(hud, (10, 10))

        pygame.display.flip()
        clock.tick(60)

        if terminated or truncated:
            obs, info = env.reset()
            frame = 0

    env.close()
    pygame.quit()


def create_state(args):
    """Navigate menus and create a fight-ready save state."""
    from fighters_common import create_fight_state, get_game_config

    config = get_game_config(GAME)
    create_fight_state(
        game=GAME,
        game_dir=SCRIPT_DIR,
        state_name=args.state_name,
        menu_sequence=config.menu_sequence,
        settle_frames=config.menu_settle_frames,
    )


def main():
    parser = argparse.ArgumentParser(description="Street Fighter II Turbo")
    sub = parser.add_subparsers(dest="command")

    play_p = sub.add_parser("play", help="Interactive play")
    play_p.add_argument("--state", default="Start", help="State to load")

    state_p = sub.add_parser("create-state", help="Create fight-ready save state")
    state_p.add_argument("--state-name", default="Fight_Ryu_vs_CPU", help="Name for saved state")

    args = parser.parse_args()
    if args.command == "play":
        play(args)
    elif args.command == "create-state":
        create_state(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

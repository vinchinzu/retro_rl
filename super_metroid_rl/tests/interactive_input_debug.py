#!/usr/bin/env python3
"""Consolidated interactive input debug tool for Super Metroid RL."""

import os
# Ensure we use X11 for display
os.environ['SDL_VIDEODRIVER'] = 'x11'

import numpy as np
import stable_retro as retro
import pygame

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
INTEGRATION_PATH = os.path.join(PROJECT_DIR, "custom_integrations")
retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

def main():
    pygame.init()
    pygame.joystick.init()

    joy = None
    if pygame.joystick.get_count() > 0:
        joy = pygame.joystick.Joystick(0)
        joy.init()
        print(f"Controller: {joy.get_name()}")

    state_name = 'BossTorizo'
    print(f"Starting interactive session in {state_name}...")
    
    try:
        env = retro.make(
            game='SuperMetroid-Snes', 
            state=state_name, 
            render_mode='rgb_array', 
            use_restricted_actions=retro.Actions.ALL
        )
    except Exception as e:
        print(f"Failed to load environment: {e}")
        return

    obs, info = env.reset()
    screen = pygame.display.set_mode((512, 500))
    pygame.display.set_caption("Input Debug - Press ESC to quit")
    font = pygame.font.SysFont('monospace', 14)
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: running = False

        keys = pygame.key.get_pressed()
        action = [0] * 12

        # Keyboard mapping
        if keys[pygame.K_RIGHT]: action[7] = 1
        if keys[pygame.K_LEFT]:  action[6] = 1
        if keys[pygame.K_DOWN]:  action[5] = 1
        if keys[pygame.K_UP]:    action[4] = 1
        if keys[pygame.K_z]: action[0] = 1  # B
        if keys[pygame.K_x]: action[8] = 1  # A
        if keys[pygame.K_a]: action[1] = 1  # Y
        if keys[pygame.K_s]: action[9] = 1  # X
        if keys[pygame.K_q]: action[10] = 1 # L
        if keys[pygame.K_w]: action[11] = 1 # R
        if keys[pygame.K_LSHIFT]: action[2] = 1 # SELECT
        if keys[pygame.K_RETURN]: action[3] = 1 # START

        # Controller mapping
        if joy:
            if joy.get_numhats() > 0:
                hat = joy.get_hat(0)
                if hat[0] > 0: action[7] = 1
                if hat[0] < 0: action[6] = 1
                if hat[1] > 0: action[4] = 1
                if hat[1] < 0: action[5] = 1
            
            DEADZONE = 0.3
            if joy.get_axis(0) > DEADZONE: action[7] = 1
            if joy.get_axis(0) < -DEADZONE: action[6] = 1
            if joy.get_axis(1) > DEADZONE: action[5] = 1
            if joy.get_axis(1) < -DEADZONE: action[4] = 1

            if joy.get_button(0): action[0] = 1   # B
            if joy.get_button(1): action[8] = 1   # A
            if joy.get_button(2): action[1] = 1   # Y
            if joy.get_button(3): action[9] = 1   # X
            if joy.get_button(4): action[10] = 1  # L
            if joy.get_button(5): action[11] = 1  # R
            if joy.get_button(6): action[2] = 1   # SELECT
            if joy.get_button(7): action[3] = 1   # START

        obs, reward, term, trunc, info = env.step(action)

        # Render
        surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
        screen.blit(pygame.transform.scale(surf, (512, 448)), (0, 0))

        # HUD
        pygame.draw.rect(screen, (20, 20, 40), (0, 448, 512, 52))
        hud_text = f"HP: {info.get('health', '?')} | Missiles: {info.get('missiles', '?')} | Item: {info.get('selected_item', '?')}"
        text_surf = font.render(hud_text, True, (255, 255, 0))
        screen.blit(text_surf, (10, 452))
        
        btn_names = ['B','Y','Sel','Sta','U','D','L','R','A','X','L','R']
        pressed = [btn_names[i] for i, v in enumerate(action) if v]
        btn_str = ' '.join(pressed) if pressed else '(none)'
        btn_surf = font.render(f"Buttons: {btn_str}", True, (0, 255, 0))
        screen.blit(btn_surf, (10, 472))

        pygame.display.flip()
        clock.tick(60)

        if term or trunc: break

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()

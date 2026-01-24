#!/usr/bin/env python3
"""
Record boss fight demos with auto-save to prevent overwrites.
Each recording gets a unique timestamp.

Usage:
    ../retro_env/bin/python record_boss.py
"""

import os
import sys
import shutil
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDING_DIR = os.path.join(SCRIPT_DIR, "recordings")
INTEGRATION_PATH = os.path.join(SCRIPT_DIR, "custom_integrations")

os.environ['SDL_VIDEODRIVER'] = 'x11'

import stable_retro as retro
retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

import pygame

def record_boss():
    """Record a boss fight demo."""
    pygame.init()
    pygame.font.init()
    pygame.joystick.init()
    font = pygame.font.SysFont('monospace', 16)

    # Initialize controller if available
    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"Controller detected: {joystick.get_name()}")

    os.makedirs(RECORDING_DIR, exist_ok=True)

    env = retro.make(
        game="SuperMetroid-Snes",
        state="BossTorizo",
        render_mode='rgb_array',
        record=RECORDING_DIR,
        use_restricted_actions=retro.Actions.ALL
    )

    obs, info = env.reset()
    screen = pygame.display.set_mode((obs.shape[1]*2, obs.shape[0]*2))
    pygame.display.set_caption("Recording Boss Fight - Press ESC when done")
    clock = pygame.time.Clock()

    running = True
    frame = 0

    print("="*50)
    print("RECORDING BOSS FIGHT")
    print("Keyboard: Arrows=Move, Z=Run, X=Jump, A=Shoot")
    if joystick:
        print(f"Controller: {joystick.get_name()}")
        print("  D-Pad/Stick=Move, A=Jump, B=Run, Y=Shoot, X=Missile")
    print("Press ESC to save and quit")
    print("="*50)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Get keyboard input
        keys = pygame.key.get_pressed()
        # SNES: [B, Y, Select, Start, Up, Down, Left, Right, A, X, L, R]
        action = [0] * 12

        # Keyboard D-Pad
        if keys[pygame.K_RIGHT]: action[7] = 1
        if keys[pygame.K_LEFT]:  action[6] = 1
        if keys[pygame.K_DOWN]:  action[5] = 1
        if keys[pygame.K_UP]:    action[4] = 1

        # Keyboard Buttons
        if keys[pygame.K_z]: action[0] = 1  # B (Run)
        if keys[pygame.K_x]: action[8] = 1  # A (Jump)
        if keys[pygame.K_a]: action[1] = 1  # Y (Shoot)
        if keys[pygame.K_s]: action[9] = 1  # X (Missile)
        if keys[pygame.K_q]: action[10] = 1 # L (Aim)
        if keys[pygame.K_w]: action[11] = 1 # R (Aim)
        if keys[pygame.K_RETURN]: action[3] = 1  # Start
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1  # Select

        # Controller input
        if joystick:
            # Left analog stick with deadzone
            DEADZONE = 0.3
            axis_x = joystick.get_axis(0)
            axis_y = joystick.get_axis(1)
            if axis_x > DEADZONE:  action[7] = 1  # Right
            if axis_x < -DEADZONE: action[6] = 1  # Left
            if axis_y > DEADZONE:  action[5] = 1  # Down
            if axis_y < -DEADZONE: action[4] = 1  # Up

            # D-Pad (hat 0)
            if joystick.get_numhats() > 0:
                hat = joystick.get_hat(0)
                if hat[0] > 0:  action[7] = 1   # Right
                if hat[0] < 0:  action[6] = 1   # Left
                if hat[1] > 0:  action[4] = 1   # Up
                if hat[1] < 0:  action[5] = 1   # Down

            # 8BitDo SN30 Pro / Xbox style controller
            # B=0, A=1, Y=2, X=3, LB=4, RB=5, Back=6, Start=7
            if joystick.get_button(0): action[0] = 1   # B -> SNES B (Run)
            if joystick.get_button(1): action[8] = 1   # A -> SNES A (Jump)
            if joystick.get_button(2): action[1] = 1   # Y -> SNES Y (Shoot)
            if joystick.get_button(3): action[9] = 1   # X -> SNES X (Missile)
            if joystick.get_button(4): action[10] = 1  # LB -> SNES L (Aim)
            if joystick.get_button(5): action[11] = 1  # RB -> SNES R (Aim)
            if joystick.get_button(6): action[2] = 1   # Back -> Select
            if joystick.get_button(7): action[3] = 1   # Start -> Start

        obs, reward, terminated, truncated, info = env.step(action)
        frame += 1

        # Render
        surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
        scaled = pygame.transform.scale(surf, (obs.shape[1]*2, obs.shape[0]*2))
        screen.blit(scaled, (0, 0))

        # HUD
        boss_hp = info.get('boss_hp', 0)
        samus_hp = info.get('health', 0)
        text = font.render(f"Frame: {frame} | Boss: {boss_hp} | Samus: {samus_hp}", True, (255, 255, 0))
        screen.blit(text, (10, 10))

        rec_text = font.render("REC", True, (255, 0, 0))
        pygame.draw.circle(screen, (255, 0, 0), (screen.get_width() - 50, 20), 8)
        screen.blit(rec_text, (screen.get_width() - 40, 12))

        pygame.display.flip()
        clock.tick(60)

        if terminated or truncated:
            print(f"Episode ended at frame {frame}")
            break

    env.close()
    pygame.quit()

    # Auto-rename recording with timestamp
    src = os.path.join(RECORDING_DIR, "SuperMetroid-Snes-BossTorizo-000000.bk2")
    if os.path.exists(src):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = os.path.join(RECORDING_DIR, f"boss_demo_{timestamp}.bk2")
        shutil.move(src, dst)
        print(f"\nSaved: {dst}")
        print(f"Frames: {frame}")
    else:
        print("Warning: Recording file not found")

if __name__ == "__main__":
    record_boss()

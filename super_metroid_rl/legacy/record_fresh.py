#!/usr/bin/env python3
"""
Start from title screen to create fresh save with default controls.

1. Start the game from title screen
2. Go through file select (creates default control config)
3. Save state after intro for ZebesStart
4. Save state before Torizo for BossTorizo
"""
import os
os.environ['SDL_VIDEODRIVER'] = 'x11'

import stable_retro as retro
import pygame
import gzip
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(SCRIPT_DIR, "custom_integrations", "SuperMetroid-Snes")
RETRO_STATE_DIR = os.path.dirname(retro.data.get_file_path("SuperMetroid-Snes", "rom.sha"))
RECORDING_DIR = os.path.join(SCRIPT_DIR, "recordings")

os.makedirs(RECORDING_DIR, exist_ok=True)

def save_state(env, filename):
    """Save state to both local and retro directories."""
    state_data = env.em.get_state()

    # Save locally
    local_path = os.path.join(STATE_DIR, filename)
    with gzip.open(local_path, "wb") as f:
        f.write(state_data)
    print(f"Saved: {local_path}")

    # Copy to retro package
    retro_path = os.path.join(RETRO_STATE_DIR, filename)
    shutil.copy(local_path, retro_path)
    print(f"Copied: {retro_path}")

pygame.init()
pygame.joystick.init()

joy = None
if pygame.joystick.get_count() > 0:
    joy = pygame.joystick.Joystick(0)
    joy.init()
    print(f"Controller: {joy.get_name()}")

# Start from absolute beginning - no state (title screen)
env = retro.make(
    game='SuperMetroid-Snes',
    state=retro.State.NONE,
    render_mode='rgb_array',
    record=RECORDING_DIR
)
obs, info = env.reset()

screen = pygame.display.set_mode((512, 500))
pygame.display.set_caption("Fresh Start - Create New Game with Default Controls")
font = pygame.font.SysFont('monospace', 14)
clock = pygame.time.Clock()

print("""
============================================================
  FRESH START - Create new save with default controls
============================================================

1. Press START at title screen
2. Select 'SAMUS A' or empty slot
3. Play intro sequence
4. When on Zebes ship: F1 = ZebesStart.state
5. Before Torizo boss: F3 = BossTorizo.state

Controls same as before. ESC to quit.
============================================================
""")

running = True
frame = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_F1:
                save_state(env, "ZebesStart.state")
            elif event.key == pygame.K_F3:
                save_state(env, "BossTorizo.state")
            elif event.key == pygame.K_F5:
                from datetime import datetime
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_state(env, f"QuickSave_{ts}.state")

    keys = pygame.key.get_pressed()
    action = [0] * 12

    # Keyboard
    if keys[pygame.K_RIGHT]: action[7] = 1
    if keys[pygame.K_LEFT]:  action[6] = 1
    if keys[pygame.K_DOWN]:  action[5] = 1
    if keys[pygame.K_UP]:    action[4] = 1
    if keys[pygame.K_z]: action[0] = 1
    if keys[pygame.K_x]: action[8] = 1
    if keys[pygame.K_a]: action[1] = 1
    if keys[pygame.K_s]: action[9] = 1
    if keys[pygame.K_q]: action[10] = 1
    if keys[pygame.K_w]: action[11] = 1
    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
    if keys[pygame.K_RETURN]: action[3] = 1

    # Controller
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

        if joy.get_button(0): action[0] = 1   # B -> Run
        if joy.get_button(1): action[8] = 1   # A -> Jump
        if joy.get_button(2): action[1] = 1   # Y -> Shoot
        if joy.get_button(3): action[9] = 1   # X -> Item Cancel
        if joy.get_button(4): action[10] = 1  # LB -> L
        if joy.get_button(5): action[11] = 1  # RB -> R
        if joy.get_button(6): action[2] = 1   # Back -> SELECT
        if joy.get_button(7): action[3] = 1   # Start

    obs, reward, term, trunc, info = env.step(action)
    frame += 1

    # Render
    surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
    screen.blit(pygame.transform.scale(surf, (512, 448)), (0, 0))

    # HUD
    pygame.draw.rect(screen, (20, 20, 40), (0, 448, 512, 52))

    missiles = info.get('missiles', 0)
    health = info.get('health', 0)
    game_state = info.get('game_state', 0)

    text1 = font.render(f"Missiles:{missiles} HP:{health} State:{game_state} Frame:{frame}", True, (255,255,0))
    screen.blit(text1, (10, 452))

    btn_names = ['B','Y','Sel','Sta','U','D','L','R','A','X','LB','RB']
    pressed = [btn_names[i] for i, v in enumerate(action) if v]
    text2 = font.render(f"Buttons: {' '.join(pressed) if pressed else '(none)'}", True, (0,255,0) if pressed else (100,100,100))
    screen.blit(text2, (10, 472))

    pygame.display.flip()
    clock.tick(60)

env.close()
pygame.quit()
print("Done!")

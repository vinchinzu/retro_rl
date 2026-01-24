"""
Shared controller/keyboard configuration for Harvest Moon Bot

SNES Button indices: B=0, Y=1, Select=2, Start=3, Up=4, Down=5, Left=6, Right=7, A=8, X=9, L=10, R=11
"""

import os

os.environ.setdefault('SDL_VIDEODRIVER', 'x11')

import pygame

# SNES BUTTON INDICES
SNES_B = 0
SNES_Y = 1
SNES_SELECT = 2
SNES_START = 3
SNES_UP = 4
SNES_DOWN = 5
SNES_LEFT = 6
SNES_RIGHT = 7
SNES_A = 8
SNES_X = 9
SNES_L = 10
SNES_R = 11

# KEYBOARD MAPPING
KEYBOARD_MAP = {
    pygame.K_z: SNES_B,       # B (cancel/run)
    pygame.K_x: SNES_Y,       # Y (use item)
    pygame.K_TAB: SNES_SELECT,
    pygame.K_RETURN: SNES_START,
    pygame.K_UP: SNES_UP,
    pygame.K_DOWN: SNES_DOWN,
    pygame.K_LEFT: SNES_LEFT,
    pygame.K_RIGHT: SNES_RIGHT,
    pygame.K_c: SNES_A,       # A (confirm)
    pygame.K_v: SNES_X,       # X (menu)
    pygame.K_a: SNES_L,       # L (item left)
    pygame.K_s: SNES_R,       # R (item right)
}

# CONTROLLER MAPPING (Xbox-style controller)
CONTROLLER_MAP = {
    0: SNES_B,      # Xbox A -> SNES B (cancel)
    1: SNES_A,      # Xbox B -> SNES A (confirm)
    2: SNES_Y,      # Xbox X -> SNES Y (use item)
    3: SNES_X,      # Xbox Y -> SNES X (menu)
    4: SNES_L,      # Xbox LB -> SNES L
    5: SNES_R,      # Xbox RB -> SNES R
    6: SNES_SELECT, # Xbox Back/Select
    7: SNES_START,  # Xbox Start
}

# HOTSWAP CHORD: L + R + SELECT (buttons 4 + 5 + 6 on Xbox controller)
HOTSWAP_CHORD = {SNES_L, SNES_R, SNES_SELECT}
HOTSWAP_KEYS = {pygame.K_a, pygame.K_s, pygame.K_TAB}  # L + R + Select on keyboard


def init_controller():
    """Initialize and return first available controller, or None."""
    pygame.joystick.init()
    if pygame.joystick.get_count() > 0:
        joy = pygame.joystick.Joystick(0)
        joy.init()
        return joy
    return None


def get_controller_action(joystick, action):
    """Read controller input and update action array in-place."""
    if joystick is None:
        return

    # D-pad via hat
    if joystick.get_numhats() > 0:
        hat = joystick.get_hat(0)
        if hat[0] < 0: action[SNES_LEFT] = 1
        if hat[0] > 0: action[SNES_RIGHT] = 1
        if hat[1] > 0: action[SNES_UP] = 1
        if hat[1] < 0: action[SNES_DOWN] = 1

    # Left stick as backup
    if joystick.get_numaxes() >= 2:
        axis_x = joystick.get_axis(0)
        axis_y = joystick.get_axis(1)
        if axis_x < -0.5: action[SNES_LEFT] = 1
        if axis_x > 0.5: action[SNES_RIGHT] = 1
        if axis_y < -0.5: action[SNES_UP] = 1
        if axis_y > 0.5: action[SNES_DOWN] = 1

    # Buttons
    for joy_btn, snes_btn in CONTROLLER_MAP.items():
        if joy_btn < joystick.get_numbuttons() and joystick.get_button(joy_btn):
            action[snes_btn] = 1


def get_keyboard_action(keys, action):
    """Read keyboard input and update action array in-place."""
    for key, btn_idx in KEYBOARD_MAP.items():
        if keys[key]:
            action[btn_idx] = 1


def sanitize_action(action):
    """Remove contradictory D-pad inputs."""
    if action[SNES_LEFT] and action[SNES_RIGHT]:
        action[SNES_LEFT] = 0
        action[SNES_RIGHT] = 0
    if action[SNES_UP] and action[SNES_DOWN]:
        action[SNES_UP] = 0
        action[SNES_DOWN] = 0


def check_hotswap_chord(joystick, keys):
    """Check if hotswap chord (L+R+SELECT) is pressed."""
    # Check keyboard
    if all(keys[k] for k in HOTSWAP_KEYS):
        return True

    # Check controller
    if joystick is not None:
        try:
            l_pressed = joystick.get_button(4) if joystick.get_numbuttons() > 4 else False
            r_pressed = joystick.get_button(5) if joystick.get_numbuttons() > 5 else False
            sel_pressed = joystick.get_button(6) if joystick.get_numbuttons() > 6 else False
            if l_pressed and r_pressed and sel_pressed:
                return True
        except:
            pass

    return False





def print_controls(joystick=None):
    """Print control scheme."""
    print("\nControls:")
    if joystick:
        print(f"  Controller: {joystick.get_name()}")
        print("    D-Pad/Stick: Movement")
        print("    A: Confirm | B: Cancel | X: Menu | Y: Use Item")
        print("    LB/RB: Cycle Items")
        print("    LB+RB+SELECT: Toggle Human/Bot Mode")
    print("  Keyboard:")
    print("    Arrows: D-Pad")
    print("    Z: Cancel (B) | C: Confirm (A) | V: Menu (X) | X: Use Item (Y)")
    print("    A/S: Cycle Items (L/R)")
    print("    A+S+TAB: Toggle Human/Bot Mode")
    print("    P: Mark current tile as no-go (debug)")

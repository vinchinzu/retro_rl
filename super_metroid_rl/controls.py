"""
Shared controller/keyboard configuration for Super Metroid RL

SNES Button indices: B=0, Y=1, Select=2, Start=3, Up=4, Down=5, Left=6, Right=7, A=8, X=9, L=10, R=11
"""

import os

# Set SDL video driver for better Arch/Wayland compatibility
os.environ.setdefault('SDL_VIDEODRIVER', 'x11')

import pygame

# =============================================================================
# SNES BUTTON INDICES
# =============================================================================
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

# =============================================================================
# KEYBOARD MAPPING
# =============================================================================
KEYBOARD_MAP = {
    pygame.K_z: SNES_B,       # B (run)
    pygame.K_x: SNES_Y,       # Y (item cancel)
    pygame.K_TAB: SNES_SELECT,
    pygame.K_RETURN: SNES_START,
    pygame.K_UP: SNES_UP,
    pygame.K_DOWN: SNES_DOWN,
    pygame.K_LEFT: SNES_LEFT,
    pygame.K_RIGHT: SNES_RIGHT,
    pygame.K_c: SNES_A,       # A (jump)
    pygame.K_v: SNES_X,       # X (shoot)
    pygame.K_a: SNES_L,       # L (aim up)
    pygame.K_s: SNES_R,       # R (aim down)
}

# =============================================================================
# CONTROLLER MAPPING (Xbox-style controller)
# =============================================================================
# Xbox button indices: A=0, B=1, X=2, Y=3, LB=4, RB=5, Select/Back=6, Start=7
CONTROLLER_MAP = {
    0: SNES_A,      # Xbox A -> SNES A (jump)
    1: SNES_B,      # Xbox B -> SNES B (run)
    2: SNES_X,      # Xbox X -> SNES X (shoot)
    3: SNES_Y,      # Xbox Y -> SNES Y (item cancel)
    4: SNES_L,      # Xbox LB -> SNES L (aim up)
    5: SNES_R,      # Xbox RB -> SNES R (aim down)
    6: SNES_SELECT, # Xbox Back/Select
    7: SNES_START,  # Xbox Start
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def init_controller():
    """Initialize and return first available controller, or None."""
    pygame.joystick.init()
    if pygame.joystick.get_count() > 0:
        joy = pygame.joystick.Joystick(0)
        joy.init()
        return joy
    return None


def get_controller_action(joystick, action):
    """
    Read controller input and update action array in-place.

    Args:
        joystick: pygame.Joystick instance
        action: numpy array of shape (12,) to update
    """
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
    """
    Read keyboard input and update action array in-place.

    Args:
        keys: pygame key state from pygame.key.get_pressed()
        action: numpy array of shape (12,) to update
    """
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


def print_controls(joystick=None):
    """Print control scheme."""
    print("\nControls:")
    if joystick:
        print(f"  Controller: {joystick.get_name()}")
        print("    D-Pad/Stick: Movement")
        print("    A: Jump | B: Run | X: Shoot | Y: Item Cancel")
        print("    LB/RB: Aim Up/Down")
        print("    SELECT: Save State | START: Save & Exit")
    print("  Keyboard:")
    print("    Arrows: D-Pad")
    print("    Z: Run (B) | C: Jump (A) | V: Shoot (X) | X: Item Cancel (Y)")
    print("    A/S: Aim Up/Down (L/R)")

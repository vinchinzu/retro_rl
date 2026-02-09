#!/usr/bin/env python3
"""
Manually create save states by playing through the game.
Press F1-F12 to save state for each character/opponent.
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

# SF2 Turbo has 12 World Warriors
SAVE_OPTIONS = [
    "Ryu",          # F1
    "EHonda",       # F2
    "Blanka",       # F3
    "Guile",        # F4
    "Ken",          # F5
    "ChunLi",       # F6
    "Zangief",      # F7
    "Dhalsim",      # F8
    "Balrog",       # F9 (Boxer/M.Bison in JP)
    "Vega",         # F10 (Claw/Balrog in JP)
    "Sagat",        # F11
    "MBison",       # F12 (Dictator/Vega in JP)
]

def main():
    config = get_game_config("sf2")
    game_dir = ROOT_DIR / config.game_dir_name
    integrations = game_dir / "custom_integrations"
    retro.data.Integrations.add_custom_path(str(integrations))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-start", action="store_true",
                       help="Start from ROM boot (for character selection)")
    parser.add_argument("--from-waypoint", action="store_true",
                       help="Start from character select waypoint (fastest!)")
    args = parser.parse_args()

    # Determine starting state
    if args.from_waypoint:
        start_state = "CharSelect_StreetFighterIITurbo"
    elif args.from_start:
        start_state = retro.State.NONE
    else:
        start_state = "Fight_StreetFighterIITurbo"

    env = retro.make(
        game=config.game_id,
        state=start_state,
        render_mode="rgb_array",
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        use_restricted_actions=retro.Actions.ALL,
    )

    # Pygame setup
    pygame.init()
    SCALE = 3
    width, height = 256 * SCALE, 224 * SCALE
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("SF2 Turbo Manual State Creator")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)

    # Controller mapping for SF2
    button_map = {
        pygame.K_x: 1,      # Y (Medium Punch)
        pygame.K_z: 9,      # X (Heavy Punch)
        pygame.K_RSHIFT: 2, # SELECT
        pygame.K_RETURN: 3, # START
        pygame.K_UP: 4,
        pygame.K_DOWN: 5,
        pygame.K_LEFT: 6,
        pygame.K_RIGHT: 7,
        pygame.K_s: 0,      # B (Light Kick)
        pygame.K_a: 8,      # A (Medium Kick)
        pygame.K_q: 10,     # L (Heavy Kick)
    }

    obs, info = env.reset()
    running = True

    # Store initial state for reset
    initial_state = env.em.get_state()

    print("\n" + "="*60)
    print("SF2 TURBO MANUAL STATE CREATOR")
    print("="*60)
    print("\nControls:")
    print("  Arrow Keys: Move")
    print("  Z: Heavy Punch   X: Medium Punch")
    print("  Q: Heavy Kick    A: Medium Kick    S: Light Kick")
    print("  ENTER: Start")
    print("  TAB: Turbo (hold for fast-forward)")
    print("\nSave States (F1-F12):")
    if args.from_start or args.from_waypoint:
        print("  (Starting from boot - save as character states)")
        for i, char in enumerate(SAVE_OPTIONS, 1):
            print(f"  F{i}: Save Fight_{char}")
    else:
        print("  (In tournament - save as opponent states)")
        for i in range(1, 13):
            if i+1 <= 12:
                print(f"  F{i}: Save Fight_vs_Opponent{i+1}")
    print("\n  R: RESET to starting point (quick!)")
    print("  ESC: Quit")
    print("="*60)
    print("\nTIP: Save the state RIGHT when the fight starts")
    print("     (when 'FIGHT!' appears and timer begins)")
    print()

    while running:
        action = np.zeros(12, dtype=np.int8)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                # Check for F1-F12 to save states
                if pygame.K_F1 <= event.key <= pygame.K_F12:
                    save_idx = event.key - pygame.K_F1
                    if save_idx < len(SAVE_OPTIONS):
                        save_state(env, config, game_dir, save_idx, args.from_start or args.from_waypoint)

                # R to reset to waypoint
                if event.key == pygame.K_r:
                    print("\n🔄 Resetting to starting point...")
                    env.em.set_state(initial_state)
                    obs, info = env.reset()
                    print("✓ Reset complete!")

        # Read current keys
        keys = pygame.key.get_pressed()
        for key, button_idx in button_map.items():
            if keys[key]:
                action[button_idx] = 1

        # Check for turbo mode (TAB)
        turbo_speed = 10 if keys[pygame.K_TAB] else 1

        # Step environment (multiple times if turbo active)
        for _ in range(turbo_speed):
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break

        # Render
        frame = env.render()
        if frame is not None:
            surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            screen.blit(pygame.transform.scale(surf, (width, height)), (0, 0))

        # Show instructions overlay
        y_offset = 10
        turbo_active = keys[pygame.K_TAB] if 'keys' in locals() else False
        turbo_indicator = " [TURBO x10]" if turbo_active else ""
        lines = [
            f"F1-F12: Save | R: Reset | TAB: Turbo{turbo_indicator}",
            f"HP: P1={info.get('health', 0)} P2={info.get('enemy_health', 0)} Timer={info.get('timer', 0)}",
        ]
        for line in lines:
            text = font.render(line, True, (255, 255, 0))
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

def save_state(env, config, game_dir, save_idx, from_start):
    """Save current state."""
    name = SAVE_OPTIONS[save_idx]

    if from_start:
        # Character state
        state_name = f"Fight_{name}"
    else:
        # Opponent state
        state_name = f"Fight_vs_Opponent{save_idx + 2}"  # +2 because F1=Opponent2

    save_path = game_dir / "custom_integrations" / config.game_id / f"{state_name}.state"

    try:
        state_data = env.em.get_state()
        with gzip.open(save_path, "wb") as f:
            f.write(state_data)
        print(f"\n✓ Saved: {state_name}.state")
    except Exception as e:
        print(f"\n✗ Failed to save {state_name}: {e}")

if __name__ == "__main__":
    main()

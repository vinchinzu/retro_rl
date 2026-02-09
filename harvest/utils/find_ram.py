#!/usr/bin/env python3
"""
RAM Address Finder & Analyzer for Harvest Moon SNES

Records gameplay sequences and correlates controller input with RAM changes
to find X/Y coordinates automatically.

Usage:
    ./run.sh python find_ram.py

Controls:
    R       - Toggle Recording (Red text = Recording)
    A       - Analyze recorded data (prints candidates to console)
    F5      - Save state
    ESC     - Exit
"""

import os
import sys
import gzip
import time
import numpy as np
import pygame
import stable_retro as retro

# Force software rendering for headless/compatibility
os.environ.setdefault('SDL_VIDEODRIVER', 'x11')
os.environ['SDL_SOFTWARE_RENDERER'] = '1'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

# utils/ is one level below harvest/, which is one level below retro_rl/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # utils/
HARVEST_DIR = os.path.dirname(SCRIPT_DIR)  # harvest/
ROOT_DIR = os.path.dirname(HARVEST_DIR)  # retro_rl/
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

INTEGRATION_PATH = os.path.join(HARVEST_DIR, "custom_integrations")
STATES_DIR = os.path.join(INTEGRATION_PATH, "HarvestMoon-Snes")

from retro_harness import (
    init_controller as _init_controller,
    controller_action,
    keyboard_action,
    sanitize_action,
    SNES_UP, SNES_DOWN, SNES_LEFT, SNES_RIGHT,
    SNES_L, SNES_R, SNES_A, SNES_B, SNES_Y, SNES_X,
)


# Wrappers for retro_harness (different signatures)
def init_controller():
    return _init_controller(pygame)


def get_controller_action(joystick, action):
    controller_action(joystick, action)


def get_keyboard_action(keys, action):
    keyboard_action(keys, action, pygame)


def print_controls(joystick=None):
    """Print RAM finder controls."""
    print("\nRAM Finder Controls:")
    if joystick:
        print(f"  Controller: {joystick.get_name()}")
    print("  Movement: Arrows/D-Pad")
    print("  R: Toggle Recording | A: Analyze | F5: Save | ESC: Exit")


retro.data.Integrations.add_custom_path(INTEGRATION_PATH)


class Recorder:
    def __init__(self):
        self.recording = False
        self.data_ram = []    # List of RAM snapshots
        self.data_input = []  # List of input vectors (dx, dy)
        
    def start(self):
        self.recording = True
        self.data_ram = []
        self.data_input = []
        print("[RECORDING] Started. Walk around! (Up/Down/Left/Right)")

    def stop(self):
        self.recording = False
        print(f"[RECORDING] Stopped. Captured {len(self.data_ram)} frames.")

    def update(self, ram, action):
        if not self.recording:
            return

        # Calculate logical input direction
        # 1 = Positive (Right/Down), -1 = Negative (Left/Up), 0 = None
        dx = 0
        dy = 0
        
        if action[SNES_RIGHT]: dx += 1
        if action[SNES_LEFT]:  dx -= 1
        
        if action[SNES_DOWN]: dy += 1
        if action[SNES_UP]:   dy -= 1

        # Tool cycling (User says X button)
        dt = 0
        if action[SNES_X]: dt = 1 # Just a flag that "Change Tool" was pressed

        self.data_ram.append(ram.copy())
        self.data_input.append((dx, dy, dt))


def analyze_correlation(recorder: Recorder):
    """Find addresses that correlate with movement."""
    if len(recorder.data_ram) < 30:
        print("[ERROR] Not enough data. Record at least 1-2 seconds of movement.")
        return

    print("\n" + "="*60)
    print("ANALYZING DATA...")
    print("="*60)

    rams = np.array(recorder.data_ram)          # (T, N_RAM)
    inputs = np.array(recorder.data_input)      # (T, 3)
    
    n_frames, n_ram = rams.shape
    
    ram_diff = np.diff(rams.astype(np.int16), axis=0) # (T-1, N_RAM)
    input_sliced = inputs[1:]                         # (T-1, 3)
    
    dx = input_sliced[:, 0]
    dy = input_sliced[:, 1]
    dt = input_sliced[:, 2] # Tool change input (X button)
    
    # Filter only frames where we are moving
    move_x_mask = dx != 0
    move_y_mask = dy != 0
    change_tool_mask = dt != 0
    
    change_tool_mask = dt != 0
    
    print(f"Frames with X movement: {np.sum(move_x_mask)}")
    print(f"Frames with Y movement: {np.sum(move_y_mask)}")
    print(f"Frames with Tool Input (X): {np.sum(change_tool_mask)}")

    # ---------------------------------------------------------
    # NOISE FILTER
    # ---------------------------------------------------------
    # Identify addresses that change when we are NOT doing anything.
    # These are likely timers, RNG, or frame counters.
    idle_mask = (dx == 0) & (dy == 0) & (dt == 0)
    print(f"Frames Idle: {np.sum(idle_mask)}")
    
    noisy_addresses = set()
    if np.sum(idle_mask) > 10:
        # Check changes during idle frames
        r_diff_idle = ram_diff[idle_mask]
        # Any column that is non-zero in r_diff_idle is noisy
        # We sum the absolute differences across all idle frames
        noise_sum = np.sum(np.abs(r_diff_idle), axis=0)
        noisy_indices = np.where(noise_sum > 0)[0]
        noisy_addresses = set(noisy_indices)
        print(f"Identified {len(noisy_addresses)} noisy addresses (timers/counters).")

    # ---------------------------------------------------------
    # ANALYZE VALUABLES (X/Y)
    # ---------------------------------------------------------
    for axis_name, mask, input_vec in [('X', move_x_mask, dx), ('Y', move_y_mask, dy)]:
        if np.sum(mask) > 10:
            r_diff = ram_diff[mask]
            in_v = input_vec[mask]
            
            # Correlation
            correlation_sum = np.sum(r_diff * in_v[:, None], axis=0)
            
            top_indices = np.argsort(correlation_sum)[::-1]
            
            print(f"\n[{axis_name} CANDIDATES] (Higher score = better)")
            count = 0
            for i in range(len(top_indices)):
                idx = top_indices[i]
                if idx in noisy_addresses:
                    continue
                    
                score = correlation_sum[idx]
                if score > 5:
                    print(f"  Addr 0x{idx:04x} ({idx:5d}): Score {score:4d} | Val: {rams[-1, idx]}")
                    count += 1
                if count >= 10:
                    break
        else:
             print(f"\n[{axis_name} INFO] Not enough movement.")

    # ---------------------------------------------------------
    # ANALYZE TOOL (X Button)
    # ---------------------------------------------------------
    # Look for values that change when X is pressed, but are STABLE otherwise
    if np.sum(change_tool_mask) > 0:
        r_diff = ram_diff[change_tool_mask]
        change_magnitude = np.sum(np.abs(r_diff), axis=0)
        
        # Penalize if it changed during movement but NO tool press (optional, but let's stick to idle noise first)
        
        top_indices = np.argsort(change_magnitude)[::-1]
        
        print(f"\n[TOOL CANDIDATES] (Val changed when X pressed)")
        count = 0
        for i in range(len(top_indices)):
            idx = top_indices[i]
            if idx in noisy_addresses:
                continue

            score = change_magnitude[idx]
            if score > 0:
                print(f"  Addr 0x{idx:04x} ({idx:5d}): Score {score:4d} | Val: {rams[-1, idx]}")
                count += 1
            if count >= 15:
                break
    else:
        print("\n[TOOL INFO] No X button presses recorded.")

    print("="*60)


def main():
    pygame.init()

    # Use existing integration loader
    env = retro.make(
        game="HarvestMoon-Snes",
        state="Y1_Spring_Day01_06h",
        inttype=retro.data.Integrations.ALL,
        render_mode="rgb_array"
    )
    obs, info = env.reset()
    h, w = obs.shape[0], obs.shape[1]
    scale = 3

    screen = pygame.display.set_mode((w * scale, h * scale))
    pygame.display.set_caption("RAM Recorder - Press 'R' to Record, 'A' to Analyze")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('monospace', 14)

    joystick = init_controller()
    recorder = Recorder()

    print("\n" + "=" * 60)
    print("DATA RECORDER & ANALYZER")
    print("=" * 60)
    print("1. Press 'R' to Start Recording")
    print("2. Move around clearly (Down for 2s, Right for 2s, etc)")
    print("3. Press 'R' to Stop")
    print("4. Press 'A' to Analyze")
    print("=" * 60)

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    if recorder.recording:
                        recorder.stop()
                    else:
                        recorder.start()
                elif event.key == pygame.K_a:
                    if not recorder.recording and len(recorder.data_ram) > 0:
                        analyze_correlation(recorder)
                    elif recorder.recording:
                        print("Stop recording first (Press R)")
                    else:
                        print("No data to analyze. Record first.")
                elif event.key == pygame.K_F5:
                    state_data = env.em.get_state()
                    with gzip.open(os.path.join(STATES_DIR, "Recorded.state"), 'wb') as f:
                        f.write(state_data)
                    print("[SAVED] Recorded.state")

        # Get inputs
        keys = pygame.key.get_pressed()
        action = np.zeros(12, dtype=np.int32)
        get_keyboard_action(keys, action)
        get_controller_action(joystick, action)
        sanitize_action(action)

        # Step
        obs, reward, term, trunc, info = env.step(action)
        
        # Record
        recorder.update(env.get_ram(), action)

        # Render
        surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
        scaled = pygame.transform.scale(surf, (w * scale, h * scale))
        screen.blit(scaled, (0, 0))

        # HUD
        status_color = (255, 0, 0) if recorder.recording else (255, 255, 0)
        status_text = f"REC: {'ON' if recorder.recording else 'OFF'} | Frames: {len(recorder.data_ram)}"
        
        text = font.render(status_text, True, status_color)
        screen.blit(text, (5, 5))

        ram = env.get_ram()
        if len(ram) > 0x0D2:
            money = (
                (int(ram[0x0D1]) & 0x0F)
                + ((int(ram[0x0D1]) >> 4) & 0x0F) * 10
                + (int(ram[0x0D2]) & 0x0F) * 100
                + ((int(ram[0x0D2]) >> 4) & 0x0F) * 1000
            )
            money_text = font.render(f"Money: ${money:,}", True, (255, 255, 255))
            screen.blit(money_text, (5, 25))
        
        instr = font.render("R=Rec/Stop A=Analyze F5=Save", True, (255, 255, 255))
        screen.blit(instr, (5, h * scale - 20))

        pygame.display.flip()
        clock.tick(60)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()

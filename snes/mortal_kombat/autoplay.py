#!/usr/bin/env python3
"""
Mortal Kombat Continuous Autoplay Bot with Manual Override.

Plays the game continuously from any starting point (Fight 1 through Shang Tsung).
Auto-detects continue screens and character select screens, and switches model
parameters on the fly if specialized models are available.

Controls:
  SPACE: Toggle Autoplay / Manual Override
  TAB:   Toggle Turbo mode
  R:     Reset emulator state
  Q/ESC: Quit

Manual controls (when override is active):
  Arrows: Move (Left/Right/Up/Down)
  A:      Low Punch
  S:      High Punch
  Z:      Low Kick
  X:      High Kick
  C:      Block
"""

import os
import time
from pathlib import Path
from collections import deque

# Ensure X11/Wayland compatibility for Pygame rendering
if not os.environ.get("SDL_VIDEODRIVER"):
    os.environ["SDL_VIDEODRIVER"] = "x11"

import numpy as np
import pygame
import cv2
import torch
from stable_baselines3 import PPO

# Set up paths
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent

import stable_retro as retro
from retro_harness.fighters.game_configs import get_game_config
from retro_harness.fighters.fighting_env import MK_FIGHTING_ACTIONS

# Configuration
FRAME_SKIP = 4
FRAME_STACK = 4
SCALE = 3
WIDTH, HEIGHT = 256 * SCALE, 224 * SCALE
HUD_HEIGHT = 160

OPPONENT_NAMES = {
    0: "Johnny Cage",
    1: "Kano",
    2: "Raiden",
    3: "Liu Kang",
    4: "Scorpion",
    5: "Sub-Zero",
    6: "Sonya",
    7: "Goro",
    8: "Shang Tsung"
}

MATCH_NAMES = {
    0: "Match 1 (Fight)",
    1: "Match 2",
    2: "Match 3",
    3: "Match 4",
    4: "Match 5",
    5: "Match 6",
    6: "Match 7 (Mirror)",
    7: "Endurance 1 (A)",
    8: "Endurance 1 (B)",
    9: "Endurance 2 (A)",
    10: "Endurance 2 (B - Goro)",
    11: "Shang Tsung (Final Boss)"
}

class AutoplayBot:
    def __init__(self, game_alias="mk1", state="Fight_LiuKang", model_dir_path=None, model_name=None):
        self.config = get_game_config(game_alias)
        self.game_dir = ROOT_DIR / self.config.game_dir_name
        self.model_dir = Path(model_dir_path) if model_dir_path else self.game_dir / "models"
        
        # Retro Integrations
        integrations = self.game_dir / "custom_integrations"
        retro.data.Integrations.add_custom_path(str(integrations))
        
        # Init retro environment (raw, no SB3 wrapper, players=1)
        self.env = retro.make(
            game=self.config.game_id,
            state=state,
            render_mode="rgb_array",
            inttype=retro.data.Integrations.CUSTOM_ONLY,
            use_restricted_actions=retro.Actions.ALL
        )
        
        # Observation Frame stack
        self.frame_queue = deque(maxlen=FRAME_STACK)
        self.reset_frame_stack()
        
        # Reset env and populate frame stack
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        self.update_frame_stack(obs)
        
        # Models dict and active model tracking
        self.models = {}
        self.active_model_name = "None"
        self.active_model = None
        self.load_models(model_name=model_name)

        # Bot and control state
        self.autoplay_active = True
        self.turbo = False
        self.frame_count = 0
        self.non_fight_frames = 0
        
        # Win/Loss counts
        self.wins = 0
        self.losses = 0
        self.last_match_id = -1
        self.last_p1_rounds = 0
        self.last_p2_rounds = 0

    def load_models(self, model_name=None):
        """Loads available models from the models directory."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        default_loaded = False
        if model_name:
            # Check if model_name is absolute or relative
            path = Path(model_name)
            if not path.is_absolute():
                path = self.model_dir / model_name
            if path.exists():
                print(f"Loading user-specified base model: {path.name}")
                self.models["default"] = PPO.load(str(path), device=device)
                self.active_model_name = path.name
                self.active_model = self.models["default"]
                default_loaded = True
            else:
                print(f"ERROR: Specified model path not found: {model_name}. Falling back to defaults.")

        if not default_loaded:
            # Load the default speedrun model
            default_candidates = [
                "mk1_speedrun_ppo_final.zip",
                "mk1_fresh_ppo_final.zip",
                "mk1_match7_ppo_final.zip",
                "mk1_multichar_ppo_2000000_steps.zip"
            ]
            
            for cand in default_candidates:
                path = self.model_dir / cand
                if path.exists():
                    print(f"Loading base speedrun model: {cand}")
                    self.models["default"] = PPO.load(str(path), device=device)
                    self.active_model_name = cand
                    self.active_model = self.models["default"]
                    default_loaded = True
                    break
                    
        if not default_loaded:
            # Fall back to the newest mk1*.zip file
            candidates = sorted(
                self.model_dir.glob("mk1*.zip"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if candidates:
                print(f"Loading latest model fallback: {candidates[0].name}")
                self.models["default"] = PPO.load(str(candidates[0]), device=device)
                self.active_model_name = candidates[0].name
                self.active_model = self.models["default"]
            else:
                print("WARNING: No pre-trained models found in models/. Running with random actions.")

        # Check for stage-specific specialized models
        spec_map = {
            "mirror": "mk1_match7_ppo_final.zip",
            "goro": "mk1_goro_ppo_final.zip",
            "shangtsung": "mk1_shangtsung_ppo_final.zip"
        }
        for key, fname in spec_map.items():
            path = self.model_dir / fname
            if path.exists():
                print(f"Loading specialized model for {key}: {fname}")
                self.models[key] = PPO.load(str(path), device=device)

    def select_model(self, match_id, opponent_id):
        """Swaps the active model on the fly based on RAM variables."""
        if not self.models:
            return
            
        # Match 7 (Mirror match against Liu Kang)
        if match_id == 6 and "mirror" in self.models:
            if self.active_model != self.models["mirror"]:
                self.active_model = self.models["mirror"]
                self.active_model_name = "mk1_match7_ppo_final.zip (Specialized Mirror)"
        # Goro (Match 10)
        elif opponent_id == 7 and "goro" in self.models:
            if self.active_model != self.models["goro"]:
                self.active_model = self.models["goro"]
                self.active_model_name = "mk1_goro_ppo_final.zip (Specialized Goro)"
        # Shang Tsung (Match 11)
        elif opponent_id == 8 and "shangtsung" in self.models:
            if self.active_model != self.models["shangtsung"]:
                self.active_model = self.models["shangtsung"]
                self.active_model_name = "mk1_shangtsung_ppo_final.zip (Specialized Shang)"
        # General default speedrun model
        else:
            if "default" in self.models and self.active_model != self.models["default"]:
                self.active_model = self.models["default"]
                # Reset display name to default loaded candidate
                self.active_model_name = "mk1_speedrun_ppo_final.zip"

    def reset_frame_stack(self):
        """Clears the frame queue."""
        self.frame_queue.clear()

    def update_frame_stack(self, rgb_frame):
        """Processes RGB frame to grayscale, resizes, and appends to frame stack."""
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        self.frame_queue.append(resized)
        
        # If queue not full yet, pad it
        while len(self.frame_queue) < FRAME_STACK:
            self.frame_queue.append(resized)

    def get_observation(self):
        """Prepares observations in shape (FRAME_STACK, 84, 84) for SB3 model."""
        return np.stack(self.frame_queue, axis=0)

    def decode_action(self, action_idx):
        """Maps discrete action index to button inputs."""
        buttons = np.zeros(12, dtype=np.int8)
        for btn, val in MK_FIGHTING_ACTIONS[action_idx].items():
            buttons[btn] = val
        return buttons

    def handle_character_select(self, p1_char):
        """Deterministic inputs to navigate cursor to Liu Kang and confirm."""
        buttons = np.zeros(12, dtype=np.int8)
        
        # If not Liu Kang (ID 3), press DOWN to move cursor from Johnny Cage (ID 0)
        # Sequence timing:
        if 180 <= self.non_fight_frames < 190:
            buttons[5] = 1  # Hold DOWN
        elif 210 <= self.non_fight_frames < 225:
            buttons[8] = 1  # Press A / Confirm (button 8)
            buttons[3] = 1  # Also mash START (button 3)
            
        return buttons

    def step(self, manual_buttons=None):
        """Steps the emulator and updates WRAM info."""
        self.frame_count += 1
        
        # 1. Read current RAM info
        ram = self.env.get_ram()
        health = int(ram[1209])
        enemy_health = int(ram[1211])
        continue_timer = int(ram[999])
        p1_char = int(ram[6514])
        p1_rounds = int(ram[6510])
        p2_rounds = int(ram[1207])
        match_id = int(ram[10])
        opponent_id = int(ram[36])

        # State updates for character select sequence
        if health == 0 and enemy_health == 0:
            self.non_fight_frames += 1
        else:
            self.non_fight_frames = 0

        # Update Match stats and score tracking
        if match_id != self.last_match_id:
            self.last_match_id = match_id
            self.last_p1_rounds = 0
            self.last_p2_rounds = 0

        # Detect round completions to update win/loss counters
        if p1_rounds > self.last_p1_rounds:
            self.last_p1_rounds = p1_rounds
            if p1_rounds >= 2:
                self.wins += 1
        if p2_rounds > self.last_p2_rounds:
            self.last_p2_rounds = p2_rounds
            if p2_rounds >= 2:
                self.losses += 1

        # Dynamic model switching based on opponent character and match index
        self.select_model(match_id, opponent_id)

        # 2. Decide button action
        buttons = np.zeros(12, dtype=np.int8)

        if not self.autoplay_active and manual_buttons is not None:
            # User manual override control
            buttons = manual_buttons
        else:
            # Autoplay control state machine
            if continue_timer > 0:
                # Continue Screen: Press START repeatedly (toggle on/off)
                if (self.frame_count // 10) % 2 == 0:
                    buttons[3] = 1  # Press START
            elif health == 0 and enemy_health == 0 and continue_timer == 0:
                # Character Select / Loading transitions
                buttons = self.handle_character_select(p1_char)
            else:
                # Active Fight Combat
                if self.active_model is not None:
                    stacked_obs = self.get_observation()
                    action_idx, _ = self.active_model.predict(stacked_obs, deterministic=True)
                    buttons = self.decode_action(int(action_idx))
                else:
                    # No model loaded fallback (no-op)
                    buttons = np.zeros(12, dtype=np.int8)

        # 3. Step the emulator
        raw_obs, reward, terminated, truncated, info = self.env.step(buttons)
        
        # Update frame stack for agent observations
        self.update_frame_stack(raw_obs)

        # Return game frame and state info
        state_info = {
            "health": health,
            "enemy_health": enemy_health,
            "continue_timer": continue_timer,
            "p1_char": p1_char,
            "p1_rounds": p1_rounds,
            "p2_rounds": p2_rounds,
            "match_id": match_id,
            "opponent_id": opponent_id,
            "non_fight_frames": self.non_fight_frames,
            "buttons": buttons
        }
        
        return raw_obs, state_info

    def reset_state(self):
        """Resets the emulator to the starting save state."""
        self.reset_frame_stack()
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        self.update_frame_stack(obs)
        self.frame_count = 0
        self.non_fight_frames = 0
        self.last_match_id = -1
        print("Emulator state reset successfully!")

    def close(self):
        self.env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Mortal Kombat Continuous Autoplay Bot")
    parser.add_argument("--state", default="Fight_LiuKang", help="Initial save state name")
    parser.add_argument("--model", default=None, help="Name or path of specific model to load (relative to models/ or absolute)")
    parser.add_argument("--model-dir", default=None, help="Path to models directory")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (useful for automated verification)")
    args = parser.parse_args()

    # Pygame Init
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT + HUD_HEIGHT))
    pygame.display.set_caption("Mortal Kombat Continuous Autoplay Bot")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)
    font_bold = pygame.font.SysFont("monospace", 18, bold=True)

    bot = AutoplayBot(state=args.state, model_dir_path=args.model_dir, model_name=args.model)
    
    running = True
    print("\n" + "=" * 60)
    print("MORTAL KOMBAT AUTOPLAY BOT LOADED")
    print("Controls:")
    print("  SPACE: Toggle Autoplay / Manual Override")
    print("  TAB:   Toggle Turbo mode")
    print("  R:     Reset emulator state")
    print("  Q/ESC: Quit")
    print("=" * 60 + "\n")

    frames_processed = 0
    while running:
        # Pygame keyboard events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    bot.autoplay_active = not bot.autoplay_active
                    print(f"Autoplay: {'ACTIVE' if bot.autoplay_active else 'MANUAL OVERRIDE'}")
                elif event.key == pygame.K_TAB:
                    bot.turbo = not bot.turbo
                    print(f"Turbo: {'ON' if bot.turbo else 'OFF'}")
                elif event.key == pygame.K_r:
                    bot.reset_state()

        # Capture manual inputs
        manual_buttons = np.zeros(12, dtype=np.int8)
        if not bot.autoplay_active:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:   manual_buttons[6] = 1 # _LEFT
            if keys[pygame.K_RIGHT]:  manual_buttons[7] = 1 # _RIGHT
            if keys[pygame.K_UP]:     manual_buttons[4] = 1 # _UP
            if keys[pygame.K_DOWN]:   manual_buttons[5] = 1 # _DOWN
            if keys[pygame.K_a]:      manual_buttons[10] = 1 # LP (_L)
            if keys[pygame.K_s]:      manual_buttons[1] = 1 # HP (_Y)
            if keys[pygame.K_z]:      manual_buttons[8] = 1 # LK (_A)
            if keys[pygame.K_x]:      manual_buttons[0] = 1 # HK (_B)
            if keys[pygame.K_c]:      manual_buttons[9] = 1 # Block (_X)

        # Step bot and get emulator frame
        # Repeat action for FRAME_SKIP times
        game_frame = None
        info = None
        
        # Calculate steps to run based on turbo
        steps_to_run = 1 if bot.turbo else FRAME_SKIP
        
        for _ in range(steps_to_run):
            game_frame, info = bot.step(manual_buttons if not bot.autoplay_active else None)

        if game_frame is not None:
            # Draw game screen
            surf = pygame.surfarray.make_surface(game_frame.swapaxes(0, 1))
            screen.blit(pygame.transform.scale(surf, (WIDTH, HEIGHT)), (0, 0))

        # Clear HUD area
        hud_rect = pygame.Rect(0, HEIGHT, WIDTH, HUD_HEIGHT)
        pygame.draw.rect(screen, (30, 30, 35), hud_rect)
        pygame.draw.line(screen, (80, 80, 90), (0, HEIGHT), (WIDTH, HEIGHT), 3)

        # Draw HUD stats
        p1_h = info["health"]
        p2_h = info["enemy_health"]
        match_id = info["match_id"]
        opp_id = info["opponent_id"]
        p1_rounds = info["p1_rounds"]
        p2_rounds = info["p2_rounds"]
        timer = info["continue_timer"]

        opp_name = OPPONENT_NAMES.get(opp_id, f"Unknown (ID {opp_id})")
        stage_name = MATCH_NAMES.get(match_id, f"Stage {match_id}")

        # Columns offsets
        col1_x = 20
        col2_x = WIDTH // 2 + 10

        # Mode Text
        if bot.autoplay_active:
            mode_text = font_bold.render("MODE: AUTOPLAY (ACTIVE)", True, (0, 255, 127))
        else:
            mode_text = font_bold.render("MODE: MANUAL OVERRIDE (SPACE TO RESUME)", True, (255, 140, 0))
        screen.blit(mode_text, (col1_x, HEIGHT + 15))

        # Column 1
        stage_text = font.render(f"Tournament Stage: {stage_name}", True, (200, 200, 210))
        opp_text = font.render(f"Current Opponent: {opp_name}", True, (200, 200, 210))
        
        health_color = (0, 255, 0) if p1_h > 50 else ((255, 165, 0) if p1_h > 20 else (255, 0, 0))
        p1_health_text = font.render(f"P1 Health: {p1_h}/161", True, health_color)
        
        opp_health_color = (0, 255, 0) if p2_h > 50 else ((255, 165, 0) if p2_h > 20 else (255, 0, 0))
        p2_health_text = font.render(f"P2 Health: {p2_h}/161", True, opp_health_color)

        screen.blit(stage_text, (col1_x, HEIGHT + 45))
        screen.blit(opp_text, (col1_x, HEIGHT + 70))
        screen.blit(p1_health_text, (col1_x, HEIGHT + 95))
        screen.blit(p2_health_text, (col1_x, HEIGHT + 120))

        # Column 2
        round_text = font.render(f"Rounds won this match: P1 {p1_rounds} - {p2_rounds} P2", True, (220, 220, 230))
        score_text = font.render(f"Total Matches Won: P1 {bot.wins} - {bot.losses} CPU", True, (220, 220, 230))
        
        turbo_color = (255, 215, 0) if bot.turbo else (150, 150, 160)
        turbo_text = font.render(f"Turbo speed: {'ENABLED (TAB to toggle)' if bot.turbo else 'DISABLED (TAB to toggle)'}", True, turbo_color)
        
        model_text = font.render(f"Active Model: {bot.active_model_name}", True, (135, 206, 250))

        screen.blit(round_text, (col2_x, HEIGHT + 45))
        screen.blit(score_text, (col2_x, HEIGHT + 70))
        screen.blit(turbo_text, (col2_x, HEIGHT + 95))
        screen.blit(model_text, (col2_x, HEIGHT + 120))

        # Draw Health Bars
        # P1 Health Bar
        pygame.draw.rect(screen, (60, 60, 65), (col1_x + 160, HEIGHT + 97, 100, 12))
        pygame.draw.rect(screen, health_color, (col1_x + 160, HEIGHT + 97, max(0, int(100 * p1_h / 161)), 12))
        # P2 Health Bar
        pygame.draw.rect(screen, (60, 60, 65), (col1_x + 160, HEIGHT + 122, 100, 12))
        pygame.draw.rect(screen, opp_health_color, (col1_x + 160, HEIGHT + 122, max(0, int(100 * p2_h / 161)), 12))

        pygame.display.flip()
        
        # Maintain framerate (60 fps under normal speed)
        # In Turbo, we want to step as fast as possible (tick(0))
        clock.tick(0 if bot.turbo else 30)

        frames_processed += 1
        if args.max_frames and frames_processed >= args.max_frames:
            print(f"Processed {frames_processed} frames. Exiting as requested by --max-frames.")
            running = False

    bot.close()
    pygame.quit()

if __name__ == "__main__":
    main()

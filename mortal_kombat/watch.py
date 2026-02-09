#!/usr/bin/env python3
"""Watch a trained fighting game agent play. Works for any game."""

import os
import sys
from pathlib import Path

# X11 for Wayland compat (must be before pygame import)
if not os.environ.get("SDL_VIDEODRIVER"):
    os.environ["SDL_VIDEODRIVER"] = "x11"

import numpy as np
import pygame
import torch

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

import stable_retro as retro
from stable_baselines3 import PPO

from fighters_common.fighting_env import (
    FIGHTING_ACTIONS,
    DirectRAMReader,
    DiscreteAction,
    FightingGameConfig,
    FightingEnv,
    FrameSkip,
    FrameStack,
    GrayscaleResize,
)
from fighters_common.game_configs import get_game_config

FRAME_SKIP = 4
FRAME_STACK = 4


# ─────────────────────────────────────────────────────────────────────────────
# Combo detection patterns for MK
# ─────────────────────────────────────────────────────────────────────────────
# Based on successful manual execution:
# Roll: RT (5f) → DN+RT (5f) → gap → LF (5f) = 28 damage
# Since DN+RT diagonal isn't in MK_FIGHTING_ACTIONS, we detect RT → LF transition
MK_COMBO_PATTERNS = {
    # Roll/Cannonball: RT followed by LF (simplified detection)
    "Roll/Cannonball": [(2, 1)],  # Action 2=RT, Action 1=LF
}


class ComboTracker:
    """Tracks action sequences to detect special moves and combos."""

    def __init__(self, patterns, window=10):
        self.patterns = patterns
        self.window = window
        self.action_history = []
        self.transition_history = []  # Track action changes, not every frame
        self.combo_counts = {name: 0 for name in patterns.keys()}
        self.last_combo = {}  # Track last frame for each combo type
        self.last_action = None
        self.cooldown = {
            "Roll/Cannonball": 15,  # Special moves need cooldown
            "Knife Throw": 15,
            "Jump→Roll": 20,
            "Uppercut": 10,
            "Crouch LK": 5,  # Pokes can be spammed
            "Blocking": 8,
        }

    def update(self, action, frame):
        """Update with new action and detect combos."""
        self.action_history.append(action)
        if len(self.action_history) > self.window:
            self.action_history.pop(0)

        # Track action transitions (when action changes)
        if action != self.last_action:
            self.transition_history.append(action)
            if len(self.transition_history) > self.window:
                self.transition_history.pop(0)
            self.last_action = action

        # Check for combo patterns (prioritize longer sequences)
        detected_combos = []
        for combo_name, pattern_list in self.patterns.items():
            # Check cooldown for this combo type
            last_frame = self.last_combo.get(combo_name, -999)
            cooldown = self.cooldown.get(combo_name, 10)
            if frame - last_frame < cooldown:
                continue

            for pattern in pattern_list:
                if self._matches_pattern(pattern):
                    detected_combos.append((combo_name, len(pattern)))

        # Return the longest matching combo
        if detected_combos:
            combo_name, _ = max(detected_combos, key=lambda x: x[1])
            self.combo_counts[combo_name] += 1
            self.last_combo[combo_name] = frame
            return combo_name
        return None

    def _matches_pattern(self, pattern):
        """Check if recent action transitions match a pattern."""
        if len(self.transition_history) < len(pattern):
            return False
        # Check if pattern appears at the end of history (most recent transitions)
        recent = self.transition_history[-len(pattern):]
        return all(recent[i] == pattern[i] for i in range(len(pattern)))

    def get_stats(self):
        """Get combo usage statistics."""
        total = sum(self.combo_counts.values())
        return total, self.combo_counts.copy()


def build_env(game_config, state):
    """Build the full wrapper stack for agent inference."""
    game_dir = ROOT_DIR / game_config.game_dir_name
    integrations = game_dir / "custom_integrations"
    retro.data.Integrations.add_custom_path(str(integrations))

    base_env = retro.make(
        game=game_config.game_id,
        state=state,
        render_mode="rgb_array",
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        use_restricted_actions=retro.Actions.ALL,
    )

    fight_config = FightingGameConfig(
        max_health=game_config.max_health,
        health_key=game_config.health_key,
        enemy_health_key=game_config.enemy_health_key,
        ram_overrides=game_config.ram_overrides,
        actions=game_config.actions,
    )

    action_map = game_config.actions or FIGHTING_ACTIONS

    env = base_env
    if game_config.ram_overrides:
        env = DirectRAMReader(env, game_config.ram_overrides)
    env = FrameSkip(env, n_skip=FRAME_SKIP)
    env = GrayscaleResize(env, width=84, height=84)
    env = FightingEnv(env, fight_config)
    env = DiscreteAction(env, action_map)
    env = FrameStack(env, n_frames=FRAME_STACK)
    return env, base_env


def watch(game_alias, state, model_path=None, show_kombos=False, char=None):
    config = get_game_config(game_alias)
    game_dir = ROOT_DIR / config.game_dir_name

    if model_path is None:
        model_dir = game_dir / "models"
        # Get all model files sorted by modification time (newest first)
        candidates = sorted(
            model_dir.glob(f"{game_alias}_*.zip"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if not candidates:
            print(f"No model found in {model_dir}")
            return
        model_path = str(candidates[0])  # Most recent

    title = f"{char} vs CPU" if char else config.display_name
    print(f"Game:  {config.display_name}")
    print(f"Model: {Path(model_path).name}")
    print(f"State: {state}")
    if show_kombos:
        print("Combo tracking: ENABLED")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPO.load(model_path, device=device)

    env, base_env = build_env(config, state)

    # Pygame setup
    pygame.init()
    SCALE = 3
    width, height = 256 * SCALE, 224 * SCALE
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 20)

    obs, info = env.reset()
    wins, losses, episodes = 0, 0, 0
    frame_count = 0
    turbo = False
    running = True

    # Initialize combo tracker if requested
    combo_tracker = None
    if show_kombos and game_alias in ["mk1", "mk2"]:
        combo_tracker = ComboTracker(MK_COMBO_PATTERNS, window=10)

    print("Keys: Q/ESC=quit  R=reset  TAB=turbo")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    frame_count = 0
                    print("Reset!")
                elif event.key == pygame.K_TAB:
                    turbo = not turbo
                    print(f"Turbo: {'ON' if turbo else 'OFF'}")

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        frame_count += 1

        # Track combos
        current_combo = None
        if combo_tracker is not None:
            current_combo = combo_tracker.update(int(action), frame_count)
            # Only print Roll/Cannonball (ignore Uppercut/Crouch LK spam)
            if current_combo and current_combo == "Roll/Cannonball":
                print(f"[Frame {frame_count}] ROLL DETECTED!")

            # Debug: show action transitions every 60 frames
            if frame_count % 60 == 0 and len(combo_tracker.transition_history) > 0:
                recent = combo_tracker.transition_history[-5:]
                print(f"[Frame {frame_count}] Recent actions: {recent}")

        # Skip rendering in turbo mode (except every 10th frame)
        if turbo and frame_count % 10 != 0:
            if terminated or truncated:
                episodes += 1
                rw = info.get("rounds_won", 0)
                rl = info.get("rounds_lost", 0)
                if rw >= 2 and rw > rl:
                    wins += 1
                else:
                    losses += 1
                wr = wins / max(1, episodes)
                result = "WIN" if rw >= 2 and rw > rl else "LOSS"
                print(f"Ep {episodes}: {result} ({rw}-{rl})  W:{wins} L:{losses} ({wr:.0%})")
                obs, info = env.reset()
                frame_count = 0
            continue

        # Render full-color frame from the base retro env
        frame = base_env.render()
        if frame is not None:
            surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            screen.blit(pygame.transform.scale(surf, (width, height)), (0, 0))

        # HUD
        p1 = info.get("health", "?")
        p2 = info.get("enemy_health", "?")
        timer = info.get("timer", "?")
        turbo_str = " [TURBO]" if turbo else ""
        line1 = f"P1:{p1}  P2:{p2}  T:{timer}  |  W:{wins} L:{losses}{turbo_str}"
        line2 = f"Frame:{frame_count}  Reward:{reward:+.3f}"

        # Add combo stats if tracking
        lines = [line1, line2]
        if combo_tracker is not None:
            total, counts = combo_tracker.get_stats()
            combo_str = " | ".join([f"{k}:{v}" for k, v in counts.items() if v > 0])
            if combo_str:
                line3 = f"Kombos: {combo_str}"
                lines.append(line3)
            if current_combo:
                line4 = f">>> {current_combo} <<<"
                lines.append(line4)

        # Render HUD lines
        y_offset = 10
        for i, line in enumerate(lines):
            shadow = font.render(line, True, (0, 0, 0))
            text = font.render(line, True, (0, 255, 0) if i < 2 else (255, 255, 0))
            screen.blit(shadow, (12, y_offset + 2))
            screen.blit(text, (10, y_offset))
            y_offset += 25

        pygame.display.flip()
        clock.tick(0 if turbo else 15)

        if terminated or truncated:
            episodes += 1
            rw = info.get("rounds_won", 0)
            rl = info.get("rounds_lost", 0)
            if rw >= 2 and rw > rl:
                wins += 1
            else:
                losses += 1
            wr = wins / max(1, episodes)
            result = "WIN" if rw >= 2 and rw > rl else "LOSS"

            # Print episode summary with combo stats
            if combo_tracker is not None:
                total, counts = combo_tracker.get_stats()
                combo_summary = ", ".join([f"{k}:{v}" for k, v in counts.items() if v > 0])
                print(f"Ep {episodes}: {result} ({rw}-{rl})  W:{wins} L:{losses} ({wr:.0%}) | Kombos: {combo_summary if combo_summary else 'None'}")
            else:
                print(f"Ep {episodes}: {result} ({rw}-{rl})  W:{wins} L:{losses} ({wr:.0%})")

            if not turbo:
                pygame.time.wait(2000)
            obs, info = env.reset()
            frame_count = 0

    env.close()
    pygame.quit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Watch trained fighting game agent")
    parser.add_argument("--game", default="mk1", help="Game alias (mk1, sf2, ssf2, mk2)")
    parser.add_argument("--char", default=None, help="Character name (auto-sets state to Fight_{char})")
    parser.add_argument("--state", default=None, help="Save state name")
    parser.add_argument("--model", default=None, help="Path to model .zip")
    parser.add_argument("--show-kombos", action="store_true", help="Track and display combo usage (MK only)")
    parser.add_argument("--practice", action="store_true", help="Use practice mode state (P2 idle)")
    args = parser.parse_args()

    default_states = {
        "mk1": "Fight_MortalKombat",
        "sf2": "Fight_StreetFighterIITurbo",
        "ssf2": "Fight_SuperStreetFighterII",
        "mk2": "Fight_MortalKombatII",
    }

    # Determine state: --state > --char > --practice > default
    if args.state:
        state = args.state
    elif args.char:
        state = f"Fight_{args.char}"
    elif args.practice:
        practice_states = {
            "mk1": "Practice_MortalKombat",
            "mk2": "Practice_MortalKombatII",
        }
        state = practice_states.get(args.game, default_states.get(args.game, "NONE"))
    else:
        state = default_states.get(args.game, "NONE")

    watch(args.game, state, args.model, show_kombos=args.show_kombos, char=args.char)

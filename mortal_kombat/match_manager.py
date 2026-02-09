#!/usr/bin/env python3
"""
Unified Match Manager for MK1 Training Pipeline.

Handles testing and state extraction for any match level (1-7+).
Single source of truth for match progression.

Usage:
    python match_manager.py test --level 2          # Test Match 2 win rate
    python match_manager.py extract --level 3       # Extract Match 3 states from Match 2 winners
    python match_manager.py validate --level 2      # Validate Match 2 states exist and are correct
    python match_manager.py list                    # List all available states
"""

import argparse
import gzip
import sys
from pathlib import Path
from PIL import Image

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from stable_baselines3 import PPO
from fighters_common.fighting_env import (
    DirectRAMReader, DiscreteAction, FightingGameConfig,
    FightingEnv, FrameSkip, FrameStack, GrayscaleResize,
)
from fighters_common.game_configs import get_game_config
import stable_retro as retro

# =============================================================================
# CONFIGURATION
# =============================================================================

GAME = "mk1"
FRAME_SKIP = 4
FRAME_STACK = 4

# All 7 playable characters
CHARACTERS = ["LiuKang", "Sonya", "JohnnyCage", "Kano", "Raiden", "SubZero", "Scorpion"]

# Best model to use (2M multichar trained)
BEST_MODEL = "mk1_multichar_ppo_final.zip"

# State naming convention
def get_state_name(character: str, match_level: int) -> str:
    """Get state name for a character at a match level."""
    if match_level == 1:
        return f"Fight_{character}"
    else:
        return f"Match{match_level}_{character}"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

def get_config():
    return get_game_config(GAME)

def setup_retro():
    config = get_config()
    game_dir = ROOT_DIR / config.game_dir_name
    integrations = game_dir / "custom_integrations"
    retro.data.Integrations.add_custom_path(str(integrations))
    return config, game_dir, integrations

def load_model(model_path=None):
    config, game_dir, _ = setup_retro()
    if model_path is None:
        model_path = game_dir / "models" / BEST_MODEL
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return PPO.load(str(model_path), device=device)

def build_env(state_name, render=False):
    """Build wrapped environment for a given state."""
    config, game_dir, integrations = setup_retro()

    base_env = retro.make(
        game=config.game_id,
        state=state_name,
        render_mode="rgb_array" if render else None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        use_restricted_actions=retro.Actions.ALL,
    )

    fight_config = FightingGameConfig(
        max_health=config.max_health,
        health_key=config.health_key,
        enemy_health_key=config.enemy_health_key,
        ram_overrides=config.ram_overrides,
        actions=config.actions,
    )

    env = base_env
    if config.ram_overrides:
        env = DirectRAMReader(env, config.ram_overrides)
    env = FrameSkip(env, n_skip=FRAME_SKIP)
    env = GrayscaleResize(env, width=84, height=84)
    env = FightingEnv(env, fight_config)
    env = DiscreteAction(env, config.actions)
    env = FrameStack(env, n_frames=FRAME_STACK)

    return env, base_env

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def play_match(model, state_name, max_frames=10000):
    """
    Play one match from a given state.
    Returns: (won: bool, rounds_won: int, rounds_lost: int, frames: int)
    """
    env, _ = build_env(state_name)
    obs, info = env.reset()

    frames = 0
    while frames < max_frames:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        frames += 1

        if terminated or truncated:
            break

    rounds_won = info.get("rounds_won", 0)
    rounds_lost = info.get("rounds_lost", 0)

    # True win = 2+ rounds AND more wins than losses (handles tiebreaker)
    won = rounds_won >= 2 and rounds_won > rounds_lost

    env.close()
    return won, rounds_won, rounds_lost, frames

def test_match_level(match_level: int, model=None, characters=None):
    """Test win rate at a specific match level."""
    if model is None:
        model = load_model()
    if characters is None:
        characters = CHARACTERS

    print(f"\nTesting Match {match_level}")
    print("=" * 50)

    results = []
    wins = 0

    for char in characters:
        state_name = get_state_name(char, match_level)

        # Check if state exists
        config, game_dir, _ = setup_retro()
        state_path = game_dir / "custom_integrations" / config.game_id / f"{state_name}.state"

        if not state_path.exists():
            print(f"  {char:<12} SKIP (no state)")
            results.append((char, None, 0, 0, 0))
            continue

        won, rw, rl, frames = play_match(model, state_name)
        status = "WIN" if won else "LOSS"
        if won:
            wins += 1

        print(f"  {char:<12} {status} ({rw}-{rl}, {frames} frames)")
        results.append((char, won, rw, rl, frames))

    tested = sum(1 for r in results if r[1] is not None)
    print("=" * 50)
    print(f"Match {match_level} Win Rate: {wins}/{tested} ({wins/tested*100:.0f}%)" if tested > 0 else "No states to test")

    return results

def extract_next_level_states(current_level: int, model=None, characters=None,
                               deterministic=True, attempts=1):
    """
    Win matches at current_level and extract states for current_level+1.
    """
    if model is None:
        model = load_model()
    if characters is None:
        characters = CHARACTERS

    next_level = current_level + 1
    config, game_dir, _ = setup_retro()
    state_dir = game_dir / "custom_integrations" / config.game_id
    screenshot_dir = game_dir / "screenshots" / f"match{next_level}_screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    mode = "deterministic" if deterministic else "stochastic"
    print(f"\nExtracting Match {next_level} states (from Match {current_level}, {mode}, {attempts} attempts)")
    print("=" * 50)

    successes = 0

    for char in characters:
        state_name = get_state_name(char, current_level)
        state_path = state_dir / f"{state_name}.state"
        next_state_path = state_dir / f"{get_state_name(char, next_level)}.state"

        if not state_path.exists():
            print(f"  {char:<12} SKIP (no Match {current_level} state)")
            continue

        if next_state_path.exists():
            print(f"  {char:<12} SKIP (Match {next_level} state already exists)")
            successes += 1
            continue

        extracted = False
        for attempt in range(attempts):
            attempt_label = f" (attempt {attempt+1}/{attempts})" if attempts > 1 else ""
            print(f"  {char:<12} Playing Match {current_level}{attempt_label}...", end=" ", flush=True)

            env, base_env = build_env(state_name, render=True)
            obs, info = env.reset()

            won = False
            saved = False
            frames = 0
            max_frames = 15000

            while frames < max_frames:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                frames += 1

                rounds_won = info.get("rounds_won", 0)
                rounds_lost = info.get("rounds_lost", 0)

                if rounds_won >= 2 and rounds_won > rounds_lost:
                    won = True
                    print(f"WON ({rounds_won}-{rounds_lost})")
                    print(f"              Waiting for Match {next_level}...", end=" ", flush=True)

                    for step in range(6000):
                        if step % 200 == 100:
                            buttons = [0] * 12
                            buttons[3] = 1  # START
                            for _ in range(4):
                                base_env.step(buttons)
                        else:
                            base_env.step([0] * 12)

                        health = base_env.data.lookup_value("health")
                        enemy_health = base_env.data.lookup_value("enemy_health")
                        timer = base_env.data.lookup_value("timer")

                        if health == config.max_health and enemy_health == config.max_health and timer > 50:
                            stable = 0
                            for _ in range(400):
                                base_env.step([0] * 12)
                                h = base_env.data.lookup_value("health")
                                e = base_env.data.lookup_value("enemy_health")
                                t = base_env.data.lookup_value("timer")
                                if h == config.max_health and e == config.max_health and t > 50:
                                    stable += 1
                                    if stable >= 120:
                                        next_state_name = get_state_name(char, next_level)
                                        save_path = state_dir / f"{next_state_name}.state"
                                        state_data = base_env.em.get_state()
                                        with gzip.open(save_path, "wb") as f:
                                            f.write(state_data)
                                        frame_img = base_env.render()
                                        img = Image.fromarray(frame_img)
                                        img.save(screenshot_dir / f"{next_state_name}.jpg", "JPEG", quality=95)
                                        print(f"SAVED {next_state_name}")
                                        successes += 1
                                        saved = True
                                        break
                                else:
                                    stable = 0
                            if saved:
                                break
                    else:
                        print("TIMEOUT (couldn't detect next match)")
                    if not saved and step < 5999:
                        print("FAILED (stability check never converged)")
                    break

                elif rounds_lost >= 2:
                    print(f"LOST ({rounds_won}-{rounds_lost})")
                    break

            if not won and frames >= max_frames:
                print("TIMEOUT")

            env.close()

            if won and saved:
                extracted = True
                break  # No more attempts needed

        if not extracted and attempts > 1:
            print(f"  {char:<12} FAILED all {attempts} attempts")

    print("=" * 50)
    print(f"Extracted: {successes}/{len(characters)} Match {next_level} states")
    return successes

def validate_level(match_level: int, save_screenshots: bool = True):
    """Validate that states exist, have correct health/timer, and save fresh screenshots."""
    config, game_dir, _ = setup_retro()
    state_dir = game_dir / "custom_integrations" / config.game_id

    screenshot_dir = None
    if save_screenshots:
        screenshot_dir = game_dir / "screenshots" / f"match{match_level}_screenshots"
        screenshot_dir.mkdir(exist_ok=True)

    print(f"\nValidating Match {match_level} states")
    print("=" * 50)

    valid = 0
    for char in CHARACTERS:
        state_name = get_state_name(char, match_level)
        state_path = state_dir / f"{state_name}.state"

        if not state_path.exists():
            print(f"  {char:<12} MISSING")
            continue

        # Load and check
        try:
            env, base_env = build_env(state_name, render=True)
            obs, info = env.reset()

            # Step a few frames with raw no-op
            for _ in range(5):
                base_env.step([0] * 12)
            obs, _, _, _, info = env.step(0)

            health = info.get("health", 0)
            enemy_health = info.get("enemy_health", 0)
            timer = base_env.data.lookup_value("timer")

            errors = []
            if health != config.max_health:
                errors.append(f"health={health}")
            if enemy_health != config.max_health:
                errors.append(f"enemy={enemy_health}")
            if timer < 50:
                errors.append(f"timer={timer}")

            if not errors:
                print(f"  {char:<12} OK (hp={health}/{enemy_health} timer={timer})")
                valid += 1
            else:
                print(f"  {char:<12} BAD ({', '.join(errors)})")

            # Save fresh screenshot
            if screenshot_dir:
                frame = base_env.render()
                if frame is not None:
                    img = Image.fromarray(frame)
                    img.save(screenshot_dir / f"{state_name}.jpg", "JPEG", quality=95)

            env.close()
        except Exception as e:
            print(f"  {char:<12} ERROR: {e}")

    print("=" * 50)
    print(f"Valid: {valid}/{len(CHARACTERS)}")
    if screenshot_dir:
        print(f"Screenshots: {screenshot_dir}")
    return valid

def list_states():
    """List all available states."""
    config, game_dir, _ = setup_retro()
    state_dir = game_dir / "custom_integrations" / config.game_id

    print("\nAvailable States")
    print("=" * 50)

    # Check each match level
    for level in range(1, 8):
        states = []
        for char in CHARACTERS:
            state_name = get_state_name(char, level)
            if (state_dir / f"{state_name}.state").exists():
                states.append(char)

        if states:
            print(f"Match {level}: {len(states)}/7 - {', '.join(states)}")

    print("=" * 50)

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MK1 Match Manager")
    parser.add_argument("command", choices=["test", "extract", "validate", "list"],
                       help="Command to run")
    parser.add_argument("--level", type=int, default=2,
                       help="Match level (default: 2)")
    parser.add_argument("--model", type=str, default=None,
                       help="Model path (default: best model)")
    parser.add_argument("--char", type=str, default=None,
                       help="Single character to test (default: all)")
    parser.add_argument("--stochastic", action="store_true",
                       help="Use stochastic predictions for extract")
    parser.add_argument("--attempts", type=int, default=1,
                       help="Number of extraction attempts per character")
    args = parser.parse_args()

    characters = [args.char] if args.char else None

    if args.command == "test":
        model = load_model(args.model)
        test_match_level(args.level, model, characters)

    elif args.command == "extract":
        model = load_model(args.model)
        extract_next_level_states(args.level, model, characters,
                                  deterministic=not args.stochastic,
                                  attempts=args.attempts)

    elif args.command == "validate":
        validate_level(args.level)

    elif args.command == "list":
        list_states()

    return 0

if __name__ == "__main__":
    sys.exit(main())

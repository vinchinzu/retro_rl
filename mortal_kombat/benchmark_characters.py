#!/usr/bin/env python3
"""
Benchmark all characters - run 10 matches each, show win rates.
Headless (no GUI).
"""

import os
import sys
from pathlib import Path

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

FRAME_SKIP = 4
FRAME_STACK = 4
CHARACTERS = ["LiuKang", "Sonya", "JohnnyCage", "Kano", "Raiden", "SubZero", "Scorpion"]
MATCHES_PER_CHAR = 10


def build_env(config, state):
    """Build environment for testing."""
    game_dir = ROOT_DIR / config.game_dir_name
    integrations = game_dir / "custom_integrations"
    retro.data.Integrations.add_custom_path(str(integrations))

    base_env = retro.make(
        game=config.game_id,
        state=state,
        render_mode=None,  # Headless
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
    return env


def test_character(model, config, char_name, num_matches, state_prefix="Fight", deterministic=True):
    """Run N matches for a character, return win/loss counts."""
    state_name = f"{state_prefix}_{char_name}"
    env = build_env(config, state_name)

    wins = 0
    losses = 0

    for match in range(num_matches):
        obs, info = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # True win = 2+ rounds AND more wins than losses (handles tiebreaker)
        rounds_won = info.get("rounds_won", 0)
        rounds_lost = info.get("rounds_lost", 0)
        if rounds_won >= 2 and rounds_won > rounds_lost:
            wins += 1
        else:
            losses += 1

        print(".", end="", flush=True)

    env.close()
    return wins, losses


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark MK1 model across all characters")
    parser.add_argument("--model", type=str, default=None, help="Path to model file")
    parser.add_argument("--matches", type=int, default=MATCHES_PER_CHAR, help="Matches per character")
    parser.add_argument("--level", type=int, default=1, help="Match level (1=Fight, 2=Match2, etc.)")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic (not deterministic) predictions")
    args = parser.parse_args()

    config = get_game_config("mk1")
    game_dir = ROOT_DIR / config.game_dir_name
    model_dir = game_dir / "models"
    state_dir = game_dir / "custom_integrations" / config.game_id

    # State prefix from level
    state_prefix = "Fight" if args.level == 1 else f"Match{args.level}"

    # Find model
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"Model not found: {args.model}")
            return 1
    else:
        # Find latest model (including multichar models)
        candidates = sorted(
            model_dir.glob("mk1*ppo*.zip"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if not candidates:
            print("No model found!")
            return 1
        model_path = candidates[0]

    matches_per_char = args.matches

    print("=" * 70)
    mode = "STOCHASTIC" if args.stochastic else "DETERMINISTIC"
    print(f"CHARACTER WIN RATE BENCHMARK - MATCH {args.level} ({mode})")
    print("=" * 70)
    print(f"\nModel: {model_path.name}")
    print(f"Matches per character: {matches_per_char}")
    print(f"State prefix: {state_prefix}_*")
    print("=" * 70 + "\n")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPO.load(str(model_path), device=device)

    results = []

    for char_name in CHARACTERS:
        # Check if state exists
        state_path = state_dir / f"{state_prefix}_{char_name}.state"
        if not state_path.exists():
            print(f"{char_name:<15} SKIP (no {state_prefix} state)")
            continue
        print(f"{char_name:<15} ", end="", flush=True)
        deterministic = not args.stochastic
        wins, losses = test_character(model, config, char_name, matches_per_char, state_prefix, deterministic=deterministic)
        win_rate = wins / matches_per_char * 100
        results.append((char_name, wins, losses, win_rate))
        print(f"  {wins}W/{losses}L ({win_rate:.0f}%)")

    # Print table
    print("\n" + "=" * 70)
    print(f"RESULTS - MATCH {args.level}")
    print("=" * 70)
    print(f"{'Character':<15} {'Wins':<8} {'Losses':<8} {'Win Rate':<10}")
    print("-" * 70)

    total_wins = 0
    total_losses = 0

    for char_name, wins, losses, win_rate in results:
        print(f"{char_name:<15} {wins:<8} {losses:<8} {win_rate:>6.0f}%")
        total_wins += wins
        total_losses += losses

    print("-" * 70)
    overall_rate = total_wins / (total_wins + total_losses) * 100
    print(f"{'OVERALL':<15} {total_wins:<8} {total_losses:<8} {overall_rate:>6.0f}%")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())

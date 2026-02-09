#!/usr/bin/env python3
"""
Speedrun Test - Play through the entire MK1 tournament and report results.

Loads a trained model and plays as LiuKang (or any character) through
the full tournament: M1-M7, Endurance 1-3, Goro, Shang Tsung.

Each match starts from its save state (not continuous play), so this tests
how well the model handles each stage independently.

Usage:
    python speedrun_test.py                            # Default: LiuKang, 5 attempts per stage
    python speedrun_test.py --char LiuKang --attempts 10  # More attempts
    python speedrun_test.py --model models/X.zip       # Specific model
    python speedrun_test.py --continuous                # Chain play (no save states)
"""

import os
import sys
import time
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from stable_baselines3 import PPO
from fighters_common.fighting_env import (
    DirectRAMReader, DiscreteAction, FightingEnv, FightingGameConfig,
    FrameSkip, FrameStack, GrayscaleResize
)
from fighters_common.game_configs import get_game_config
import stable_retro as retro

# Full tournament stages (SNES MK1: 2 endurance rounds, Goro = E2 opp2)
STAGES = [
    ("Fight",       "Match 1"),
    ("Match2",      "Match 2"),
    ("Match3",      "Match 3"),
    ("Match4",      "Match 4"),
    ("Match5",      "Match 5"),
    ("Match6",      "Match 6"),
    ("Match7",      "Mirror Match"),
    ("Endurance1",  "Endurance 1 (opp 1)"),
    ("Endurance1B", "Endurance 1 (opp 2)"),
    ("Endurance2",  "Endurance 2 (opp 1)"),
    ("Goro",        "Goro (=E2 opp 2)"),
    ("ShangTsung",  "Shang Tsung"),
]


def build_env(config, game_dir, state):
    """Build wrapped environment for model evaluation."""
    retro.data.Integrations.add_custom_path(str(game_dir / "custom_integrations"))

    base_env = retro.make(
        game=config.game_id,
        state=state,
        render_mode="rgb_array",
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
    env = FrameSkip(env, n_skip=4)
    env = GrayscaleResize(env, width=84, height=84)
    env = FightingEnv(env, fight_config)
    env = DiscreteAction(env, config.actions)
    env = FrameStack(env, n_frames=4)
    return env


def test_stage(model, config, game_dir, state_name, attempts=5):
    """Test a single stage, return (wins, losses, avg_frames)."""
    wins = 0
    losses = 0
    frame_counts = []

    for attempt in range(attempts):
        env = build_env(config, game_dir, state_name)
        obs, info = env.reset()
        frames = 0

        for frame in range(15000):  # Max ~4 minutes
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            frames += 1

            if terminated or truncated:
                rounds_won = info.get("rounds_won", 0)
                rounds_lost = info.get("rounds_lost", 0)
                if rounds_won >= 2 and rounds_won > rounds_lost:
                    wins += 1
                else:
                    losses += 1
                frame_counts.append(frames)
                break
        else:
            losses += 1
            frame_counts.append(frames)

        env.close()

    avg_frames = np.mean(frame_counts) if frame_counts else 0
    return wins, losses, avg_frames


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MK1 Tournament Speedrun Test")
    parser.add_argument("--char", default="LiuKang", help="Character to test")
    parser.add_argument("--attempts", type=int, default=5, help="Attempts per stage")
    parser.add_argument("--model", default=None, help="Model path")
    args = parser.parse_args()

    config = get_game_config("mk1")
    game_dir = ROOT_DIR / config.game_dir_name
    state_dir = game_dir / "custom_integrations" / config.game_id

    # Find model
    if args.model:
        model_path = Path(args.model)
    else:
        model_dir = game_dir / "models"
        for candidate in [
            "mk1_speedrun_ppo_final.zip",
            "mk1_fresh_ppo_final.zip",
            "mk1_match7_ppo_final.zip",
            "mk1_match4_ppo_final.zip",
            "mk1_multichar_ppo_2000000_steps.zip",
        ]:
            model_path = model_dir / candidate
            if model_path.exists():
                break

    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPO.load(str(model_path), device=device)
    print(f"Model: {model_path.name}")
    print(f"Character: {args.char}")
    print(f"Attempts per stage: {args.attempts}")
    print()

    # Test each stage
    results = []
    print(f"{'Stage':<25} {'Win%':>6} {'W':>4} {'L':>4} {'Avg Frames':>10}")
    print("-" * 55)

    for prefix, display_name in STAGES:
        state_name = f"{prefix}_{args.char}"
        if not (state_dir / f"{state_name}.state").exists():
            print(f"{display_name:<25} {'SKIP':>6}  (no state)")
            results.append((display_name, 0, 0, args.attempts, 0))
            continue

        wins, losses, avg_frames = test_stage(
            model, config, game_dir, state_name, args.attempts
        )
        win_rate = wins / max(1, wins + losses)
        results.append((display_name, win_rate, wins, losses, avg_frames))

        marker = " ***" if win_rate == 0 else ""
        print(f"{display_name:<25} {win_rate:>5.0%} {wins:>4} {losses:>4} {avg_frames:>10.0f}{marker}")

    # Summary
    print("-" * 55)
    total_wins = sum(r[2] for r in results)
    total_losses = sum(r[3] for r in results)
    overall_wr = total_wins / max(1, total_wins + total_losses)
    print(f"{'OVERALL':<25} {overall_wr:>5.0%} {total_wins:>4} {total_losses:>4}")

    # Bottleneck analysis
    print("\nBottleneck Analysis:")
    zero_stages = [r[0] for r in results if r[1] == 0 and r[2] + r[3] > 0]
    weak_stages = [r[0] for r in results if 0 < r[1] < 0.5]
    strong_stages = [r[0] for r in results if r[1] >= 0.5]

    if zero_stages:
        print(f"  Cannot win: {', '.join(zero_stages)}")
    if weak_stages:
        print(f"  Weak (<50%): {', '.join(weak_stages)}")
    if strong_stages:
        print(f"  Strong (50%+): {', '.join(strong_stages)}")

    # Simulated tournament success rate
    # Probability of winning ALL stages in sequence
    if all(r[1] > 0 for r in results if r[2] + r[3] > 0):
        chain_prob = 1.0
        for r in results:
            if r[2] + r[3] > 0:
                chain_prob *= r[1]
        print(f"\n  Estimated full-clear probability: {chain_prob:.1%}")
    else:
        print(f"\n  Full clear impossible until all stages have >0% win rate")


if __name__ == "__main__":
    main()

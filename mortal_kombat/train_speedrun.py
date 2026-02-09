#!/usr/bin/env python3
"""
Full Tournament Speedrun Training - LiuKang through the entire MK1 game.

Auto-discovers all LiuKang states (Fight through ShangTsung) and trains with
progressive weighting that emphasizes harder stages. Includes multi-character
general preservation to prevent overfitting.

Training mix:
  85% LiuKang states (weighted by difficulty tier)
  15% Other characters M1 states (general skill preservation)

Difficulty tiers (within LiuKang states):
  10% M1-M3    (early matches, already good at these)
  15% M4-M6    (medium difficulty)
  10% M7       (mirror match)
  25% Endurance (new content, need to learn)
  25% Goro     (sub-boss, unique fighting style)
  15% ShangTsung (final boss, shapeshifter)

Usage:
    python train_speedrun.py                        # Default: 8M steps
    python train_speedrun.py --steps 12000000       # Longer run
    python train_speedrun.py --lr 5e-5              # Custom learning rate
    python train_speedrun.py --fresh                # Start from scratch
"""

import sys
import random
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from stable_baselines3.common.callbacks import BaseCallback
from fighters_common import train_ppo
from fighters_common.fighting_env import make_fighting_env
from fighters_common.game_configs import get_game_config

CHARACTERS = ["LiuKang", "Sonya", "JohnnyCage", "Kano", "Raiden", "SubZero", "Scorpion"]

# Difficulty tiers: (state_prefixes, weight)
# SNES MK1 has 2 endurance rounds (not 3). Goro = Endurance2B (alias).
# Total: 12 fights across 12 unique states.
LIUKANG_TIERS = [
    (["Fight", "Match2", "Match3"],                           0.10, "Easy (M1-M3)"),
    (["Match4", "Match5", "Match6"],                          0.15, "Medium (M4-M6)"),
    (["Match7"],                                              0.10, "Mirror (M7)"),
    (["Endurance1", "Endurance1B", "Endurance2"],             0.20, "Endurance"),
    (["Goro"],                                                0.25, "Goro (sub-boss)"),
    (["ShangTsung"],                                          0.20, "Shang Tsung (final)"),
]

GENERAL_WEIGHT = 0.15  # Weight for non-LiuKang general preservation


def discover_states():
    """Auto-discover all available states."""
    config = get_game_config("mk1")
    game_dir = ROOT_DIR / config.game_dir_name
    state_dir = game_dir / "custom_integrations" / config.game_id

    # LiuKang states by tier
    liukang_tiers = []
    for prefixes, weight, name in LIUKANG_TIERS:
        states = []
        for prefix in prefixes:
            state_name = f"{prefix}_LiuKang"
            if (state_dir / f"{state_name}.state").exists():
                states.append(state_name)
        if states:
            liukang_tiers.append((states, weight, name))

    # General preservation states (all chars, M1 only)
    general_states = []
    for char in CHARACTERS:
        state_name = f"Fight_{char}"
        if (state_dir / f"{state_name}.state").exists():
            general_states.append(state_name)

    return liukang_tiers, general_states


class SpeedrunMetrics(BaseCallback):
    """Track fight metrics with per-tier reporting."""

    def __init__(self, report_interval=500, verbose=1):
        super().__init__(verbose)
        self.report_interval = report_interval
        self.wins = 0
        self.losses = 0
        self.episodes = 0
        self.episode_rewards = []
        self.milestones = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue

            self.episodes += 1
            ep_reward = info["episode"]["r"]
            self.episode_rewards.append(ep_reward)

            rounds_won = info.get("rounds_won", 0)
            rounds_lost = info.get("rounds_lost", 0)
            won = rounds_won >= 2 and rounds_won > rounds_lost

            if won:
                self.wins += 1
            else:
                self.losses += 1

            if self.episodes % 10 == 0:
                recent = self.episode_rewards[-100:]
                wr = self.wins / max(1, self.wins + self.losses)
                print(
                    f"  Ep {self.episodes}: "
                    f"avg_reward={np.mean(recent):.2f} "
                    f"win_rate={wr:.1%} "
                    f"({self.wins}W/{self.losses}L)"
                )

            if self.episodes % self.report_interval == 0:
                self._progress_report()

        return True

    def _progress_report(self):
        total = self.wins + self.losses
        wr = self.wins / max(1, total)
        recent_rewards = self.episode_rewards[-200:]
        avg_r = np.mean(recent_rewards) if recent_rewards else 0

        self.milestones.append({
            "step": self.num_timesteps,
            "episodes": self.episodes,
            "win_rate": wr,
            "avg_reward": avg_r,
        })

        print(f"\n{'='*70}")
        print(f"SPEEDRUN PROGRESS @ Ep {self.episodes} | Step {self.num_timesteps:,}")
        print(f"{'='*70}")
        print(f"  Win rate:   {wr:.1%} ({self.wins}W/{self.losses}L)")
        print(f"  Avg reward: {avg_r:.3f} (last 200 episodes)")

        if len(self.milestones) >= 3:
            recent = self.milestones[-3:]
            wr_trend = recent[-1]["win_rate"] - recent[0]["win_rate"]
            print(f"  Trend:      {wr_trend:+.1%} over last 3 reports")

            if wr_trend < -0.03:
                print(f"  >>> WARNING: WIN RATE DECLINING - possible overtraining")
            elif len(self.milestones) >= 6:
                long = self.milestones[-6:]
                wr_range = max(m["win_rate"] for m in long) - min(m["win_rate"] for m in long)
                if wr_range < 0.02:
                    print(f"  >>> PLATEAU DETECTED at ~{np.mean([m['win_rate'] for m in long]):.1%}")

        if len(self.milestones) >= 2:
            best = max(self.milestones, key=lambda m: m["win_rate"])
            print(f"  Best so far: {best['win_rate']:.1%} at step {best['step']:,}")

        print(f"{'='*70}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Full tournament speedrun training")
    parser.add_argument("--steps", type=int, default=8_000_000, help="Total steps (default 8M)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint-freq", type=int, default=500_000, help="Checkpoint every N steps")
    parser.add_argument("--report-freq", type=int, default=500, help="Report every N episodes")
    parser.add_argument("--fresh", action="store_true", help="Start from scratch")
    cli_args = parser.parse_args()

    liukang_tiers, general_states = discover_states()

    total_liukang = sum(len(s) for s, _, _ in liukang_tiers)
    if total_liukang == 0:
        print("ERROR: No LiuKang states found!")
        sys.exit(1)

    LEARNING_RATE = cli_args.lr
    TOTAL_STEPS = cli_args.steps
    estimated_hours = TOTAL_STEPS / (440 * 3600)

    print("=" * 70)
    print("MK1 SPEEDRUN TRAINING - LiuKang Full Tournament")
    print(f"LiuKang states: {total_liukang} | General states: {len(general_states)}")
    print(f"Steps: {TOTAL_STEPS:,} | LR: {LEARNING_RATE} | Checkpoints: {cli_args.checkpoint_freq:,}")
    print(f"Estimated time: {estimated_hours:.1f} hours (at ~440 fps)")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Normalize tier weights (in case some tiers are empty)
    active_tiers = [(s, w, n) for s, w, n in liukang_tiers if s]
    total_tier_weight = sum(w for _, w, _ in active_tiers)
    active_tiers = [(s, w / total_tier_weight, n) for s, w, n in active_tiers]

    print(f"\nLiuKang tiers ({int((1-GENERAL_WEIGHT)*100)}% of training):")
    for states, weight, name in active_tiers:
        actual_weight = weight * (1 - GENERAL_WEIGHT)
        print(f"  {name}: {int(actual_weight*100)}% ({len(states)} states: {', '.join(states)})")
    print(f"\nGeneral preservation ({int(GENERAL_WEIGHT*100)}%): {len(general_states)} M1 states")
    print("=" * 70)

    # Find best model
    config = get_game_config("mk1")
    game_dir = ROOT_DIR / config.game_dir_name
    model_dir = game_dir / "models"

    model_path = None
    if not cli_args.fresh:
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
        else:
            candidates = sorted(model_dir.glob("mk1*ppo*.zip"),
                              key=lambda p: p.stat().st_mtime, reverse=True)
            model_path = candidates[0] if candidates else None

        if not model_path or not model_path.exists():
            print("ERROR: No base model found! Use --fresh to start from scratch.")
            sys.exit(1)

        print(f"\nBase model: {model_path.name}")
    else:
        print("\nStarting FRESH (no pretrained model)")

    print("=" * 70 + "\n")

    # Monkey-patch env creation for weighted random state selection
    _original_make_env = make_fighting_env

    def speedrun_make_env(*args, **kwargs):
        roll = random.random()

        if roll < GENERAL_WEIGHT:
            # General preservation
            state = random.choice(general_states)
        else:
            # LiuKang tier selection
            tier_roll = random.random()
            cumulative = 0
            chosen_states = active_tiers[0][0]  # fallback
            for states, weight, _ in active_tiers:
                cumulative += weight
                if tier_roll < cumulative:
                    chosen_states = states
                    break
            state = random.choice(chosen_states)

        kwargs['state'] = state
        if random.random() < 0.002:
            print(f"  [Env] {state}")
        return _original_make_env(*args, **kwargs)

    train_ppo.make_fighting_env = speedrun_make_env

    # Replace metrics callback
    train_ppo.FightMetricsCallback = lambda verbose=0: SpeedrunMetrics(
        report_interval=cli_args.report_freq, verbose=verbose
    )

    # Override configs
    train_ppo.TrainConfig.LEARNING_RATE = LEARNING_RATE
    train_ppo.TrainConfig.CHECKPOINT_FREQ = cli_args.checkpoint_freq

    MODEL_PREFIX = "mk1_speedrun_ppo"

    sys.argv = [sys.argv[0]]
    sys.argv.extend(['--game', 'mk1'])
    sys.argv.extend(['--steps', str(TOTAL_STEPS)])
    if model_path:
        sys.argv.extend(['--load', str(model_path)])
    sys.argv.extend(['--prefix', MODEL_PREFIX])

    start_time = time.time()
    train_ppo.main()
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"Duration: {elapsed/3600:.1f} hours")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Register
    try:
        from model_registry import Registry
        reg = Registry()
        final_name = f"{MODEL_PREFIX}_final.zip"
        tier_str = ", ".join(f"{int(w*(1-GENERAL_WEIGHT)*100)}% {n}"
                           for _, w, n in active_tiers)
        reg.register(
            final_name,
            parent=model_path.name if model_path else "fresh",
            script="train_speedrun.py",
            steps=TOTAL_STEPS,
            lr=LEARNING_RATE,
            training_mix=f"85% LiuKang ({tier_str}), 15% general M1",
            notes=f"Full tournament speedrun, {total_liukang} LiuKang states, "
                  f"{elapsed/3600:.1f}h, {TOTAL_STEPS:,} steps",
        )
        print(f"\nRegistered {final_name} in model registry.")
    except Exception as e:
        print(f"\nWarning: Could not register model: {e}")


if __name__ == '__main__':
    main()

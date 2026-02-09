#!/usr/bin/env python3
"""
PPO training pipeline for fighting games.

Unified trainer that works with any of the supported fighting games.
Uses stable-baselines3 PPO with CNN policy, frame stacking, and
health-delta reward shaping.

Usage:
    # Train SF2 Turbo (from a fight-ready save state)
    python train_ppo.py --game sf2 --state Fight_Ryu_vs_CPU --steps 500000

    # Train Mortal Kombat II
    python train_ppo.py --game mk2 --state Fight_LiuKang_vs_CPU --steps 500000

    # Resume training from checkpoint
    python train_ppo.py --game sf2 --state Fight_Ryu_vs_CPU --load models/sf2_ppo_best.zip

    # Create a fight state from title screen, then train
    python train_ppo.py --game sf2 --create-state --steps 500000

    # Evaluate a trained model (renders to screen)
    python train_ppo.py --game sf2 --state Fight_Ryu_vs_CPU --eval --load models/sf2_final.zip
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from fighters_common.fighting_env import FightingGameConfig, make_fighting_env
from fighters_common.game_configs import GAME_REGISTRY, get_game_config
from fighters_common.menu_nav import create_fight_state


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
class TrainConfig:
    TOTAL_STEPS = 500_000
    LEARNING_RATE = 3e-4
    ENT_COEF_START = 0.01
    ENT_COEF_END = 0.001
    BATCH_SIZE = 256
    N_STEPS = 2048          # Steps per rollout
    N_EPOCHS = 4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    N_ENVS = 4              # Parallel environments
    FRAME_SKIP = 4
    FRAME_STACK = 4
    FEATURES_DIM = 512
    CHECKPOINT_FREQ = 100_000


# ─────────────────────────────────────────────────────────────────────────────
# CNN Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────
class FighterCNN(BaseFeaturesExtractor):
    """CNN for 84x84 grayscale frame stacks."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]  # frame_stack channels

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────
class EntropySchedule(BaseCallback):
    """Linearly decay entropy coefficient during training."""

    def __init__(self, start: float, end: float, total_steps: int, verbose=0):
        super().__init__(verbose)
        self.start = start
        self.end = end
        self.total_steps = total_steps

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.total_steps)
        self.model.ent_coef = self.start + (self.end - self.start) * progress
        if self.num_timesteps % 10000 == 0 and self.verbose:
            print(f"  Step {self.num_timesteps}: ent_coef={self.model.ent_coef:.5f}")
        return True


class FightMetricsCallback(BaseCallback):
    """Log fighting-game specific metrics."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.wins = 0
        self.losses = 0
        self.episodes = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episodes += 1
                ep_reward = info["episode"]["r"]
                self.episode_rewards.append(ep_reward)
                rounds_won = info.get("rounds_won", 0)
                rounds_lost = info.get("rounds_lost", 0)
                if rounds_won >= 2 and rounds_won > rounds_lost:
                    self.wins += 1
                else:
                    self.losses += 1

                if self.episodes % 10 == 0:
                    recent = self.episode_rewards[-100:]
                    win_rate = self.wins / max(1, self.wins + self.losses)
                    print(
                        f"  Ep {self.episodes}: "
                        f"avg_reward={np.mean(recent):.2f} "
                        f"win_rate={win_rate:.1%} "
                        f"({self.wins}W/{self.losses}L)"
                    )
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory for vectorized envs
# ─────────────────────────────────────────────────────────────────────────────
def _make_env_fn(game_id: str, state: str, game_dir: Path, config, monitor_dir: str, rank: int, practice: bool = False):
    """Factory function for SubprocVecEnv."""
    def _init():
        env = make_fighting_env(
            game=game_id,
            state=state,
            game_dir=game_dir,
            config=config,
            frame_skip=TrainConfig.FRAME_SKIP,
            frame_stack=TrainConfig.FRAME_STACK,
            monitor_dir=monitor_dir,
            practice=practice,
        )
        return env
    return _init


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game_config = get_game_config(args.game)
    game_dir = ROOT_DIR / game_config.game_dir_name

    print(f"Game: {game_config.display_name}")
    print(f"Game ID: {game_config.game_id}")
    print(f"Device: {device}")
    practice = getattr(args, "practice", False)
    print(f"State: {args.state}")
    print(f"Steps: {args.steps}")
    print(f"Envs: {args.n_envs}")
    if practice:
        print("Mode: PRACTICE (2P null bot)")

    # Dirs
    model_dir = game_dir / "models"
    log_dir = game_dir / "logs"
    monitor_dir = str(game_dir / "monitor")
    model_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    # Fighting env config
    env_config = FightingGameConfig(
        max_health=game_config.max_health,
        health_key=game_config.health_key,
        enemy_health_key=game_config.enemy_health_key,
        timer_key=game_config.timer_key,
        round_length_frames=game_config.round_length_frames,
        ram_overrides=game_config.ram_overrides,
        actions=game_config.actions,
    )

    # Create vectorized envs
    env_fns = [
        _make_env_fn(game_config.game_id, args.state, game_dir, env_config, monitor_dir, i, practice=practice)
        for i in range(args.n_envs)
    ]

    if args.n_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    # Policy
    policy_kwargs = dict(
        features_extractor_class=FighterCNN,
        features_extractor_kwargs=dict(features_dim=TrainConfig.FEATURES_DIM),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )

    if args.load and os.path.exists(args.load):
        print(f"Loading model from {args.load}")
        model = PPO.load(
            args.load,
            env=env,
            device=device,
            custom_objects={
                "learning_rate": TrainConfig.LEARNING_RATE,
                "clip_range": TrainConfig.CLIP_RANGE,
            },
        )
    else:
        print("Starting fresh PPO training")
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            device=device,
            policy_kwargs=policy_kwargs,
            learning_rate=TrainConfig.LEARNING_RATE,
            ent_coef=TrainConfig.ENT_COEF_START,
            n_steps=TrainConfig.N_STEPS,
            batch_size=TrainConfig.BATCH_SIZE,
            n_epochs=TrainConfig.N_EPOCHS,
            clip_range=TrainConfig.CLIP_RANGE,
            gae_lambda=TrainConfig.GAE_LAMBDA,
            gamma=TrainConfig.GAMMA,
            tensorboard_log=str(log_dir),
        )

    # Model naming
    model_prefix = args.prefix or f"{args.game}_ppo"

    # Callbacks
    callbacks = [
        EntropySchedule(TrainConfig.ENT_COEF_START, TrainConfig.ENT_COEF_END, args.steps, verbose=1),
        FightMetricsCallback(verbose=1),
        CheckpointCallback(
            save_freq=max(TrainConfig.CHECKPOINT_FREQ // args.n_envs, 1),
            save_path=str(model_dir),
            name_prefix=model_prefix,
        ),
    ]

    print(f"\nTraining for {args.steps} steps...")
    model.learn(total_timesteps=args.steps, callback=callbacks)

    final_path = str(model_dir / f"{model_prefix}_final.zip")
    model.save(final_path)
    print(f"\nTraining complete. Model saved to {final_path}")

    env.close()


def evaluate(args):
    """Run a trained model with rendering for visual evaluation."""
    game_config = get_game_config(args.game)
    game_dir = ROOT_DIR / game_config.game_dir_name

    if not args.load:
        # Try to find latest model
        model_dir = game_dir / "models"
        candidates = sorted(model_dir.glob(f"{args.game}_ppo*.zip"))
        if not candidates:
            print("No trained model found. Train first with --steps")
            return
        args.load = str(candidates[-1])

    print(f"Evaluating {args.load}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_config = FightingGameConfig(
        max_health=game_config.max_health,
        health_key=game_config.health_key,
        enemy_health_key=game_config.enemy_health_key,
        ram_overrides=game_config.ram_overrides,
        actions=game_config.actions,
    )

    env = make_fighting_env(
        game=game_config.game_id,
        state=args.state,
        game_dir=game_dir,
        config=env_config,
        render_mode="rgb_array",
        frame_skip=TrainConfig.FRAME_SKIP,
        frame_stack=TrainConfig.FRAME_STACK,
    )

    model = PPO.load(args.load, device=device)

    # Render with pygame
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((84 * 6, 84 * 6))
    pygame.display.set_caption(f"Eval: {game_config.display_name}")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    wins = 0
    losses = 0
    episodes = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # Render the last grayscale frame scaled up
        frame = obs[-1] if obs.ndim == 3 else obs
        rgb = np.stack([frame, frame, frame], axis=-1) if frame.ndim == 2 else frame
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        screen.blit(pygame.transform.scale(surf, screen.get_size()), (0, 0))

        font = pygame.font.SysFont("monospace", 18)
        hud = font.render(
            f"W:{info.get('rounds_won', 0)} L:{info.get('rounds_lost', 0)} | "
            f"Dmg:{info.get('episode_damage_dealt', 0)} Taken:{info.get('episode_damage_taken', 0)}",
            True, (0, 255, 0),
        )
        screen.blit(hud, (10, 10))
        pygame.display.flip()
        clock.tick(15)  # Slow for visibility (frame skip already 4x)

        if terminated or truncated:
            episodes += 1
            rw = info.get("rounds_won", 0)
            rl = info.get("rounds_lost", 0)
            if rw >= 2 and rw > rl:
                wins += 1
            else:
                losses += 1
            print(f"Episode {episodes}: W={wins} L={losses} (rate={wins/max(1,episodes):.0%})")
            obs, info = env.reset()

    env.close()
    pygame.quit()


def do_create_state(args):
    """Create a fight-ready save state by navigating menus."""
    game_config = get_game_config(args.game)
    game_dir = ROOT_DIR / game_config.game_dir_name
    state_name = args.state or f"Fight_{game_config.game_id.split('-')[0]}"
    create_fight_state(
        game=game_config.game_id,
        game_dir=game_dir,
        state_name=state_name,
        menu_sequence=game_config.menu_sequence,
        settle_frames=game_config.menu_settle_frames,
    )


def main():
    parser = argparse.ArgumentParser(description="Fighting Game PPO Trainer")
    parser.add_argument("--game", required=True, help="Game ID or alias (sf2, ssf2, mk1, mk2)")
    parser.add_argument("--state", default="NONE", help="Save state to load")
    parser.add_argument("--steps", type=int, default=TrainConfig.TOTAL_STEPS, help="Training steps")
    parser.add_argument("--n-envs", type=int, default=TrainConfig.N_ENVS, help="Parallel envs")
    parser.add_argument("--load", type=str, default=None, help="Model checkpoint to load")
    parser.add_argument("--prefix", type=str, default=None, help="Model name prefix (default: {game}_ppo)")
    parser.add_argument("--eval", action="store_true", help="Evaluate mode (render)")
    parser.add_argument("--create-state", action="store_true", help="Create fight state from menus")
    parser.add_argument("--practice", action="store_true", help="Practice mode: 2P with idle P2 (null bot)")

    args = parser.parse_args()

    if args.create_state:
        do_create_state(args)
    elif args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()

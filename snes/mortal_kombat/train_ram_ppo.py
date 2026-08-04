#!/usr/bin/env python3
"""
RAM-vector PPO training for Mortal Kombat (MLP policy).

Uses make_ram_fighting_env() instead of CNN frame stacks. Same 32 discrete
actions and FightingEnv reward shaping as pixel PPO.

Usage:
    cd mortal_kombat
    uv run python train_ram_ppo.py --state Fight_LiuKang --steps 1000000
    uv run python train_ram_ppo.py --state Fight_LiuKang --eval --load models/mk1_ram_ppo_final.zip
"""

from __future__ import annotations

import argparse
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from retro_harness.fighters.fighting_env import FightingGameConfig
from retro_harness.fighters.game_configs import get_game_config
from retro_harness.fighters.ram_observation import (
    MK1_RAM_FEATURES,
    MK1_RAM_FEATURES_V1,
    make_ram_fighting_env,
)
from retro_harness.fighters.train_ppo import (
    EntropySchedule,
    FightMetricsCallback,
    TrainConfig,
)


class RamTrainConfig:
    """Hyperparameters tuned for small RAM observations."""

    TOTAL_STEPS = 1_000_000
    LEARNING_RATE = 3e-4
    ENT_COEF_START = 0.01
    ENT_COEF_END = 0.0002
    BATCH_SIZE = 256
    N_STEPS = 2048
    N_EPOCHS = 4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    N_ENVS = 8
    FRAME_SKIP = 4
    CHECKPOINT_FREQ = 100_000
    NET_ARCH = dict(pi=[256, 128], vf=[256, 128])


def _make_env_fn(
    game_id: str,
    state: str,
    game_dir: Path,
    config: FightingGameConfig,
    monitor_dir: str,
    features: tuple,
):
    def _init():
        return make_ram_fighting_env(
            game=game_id,
            state=state,
            game_dir=game_dir,
            config=config,
            frame_skip=RamTrainConfig.FRAME_SKIP,
            monitor_dir=monitor_dir,
            features=features,
        )

    return _init


def _resolve_features(name: str):
    if name == "v1":
        return MK1_RAM_FEATURES_V1
    return MK1_RAM_FEATURES


def train(args: argparse.Namespace) -> None:
    # MLP policies train faster on CPU (SB3 recommendation for non-CNN).
    device = torch.device("cpu")
    game_config = get_game_config(args.game)
    game_dir = ROOT_DIR / game_config.game_dir_name

    print(f"Game: {game_config.display_name}")
    features = _resolve_features(args.features)
    print(f"Policy: MlpPolicy (RAM obs, {len(features)} dims, {args.features})")
    print(f"Device: {device}")
    print(f"State: {args.state}")
    print(f"Steps: {args.steps}")
    print(f"Envs: {args.n_envs}")

    model_dir = game_dir / "models"
    monitor_dir = str(game_dir / "monitor")
    model_dir.mkdir(exist_ok=True)

    env_config = FightingGameConfig(
        max_health=game_config.max_health,
        health_key=game_config.health_key,
        enemy_health_key=game_config.enemy_health_key,
        timer_key=game_config.timer_key,
        round_length_frames=game_config.round_length_frames,
        ram_overrides=game_config.ram_overrides,
        actions=game_config.actions,
    )

    env_fns = [
        _make_env_fn(
            game_config.game_id,
            args.state,
            game_dir,
            env_config,
            monitor_dir,
            features,
        )
        for _ in range(args.n_envs)
    ]
    env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)

    policy_kwargs = dict(net_arch=RamTrainConfig.NET_ARCH)

    if args.load and os.path.exists(args.load):
        print(f"Loading model from {args.load}")
        model = PPO.load(
            args.load,
            env=env,
            device=device,
            custom_objects={
                "learning_rate": RamTrainConfig.LEARNING_RATE,
                "clip_range": RamTrainConfig.CLIP_RANGE,
            },
        )
    else:
        print("Starting fresh RAM PPO training")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,
            policy_kwargs=policy_kwargs,
            learning_rate=RamTrainConfig.LEARNING_RATE,
            ent_coef=RamTrainConfig.ENT_COEF_START,
            n_steps=RamTrainConfig.N_STEPS,
            batch_size=RamTrainConfig.BATCH_SIZE,
            n_epochs=RamTrainConfig.N_EPOCHS,
            clip_range=RamTrainConfig.CLIP_RANGE,
            gae_lambda=RamTrainConfig.GAE_LAMBDA,
            gamma=RamTrainConfig.GAMMA,
            tensorboard_log=None,
        )

    model_prefix = args.prefix or "mk1_ram_v2_ppo"
    callbacks = [
        EntropySchedule(
            RamTrainConfig.ENT_COEF_START,
            RamTrainConfig.ENT_COEF_END,
            args.steps,
            verbose=1,
        ),
        FightMetricsCallback(verbose=1),
        CheckpointCallback(
            save_freq=max(RamTrainConfig.CHECKPOINT_FREQ // args.n_envs, 1),
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


def evaluate(args: argparse.Namespace) -> None:
    """Headless eval loop for a RAM-trained model."""
    game_config = get_game_config(args.game)
    game_dir = ROOT_DIR / game_config.game_dir_name

    if not args.load:
        model_dir = game_dir / "models"
        candidates = sorted(model_dir.glob("mk1_ram_ppo*.zip"))
        if not candidates:
            print("No RAM model found. Train first with train_ram_ppo.py")
            return
        args.load = str(candidates[-1])

    env_config = FightingGameConfig(
        max_health=game_config.max_health,
        health_key=game_config.health_key,
        enemy_health_key=game_config.enemy_health_key,
        timer_key=game_config.timer_key,
        round_length_frames=game_config.round_length_frames,
        ram_overrides=game_config.ram_overrides,
        actions=game_config.actions,
    )

    env = make_ram_fighting_env(
        game=game_config.game_id,
        state=args.state,
        game_dir=game_dir,
        config=env_config,
        frame_skip=RamTrainConfig.FRAME_SKIP,
        features=_resolve_features(args.features),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPO.load(args.load, device=device)

    wins = 0
    losses = 0
    episodes = 0

    for _ in range(args.episodes):
        obs, info = env.reset()
        for _ in range(15000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                episodes += 1
                rw = info.get("rounds_won", 0)
                rl = info.get("rounds_lost", 0)
                if rw >= 2 and rw > rl:
                    wins += 1
                else:
                    losses += 1
                break

    env.close()
    rate = wins / max(1, wins + losses)
    print(f"Eval: {wins}W / {losses}L over {episodes} episodes ({rate:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="MK1 RAM-vector PPO trainer")
    parser.add_argument("--game", default="mk1", help="Game alias (default: mk1)")
    parser.add_argument(
        "--state",
        default="Fight_LiuKang",
        help="Save state (default: Fight_LiuKang for E017)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=RamTrainConfig.TOTAL_STEPS,
        help="Training steps",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=RamTrainConfig.N_ENVS,
        help="Parallel envs",
    )
    parser.add_argument("--load", type=str, default=None, help="Checkpoint to resume")
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Model name prefix (default: mk1_ram_ppo)",
    )
    parser.add_argument(
        "--features",
        choices=["v1", "v2"],
        default="v2",
        help="RAM feature set (default: v2 with spacing)",
    )
    parser.add_argument("--eval", action="store_true", help="Evaluate a trained model")
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Eval episodes (eval mode only)",
    )
    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Metroid HRL - Generic Training Script
Supports training on various states (BossTorizo, LandingSite, etc.)

Usage:
    python train.py --state LandingSite --scenario nav --steps 100000
    python train.py --state BossTorizo --scenario boss --load models/boss_best.zip
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from functools import partial
import gymnasium as gym

from metroid_env import make_metroid_env, MetroidConfig, DATA_DIR, MODEL_DIR

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    SB3_AVAILABLE = True
except ImportError:
    print("Stable Baselines 3 required.")
    sys.exit(1)

# =============================================================================
# POLICY NETWORK
# =============================================================================
class MetroidCNNExtractor(BaseFeaturesExtractor):
    """CNN feature extractor matching the environment's frame stack."""
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# =============================================================================
# TRAINING LOOP
# =============================================================================
def train_ppo(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training on State: {args.state} | Scenario: {args.scenario}")
    
    resume = bool(args.load and args.load.endswith(".zip") and os.path.exists(args.load))
    
    # Entropy scheduling
    initial_ent_coef = MetroidConfig.ENT_COEF_START
    final_ent_coef = MetroidConfig.ENT_COEF_END
    total_timesteps = args.steps

    # Create Env
    env = DummyVecEnv([partial(make_metroid_env, state=args.state, scenario=args.scenario)])

    # Callback
    class EntropyScheduleCallback(BaseCallback):
        def __init__(self, initial_ent: float, final_ent: float, total_steps: int, verbose=0):
            super().__init__(verbose)
            self.initial_ent = initial_ent
            self.final_ent = final_ent
            self.total_steps = total_steps
        def _on_step(self) -> bool:
            progress = min(1.0, self.num_timesteps / self.total_steps)
            new_ent = self.initial_ent + (self.final_ent - self.initial_ent) * progress
            self.model.ent_coef = new_ent
            if self.num_timesteps % 5000 == 0:
                print(f"  Step {self.num_timesteps}: ent_coef = {new_ent:.4f}")
            return True

    policy_kwargs = dict(
        features_extractor_class=MetroidCNNExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256], vf=[256])
    )

    if resume:
        print(f"Resuming PPO from {args.load}...")
        try:
            custom_objects = {
                'learning_rate': MetroidConfig.LEARNING_RATE,
                'lr_schedule': lambda _: MetroidConfig.LEARNING_RATE,
                'clip_range': lambda _: MetroidConfig.CLIP_RANGE,
            }
            model = PPO.load(args.load, env=env, device=device, custom_objects=custom_objects)
            model.ent_coef = initial_ent_coef 
            
            # Reset buffer if switching scenarios? Maybe unsafe if observation space changed (it shouldn't have)
            # If resume from Boss to Nav, might be weird due to different Value function landscape.
            # But technically possible.
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        print("Starting fresh PPO training...")
        model = PPO("CnnPolicy", env, verbose=1, device=device, policy_kwargs=policy_kwargs,
                    learning_rate=MetroidConfig.LEARNING_RATE, ent_coef=initial_ent_coef, 
                    n_steps=2048, batch_size=MetroidConfig.BATCH_SIZE, 
                    n_epochs=MetroidConfig.N_EPOCHS, clip_range=MetroidConfig.CLIP_RANGE, 
                    gae_lambda=MetroidConfig.GAE_LAMBDA, gamma=MetroidConfig.GAMMA)

    checkpoint_callback = CheckpointCallback(
        save_freq=20000, 
        save_path=MODEL_DIR, 
        name_prefix=f"{args.state}_{args.scenario}_ppo"
    )

    print(f"Training for {args.steps} steps...")
    model.learn(total_timesteps=args.steps, callback=[
        EntropyScheduleCallback(initial_ent_coef, final_ent_coef, total_timesteps),
        checkpoint_callback
    ])
    
    save_path = os.path.join(MODEL_DIR, f"{args.state}_{args.scenario}_final.zip")
    model.save(save_path)
    print(f"Training complete. Saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Metroid RL Trainer")
    parser.add_argument('--state', type=str, default='BossTorizo', help='Retro State to load')
    parser.add_argument('--scenario', type=str, default='boss', choices=['boss', 'nav'], help='Scenario type')
    parser.add_argument('--steps', type=int, default=100000, help='Training steps')
    parser.add_argument('--load', type=str, help='Model to load')

    args = parser.parse_args()
    train_ppo(args)

if __name__ == "__main__":
    main()

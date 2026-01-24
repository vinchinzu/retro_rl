#!/usr/bin/env python3
"""
Curriculum Training for Super Metroid Route: ZebesStart -> Bomb Torizo

Key insight: Train each room transition separately using [from X] states,
then combine models for full route execution.

Route (Descent to Morph Ball):
  1. Landing Site (ZebesStart) -> Parlor and Alcatraz
  2. Parlor and Alcatraz [from Landing Site] -> Climb
  3. Climb [from Parlor and Alcatraz] -> Pit Room
  4. Pit Room [from Climb] -> Blue Brinstar Elevator Room
  5. Blue Brinstar Elevator Room [from Pit Room] -> Morph Ball Room
  6. Morph Ball Room [from Blue Brinstar Elevator Room] -> Collect Morph Ball

Route (Return to Torizo):
  7. Return through Flyway -> Bomb Torizo Room
  8. Bomb Torizo Room [from Flyway] -> Boss Fight

Usage:
    # Train a specific segment
    python train_curriculum.py train --segment landing_site --steps 50000

    # Train all segments sequentially
    python train_curriculum.py train-all --steps-per-segment 50000

    # Run the full route with trained models
    python train_curriculum.py run --render

    # List available segments
    python train_curriculum.py list-segments
"""

import os
import sys
import json
import time
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Error: stable_baselines3 required")
    sys.exit(1)

import stable_retro as retro

# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTEGRATION_PATH = os.path.join(SCRIPT_DIR, "custom_integrations")
retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
WORLD_MAP_PATH = os.path.join(SCRIPT_DIR, "world_map.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# =============================================================================
# ROUTE SEGMENTS
# =============================================================================
@dataclass
class RouteSegment:
    """Definition of a single room transition to train."""
    name: str               # Segment identifier
    start_state: str        # Retro state to load
    target_room_id: int     # Room ID that signals success
    direction: str          # 'left', 'right', 'down', 'up', 'collect'
    max_steps: int          # Max model steps per episode
    description: str        # Human readable description

# Load world map for room IDs
def load_world_map() -> Dict[str, int]:
    if os.path.exists(WORLD_MAP_PATH):
        with open(WORLD_MAP_PATH, 'r') as f:
            data = json.load(f)
            return {k: int(v, 16) for k, v in data.items()}
    return {}

WORLD_MAP = load_world_map()

# Define the training segments for ZebesStart -> Bomb Torizo route
ROUTE_SEGMENTS = {
    # ==========================================================================
    # DESCENT PHASE (going down/left toward Morph Ball)
    # ==========================================================================
    "landing_site": RouteSegment(
        name="landing_site",
        start_state="ZebesStart",
        target_room_id=WORLD_MAP.get("Parlor and Alcatraz", 0x92fd),
        direction="left",
        max_steps=2000,
        description="Landing Site -> Parlor (go left)"
    ),
    "parlor_descent": RouteSegment(
        name="parlor_descent",
        start_state="Parlor and Alcatraz [from Landing Site]",
        target_room_id=WORLD_MAP.get("Climb", 0x96ba),
        direction="down",
        max_steps=4000,  # Increased - this is a hard room
        description="Parlor -> Climb (go down left) [HARD - needs demos]"
    ),
    "climb_descent": RouteSegment(
        name="climb_descent",
        start_state="Climb [from Parlor and Alcatraz]",
        target_room_id=WORLD_MAP.get("Pit Room", 0x975c),
        direction="down",
        max_steps=2000,
        description="Climb -> Pit Room (go down)"
    ),
    "pit_room_descent": RouteSegment(
        name="pit_room_descent",
        start_state="Pit Room [from Climb]",
        target_room_id=WORLD_MAP.get("Blue Brinstar Elevator Room", 0x97b5),
        direction="down",
        max_steps=2000,
        description="Pit Room -> Blue Brinstar Elevator (go down)"
    ),
    "elevator_descent": RouteSegment(
        name="elevator_descent",
        start_state="Blue Brinstar Elevator Room [from Pit Room]",
        target_room_id=WORLD_MAP.get("Morph Ball Room", 0x9e9f),
        direction="down",
        max_steps=3000,
        description="Elevator -> Morph Ball Room (go down)"
    ),
    "morph_ball_collect": RouteSegment(
        name="morph_ball_collect",
        start_state="Morph Ball Room [from Blue Brinstar Elevator Room]",
        target_room_id=0,  # Special: check for morph ball item
        direction="collect",
        max_steps=3000,
        description="Collect Morph Ball item"
    ),

    # ==========================================================================
    # RETURN PHASE (going up/right toward Bomb Torizo)
    # These are HARDER - require wall jumps and platforming UP
    # ==========================================================================
    "morph_ball_return": RouteSegment(
        name="morph_ball_return",
        start_state="Morph Ball Room [from Construction Zone]",  # After getting item
        target_room_id=WORLD_MAP.get("Blue Brinstar Elevator Room", 0x97b5),
        direction="up",
        max_steps=4000,
        description="Morph Ball Room -> Elevator (go up) [HARD - needs demos]"
    ),
    "elevator_return": RouteSegment(
        name="elevator_return",
        start_state="Blue Brinstar Elevator Room [from Morph Ball Room]",
        target_room_id=WORLD_MAP.get("Pit Room", 0x975c),
        direction="up",
        max_steps=3000,
        description="Elevator -> Pit Room (go up) [HARD - needs demos]"
    ),
    "pit_room_return": RouteSegment(
        name="pit_room_return",
        start_state="Pit Room [from Blue Brinstar Elevator Room]",
        target_room_id=WORLD_MAP.get("Climb", 0x96ba),
        direction="up",
        max_steps=3000,
        description="Pit Room -> Climb (go up) [HARD - needs demos]"
    ),
    "climb_return": RouteSegment(
        name="climb_return",
        start_state="Climb [from Pit Room]",
        target_room_id=WORLD_MAP.get("Parlor and Alcatraz", 0x92fd),
        direction="up",
        max_steps=4000,
        description="Climb -> Parlor (go up) [HARD - needs demos]"
    ),
    "parlor_to_flyway": RouteSegment(
        name="parlor_to_flyway",
        start_state="Parlor and Alcatraz [from Climb]",
        target_room_id=WORLD_MAP.get("Flyway", 0x9879),
        direction="right",
        max_steps=3000,
        description="Parlor -> Flyway (go right)"
    ),
    "flyway_to_torizo": RouteSegment(
        name="flyway_to_torizo",
        start_state="Flyway [from Parlor and Alcatraz]",
        target_room_id=WORLD_MAP.get("Bomb Torizo Room", 0x9804),
        direction="right",
        max_steps=2000,
        description="Flyway -> Bomb Torizo Room (go right)"
    ),
}

# Training order - descent first, then return
DESCENT_ORDER = ["landing_site", "parlor_descent", "climb_descent", "pit_room_descent", "elevator_descent", "morph_ball_collect"]
RETURN_ORDER = ["morph_ball_return", "elevator_return", "pit_room_return", "climb_return", "parlor_to_flyway", "flyway_to_torizo"]
TRAINING_ORDER = DESCENT_ORDER + RETURN_ORDER

# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================
def find_latest_checkpoint(segment_name: str) -> Optional[str]:
    """Return latest checkpoint or final model path for a segment."""
    prefix = f"segment_{segment_name}_"
    candidates = []
    for fname in os.listdir(MODEL_DIR):
        if fname.startswith(prefix) and fname.endswith("_steps.zip"):
            candidates.append(os.path.join(MODEL_DIR, fname))

    if candidates:
        candidates.sort(key=os.path.getmtime)
        return candidates[-1]

    final_path = os.path.join(MODEL_DIR, f"segment_{segment_name}.zip")
    if os.path.exists(final_path):
        return final_path

    return None

# =============================================================================
# WRAPPERS
# =============================================================================
class FrameStack(gym.Wrapper):
    """Stack n_frames RGB frames for motion perception."""
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = None
        old_shape = env.observation_space.shape
        h, w = old_shape[0] // 2, old_shape[1] // 2
        new_shape = (n_frames * 3, h, w)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=new_shape, dtype=np.uint8
        )

    def _get_frame(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            return obs[::2, ::2, :].transpose(2, 0, 1)
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if isinstance(obs, np.ndarray) and len(obs.shape) == 3 and obs.shape[2] == 3:
            info['rgb_frame'] = obs
        frame = self._get_frame(obs)
        self.frames = np.concatenate([frame] * self.n_frames, axis=0)
        return self.frames.copy(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(obs, np.ndarray) and len(obs.shape) == 3 and obs.shape[2] == 3:
            info['rgb_frame'] = obs
        frame = self._get_frame(obs)
        self.frames = np.roll(self.frames, shift=-3, axis=0)
        self.frames[-3:] = frame
        return self.frames.copy(), reward, terminated, truncated, info


class SanitizeAction(gym.ActionWrapper):
    """Prevent contradictory D-pad inputs."""
    def action(self, action):
        if action[6] and action[7]: action[6] = 0; action[7] = 0
        if action[4] and action[5]: action[4] = 0; action[5] = 0
        return action


# Discrete action mapping
_B, _Y, _SELECT, _START, _UP, _DOWN, _LEFT, _RIGHT, _A, _X, _L, _R = range(12)

DISCRETE_ACTIONS = [
    {_LEFT: 1}, {_RIGHT: 1},
    {_LEFT: 1, _X: 1}, {_RIGHT: 1, _X: 1},
    {_X: 1},
    {_UP: 1, _X: 1}, {_UP: 1, _LEFT: 1, _X: 1}, {_UP: 1, _RIGHT: 1, _X: 1},
    {_A: 1, _X: 1}, {_A: 1}, {_A: 1, _LEFT: 1}, {_A: 1, _RIGHT: 1},
    {_B: 1, _LEFT: 1}, {_B: 1, _RIGHT: 1},
    {_DOWN: 1}, {_DOWN: 1, _X: 1}, {_DOWN: 1, _LEFT: 1}, {_DOWN: 1, _RIGHT: 1},
    {_A: 1, _UP: 1, _X: 1}, {_A: 1, _LEFT: 1, _X: 1}, {_A: 1, _RIGHT: 1, _X: 1},
    {_B: 1, _LEFT: 1, _X: 1}, {_B: 1, _RIGHT: 1, _X: 1},
    {_B: 1, _A: 1, _LEFT: 1}, {_B: 1, _A: 1, _RIGHT: 1},
    {},
]


class DiscreteAction(gym.ActionWrapper):
    def __init__(self, env, action_map):
        super().__init__(env)
        self.action_map = action_map
        self.action_space = gym.spaces.Discrete(len(action_map))

    def action(self, action):
        idx = int(action)
        idx = max(0, min(idx, len(self.action_map) - 1))
        mapped = np.zeros(12, dtype=np.int8)
        for button_idx, pressed in self.action_map[idx].items():
            mapped[button_idx] = pressed
        return mapped


class ActionHoldRepeat(gym.Wrapper):
    """Repeat actions for smoother movement."""
    def __init__(self, env, min_hold=2, max_hold=5):
        super().__init__(env)
        self.min_hold = min_hold
        self.max_hold = max_hold

    def step(self, action):
        repeat = np.random.randint(self.min_hold, self.max_hold + 1)
        total_reward = 0.0
        obs = None
        for _ in range(repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


# =============================================================================
# SEGMENT REWARD WRAPPER
# =============================================================================
class SegmentReward(gym.Wrapper):
    """
    Reward wrapper for training individual route segments.
    Terminates successfully when target room is reached.
    """
    def __init__(self, env, segment: RouteSegment):
        super().__init__(env)
        self.segment = segment
        self.prev_x = None
        self.prev_y = None
        self.prev_hp = None
        self.prev_items = None
        self.prev_door_transition = None
        self.frame_count = 0
        self.start_room = None
        self.left_start_room = False
        self.demo_feats = None
        self.demo_actions = None
        self.demo_reward_scale = 0.15

        demo_path = os.path.join(SCRIPT_DIR, "boss_data", "nav_demos.npz")
        if os.path.exists(demo_path):
            data = np.load(demo_path)
            demo_obs = data.get("obs")
            demo_actions = data.get("acts")
            if demo_obs is not None and demo_actions is not None and len(demo_obs) > 0:
                sample_count = min(64, len(demo_obs))
                rng = np.random.default_rng(0)
                idx = rng.choice(len(demo_obs), size=sample_count, replace=False)
                demo_obs = demo_obs[idx]
                demo_actions = demo_actions[idx]

                demo_gray = (
                    demo_obs[:, 0].astype(np.float32) * 0.299
                    + demo_obs[:, 1].astype(np.float32) * 0.587
                    + demo_obs[:, 2].astype(np.float32) * 0.114
                )
                demo_small = demo_gray[:, ::8, ::8][:, :14, :16]
                self.demo_feats = demo_small.reshape(sample_count, -1)
                self.demo_actions = demo_actions.astype(np.int8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_x = info.get('samus_x', 0)
        self.prev_y = info.get('samus_y', 0)
        self.prev_hp = info.get('health', 99)
        self.prev_items = info.get('collected_items', 0) or info.get('items', 0)
        self.prev_door_transition = info.get('door_transition', 0)
        self.frame_count = 0
        self.start_room = info.get('room_id', 0)
        self.left_start_room = False
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frame_count += 1
        shaped_reward = 0.0

        x = info.get('samus_x', 0)
        y = info.get('samus_y', 0)
        hp = info.get('health', 99)
        room_id = info.get('room_id', 0)
        items = info.get('collected_items', 0) or info.get('items', 0)
        door_transition = info.get('door_transition', 0)

        # Check for segment completion
        if self.segment.direction == "collect":
            # Check if morph ball was collected (bit 0 of items)
            if self.prev_items is not None:
                if (items & 0x1) and not (self.prev_items & 0x1):
                    shaped_reward += 5000.0
                    terminated = True
                    print(f"[SUCCESS] Morph Ball collected in {self.frame_count} frames!")
        else:
            # Check if reached target room
            if room_id == self.segment.target_room_id and room_id != self.start_room:
                shaped_reward += 2000.0
                # Speed bonus
                frames_saved = max(0, (self.segment.max_steps * 3) - self.frame_count)
                shaped_reward += frames_saved * 0.5
                terminated = True
                print(f"[SUCCESS] Reached target room in {self.frame_count} frames!")

        # Direction-based progress rewards
        if self.prev_x is not None:
            dx = x - self.prev_x
            dy = y - self.prev_y

            if self.segment.direction == "left" and dx < 0:
                shaped_reward += abs(dx) * 0.3
            elif self.segment.direction == "left" and dx > 0:
                shaped_reward -= abs(dx) * 0.4
            elif self.segment.direction == "right" and dx > 0:
                shaped_reward += dx * 0.3
            elif self.segment.direction == "right" and dx < 0:
                shaped_reward -= abs(dx) * 0.4
            elif self.segment.direction == "down" and dy > 0:
                shaped_reward += dy * 0.2
            elif self.segment.direction == "down" and dy < 0:
                shaped_reward -= abs(dy) * 0.3
            elif self.segment.direction == "up" and dy < 0:
                shaped_reward += abs(dy) * 0.2
            elif self.segment.direction == "up" and dy > 0:
                shaped_reward -= abs(dy) * 0.3
            if self.segment.name == "parlor_descent" and dx < 0:
                shaped_reward += abs(dx) * 0.1

        # Action intent reward (encourage consistent directional input)
        if self.segment.direction == "left":
            shaped_reward += 0.08 if action[6] else 0.0
            shaped_reward -= 0.08 if action[7] else 0.0
        elif self.segment.direction == "right":
            shaped_reward += 0.08 if action[7] else 0.0
            shaped_reward -= 0.08 if action[6] else 0.0
        elif self.segment.direction == "up":
            shaped_reward += 0.06 if action[4] else 0.0
            shaped_reward -= 0.06 if action[5] else 0.0
        elif self.segment.direction == "down":
            shaped_reward += 0.06 if action[5] else 0.0
            shaped_reward -= 0.06 if action[4] else 0.0

        # Demo matching reward (lightweight, optional)
        shaped_reward += self._demo_match_reward(obs, action)

        # Door interaction reward (helps exit loops)
        if door_transition and not self.prev_door_transition:
            shaped_reward += 50.0

        if room_id != self.start_room:
            self.left_start_room = True

        # Penalize leaving the target path in hard segments
        if self.segment.name == "parlor_descent":
            if self.left_start_room and room_id == self.start_room:
                shaped_reward -= 2000.0
                terminated = True
            elif room_id not in (self.start_room, self.segment.target_room_id):
                shaped_reward -= 500.0
                terminated = True

        # Damage penalty (reduced compared to death penalty)
        if hp < self.prev_hp:
            shaped_reward -= (self.prev_hp - hp) * 2.0

        # Death penalty
        if hp <= 0:
            shaped_reward -= 300.0
            terminated = True

        # Small time penalty to encourage speed
        shaped_reward -= 0.2

        # Update state
        self.prev_x = x
        self.prev_y = y
        self.prev_hp = hp
        self.prev_items = items
        self.prev_door_transition = door_transition

        reward += shaped_reward
        info['segment'] = self.segment.name
        info['target_room'] = self.segment.target_room_id
        return obs, reward, terminated, truncated, info

    def _demo_match_reward(self, obs, action):
        if self.demo_feats is None or obs is None:
            return 0.0

        if isinstance(obs, np.ndarray):
            if obs.ndim == 3 and obs.shape[2] == 3:
                gray = (
                    obs[..., 0].astype(np.float32) * 0.299
                    + obs[..., 1].astype(np.float32) * 0.587
                    + obs[..., 2].astype(np.float32) * 0.114
                )
            elif obs.ndim == 3 and obs.shape[0] == 3:
                gray = (
                    obs[0].astype(np.float32) * 0.299
                    + obs[1].astype(np.float32) * 0.587
                    + obs[2].astype(np.float32) * 0.114
                )
            else:
                return 0.0
        else:
            return 0.0

        h, w = gray.shape
        step_h = max(1, h // 14)
        step_w = max(1, w // 16)
        small = gray[::step_h, ::step_w][:14, :16].reshape(1, -1)

        diffs = self.demo_feats - small
        idx = int(np.argmin(np.sum(diffs * diffs, axis=1)))
        demo_action = self.demo_actions[idx]

        demo_pressed = demo_action > 0
        action_pressed = np.asarray(action) > 0
        demo_count = int(demo_pressed.sum())
        if demo_count == 0:
            return 0.0

        matches = int((demo_pressed & action_pressed).sum())
        return (matches / demo_count) * self.demo_reward_scale


# =============================================================================
# CNN FEATURE EXTRACTOR
# =============================================================================
class MetroidCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]

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
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


# =============================================================================
# ENTROPY SCHEDULE CALLBACK
# =============================================================================
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
            print(f"  Step {self.num_timesteps}: ent_coef = {new_ent:.5f}")
        return True


# =============================================================================
# ENVIRONMENT FACTORY
# =============================================================================
def make_segment_env(segment: RouteSegment, render_mode: str = "rgb_array") -> gym.Env:
    """Create environment for training a specific route segment."""

    env = retro.make(
        game="SuperMetroid-Snes",
        state=segment.start_state,
        use_restricted_actions=retro.Actions.ALL,
        render_mode=render_mode
    )

    env = SanitizeAction(env)
    env = SegmentReward(env, segment)
    env = DiscreteAction(env, DISCRETE_ACTIONS)
    env = ActionHoldRepeat(env, min_hold=2, max_hold=4)
    env = FrameStack(env, n_frames=4)
    env = TimeLimit(env, max_episode_steps=segment.max_steps)

    return env


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================
def train_segment(
    segment_name: str,
    steps: int = 50000,
    load_path: Optional[str] = None,
    device: str = "cuda"
) -> str:
    """Train a single route segment."""

    if segment_name not in ROUTE_SEGMENTS:
        print(f"Error: Unknown segment '{segment_name}'")
        print(f"Available: {list(ROUTE_SEGMENTS.keys())}")
        return None

    segment = ROUTE_SEGMENTS[segment_name]

    print("="*60)
    print(f"Training Segment: {segment_name}")
    print(f"Description: {segment.description}")
    print(f"Start State: {segment.start_state}")
    print(f"Target Room: 0x{segment.target_room_id:04X}")
    print("="*60)

    def env_fn():
        return make_segment_env(segment)

    env = DummyVecEnv([env_fn])
    env = VecMonitor(env, filename=os.path.join(LOG_DIR, f"segment_{segment_name}"))

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not load_path:
        load_path = find_latest_checkpoint(segment_name)

    if load_path and os.path.exists(load_path):
        print(f"Loading model from {load_path}")
        model = PPO.load(load_path, env=env, device=device)
    else:
        print("Creating new PPO model")
        policy_kwargs = dict(
            features_extractor_class=MetroidCNNExtractor,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=dict(pi=[256, 128], vf=[256, 128])
        )

        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            device=device,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            ent_coef=0.02,
            n_steps=1024,
            batch_size=256,
            n_epochs=4,
            clip_range=0.2,
            gae_lambda=0.95,
            gamma=0.99,
            tensorboard_log=LOG_DIR
        )

    # Callbacks
    ent_callback = EntropyScheduleCallback(0.02, 0.005, steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=MODEL_DIR,
        name_prefix=f"segment_{segment_name}"
    )

    print(f"Training for {steps} steps...")
    model.learn(
        total_timesteps=steps,
        callback=[ent_callback, checkpoint_callback],
        progress_bar=True
    )

    save_path = os.path.join(MODEL_DIR, f"segment_{segment_name}.zip")
    model.save(save_path)
    print(f"Model saved to {save_path}")

    env.close()
    return save_path


def train_all_segments(steps_per_segment: int = 50000, device: str = "cuda"):
    """Train all segments in order."""

    print("="*60)
    print("CURRICULUM TRAINING - ALL SEGMENTS")
    print("="*60)
    print(f"Training order: {TRAINING_ORDER}")
    print(f"Steps per segment: {steps_per_segment}")
    print("="*60)

    for i, segment_name in enumerate(TRAINING_ORDER):
        print(f"\n[{i+1}/{len(TRAINING_ORDER)}] Training segment: {segment_name}")
        train_segment(segment_name, steps=steps_per_segment, device=device)
        print(f"Completed: {segment_name}")

    print("\n" + "="*60)
    print("CURRICULUM TRAINING COMPLETE")
    print("="*60)


# =============================================================================
# RUN WITH TRAINED MODELS
# =============================================================================
def run_full_route(render: bool = True, device: str = "cuda", start_state: str = "ZebesStart"):
    """Run the full route using trained segment models."""

    if render:
        import pygame

    print("="*60)
    print("Running Full Route with Trained Models")
    print(f"Start state: {start_state}")
    print("="*60)

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load all segment models
    models = {}
    for segment_name in TRAINING_ORDER:
        model_path = os.path.join(MODEL_DIR, f"segment_{segment_name}.zip")
        if os.path.exists(model_path):
            print(f"Loading: {segment_name}")
            models[segment_name] = PPO.load(model_path, device=device)
        else:
            print(f"Warning: Model not found for {segment_name}")

    if not models:
        print("No models found. Run training first.")
        return

    # Create base environment from ZebesStart
    env = retro.make(
        game="SuperMetroid-Snes",
        state=start_state,
        use_restricted_actions=retro.Actions.ALL,
        render_mode='rgb_array'
    )
    env = SanitizeAction(env)
    env = DiscreteAction(env, DISCRETE_ACTIONS)
    env = FrameStack(env, n_frames=4)

    # Initialize pygame
    if render:
        pygame.init()
        screen = pygame.display.set_mode((256*2, 224*2))
        pygame.display.set_caption("Super Metroid - Curriculum Bot")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont('monospace', 14)

    obs, info = env.reset()

    # Determine current segment based on room
    current_segment_idx = 0
    for idx, name in enumerate(TRAINING_ORDER):
        if ROUTE_SEGMENTS[name].start_state == start_state:
            current_segment_idx = idx
            break
    print(f"Starting segment: {TRAINING_ORDER[current_segment_idx]}")

    running = True
    frame = 0
    total_reward = 0

    while running and current_segment_idx < len(TRAINING_ORDER):
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

        segment_name = TRAINING_ORDER[current_segment_idx]
        segment = ROUTE_SEGMENTS[segment_name]

        # Get model for current segment
        model = models.get(segment_name)
        if model is None:
            print(f"No model for {segment_name}, using random")
            action = np.random.randint(0, len(DISCRETE_ACTIONS))
        else:
            action, _ = model.predict(obs, deterministic=False)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        frame += 1
        total_reward += reward

        # Check for segment transition
        room_id = info.get('room_id', 0)
        items = info.get('collected_items', 0) or info.get('items', 0)

        if segment.direction == "collect":
            if items & 0x1:  # Morph ball collected
                print(f"[MILESTONE] Morph Ball collected at frame {frame}")
                current_segment_idx += 1
        elif room_id == segment.target_room_id:
            print(f"[SEGMENT] Completed {segment_name} at frame {frame}")
            current_segment_idx += 1

        # Render
        if render:
            rgb = info.get('rgb_frame')
            if rgb is not None:
                surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
                scaled = pygame.transform.scale(surf, (256*2, 224*2))
                screen.blit(scaled, (0, 0))

                # HUD
                texts = [
                    f"Segment: {segment_name}",
                    f"Frame: {frame}",
                    f"Room: 0x{room_id:04X}",
                    f"Target: 0x{segment.target_room_id:04X}",
                    f"Reward: {total_reward:.0f}",
                ]

                for i, txt in enumerate(texts):
                    text_surf = font.render(txt, True, (0, 255, 0))
                    screen.blit(text_surf, (10, 10 + i * 16))

                pygame.display.flip()
                clock.tick(60)

        if terminated:
            print(f"Episode terminated at frame {frame}")
            break

    env.close()
    if render:
        pygame.quit()

    print(f"\nRun complete: {frame} frames, reward: {total_reward:.0f}")


def list_segments():
    """List all available training segments."""
    print("\n" + "="*60)
    print("AVAILABLE TRAINING SEGMENTS")
    print("="*60)

    for name, segment in ROUTE_SEGMENTS.items():
        model_path = os.path.join(MODEL_DIR, f"segment_{name}.zip")
        trained = "TRAINED" if os.path.exists(model_path) else "not trained"
        print(f"\n  {name} [{trained}]")
        print(f"    State: {segment.start_state}")
        print(f"    Direction: {segment.direction}")
        print(f"    Max steps: {segment.max_steps}")
        print(f"    {segment.description}")

    print("\n" + "="*60)
    print(f"Training order: {' -> '.join(TRAINING_ORDER)}")
    print("="*60)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Curriculum Training for Super Metroid")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Train single segment
    train = subparsers.add_parser('train', help='Train a single segment')
    train.add_argument('--segment', type=str, required=True, help='Segment name')
    train.add_argument('--steps', type=int, default=50000, help='Training steps')
    train.add_argument('--load', type=str, help='Model to resume from')
    train.add_argument('--device', type=str, default='cuda', help='Device')

    # Train all segments
    train_all = subparsers.add_parser('train-all', help='Train all segments')
    train_all.add_argument('--steps-per-segment', type=int, default=50000)
    train_all.add_argument('--device', type=str, default='cuda')

    # Run full route
    run = subparsers.add_parser('run', help='Run with trained models')
    run.add_argument('--render', action='store_true', help='Enable rendering')
    run.add_argument('--start-state', type=str, default="ZebesStart", help='Start state (e.g., \"Parlor and Alcatraz [from Landing Site]\")')
    run.add_argument('--device', type=str, default='cuda')

    # List segments
    subparsers.add_parser('list-segments', help='List available segments')

    args = parser.parse_args()

    if args.command == 'train':
        train_segment(args.segment, steps=args.steps, load_path=args.load, device=args.device)
    elif args.command == 'train-all':
        train_all_segments(steps_per_segment=args.steps_per_segment, device=args.device)
    elif args.command == 'run':
        run_full_route(render=args.render, device=args.device, start_state=args.start_state)
    elif args.command == 'list-segments':
        list_segments()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

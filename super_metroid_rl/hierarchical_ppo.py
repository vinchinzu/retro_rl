#!/usr/bin/env python3
"""
Hierarchical PPO Pipeline for Super Metroid
Goal: ZebesStart -> Get Morph Ball -> Beat Torizo Boss

Uses:
- Segment-based model selection (navigation vs boss)
- Demo data for behavioral cloning warmstart
- Performance tracking against best_times.json
"""

import os
import sys
import json
import time
import glob
import argparse
from enum import Enum
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

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
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable_baselines3 not available")

import stable_retro as retro

# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTEGRATION_PATH = os.path.join(SCRIPT_DIR, "custom_integrations")
retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
DATA_DIR = os.path.join(SCRIPT_DIR, "boss_data")
DEMO_DIR = os.path.join(SCRIPT_DIR, "demos")
BEST_TIMES_PATH = os.path.join(SCRIPT_DIR, "best_times.json")
WORLD_MAP_PATH = os.path.join(SCRIPT_DIR, "world_map.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# =============================================================================
# GAME STATE DEFINITIONS
# =============================================================================
class GamePhase(Enum):
    """High-level mission phases"""
    DESCENT = 1      # ZebesStart -> Morph Ball Room
    RETURN = 2       # Morph Ball Room -> Bomb Torizo Room
    BOSS = 3         # Fight Torizo
    COMPLETE = 4     # Victory

@dataclass
class RoomInfo:
    """Information about a specific room"""
    name: str
    room_id: int
    segment: str  # 'nav' or 'boss'
    descent_dir: Optional[str] = None
    return_dir: Optional[str] = None

# Key rooms on the ZebesStart -> Torizo route
ROUTE_ROOMS = {
    0x91F8: RoomInfo("Landing Site", 0x91F8, "nav", "left", "right"),
    0x92FD: RoomInfo("Parlor and Alcatraz", 0x92FD, "nav", "down", "up"),
    0x96BA: RoomInfo("Climb", 0x96BA, "nav", "down", "up"),
    0x975C: RoomInfo("Pit Room", 0x975C, "nav", "down", "up"),
    0x97B5: RoomInfo("Blue Brinstar Elevator", 0x97B5, "nav", "down", "up"),
    0x9E9F: RoomInfo("Morph Ball Room", 0x9E9F, "nav", "collect", "right"),
    0x9804: RoomInfo("Bomb Torizo Room", 0x9804, "boss", None, "boss"),
    0x9879: RoomInfo("Flyway", 0x9879, "nav", None, "left"),
}

# Room IDs where we switch to boss model
BOSS_ROOMS = {0x9804}  # Bomb Torizo Room

# =============================================================================
# WRAPPERS (reused from existing code)
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
    {_B: 1, _A: 1, _LEFT: 1}, {_B: 1, _A: 1, _RIGHT: 1},  # Spin jump
    {},  # No-op
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


class ForceMissiles(gym.Wrapper):
    """Force missiles selected for boss fights."""
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._force_missiles()
        return obs, info

    def step(self, action):
        self._force_missiles()
        return self.env.step(action)

    def _force_missiles(self):
        try:
            missiles = self.unwrapped.data.lookup_value('missiles')
            if missiles > 0:
                self.unwrapped.data.set_value('selected_item', 1)
        except:
            pass


class ActionHoldRepeat(gym.Wrapper):
    """Repeat actions for sampled hold length (smoother movement)."""
    def __init__(self, env, min_hold=2, max_hold=6):
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
# HIERARCHICAL REWARD WRAPPER
# =============================================================================
class HierarchicalReward(gym.Wrapper):
    """
    Reward shaping that adapts based on current room/phase.
    Tracks room transitions and applies segment-specific rewards.
    """
    def __init__(self, env, phase: GamePhase = GamePhase.DESCENT):
        super().__init__(env)
        self.phase = phase
        self.prev_room = None
        self.prev_hp = None
        self.prev_boss_hp = None
        self.prev_x = None
        self.prev_y = None
        self.has_morph_ball = False
        self.room_entry_frame = 0
        self.frame_count = 0
        self.rooms_visited = set()
        self.total_damage_dealt = 0

        # Load best times for comparison
        self.best_times = self._load_best_times()

    def _load_best_times(self) -> Dict:
        if os.path.exists(BEST_TIMES_PATH):
            with open(BEST_TIMES_PATH, 'r') as f:
                return json.load(f)
        return {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_room = info.get('room_id', 0)
        self.prev_hp = info.get('health', 99)
        self.prev_boss_hp = 0
        self.prev_x = info.get('samus_x', 0)
        self.prev_y = info.get('samus_y', 0)
        self.has_morph_ball = False
        self.room_entry_frame = 0
        self.frame_count = 0
        self.rooms_visited = {self.prev_room}
        self.total_damage_dealt = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frame_count += 1
        shaped_reward = 0.0

        # Extract state
        room_id = info.get('room_id', 0)
        hp = info.get('health', 99)
        x = info.get('samus_x', 0)
        y = info.get('samus_y', 0)
        items = info.get('collected_items', 0) or info.get('items', 0)
        boss_hp = info.get('boss_hp', 0) or info.get('enemy0_hp', 0)

        # Check morph ball acquisition
        if not self.has_morph_ball and (items & 0x1):
            self.has_morph_ball = True
            shaped_reward += 2000.0  # Major milestone
            self.phase = GamePhase.RETURN
            print(f"[MILESTONE] Morph Ball acquired! Switching to RETURN phase")

        # Room transition rewards
        if room_id != self.prev_room:
            # New room bonus
            if room_id not in self.rooms_visited:
                shaped_reward += 500.0
                self.rooms_visited.add(room_id)
            else:
                shaped_reward += 50.0  # Revisit (needed for return trip)

            # Calculate room completion time
            room_frames = self.frame_count - self.room_entry_frame
            room_info = ROUTE_ROOMS.get(self.prev_room)
            if room_info:
                print(f"[ROOM] Left {room_info.name} in {room_frames} frames ({room_frames/60:.1f}s)")

            self.room_entry_frame = self.frame_count

            # Check if entered boss room
            if room_id in BOSS_ROOMS:
                self.phase = GamePhase.BOSS
                shaped_reward += 1000.0
                print(f"[PHASE] Entered boss room! Switching to BOSS phase")

        # Phase-specific rewards
        if self.phase == GamePhase.BOSS:
            # Boss fight rewards
            if self.prev_boss_hp is None:
                self.prev_boss_hp = boss_hp

            # Damage dealt to boss
            if boss_hp < self.prev_boss_hp and self.prev_boss_hp > 0:
                dmg = self.prev_boss_hp - boss_hp
                shaped_reward += dmg * 30.0
                self.total_damage_dealt += dmg
                print(f"  HIT! Dealt {dmg} dmg (total: {self.total_damage_dealt})")

            # Boss killed
            if self.prev_boss_hp > 0 and boss_hp == 0:
                shaped_reward += 5000.0
                self.phase = GamePhase.COMPLETE
                print(f"[VICTORY] Torizo defeated in {self.frame_count} frames!")
                terminated = True

            self.prev_boss_hp = boss_hp

        else:
            # Navigation rewards
            dx = x - self.prev_x if self.prev_x else 0

            # Direction-based progress (phase-dependent)
            if self.phase == GamePhase.DESCENT:
                # Reward moving left (toward morph ball)
                if dx < 0:
                    shaped_reward += abs(dx) * 0.1
            elif self.phase == GamePhase.RETURN:
                # Reward moving right (toward torizo)
                if dx > 0:
                    shaped_reward += dx * 0.1

        # Damage taken penalty
        if hp < self.prev_hp:
            shaped_reward -= (self.prev_hp - hp) * 5.0

        # Death penalty
        if hp <= 0:
            shaped_reward -= 500.0
            terminated = True

        # Small time penalty
        shaped_reward -= 0.05

        # Update state
        self.prev_room = room_id
        self.prev_hp = hp
        self.prev_x = x
        self.prev_y = y

        # Add tracking info
        info['phase'] = self.phase.name
        info['has_morph_ball'] = self.has_morph_ball
        info['rooms_visited'] = len(self.rooms_visited)
        info['total_damage_dealt'] = self.total_damage_dealt

        reward += shaped_reward
        return obs, reward, terminated, truncated, info


# =============================================================================
# HIERARCHICAL MODEL SELECTOR
# =============================================================================
class HierarchicalModelSelector:
    """
    Selects and switches between segment-specific models based on game state.

    Models:
    - nav_model: General navigation (ZebesStart -> Morph Ball -> Flyway)
    - boss_model: Torizo boss fight (pre-trained)
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.nav_model = None
        self.boss_model = None
        self.current_model = None
        self.current_segment = "nav"

        # Load pre-trained boss model
        boss_path = os.path.join(MODEL_DIR, "boss_ppo.zip")
        if os.path.exists(boss_path):
            print(f"Loading boss model from {boss_path}")
            self.boss_model = PPO.load(boss_path, device=self.device)
        else:
            print(f"Warning: Boss model not found at {boss_path}")

    def load_nav_model(self, path: str):
        """Load navigation model."""
        if os.path.exists(path):
            print(f"Loading nav model from {path}")
            self.nav_model = PPO.load(path, device=self.device)
        else:
            print(f"Warning: Nav model not found at {path}")

    def select_model(self, info: Dict) -> Tuple[Optional[PPO], str]:
        """
        Select appropriate model based on current game state.
        Returns (model, segment_name).
        """
        room_id = info.get('room_id', 0)

        # Check if in boss room
        if room_id in BOSS_ROOMS:
            if self.current_segment != "boss":
                print(f"[SELECTOR] Switching to boss model (room: 0x{room_id:04X})")
                self.current_segment = "boss"
            return self.boss_model, "boss"

        # Otherwise use navigation model
        if self.current_segment != "nav":
            print(f"[SELECTOR] Switching to nav model (room: 0x{room_id:04X})")
            self.current_segment = "nav"
        return self.nav_model, "nav"

    def get_action(self, obs: np.ndarray, info: Dict) -> int:
        """Get action from appropriate model."""
        model, segment = self.select_model(info)

        if model is None:
            # Fallback: random action
            return np.random.randint(0, len(DISCRETE_ACTIONS))

        action, _ = model.predict(obs, deterministic=False)
        return action


# =============================================================================
# PERFORMANCE TRACKER
# =============================================================================
class PerformanceTracker:
    """Track and compare performance against best times."""

    def __init__(self):
        self.best_times = self._load_best_times()
        self.current_run = {
            'start_time': None,
            'room_splits': {},
            'total_frames': 0,
            'success': False
        }

    def _load_best_times(self) -> Dict:
        if os.path.exists(BEST_TIMES_PATH):
            with open(BEST_TIMES_PATH, 'r') as f:
                return json.load(f)
        return {}

    def _save_best_times(self):
        with open(BEST_TIMES_PATH, 'w') as f:
            json.dump(self.best_times, f, indent=4)

    def start_run(self):
        self.current_run = {
            'start_time': time.time(),
            'room_splits': {},
            'total_frames': 0,
            'success': False
        }

    def record_room_split(self, room_name: str, frames: int):
        self.current_run['room_splits'][room_name] = frames

        # Compare to best
        if room_name in self.best_times:
            best = self.best_times[room_name]['best_frames']
            diff = frames - best
            status = "NEW BEST!" if diff < 0 else f"+{diff} frames"
            print(f"  Split: {room_name} - {frames} frames ({status})")
        else:
            print(f"  Split: {room_name} - {frames} frames (first recorded)")

    def finish_run(self, success: bool, total_frames: int):
        self.current_run['success'] = success
        self.current_run['total_frames'] = total_frames

        if success:
            print(f"\n=== RUN COMPLETE ===")
            print(f"Total time: {total_frames} frames ({total_frames/60:.1f}s)")

            # Update best times for each room
            for room_name, frames in self.current_run['room_splits'].items():
                if room_name not in self.best_times:
                    self.best_times[room_name] = {
                        'best_frames': frames,
                        'best_seconds': frames / 60.0,
                        'history': []
                    }

                if frames < self.best_times[room_name]['best_frames']:
                    self.best_times[room_name]['best_frames'] = frames
                    self.best_times[room_name]['best_seconds'] = frames / 60.0

                self.best_times[room_name]['history'].append({
                    'frames': frames,
                    'seconds': frames / 60.0,
                    'timestamp': int(time.time())
                })

            self._save_best_times()


# =============================================================================
# CNN FEATURE EXTRACTOR (for SB3)
# =============================================================================
if SB3_AVAILABLE:
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
if SB3_AVAILABLE:
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

            if self.num_timesteps % 10000 == 0:
                print(f"  Step {self.num_timesteps}: ent_coef = {new_ent:.5f}")
            return True


# =============================================================================
# ENVIRONMENT FACTORY
# =============================================================================
def make_hierarchical_env(
    state: str = "ZebesStart",
    phase: GamePhase = GamePhase.DESCENT,
    max_steps: int = 4000,
    render_mode: str = "rgb_array",
    record_dir: Optional[str] = None
) -> gym.Env:
    """Create environment with hierarchical reward wrapper.

    Note: max_steps refers to model actions (after ActionHoldRepeat).
    With avg hold of 3.5 frames, 4000 steps ≈ 14000 game frames ≈ 4 minutes.
    """

    env = retro.make(
        game="SuperMetroid-Snes",
        state=state,
        use_restricted_actions=retro.Actions.ALL,
        render_mode=render_mode,
        record=record_dir
    )

    env = SanitizeAction(env)

    # Add ForceMissiles for boss phases
    if phase == GamePhase.BOSS:
        env = ForceMissiles(env)

    env = HierarchicalReward(env, phase=phase)
    env = DiscreteAction(env, DISCRETE_ACTIONS)
    env = ActionHoldRepeat(env, min_hold=2, max_hold=5)
    env = FrameStack(env, n_frames=4)

    # TimeLimit goes AFTER ActionHoldRepeat so it counts model decisions, not frames
    env = TimeLimit(env, max_episode_steps=max_steps)

    return env


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================
def train_navigation(
    steps: int = 500000,
    load_path: Optional[str] = None,
    save_name: str = "nav_ppo"
):
    """Train navigation model (ZebesStart -> Morph Ball -> Flyway)."""
    if not SB3_AVAILABLE:
        print("Error: stable_baselines3 required")
        return

    print("="*60)
    print("Training Navigation Model")
    print("="*60)

    def env_fn():
        return make_hierarchical_env(
            state="ZebesStart",
            phase=GamePhase.DESCENT,
            max_steps=8000  # ~2 minutes per episode
        )

    env = DummyVecEnv([env_fn])
    env = VecMonitor(env, filename=os.path.join(LOG_DIR, f"monitor_{save_name}"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
            learning_rate=1e-4,
            ent_coef=0.02,
            n_steps=2048,
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
        name_prefix=f"{save_name}_checkpoint"
    )

    print(f"Training for {steps} steps...")
    model.learn(
        total_timesteps=steps,
        callback=[ent_callback, checkpoint_callback],
        progress_bar=True
    )

    save_path = os.path.join(MODEL_DIR, f"{save_name}.zip")
    model.save(save_path)
    print(f"Model saved to {save_path}")

    env.close()
    return save_path


def train_full_route(
    steps: int = 1000000,
    nav_load: Optional[str] = None
):
    """
    Train the full ZebesStart -> Torizo route.
    Uses pre-trained boss model, focuses on navigation training.
    """
    print("="*60)
    print("Training Full Route (ZebesStart -> Torizo)")
    print("="*60)

    # First, ensure we have a navigation model
    nav_path = nav_load or os.path.join(MODEL_DIR, "nav_ppo.zip")

    if not os.path.exists(nav_path):
        print("No navigation model found. Training navigation first...")
        nav_path = train_navigation(steps=steps//2, save_name="nav_ppo")

    # Continue training navigation with full route
    train_navigation(
        steps=steps,
        load_path=nav_path,
        save_name="nav_ppo_full"
    )


# =============================================================================
# RUN BOT (INFERENCE)
# =============================================================================
def run_hierarchical_bot(
    nav_model_path: Optional[str] = None,
    render: bool = True
):
    """Run the hierarchical bot from ZebesStart to Torizo."""
    if render:
        import pygame

    print("="*60)
    print("Running Hierarchical Bot")
    print("="*60)

    # Initialize selector
    selector = HierarchicalModelSelector()

    # Load navigation model
    nav_path = nav_model_path or os.path.join(MODEL_DIR, "nav_ppo.zip")
    if os.path.exists(nav_path):
        selector.load_nav_model(nav_path)
    else:
        print(f"Warning: Nav model not found at {nav_path}")

    # Create environment
    env = make_hierarchical_env(
        state="ZebesStart",
        phase=GamePhase.DESCENT,
        max_steps=15000,  # ~4 minutes
        render_mode="rgb_array"
    )

    # Initialize pygame for rendering
    if render:
        pygame.init()
        screen = pygame.display.set_mode((256*2, 224*2))
        pygame.display.set_caption("Super Metroid - Hierarchical Bot")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont('monospace', 14)

    # Initialize tracker
    tracker = PerformanceTracker()

    # Run episode
    obs, info = env.reset()
    tracker.start_run()

    running = True
    frame = 0
    prev_room = info.get('room_id', 0)
    room_start_frame = 0

    while running:
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

        # Get action from hierarchical selector
        action = selector.get_action(obs, info)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        frame += 1

        # Track room transitions
        room_id = info.get('room_id', 0)
        if room_id != prev_room:
            room_frames = frame - room_start_frame
            room_info = ROUTE_ROOMS.get(prev_room)
            room_name = room_info.name if room_info else f"0x{prev_room:04X}"
            tracker.record_room_split(room_name, room_frames)
            room_start_frame = frame
            prev_room = room_id

        # Render
        if render:
            rgb = info.get('rgb_frame')
            if rgb is not None:
                surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
                scaled = pygame.transform.scale(surf, (256*2, 224*2))
                screen.blit(scaled, (0, 0))

                # HUD
                phase = info.get('phase', 'UNKNOWN')
                morph = "YES" if info.get('has_morph_ball', False) else "NO"
                rooms = info.get('rooms_visited', 0)
                dmg = info.get('total_damage_dealt', 0)

                texts = [
                    f"Phase: {phase}",
                    f"Morph Ball: {morph}",
                    f"Rooms: {rooms}",
                    f"Frame: {frame}",
                    f"Segment: {selector.current_segment}",
                ]

                if phase == "BOSS":
                    texts.append(f"Damage: {dmg}")

                for i, txt in enumerate(texts):
                    text_surf = font.render(txt, True, (0, 255, 0))
                    screen.blit(text_surf, (10, 10 + i * 16))

                pygame.display.flip()
                clock.tick(60)
        else:
            # Headless mode - periodic status output
            if frame % 300 == 0:  # Every 5 seconds
                phase = info.get('phase', 'UNKNOWN')
                morph = "YES" if info.get('has_morph_ball', False) else "NO"
                rooms = info.get('rooms_visited', 0)
                room_id = info.get('room_id', 0)
                room_info = ROUTE_ROOMS.get(room_id)
                room_name = room_info.name if room_info else f"0x{room_id:04X}"
                print(f"[F{frame}] Phase: {phase} | Room: {room_name} | Morph: {morph} | Visited: {rooms}")

        if terminated or truncated:
            success = info.get('phase') == 'COMPLETE'
            tracker.finish_run(success, frame)
            break

    env.close()
    if render:
        pygame.quit()


# =============================================================================
# DEMO EXTRACTION FOR BC
# =============================================================================
def extract_demos_for_bc(output_path: Optional[str] = None):
    """
    Extract training data from demo recordings for behavioral cloning.
    """
    print("="*60)
    print("Extracting Demo Data for BC")
    print("="*60)

    # Find all Zebes demos
    demo_patterns = [
        os.path.join(DEMO_DIR, "*ZebesStart*.bk2"),
        os.path.join(DEMO_DIR, "*Zebes*.bk2"),
    ]

    bk2_files = []
    for pattern in demo_patterns:
        bk2_files.extend(glob.glob(pattern))
    bk2_files = list(set(bk2_files))

    if not bk2_files:
        print("No demo files found!")
        return

    print(f"Found {len(bk2_files)} demo files")

    all_obs = []
    all_acts = []

    for bk2_path in bk2_files:
        print(f"Processing: {os.path.basename(bk2_path)}")

        try:
            movie = retro.Movie(bk2_path)
            movie.step()

            game = movie.get_game()
            state = movie.get_state()

            env = retro.make(
                game=game,
                state=retro.State.NONE,
                use_restricted_actions=retro.Actions.ALL,
                render_mode='rgb_array'
            )
            env.initial_state = state
            obs, info = env.reset()

            frames = 0
            while movie.step():
                keys = []
                for p in range(movie.players):
                    for i in range(env.num_buttons):
                        keys.append(int(movie.get_key(i, p)))

                action = np.array(keys[:12], dtype=np.int32)

                # Sanitize
                if action[6] and action[7]: action[6] = 0; action[7] = 0
                if action[4] and action[5]: action[4] = 0; action[5] = 0

                # Store frame
                rgb_small = obs[::2, ::2, :].transpose(2, 0, 1)
                all_obs.append(rgb_small)
                all_acts.append(action)

                obs, _, terminated, truncated, _ = env.step(action)
                frames += 1

                if terminated or truncated:
                    break

            env.close()
            print(f"  Extracted {frames} frames")

        except Exception as e:
            print(f"  Error processing {bk2_path}: {e}")
            continue

    if all_obs:
        output = output_path or os.path.join(DATA_DIR, "nav_demos.npz")
        all_obs = np.array(all_obs)
        all_acts = np.array(all_acts)
        np.savez_compressed(output, obs=all_obs, acts=all_acts)
        print(f"Saved {len(all_obs)} frames to {output}")
    else:
        print("No frames extracted!")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Hierarchical PPO for Super Metroid")

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Train navigation
    train_nav = subparsers.add_parser('train-nav', help='Train navigation model')
    train_nav.add_argument('--steps', type=int, default=500000)
    train_nav.add_argument('--load', type=str, help='Model to resume from')

    # Train full route
    train_full = subparsers.add_parser('train-full', help='Train full route')
    train_full.add_argument('--steps', type=int, default=1000000)
    train_full.add_argument('--nav-load', type=str, help='Nav model to start from')

    # Run bot
    run = subparsers.add_parser('run', help='Run hierarchical bot')
    run.add_argument('--nav-model', type=str, help='Navigation model path')
    run.add_argument('--no-render', action='store_true', help='Disable rendering')

    # Extract demos
    extract = subparsers.add_parser('extract-demos', help='Extract BC training data')
    extract.add_argument('--output', type=str, help='Output path')

    args = parser.parse_args()

    if args.command == 'train-nav':
        train_navigation(steps=args.steps, load_path=args.load)
    elif args.command == 'train-full':
        train_full_route(steps=args.steps, nav_load=args.nav_load)
    elif args.command == 'run':
        run_hierarchical_bot(nav_model_path=args.nav_model, render=not args.no_render)
    elif args.command == 'extract-demos':
        extract_demos_for_bc(output_path=args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

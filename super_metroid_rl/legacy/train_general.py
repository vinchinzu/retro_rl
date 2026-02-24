#!/usr/bin/env python3
"""
Bomb Torizo Boss Fight - Training Script
Combines Behavioral Cloning from demos + PPO fine-tuning

Usage:
    # Step 1: Extract training data from bk2 demos
    ../retro_env/bin/python train_boss.py --extract

    # Step 2: Train BC model on demos
    ../retro_env/bin/python train_boss.py --train-bc --epochs 20

    # Step 3: (Optional) Fine-tune with PPO
    ../retro_env/bin/python train_boss.py --train-ppo --load models/boss_bc.pth --steps 100000

    # Step 4: Run the bot
    ../retro_env/bin/python train_boss.py --play
"""

import os
import sys
import argparse
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import json

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

os.environ['SDL_VIDEODRIVER'] = 'x11'

import stable_retro as retro

# =============================================================================
# CONFIGURATION
# =============================================================================
class GameConfig:
    # -------------------------------------------------------------------------
    # Global Hyperparameters
    # -------------------------------------------------------------------------
    LEARNING_RATE = 1e-4
    ENT_COEF_START = 0.05
    ENT_COEF_END = 0.005
    BATCH_SIZE = 256
    N_EPOCHS = 4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    
    # -------------------------------------------------------------------------
    # State Definitions
    # -------------------------------------------------------------------------
    STATES = {
        "BossTorizo": {
            "reward_scale": 1.0,
            "opener": True,
            "rewards": {
                "kill": 5000.0,
                "damage_dealt": 30.0,
                "damage_taken": -80.0,
                "health_recovered": 50.0,
                "death": -500.0,
                "item_pickup": 40.0,
                "orb_hit": 50.0,
                "shoot_facing": 5.0,
                "shoot_away": -8.0,
                "headshot": 15.0,
                "generic_hit": 5.0,
                "rapid_fire": 3.0,
                "no_shoot": -5.0,
                "still": -1.0,
                "missile_waste": -20.0,
                "time": -0.05
            }
        },
        "ZebesStart": {
            "reward_scale": 0.5,
            "max_steps": 2048, # Short episodes to encourage rapid retry of first traversal
            "opener": False,
            "rewards": {
                "kill": 100.0,
                "damage_dealt": 5.0,
                "damage_taken": -5.0,
                "health_recovered": 100.0,
                "death": -50.0,
                "item_pickup": 100.0,
                "orb_hit": 10.0,
                "shoot_facing": 1.0,
                "shoot_away": -1.0,
                "headshot": 0.0,
                "generic_hit": 1.0,
                "rapid_fire": 0.0,
                "no_shoot": 0.0,
                "still": -5.0,
                "missile_waste": 0.0,
                "time": -0.01,
                # Navigation Specifics
                "room_change": 500.0, # Big reward for leaving room
                "door_open": 50.0,    # Reward for opening door (detected by map/event bit hopefully, or just proximity + shot?)
                "progress_right": 2.0 # Reward for moving right (Samus X)
            }
        }
    }

    @staticmethod
    def get(state_name):
        return GameConfig.STATES.get(state_name, GameConfig.STATES["BossTorizo"])

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTEGRATION_PATH = os.path.join(SCRIPT_DIR, "custom_integrations")
retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

# Directories
RECORDING_DIR = os.path.join(SCRIPT_DIR, "recordings")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(LOG_DIR, exist_ok=True)
DATA_DIR = os.path.join(SCRIPT_DIR, "boss_data")
HOLD_DATA_PATH = os.path.join(DATA_DIR, "boss_hold_lengths.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# =============================================================================
# FRAME STACKING WRAPPER
# =============================================================================
class FrameStack(gym.Wrapper):
    """Stack n_frames grayscale frames for motion perception."""
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = None
        old_shape = env.observation_space.shape
        # Update observation space for stacked frames: (n_frames * 3, H/2, W/2)
        h, w = old_shape[0] // 2, old_shape[1] // 2
        new_shape = (n_frames * 3, h, w)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=new_shape, dtype=np.uint8
        )

    def _get_frame(self, obs):
        """Convert observation to RGB frame."""
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            # RGB (H, W, 3) -> RGB downsampled (112, 128, 3)
            return obs[::2, ::2, :].transpose(2, 0, 1) # (3, 112, 128)
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if isinstance(obs, np.ndarray) and len(obs.shape) == 3 and obs.shape[2] == 3:
            info['rgb_frame'] = obs
        frame = self._get_frame(obs)
        self.frames = np.concatenate([frame] * self.n_frames, axis=0) # (12, 112, 128)
        return self.frames.copy(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(obs, np.ndarray) and len(obs.shape) == 3 and obs.shape[2] == 3:
            info['rgb_frame'] = obs
        frame = self._get_frame(obs)
        # Shift and replace for (C*N, H, W)
        self.frames = np.roll(self.frames, shift=-3, axis=0)
        self.frames[-3:] = frame
        return self.frames.copy(), reward, terminated, truncated, info

# =============================================================================
# STATE AUGMENTED OBSERVATION WRAPPER
# =============================================================================
class StateAugmented(gym.Wrapper):
    """Add normalized game state to info for auxiliary learning."""
    def __init__(self, env):
        super().__init__(env)
        # Keep observation space unchanged, state goes in info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Add normalized state vector to info
        boss_hp = info.get('boss_hp', 0) or info.get('enemy0_hp', 0)
        samus_hp = info.get('health', 99)
        samus_x = info.get('samus_x', 0)
        samus_y = info.get('samus_y', 0)

        info['state_vec'] = np.array([
            boss_hp / 800.0,      # Normalize boss HP
            samus_hp / 99.0,      # Normalize Samus HP
            samus_x / 256.0,      # Normalize X position
            samus_y / 256.0,      # Normalize Y position
        ], dtype=np.float32)

        return obs, reward, terminated, truncated, info

# =============================================================================
# WRAPPERS
# =============================================================================
class RoomDetector(gym.Wrapper):
    """Logs and provides access to current Room ID."""
    def __init__(self, env):
        super().__init__(env)
        self.current_room = None
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Room ID is expected to be in info from data.json
        self.current_room = info.get('room_id', 0)
        return obs, reward, terminated, truncated, info

class GeneralReward(gym.Wrapper):
    """
    Generalized reward wrapper that adapts to the current state configuration.
    """
    def __init__(self, env, state_config):
        super().__init__(env)
        self.cfg = state_config["rewards"]
        self.scale = state_config.get("reward_scale", 1.0)
        
        # State tracking
        self.prev_boss_hp = None
        self.prev_samus_hp = None
        self.prev_missiles = None
        self.steps_no_shot = 0
        self.steps_still = 0
        self.prev_x = 0
        self.prev_y = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Initialize heuristics using info keys if available
        self.prev_boss_hp = info.get('enemy0_hp', 0)
        self.prev_samus_hp = info.get('health', 99)
        self.prev_missiles = info.get('missiles', 0)
        self.steps_no_shot = 0
        self.steps_still = 0
        self.prev_x = info.get('samus_x', 0)
        self.prev_y = info.get('samus_y', 0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # --- Extract State ---
        boss_hp = info.get('enemy0_hp', 0)
        # Fallback for some states where enemy might be different or not present
        if boss_hp == 0: boss_hp = info.get('enemy1_hp', 0)
        
        samus_hp = info.get('health', 0)
        missiles = info.get('missiles', 0)
        samus_x = info.get('samus_x', 0)
        samus_y = info.get('samus_y', 0)
        
        shaped_reward = 0.0
        
        # --- 1. Combat / Survival Rewards ---
        
        # Damage Dealt
        if self.prev_boss_hp is not None and boss_hp < self.prev_boss_hp and self.prev_boss_hp > 0:
            dmg = self.prev_boss_hp - boss_hp
            shaped_reward += dmg * self.cfg["damage_dealt"]
            info['total_damage_dealt'] = info.get('total_damage_dealt', 0) + dmg
        
        # Kill
        if self.prev_boss_hp is not None and self.prev_boss_hp > 0 and boss_hp == 0:
            shaped_reward += self.cfg["kill"]
            
        # Damage Taken
        if samus_hp < self.prev_samus_hp:
            dmg_taken = self.prev_samus_hp - samus_hp
            shaped_reward += dmg_taken * self.cfg["damage_taken"]
            
        # Health Recovered
        if samus_hp > self.prev_samus_hp:
            diff = samus_hp - self.prev_samus_hp
            shaped_reward += diff * self.cfg["health_recovered"]

        # Death
        if samus_hp <= 0:
            shaped_reward += self.cfg["death"]
            terminated = True
            
        # --- 2. Resources ---
        if missiles > self.prev_missiles:
            shaped_reward += self.cfg["item_pickup"]
            
        # --- 3. Positional / Action Heuristics ---
        
        samus_dx = samus_x - self.prev_x
        samus_dy = samus_y - self.prev_y
        
        # Check if shooting (Action index 9 is 'A' in our map, wait, let's check DISCRETE_ACTIONS or raw)
        # We assume discrete action input to this wrapper? No, this wrapper wraps the env, usually *before* DiscreteAction if we want raw keys,
        # but Stable Baselines passes discrete index.
        # Actually, let's look at where we put this wrapper. It's usually inner.
        # `play_bot` applies Wrappers then DiscreteAction.
        # `make_env` applies Wrappers then DiscreteAction.
        # So `step(action)` here receives the MultiBinary actions (list of 12 ints/bools).
        
        # SNES Layout: B, Y, Select, Start, Up, Down, Left, Right, A, X, L, R
        # Indices:     0, 1, 2,      3,     4,  5,    6,    7,     8, 9, 10, 11
        # Wait, check `retro.Actions.ALL` mapping or our local `check_buttons.py` output if we had it.
        # Standard Retro SNES: B, Y, Select, Start, Up, Down, Left, Right, A, X, L, R
        # Let's verify `DISCRETE_ACTIONS` in original file.
        # It used: [['L'], ['R'], ...] 
        # But here `action` is the *input* to the environment.
        # If we wrap `GeneralReward(env)`, `env` expects MultiBinary.
        # So `action` passed to `GeneralReward.step` depends on what's wrapping IT.
        # Usually: Model -> DiscreteWrapper -> GeneralReward -> RetroEnv
        # So `GeneralReward` sees `action` as Discrete Index? 
        # NO. Wrappers wrap from inside out.
        # Env = Retro
        # Env = GeneralReward(Env)
        # Env = DiscreteAction(Env)
        # Model calls DiscreteAction.step(int) -> converts to list -> calls GeneralReward.step(list)
        # So `action` here is a LIST (MultiBinary).
        
        # X is shoot (Index 9 normally for SNES in Retro? Let's check).
        # In `train_boss.py`:
        # ACTION_KEYS = ['B', 'Y', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'X', 'L', 'R']
        # index 9 is X.
        
        is_shooting = action[9] if len(action) > 9 else 0
        
        if is_shooting:
            self.steps_no_shot = 0
            
            # Heuristic: Shooting while facing boss
            # Need boss X.
            boss_x = info.get('enemy0_hp', 0) and info.get('enemy0_x', 0)
            if boss_x == 0: boss_x = info.get('enemy1_x', 0) # Flash back or other part
            
            if boss_x > 0:
                dx_to_boss = boss_x - samus_x
                # If boss is right (dx > 0) and we are pressing Right (idx 7) or not pressing Left
                # Simple check: Moving towards boss?
                if samus_dx > 0 and dx_to_boss > 0:
                    shaped_reward += self.cfg["shoot_facing"]
                elif samus_dx < 0 and dx_to_boss < 0:
                    shaped_reward += self.cfg["shoot_facing"]
                
                # Penalty for shooting away
                if (samus_dx < 0 and dx_to_boss > 0) or (samus_dx > 0 and dx_to_boss < 0):
                    shaped_reward += self.cfg["shoot_away"]

        else:
            self.steps_no_shot += 1
            if self.steps_no_shot > 60: # 1 second
                shaped_reward += self.cfg["no_shoot"]
                
        # Movement check
        if samus_dx == 0 and samus_dy == 0:
            self.steps_still += 1
            if self.steps_still > 30:
                shaped_reward += self.cfg["still"]
        else:
            self.steps_still = 0
            
        # Time penalty
        shaped_reward += self.cfg["time"]
        
        # Apply scaling
        final_reward = reward + (shaped_reward * self.scale)
        
        # State Update
        self.prev_boss_hp = boss_hp
        self.prev_samus_hp = samus_hp
        self.prev_missiles = missiles
        self.prev_x = samus_x
        self.prev_y = samus_y
        
        return obs, final_reward, terminated, truncated, info

# =============================================================================
# POLICY NETWORK
# =============================================================================
class BossPolicy(nn.Module):
    """CNN policy for boss fight - takes stacked grayscale frames, outputs button logits."""
    def __init__(self, input_shape=(112, 128), num_actions=12, n_frames=4):
        super(BossPolicy, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_frames * 3, 32, kernel_size=8, stride=4),  # 12 channels for RGB stack
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, n_frames * 3, input_shape[0], input_shape[1])
            self.feature_size = self.features(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = x.float() / 255.0
        x = self.features(x)
        x = self.fc(x)
        return x

# =============================================================================
# DATA EXTRACTION FROM BK2
# =============================================================================
def _sanitize_action_vec(action_vec):
    action = action_vec.copy()
    if action[_LEFT] and action[_RIGHT]:
        action[_LEFT] = 0
        action[_RIGHT] = 0
    if action[_UP] and action[_DOWN]:
        action[_UP] = 0
        action[_DOWN] = 0
    return action

def _build_action_lookup():
    lookup = {}
    for idx, mapping in enumerate(DISCRETE_ACTIONS):
        vec = np.zeros(12, dtype=np.int8)
        for button_idx, pressed in mapping.items():
            vec[button_idx] = pressed
        lookup[tuple(vec.tolist())] = idx
    return lookup

def _save_hold_lengths(hold_lengths):
    data = {str(k): v for k, v in hold_lengths.items()}
    with open(HOLD_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def _load_hold_lengths():
    if not os.path.exists(HOLD_DATA_PATH):
        return None
    with open(HOLD_DATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    hold_lengths = {}
    for key, vals in raw.items():
        try:
            idx = int(key)
        except ValueError:
            continue
        cleaned = [int(v) for v in vals if int(v) > 0]
        if cleaned:
            hold_lengths[idx] = cleaned
    return hold_lengths

def extract_demos():
    """Extract training data from all Torizo bk2 recordings."""
    # Find both old-style and new-style recordings
    bk2_files = glob.glob(os.path.join(RECORDING_DIR, "*Torizo*.bk2"))
    bk2_files += glob.glob(os.path.join(RECORDING_DIR, "boss_demo_*.bk2"))
    bk2_files = list(set(bk2_files))  # Remove duplicates

    if not bk2_files:
        print("No Torizo recordings found!")
        return

    print(f"Found {len(bk2_files)} Torizo recordings")

    all_obs = []
    all_acts = []
    discrete_obs = []
    discrete_acts = []
    hold_lengths = {i: [] for i in range(len(DISCRETE_ACTIONS))}
    action_lookup = _build_action_lookup()

    for bk2_path in bk2_files:
        print(f"\nProcessing: {os.path.basename(bk2_path)}")

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
        demo_obs = []
        demo_acts = []
        last_idx = None
        hold_len = 0

        while movie.step():
            # Get action from movie
            keys = []
            for p in range(movie.players):
                for i in range(env.num_buttons):
                    keys.append(int(movie.get_key(i, p)))

            action = np.array(keys[:12], dtype=np.int32)
            action = _sanitize_action_vec(action)

            # Preprocess observation (RGB, downsample)
            rgb_small = obs[::2, ::2, :].transpose(2, 0, 1) # (3, 112, 128)

            demo_obs.append(rgb_small)
            demo_acts.append(action)

            # Map to discrete action index for hold-length mining
            action_idx = action_lookup.get(tuple(action.tolist()))
            if action_idx is None:
                if last_idx is not None and hold_len > 0:
                    hold_lengths[last_idx].append(hold_len)
                last_idx = None
                hold_len = 0
            else:
                discrete_obs.append(rgb_small)
                discrete_acts.append(action_idx)
                if last_idx == action_idx:
                    hold_len += 1
                else:
                    if last_idx is not None and hold_len > 0:
                        hold_lengths[last_idx].append(hold_len)
                    last_idx = action_idx
                    hold_len = 1

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            frames += 1

            if terminated or truncated:
                print(f"  Episode ended at frame {frames}")
                break

        env.close()
        print(f"  Extracted {frames} frames")
        if last_idx is not None and hold_len > 0:
            hold_lengths[last_idx].append(hold_len)

        all_obs.extend(demo_obs)
        all_acts.extend(demo_acts)

    # Save as npz
    all_obs = np.array(all_obs)
    all_acts = np.array(all_acts)

    save_path = os.path.join(DATA_DIR, "boss_demos.npz")
    np.savez_compressed(save_path, obs=all_obs, acts=all_acts)
    print(f"\nSaved {len(all_obs)} frames to {save_path}")
    print(f"  Obs shape: {all_obs.shape}")
    print(f"  Acts shape: {all_acts.shape}")

    if discrete_obs and discrete_acts:
        discrete_path = os.path.join(DATA_DIR, "boss_demos_discrete.npz")
        np.savez_compressed(
            discrete_path,
            obs=np.array(discrete_obs),
            acts=np.array(discrete_acts, dtype=np.int64)
        )
        print(f"Saved discrete demo set to {discrete_path}")

    _save_hold_lengths(hold_lengths)
    print(f"Saved hold-length data to {HOLD_DATA_PATH}")

# =============================================================================
# BEHAVIORAL CLONING TRAINING
# =============================================================================
class DemoDataset(Dataset):
    """Dataset with frame stacking for BC training."""
    def __init__(self, npz_path, n_frames=4):
        data = np.load(npz_path)
        raw_obs = data['obs']  # Shape: (N, H, W)
        self.acts = data['acts'].astype(np.float32)
        self.n_frames = n_frames

        # Create frame-stacked observations
        # Pad beginning with repeated first frame
        n_samples = len(raw_obs)
        c, h, w = raw_obs.shape[1], raw_obs.shape[2], raw_obs.shape[3]
        self.obs = np.zeros((n_samples, n_frames * 3, h, w), dtype=np.uint8)

        for i in range(n_samples):
            for j in range(n_frames):
                # Get frame index, clamped to valid range
                frame_idx = max(0, i - (n_frames - 1 - j))
                self.obs[i, j*3 : (j+1)*3] = raw_obs[frame_idx]

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.acts[idx]

def train_bc(epochs=20, n_frames=4):
    """Train behavioral cloning model on extracted demos with frame stacking."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_path = os.path.join(DATA_DIR, "boss_demos.npz")
    if not os.path.exists(data_path):
        print(f"No training data found at {data_path}")
        print("Run with --extract first!")
        return

    print(f"Loading data from {data_path}...")
    dataset = DemoDataset(data_path, n_frames=n_frames)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    input_shape = (dataset[0][0].shape[1], dataset[0][0].shape[2])
    print(f"Input shape: {input_shape}, Frames stacked: {n_frames}, Total samples: {len(dataset)}")

    policy = BossPolicy(input_shape, 12, n_frames=n_frames).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    print(f"\nTraining for {epochs} epochs...")
    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0
        policy.train()

        for obs, acts in dataloader:
            obs, acts = obs.to(device), acts.to(device)

            optimizer.zero_grad()
            logits = policy(obs)
            loss = criterion(logits, acts)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(MODEL_DIR, "boss_bc.pth")
            torch.save(policy.state_dict(), save_path)
            print(f"  Saved best model (loss={best_loss:.4f})")

    print(f"\nTraining complete! Best loss: {best_loss:.4f}")

# =============================================================================
# BOSS FIGHT REWARD WRAPPER (for PPO fine-tuning)
# =============================================================================
class BossReward(gym.Wrapper):
    """Custom rewards for Bomb Torizo boss fight.

    Reward structure (rebalanced for clearer gradients):
    - PRIMARY: Damage dealt to boss (+20/point) - consistent progress
    - PRIMARY: Kill bonus (+5000) - terminal goal
    - SECONDARY: Damage taken (-80/point) - strong survival incentive (1 HP lost ~= 4 HP dealt)
    - SECONDARY: Health recovered (+50/point) - incentive to farm drops (yellow orbs)
    - SECONDARY: Death penalty (-500) - terminal penalty
    - SHAPING: Movement/combat behavior bonuses (small, consolidated)
    """
    def __init__(self, env):
        super().__init__(env)
        self.prev_boss_hp = None
        self.prev_samus_hp = None
        self.prev_samus_pos = None
        self.still_frames = 0
        self.frames_since_shot = 0
        self.total_damage_dealt = 0
        self.episode_reward = 0
        self.boss_triggered = False
        self.frame_count = 0
        self.prev_missiles = 0
        self.prev_enemy1_hp = 0
        self.prev_enemy2_hp = 0

    def reset(self, **kwargs):
        self.prev_boss_hp = None
        self.prev_samus_hp = None
        self.prev_samus_pos = None
        self.still_frames = 0
        self.frames_since_shot = 0
        self.total_damage_dealt = 0
        self.episode_reward = 0
        self.boss_triggered = False
        self.frame_count = 0
        self.prev_missiles = 0
        self.prev_enemy1_hp = 0
        self.prev_enemy2_hp = 0
        obs, info = self.env.reset(**kwargs)
        info['total_damage_dealt'] = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frame_count += 1

        # Get current values
        boss_hp = info.get('boss_hp', 0) or info.get('enemy0_hp', 0)
        samus_hp = info.get('health', 99)
        missiles = info.get('missiles', 0)
        enemy1_hp = info.get('enemy1_hp', 0)
        enemy2_hp = info.get('enemy2_hp', 0)
        selected_item = info.get('selected_item', 0) # 1 = Missiles

        # Detect boss trigger (HP goes from 0 to 800)
        if not self.boss_triggered and boss_hp >= 800:
            self.boss_triggered = True
            print(f"  Boss triggered at frame {self.frame_count}!")

        # Debug output every 100 frames
        if self.frame_count % 100 == 0 and self.frame_count > 0:
            missiles_dbg = info.get('missiles', 0)
            selected = info.get('selected_item', -1)
            samus_x_dbg = info.get('samus_x', 0)
            boss_x_dbg = info.get('enemy0_x', 0) or info.get('boss_x', 0)
            print(f"  [F{self.frame_count}] Boss:{boss_hp} Samus:{samus_hp} Missiles:{missiles_dbg} Selected:{selected} "
                  f"SamusX:{samus_x_dbg} BossX:{boss_x_dbg}")

        boss_active = self.boss_triggered or boss_hp > 0
        shaped_reward = 0.0

        # Initialize on first step
        if self.prev_boss_hp is None:
            self.prev_boss_hp = boss_hp
            self.prev_samus_hp = samus_hp
            self.prev_samus_pos = (info.get('samus_x', 0), info.get('samus_y', 0))
            self.prev_missiles = missiles
            self.prev_enemy1_hp = enemy1_hp
            self.prev_enemy2_hp = enemy2_hp

        # Track shooting (action is now discrete index, need to check mapped action)
        shoot_actions = {2, 3, 4, 5, 6, 7, 8, 15, 18, 19, 20}
        # Actions that aim UP while shooting (critical for hitting boss head)
        aim_up_shoot_actions = {5, 6, 7, 18}  # UP+X, UP+LEFT+X, UP+RIGHT+X, JUMP+UP+X

        is_shooting = action in shoot_actions if isinstance(action, (int, np.integer)) else action[9] == 1
        is_aiming_up_shoot = action in aim_up_shoot_actions if isinstance(action, (int, np.integer)) else False

        if is_shooting:
            self.frames_since_shot = 0
        else:
            self.frames_since_shot += 1

        # =====================================================================
        # PRIMARY REWARDS - Core objectives (ATTACK FOCUSED)
        # =====================================================================

        # MASSIVE reward for damaging boss - this is THE goal
        if boss_hp < self.prev_boss_hp and self.prev_boss_hp > 0:
            damage = self.prev_boss_hp - boss_hp
            shaped_reward += damage * BossFightConfig.REWARD_DAMAGE_DEALT
            self.total_damage_dealt += damage
            print(f"  HIT! Dealt {damage} damage, total: {self.total_damage_dealt}")

        # Bonus for killing boss
        if self.prev_boss_hp > 0 and boss_hp == 0:
            shaped_reward += BossFightConfig.REWARD_BOSS_KILL
            print("BOSS DEFEATED!")
            terminated = True

        # =====================================================================
        # SECONDARY REWARDS - Survival (reduced importance)
        # =====================================================================

        # Heavy penalty for taking damage
        if samus_hp < self.prev_samus_hp:
            damage_taken = self.prev_samus_hp - samus_hp
            shaped_reward += damage_taken * BossFightConfig.REWARD_DAMAGE_TAKEN # Negative value
            print(f"  OUCH! Took {damage_taken} damage!")

        # NEW: Reward for RECOVERING health (farming drops)
        elif samus_hp > self.prev_samus_hp:
            health_gained = samus_hp - self.prev_samus_hp
            shaped_reward += health_gained * BossFightConfig.REWARD_HEALTH_RECOVERED
            print(f"  RECOVERED! Gained {health_gained} HP!")

        # Death penalty (moderate - dying while attacking is okay)
        if samus_hp <= 0:
            # Bonus if we dealt damage before dying to encourage contact
            if self.total_damage_dealt > 0:
                shaped_reward += self.total_damage_dealt * 10.0
            shaped_reward += BossFightConfig.REWARD_DEATH
            terminated = True

        # ---------------------------------------------------------------------
        # 6. Navigation / Exploration
        # ---------------------------------------------------------------------
        # Room Change Reward
        current_room = info.get('room_id', 0)
        # Initialize prev_room on first step
        if not hasattr(self, 'prev_room') or self.prev_room is None:
            self.prev_room = current_room

        if current_room != self.prev_room and self.prev_room != 0:
            shaped_reward += self.cfg.get("room_change", 0.0)
            print(f"  ROOM CHANGE! {self.prev_room} -> {current_room}")
        
        self.prev_room = current_room

        # X-Progress Reward (Simple right-moving heuristic for ZebesStart)
        if self.cfg.get("progress_right", 0.0) > 0:
            dx = samus_x - self.prev_x
            if dx > 0:
                 shaped_reward += dx * self.cfg["progress_right"]

        # ---------------------------------------------------------------------
        # 7. Shaping Rewards - Behavior guidance
        # ---------------------------------------------------------------------

        # Movement tracking
        samus_x = info.get('samus_x', 0)
        samus_y = info.get('samus_y', 0)
        curr_pos = (samus_x, samus_y)
        vel_y = info.get('velocity_y', 0)
        vel_x = info.get('velocity_x', 0)
        boss_x = info.get('enemy0_x', 0) or info.get('boss_x', 128)  # Default to center

        if curr_pos == self.prev_samus_pos:
            self.still_frames += 1
        else:
            self.still_frames = 0
            self.prev_samus_pos = curr_pos

        if boss_active:
            distance_to_boss = abs(samus_x - boss_x)
            boss_on_right = boss_x > samus_x

            # Facing direction check (heuristic based on actions)
            is_moving_right = action[7] == 1
            is_moving_left = action[6] == 1
            
            # SHOOTING DIRECTION REWARDS
            if is_shooting:
                # Reward for facing/moving towards boss while shooting
                if (boss_on_right and is_moving_right) or (not boss_on_right and is_moving_left):
                    shaped_reward += BossFightConfig.REWARD_SHOOT_FACING
                # Penalty for facing/moving away from boss while shooting
                elif (boss_on_right and is_moving_left) or (not boss_on_right and is_moving_right):
                    shaped_reward += BossFightConfig.PENALTY_SHOOT_AWAY

                # Reward for shooting while close to boss and aiming up
                if is_aiming_up_shoot:
                    if distance_to_boss < 80:
                        shaped_reward += BossFightConfig.REWARD_HEADSHOT_RANGE

                # Moderate reward for any shooting (baseline)
                if distance_to_boss < 80:
                    shaped_reward += BossFightConfig.REWARD_GENERIC_HIT
                else:
                    shaped_reward += 2.0

            # RAPID FIRE REWARD
            # Give a small bonus for shooting as soon as cooldown allows
            shot_cooldown = info.get('shot_cooldown', 0)
            if shot_cooldown == 0 and is_shooting:
                 shaped_reward += BossFightConfig.REWARD_RAPID_FIRE

            # Strong penalty for NOT shooting (make shooting mandatory)
            if self.frames_since_shot > 10:
                shaped_reward += BossFightConfig.PENALTY_NO_SHOOT

            # Movement/Dodge rewards
            # Stillness penalty
            if self.still_frames > 60:
                shaped_reward += BossFightConfig.PENALTY_STILL

        # =====================================================================
        # MISSILE & ORB REWARDS
        # =====================================================================
        
        # 1. Item Pickup Reward (Missiles)
        if missiles > self.prev_missiles:
            diff = missiles - self.prev_missiles
            shaped_reward += diff * BossFightConfig.REWARD_ITEM_PICKUP
            print(f"  PICKUP! Gained {diff} missiles!")

        # 2. Missile Waste Penalty
        is_firing_missile = (selected_item == 1 and is_shooting)
        if is_firing_missile:
            if not self.boss_triggered:
                shaped_reward -= 30.0 # Penalty for shooting statue
            else:
                # Penalty for firing missiles in wrong direction
                boss_on_right = boss_x > samus_x
                if (boss_on_right and vel_x < -5) or (not boss_on_right and vel_x > 5):
                    shaped_reward += BossFightConfig.PENALTY_MISSILE_WASTE
                    print("  MISSILE WASTED! Firing away from boss.")

        # 3. Orb Kill Reward (Encourage drop farming)
        if enemy1_hp < self.prev_enemy1_hp and self.prev_enemy1_hp > 0:
            shaped_reward += BossFightConfig.REWARD_ORB_HIT
            print(f"  ORB HIT! (Slot 1 HP: {enemy1_hp})")
        if enemy2_hp < self.prev_enemy2_hp and self.prev_enemy2_hp > 0:
            shaped_reward += BossFightConfig.REWARD_ORB_HIT
            print(f"  ORB HIT! (Slot 2 HP: {enemy2_hp})")

        # Small time penalty to encourage speed
        shaped_reward += BossFightConfig.PENALTY_TIME

        self.prev_boss_hp = boss_hp
        self.prev_samus_hp = samus_hp
        self.prev_missiles = missiles
        self.prev_enemy1_hp = enemy1_hp
        self.prev_enemy2_hp = enemy2_hp

        reward += shaped_reward
        self.episode_reward += reward

        info['shaped_reward'] = shaped_reward
        info['total_damage_dealt'] = self.total_damage_dealt

        return obs, reward, terminated, truncated, info

# =============================================================================
# ACTION SANITIZER
# =============================================================================
class SanitizeAction(gym.ActionWrapper):
    """Resolve contradictory D-pad inputs."""
    def action(self, action):
        # Left (6) and Right (7) cannot both be pressed
        if action[6] and action[7]:
            action[6] = 0
            action[7] = 0
        # Up (4) and Down (5) cannot both be pressed
        if action[4] and action[5]:
            action[4] = 0
            action[5] = 0
        return action

# =============================================================================
# DISCRETE ACTION SET (PPO)
# =============================================================================
_B = 0
_Y = 1
_SELECT = 2
_START = 3
_UP = 4
_DOWN = 5
_LEFT = 6
_RIGHT = 7
_A = 8
_X = 9
_L = 10
_R = 11

DISCRETE_ACTIONS = [
    # Movement + core combat
    # NOTE: X is shoot button, Y is item cancel in Super Metroid default config
    {_LEFT: 1},
    {_RIGHT: 1},
    {_LEFT: 1, _X: 1},           # Move left + shoot
    {_RIGHT: 1, _X: 1},          # Move right + shoot
    {_X: 1},                     # Shoot
    {_UP: 1, _X: 1},             # Aim up + shoot
    {_UP: 1, _LEFT: 1, _X: 1},   # Aim up-left + shoot
    {_UP: 1, _RIGHT: 1, _X: 1},  # Aim up-right + shoot
    {_A: 1, _X: 1},              # Jump + shoot
    {_A: 1},                     # Jump
    {_A: 1, _LEFT: 1},           # Jump left
    {_A: 1, _RIGHT: 1},          # Jump right
    {_B: 1, _LEFT: 1},           # Run left
    {_B: 1, _RIGHT: 1},          # Run right
    # Crouch/dodge actions
    {_DOWN: 1},                  # Crouch
    {_DOWN: 1, _X: 1},           # Crouch + shoot
    {_DOWN: 1, _LEFT: 1},        # Crouch left (aim down-left)
    {_DOWN: 1, _RIGHT: 1},       # Crouch right (aim down-right)
    # Jump + aim up combos (hit boss head while airborne)
    {_A: 1, _UP: 1, _X: 1},      # Jump + aim up + shoot
    {_A: 1, _LEFT: 1, _X: 1},    # Jump left + shoot
    {_A: 1, _RIGHT: 1, _X: 1},   # Jump right + shoot
    {},                          # No-op
]

class DiscreteAction(gym.ActionWrapper):
    """Map discrete action index to a MultiBinary action array."""
    def __init__(self, env, action_map):
        super().__init__(env)
        self.action_map = action_map
        self.action_space = gym.spaces.Discrete(len(action_map))
        self._action_count = 0

    def action(self, action):
        idx = int(action)
        idx = max(0, min(idx, len(self.action_map) - 1))
        mapped = np.zeros(12, dtype=np.int8)
        for button_idx, pressed in self.action_map[idx].items():
            mapped[button_idx] = pressed

        # Debug: print action every 50 steps
        self._action_count += 1
        if self._action_count % 50 == 0:
            buttons = "B Y SEL STA U D L R A X L R".split()
            active = [buttons[i] for i in range(12) if mapped[i]]
            # print(f"    Action {idx}: {active}")

        return mapped

# =============================================================================
# FORCE MISSILE SELECTION
# =============================================================================
class ForceMissiles(gym.Wrapper):
    """Force missiles selected when available to enable damage output."""
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._force_missiles()
        return obs, info

    def step(self, action):
        # Force missiles BEFORE the step so shots use missiles
        self._force_missiles()
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Also force after to maintain selection
        self._force_missiles()
        return obs, reward, terminated, truncated, info

    def _force_missiles(self):
        try:
            # Check current missile count
            missiles = self.unwrapped.data.lookup_value('missiles')
            if missiles > 0:
                # Force missiles to be selected
                self.unwrapped.data.set_value('selected_item', 1)
            else:
                # Deselect if none left to allow normal beam shooting
                # Only force 0 if it was previously 1 to avoid fighting the user
                if self.unwrapped.data.lookup_value('selected_item') == 1:
                    self.unwrapped.data.set_value('selected_item', 0)
        except Exception:
            pass

# =============================================================================
# SCRIPTED OPENER + ACTION HOLD REPEAT
# =============================================================================
class ScriptedOpener(gym.Wrapper):
    """Override actions for an opening window to seed damage behavior.

    Aggressive attack pattern - move toward boss and shoot constantly.
    """
    def __init__(self, env, opener_steps=240):
        super().__init__(env)
        self.opener_steps = opener_steps
        self._step_count = 0
        # Aggressive attack pattern - mostly aim up + shoot while moving right
        # 7 = UP+RIGHT+Y (move right while shooting up) - primary attack
        # 5 = UP+Y (shoot straight up)
        # 18 = JUMP+UP+Y (jump while shooting up)
        self.attack_pattern = [7, 7, 7, 5, 7, 7, 18, 5, 7, 7]

    def reset(self, **kwargs):
        self._step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        if self._step_count < self.opener_steps:
            # Cycle through attack pattern every 8 frames (faster cycling)
            pattern_idx = (self._step_count // 8) % len(self.attack_pattern)
            action = self.attack_pattern[pattern_idx]
        self._step_count += 1
        return self.env.step(action)

class ActionHoldRepeat(gym.Wrapper):
    """Repeat discrete actions for a sampled hold length."""
    def __init__(self, env, hold_sampler):
        super().__init__(env)
        self.hold_sampler = hold_sampler
        self.render_fn = None

    def step(self, action):
        repeat = max(1, int(self.hold_sampler(action)))
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = None
        for i in range(repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if self.render_fn:
                self.render_fn(obs, info)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

# =============================================================================
# PPO FINE-TUNING
# =============================================================================
if SB3_AVAILABLE:
    class BossCNNExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
            super(BossCNNExtractor, self).__init__(observation_space, features_dim)
            n_input_channels = observation_space.shape[0]
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4), # 12 channels for RGB stack
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
# ENVIRONMENT FACTORY
# =============================================================================
def make_env(state="BossTorizo"):
    config = GameConfig.get(state)
    env = retro.make(
        game="SuperMetroid-Snes",
        state=state,
        use_restricted_actions=retro.Actions.ALL,
        render_mode='rgb_array'
    )
    
    # Add TimeLimit wrapper to ensure resets
    max_steps = config.get("max_steps", 4000)
    env = TimeLimit(env, max_episode_steps=max_steps)
    
    env = SanitizeAction(env)
    env = ForceMissiles(env)
    env = StateAugmented(env)
    env = GeneralReward(env, config)
    env = RoomDetector(env)
    
    env = DiscreteAction(env, DISCRETE_ACTIONS)
    
    # Custom hold lengths
    hold_lengths = _load_hold_lengths()
    if not hold_lengths:
        hold_lengths = {}
        
    def hold_sampler(action_idx):
        holds = hold_lengths.get(int(action_idx)) if hold_lengths else None
        if holds:
            return np.random.choice(holds)
        return np.random.randint(2, 8) 

    if config.get("opener", False):
        env = ScriptedOpener(env, opener_steps=240)
        
    env = ActionHoldRepeat(env, hold_sampler)
    env = FrameStack(env, n_frames=4)
    
    return env

class EntropyScheduleCallback(BaseCallback):
    """
    Custom callback to schedule entropy coefficient decay.
    """
    def __init__(self, initial_ent: float, final_ent: float, total_steps: int, verbose=0):
        super().__init__(verbose)
        self.initial_ent = initial_ent
        self.final_ent = final_ent
        self.total_steps = total_steps

    def _on_step(self) -> bool:
        # progress_remaining goes from 1.0 (start) to 0.0 (end)
        # We want to go from initial to final
        # But num_timesteps goes 0 -> total_steps
        current_step = self.num_timesteps
        progress = min(1.0, current_step / self.total_steps)
        
        new_ent = self.initial_ent + (self.final_ent - self.initial_ent) * progress
        
        self.model.ent_coef = new_ent
        
        # Log to SB3 logger
        if self.logger is not None:
             self.logger.record("train/ent_coef", new_ent)
             
        if current_step % 10000 == 0:
            print(f"  Step {current_step}: ent_coef = {new_ent:.5f}")
            
        return True


# =============================================================================
# PPO TRAINING
# =============================================================================
def train_ppo(state="BossTorizo", steps=None, load_path=None):
    config = GameConfig.get(state)
    steps = steps if steps else config["max_steps"]
    
    # Wrap make_env for VecEnv
    def env_fn():
        return make_env(state=state)

    # Vectorize
    env = DummyVecEnv([env_fn])
    env = VecMonitor(env, filename=os.path.join(LOG_DIR, f"monitor_{state}"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resume_from_ppo = bool(load_path and load_path.endswith(".zip") and os.path.exists(load_path))

    if resume_from_ppo:
        print(f"Loading model from {load_path}")
        model = PPO.load(load_path, env=env, device=device)
    else:
        print("Creating new PPO model")
        policy_kwargs = dict(
            features_extractor_class=BossCNNExtractor,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=dict(pi=[256], vf=[256])  # Added hidden layers for capacity
        )
        
        initial_ent_coef = GameConfig.ENT_COEF_START
        final_ent_coef = GameConfig.ENT_COEF_END
        
        model = PPO("CnnPolicy", env, verbose=1, device=device, policy_kwargs=policy_kwargs,
                    learning_rate=GameConfig.LEARNING_RATE, ent_coef=initial_ent_coef, 
                    n_steps=2048, batch_size=GameConfig.BATCH_SIZE, 
                    n_epochs=GameConfig.N_EPOCHS, clip_range=GameConfig.CLIP_RANGE, 
                    gae_lambda=GameConfig.GAE_LAMBDA, gamma=GameConfig.GAMMA)

    if load_path and load_path.endswith(".pth") and os.path.exists(load_path):
        print(f"Loading BC feature weights from {load_path}...")
        bc_state_dict = torch.load(load_path, map_location=device)
        
        sb3_state_dict = model.policy.state_dict()
        
        mapping = {
            "features.0.weight": "features_extractor.cnn.0.weight",
            "features.0.bias": "features_extractor.cnn.0.bias",
            "features.2.weight": "features_extractor.cnn.2.weight",
            "features.2.bias": "features_extractor.cnn.2.bias",
            "features.4.weight": "features_extractor.cnn.4.weight",
            "features.4.bias": "features_extractor.cnn.4.bias",
            "fc.0.weight": "features_extractor.linear.0.weight",
            "fc.0.bias": "features_extractor.linear.0.bias",
        }
        
        loaded_count = 0
        for bc_key, sb3_key in mapping.items():
            if bc_key in bc_state_dict and sb3_key in sb3_state_dict:
                sb3_state_dict[sb3_key] = bc_state_dict[bc_key]
                loaded_count += 1
        
        model.policy.load_state_dict(sb3_state_dict)
        print(f"Successfully mapped {loaded_count} feature layers from BC to PPO policy.")

    # Create entropy schedule callback
    ent_callback = EntropyScheduleCallback(initial_ent_coef, final_ent_coef, steps)

    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=MODEL_DIR,
        name_prefix="boss_ppo_checkpoint"
    )

    print(f"Starting PPO fine-tuning for {steps} steps...")
    print(f"  Entropy schedule: {initial_ent_coef} -> {final_ent_coef}")
    print(f"  Frame stack: 4 frames")
    print(f"  Action space: {len(DISCRETE_ACTIONS)} discrete actions")

    model.learn(
        total_timesteps=steps, 
        reset_num_timesteps=not resume_from_ppo, 
        callback=[ent_callback, checkpoint_callback]
    )

    save_path = os.path.join(MODEL_DIR, "boss_ppo.zip")
    model.save(save_path)
    print(f"PPO model saved to {save_path}")
    env.close()

# =============================================================================
# PLAY / RUN BOT
# =============================================================================
def play_bot(model_path=None, state="BossTorizo", record=False):
    """Run the trained bot against a specific state."""
    import pygame

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = GameConfig.get(state)

    # Find model
    if model_path is None:
        ppo_path = os.path.join(MODEL_DIR, f"ppo_{state}_final.zip")
        if os.path.exists(ppo_path):
            model_path = ppo_path
        else:
             # Fallback to generic
            model_path = os.path.join(MODEL_DIR, "boss_ppo.zip")
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    print(f"Loading model: {model_path} for state {state}")
    is_ppo = model_path.endswith(".zip")

    # Create environment
    env = retro.make(
        game="SuperMetroid-Snes",
        state=state,
        use_restricted_actions=retro.Actions.ALL,
        render_mode='rgb_array',
        record='.' if record else False
    )
    raw_env = env
    env = SanitizeAction(env)
    env = ForceMissiles(env)
    env = StateAugmented(env)
    env = GeneralReward(env, config)
    env = RoomDetector(env)

    if is_ppo:
        env = DiscreteAction(env, DISCRETE_ACTIONS)
        hold_lengths = _load_hold_lengths()
        if not hold_lengths:
            hold_lengths = {}
        def hold_sampler(action_idx):
            holds = hold_lengths.get(int(action_idx))
            if holds:
                return np.random.choice(holds)
            return np.random.randint(2, 5)

        if config.get("opener", False):
            env = ScriptedOpener(env, opener_steps=240)
            
        env = ActionHoldRepeat(env, hold_sampler)
        env = FrameStack(env, n_frames=4)
    # ... rest of play_bot is display mostly ...


    # Init pygame
    pygame.init()
    obs, info = env.reset()
    # For PPO with frame stacking, display size is based on single frame
    if is_ppo:
        display_h, display_w = 112 * 2, 128 * 2
    else:
        display_h, display_w = obs.shape[0] * 2, obs.shape[1] * 2
    screen = pygame.display.set_mode((display_w, display_h))
    pygame.display.set_caption("Bomb Torizo - Boss Fight Bot")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('monospace', 16)

    # Load model
    if is_ppo:
        model = PPO.load(model_path, device=device)
    else:
        gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
        gray_small = gray[::2, ::2]
        input_shape = (gray_small.shape[0], gray_small.shape[1])

        # BC model was trained with 1 frame, whereas PPO uses 4
        policy = BossPolicy(input_shape, 12, n_frames=1).to(device)
        policy.load_state_dict(torch.load(model_path, map_location=device))
        policy.eval()

    # Define render function
    def render_frame(f_obs, f_info):
        nonlocal frame
        # Use RGB frame from info if available (PPO/FrameStack case)
        # Otherwise use the observation itself (BC case)
        rgb = f_info.get('rgb_frame')
        if rgb is None:
            if len(f_obs.shape) == 3 and f_obs.shape[2] == 3:
                rgb = f_obs
            elif len(f_obs.shape) == 3 and f_obs.shape[0] <= 4:
                # Might be (C, H, W), take first channel and make gray-RGB
                rgb = np.stack([f_obs[0]] * 3, axis=-1)
            else:
                # Assume grayscale (H, W) or (4, H, W)
                single = f_obs[-1] if len(f_obs.shape) == 3 else f_obs
                rgb = np.stack([single] * 3, axis=-1)

        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        scaled = pygame.transform.scale(surf, (display_w, display_h))
        screen.blit(scaled, (0, 0))

        # HUD
        # Check multiple potential HP keys
        b_hp = f_info.get('boss_hp', 0) or f_info.get('enemy0_hp', 0) or f_info.get('enemy1_hp', 0)
        s_hp = f_info.get('health', 0)
        dmg = f_info.get('total_damage_dealt', 0)

        texts = [
            f"Samus HP: {s_hp}",
            f"Boss HP: {b_hp}",
            f"Damage Dealt: {dmg}",
            f"Frame: {frame}",
            f"W/L: {wins}/{losses}"
        ]

        for i, txt in enumerate(texts):
            c = (0, 255, 0) if 'Samus' in txt else (255, 255, 0)
            text_surf = font.render(txt, True, c)
            screen.blit(text_surf, (10, 10 + i * 20))

        pygame.display.flip()
        clock.tick(60)
        frame += 1

    # Attach render function to ActionHoldRepeat if it exists in the stack
    curr = env
    while hasattr(curr, 'env'):
        if isinstance(curr, ActionHoldRepeat):
            curr.render_fn = render_frame
            break
        curr = curr.env

    running = True
    frame = 0
    wins = 0
    losses = 0

    print("\nStarting boss fight! Press ESC to quit.")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        if is_ppo:
            # PPO prediction - obs is already (4, 112, 128) from FrameStack
            action, _ = model.predict(obs, deterministic=False)
        else:
            # BC prediction
            # Preprocess
            gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
            gray_small = gray[::2, ::2]

            # Predict
            with torch.no_grad():
                t_obs = torch.from_numpy(gray_small).unsqueeze(0).unsqueeze(0).to(device)
                logits = policy(t_obs)
                probs = torch.sigmoid(logits)

                # Scale up probabilities to make actions more likely
                boosted_probs = torch.clamp(probs * 3.0, 0, 1)

                # Sample from boosted probabilities
                rand = torch.rand_like(boosted_probs)
                action = (rand < boosted_probs).cpu().numpy()[0].astype(int)

            # Heuristics (BC only for stability)
            # Always shoot during boss fight (X button = index 9)
            if frame % 3 == 0:
                action[9] = 1  # Shoot frequently

            # Bomb Torizo specific aiming: Need to aim up to hit the head
            # SNES Index: 4=Up
            action[4] = 1

            # Add some movement if model is too passive (BC model only)
            if frame % 30 < 15:
                action[7] = max(action[7], int(probs[0, 7] > 0.05))  # Right bias
            else:
                action[6] = max(action[6], int(probs[0, 6] > 0.05))  # Left bias

        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # If ActionHoldRepeat didn't render (e.g. BC mode or just one frame), render now
        # Actually, render_frame increments 'frame', so we should be careful.
        # If ActionHoldRepeat IS used, it will have rendered 'repeat' times.
        # If it's NOT used, we render here.
        # Check if ActionHoldRepeat is in the stack
        has_hold_wrapper = False
        target = env
        while hasattr(target, 'env'):
            if isinstance(target, ActionHoldRepeat):
                has_hold_wrapper = True
                break
            target = target.env
        
        if not has_hold_wrapper:
            render_frame(obs, info)

        # Debug output every 5 seconds (approx)
        if frame % 300 < 10 and frame > 0: # Small window to avoid missing it due to frame skips
            b_hp_dbg = info.get('enemy0_hp', 0) or info.get('enemy1_hp', 0)
            s_hp_dbg = info.get('health', 0)
            dmg_dbg = info.get('total_damage_dealt', 0)
            print(f"Frame {frame}: Boss HP={b_hp_dbg}, Samus HP={s_hp_dbg}, Damage={dmg_dbg}")

        if terminated or truncated:
            if info.get('health', 0) > 0:
                wins += 1
                print(f"WIN! (Frame {frame}, Boss HP={info.get('boss_hp', 0) or info.get('enemy0_hp', 0)})")
            else:
                losses += 1
                print(f"LOSS (Frame {frame}, Samus HP={info.get('health', 0)})")

            if record:
                print("Recording complete. Exiting...")
                running = False
            else:
                obs, info = env.reset()
                frame = 0
                time.sleep(1)

    env.close()
    pygame.quit()
    print(f"\nFinal record: {wins} wins, {losses} losses")

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Super Metroid General Trainer")
    parser.add_argument('--extract', action='store_true', help='Extract data from bk2 demos')
    parser.add_argument('--train-bc', action='store_true', help='Train BC model')
    parser.add_argument('--train-ppo', action='store_true', help='Train with PPO')
    parser.add_argument('--play', action='store_true', help='Run the bot')
    parser.add_argument('--record', action='store_true', help='Record .bk2 movie during play')
    parser.add_argument('--state', type=str, default="BossTorizo", help='Game state to train/play (e.g., BossTorizo, ZebesStart)')
    parser.add_argument('--epochs', type=int, default=20, help='BC training epochs')
    parser.add_argument('--steps', type=int, default=0, help='PPO training steps (0 = use config max)')
    parser.add_argument('--load', type=str, help='Model to load')

    args = parser.parse_args()

    if args.extract:
        extract_demos()
    elif args.train_bc:
        train_bc(epochs=args.epochs)
    elif args.train_ppo:
        if not SB3_AVAILABLE:
            print("Stable Baselines 3 not found.")
            return
        train_ppo(state=args.state, steps=args.steps if args.steps > 0 else None, load_path=args.load)
    elif args.play:
        play_bot(model_path=args.load, state=args.state, record=args.record)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

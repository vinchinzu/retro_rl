
import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import stable_retro as retro
from stable_baselines3.common.monitor import Monitor
import json

# =============================================================================
# CONFIGURATION
# =============================================================================
class MetroidConfig:
    # -------------------------------------------------------------------------
    # Hyperparameters
    # -------------------------------------------------------------------------
    TOTAL_STEPS = 500000             
    LEARNING_RATE = 1e-4             
    ENT_COEF_START = 0.02            
    ENT_COEF_END = 0.001             
    BATCH_SIZE = 512                 
    N_EPOCHS = 4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2

    # -------------------------------------------------------------------------
    # Rewards (Boss Scenario)
    # -------------------------------------------------------------------------
    REWARD_BOSS_KILL = 5000.0        
    REWARD_DAMAGE_DEALT = 30.0       
    REWARD_DAMAGE_TAKEN = -100.0     
    REWARD_HEALTH_RECOVERED = 60.0   
    REWARD_DEATH = -500.0            
    REWARD_ITEM_PICKUP = 300.0       
    REWARD_ORB_HIT = 50.0            
    
    # -------------------------------------------------------------------------
    # Rewards (Navigation Scenario)
    # -------------------------------------------------------------------------
    REWARD_NAV_PROGRESS = 10.0      # E.g. moving left/right towards goal
    REWARD_DOOR_ENTRY = 1000.0
    PENALTY_NAV_STILL = -1.0
    PENALTY_TIME = -0.1


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTEGRATION_PATH = os.path.join(SCRIPT_DIR, "custom_integrations")
retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

DATA_DIR = os.path.join(SCRIPT_DIR, "boss_data")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
try:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
except OSError:
    pass

# =============================================================================
# WRAPPERS
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

class StateAugmented(gym.Wrapper):
    """Add normalized game state to info for auxiliary learning."""
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        boss_hp = info.get('boss_hp', 0) or info.get('enemy0_hp', 0)
        samus_hp = info.get('health', 99)
        samus_x = info.get('samus_x', 0)
        samus_y = info.get('samus_y', 0)

        info['state_vec'] = np.array([
            boss_hp / 800.0,      
            samus_hp / 99.0,      
            samus_x / 256.0,      
            samus_y / 256.0,      
        ], dtype=np.float32)

        return obs, reward, terminated, truncated, info

class MetroidReward(gym.Wrapper):
    """Base Reward Wrapper. Dispatches to specific logic based on scenario."""
    def __init__(self, env, scenario='nav', delayed_termination=False):
        super().__init__(env)
        self.scenario = scenario
        self.delayed_termination = delayed_termination
        
        # Tracking
        self.prev_boss_hp = None
        self.prev_samus_hp = None
        self.prev_samus_x = None
        self.prev_missiles = 0
        self.total_damage_dealt = 0
        self.boss_killed = False
        self.frames_post_kill = 0
        self.frame_count = 0

    def reset(self, **kwargs):
        self.prev_boss_hp = None
        self.prev_samus_hp = None
        self.prev_samus_x = None
        self.prev_missiles = 0
        self.total_damage_dealt = 0
        self.boss_killed = False
        self.frames_post_kill = 0
        self.frame_count = 0
        obs, info = self.env.reset(**kwargs)
        info['total_damage_dealt'] = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frame_count += 1
        shaped_reward = 0.0

        # State extraction
        boss_hp = info.get('boss_hp', 0) or info.get('enemy0_hp', 0)
        samus_hp = info.get('health', 99)
        samus_x = info.get('samus_x', 0)
        missiles = info.get('missiles', 0)

        # Initialization
        if self.prev_boss_hp is None:
            self.prev_boss_hp = boss_hp
            self.prev_samus_hp = samus_hp
            self.prev_samus_x = samus_x
            self.prev_missiles = missiles

        # Dispatch
        if self.scenario == 'boss':
            shaped_reward, terminated = self._boss_reward(boss_hp, samus_hp, missiles, terminated)
        elif self.scenario == 'nav':
            shaped_reward, terminated = self._nav_reward(samus_x, samus_hp, terminated)

        # Common Time Penalty
        shaped_reward += MetroidConfig.PENALTY_TIME

        # Update State
        self.prev_boss_hp = boss_hp
        self.prev_samus_hp = samus_hp
        self.prev_samus_x = samus_x
        self.prev_missiles = missiles

        reward += shaped_reward
        info['shaped_reward'] = shaped_reward
        info['total_damage_dealt'] = self.total_damage_dealt
        return obs, reward, terminated, truncated, info

    def _boss_reward(self, boss_hp, samus_hp, missiles, terminated):
        r = 0.0
        # Damage
        if boss_hp < self.prev_boss_hp and self.prev_boss_hp > 0:
            dmg = self.prev_boss_hp - boss_hp
            r += dmg * MetroidConfig.REWARD_DAMAGE_DEALT
            self.total_damage_dealt += dmg
        
        # Kill
        if self.prev_boss_hp > 0 and boss_hp == 0:
            if not self.boss_killed:
                r += MetroidConfig.REWARD_BOSS_KILL
                # Speed Bonus
                frames_saved = max(0, 4000 - self.frame_count)
                r += frames_saved * MetroidConfig.REWARD_TIME_BONUS
                self.boss_killed = True
            if not self.delayed_termination:
                terminated = True
        
        # Delayed Term
        if self.boss_killed and self.delayed_termination:
            self.frames_post_kill += 1
            if self.frames_post_kill > 300: terminated = True
            else: terminated = False

        # Survival
        if samus_hp < self.prev_samus_hp:
            r += (self.prev_samus_hp - samus_hp) * MetroidConfig.REWARD_DAMAGE_TAKEN
        
        if samus_hp <= 0:
            r += MetroidConfig.REWARD_DEATH
            terminated = True
            
        # Items
        if missiles > self.prev_missiles:
            r += (missiles - self.prev_missiles) * MetroidConfig.REWARD_ITEM_PICKUP

        return r, terminated

    def _nav_reward(self, samus_x, samus_hp, terminated):
        r = 0.0
        # Simple Left Progress (Landing Site -> Left Door)
        # Landing Site Ship is around X=1200? Need to check.
        # Assuming we want to go LEFT (decreasing X).
        if samus_x < self.prev_samus_x:
            r += (self.prev_samus_x - samus_x) * 0.1 # Small progress reward
        
        # Death
        if samus_hp <= 0:
            r += MetroidConfig.REWARD_DEATH
            terminated = True
            
        return r, terminated

class SanitizeAction(gym.ActionWrapper):
    def action(self, action):
        if action[6] and action[7]: action[6] = 0; action[7] = 0
        if action[4] and action[5]: action[4] = 0; action[5] = 0
        return action

# Discrete Actions
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
    {_B: 1, _A: 1, _LEFT: 1, _X: 1}, {_B: 1, _A: 1, _RIGHT: 1, _X: 1},
    {_B: 1, _RIGHT: 1, _R: 1, _X: 1}, {_B: 1, _LEFT: 1, _R: 1, _X: 1},
    {_B: 1, _A: 1, _RIGHT: 1, _R: 1, _X: 1}, {_B: 1, _A: 1, _LEFT: 1, _R: 1, _X: 1},
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

class ForceMissiles(gym.Wrapper):
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
            if missiles > 0: self.unwrapped.data.set_value('selected_item', 1)
        except: pass

class ScriptedOpener(gym.Wrapper):
    def __init__(self, env, opener_steps=240):
        super().__init__(env)
        self.opener_steps = opener_steps
        self._step_count = 0
        self.attack_pattern = [7, 7, 7, 5, 7, 7, 18, 5, 7, 7]
    def reset(self, **kwargs):
        self._step_count = 0
        return self.env.reset(**kwargs)
    def step(self, action):
        if self._step_count < self.opener_steps:
            pattern_idx = (self._step_count // 8) % len(self.attack_pattern)
            action = self.attack_pattern[pattern_idx]
        self._step_count += 1
        return self.env.step(action)

class ActionHoldRepeat(gym.Wrapper):
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
            if self.render_fn: self.render_fn(obs, info)
            if terminated or truncated: break
        return obs, total_reward, terminated, truncated, info

def hold_sampler(action_idx):
    return 4

def make_metroid_env(state, scenario='nav', render_mode="rgb_array", delayed_termination=False):
    """
    Factory to create Metroid Env.
    Args:
        state: retro State string (e.g., 'BossTorizo', 'LandingSite')
        scenario: 'boss' (forced missiles, opener) or 'nav' (general traversal)
    """
    env = retro.make(
        game="SuperMetroid-Snes",
        state=state,
        use_restricted_actions=retro.Actions.ALL,
        render_mode=render_mode
    )
    env = SanitizeAction(env)
    
    if scenario == 'boss':
        env = ForceMissiles(env)
    
    env = StateAugmented(env)
    env = MetroidReward(env, scenario=scenario, delayed_termination=delayed_termination)
    env = DiscreteAction(env, DISCRETE_ACTIONS)
    
    if scenario == 'boss':
        env = ScriptedOpener(env, opener_steps=240)
        
    env = ActionHoldRepeat(env, hold_sampler)
    env = FrameStack(env, n_frames=4)
    
    monitor_path = os.path.join(DATA_DIR, f"{state}_{scenario}_monitor_{int(time.time())}.csv")
    env = Monitor(env, filename=monitor_path)
    return env

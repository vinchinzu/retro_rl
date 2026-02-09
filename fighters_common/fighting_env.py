"""
Gymnasium wrappers for fighting game RL training.

Provides observation preprocessing, reward shaping, discrete action mapping,
and environment factory for all supported fighting games.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import cv2
import gymnasium as gym
import numpy as np
import stable_retro as retro
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor


# ─────────────────────────────────────────────────────────────────────────────
# Discrete action map for fighting games
# ─────────────────────────────────────────────────────────────────────────────
_B, _Y, _SELECT, _START, _UP, _DOWN, _LEFT, _RIGHT, _A, _X, _L, _R = range(12)

# Comprehensive fighting game action set covering movement, attacks, and combos
FIGHTING_ACTIONS = [
    # Movement (0-5)
    {},                                     # 0: No-op
    {_LEFT: 1},                             # 1: Walk left
    {_RIGHT: 1},                            # 2: Walk right
    {_UP: 1},                               # 3: Jump
    {_DOWN: 1},                             # 4: Crouch
    {_DOWN: 1, _LEFT: 1},                   # 5: Crouch-block / down-back

    # Basic attacks (6-11)
    {_Y: 1},                                # 6: Light punch (Y)
    {_X: 1},                                # 7: Medium punch (X)
    {_B: 1},                                # 8: Light kick (B)
    {_A: 1},                                # 9: Medium kick (A)
    {_L: 1},                                # 10: Heavy punch (L)
    {_R: 1},                                # 11: Heavy kick (R)

    # Directional attacks (12-19)
    {_DOWN: 1, _Y: 1},                      # 12: Crouch light punch
    {_DOWN: 1, _X: 1},                      # 13: Crouch medium punch
    {_DOWN: 1, _B: 1},                      # 14: Crouch light kick (sweep)
    {_DOWN: 1, _A: 1},                      # 15: Crouch medium kick
    {_UP: 1, _Y: 1},                        # 16: Jump light punch
    {_UP: 1, _A: 1},                        # 17: Jump kick
    {_LEFT: 1, _Y: 1},                      # 18: Walk-back + light punch
    {_RIGHT: 1, _Y: 1},                     # 19: Walk-forward + light punch

    # Jump-direction attacks (20-23)
    {_UP: 1, _LEFT: 1},                     # 20: Jump back
    {_UP: 1, _RIGHT: 1},                    # 21: Jump forward
    {_UP: 1, _RIGHT: 1, _A: 1},             # 22: Jump-forward kick
    {_UP: 1, _LEFT: 1, _A: 1},              # 23: Jump-back kick

    # Blocking (24-25)
    {_LEFT: 1, _DOWN: 1},                   # 24: Low block (crouch-back)
    {_LEFT: 1},                             # 25: Stand block (same as walk-back)

    # Throw / close attacks (26-27)
    {_RIGHT: 1, _L: 1},                     # 26: Forward + heavy punch (throw)
    {_RIGHT: 1, _R: 1},                     # 27: Forward + heavy kick

    # Multi-button (28-31)
    {_Y: 1, _B: 1},                         # 28: LP + LK
    {_X: 1, _A: 1},                         # 29: MP + MK
    {_L: 1, _R: 1},                         # 30: HP + HK
    {_DOWN: 1, _RIGHT: 1, _Y: 1},           # 31: QCF + LP (fireball motion approx)
]

# MK-specific action set: dedicated Block button (X), tuned for Kano strategy.
#
# Key Kano moves (from competitive guide):
#   - Crouch LK: best poke in game, spam it. Counters jabs and grounded jump attacks.
#   - Uppercut (Crouch LP): longest range in game, perfect punisher.
#   - Roll/Cannonball: most powerful move. Tiny startup, punishes everything.
#     Input: Hold Block + circle motion (or Block+Back, Block+Down, Block+Forward).
#   - Knife Throw: Hold Block, Back, Forward, release Block + LP. 14% damage.
#   - Jump Kick: one of best in game. Hits ducked opponents, wins aerial duels.
#   - Basic combos: JK→Roll, aaHP×2→Roll, Uppercut→JP→Roll
#
# Block+direction actions (25-28) are intermediates the agent chains for specials.
MK_FIGHTING_ACTIONS = [
    # Movement (0-6)
    {},                                     # 0: No-op
    {_LEFT: 1},                             # 1: Walk back
    {_RIGHT: 1},                            # 2: Walk forward
    {_UP: 1},                               # 3: Jump
    {_DOWN: 1},                             # 4: Crouch
    {_UP: 1, _LEFT: 1},                     # 5: Jump back
    {_UP: 1, _RIGHT: 1},                    # 6: Jump forward

    # Basic attacks (7-10): Y=HP, L=LP, B=HK, A=LK
    {_Y: 1},                                # 7: High Punch (jab pressure)
    {_L: 1},                                # 8: Low Punch
    {_B: 1},                                # 9: High Kick (Roundhouse)
    {_A: 1},                                # 10: Low Kick

    # Block (11-12): X = Block button in MK
    {_X: 1},                                # 11: Stand Block
    {_DOWN: 1, _X: 1},                      # 12: Crouch Block

    # KEY CROUCH ATTACKS (13-16) — uppercut & crouch LK are Kano's best moves
    {_DOWN: 1, _L: 1},                      # 13: Crouch LP = UPPERCUT (longest range punisher)
    {_DOWN: 1, _Y: 1},                      # 14: Crouch HP
    {_DOWN: 1, _A: 1},                      # 15: Crouch LK = BEST POKE (spam this)
    {_DOWN: 1, _B: 1},                      # 16: Crouch HK (sweep)

    # Jump attacks (17-22) — Jump Kick is one of best in game
    {_UP: 1, _Y: 1},                        # 17: Jump HP (combo starter in corner)
    {_UP: 1, _A: 1},                        # 18: Jump LK
    {_UP: 1, _RIGHT: 1, _A: 1},             # 19: Jump-forward LK (hits OTG)
    {_UP: 1, _LEFT: 1, _A: 1},              # 20: Jump-back LK
    {_UP: 1, _RIGHT: 1, _B: 1},             # 21: Jump-forward HK
    {_UP: 1, _LEFT: 1, _B: 1},              # 22: Jump-back HK

    # Directional attacks (23-24)
    {_RIGHT: 1, _Y: 1},                     # 23: Forward + HP
    {_LEFT: 1, _Y: 1},                      # 24: Back + HP

    # Block + direction intermediates (25-28) for special move inputs.
    # Agent chains these for: Knife Throw (BL+B → BL+F → LP),
    # Roll/Cannonball (BL+B → BL+D → BL+F → release)
    {_X: 1, _LEFT: 1},                      # 25: Block + Back
    {_X: 1, _RIGHT: 1},                     # 26: Block + Forward
    {_X: 1, _DOWN: 1, _RIGHT: 1},            # 27: Block + Down-Forward (roll intermediate)
    {_X: 1, _UP: 1},                        # 28: Block + Up

    # Kick combos (29-30)
    {_RIGHT: 1, _B: 1},                     # 29: Forward + HK (Roundhouse approach)
    {_LEFT: 1, _B: 1},                      # 30: Back + HK

    # Multi-button (31)
    {_Y: 1, _A: 1},                         # 31: HP + LK
]


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass (duplicated lightly to avoid circular imports)
# ─────────────────────────────────────────────────────────────────────────────
class FightingGameConfig:
    """Minimal config used by env wrappers. Full config in game_configs.py."""
    def __init__(self, max_health=176, health_key="health",
                 enemy_health_key="enemy_health", timer_key="timer",
                 round_length_frames=5400, ram_overrides=None, actions=None):
        self.max_health = max_health
        self.health_key = health_key
        self.enemy_health_key = enemy_health_key
        self.timer_key = timer_key
        self.round_length_frames = round_length_frames
        # For games where data.json can't map high addresses (> 0x2000),
        # provide direct RAM reads: {"info_key": ram_offset}
        self.ram_overrides = ram_overrides or {}
        # Per-game action space (defaults to SF2-style FIGHTING_ACTIONS)
        self.actions = actions


# ─────────────────────────────────────────────────────────────────────────────
# Observation wrappers
# ─────────────────────────────────────────────────────────────────────────────
class GrayscaleResize(gym.ObservationWrapper):
    """Convert RGB to grayscale and resize to (84, 84)."""
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, 1), dtype=np.uint8
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return resized[:, :, np.newaxis]


class FrameStack(gym.Wrapper):
    """Stack n grayscale frames along channel dim -> (n, H, W) for CNN."""
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = None
        h, w, c = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(n_frames, h, w), dtype=np.uint8
        )

    def _get_frame(self, obs):
        # (H, W, 1) -> (H, W)
        return obs[:, :, 0] if obs.ndim == 3 else obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = self._get_frame(obs)
        self.frames = np.stack([frame] * self.n_frames, axis=0)
        return self.frames.copy(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self._get_frame(obs)
        self.frames = np.roll(self.frames, shift=-1, axis=0)
        self.frames[-1] = frame
        return self.frames.copy(), reward, terminated, truncated, info


class FrameSkip(gym.Wrapper):
    """Repeat each action for n frames, summing rewards."""
    def __init__(self, env, n_skip=4):
        super().__init__(env)
        self.n_skip = n_skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        obs = None
        for _ in range(self.n_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


# ─────────────────────────────────────────────────────────────────────────────
# Discrete action wrapper
# ─────────────────────────────────────────────────────────────────────────────
class DiscreteAction(gym.ActionWrapper):
    """Map discrete action index to SNES button array."""
    def __init__(self, env, action_map=None):
        super().__init__(env)
        self.action_map = action_map or FIGHTING_ACTIONS
        self.action_space = spaces.Discrete(len(self.action_map))

    def action(self, action_idx):
        idx = int(action_idx)
        idx = max(0, min(idx, len(self.action_map) - 1))
        buttons = np.zeros(12, dtype=np.int8)
        for btn, val in self.action_map[idx].items():
            buttons[btn] = val
        return buttons


# ─────────────────────────────────────────────────────────────────────────────
# Direct RAM reader for high-address games (e.g., MK2 SNES)
# ─────────────────────────────────────────────────────────────────────────────
class DirectRAMReader(gym.Wrapper):
    """
    Read RAM values directly via env.get_ram() and inject into info dict.

    Needed for SNES games where data.json can't map addresses >= 0x2000
    due to stable-retro's limited WRAM mapping.
    """

    def __init__(self, env, ram_map: dict[str, int]):
        """
        Args:
            env: RetroEnv (must be unwrapped or close to it)
            ram_map: {"variable_name": ram_offset} e.g. {"health": 0x2EFC}
        """
        super().__init__(env)
        self.ram_map = ram_map

    def _read_ram_vars(self, info):
        try:
            ram = self.unwrapped.get_ram()
            for key, offset in self.ram_map.items():
                if offset < len(ram):
                    info[key] = int(ram[offset])
        except Exception:
            pass
        return info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, self._read_ram_vars(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self._read_ram_vars(info)


# ─────────────────────────────────────────────────────────────────────────────
# 2P practice mode: P2 is human-idle (null bot)
# ─────────────────────────────────────────────────────────────────────────────
class NullP2Wrapper(gym.ActionWrapper):
    """
    Adapter for 2P practice states: pads P1's 12-button action to 24 buttons
    with P2 getting no input (idle). Requires a save state created in 2P VS mode.
    """

    def __init__(self, env):
        super().__init__(env)
        # Expose only P1's 12 buttons to upstream wrappers
        self.action_space = spaces.MultiBinary(12)

    def action(self, p1_action):
        full = np.zeros(24, dtype=np.int8)
        full[:12] = p1_action
        return full


# ─────────────────────────────────────────────────────────────────────────────
# Reward shaping wrapper
# ─────────────────────────────────────────────────────────────────────────────
class FightingEnv(gym.Wrapper):
    """
    Fighting game reward wrapper.

    Reward components:
    - Health delta: reward for damaging enemy, penalty for taking damage
    - Win bonus: large reward for winning a round (enemy health -> 0)
    - Loss penalty: penalty for losing a round (own health -> 0)
    - Time penalty: small per-step penalty to encourage aggression
    """

    # Reward weights
    REWARD_DAMAGE_DEALT = 1.0       # Per point of enemy health lost
    REWARD_DAMAGE_TAKEN = -0.5      # Per point of own health lost
    REWARD_ROUND_WIN = 50.0         # Bonus for winning a round
    REWARD_ROUND_LOSS = -50.0       # Penalty for losing a round
    REWARD_DOUBLE_KO = -100.0       # Heavy penalty for double-KO (both die = no progress)
    REWARD_MATCH_WIN = 200.0        # Bonus for winning the match
    REWARD_TIME_PENALTY = -0.001    # Per-frame time penalty
    REWARD_TIMEOUT_ROUND = -0.15    # Flat penalty for ANY timeout round (discourages passive play)

    def __init__(self, env, config: Optional[FightingGameConfig] = None):
        super().__init__(env)
        self.config = config or FightingGameConfig()
        self.reward_scale = 1.0 / self.config.max_health
        self.prev_health = None
        self.prev_enemy_health = None
        self.rounds_won = 0
        self.rounds_lost = 0
        self.frame_count = 0
        self.episode_damage_dealt = 0
        self.episode_damage_taken = 0
        # Timeout detection state
        self._ko_this_round = False
        self._fighting_health = None       # Health during fighting (not refill)
        self._fighting_enemy_health = None  # Enemy health during fighting (not refill)
        self.timeout_rounds = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_health = info.get(self.config.health_key, self.config.max_health)
        self.prev_enemy_health = info.get(self.config.enemy_health_key, self.config.max_health)
        self.rounds_won = 0
        self.rounds_lost = 0
        self.frame_count = 0
        self.episode_damage_dealt = 0
        self.episode_damage_taken = 0
        # Timeout detection state
        self._ko_this_round = False
        self._fighting_health = self.prev_health
        self._fighting_enemy_health = self.prev_enemy_health
        self.timeout_rounds = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frame_count += 1

        health = info.get(self.config.health_key, self.prev_health)
        enemy_health = info.get(self.config.enemy_health_key, self.prev_enemy_health)

        shaped_reward = 0.0

        # Damage dealt to enemy
        if self.prev_enemy_health is not None and enemy_health < self.prev_enemy_health:
            dmg = self.prev_enemy_health - enemy_health
            shaped_reward += dmg * self.REWARD_DAMAGE_DEALT * self.reward_scale
            self.episode_damage_dealt += dmg

        # Damage taken
        if self.prev_health is not None and health < self.prev_health:
            dmg = self.prev_health - health
            shaped_reward += dmg * self.REWARD_DAMAGE_TAKEN * self.reward_scale
            self.episode_damage_taken += dmg

        # Track "fighting" health: only update when health is decreasing or stable
        # (never during refill animation after a round ends). This captures the
        # last health values from actual combat for timeout winner detection.
        max_h = self.config.max_health
        health_near_max = health >= max_h - 10
        enemy_near_max = enemy_health >= max_h - 10

        if health <= (self.prev_health or max_h):
            self._fighting_health = health
        if enemy_health <= (self.prev_enemy_health or max_h):
            self._fighting_enemy_health = enemy_health

        # Round detection: KO
        enemy_died = (self.prev_enemy_health is not None
                      and self.prev_enemy_health > 0 and enemy_health <= 0)
        player_died = (self.prev_health is not None
                       and self.prev_health > 0 and health <= 0)

        if enemy_died and player_died:
            # Double-KO: both die simultaneously. Heavy penalty - trading kills
            # doesn't win matches. Count for both sides but penalize hard.
            self.rounds_won += 1
            self.rounds_lost += 1
            self._ko_this_round = True
            shaped_reward += self.REWARD_DOUBLE_KO * self.reward_scale
        elif enemy_died:
            self.rounds_won += 1
            self._ko_this_round = True
            shaped_reward += self.REWARD_ROUND_WIN * self.reward_scale
        elif player_died:
            self.rounds_lost += 1
            self._ko_this_round = True
            shaped_reward += self.REWARD_ROUND_LOSS * self.reward_scale

        # Round detection: TIMEOUT
        # When the in-game timer expires, the game resets both health bars to max.
        # Detect this as: both health near max AND at least one fighting health
        # was significantly below max AND no KO happened this round.
        if (health_near_max and enemy_near_max
                and not self._ko_this_round
                and self._fighting_health is not None
                and self._fighting_enemy_health is not None
                and (self._fighting_health < max_h - 10
                     or self._fighting_enemy_health < max_h - 10)):
            # Timeout detected — determine winner by who had more health
            self.timeout_rounds += 1
            if self._fighting_health > self._fighting_enemy_health:
                # Player was winning → timeout win (reduced reward)
                self.rounds_won += 1
                shaped_reward += self.REWARD_ROUND_WIN * self.reward_scale * 0.5
            elif self._fighting_enemy_health > self._fighting_health:
                # Enemy was winning → timeout loss (full penalty)
                self.rounds_lost += 1
                shaped_reward += self.REWARD_ROUND_LOSS * self.reward_scale
            else:
                # Draw → treat as loss
                self.rounds_lost += 1
                shaped_reward += self.REWARD_ROUND_LOSS * self.reward_scale
            # Flat timeout penalty in ALL cases (discourages running clock)
            shaped_reward += self.REWARD_TIMEOUT_ROUND
            # Reset fighting health for next round
            self._fighting_health = max_h
            self._fighting_enemy_health = max_h

        # Clear KO flag when health bars reset (new round starting)
        if health_near_max and enemy_near_max:
            self._ko_this_round = False

        # Time penalty
        shaped_reward += self.REWARD_TIME_PENALTY

        # OVERRIDE base environment's termination - only WE decide when match ends
        # Match win requires strictly more round wins than losses
        terminated = False
        truncated = False

        if self.rounds_won >= 2 and self.rounds_won > self.rounds_lost:
            shaped_reward += self.REWARD_MATCH_WIN * self.reward_scale
            terminated = True
        elif self.rounds_lost >= 2 and self.rounds_lost > self.rounds_won:
            terminated = True
        elif self.rounds_won >= 2 and self.rounds_lost >= 2:
            # Draw (e.g. 2-2 from double-KOs) - treat as loss
            terminated = True
        elif self.frame_count >= self.config.round_length_frames * 3:
            truncated = True

        # Update tracking
        self.prev_health = health
        self.prev_enemy_health = enemy_health

        # Augment info
        info["shaped_reward"] = shaped_reward
        info["rounds_won"] = self.rounds_won
        info["rounds_lost"] = self.rounds_lost
        info["episode_damage_dealt"] = self.episode_damage_dealt
        info["episode_damage_taken"] = self.episode_damage_taken
        info["timeout_rounds"] = self.timeout_rounds

        return obs, shaped_reward, terminated, truncated, info


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────
def make_fighting_env(
    game: str,
    state: str,
    game_dir: str | Path,
    config: Optional[FightingGameConfig] = None,
    render_mode: str = "rgb_array",
    frame_skip: int = 4,
    frame_stack: int = 4,
    monitor_dir: Optional[str] = None,
    practice: bool = False,
    combos: list[dict] | None = None,
):
    """
    Create a fully wrapped fighting game environment for PPO training.

    Args:
        practice: If True, use 2P mode with a Practice_* state so P2 is idle (null bot).
        combos: Optional list of combo definitions for ComboFrameSkip. When provided,
                uses ComboFrameSkip instead of regular FrameSkip, and appends combo
                actions to the discrete action space.

    Wrapper stack:
        RetroEnv -> [NullP2Wrapper] -> FrameSkip/ComboFrameSkip -> GrayscaleResize -> FightingEnv -> DiscreteAction -> FrameStack -> Monitor
    """
    game_dir = Path(game_dir).resolve()
    integrations_path = game_dir / "custom_integrations"
    if integrations_path.exists():
        retro.data.Integrations.add_custom_path(str(integrations_path))

    if state == "NONE":
        state = retro.State.NONE

    retro_kwargs = dict(
        game=game,
        state=state,
        render_mode=render_mode,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        use_restricted_actions=retro.Actions.ALL,
    )
    if practice:
        retro_kwargs["players"] = 2

    env = retro.make(**retro_kwargs)

    # In practice mode, pad P1 actions to 24 buttons (P2 gets no input = idle)
    if practice:
        env = NullP2Wrapper(env)

    # Wrapper stack
    # Direct RAM reader must be first (needs access to unwrapped env)
    if config and config.ram_overrides:
        env = DirectRAMReader(env, config.ram_overrides)
    if combos:
        from fighters_common.combo_wrapper import ComboFrameSkip
        env = ComboFrameSkip(env, combos=combos, n_skip=frame_skip)
    elif frame_skip > 1:
        env = FrameSkip(env, n_skip=frame_skip)
    env = GrayscaleResize(env, width=84, height=84)
    env = FightingEnv(env, config=config)
    action_map = (config.actions if config and config.actions else FIGHTING_ACTIONS)
    if combos:
        from fighters_common.combo_wrapper import get_combo_actions
        action_map = list(action_map) + get_combo_actions(combos)
    env = DiscreteAction(env, action_map)
    if frame_stack > 1:
        env = FrameStack(env, n_frames=frame_stack)

    # Monitor
    if monitor_dir:
        os.makedirs(monitor_dir, exist_ok=True)
        monitor_path = os.path.join(monitor_dir, f"{game}_{state}_{int(time.time())}.csv")
        env = Monitor(env, filename=monitor_path)

    return env

"""
RAM vector observations for fighting-game RL.

Replaces GrayscaleResize + FrameStack with a small normalized feature vector
read from stable-retro info keys (data.json) or DirectRAMReader overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from fighters_common.fighting_env import (
    DirectRAMReader,
    DiscreteAction,
    FightingEnv,
    FightingGameConfig,
    FIGHTING_ACTIONS,
    FrameSkip,
    NullP2Wrapper,
)


class RamFeatureKind(str, Enum):
    """How a RAM observation feature is sourced."""

    INFO = "info"
    DELTA = "delta"
    COMPUTED = "computed"


@dataclass(frozen=True)
class RamFeatureSpec:
    """Single normalized feature in the RAM observation vector."""

    name: str
    kind: RamFeatureKind
    info_key: str = ""
    scale: float = 1.0
    clip_min: float = 0.0
    clip_max: float = 1.0
    compute: str = ""


# MK1 SNES v1 — health/timer/rounds only (E017 discard: no spacing).
MK1_RAM_FEATURES_V1: tuple[RamFeatureSpec, ...] = (
    RamFeatureSpec("p1_health", RamFeatureKind.INFO, "health", scale=161.0),
    RamFeatureSpec("p2_health", RamFeatureKind.INFO, "enemy_health", scale=161.0),
    RamFeatureSpec(
        "p1_health_delta",
        RamFeatureKind.DELTA,
        "health",
        scale=161.0,
        clip_min=-1.0,
        clip_max=1.0,
    ),
    RamFeatureSpec(
        "p2_health_delta",
        RamFeatureKind.DELTA,
        "enemy_health",
        scale=161.0,
        clip_min=-1.0,
        clip_max=1.0,
    ),
    RamFeatureSpec("timer", RamFeatureKind.INFO, "timer", scale=154.0),
    RamFeatureSpec("p2_char_id", RamFeatureKind.INFO, "p2_character", scale=6.0),
    RamFeatureSpec("p1_rounds", RamFeatureKind.INFO, "p1_rounds", scale=2.0),
    RamFeatureSpec("p2_rounds", RamFeatureKind.INFO, "p2_rounds", scale=2.0),
    RamFeatureSpec("match_counter", RamFeatureKind.INFO, "match_counter", scale=11.0),
)

# MK1 SNES v2 — adds screen positions (0xDA P1 X, 0x174 P2 X) + distance.
MK1_RAM_FEATURES: tuple[RamFeatureSpec, ...] = MK1_RAM_FEATURES_V1 + (
    RamFeatureSpec("p1_x", RamFeatureKind.INFO, "p1_x", scale=255.0),
    RamFeatureSpec("p2_x", RamFeatureKind.INFO, "p2_x", scale=255.0),
    RamFeatureSpec("p1_y", RamFeatureKind.INFO, "p1_y", scale=255.0),
    RamFeatureSpec(
        "distance_x",
        RamFeatureKind.COMPUTED,
        compute="distance_x",
        scale=255.0,
    ),
)


def build_ram_features(
    features: tuple[RamFeatureSpec, ...],
    info: dict,
    prev_values: dict[str, float],
) -> tuple[np.ndarray, dict[str, float]]:
    """Build a normalized RAM observation vector and updated previous values."""
    values: list[float] = []
    next_prev = dict(prev_values)

    for spec in features:
        if spec.kind is RamFeatureKind.INFO:
            raw = float(info.get(spec.info_key, 0))
            next_prev[spec.info_key] = raw
            norm = raw / spec.scale if spec.scale else raw
            values.append(float(np.clip(norm, spec.clip_min, spec.clip_max)))
            continue

        if spec.kind is RamFeatureKind.COMPUTED:
            raw = _compute_feature(spec, info)
            norm = raw / spec.scale if spec.scale else raw
            values.append(float(np.clip(norm, spec.clip_min, spec.clip_max)))
            continue

        raw = float(info.get(spec.info_key, prev_values.get(spec.info_key, 0)))
        prev = prev_values.get(spec.info_key, raw)
        delta = raw - prev
        next_prev[spec.info_key] = raw
        norm = delta / spec.scale if spec.scale else delta
        values.append(float(np.clip(norm, spec.clip_min, spec.clip_max)))

    return np.asarray(values, dtype=np.float32), next_prev


def _compute_feature(spec: RamFeatureSpec, info: dict) -> float:
    """Evaluate a computed RAM feature from current info."""
    if spec.compute == "distance_x":
        p1_x = float(info.get("p1_x", 0))
        p2_x = float(info.get("p2_x", 0))
        return abs(p2_x - p1_x)
    raise ValueError(f"Unknown computed RAM feature: {spec.compute}")


MK1_RAM_INFO_KEYS: tuple[str, ...] = (
    "health",
    "enemy_health",
    "timer",
    "p2_character",
    "p1_rounds",
    "p2_rounds",
    "match_counter",
    "p1_x",
    "p1_y",
    "p2_x",
)


class DataJsonInfoEnricher(gym.Wrapper):
    """Inject data.json RAM fields into info via lookup_value (works on reset)."""

    def __init__(self, env: gym.Env, keys: tuple[str, ...] = MK1_RAM_INFO_KEYS):
        super().__init__(env)
        self.keys = keys

    def _enrich(self, info: dict) -> dict:
        try:
            for key in self.keys:
                info[key] = self.unwrapped.data.lookup_value(key)
        except Exception:
            pass
        return info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, self._enrich(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self._enrich(info)


class RamObservation(gym.ObservationWrapper):
    """Expose normalized RAM features as the policy observation vector."""

    def __init__(
        self,
        env: gym.Env,
        features: tuple[RamFeatureSpec, ...] = MK1_RAM_FEATURES,
    ):
        super().__init__(env)
        self.features = features
        self._prev_values: dict[str, float] = {}
        self._last_info: dict = {}
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(features),),
            dtype=np.float32,
        )

    def _vector_from_info(self, info: dict) -> np.ndarray:
        vector, self._prev_values = build_ram_features(
            self.features, info, self._prev_values
        )
        return vector

    def observation(self, observation):
        del observation
        return self._vector_from_info(self._last_info)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_values = {}
        self._last_info = info
        return self._vector_from_info(info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_info = info
        return self._vector_from_info(info), reward, terminated, truncated, info


def make_ram_fighting_env(
    game: str,
    state: str,
    game_dir: str,
    config: Optional[FightingGameConfig] = None,
    render_mode: str = "rgb_array",
    frame_skip: int = 4,
    monitor_dir: Optional[str] = None,
    practice: bool = False,
    combos: list[dict] | None = None,
    randomize_state: bool = False,
    features: tuple[RamFeatureSpec, ...] = MK1_RAM_FEATURES,
):
    """
    Create a RAM-observation fighting environment for MLP PPO training.

    Wrapper stack (no GrayscaleResize / FrameStack):
        RetroEnv -> [NullP2Wrapper] -> [DirectRAMReader] -> FrameSkip
        -> DataJsonInfoEnricher -> FightingEnv -> RamObservation -> DiscreteAction
        -> Monitor
    """
    import os
    import time
    from pathlib import Path

    import stable_retro as retro
    from stable_baselines3.common.monitor import Monitor

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

    if practice:
        env = NullP2Wrapper(env)

    if config and config.ram_overrides:
        env = DirectRAMReader(env, config.ram_overrides)

    if combos:
        from fighters_common.combo_wrapper import ComboFrameSkip

        env = ComboFrameSkip(env, combos=combos, n_skip=frame_skip)
    elif frame_skip > 1:
        env = FrameSkip(env, n_skip=frame_skip)

    env = DataJsonInfoEnricher(env)
    env = FightingEnv(env, config=config, randomize_state=randomize_state)
    env = RamObservation(env, features=features)

    action_map = config.actions if config and config.actions else FIGHTING_ACTIONS
    if combos:
        from fighters_common.combo_wrapper import get_combo_actions

        action_map = list(action_map) + get_combo_actions(combos)
    env = DiscreteAction(env, action_map)

    if monitor_dir:
        os.makedirs(monitor_dir, exist_ok=True)
        monitor_path = os.path.join(
            monitor_dir, f"{game}_{state}_ram_{int(time.time())}.csv"
        )
        env = Monitor(env, filename=monitor_path)

    return env


def build_eval_env(
    game: str,
    state: str,
    game_dir: str,
    config: Optional[FightingGameConfig] = None,
    *,
    ram: bool = False,
    frame_skip: int = 4,
    frame_stack: int = 4,
    features: tuple[RamFeatureSpec, ...] = MK1_RAM_FEATURES,
):
    """Build an evaluation env without Monitor (pixel CNN or RAM MLP)."""
    if ram:
        return make_ram_fighting_env(
            game=game,
            state=state,
            game_dir=game_dir,
            config=config,
            frame_skip=frame_skip,
            monitor_dir=None,
            features=features,
        )

    from fighters_common.fighting_env import (
        FrameStack,
        GrayscaleResize,
        make_fighting_env,
    )

    return make_fighting_env(
        game=game,
        state=state,
        game_dir=game_dir,
        config=config,
        frame_skip=frame_skip,
        frame_stack=frame_stack,
        monitor_dir=None,
    )

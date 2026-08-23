"""MK1 RAM+hitbox observation wrapper (v3).

Reads ``get_ram()`` every step via ``parse_ram`` — not pixels. Used for
overnight specialist training. Observation dim is 20 (incompatible with
v1/v2 MLP checkpoints).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from retro_harness.fighters.fighting_env import (
    DirectRAMReader,
    DiscreteAction,
    FightingEnv,
    FightingGameConfig,
    FIGHTING_ACTIONS,
    FrameSkip,
)
from retro_harness.fighters.ram_observation import DataJsonInfoEnricher
from mortal_kombat.ram import V3_DIM, parse_ram, snapshot_features


class MkRamObservation(gym.Wrapper):
    """Replace pixels with the 20-dim hitbox RAM vector."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._prev = (0, 0)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(V3_DIM,), dtype=np.float32
        )

    def _vector(self, info: dict) -> np.ndarray:
        snap = parse_ram(self.unwrapped.get_ram())
        vector, self._prev = snapshot_features(snap, self._prev)
        info["p1_x"] = snap.p1.x
        info["p2_x"] = snap.p2.x
        info["p1_y"] = snap.p1.y
        info["p2_y"] = snap.p2.y
        info["distance_x"] = snap.distance_x
        info["p1_state"] = snap.p1.state
        info["p2_state"] = snap.p2.state
        info["bodies_overlap"] = int(snap.bodies_overlap)
        info["p1_hit_connects"] = int(snap.p1_hit_connects)
        return vector

    def reset(self, **kwargs):
        _obs, info = self.env.reset(**kwargs)
        self._prev = (0, 0)
        return self._vector(info), info

    def step(self, action):
        _obs, reward, terminated, truncated, info = self.env.step(action)
        return self._vector(info), reward, terminated, truncated, info


def make_mk_ram_env(
    game: str,
    state: str,
    game_dir: str | Path,
    config: Optional[FightingGameConfig] = None,
    render_mode: str = "rgb_array",
    frame_skip: int = 4,
    monitor_dir: Optional[str] = None,
    randomize_state: bool = False,
):
    """RAM+hitbox fighting env (no grayscale / frame-stack)."""
    import os
    import time

    import stable_retro as retro
    from stable_baselines3.common.monitor import Monitor

    game_dir = Path(game_dir).resolve()
    integrations = game_dir / "custom_integrations"
    if integrations.exists():
        retro.data.Integrations.add_custom_path(str(integrations))

    if state == "NONE":
        state = retro.State.NONE

    env = retro.make(
        game=game,
        state=state,
        render_mode=render_mode,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        use_restricted_actions=retro.Actions.ALL,
    )
    if config and config.ram_overrides:
        env = DirectRAMReader(env, config.ram_overrides)
    if frame_skip > 1:
        env = FrameSkip(env, n_skip=frame_skip)
    env = DataJsonInfoEnricher(env)
    env = FightingEnv(env, config=config, randomize_state=randomize_state)
    env = MkRamObservation(env)
    action_map = config.actions if config and config.actions else FIGHTING_ACTIONS
    env = DiscreteAction(env, action_map)
    if monitor_dir:
        os.makedirs(monitor_dir, exist_ok=True)
        path = os.path.join(monitor_dir, f"{game}_{state}_v3_{int(time.time())}.csv")
        env = Monitor(env, filename=path)
    return env

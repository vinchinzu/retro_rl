"""Gymnasium env: Bomb Torizo fight on structured ``feature_vector`` obs.

Vision-free. Observation is the 14-float combat feature vector. Actions are
the shared discrete combat table. Reward shapes boss damage, Samus damage,
and fight length — suitable for short PPO/SB3 loops before distilling back
to a deterministic controller.

Gymnasium env only — no dedicated probe CLI. Distill back to
``combat.bomb_torizo.play_bomb_torizo_fight``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from retro_harness.actions import buttons
from retro_harness.env import make_env, read_state_bytes
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.combat.actions import (
    N_COMBAT_ACTIONS,
    action_vector,
    nearest_action_id,
)
from super_metroid.combat.audit import structured_combat_audit_info
from super_metroid.combat.bomb_torizo import fight_bomb_torizo_action
from super_metroid.combat.features import (
    FEATURE_DIM,
    bomb_torizo_catalog,
    feature_vector,
    features_from_state,
)
from super_metroid.paths import GAME, GAME_DIR, SCRATCH_STATE_DIR
from super_metroid.ram import parse_state

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - optional dependency path
    raise ImportError(
        "gymnasium is required for combat.env; install with "
        "`uv sync --extra ml`"
    ) from exc

ROOM_BOMB_TORIZO = 0x9804
DEFAULT_STATE_NAME = "BossTorizo"
NATURAL_ACTIVE_STATE = SCRATCH_STATE_DIR / "natural_bomb_torizo_active.state"


class BombTorizoFeatureEnv(gym.Env):
    """Single-boss fight env with full-knowledge feature observations."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        *,
        state: str | Path = DEFAULT_STATE_NAME,
        max_episode_frames: int = 4_000,
        damage_penalty: float = 0.5,
        frame_penalty: float = 0.001,
        win_bonus: float = 50.0,
        lose_penalty: float = 10.0,
        require_active: bool = True,
        unlimited_energy: bool = True,
        unlimited_ammo: bool = True,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.state_spec = state
        self.max_episode_frames = max_episode_frames
        self.damage_penalty = damage_penalty
        self.frame_penalty = frame_penalty
        self.win_bonus = win_bonus
        self.lose_penalty = lose_penalty
        self.require_active = require_active
        self.unlimited_energy = unlimited_energy
        self.unlimited_ammo = unlimited_ammo
        self.render_mode = render_mode

        self.catalog = bomb_torizo_catalog()
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(FEATURE_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(N_COMBAT_ACTIONS)

        self._env: Any | None = None
        self._assist = UnlimitedResourcesAssist(
            unlimited_energy=unlimited_energy,
            unlimited_ammo=unlimited_ammo,
        )
        self._frame = 0
        self._prev_enemy_hp = 0
        self._prev_samus_health = 0
        self._episode_damage_taken = 0
        self._last_obs: np.ndarray | None = None
        self._last_rgb: np.ndarray | None = None

    # ------------------------------------------------------------------ API

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._assist = UnlimitedResourcesAssist(
            unlimited_energy=self.unlimited_energy,
            unlimited_ammo=self.unlimited_ammo,
        )
        # stable-retro allows only one emulator instance per process — reuse it.
        if self._env is None:
            self._env = self._make_backend()
            obs_rgb, _ = self._env.reset()
            self._last_rgb = obs_rgb
        self._load_state_if_needed()
        if self._last_rgb is None:
            # Path-loaded named states still need a step buffer for render.
            self._last_rgb, _, _, _, _ = self._env.step(action_vector(0))

        # Settle a few frames so RAM is coherent after load.
        for _ in range(2):
            self._last_rgb, _, _, _, _ = self._env.step(action_vector(0))

        state = parse_state(self._env.get_ram(), frame=0)
        if state.room_id != ROOM_BOMB_TORIZO:
            raise RuntimeError(
                f"BombTorizoFeatureEnv expected room 0x{ROOM_BOMB_TORIZO:04X}, "
                f"got 0x{state.room_id:04X} from {self.state_spec!r}"
            )
        if state.selected_item != 1 and state.max_missiles > 0:
            for _ in range(8):
                if parse_state(self._env.get_ram()).selected_item == 1:
                    break
                self._last_rgb, _, _, _, _ = self._env.step(buttons("SELECT"))
                state = parse_state(self._env.get_ram())
                self._assist.apply(self._env.data, state)

        state = parse_state(self._env.get_ram(), frame=0)
        feat = features_from_state(state, self.catalog)
        if self.require_active and not feat.enemy_active and not feat.enemy_defeated:
            raise RuntimeError(
                f"Torizo not combat-active on reset (spritemap "
                f"0x{feat.enemy_spritemap:04X}). Capture natural activation "
                f"or use BossTorizo."
            )

        self._frame = 0
        self._prev_enemy_hp = state.enemy0_hp
        self._prev_samus_health = state.health
        self._episode_damage_taken = 0
        self._last_obs = feature_vector(feat)
        info = {
            "features": feat.to_dict(),
            "state": self.state_spec if isinstance(self.state_spec, str) else str(self.state_spec),
            **structured_combat_audit_info(self._assist.telemetry),
        }
        return self._last_obs.copy(), info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._env is None:
            raise RuntimeError("env.reset() must be called before step()")

        action_id = int(action)
        vec = action_vector(action_id)
        self._last_rgb, _, _, _, _ = self._env.step(vec)
        self._frame += 1

        # Pre-assist health for true damage signal.
        raw = parse_state(self._env.get_ram(), frame=self._frame)
        damage = max(0, self._prev_samus_health - raw.health)
        self._assist.apply(self._env.data, raw)
        state = parse_state(self._env.get_ram(), frame=self._frame)

        boss_damage = max(0, self._prev_enemy_hp - state.enemy0_hp)
        self._episode_damage_taken += damage
        reward = (
            float(boss_damage)
            - self.damage_penalty * float(damage)
            - self.frame_penalty
        )

        feat = features_from_state(state, self.catalog)
        terminated = False
        truncated = False
        if feat.enemy_defeated or state.enemy0_hp == 0:
            reward += self.win_bonus
            terminated = True
        elif self._frame >= self.max_episode_frames:
            reward -= self.lose_penalty
            truncated = True
        elif state.room_id != ROOM_BOMB_TORIZO:
            reward -= self.lose_penalty
            terminated = True

        self._prev_enemy_hp = state.enemy0_hp
        self._prev_samus_health = state.health
        self._last_obs = feature_vector(feat)
        info = {
            "features": feat.to_dict(),
            "boss_damage": boss_damage,
            "samus_damage": damage,
            "frame": self._frame,
            "episode_damage_taken": self._episode_damage_taken,
            "assist": {
                "energy_restored": self._assist.telemetry.energy.restored,
                "deaths": self._assist.telemetry.deaths,
                "maximum_single_frame_damage": (
                    self._assist.telemetry.maximum_single_frame_damage
                ),
            },
            **structured_combat_audit_info(self._assist.telemetry),
        }
        return self._last_obs.copy(), reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        return self._last_rgb

    def close(self) -> None:
        self._close_env()

    # ----------------------------------------------------------- baselines

    def strategy_action(self) -> int:
        """Project the full-knowledge strategy onto the discrete action table."""
        if self._env is None:
            raise RuntimeError("env.reset() required")
        state = parse_state(self._env.get_ram(), frame=self._frame)
        names = fight_bomb_torizo_action(state, self._frame)
        return nearest_action_id(names)

    # --------------------------------------------------------------- intern

    def _make_backend(self) -> Any:
        # Named integration states load via retro; paths load after reset.
        if isinstance(self.state_spec, Path) or (
            isinstance(self.state_spec, str)
            and (self.state_spec.endswith(".state") or "/" in self.state_spec)
        ):
            return make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
        return make_env(GAME, str(self.state_spec), GAME_DIR, render_mode="rgb_array")

    def _load_state_if_needed(self) -> None:
        assert self._env is not None
        path: Path | None = None
        if isinstance(self.state_spec, Path):
            path = self.state_spec
        elif isinstance(self.state_spec, str) and (
            self.state_spec.endswith(".state") or "/" in self.state_spec
        ):
            path = Path(self.state_spec)
            if not path.is_absolute():
                candidate = GAME_DIR / path
                path = candidate if candidate.exists() else Path(self.state_spec)
        if path is not None:
            if not path.exists():
                raise FileNotFoundError(f"combat env state not found: {path}")
            self._env.em.set_state(read_state_bytes(path))

    def _close_env(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


def resolve_state_spec(name: str) -> str | Path:
    """Resolve CLI state name to BossTorizo / path / natural scratch."""
    if name in ("natural", "natural-active", "natural_active"):
        return NATURAL_ACTIVE_STATE
    path = Path(name)
    if path.suffix == ".state" or path.exists():
        return path
    return name

"""Gymnasium env: practice-ROM pins, Survival, Join terminal, SM_ACTIONS."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from retro_harness.env import make_env, read_state_bytes
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.generalist.corpus import CorpusRow, assert_practice_rom, load_rows
from super_metroid.generalist.goals import is_join
from super_metroid.generalist.obs import N_ACTIONS, OBS_DIM, observe
from super_metroid.generalist.solid import (
    CollisionDependencyError,
    RoomSolid,
    editor_rooms_dir,
    load_room_solid,
    require_row_solids,
)
from super_metroid.generalist.steering import (
    SteeringTarget,
    capabilities_from_state,
    load_room_graph,
    steering_distance,
    steering_target,
)
from super_metroid.paths import GAME_DIR, PRACTICE_GAME, SHARED_PRACTICE_ROM
from super_metroid.platformer_levels import SM_ACTIONS
from super_metroid.ram import GS_DEAD, GS_ORDINARY, parse_env_state

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - optional dependency path
    raise ImportError(
        "gymnasium is required for generalist.env; install with `uv sync --extra ml`"
    ) from exc

STALL_FRAMES = 240
MAX_EPISODE_FRAMES = 1_800
JOIN_REWARD = 1.0
DEATH_REWARD = -1.0
STALL_REWARD = -0.05
DISTANCE_CLIP = 64.0
FRAME_SKIP = 4


def _opi_key(state: Any) -> tuple[int, int, int, int]:
    return (int(state.room_id), int(state.samus_x), int(state.samus_y), int(state.pose))


class GeneralistEnv(gym.Env):
    """Reset from a practice-repertoire pin; terminate on Join / death / stall."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        *,
        rows: list[CorpusRow] | None = None,
        area: str | None = "crateria",
        frame_skip: int = FRAME_SKIP,
        max_episode_frames: int = MAX_EPISODE_FRAMES,
        stall_frames: int = STALL_FRAMES,
        render_mode: str | None = None,
        survival: bool = True,
    ) -> None:
        super().__init__()
        self.rows = list(rows) if rows is not None else load_rows(
            area=area, exclude_ceres=area == "crateria", dedupe=True
        )
        if not self.rows:
            raise RuntimeError("generalist corpus is empty (capture practice states first)")
        self._collision_root = editor_rooms_dir()
        self._solids = require_row_solids(self.rows, root=self._collision_root)
        self._graph = load_room_graph()
        assert_practice_rom(SHARED_PRACTICE_ROM)
        self.frame_skip = max(1, int(frame_skip))
        self.max_episode_frames = int(max_episode_frames)
        self.stall_frames = int(stall_frames)
        self.render_mode = render_mode
        self.survival = survival
        self.observation_space = spaces.Box(
            low=-4.0, high=4.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._env: Any | None = None
        self._assist = UnlimitedResourcesAssist(
            unlimited_energy=survival, unlimited_ammo=survival
        )
        self._row = self.rows[0]
        self._goal = self._row.goal()
        self._frame = 0
        self._prev_action = 0
        self._prev_distance = 0.0
        self._stall = 0
        self._last_opi: tuple[int, int, int, int] | None = None
        self._last_obs: np.ndarray | None = None
        self._last_rgb: np.ndarray | None = None
        self._last_info: dict[str, Any] = {}
        self._last_target: SteeringTarget | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        if "row" in options:
            self._row = options["row"]
        elif "session_id" in options:
            match = [row for row in self.rows if row.session_id == options["session_id"]]
            if not match:
                raise KeyError(options["session_id"])
            self._row = match[0]
        else:
            index = int(self.np_random.integers(0, len(self.rows)))
            self._row = self.rows[index]
        self._goal = self._row.goal()
        self._assist = UnlimitedResourcesAssist(
            unlimited_energy=self.survival, unlimited_ammo=self.survival
        )
        if self._env is None:
            self._env = make_env(
                PRACTICE_GAME, "NONE", GAME_DIR, render_mode="rgb_array"
            )
            self._last_rgb, _ = self._env.reset()
        path = Path(self._row.state_path)
        self._env.em.set_state(read_state_bytes(path))
        idle = np.asarray(SM_ACTIONS[25], dtype=np.int8)
        for _ in range(2):
            self._last_rgb, _, _, _, _ = self._env.step(idle)
        state = parse_env_state(self._env, frame=0, mode="nav")
        if int(state.game_state) != GS_ORDINARY:
            raise RuntimeError(
                f"{self._row.session_id} loaded gs={state.game_state}, want 8"
            )
        self._frame = 0
        self._prev_action = 0
        self._last_target = self._target_for(state)
        self._prev_distance = steering_distance(state, self._last_target)
        self._stall = 0
        self._last_opi = _opi_key(state)
        self._last_obs = self._observe(
            state, prev_action=0, target=self._last_target
        )
        info = self._info(state, reason="reset", joined=False)
        self._last_info = info
        return self._last_obs.copy(), info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._env is None or self._last_obs is None:
            raise RuntimeError("env.reset() must be called before step()")
        action_id = int(action) % N_ACTIONS
        vec = np.asarray(SM_ACTIONS[action_id], dtype=np.int8)
        reward = 0.0
        terminated = False
        truncated = False
        reason = "step"
        joined = False
        state = parse_env_state(self._env, frame=self._frame, mode="nav")
        try:
            for _hold in range(self.frame_skip):
                self._last_rgb, _, _, _, _ = self._env.step(vec)
                self._frame += 1
                raw = parse_env_state(self._env, frame=self._frame, mode="nav")
                self._assist.apply(self._env.data, raw)
                state = parse_env_state(self._env, frame=self._frame, mode="nav")
                target = self._target_for(state)
                dist = steering_distance(state, target)
                delta = self._prev_distance - dist
                reward += float(
                    max(-DISTANCE_CLIP, min(DISTANCE_CLIP, delta)) / DISTANCE_CLIP
                )
                self._prev_distance = dist
                self._last_target = target
                key = _opi_key(state)
                if key == self._last_opi:
                    self._stall += 1
                else:
                    self._stall = 0
                    self._last_opi = key
                if is_join(state, self._goal):
                    reward += JOIN_REWARD
                    terminated = True
                    joined = True
                    reason = "join"
                    break
                if int(state.health) <= 0 or int(state.game_state) in GS_DEAD:
                    reward += DEATH_REWARD
                    terminated = True
                    reason = "death"
                    break
                if self._stall >= self.stall_frames:
                    reward += STALL_REWARD
                    truncated = True
                    reason = "stall"
                    break
                if self._frame >= self.max_episode_frames:
                    truncated = True
                    reason = "timeout"
                    break
            self._last_obs = self._observe(
                state, prev_action=action_id, target=self._last_target
            )
        except CollisionDependencyError:
            # Keep the vec-env worker alive, but do not count missing geometry as a stall.
            truncated = True
            reason = "unmapped_room"
            if self._last_obs is None:
                self._last_obs = np.zeros(OBS_DIM, dtype=np.float32)
        self._prev_action = action_id
        info = self._info(state, reason=reason, joined=joined)
        self._last_info = info
        return self._last_obs.copy(), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        return self._last_rgb

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def _solid_for(self, state: Any) -> RoomSolid:
        room = int(state.room_id)
        if room not in self._solids:
            solid = load_room_solid(room, self._collision_root)
            if solid is None:
                raise CollisionDependencyError(
                    f"editor collision missing for active room 0x{room:04X} in "
                    f"{self._collision_root}; set SUPER_METROID_EDITOR_NAV to an "
                    "sm_nav export containing every room the episode can enter"
                )
            self._solids[room] = solid
        return self._solids[room]

    def _target_for(self, state: Any) -> SteeringTarget:
        return steering_target(
            state,
            self._goal,
            self._solid_for(state),
            graph=self._graph,
            capabilities=capabilities_from_state(state),
        )

    def _observe(
        self,
        state: Any,
        *,
        prev_action: int,
        target: SteeringTarget | None = None,
    ) -> np.ndarray:
        solid = self._solid_for(state)
        selected = target or steering_target(
            state,
            self._goal,
            solid,
            graph=self._graph,
            capabilities=capabilities_from_state(state),
        )
        return observe(
            state,
            self._goal,
            ram=self._env.get_ram(),
            prev_action=prev_action,
            solid=solid.is_solid,
            steer_x=selected.x,
            steer_y=selected.y,
        )

    def _info(self, state: Any, *, reason: str, joined: bool) -> dict[str, Any]:
        telemetry = self._assist.telemetry
        target = getattr(self, "_last_target", None)
        return {
            "session_id": self._row.session_id,
            "goal_session_id": self._goal.session_id,
            "reason": reason,
            "join": joined,
            "frame": self._frame,
            "room": f"0x{int(state.room_id):04X}",
            "xy": [int(state.samus_x), int(state.samus_y)],
            "pose": int(state.pose),
            "gs": int(state.game_state),
            "stall": self._stall,
            "refills": int(telemetry.energy.writes),
            "steer_kind": None if target is None else target.kind,
            "steer_xy": None if target is None else [target.x, target.y],
            "steer_next_room": (
                None
                if target is None or target.next_room_id is None
                else f"0x{target.next_room_id:04X}"
            ),
            "steer_remaining_doors": (
                None if target is None else target.remaining_doors
            ),
            "steer_distance": (
                None if target is None else steering_distance(state, target)
            ),
            "steer_route": (
                []
                if target is None
                else [f"0x{room_id:04X}" for room_id in target.route_rooms]
            ),
            "practice_only": True,
        }


__all__ = [
    "FRAME_SKIP",
    "GeneralistEnv",
    "MAX_EPISODE_FRAMES",
    "STALL_FRAMES",
]

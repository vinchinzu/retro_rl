"""Join-rate eval and the occupancy-aware walk-to-steer heuristic."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from super_metroid.generalist.env import GeneralistEnv
from super_metroid.generalist.obs import GRID_SIZE, GeneralistObs, N_GRID

CENTER = GRID_SIZE // 2
JUMP_ACTIONS = frozenset({8, 9, 10, 11, 18, 19, 20, 23, 24})


def _jump_toward(rel_x: float, *, shoot: bool, previous_action: int) -> int:
    """Pulse Jump: release A for one decision after every jump action."""

    if previous_action in JUMP_ACTIONS:
        if rel_x > 0.02:
            return 3 if shoot else 1
        if rel_x < -0.02:
            return 2 if shoot else 0
        return 25
    if rel_x > 0.02:
        return 20 if shoot else 11
    if rel_x < -0.02:
        return 19 if shoot else 10
    return 18 if shoot else 9


def heuristic_action(obs: GeneralistObs | np.ndarray) -> int:
    """Door-aware BC teacher over the locked observation schema.

    Cross-room steering shoots horizontal blue caps instead of walking into
    them forever.  During a door transition it idles rather than cloning an
    input aimed back through the entry door.  Same-room behavior remains the
    original occupancy-row walk/jump teacher.
    """

    parts = obs if isinstance(obs, GeneralistObs) else GeneralistObs.from_array(obs)
    if parts.ordinary < 0.5 or parts.door_transition > 0.5:
        return 25

    rel_x = parts.goal_dx
    rel_y = parts.goal_dy
    previous_action = parts.previous_action
    same_room = parts.same_room
    grid = parts.grid
    dir_c = 1 if rel_x > 0 else -1
    ahead = CENTER + dir_c
    blocked = False
    if 0 <= ahead < GRID_SIZE:
        blocked = float(grid[CENTER, ahead]) > 0.5
        two = CENTER + 2 * dir_c
        if 0 <= two < GRID_SIZE:
            blocked = blocked or float(grid[CENTER, two]) > 0.5
    if not same_room:
        if blocked or (abs(rel_y) > abs(rel_x) and rel_y < -0.05):
            return _jump_toward(
                rel_x, shoot=True, previous_action=previous_action
            )
        if abs(rel_x) > 0.02:
            return 3 if rel_x > 0 else 2  # run into and shoot horizontal cap
        if rel_y > 0.05:
            return 14  # enter a down door / elevator without morphing
        return 5  # hold up and shoot an up-facing cap
    if blocked or (abs(rel_y) > abs(rel_x) and rel_y < -0.05):
        return _jump_toward(
            rel_x, shoot=False, previous_action=previous_action
        )
    if abs(rel_x) >= abs(rel_y):
        return 1 if rel_x > 0 else 0
    if rel_y > 0.05:
        return 16 if rel_x < 0 else 17
    return 25


def act(state: Any, goal: Any, obs: GeneralistObs | np.ndarray) -> int:
    """Contractor runtime surface. Heuristic today."""

    del state, goal
    return heuristic_action(obs)


def _act(env: GeneralistEnv, policy: Any, obs: np.ndarray) -> int:
    if policy == "random":
        return int(env.action_space.sample())
    if policy == "heuristic":
        return act(None, None, obs)
    action, _ = policy.predict(obs, deterministic=True)
    return int(action)


def eval_join_rate(
    env: GeneralistEnv,
    policy: Any,
    *,
    episodes: int = 8,
    session_id: str | None = None,
) -> dict[str, Any]:
    joins = 0
    stalls = 0
    deaths = 0
    timeouts = 0
    frames: list[int] = []
    refills = 0
    occupancy_hits = 0
    for _ in range(episodes):
        options = {"session_id": session_id} if session_id else {}
        obs, info = env.reset(options=options)
        if float(np.max(np.abs(obs[:N_GRID]))) > 0.0:
            occupancy_hits += 1
        done = False
        while not done:
            obs, _reward, terminated, truncated, info = env.step(_act(env, policy, obs))
            done = terminated or truncated
        frames.append(int(info["frame"]))
        refills += int(info["refills"])
        reason = info["reason"]
        if reason == "join":
            joins += 1
        elif reason == "stall":
            stalls += 1
        elif reason == "death":
            deaths += 1
        elif reason == "timeout":
            timeouts += 1
    n = max(1, episodes)
    return {
        "episodes": episodes,
        "session_id": session_id,
        "join_rate": joins / n,
        "stall_rate": stalls / n,
        "death_rate": deaths / n,
        "timeout_rate": timeouts / n,
        "mean_frames": float(sum(frames) / n),
        "refills": refills,
        "occupancy_filled": occupancy_hits / n,
        "joins": joins,
        "stalls": stalls,
    }


def eval_per_session(
    env: GeneralistEnv,
    policy: Any,
    *,
    episodes: int = 8,
) -> dict[str, Any]:
    by_session: dict[str, Any] = {}
    joins = 0
    stalls = 0
    n = 0
    frames = 0.0
    occ = 0.0
    policy_name = policy if isinstance(policy, str) else type(policy).__name__
    for row in env.rows:
        report = eval_join_rate(
            env, policy, episodes=episodes, session_id=row.session_id
        )
        by_session[row.session_id] = report
        print(
            json.dumps(
                {
                    "event": "eval_session",
                    "policy": policy_name,
                    "session_id": row.session_id,
                    "join_rate": report["join_rate"],
                    "stall_rate": report["stall_rate"],
                    "occupancy_filled": report["occupancy_filled"],
                    "mean_frames": report["mean_frames"],
                }
            ),
            flush=True,
        )
        joins += report["joins"]
        stalls += report["stalls"]
        n += report["episodes"]
        frames += report["mean_frames"] * report["episodes"]
        occ += report["occupancy_filled"] * report["episodes"]
    n = max(1, n)
    return {
        "episodes": n,
        "join_rate": joins / n,
        "stall_rate": stalls / n,
        "mean_frames": frames / n,
        "occupancy_filled": occ / n,
        "sessions": len(by_session),
        "by_session": by_session,
    }


__all__ = [
    "act",
    "eval_join_rate",
    "eval_per_session",
    "heuristic_action",
]

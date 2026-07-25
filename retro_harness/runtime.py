"""Tiny compatibility boundary around Gym and Gymnasium environment APIs."""

from __future__ import annotations

from typing import Any


def reset_env(env: Any) -> tuple[Any, dict[str, Any]]:
    """Reset an environment and normalize old/new API return values."""

    result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], result[1] if isinstance(result[1], dict) else {}
    return result, {}


def step_env(
    env: Any,
    action: Any,
) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
    """Step an environment and always return the Gymnasium five-tuple."""

    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return (
            obs,
            reward,
            bool(terminated),
            bool(truncated),
            info if isinstance(info, dict) else {},
        )
    obs, reward, done, info = result
    return obs, reward, bool(done), False, info if isinstance(info, dict) else {}


__all__ = ["reset_env", "step_env"]

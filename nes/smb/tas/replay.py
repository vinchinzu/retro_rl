"""Shared NES-9 replay helpers for TAS adapt (no L+R sanitize).

All collect / probe / chain code should use these instead of local ``_act``
copies. Simultaneous Left+Right is preserved by design — never pipe frames
through ``sanitize_action``.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

IDLE = np.zeros(9, dtype=np.int8)


def to_action9(frame: list[int] | tuple[int, ...] | np.ndarray | Any) -> np.ndarray:
    """Map a 9-slot (or shorter) button frame to ``np.int8`` action."""
    action = np.zeros(9, dtype=np.int8)
    for j, v in enumerate(list(frame)[:9]):
        action[j] = int(v)
    return action


def get_em(env: Any) -> Any:
    """Return the underlying emulator object (``env.em`` or unwrapped)."""
    if hasattr(env, "em"):
        return env.em
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and hasattr(unwrapped, "em"):
        return unwrapped.em
    raise AttributeError("env has no .em (and unwrapped has no .em)")


def get_state(env: Any) -> Any:
    """Savestate bytes from the emulator."""
    return get_em(env).get_state()


def set_state(env: Any, state: Any) -> None:
    """Restore savestate bytes."""
    get_em(env).set_state(state)


def idle_until(
    env: Any,
    predicate: Callable[[Any], bool],
    *,
    max_wait: int = 600,
    read_snap: Callable[[Any], Any] | None = None,
) -> tuple[int, Any]:
    """Step idle until *predicate(snap)* or *max_wait*.

    Returns ``(wait_frames, last_snapshot)``. Does not count a final idle after
    the predicate already holds on entry.
    """
    if read_snap is None:
        from smb.ram import read_snapshot

        def read_snap(e: Any) -> Any:  # type: ignore[misc]
            return read_snapshot(e.get_ram(), 0)

    wait = 0
    snap = read_snap(env)
    while wait < max_wait:
        snap = read_snap(env)
        if predicate(snap):
            return wait, snap
        env.step(IDLE)
        wait += 1
    return wait, snap


def play_body(
    env: Any,
    frames: list[list[int]] | list,
    *,
    start: int = 0,
    n: int | None = None,
) -> None:
    """Play ``frames[start:start+n]`` (or to end) without observation."""
    body = frames[start:] if n is None else frames[start : start + n]
    for fr in body:
        env.step(to_action9(fr))


def make_level1_env() -> Any:
    """Fresh ``Level1_1`` fceumm env (rgb_array, already ``reset()``)."""
    from retro_harness.env import make_env
    from smb.paths import GAME_DIR, GAME_V0

    env = make_env(GAME_V0, "Level1_1", GAME_DIR, render_mode="rgb_array")
    env.reset()
    return env

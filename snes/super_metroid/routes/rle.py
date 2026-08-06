"""Shared open-loop RLE script types, JSON loader, and frame player.

Used by Spazer hops and Wave double-chamber gate-open. Controllers own
room-specific seating / hit-abort / retry wrappers around :func:`play_script`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import hold
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.geometry import POSE_KNOCKBACK

# (frames, buttons) — one-frame holds played by :func:`play_script`.
RleScript = tuple[tuple[int, tuple[str, ...]], ...]

StopWhen = Callable[[SuperMetroidState], bool]

# Knockback + Chozo item-grab (164) freeze open-loop RLE until cleared.
_RLE_LAG_POSES = POSE_KNOCKBACK | frozenset({164})


def _is_rle_lag_pose(state: SuperMetroidState) -> bool:
    return int(state.pose) in _RLE_LAG_POSES


def _break_rle_lag(
    session: ControllerSession,
    *,
    reason: str = "rle_lag",
    budget: int = 40,
) -> None:
    """Clear knockback / item-grab lag poses that freeze open-loop RLE."""
    for _ in range(budget):
        if not _is_rle_lag_pose(session.state):
            return
        hold(session, 1, "A", reason=reason)
        hold(session, 2, reason=reason)


def load_rle_json(path: str | Path) -> RleScript:
    """Load an RLE script from JSON.

    Accepts a list of ``{"n": frames, "b": [buttons...]}`` objects or
    ``[n, buttons]`` two-element lists. Button lists may be empty.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"RLE JSON must be a list, got {type(raw).__name__}: {path}")
    runs: list[tuple[int, tuple[str, ...]]] = []
    for i, row in enumerate(raw):
        if isinstance(row, dict):
            if "n" not in row:
                raise ValueError(f"RLE row {i} missing 'n': {path}")
            n = int(row["n"])
            if "b" in row:
                buttons = row["b"]
            elif "buttons" in row:
                buttons = row["buttons"]
            else:
                buttons = ()
            if buttons is None:
                buttons = ()
            btns = tuple(str(b) for b in buttons)
        elif isinstance(row, (list, tuple)) and len(row) == 2:
            n = int(row[0])
            buttons = row[1] or ()
            btns = tuple(str(b) for b in buttons)
        else:
            raise ValueError(
                f"RLE row {i} must be {{n,b}} or [n, buttons], got {row!r}: {path}"
            )
        if n < 0:
            raise ValueError(f"RLE row {i} has negative n={n}: {path}")
        runs.append((n, btns))
    return tuple(runs)


def play_script(
    session: ControllerSession,
    runs: RleScript,
    *,
    reason: str,
    room_id: int,
    stop_when: StopWhen | None = None,
    on_lag: str = "ignore",
) -> SuperMetroidState:
    """Play a fixed RLE button script (one-frame holds).

    ``on_lag``:
      * ``ignore`` — leave lag poses alone (door-hop timing may need them).
      * ``break`` — clear lag before each frame (top-drop recovery).
    """
    if on_lag not in ("ignore", "break"):
        raise ValueError(f"on_lag must be 'ignore' or 'break', got {on_lag!r}")
    state = session.state
    for n, btns in runs:
        for _ in range(n):
            if int(session.state.room_id) != room_id:
                return session.state
            if stop_when is not None and stop_when(session.state):
                return session.state
            if on_lag == "break" and _is_rle_lag_pose(session.state):
                _break_rle_lag(session, reason=f"{reason}_lag")
            state = hold(session, 1, *btns, reason=reason)
            if stop_when is not None and stop_when(session.state):
                return session.state
    return state


__all__ = [
    "RleScript",
    "StopWhen",
    "load_rle_json",
    "play_script",
]

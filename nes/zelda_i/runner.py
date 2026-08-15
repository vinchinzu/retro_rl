"""Shared segment-runner helpers for Zelda I scripts.

Thin CLIs should use this module instead of re-copying sys.path / make_env /
assist / JSON report boilerplate. Path logic stays in library controllers.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR

# Repo layout: nes/zelda_i/runner.py → parents[2] = repo root, parents[1] = nes/
_REPO_ROOT = Path(__file__).resolve().parents[2]
_NES_ROOT = Path(__file__).resolve().parents[1]


def ensure_import_paths() -> None:
    """Insert repo + nes roots on sys.path (idempotent)."""
    for p in (_REPO_ROOT, _NES_ROOT):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_state: str | None = None,
    default_tag: str = "run",
    default_trials: int = 1,
) -> argparse.ArgumentParser:
    """Standard flags shared by segment scripts."""
    parser.add_argument(
        "--from-state",
        default=default_state,
        help="Integration save-state name (None = natural boot where supported)",
    )
    parser.add_argument("--tag", default=default_tag)
    parser.add_argument("--trials", type=int, default=default_trials)
    parser.add_argument(
        "--infinite-life",
        action="store_true",
        help="Survival assist (not Clean STATUS)",
    )
    parser.add_argument(
        "--save-state",
        action="store_true",
        help="Write checkpoint on success when controller supports it",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def make_assist(enabled: bool):
    """Return UnlimitedHealthAssist or None."""
    if not enabled:
        return None
    from zelda_i.assist import UnlimitedHealthAssist

    return UnlimitedHealthAssist()


def open_env(
    *,
    from_state: str | None = None,
    seed: int = 0,
    headless: bool = True,
):
    """Create fceumm env for LegendOfZelda-Nes; optionally load a save state."""
    from retro_harness.env import load_state, make_env
    from retro_harness.segment_runner import configure_headless

    if headless:
        configure_headless()
    env = make_env(GAME, seed=seed)
    if from_state:
        load_state(env, GAME_DIR, GAME, from_state)
    return env


_STOP_PHASES = frozenset({"FAILED", "DONE"})


def controller_stopped(controller: Any) -> bool:
    """True when a controller reports success, fail, or a terminal phase.

    Accepts Enum phases (``.name``) and string phases (L3 raft ``"failed"``).
    """
    if getattr(controller, "success", False) or getattr(controller, "failed", False):
        return True
    phase = getattr(controller, "phase", None)
    if phase is None:
        return False
    name = getattr(phase, "name", None)
    token = str(name if name is not None else phase)
    return token.upper() in _STOP_PHASES


def run_controller(
    controller: Any,
    *,
    from_state: str | None,
    infinite_life: bool = False,
    max_frames: int | None = None,
    seed: int = 0,
    on_frame: Callable[[Any, Any, int], None] | None = None,
    step_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Step ``controller`` until success/fail/timeout; return report dict.

    Controller must expose ``.step(snap) -> FrameAction``, ``.success``,
    and preferably ``.report()``. Optional ``.phase`` / ``.failed`` / ``.max_frames``.
    """
    from zelda_i.ram import read_snapshot

    assist = make_assist(infinite_life)
    env = open_env(from_state=from_state, seed=seed)
    extra = step_kwargs or {}
    limit = max_frames
    if limit is None:
        limit = int(getattr(controller, "max_frames", 30000) or 30000)

    try:
        for frame in range(limit):
            snap = read_snapshot(env.get_ram())
            action = controller.step(snap, **extra)
            env.step(action.action)
            if assist is not None:
                assist.apply_env(env, frame=frame)
            if on_frame is not None:
                on_frame(env, controller, frame)
            if controller_stopped(controller):
                break
    finally:
        env.close()

    report: dict[str, Any] = {}
    if hasattr(controller, "report"):
        report = dict(controller.report())
    report.setdefault("success", bool(getattr(controller, "success", False)))
    report["from_state"] = from_state
    report["infinite_life"] = infinite_life
    if assist is not None and hasattr(assist, "report"):
        report["assist"] = assist.report()
    return report


def write_report(name: str, payload: dict[str, Any], *, tag: str = "") -> Path:
    """Write JSON under recordings/; return path."""
    from retro_harness.segment_runner import write_json_report

    stem = f"{name}_{tag}" if tag else name
    out = RECORDINGS_DIR / f"{stem}.json"
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    write_json_report(out, payload)
    return out


__all__ = [
    "add_common_args",
    "controller_stopped",
    "ensure_import_paths",
    "make_assist",
    "open_env",
    "run_controller",
    "write_report",
]

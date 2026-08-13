"""Shared CLI harness for Super Metroid combat / pin probes.

Probe scripts used to copy ``_Session``, ``_resolve_state``, and ``_open_env``.
Import from here instead of recutting those helpers.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from retro_harness.actions import idle_action
from retro_harness.env import make_env, read_state_bytes
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.paths import GAME, GAME_DIR, INTEGRATION_DIR, SCRATCH_STATE_DIR
from super_metroid.ram import parse_state

DEFAULT_STATE_DIRS: tuple[Path, ...] = (
    GAME_DIR,
    INTEGRATION_DIR,
    SCRATCH_STATE_DIR,
    GAME_DIR / "tasks",
)


class ProbeSession:
    """Minimal ControllerSession for pin-local combat / suffix probes."""

    def __init__(self, env: Any, assist: UnlimitedResourcesAssist) -> None:
        self.env = env
        self.assist = assist
        self.frame = 0
        self.action_reasons: Counter[str] = Counter()
        self.state = parse_state(env.get_ram(), frame=0)

    def step(self, action: Any, reason: str) -> Any:
        self.env.step(action)
        self.frame += 1
        self.state = parse_state(self.env.get_ram(), frame=self.frame)
        self.assist.apply(self.env.data, self.state)
        self.action_reasons[reason] += 1
        return self.state


def resolve_named_state(
    name: str,
    named: Mapping[str, Path] | None = None,
    extra_dirs: Sequence[Path] = (),
) -> Path:
    """Resolve a CLI ``--state`` token to a ``.state`` path."""
    key = name.strip()
    if named and key in named:
        return named[key]
    path = Path(key)
    search = (path, *(d / path.name for d in (*DEFAULT_STATE_DIRS, *extra_dirs)))
    if path.suffix == ".state" or "/" in key or path.exists():
        if not path.is_absolute():
            for candidate in search:
                if candidate.exists():
                    return candidate
        return path
    for directory in (*DEFAULT_STATE_DIRS, *extra_dirs):
        candidate = directory / f"{key}.state"
        if candidate.exists():
            return candidate
    return path


def open_state_env(
    spec: Path | str,
    *,
    settle: int = 4,
    missing_hint: str = "",
) -> tuple[Any, str]:
    """Boot a headless env from a save-state path or retro integration name."""
    if isinstance(spec, Path) or str(spec).endswith(".state") or "/" in str(spec):
        path = spec if isinstance(spec, Path) else Path(spec)
        if not path.is_absolute():
            for candidate in (path, GAME_DIR / path, SCRATCH_STATE_DIR / path.name):
                if candidate.exists():
                    path = candidate
                    break
        if not path.exists():
            hint = f"\n{missing_hint}" if missing_hint else ""
            raise FileNotFoundError(f"state not found: {path}{hint}")
        env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
        env.reset()
        env.em.set_state(read_state_bytes(path))
        for _ in range(settle):
            env.step(idle_action())
        return env, str(path)
    env = make_env(GAME, str(spec), GAME_DIR, render_mode="rgb_array")
    env.reset()
    return env, str(spec)


def write_json_report(report: dict[str, Any], path: Path | None = None) -> None:
    """Print a probe report and optionally persist it."""
    text = json.dumps(report, indent=2)
    print(text)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")

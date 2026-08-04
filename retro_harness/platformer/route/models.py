"""Route models, registry, and recording discovery helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from retro_harness.platformer.level_config import LevelConfig


@dataclass
class RouteSegment:
    """One segment of a speedrun route."""

    config_id: str          # registered LevelConfig ID
    label: str = ""         # human-readable label (e.g. "1-1", "8-4 seg3")
    recording: str = ""     # explicit path to recording JSON (relative to runs_dir or absolute)
    neuro_checkpoint: str = ""  # path to neuro_best.json (relative to runs_dir); plays live after recording


@dataclass
class RouteConfig:
    """Ordered list of segments forming a complete speedrun route."""

    route_id: str
    display_name: str
    segments: list[RouteSegment] = field(default_factory=list)


# -- Route registry ----------------------------------------------------------

ROUTE_REGISTRY: dict[str, RouteConfig] = {}


def register_route(route: RouteConfig, *aliases: str) -> None:
    """Register a route config."""
    ROUTE_REGISTRY[route.route_id] = route
    for alias in aliases:
        ROUTE_REGISTRY[alias] = route


def get_platformer_route(route_id: str) -> RouteConfig:
    """Look up a platformer speedrun route by ID or alias.

    Prefer this name over bare ``get_route`` — adventure catalogs use
    ``get_named_route`` for a different type.
    """
    key = route_id.lower() if route_id.lower() in ROUTE_REGISTRY else route_id
    if key not in ROUTE_REGISTRY:
        available = sorted(set(r.route_id for r in ROUTE_REGISTRY.values()))
        raise KeyError(f"Unknown route '{route_id}'. Available: {available}")
    return ROUTE_REGISTRY[key]


# Compat alias (prefer get_platformer_route in new code).
get_route = get_platformer_route


def list_routes() -> list[RouteConfig]:
    """Return deduplicated list of all registered routes."""
    seen: set[str] = set()
    result: list[RouteConfig] = []
    for route in ROUTE_REGISTRY.values():
        if route.route_id not in seen:
            seen.add(route.route_id)
            result.append(route)
    return result


# -- Recording discovery -----------------------------------------------------

def find_best_recording(config: LevelConfig) -> Path | None:
    """Find the best available recording for a level config.

    Priority: hillclimb (if completed) > recording_000.
    Hillclimb results that didn't complete are skipped in favor of
    the original recording which may have raw buttons for faithful replay.
    """
    runs = config.runs_dir
    if not runs.exists():
        return None

    hill = runs / "hillclimb_best_final.json"
    if hill.exists():
        try:
            data = json.loads(hill.read_text())
            if data.get("completed", False):
                return hill
        except (json.JSONDecodeError, KeyError):
            pass

    # Check all recording_*.json for a completed one (prefer highest number)
    for rec_path in sorted(runs.glob("recording_*.json"), reverse=True):
        if "_raw" in rec_path.stem:
            continue
        try:
            data = json.loads(rec_path.read_text())
            if data.get("completed", False):
                return rec_path
        except (json.JSONDecodeError, KeyError):
            pass

    rec = runs / "recording_000.json"
    if rec.exists():
        return rec

    return None


def _load_practice_seeds(
    practice_dir: Path,
    min_frames: int = 60,
) -> list[tuple[list[int] | list[list[int]], bool]]:
    """Load practice attempts as ``(frames, is_raw)`` seed pairs.

    Faithful companion ``*_raw.json`` inputs take precedence. Older attempts
    that only contain action indices remain supported as a fallback.
    """
    from retro_harness.platformer.bk2_extract import load_raw_buttons

    seed_files = sorted(practice_dir.glob("attempt_*.json"))
    seed_files = [f for f in seed_files if "_raw" not in f.stem]

    seeds: list[tuple[list[int] | list[list[int]], bool]] = []
    for f in seed_files:
        try:
            raw = load_raw_buttons(f)
            if raw is not None:
                if len(raw) >= min_frames:
                    seeds.append((raw, True))
                continue
            data = json.loads(f.read_text())
            actions = data["actions"]
            if len(actions) >= min_frames:
                seeds.append((actions, False))
        except (KeyError, OSError, TypeError, json.JSONDecodeError):
            pass
    return seeds


def load_recording_data(path: Path) -> tuple[list, bool]:
    """Load actions from a recording JSON.

    Returns (actions, is_raw).
    Prefers raw buttons (companion _raw.json or embedded) for faithful replay.
    Falls back to action indices if no raw data available.
    """
    from retro_harness.platformer.bk2_extract import load_raw_buttons

    raw = load_raw_buttons(path)
    if raw is not None:
        return raw, True
    data = json.loads(path.read_text())
    return data["actions"], False


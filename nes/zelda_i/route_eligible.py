"""Classify Zelda I checkpoints as lab fixtures vs route-eligible pins.

Single source of truth for compose / stitch: a state is a route pin only when
provenance honestly says so. Name globs, poke/loader writes, and explicit
``route_eligible=false`` cannot be overridden by a friendly filename.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Literal

EligibilityClass = Literal["lab_fixture", "route_pin", "unknown"]

# Name globs that are never route pins (lab / poke / incomplete inventory).
INELIGIBLE_NAME_GLOBS: tuple[tuple[str, str], ...] = (
    ("*ReconFixture", "name matches *ReconFixture"),
    ("L5_Room_*", "name matches L5_Room_*"),
    (
        "Level5Entrance",
        "old Level5Entrance lacks Raft/Stepladder/bombs/TF",
    ),
)

# May become route pins only when provenance is honest.
KNOWN_ROUTE_PIN_NAMES: frozenset[str] = frozenset(
    {
        "Level1Complete",
        "Level1ExitOverworld",
        "Level5EntranceFromL4",
        "Level5Complete",
    }
)

# Assisted / development pins: ``route_eligible=true`` is enough (no natural_entry).
DOCUMENTED_ASSISTED_PINS: frozenset[str] = frozenset(
    {
        "Level5EntranceFromL4",
        "Level5Complete",
    }
)

# Clean natural-entry pins: ``natural_entry=true`` is the documented grant.
DOCUMENTED_NATURAL_PINS: frozenset[str] = frozenset(
    {
        "Level1Complete",
        "Level1ExitOverworld",
    }
)

# Nested maps walked for flags (write_state_provenance + stitch manifests).
_NESTED_MAP_KEYS: tuple[str, ...] = ("request", "selected_trial")

# Truthy values deny eligibility (inventory / door / loader pokes).
_DENY_TRUE_FLAGS: tuple[tuple[str, str], ...] = (
    ("fixture_only", "fixture_only=true"),
    ("inventory_poke", "inventory poke"),
    ("door_poke", "door poke"),
    ("key_poke", "key poke"),
    ("bomb_count_poke", "bomb count poke"),
    ("selected_item_poke", "selected-item poke"),
)

# Non-empty lists mean loader / inventory / door writes.
_DENY_NONEMPTY_LISTS: tuple[tuple[str, str], ...] = (
    ("fixture_writes", "loader/fixture writes"),
    ("loader_writes", "loader writes"),
    ("inventory_writes", "inventory writes"),
    ("door_writes", "door writes"),
)


@dataclass(frozen=True)
class Eligibility:
    """Mechanical lab-fixture vs route-pin decision."""

    name: str
    eligible: bool
    reasons: tuple[str, ...]
    class_: EligibilityClass

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "eligible": self.eligible,
            "reasons": list(self.reasons),
            "class": self.class_,
        }


def load_provenance(path: Path) -> dict[str, Any]:
    """Read a ``*_provenance.json`` / stitch sidecar. Empty dict on missing file."""
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def classify(
    name: str,
    provenance: Mapping[str, Any] | Path | None = None,
) -> Eligibility:
    """Return eligibility for ``name`` plus optional provenance dict or sidecar."""
    payload = _as_mapping(provenance)
    maps = _walk_maps(payload)
    reasons: list[str] = []

    for glob, reason in INELIGIBLE_NAME_GLOBS:
        if fnmatch(name, glob):
            reasons.append(reason)

    route_eligible = _explicit_bool(maps, "route_eligible")
    if route_eligible is False:
        reasons.append("route_eligible=false")

    for key, reason in _DENY_TRUE_FLAGS:
        if _explicit_bool(maps, key) is True:
            reasons.append(reason)

    for key, reason in _DENY_NONEMPTY_LISTS:
        if _nonempty_list(maps, key):
            reasons.append(reason)

    deny = bool(reasons)
    grant_reasons = () if deny else _grant_reasons(name, maps, route_eligible)
    if grant_reasons:
        return Eligibility(
            name=name,
            eligible=True,
            reasons=grant_reasons,
            class_="route_pin",
        )

    if deny:
        class_: EligibilityClass = "lab_fixture"
        if not reasons:
            reasons.append("ineligible")
    elif name in KNOWN_ROUTE_PIN_NAMES:
        class_ = "unknown"
        reasons.append("known pin missing honest route_eligible provenance")
    else:
        class_ = "unknown"
        reasons.append(
            "default ineligible without route_eligible=true and natural_entry=true"
        )

    return Eligibility(
        name=name,
        eligible=False,
        reasons=tuple(reasons),
        class_=class_,
    )


def require_route_pin(
    name: str,
    provenance: Mapping[str, Any] | Path | None = None,
) -> Eligibility:
    """Fail closed: raise if ``name`` is not a route pin."""
    verdict = classify(name, provenance)
    if not verdict.eligible:
        detail = "; ".join(verdict.reasons) or verdict.class_
        raise ValueError(f"{name} is not route_eligible ({verdict.class_}): {detail}")
    return verdict


def _as_mapping(
    provenance: Mapping[str, Any] | Path | None,
) -> Mapping[str, Any]:
    if provenance is None:
        return {}
    if isinstance(provenance, Path):
        return load_provenance(provenance)
    return provenance


def _walk_maps(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    maps: list[Mapping[str, Any]] = [payload]
    for key in _NESTED_MAP_KEYS:
        value = payload.get(key)
        if isinstance(value, Mapping):
            maps.append(value)
    return maps


def _explicit_bool(maps: list[Mapping[str, Any]], key: str) -> bool | None:
    for mapping in maps:
        if key not in mapping:
            continue
        value = mapping[key]
        if isinstance(value, bool):
            return value
        if value is None:
            continue
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
    return None


def _nonempty_list(maps: list[Mapping[str, Any]], key: str) -> bool:
    for mapping in maps:
        value = mapping.get(key)
        if isinstance(value, list) and len(value) > 0:
            return True
    return False


def _grant_reasons(
    name: str,
    maps: list[Mapping[str, Any]],
    route_eligible: bool | None,
) -> tuple[str, ...]:
    natural_entry = _explicit_bool(maps, "natural_entry") is True
    if route_eligible is True and natural_entry:
        return ("route_eligible=true and natural_entry=true",)
    if route_eligible is True and name in DOCUMENTED_ASSISTED_PINS:
        return (f"documented assisted pin {name} with route_eligible=true",)
    if natural_entry and name in DOCUMENTED_NATURAL_PINS:
        return (f"documented Clean natural-entry pin {name}",)
    return ()


__all__ = [
    "DOCUMENTED_ASSISTED_PINS",
    "DOCUMENTED_NATURAL_PINS",
    "Eligibility",
    "EligibilityClass",
    "INELIGIBLE_NAME_GLOBS",
    "KNOWN_ROUTE_PIN_NAMES",
    "classify",
    "load_provenance",
    "require_route_pin",
]

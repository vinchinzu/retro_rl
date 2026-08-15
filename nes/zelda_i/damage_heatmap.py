"""Rank Survival assist ``damage_by_location`` for later Clean residual.

Parses ``UnlimitedHealthAssist`` report dicts (bare telemetry or a runner
JSON with a nested ``assist`` block). No ROM. Does not write recordings.

Known L5 whistle-suffix heat (STITCH_MAP / ``l5_whistle04_to_tf_stitch``):
Digdogger ``0x24``=27, first-key Gibdos ``0x66``=10, west Gibdos ``0x26``=4,
north Dodongos ``0x56``=1, east Zols ``0x57``=1.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# STITCH_MAP L5 suffix totals — fixture tests, not a STATUS claim.
L5_SUFFIX_HEAT: dict[str, int] = {
    "L5:0x24": 27,
    "L5:0x66": 10,
    "L5:0x26": 4,
    "L5:0x56": 1,
    "L5:0x57": 1,
}


@dataclass(frozen=True)
class RoomHeat:
    location: str
    damage: int
    writes: int
    sources: tuple[str, ...]


def l5_suffix_fixture_report() -> dict[str, Any]:
    """Minimal Survival-shaped report matching the published L5 suffix heat."""
    return {
        "enabled": True,
        "class": "survival",
        "kind": "unlimited_health",
        "total_damage": sum(L5_SUFFIX_HEAT.values()),
        "damage_by_location": dict(L5_SUFFIX_HEAT),
        "health": {"restored": 44, "writes": 20},
    }


def extract_assist_block(report: Mapping[str, Any]) -> Mapping[str, Any]:
    """Prefer nested ``assist``; else treat the mapping as telemetry."""
    nested = report.get("assist")
    if isinstance(nested, Mapping) and (
        "damage_by_location" in nested or "total_damage" in nested
    ):
        return nested
    note = report.get("health_drop_note")
    if isinstance(note, Mapping) and "damage_by_location" in note:
        merged = dict(note)
        if "health" in report and "health" not in merged:
            merged["health"] = report["health"]
        if isinstance(report.get("assist"), Mapping):
            assist = report["assist"]
            if "damage_samples" in assist:
                merged.setdefault("damage_samples", assist["damage_samples"])
            if "health" in assist:
                merged.setdefault("health", assist["health"])
        return merged
    return report


def _report_source(report: Mapping[str, Any], fallback: str) -> str:
    for key in ("_source", "source", "segment", "tag"):
        value = report.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _location_from_sample(sample: Mapping[str, Any]) -> str | None:
    loc = sample.get("location")
    if isinstance(loc, str) and loc:
        return loc
    level = sample.get("level")
    screen = sample.get("screen")
    if level is None or screen is None:
        return None
    try:
        return f"L{int(level)}:0x{int(screen):02x}"
    except (TypeError, ValueError):
        return None


def _writes_by_location(block: Mapping[str, Any]) -> Counter[str]:
    writes: Counter[str] = Counter()
    samples = block.get("damage_samples")
    if isinstance(samples, Sequence):
        for sample in samples:
            if not isinstance(sample, Mapping):
                continue
            loc = _location_from_sample(sample)
            if loc:
                writes[loc] += 1
    return writes


def _damage_by_location(block: Mapping[str, Any]) -> dict[str, int]:
    raw = block.get("damage_by_location")
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, int] = {}
    for key, value in raw.items():
        loc = str(key)
        try:
            amount = int(value)
        except (TypeError, ValueError):
            continue
        if amount:
            out[loc] = out.get(loc, 0) + amount
    return out


def rank_damage(
    report: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    source: str = "",
) -> list[RoomHeat]:
    """Hottest rooms first. Ties break on location id."""
    reports: Sequence[Mapping[str, Any]]
    if isinstance(report, Mapping):
        reports = (report,)
    else:
        reports = report

    damage: Counter[str] = Counter()
    writes: Counter[str] = Counter()
    sources: dict[str, list[str]] = {}

    for index, raw in enumerate(reports):
        if not isinstance(raw, Mapping):
            continue
        block = extract_assist_block(raw)
        label = _report_source(raw, source or f"report_{index}")
        loc_writes = _writes_by_location(block)
        for loc, amount in _damage_by_location(block).items():
            damage[loc] += amount
            writes[loc] += loc_writes.get(loc, 0)
            if label not in sources.setdefault(loc, []):
                sources[loc].append(label)
        # Locations that only appear in samples (no damage_by_location row).
        for loc, count in loc_writes.items():
            if loc not in damage:
                writes[loc] += count
                if label not in sources.setdefault(loc, []):
                    sources[loc].append(label)

    ranked = [
        RoomHeat(
            location=loc,
            damage=int(damage[loc]),
            writes=int(writes[loc]),
            sources=tuple(sources.get(loc, ())),
        )
        for loc in damage
    ]
    ranked.sort(key=lambda room: (-room.damage, room.location))
    return ranked


def load_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    payload.setdefault("_source", Path(path).name)
    return payload


def rank_report_paths(paths: Iterable[str | Path]) -> list[RoomHeat]:
    reports = [load_report(path) for path in paths]
    return rank_damage(reports)


def format_heatmap_table(rooms: Sequence[RoomHeat]) -> str:
    if not rooms:
        return "location     damage writes sources\n(no damage_by_location)"
    headers = ("location", "damage", "writes", "sources")
    rows = [
        (
            room.location,
            str(room.damage),
            str(room.writes),
            ",".join(room.sources) if room.sources else "-",
        )
        for room in rooms
    ]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = " ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    body = [
        " ".join(
            cell.rjust(widths[i]) if i in (1, 2) else cell.ljust(widths[i])
            for i, cell in enumerate(row)
        )
        for row in rows
    ]
    return "\n".join([line, *body])


__all__ = [
    "L5_SUFFIX_HEAT",
    "RoomHeat",
    "extract_assist_block",
    "format_heatmap_table",
    "l5_suffix_fixture_report",
    "load_report",
    "rank_damage",
    "rank_report_paths",
]

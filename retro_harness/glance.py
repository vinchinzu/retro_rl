"""Leftover glance: grade a still dict. No MP4.

A leftover mapping in, miss strings out. Games own field names
(room/pose/gs vs screen/mode/triforce vs tilemap/clock). Dual-run JSON
(``runs[]`` / ``final``) is the shared report shape.

Empty misses = glance pass. Leftover is still returned on a miss so the
next agent boots the still, not the pin.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

FinalGrader = Callable[[Mapping[str, Any]], Sequence[str]]
RunPreparer = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]

__all__ = [
    "GlanceLeftover",
    "LeaveMiss",
    "band_miss",
    "grade_report",
    "in_band",
    "parse_int",
    "pick",
    "xy_of",
]


@dataclass(frozen=True)
class GlanceLeftover:
    """Stand leftover. ``leftover`` is present even when ``misses`` is not empty."""

    ok: bool
    leftover: dict[str, Any] = field(default_factory=dict)
    misses: list[str] = field(default_factory=list)


class LeaveMiss(RuntimeError):
    """Hop leave failed. Next agent boots ``.leftover`` (the still), not the pin."""

    hop_id: str
    leftover: dict[str, Any]
    misses: list[str]

    def __init__(
        self,
        hop_id: str,
        leftover: Mapping[str, Any],
        misses: list[str],
        message: str | None = None,
    ) -> None:
        self.hop_id = hop_id
        self.leftover = dict(leftover)
        self.misses = list(misses)
        super().__init__(message or _default_leave_message(hop_id, self.leftover, self.misses))


def _default_leave_message(
    hop_id: str,
    leftover: Mapping[str, Any],
    misses: Sequence[str],
) -> str:
    bits = [f"{hop_id}: leftover={dict(leftover)}"]
    if misses:
        bits.append("misses: " + "; ".join(misses))
    return "; ".join(bits)


def parse_int(value: Any, default: int = 0) -> int:
    """Accept ``0x3A``, ``'0x3a'``, or int. ``None`` → ``default``."""
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text.startswith("0x"):
        return int(text, 16)
    return int(text)


def pick(mapping: Mapping[str, Any], *keys: str) -> Any:
    """First present non-None key."""
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def xy_of(
    final: Mapping[str, Any],
    *,
    x_keys: tuple[str, ...] = ("x",),
    y_keys: tuple[str, ...] = ("y",),
) -> tuple[int, int]:
    """``xy`` pair, else the first matching x/y aliases."""
    if "xy" in final and final["xy"] is not None:
        pair = list(final["xy"])
        return int(pair[0]), int(pair[1])
    x = pick(final, *x_keys)
    y = pick(final, *y_keys)
    return int(x), int(y)


def in_band(value: int, band: int | tuple[int, int] | None) -> bool:
    """Inclusive band. ``None`` is a free pass; a lone int is ``(n, n)``."""
    if band is None:
        return True
    lo, hi = band if isinstance(band, tuple) else (band, band)
    return int(lo) <= int(value) <= int(hi)


def band_miss(name: str, value: int, band: tuple[int, int]) -> str | None:
    if in_band(value, band):
        return None
    return f"{name}={value} not in [{band[0]}, {band[1]}]"


def grade_report(
    report: Mapping[str, Any],
    grade_final: FinalGrader,
    *,
    prepare_final: RunPreparer | None = None,
) -> list[str]:
    """Grade a dual/probe JSON. Both runs must glance-pass when present."""
    misses: list[str] = []
    if report.get("success") is False:
        misses.append("success is false")
    runs = list(report.get("runs") or ())
    if not runs:
        final = report.get("final")
        if not isinstance(final, Mapping):
            return misses + ["missing final"]
        return misses + list(grade_final(final))
    for i, run in enumerate(runs, start=1):
        if not isinstance(run, Mapping):
            misses.append(f"run {i} not an object")
            continue
        if run.get("success") is False:
            misses.append(f"run {i} success is false")
        final = run.get("final")
        if not isinstance(final, Mapping):
            misses.append(f"run {i} missing final")
            continue
        if prepare_final is not None:
            final = prepare_final(final, run)
        for reason in grade_final(final):
            misses.append(f"run {i}: {reason}")
    return misses

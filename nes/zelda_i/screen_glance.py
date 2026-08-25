"""Human-eye screen leave checks from a dual-report final dict.

A glance still (or RAM dump) is enough: wrong room, still cave, fanfare
when the leave is play, TF bit missing, hearts nibble 0xF, xy not in the
door/play band, keys/bombs not the owned HUD counts, door poke. No MP4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from zelda_i.ram import CAVE_MODE, PLAY_MODE

FANFARE_MODE = 18
_HEART_FILL_NIBBLE = 0x0F


@dataclass(frozen=True)
class LeaveSpec:
    """What a human would check in a couple of seconds on a still."""

    hop: str
    room: int
    x: tuple[int, int]
    y: tuple[int, int]
    mode: int = PLAY_MODE
    triforce_bits: int = 0
    keys: int | None = None
    bombs: int | None = None
    hearts_lo_eq_hi: bool = True
    allow_cave: bool = False
    allow_fanfare: bool = False
    require_progression_writes: int = 0


def parse_room(value: Any) -> int:
    """Accept ``0x3A``, ``'0x3a'``, or int."""
    return _as_int(value)


def _as_int(value: Any, default: int = 0) -> int:
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


def _pick(final: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in final and final[key] is not None:
            return final[key]
    return None


def _xy(final: Mapping[str, Any]) -> tuple[int, int]:
    if "xy" in final and final["xy"] is not None:
        pair = list(final["xy"])
        return int(pair[0]), int(pair[1])
    x = _pick(final, "x", "link_x")
    y = _pick(final, "y", "link_y")
    return int(x), int(y)


def _hearts_lo_nibble(health: int) -> int:
    return int(health) & 0x0F


def _hearts_hi_nibble(health: int) -> int:
    return (int(health) >> 4) & 0x0F


def grade_final(final: Mapping[str, Any], spec: LeaveSpec) -> list[str]:
    """Human-readable miss reasons (empty = glance pass)."""
    misses: list[str] = []
    room = parse_room(_pick(final, "room", "screen") or 0)
    if room != spec.room:
        misses.append(f"room 0x{room:02X} != 0x{spec.room:02X}")
    x, y = _xy(final)
    if not (spec.x[0] <= x <= spec.x[1]):
        misses.append(f"x={x} not in [{spec.x[0]}, {spec.x[1]}]")
    if not (spec.y[0] <= y <= spec.y[1]):
        misses.append(f"y={y} not in [{spec.y[0]}, {spec.y[1]}]")
    mode = _as_int(_pick(final, "mode"), default=-1)
    if mode == CAVE_MODE and not spec.allow_cave:
        misses.append(f"mode={mode} cave (allow_cave=False)")
    elif mode == FANFARE_MODE and not spec.allow_fanfare:
        misses.append(f"mode={mode} fanfare (allow_fanfare=False)")
    elif mode != spec.mode:
        misses.append(f"mode={mode} != {spec.mode}")
    tf = _as_int(_pick(final, "triforce", "tf"), default=0)
    missing_tf = spec.triforce_bits & ~tf
    if missing_tf:
        misses.append(
            f"triforce=0x{tf:02X} missing bits 0x{missing_tf:02X}"
        )
    health_raw = _pick(final, "health", "hearts")
    if health_raw is None:
        if spec.hearts_lo_eq_hi:
            misses.append("missing health")
    else:
        health = _as_int(health_raw)
        lo = _hearts_lo_nibble(health)
        hi = _hearts_hi_nibble(health)
        if lo == _HEART_FILL_NIBBLE:
            misses.append(f"health=0x{health:02X} low nibble 0xF")
        elif spec.hearts_lo_eq_hi and lo != hi:
            misses.append(f"health=0x{health:02X} lo!=hi")
    if spec.keys is not None:
        got = _pick(final, "keys")
        if got is None or _as_int(got) != spec.keys:
            misses.append(f"keys={got} != {spec.keys}")
    if spec.bombs is not None:
        got = _pick(final, "bombs")
        if got is None or _as_int(got) != spec.bombs:
            misses.append(f"bombs={got} != {spec.bombs}")
    writes = _as_int(_pick(final, "progression_writes"), default=0)
    if writes != spec.require_progression_writes:
        misses.append(
            f"progression_writes={writes} != {spec.require_progression_writes}"
        )
    if _pick(final, "doors_poked"):
        misses.append("doors_poked")
    return misses


def _with_run_fields(
    final: Mapping[str, Any], run: Mapping[str, Any]
) -> Mapping[str, Any]:
    extra = {}
    for key in ("progression_writes", "doors_poked"):
        if key not in final and key in run:
            extra[key] = run[key]
    if not extra:
        return final
    merged = dict(final)
    merged.update(extra)
    return merged


def grade_report(report: Mapping[str, Any], spec: LeaveSpec) -> list[str]:
    """Grade a dual/probe JSON. Both runs must glance-pass when present."""
    misses: list[str] = []
    if report.get("success") is False:
        misses.append("success is false")
    runs = list(report.get("runs") or ())
    if not runs:
        final = report.get("final")
        if not isinstance(final, Mapping):
            return misses + ["missing final"]
        return misses + grade_final(final, spec)
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
        final = _with_run_fields(final, run)
        for reason in grade_final(final, spec):
            misses.append(f"run {i}: {reason}")
    return misses

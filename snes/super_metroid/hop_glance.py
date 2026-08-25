"""Human-eye hop leave checks from a dual-report final dict.

A glance still (or RAM dump) is enough: wrong room, not gs=8, still morph
when the door needs stand, boss alive, xy not in the door band. No MP4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from super_metroid.routes.controller_common import MORPH_POSES

STAND_POSES = frozenset({1, 2, 9, 10, 12, 27, 28, 137, 138})
AIR_POSES = frozenset({19, 20, 21, 25, 81, 82})

_POSE_CLASS = {
    "stand": STAND_POSES,
    "morph": MORPH_POSES,
    "air": AIR_POSES,
}


@dataclass(frozen=True)
class LeaveSpec:
    """What a human would check in a couple of seconds on a still."""

    hop: str
    room: int
    x: tuple[int, int]
    y: tuple[int, int]
    pose_class: str = "any"
    gs: int = 8
    dt: int = 0
    boss_bit: int | None = None
    min_health: int = 1


def parse_room(value: Any) -> int:
    """Accept ``0xCD13``, ``'0xcd13'``, or int."""
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text.startswith("0x"):
        return int(text, 16)
    return int(text)


def pose_class(pose: int) -> str:
    """stand / morph / air / other."""
    p = int(pose)
    if p in MORPH_POSES:
        return "morph"
    if p in STAND_POSES:
        return "stand"
    if p in AIR_POSES:
        return "air"
    return "other"


def _xy(final: Mapping[str, Any]) -> tuple[int, int]:
    if "xy" in final and final["xy"] is not None:
        pair = list(final["xy"])
        return int(pair[0]), int(pair[1])
    return int(final["x"]), int(final["y"])


def grade_final(final: Mapping[str, Any], spec: LeaveSpec) -> list[str]:
    """Human-readable miss reasons (empty = glance pass)."""
    misses: list[str] = []
    room = parse_room(final.get("room", final.get("room_id", 0)))
    if room != spec.room:
        misses.append(f"room 0x{room:04X} != 0x{spec.room:04X}")
    x, y = _xy(final)
    if not (spec.x[0] <= x <= spec.x[1]):
        misses.append(f"x={x} not in [{spec.x[0]}, {spec.x[1]}]")
    if not (spec.y[0] <= y <= spec.y[1]):
        misses.append(f"y={y} not in [{spec.y[0]}, {spec.y[1]}]")
    pose = int(final.get("pose", -1))
    allowed = _POSE_CLASS.get(spec.pose_class)
    if allowed is not None and pose not in allowed:
        misses.append(
            f"pose {pose} ({pose_class(pose)}) not {spec.pose_class}"
        )
    gs = int(final.get("gs", final.get("game_state", -1)))
    if gs != spec.gs:
        misses.append(f"gs={gs} != {spec.gs}")
    dt = int(final.get("dt", final.get("door_transition", 0)))
    if dt != spec.dt:
        misses.append(f"dt={dt} != {spec.dt}")
    if spec.boss_bit is not None:
        boss = int(final.get("boss", final.get("boss_bit", 0)))
        if boss != spec.boss_bit:
            misses.append(f"boss={boss} != {spec.boss_bit}")
    health = int(final.get("health", spec.min_health))
    if health < spec.min_health:
        misses.append(f"health={health} < {spec.min_health}")
    return misses


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
        if spec.boss_bit is not None and "boss" not in final and "boss" in run:
            final = dict(final)
            final["boss"] = run["boss"]
        for reason in grade_final(final, spec):
            misses.append(f"run {i}: {reason}")
    return misses

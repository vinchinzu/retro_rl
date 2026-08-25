"""Human-eye hop leave checks from a dual-report final dict.

A glance still (or RAM dump) is enough: wrong room, not gs=8, still morph
when the door needs stand, boss alive, xy not in the door band. No MP4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from super_metroid.routes.controller_common import MORPH_POSES
from super_metroid.routes.kpdr.room_ids import ROOM_WS_BASEMENT, ROOM_WS_MAIN

__all__ = [
    "LeaveMiss",
    "LeaveSpec",
    "PHANTOON_LEAVE",
    "WS_BASEMENT_TO_MAIN",
    "WS_ENTRANCE_TO_MAIN",
    "final_from_state",
    "grade_final",
    "grade_leave",
    "grade_report",
    "parse_room",
    "pose_class",
]

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


# Canonical dest glances — compose and tests import these (do not copy numbers).
WS_ENTRANCE_TO_MAIN = LeaveSpec(
    hop="ws_entrance_to_main",
    room=ROOM_WS_MAIN,
    x=(1000, 1100),
    y=(880, 940),
    pose_class="stand",
)
PHANTOON_LEAVE = LeaveSpec(
    hop="phantoon_loot_exit",
    room=ROOM_WS_BASEMENT,
    x=(1200, 1280),
    y=(120, 160),
    pose_class="stand",
    boss_bit=1,
)
# Residual hop dest (not on POST_ICE_SPINE). Main Shaft floor hatch ~(1144,1900).
# RED leftover is the still (basement hatch ~(630-690, 160-190)), not this band.
WS_BASEMENT_TO_MAIN = LeaveSpec(
    hop="ws_basement_to_main",
    room=ROOM_WS_MAIN,
    x=(1100, 1200),
    y=(1800, 2000),
    pose_class="stand",
)


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
        *,
        room_label: str | None = None,
        to_room: int | None = None,
    ) -> None:
        self.hop_id = hop_id
        self.leftover = dict(leftover)
        self.misses = list(misses)
        super().__init__(
            _leave_miss_message(
                hop_id, self.leftover, self.misses, room_label=room_label, to_room=to_room
            )
        )


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


def _int_attr(state: Any, *names: str, default: int = 0) -> int:
    for name in names:
        if hasattr(state, name):
            val = getattr(state, name)
            if val is not None:
                return int(val)
    return int(default)


def _boss_from_state(state: Any) -> int | None:
    for name in ("boss", "boss_bit"):
        if hasattr(state, name):
            val = getattr(state, name)
            if val is not None:
                return int(val)
    bits = getattr(state, "boss_bits", None)
    if bits is None:
        return None
    area = _int_attr(state, "area_index", default=3)
    try:
        return int(bits[area]) & 1
    except (IndexError, TypeError):
        return None


def final_from_state(state: Any) -> dict[str, Any]:
    """Glance still from SuperMetroidState or a room/x/y/pose/gs/dt/health duck."""
    room = _int_attr(state, "room_id", "room")
    x = _int_attr(state, "samus_x", "x")
    y = _int_attr(state, "samus_y", "y")
    final: dict[str, Any] = {
        "room": f"0x{room:04X}",
        "xy": [x, y],
        "pose": _int_attr(state, "pose", default=-1),
        "gs": _int_attr(state, "game_state", "gs", default=-1),
        "dt": _int_attr(state, "door_transition", "dt"),
        "health": _int_attr(state, "health"),
    }
    boss = _boss_from_state(state)
    if boss is not None:
        final["boss"] = boss
    return final


def grade_leave(final: Mapping[str, Any], spec: LeaveSpec) -> dict[str, Any]:
    """Leftover is the final still, whether or not glance passes.

    Miss reasons: :func:`grade_final`. Callers must keep leftover on miss.
    """
    leftover = dict(final)
    if leftover.get("xy") is None:
        leftover["xy"] = list(_xy(leftover))
    del spec
    return leftover


def _leave_miss_message(
    hop_id: str,
    leftover: Mapping[str, Any],
    misses: list[str],
    *,
    room_label: str | None,
    to_room: int | None,
) -> str:
    bits: list[str] = []
    got = parse_room(leftover.get("room", leftover.get("room_id", 0)))
    if to_room is not None and got != to_room:
        label = room_label or hop_id
        bits.append(f"expected {label} 0x{to_room:04X}, got 0x{got:04X}")
    try:
        x, y = _xy(leftover)
        xy_text = f"[{x}, {y}]"
    except (KeyError, TypeError, ValueError):
        xy_text = str(leftover.get("xy"))
    bits.append(
        f"leftover xy={xy_text} pose={leftover.get('pose')} gs={leftover.get('gs')}"
    )
    if misses:
        bits.append("misses: " + "; ".join(misses))
    return f"{hop_id}: " + "; ".join(bits)


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

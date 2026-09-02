"""Human-eye hop leave checks from a dual-report final dict.

A glance still (or RAM dump) is enough: wrong room, not gs=8, still morph
when the door needs stand, boss alive, xy not in the door band. No MP4.
Dest-glance numbers live in :mod:`super_metroid.leave_specs`.
"""

from __future__ import annotations

from typing import Any, Mapping

from retro_harness.glance import (
    LeaveMiss as _HarnessLeaveMiss,
    band_miss,
    grade_report as _grade_report,
    parse_int,
    xy_of,
)
from super_metroid.leave_specs import LeaveSpec
from super_metroid.routes.controller_common import MORPH_POSES

__all__ = [
    "LeaveMiss",
    "final_from_state",
    "grade_final",
    "grade_report",
    "parse_room",
    "pose_class",
    "raise_leave_miss",
]

STAND_POSES = frozenset({1, 2, 9, 10, 12, 27, 28, 137, 138})
AIR_POSES = frozenset({19, 20, 21, 25, 81, 82})

_POSE_CLASS = {
    "stand": STAND_POSES,
    "morph": MORPH_POSES,
    "air": AIR_POSES,
    # Doorway leave: stand or spin through. Morph in the door is a miss.
    "door": STAND_POSES | AIR_POSES,
}


class LeaveMiss(_HarnessLeaveMiss):
    """SM leftover miss with room-label wording."""

    def __init__(
        self,
        hop_id: str,
        leftover: Mapping[str, Any],
        misses: list[str],
        *,
        room_label: str | None = None,
        to_room: int | None = None,
    ) -> None:
        super().__init__(
            hop_id,
            leftover,
            misses,
            message=_leave_miss_message(
                hop_id, leftover, misses, room_label=room_label, to_room=to_room
            ),
        )


def parse_room(value: Any) -> int:
    """Accept ``0xCD13``, ``'0xcd13'``, or int."""
    return parse_int(value)


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


def raise_leave_miss(
    state: Any,
    hop_id: str,
    spec: LeaveSpec,
    *,
    room_label: str,
    to_room: int,
    exc: BaseException | None = None,
) -> None:
    """Grade ``state`` against ``spec`` and raise :class:`LeaveMiss`. Never returns."""
    leftover = final_from_state(state)
    misses = list(grade_final(leftover, spec))
    if exc is not None:
        misses.append(f"{type(exc).__name__}: {exc}")
    raise LeaveMiss(
        hop_id,
        leftover,
        misses or ["leave failed"],
        room_label=room_label,
        to_room=to_room,
    ) from exc


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
        x, y = xy_of(leftover)
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
    x, y = xy_of(final)
    x_miss = band_miss("x", x, spec.x)
    if x_miss:
        misses.append(x_miss)
    y_miss = band_miss("y", y, spec.y)
    if y_miss:
        misses.append(y_miss)
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
    health = final.get("health")
    if health is None:
        misses.append("missing health")
    elif int(health) < spec.min_health:
        misses.append(f"health={int(health)} < {spec.min_health}")
    return misses


def grade_report(report: Mapping[str, Any], spec: LeaveSpec) -> list[str]:
    """Grade a dual/probe JSON. Both runs must glance-pass when present."""

    def prepare_final(
        final: Mapping[str, Any], run: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if spec.boss_bit is not None and "boss" not in final and "boss" in run:
            merged = dict(final)
            merged["boss"] = run["boss"]
            return merged
        return final

    return _grade_report(
        report,
        lambda final: grade_final(final, spec),
        prepare_final=prepare_final,
    )

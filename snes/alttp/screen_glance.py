"""Human-eye screen leave checks from a dual-report final dict.

A glance still (or RAM dump) is enough: wrong room, not control-ready,
xy off the door/spawn band, sword below min, Zelda tagalong missing.
No MP4. Hop leftover is progress: grade_* always return leftover even
when misses is non-empty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from retro_harness.glance import (
    GlanceLeftover,
    band_miss,
    grade_report as _grade_report,
    parse_int,
    pick as _pick,
    xy_of,
)
from alttp.ram import (
    FIGHTER_SWORD_LEVEL,
    HYRULE_CASTLE_B1_PIT_ROOM,
    HYRULE_CASTLE_NORTH_CONNECTOR_ROOM,
    HYRULE_CASTLE_NW_ROOM,
)

# Control-ready modules: $10 == 0x07 indoors, 0x09 overworld; $11 == 0.
MODULE_INDOORS = 0x07
MODULE_OVERWORLD = 0x09

# maps/room_50.json entry_spawn / east_door_approach; pad is door default.
_ROOM_50_SPAWN = (448, 2680)
_ROOM_50_EAST = (480, 2680)
_ROOM_50_PAD = 12

# maps/room_01.json down_to_0x72: corridor default 4, well approach 2.
_ROOM_01_CORRIDOR = (760, 120)
_ROOM_01_WELL = (760, 99)
_ROOM_01_CORRIDOR_PAD = 4
_ROOM_01_WELL_PAD = 2

# maps/room_72.json north_to_0x01: guard landing / stair approach; pad default 12.
_ROOM_72_GUARD = (1273, 3665)
_ROOM_72_APPROACH = (1272, 3656)
_ROOM_72_PAD = 12

__all__ = [
    "GlanceLeftover",
    "LeaveSpec",
    "MODULE_INDOORS",
    "MODULE_OVERWORLD",
    "ROOM_01",
    "ROOM_01_LEAVE",
    "ROOM_50",
    "ROOM_50_LEAVE",
    "ROOM_72",
    "ROOM_72_LEAVE",
    "grade_controller",
    "grade_final",
    "grade_leftover",
    "grade_report",
    "leftover_from_mapping",
    "leftover_from_snapshot",
    "parse_room",
]


@dataclass(frozen=True)
class LeaveSpec:
    """What a human would check in a couple of seconds on a still."""

    hop: str
    room: int
    x: tuple[int, int]
    y: tuple[int, int]
    module: int = MODULE_INDOORS
    submodule: int = 0
    sword_min: int = FIGHTER_SWORD_LEVEL
    follower: int | None = None
    keys: int | None = None
    indoors: bool | None = None


def _xy_band(
    a: tuple[int, int],
    b: tuple[int, int],
    pad_a: int,
    pad_b: int | None = None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Inclusive xy band: each point ± its pad, then union."""
    pb = pad_a if pad_b is None else pad_b
    return (
        (min(a[0] - pad_a, b[0] - pb), max(a[0] + pad_a, b[0] + pb)),
        (min(a[1] - pad_a, b[1] - pb), max(a[1] + pad_a, b[1] + pb)),
    )


_ROOM_50_X, _ROOM_50_Y = _xy_band(_ROOM_50_SPAWN, _ROOM_50_EAST, _ROOM_50_PAD)
_ROOM_01_X, _ROOM_01_Y = _xy_band(
    _ROOM_01_CORRIDOR, _ROOM_01_WELL, _ROOM_01_CORRIDOR_PAD, _ROOM_01_WELL_PAD
)
_ROOM_72_X, _ROOM_72_Y = _xy_band(_ROOM_72_GUARD, _ROOM_72_APPROACH, _ROOM_72_PAD)


# Published leftover: verified tip room 0x50 at spawn / east door.
ROOM_50 = LeaveSpec(
    hop="room_50",
    room=HYRULE_CASTLE_NW_ROOM,
    x=_ROOM_50_X,
    y=_ROOM_50_Y,
    module=MODULE_INDOORS,
    submodule=0,
    sword_min=FIGHTER_SWORD_LEVEL,
    follower=None,
    indoors=True,
)
ROOM_50_LEAVE = ROOM_50

# F1 well leftover: room 0x01 north-wall stair column (isolated, not continuous).
ROOM_01 = LeaveSpec(
    hop="room_01",
    room=HYRULE_CASTLE_NORTH_CONNECTOR_ROOM,
    x=_ROOM_01_X,
    y=_ROOM_01_Y,
    module=MODULE_INDOORS,
    submodule=0,
    sword_min=FIGHTER_SWORD_LEVEL,
    follower=None,
    indoors=True,
)
ROOM_01_LEAVE = ROOM_01

# B1 landing leftover: room 0x72 stair column (isolated, not continuous).
ROOM_72 = LeaveSpec(
    hop="room_72",
    room=HYRULE_CASTLE_B1_PIT_ROOM,
    x=_ROOM_72_X,
    y=_ROOM_72_Y,
    module=MODULE_INDOORS,
    submodule=0,
    sword_min=FIGHTER_SWORD_LEVEL,
    follower=None,
    indoors=True,
)
ROOM_72_LEAVE = ROOM_72


def parse_room(value: Any) -> int:
    """Accept ``0x50``, ``'0x50'``, or int. Masks to the room base id."""
    return parse_int(value) & 0xFF


def leftover_from_mapping(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Copy leftover and fill room/module/xy aliases so grade_final can read it."""
    leftover = dict(raw)
    room = _pick(leftover, "room", "room_base_id", "room_id")
    if room is not None:
        leftover.setdefault("room", parse_room(room))
    module = _pick(leftover, "module", "game_mode")
    if module is not None:
        leftover.setdefault("module", parse_int(module))
    if leftover.get("xy") is not None:
        pair = list(leftover["xy"])
        leftover["xy"] = [int(pair[0]), int(pair[1])]
        leftover.setdefault("x", int(pair[0]))
        leftover.setdefault("y", int(pair[1]))
    else:
        x = _pick(leftover, "x", "link_x")
        y = _pick(leftover, "y", "link_y")
        if x is not None and y is not None:
            leftover["xy"] = [int(x), int(y)]
            leftover.setdefault("x", int(x))
            leftover.setdefault("y", int(y))
    sword = _pick(leftover, "sword", "sword_level")
    if sword is not None:
        leftover.setdefault("sword", parse_int(sword))
    screen = _pick(leftover, "screen", "screen_id")
    if screen is not None:
        leftover.setdefault("screen", parse_int(screen))
    keys = _pick(leftover, "keys", "num_keys", "dungeon_key_count")
    if keys is not None:
        leftover.setdefault("keys", parse_int(keys))
    follower = leftover.get("follower")
    if follower is not None:
        leftover["follower"] = parse_int(follower)
    submodule = leftover.get("submodule")
    if submodule is not None:
        leftover["submodule"] = parse_int(submodule)
    if leftover.get("indoors") is not None:
        leftover["indoors"] = bool(leftover["indoors"])
    return leftover


def leftover_from_snapshot(snap: Any) -> dict[str, Any]:
    """Build leftover a later hop can boot from AlttpSnapshot or a still dict."""
    if isinstance(snap, Mapping):
        if not snap:
            return {}
        return _boot_fields(leftover_from_mapping(snap))
    room_raw = getattr(snap, "room_base_id", None)
    if room_raw is None:
        room_raw = getattr(snap, "room_id", getattr(snap, "room", 0))
    x = int(getattr(snap, "link_x", getattr(snap, "x", 0)))
    y = int(getattr(snap, "link_y", getattr(snap, "y", 0)))
    keys_raw = getattr(snap, "num_keys", None)
    if keys_raw is None:
        keys_raw = getattr(snap, "keys", 0)
    leftover: dict[str, Any] = {
        "room": int(room_raw) & 0xFF,
        "module": int(getattr(snap, "game_mode", getattr(snap, "module", -1))),
        "submodule": int(getattr(snap, "submodule", 0)),
        "x": x,
        "y": y,
        "xy": [x, y],
        "sword": int(getattr(snap, "sword_level", getattr(snap, "sword", 0))),
        "follower": int(getattr(snap, "follower", 0)),
        "keys": int(keys_raw),
        "indoors": bool(getattr(snap, "indoors", False)),
        "screen": int(getattr(snap, "screen_id", getattr(snap, "screen", 0))),
    }
    return leftover


def _boot_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Guarantee the leftover keys a later hop boots."""
    x = parse_int(_pick(payload, "x", "link_x"), default=0)
    y = parse_int(_pick(payload, "y", "link_y"), default=0)
    room_raw = _pick(payload, "room", "room_base_id", "room_id")
    keys_raw = _pick(payload, "keys", "num_keys", "dungeon_key_count")
    return {
        "room": parse_room(room_raw) if room_raw is not None else 0,
        "module": parse_int(_pick(payload, "module", "game_mode"), default=-1),
        "submodule": parse_int(payload.get("submodule"), default=0),
        "x": x,
        "y": y,
        "sword": parse_int(_pick(payload, "sword", "sword_level"), default=0),
        "follower": parse_int(payload.get("follower"), default=0),
        "keys": parse_int(keys_raw, default=0),
        "indoors": bool(payload.get("indoors", False)),
        "screen": parse_int(_pick(payload, "screen", "screen_id"), default=0),
        "xy": [x, y],
    }


def grade_final(final: Mapping[str, Any], spec: LeaveSpec) -> list[str]:
    """Human-readable miss reasons (empty = glance pass)."""
    row = leftover_from_mapping(final)
    misses: list[str] = []
    room_raw = _pick(row, "room", "room_base_id", "room_id")
    room = parse_room(room_raw) if room_raw is not None else -1
    if room != spec.room:
        misses.append(f"room 0x{room:02X} != 0x{spec.room:02X}")
    x_raw = _pick(row, "x", "link_x")
    y_raw = _pick(row, "y", "link_y")
    if x_raw is None or y_raw is None:
        misses.append("missing xy")
    else:
        x, y = xy_of(row, x_keys=("x", "link_x"), y_keys=("y", "link_y"))
        x_miss = band_miss("x", x, spec.x)
        if x_miss:
            misses.append(x_miss)
        y_miss = band_miss("y", y, spec.y)
        if y_miss:
            misses.append(y_miss)
    module_raw = _pick(row, "module", "game_mode")
    module = parse_int(module_raw, default=-1) if module_raw is not None else -1
    if module != spec.module:
        misses.append(f"module=0x{module:02X} != 0x{spec.module:02X}")
    sub_raw = row.get("submodule")
    sub = parse_int(sub_raw, default=-1) if sub_raw is not None else -1
    if sub != spec.submodule:
        misses.append(f"submodule={sub} != {spec.submodule}")
    sword_raw = _pick(row, "sword", "sword_level")
    sword = parse_int(sword_raw, default=0) if sword_raw is not None else 0
    if sword < spec.sword_min:
        misses.append(f"sword={sword} < {spec.sword_min}")
    if spec.follower is not None:
        got = parse_int(_pick(row, "follower"), default=0)
        if got != spec.follower:
            misses.append(f"follower={got} != {spec.follower}")
    if spec.keys is not None:
        got = _pick(row, "keys", "num_keys", "dungeon_key_count")
        if got is None or parse_int(got) != spec.keys:
            misses.append(f"keys={got} != {spec.keys}")
    if spec.indoors is not None:
        if "indoors" not in row:
            misses.append("missing indoors")
        elif bool(row["indoors"]) != spec.indoors:
            misses.append(f"indoors={bool(row['indoors'])} != {spec.indoors}")
    return misses


def grade_report(report: Mapping[str, Any], spec: LeaveSpec) -> list[str]:
    """Grade a dual/probe JSON. Both runs must glance-pass when present."""
    return _grade_report(
        report,
        lambda final: grade_final(final, spec),
        prepare_final=lambda final, _run: leftover_from_mapping(final),
    )


def leftover_from_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Pull leftover from a stage / dual / probe report."""
    nested = report.get("controller")
    if isinstance(nested, Mapping):
        raw = nested.get("leftover")
        if isinstance(raw, Mapping) and raw:
            return leftover_from_snapshot(raw)
    raw = report.get("leftover")
    if isinstance(raw, Mapping) and raw:
        return leftover_from_snapshot(raw)
    final = report.get("final")
    if isinstance(final, Mapping) and final:
        return leftover_from_snapshot(final)
    return {}


def leftover_from_controller(controller: Any) -> dict[str, Any]:
    """Read controller.leftover, else report leftover, else a last snapshot."""
    raw = getattr(controller, "leftover", None)
    if isinstance(raw, Mapping) and raw:
        return leftover_from_snapshot(raw)
    report_fn = getattr(controller, "report", None)
    if callable(report_fn):
        nested = report_fn()
        if isinstance(nested, Mapping):
            pulled = leftover_from_report(nested)
            if pulled:
                return pulled
    for attr in ("snap", "last_snap", "snapshot"):
        snap = getattr(controller, attr, None)
        if snap is not None and (
            hasattr(snap, "link_x") or hasattr(snap, "room_id") or hasattr(snap, "room")
        ):
            return leftover_from_snapshot(snap)
    if isinstance(raw, Mapping):
        return leftover_from_snapshot(raw)
    return {}


def grade_leftover(leftover: Mapping[str, Any] | Any, spec: LeaveSpec) -> GlanceLeftover:
    """Grade a leftover still. leftover is always returned."""
    if isinstance(leftover, Mapping):
        payload = leftover_from_snapshot(leftover) if leftover else {}
    else:
        payload = leftover_from_snapshot(leftover) if leftover is not None else {}
    if not payload:
        return GlanceLeftover(ok=False, leftover={}, misses=["missing leftover"])
    misses = grade_final(payload, spec)
    return GlanceLeftover(ok=not misses, leftover=payload, misses=misses)


def grade_controller(controller: Any, spec: LeaveSpec) -> GlanceLeftover:
    """Grade leftover on a hop controller. leftover is always returned."""
    leftover = leftover_from_controller(controller)
    return grade_leftover(leftover, spec)


def room_50_glance(controller: Any) -> GlanceLeftover:
    """Published leftover: room 0x50 spawn / east door, fighter sword, no Zelda."""
    return grade_controller(controller, ROOM_50)


def room_01_glance(controller: Any) -> GlanceLeftover:
    """F1 well leftover: room 0x01 north-wall stair, fighter sword, no Zelda."""
    return grade_controller(controller, ROOM_01)


def room_72_glance(controller: Any) -> GlanceLeftover:
    """B1 landing leftover: room 0x72 stair column, fighter sword, no Zelda."""
    return grade_controller(controller, ROOM_72)

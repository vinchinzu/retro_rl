"""Human-eye screen leave checks from a dual-report final dict.

A glance still (or RAM dump) is enough: wrong room, still cave, fanfare
when the leave is play, TF bit missing, hearts nibble 0xF, xy not in the
door/play band, keys/bombs not the owned HUD counts, door poke. No MP4.

Hop leftover is progress: grade_* always return leftover even when misses
is non-empty. Dest RAM success stays a separate predicate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from zelda_i.ram import CAVE_MODE, PLAY_MODE

FANFARE_MODE = 18
# stairs3a-warp dest is RAM mode 9 cellar 0x08. Not cave (11).
PASSAGE_MODE = 9
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


# Published leftover: l6_clear3a_continuous_v1 play 0x3A (144,141).
CLEAR_3A = LeaveSpec(
    hop="level6-clear3a",
    room=0x3A,
    x=(128, 160),
    y=(133, 149),
    triforce_bits=0x1F,
    keys=4,
    bombs=8,
)

# Cellar 0x08 B endpoint from ROM AttrB[0x08]=0x1D; AttrC lands (96,157).
CELLAR08_LEAVE = LeaveSpec(
    hop="level6-cellar08",
    room=0x1D,
    x=(88, 112),
    y=(149, 165),
    triforce_bits=0x1F,
    keys=4,
    bombs=8,
    hearts_lo_eq_hi=False,
)

# Live leftover: l6_south1d_continuous play 0x2D (120,77).
SOUTH1D_LEAVE = LeaveSpec(
    hop="level6-south1d",
    room=0x2D,
    x=(112, 128),
    y=(69, 93),
    triforce_bits=0x1F,
    keys=4,
    bombs=8,
    hearts_lo_eq_hi=False,
)

# Live leftover: l6_west2d_continuous play 0x2C (224,141); keys stay 4.
WEST2D_LEAVE = LeaveSpec(
    hop="level6-west2d",
    room=0x2C,
    x=(208, 232),
    y=(133, 149),
    triforce_bits=0x1F,
    keys=4,
    bombs=8,
    hearts_lo_eq_hi=False,
)

# Live leftover: l6_north2c_continuous play 0x1C (120,205); keys 4→3.
NORTH2C_LEAVE = LeaveSpec(
    hop="level6-north2c",
    room=0x1C,
    x=(112, 128),
    y=(189, 221),
    triforce_bits=0x1F,
    keys=3,
    bombs=8,
    hearts_lo_eq_hi=False,
)

# Live leftover: l1_bow22_x112_v2 play 0x22 (224,141); keys 1→0. ADDR_BOW=0.
BOW22_LEAVE = LeaveSpec(
    hop="level1-bow",
    room=0x22,
    x=(208, 232),
    y=(133, 149),
    keys=0,
    hearts_lo_eq_hi=False,
)

# Live leftover: l1_bow_cellar mode 9 0x7F (128,141) tile 0x71. ADDR_BOW=0.
BOW_CELLAR_LEAVE = LeaveSpec(
    hop="level1-bow-cellar",
    room=0x7F,
    x=(120, 136),
    y=(133, 149),
    mode=PASSAGE_MODE,
    keys=0,
    hearts_lo_eq_hi=False,
)

# stairs3a-warp dest: mode 9 cellar 0x08 (208,93). Walk-on stairs BLOCKED.
# Spec documents dest; leftover documents where we actually stopped.
STAIRS3A_DEST = LeaveSpec(
    hop="level6-stairs3a-warp",
    room=0x08,
    x=(200, 216),
    y=(85, 101),
    mode=PASSAGE_MODE,
    triforce_bits=0x1F,
    keys=4,
    bombs=8,
    hearts_lo_eq_hi=False,
)


@dataclass
class GlanceLeftover:
    """Glance result. leftover is present even when misses is non-empty."""

    ok: bool
    leftover: dict[str, Any] = field(default_factory=dict)
    misses: list[str] = field(default_factory=list)


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


def leftover_from_mapping(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Copy leftover and fill room/xy aliases so grade_final can read it."""
    leftover = dict(raw)
    room = _pick(leftover, "room", "screen")
    if room is not None:
        leftover.setdefault("room", parse_room(room))
        leftover.setdefault("screen", parse_room(room))
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
    tf = _pick(leftover, "triforce", "tf")
    if tf is not None:
        leftover.setdefault("triforce", _as_int(tf))
    health = _pick(leftover, "health", "hearts")
    if health is not None:
        leftover.setdefault("health", _as_int(health))
    return leftover


def leftover_from_snapshot(snap: Any) -> dict[str, Any]:
    """Build leftover from a ZeldaSnapshot (or duck-typed last frame)."""
    x = int(getattr(snap, "link_x", getattr(snap, "x", 0)))
    y = int(getattr(snap, "link_y", getattr(snap, "y", 0)))
    screen = int(getattr(snap, "screen", getattr(snap, "room", 0)))
    leftover: dict[str, Any] = {
        "x": x,
        "y": y,
        "xy": [x, y],
        "mode": int(getattr(snap, "mode", -1)),
        "screen": screen,
        "room": screen,
        "keys": int(getattr(snap, "keys", 0)),
        "bombs": int(getattr(snap, "bombs", 0)),
        "triforce": int(getattr(snap, "triforce", 0)),
    }
    health = getattr(snap, "health", None)
    if health is None:
        health = getattr(snap, "hearts", None)
    if health is not None:
        leftover["health"] = int(health)
    tile = getattr(snap, "colliding_tile", getattr(snap, "tile", None))
    if tile is not None:
        leftover["tile"] = int(tile)
    for key in ("rod", "bow", "arrows", "map"):
        if hasattr(snap, key):
            leftover[key] = int(getattr(snap, key) or 0)
    return leftover


def leftover_from_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Pull leftover from a stage / dual / probe report."""
    nested = report.get("controller")
    if isinstance(nested, Mapping):
        raw = nested.get("leftover")
        if isinstance(raw, Mapping) and raw:
            return leftover_from_mapping(raw)
    raw = report.get("leftover")
    if isinstance(raw, Mapping) and raw:
        return leftover_from_mapping(raw)
    final = report.get("final")
    if isinstance(final, Mapping) and final:
        return leftover_from_mapping(final)
    return {}


def leftover_from_controller(controller: Any) -> dict[str, Any]:
    """Read controller.leftover, else report leftover, else a last snapshot."""
    raw = getattr(controller, "leftover", None)
    if isinstance(raw, Mapping) and raw:
        return leftover_from_mapping(raw)
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
            hasattr(snap, "link_x") or hasattr(snap, "screen")
        ):
            return leftover_from_snapshot(snap)
    if isinstance(raw, Mapping):
        return leftover_from_mapping(raw)
    return {}


def _grade_leftover(leftover: Mapping[str, Any], spec: LeaveSpec) -> GlanceLeftover:
    payload = leftover_from_mapping(leftover) if leftover else {}
    if not payload:
        return GlanceLeftover(ok=False, leftover={}, misses=["missing leftover"])
    misses = grade_final(payload, spec)
    return GlanceLeftover(ok=not misses, leftover=payload, misses=misses)


def grade_controller(controller: Any, spec: LeaveSpec) -> GlanceLeftover:
    """Grade leftover on a hop controller. leftover is always returned."""
    leftover = leftover_from_controller(controller)
    return _grade_leftover(leftover, spec)


def grade_stage_report(report: Mapping[str, Any], spec: LeaveSpec) -> GlanceLeftover:
    """Grade leftover from report[controller][leftover] / leftover / final."""
    leftover = leftover_from_report(report)
    return _grade_leftover(leftover, spec)


def clear3a_glance(controller: Any) -> GlanceLeftover:
    """Published leftover play 0x3A (144,141). Dest RAM is separate."""
    return grade_controller(controller, CLEAR_3A)


def cellar08_glance(controller: Any) -> GlanceLeftover:
    """Decoded cellar 0x08 B-side leftover play 0x1D (96,157)."""
    return grade_controller(controller, CELLAR08_LEAVE)


def south1d_glance(controller: Any) -> GlanceLeftover:
    """Live 0x1D south leftover play 0x2D (120,77)."""
    return grade_controller(controller, SOUTH1D_LEAVE)


def west2d_glance(controller: Any) -> GlanceLeftover:
    """Live 0x2D west leftover play 0x2C (224,141)."""
    return grade_controller(controller, WEST2D_LEAVE)


def north2c_glance(controller: Any) -> GlanceLeftover:
    """Predicted 0x2C KEY-UP leftover play 0x1C south mouth; keys 4→3."""
    return grade_controller(controller, NORTH2C_LEAVE)


def east3a_glance(controller: Any) -> GlanceLeftover:
    """Legacy east-wall diagnostic has no route-eligible leave."""
    return grade_controller(controller, CLEAR_3A)


level6_clear3a_glance = clear3a_glance

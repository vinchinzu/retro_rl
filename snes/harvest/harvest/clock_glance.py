"""Human-eye hop leave checks from a probe JSON / still dict.

A glance still (or RAM dump) is enough: wrong tilemap, clock frozen,
CrossMap origin-return without shop, plot not cleared, missing wallet
delta. No MP4. HEADLESS: RAM identity only — no emulator, no ROM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from harvest.core.game_clock import (
    ClockTime,
    ClockTimeline,
    clock_from_mapping,
    compare_frame_benches,
    format_segment_time,
)

# harvest.maps.map_config MAP_REGISTRY / ram_catalog ``tilemap``.
FARM_TILEMAP = 0x00
TOWN_TILEMAP = 0x04
SHOP_TILEMAP = 0x1C
HOUSE_TILEMAP = 0x15
MOUNTAIN_TILEMAP = 0x10  # mountain_spring


@dataclass(frozen=True)
class LeaveSpec:
    """What a human would check in a couple of seconds on a still."""

    hop: str
    tilemap: int
    hour: int | tuple[int, int] | None = None
    minute: int | tuple[int, int] | None = None
    clock: ClockTime | None = None
    money_delta: int | None = None
    shipping_delta: int | None = None
    require_shop_interior: bool = False
    require_plot_cleared: bool = False
    forbid_tilemaps: tuple[int, ...] = ()
    clock_must_advance: bool = True
    require_empty: tuple[str, ...] = ()


HopSpec = LeaveSpec


def parse_tilemap(value: Any) -> int:
    """Accept ``0x1C``, ``'0x1c'``, or int."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return int(text, 16) if text.startswith("0x") else int(text)


def glance_bench(before: int | None, after: int | None) -> dict:
    """Play-clock split. 60 fps via format_segment_time; do not reimplement."""
    table = compare_frame_benches(before, after)
    play = format_segment_time(after)
    table.update(frames=play["frames"], seconds=play["seconds"], clock=play["clock"])
    return table


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, (bool, list, dict)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    out.update(overlay)
    return out

def _nested_pick(row: Mapping[str, Any], *keys: str) -> Any:
    for src in (row, row.get("player"), row.get("map"), row.get("date")):
        if isinstance(src, Mapping):
            for key in keys:
                if key in src and src[key] is not None:
                    return src[key]
    return None

def _tilemap_of(row: Mapping[str, Any]) -> int | None:
    raw = _nested_pick(row, "tilemap", "map_id")
    return parse_tilemap(raw) if raw is not None else None

def _clock_of(row: Mapping[str, Any]) -> ClockTime | None:
    for src in (row, row.get("date")):
        if not isinstance(src, Mapping):
            continue
        clock = clock_from_mapping(src)
        if clock is not None:
            return clock
        text = src.get("clock")
        if isinstance(text, str) and ":" in text:
            hour, _, rest = text.partition(":")
            return ClockTime(int(hour), int(rest.split(":")[0]))
    return None

def _sample_rows(row: Mapping[str, Any]) -> list:
    for key in ("clock_samples", "samples", "timeline"):
        val = row.get(key)
        if isinstance(val, list):
            return val
        if isinstance(val, Mapping) and isinstance(val.get("samples"), list):
            return list(val["samples"])
    return []

def _iter_maps(row: Mapping[str, Any]) -> list[int]:
    blobs: list[Any] = list(_sample_rows(row))
    for key in ("maps_seen", "tilemaps_seen", "maps"):
        val = row.get(key)
        if isinstance(val, (list, tuple)):
            blobs.extend(val)
    seen: list[int] = []
    for item in blobs:
        try:
            tm = _tilemap_of(item) if isinstance(item, Mapping) else parse_tilemap(item)
        except (TypeError, ValueError):
            continue
        if tm is not None:
            seen.append(tm)
    return seen

def _named_pair(row: Mapping[str, Any], *names: str) -> tuple[int | None, int | None]:
    before = after = None
    res = row["resources"] if isinstance(row.get("resources"), Mapping) else {}
    origin = row.get("start") if isinstance(row.get("start"), Mapping) else row.get("boot")
    origin = origin if isinstance(origin, Mapping) else {}
    for name in names:
        nested = row.get(name)
        if isinstance(nested, Mapping):
            before = before if before is not None else _as_int(nested.get("before"))
            after = after if after is not None else _as_int(nested.get("after"))
        elif after is None:
            after = _as_int(nested)
        before = before if before is not None else _as_int(row.get(f"{name}_before"))
        before = before if before is not None else _as_int(origin.get(name))
        if row.get(f"{name}_after") is not None:
            after = _as_int(row.get(f"{name}_after"))
        after = after if after is not None else _as_int(res.get(name))
    return before, after

def _delta_miss(label: str, before: int | None, after: int | None, required: int) -> str | None:
    if before is None or after is None:
        return f"{label} delta missing (need before/after, want {required:+d})"
    actual = after - before
    ok = actual == 0 if required == 0 else actual <= required if required < 0 else actual >= required
    if not ok:
        return f"{label} delta {actual:+d} does not meet {required:+d}"
    return None

def _clock_frozen(row: Mapping[str, Any]) -> str | None:
    marks = []
    for sample in _sample_rows(row):
        if isinstance(sample, Mapping) and _clock_of(sample) is not None:
            item = dict(sample)
            tm = _tilemap_of(item)
            if tm is not None:
                item["tilemap"] = tm
            marks.append(item)
    if len(marks) < 2:
        return None
    timeline = ClockTimeline.from_samples(marks)
    if len({m.frame for m in timeline.samples}) < 2:
        return None
    clocks = {(m.clock.hour, m.clock.minute) for m in timeline.samples}
    return None if len(clocks) > 1 else f"clock frozen {timeline.samples[0].clock}"

def _in_band(value: int, band: int | tuple[int, int] | None) -> bool:
    if band is None:
        return True
    lo, hi = band if isinstance(band, tuple) else (band, band)
    return int(lo) <= value <= int(hi)


def grade_final(final: Mapping[str, Any], spec: LeaveSpec) -> list[str]:
    """Human-readable miss reasons (empty = glance pass)."""
    misses: list[str] = []
    inner = final.get("final")
    row = _merge(final, inner) if isinstance(inner, Mapping) else final
    tm = _tilemap_of(row)
    if tm is None:
        misses.append("missing tilemap")
    else:
        if tm != spec.tilemap:
            misses.append(f"tilemap 0x{tm:02X} != 0x{spec.tilemap:02X}")
        if tm in spec.forbid_tilemaps:
            misses.append(f"tilemap 0x{tm:02X} forbidden")
    clock = _clock_of(row)
    needs_clock = spec.clock is not None or spec.hour is not None or spec.minute is not None
    if needs_clock and clock is None:
        misses.append("missing clock")
    elif clock is not None:
        if spec.clock is not None and clock != spec.clock:
            misses.append(f"clock {clock} != {spec.clock}")
        if not _in_band(clock.hour, spec.hour):
            misses.append(f"hour={clock.hour} not in {spec.hour}")
        if not _in_band(clock.minute, spec.minute):
            misses.append(f"minute={clock.minute} not in {spec.minute}")
    frozen = _clock_frozen(row) if spec.clock_must_advance else None
    if frozen:
        misses.append(frozen)
    plot = row.get("plot_cleared")
    plot_ok = str(plot).strip().lower() in {"1", "true", "yes"} if isinstance(plot, str) else bool(plot)
    if spec.require_plot_cleared and not plot_ok:
        misses.append("plot not cleared")
    if spec.money_delta is not None:
        miss = _delta_miss("money", *_named_pair(row, "money"), spec.money_delta)
        if miss:
            misses.append(miss)
    if spec.shipping_delta is not None:
        miss = _delta_miss("shipping", *_named_pair(row, "shipping_money", "shipping"), spec.shipping_delta)
        if miss:
            misses.append(miss)
    shop = row.get("shop_seen") is True or SHOP_TILEMAP in _iter_maps(row)
    if spec.require_shop_interior and not shop:
        pairs = [_named_pair(row, *n) for n in (("money",), ("shipping_money", "shipping"), ("potato_seeds", "stock"))]
        moved = any(b is not None and a is not None and b != a for b, a in pairs)
        origin_miss = tm == FARM_TILEMAP and not moved
        misses.append("shop miss: returned to origin without 0x1C/stock/wallet delta" if origin_miss else "shop 0x1C not seen")
    if spec.require_empty:
        debris = row.get("debris") if isinstance(row.get("debris"), Mapping) else {}
        for key in spec.require_empty:
            raw = debris.get(key) if isinstance(debris, Mapping) and key in debris else row.get(key)
            count = _as_int(raw)
            if count is None:
                misses.append(f"{key} remaining unknown (need 0)")
            elif count != 0:
                misses.append(f"{key} remaining {count}")
    return misses


def grade_report(report: Mapping[str, Any], spec: LeaveSpec) -> list[str]:
    """Grade a probe JSON. Both runs must glance-pass when present."""
    misses: list[str] = []
    if report.get("success") is False:
        misses.append("success is false")
    runs = list(report.get("runs") or ())
    if not runs:
        final = report.get("final")
        return misses + grade_final(_merge(report, final) if isinstance(final, Mapping) else report, spec)
    for i, run in enumerate(runs, start=1):
        if not isinstance(run, Mapping):
            misses.append(f"run {i} not an object")
            continue
        if run.get("success") is False:
            misses.append(f"run {i} success is false")
        final = run.get("final")
        payload = _merge(_merge(report, run), final) if isinstance(final, Mapping) else _merge(report, run)
        if not isinstance(final, Mapping) and _tilemap_of(payload) is None:
            misses.append(f"run {i} missing final")
            continue
        misses.extend(f"run {i}: {reason}" for reason in grade_final(payload, spec))
    return misses


def _clock_from_snapshot(row: Mapping[str, Any]) -> ClockTime | None:
    clock = _clock_of(row)
    if clock is not None:
        return clock
    nested = row.get("clock")
    if not isinstance(nested, Mapping):
        return None
    clock = clock_from_mapping(nested)
    if clock is not None:
        return clock
    text = nested.get("clock")
    if isinstance(text, str) and ":" in text:
        hour_s, _, rest = text.partition(":")
        try:
            return ClockTime(int(hour_s), int(rest.split(":")[0]))
        except ValueError:
            return None
    return None


def leftover_from_snapshot(snap: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize a probe ``_snapshot`` (or still) to grade_final keys.

    Probe snapshots nest ``clock`` as ``{hour, minute, clock}`` and store
    pixel ``pos`` / tile xy. Debris ``samples`` are not clock samples and
    are omitted so leftover is a stand, not a scan dump.
    """
    if not isinstance(snap, Mapping) or not snap:
        return {}
    row = dict(snap)
    for key in ("final", "leftover", "end"):
        inner = snap.get(key)
        if isinstance(inner, Mapping):
            row = _merge(row, inner)
            break

    leftover: dict[str, Any] = {}
    tm = _tilemap_of(row)
    if tm is not None:
        leftover["tilemap"] = tm

    clock = _clock_from_snapshot(row)
    if clock is not None:
        leftover["hour"] = int(clock.hour)
        leftover["minute"] = int(clock.minute)
        leftover["clock"] = clock.to_dict()

    pos = row.get("pos")
    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        leftover["x"] = int(pos[0])
        leftover["y"] = int(pos[1])
    else:
        x = _as_int(row.get("x"))
        y = _as_int(row.get("y"))
        if x is not None:
            leftover["x"] = x
        if y is not None:
            leftover["y"] = y

    tile = row.get("tile")
    if isinstance(tile, (list, tuple)) and len(tile) >= 2:
        leftover["tile"] = [int(tile[0]), int(tile[1])]
    elif "x" in leftover and "y" in leftover:
        leftover["tile"] = [int(leftover["x"]) // 16, int(leftover["y"]) // 16]

    for key in (
        "carry",
        "debris",
        "stamina",
        "money",
        "shipping",
        "shipping_money",
        "plot_cleared",
        "money_before",
        "money_after",
        "shop_seen",
    ):
        if key in row and row[key] is not None:
            leftover[key] = row[key]
    return leftover


@dataclass(frozen=True)
class GlanceLeftover:
    """Stand leftover. ``leftover`` is present even when ``misses`` is not empty."""

    ok: bool
    leftover: dict[str, Any]
    misses: list[str]


def grade_leftover(final: Mapping[str, Any], spec: LeaveSpec) -> GlanceLeftover:
    leftover = leftover_from_snapshot(final)
    misses = grade_final(leftover, spec)
    return GlanceLeftover(ok=not misses, leftover=leftover, misses=list(misses))


def leftover_json(
    snapshot: Mapping[str, Any] | None,
    spec: LeaveSpec,
    *,
    ok: bool,
    **fields: Any,
) -> dict[str, Any]:
    """Probe JSON with leftover still + glance_misses. Tests do not exec the CLI."""
    glance = grade_leftover(snapshot or {}, spec)
    payload = dict(fields)
    payload["ok"] = ok
    payload["final"] = glance.leftover
    payload["leftover"] = glance.leftover
    payload["glance_misses"] = glance.misses
    return payload


# D2 leftover: location stand vs posts-gone. Fail uses the stand so the next
# agent takes off from that still, not a bushes re-run. Hour 18 is legal
# (ADR-0003); do not treat advancing minutes as a frozen-clock miss.
FENCE_STAND = LeaveSpec(
    hop="FENCE_STAND",
    tilemap=FARM_TILEMAP,
    clock_must_advance=False,
)
FENCE_DUMP = FENCE_STAND
D2_FENCE_LEFTOVER = FENCE_STAND
FENCE_DUMP_DONE = LeaveSpec(
    hop="FENCE_DUMP_DONE",
    tilemap=FARM_TILEMAP,
    clock_must_advance=False,
    require_empty=("fences",),
)

_DONE_EMPTY = {
    "fences": ("fences",),
    "stones": ("stones",),
    "bushes": (),
    "rocks": ("large_rocks",),
    "stumps": ("stumps",),
    "all": ("fences", "stones", "large_rocks", "stumps"),
}


def d2_leftover_spec(section: str = "fences", *, done: bool = False) -> LeaveSpec:
    """Fail path: farm stand; success still enforces exhaustive fences/stones."""
    if section == "fences":
        return FENCE_DUMP_DONE if done else FENCE_STAND
    empty = _DONE_EMPTY.get(section, ())
    return LeaveSpec(
        hop=f"d2_{section}_{'done' if done else 'stand'}",
        tilemap=FARM_TILEMAP,
        clock_must_advance=False,
        require_empty=empty if done else (),
    )

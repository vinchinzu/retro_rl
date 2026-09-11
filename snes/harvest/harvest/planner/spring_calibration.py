"""Calibrate ``spring_opt.CostModel`` frame costs from real campaign logs.

Parses `[RUN] f=<frame> date=S<season>D<day> HH:MM $<money> ...` heartbeat
lines and `[DAY_PLAN] Starting phase N/M: NAME (kind)` / `[DAY_PLAN] NAME ->
...` / `[DAY_PLAN] Phase NAME FAILURE: ...` phase markers out of
`logs/spring_d3_30/run*.log`, and reports measured frame deltas per phase
name by linearly interpolating each phase-boundary log line's frame number
between the two nearest `[RUN]` heartbeats (heartbeats land roughly every
2000 frames, so this is "right order of magnitude", not cycle-accurate —
see the module-level caveat in ``PhaseStats``).

Pure Python, no emulator. Used by ``harvest.scripts.spring_plan --calibrate``.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_RUN_RE = re.compile(r"^\[RUN\] f=(\d+) date=S(\d)D(\d+) ")
_START_RE = re.compile(r"^\[DAY_PLAN\] Starting phase \d+/\d+: (\w+)")
_END_OK_RE = re.compile(r"^\[DAY_PLAN\] (\w+) -> (.*)$")
_END_FAIL_RE = re.compile(r"^\[DAY_PLAN\] Phase (\w+) FAILURE: (.*)$")
_DAYCHANGE_RE = re.compile(r"^\[RUN\] day change \S+ -> \S+ at frame=(\d+)")

# Known runtime bugs (rr-20w.3.x) that inflate a phase's frame cost with
# recovery/retry work unrelated to the phase's steady-state cost. Spans
# containing these are excluded from calibration, not just discounted.
_BUG_MARKERS = ("select_carry", "swap_preserve_hoe", "carry slot swap timeout")


@dataclass(frozen=True)
class PhaseSample:
    phase: str
    log: str
    line_start: int
    line_end: int
    frame_start: float
    frame_end: float
    ok: bool
    clean: bool
    detail: str

    @property
    def delta(self) -> float:
        return self.frame_end - self.frame_start


@dataclass(frozen=True)
class PhaseStats:
    phase: str
    n: int
    min_f: float
    median_f: float
    max_f: float
    samples: Tuple[PhaseSample, ...] = field(default_factory=tuple)


def _run_anchors(lines: Sequence[str]) -> List[Tuple[int, int]]:
    anchors = []
    for i, line in enumerate(lines):
        m = _RUN_RE.match(line)
        if m:
            anchors.append((i, int(m.group(1))))
    return anchors


def _frame_at(line_idx: int, anchors: Sequence[Tuple[int, int]]) -> Optional[float]:
    """Interpolate a frame number for ``line_idx`` between RUN heartbeats.

    Linear in *line index*, not real time — the honest caveat is that a
    span with no anchor inside it is only as good as "roughly proportional
    log-line density between its two bracketing anchors", which can be
    badly wrong for a phase that logs unusually much or little (e.g. a
    stuck nav loop that prints many "Push-facing block" lines quickly).
    Prefer samples where the span itself is short (few lines) relative to
    the anchor gap, or spans multiple anchors — see PhaseStats callers.
    """
    if not anchors:
        return None
    if line_idx <= anchors[0][0]:
        if len(anchors) >= 2 and anchors[1][0] != anchors[0][0]:
            (i0, f0), (i1, f1) = anchors[0], anchors[1]
            slope = (f1 - f0) / (i1 - i0)
            return max(0.0, f0 + slope * (line_idx - i0))
        return float(anchors[0][1])
    if line_idx >= anchors[-1][0]:
        if len(anchors) >= 2 and anchors[-1][0] != anchors[-2][0]:
            (i0, f0), (i1, f1) = anchors[-2], anchors[-1]
            slope = (f1 - f0) / (i1 - i0)
            return f1 + slope * (line_idx - i1)
        return float(anchors[-1][1])
    for k in range(len(anchors) - 1):
        i0, f0 = anchors[k]
        i1, f1 = anchors[k + 1]
        if i0 <= line_idx <= i1:
            if i1 == i0:
                return float(f0)
            frac = (line_idx - i0) / (i1 - i0)
            return f0 + frac * (f1 - f0)
    return float(anchors[-1][1])


def parse_phase_samples(path: str | Path) -> List[PhaseSample]:
    """Extract one sample per (start, completion-or-failure) phase attempt."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    anchors = _run_anchors(lines)

    samples: List[PhaseSample] = []
    open_phase: Optional[str] = None
    open_start: Optional[int] = None
    for i, line in enumerate(lines):
        m = _START_RE.match(line)
        if m:
            open_phase, open_start = m.group(1), i
            continue
        if open_phase is None:
            continue
        m_ok = _END_OK_RE.match(line)
        if m_ok and m_ok.group(1) == open_phase:
            span_text = "\n".join(lines[open_start:i + 1])
            clean = not any(bug in span_text for bug in _BUG_MARKERS)
            f0, f1 = _frame_at(open_start, anchors), _frame_at(i, anchors)
            if f0 is not None and f1 is not None:
                samples.append(PhaseSample(
                    phase=open_phase, log=str(path), line_start=open_start,
                    line_end=i, frame_start=f0, frame_end=f1, ok=True,
                    clean=clean, detail=m_ok.group(2),
                ))
            open_phase = None
            continue
        m_fail = _END_FAIL_RE.match(line)
        if m_fail and m_fail.group(1) == open_phase:
            f0, f1 = _frame_at(open_start, anchors), _frame_at(i, anchors)
            if f0 is not None and f1 is not None:
                samples.append(PhaseSample(
                    phase=open_phase, log=str(path), line_start=open_start,
                    line_end=i, frame_start=f0, frame_end=f1, ok=False,
                    clean=False, detail=m_fail.group(2),
                ))
            open_phase = None
            continue
    return samples


def parse_daychange_frames(path: str | Path) -> List[int]:
    """Absolute frame at each overnight rollover — the flat per-day cost
    from ``docs/tasks/rr-20w-idle-day.md`` (idle time included)."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    return [int(m.group(1)) for m in (
        _DAYCHANGE_RE.match(line) for line in lines
    ) if m]


def parse_return_home_to_daychange(path: str | Path) -> List[float]:
    """Frames from `[MULTI_DAY] Start return_home` to that day's rollover.

    This is the practical measurement for ``CostModel.home_sleep_f`` — the
    "day is over, go home and sleep" tail the planner pays regardless of
    how the day's work went.
    """
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    anchors = _run_anchors(lines)
    deltas: List[float] = []
    pending_start: Optional[int] = None
    for i, line in enumerate(lines):
        if "[MULTI_DAY] Start return_home" in line:
            pending_start = i
            continue
        m = _DAYCHANGE_RE.match(line)
        if m and pending_start is not None:
            f0 = _frame_at(pending_start, anchors)
            f1 = float(m.group(1))
            if f0 is not None:
                deltas.append(f1 - f0)
            pending_start = None
    return deltas


def aggregate(samples: Sequence[PhaseSample]) -> Dict[str, PhaseStats]:
    by_phase: Dict[str, List[PhaseSample]] = {}
    for s in samples:
        by_phase.setdefault(s.phase, []).append(s)
    out: Dict[str, PhaseStats] = {}
    for phase, group in by_phase.items():
        clean_ok = [s for s in group if s.ok and s.clean and s.delta > 0]
        if not clean_ok:
            continue
        deltas = [s.delta for s in clean_ok]
        out[phase] = PhaseStats(
            phase=phase, n=len(deltas), min_f=min(deltas),
            median_f=statistics.median(deltas), max_f=max(deltas),
            samples=tuple(clean_ok),
        )
    return out


# ── Mapping to CostModel fields ──────────────────────────────────────────

# (field, default, measured getter, note). The getter takes the aggregate
# phase-stats dict (+ optional extra series) and returns (value, n, method)
# or None if unmeasured. This mirrors, in code, the reasoning written out in
# CostModel's own field comments — kept here too so `--calibrate` prints a
# self-contained measured-vs-default table without re-deriving the ratios
# by hand each time.


def build_report(log_paths: Sequence[str | Path]) -> str:
    all_samples: List[PhaseSample] = []
    daychange_deltas: List[int] = []
    return_home_deltas: List[float] = []
    for p in log_paths:
        all_samples.extend(parse_phase_samples(p))
        changes = parse_daychange_frames(p)
        daychange_deltas.extend(
            b - a for a, b in zip(changes, changes[1:])
        )
        return_home_deltas.extend(parse_return_home_to_daychange(p))

    stats = aggregate(all_samples)

    lines: List[str] = []
    lines.append(f"Calibration source logs: {[str(p) for p in log_paths]}")
    lines.append("")
    lines.append(f"{'phase':22s} {'n':>3s} {'min':>8s} {'median':>8s} {'max':>8s}")
    for phase in sorted(stats):
        s = stats[phase]
        lines.append(f"{phase:22s} {s.n:3d} {s.min_f:8.0f} {s.median_f:8.0f} {s.max_f:8.0f}")

    lines.append("")
    if daychange_deltas:
        lines.append(
            f"day-change frame deltas (flat per-day cost, idle included): "
            f"n={len(daychange_deltas)} min={min(daychange_deltas)} "
            f"median={statistics.median(daychange_deltas):.0f} max={max(daychange_deltas)}"
        )
    if return_home_deltas:
        lines.append(
            f"return_home-start -> day-change (home_sleep_f evidence): "
            f"n={len(return_home_deltas)} min={min(return_home_deltas):.0f} "
            f"median={statistics.median(return_home_deltas):.0f} max={max(return_home_deltas):.0f}"
        )

    lines.append("")
    lines.append("Field-by-field vs. harvest.planner.spring_opt.CostModel defaults:")
    from harvest.planner.spring_opt import CostModel
    base = CostModel()
    mapping = [
        ("grape_first_f", base.grape_first_f, stats.get("MOUNTAIN_BERRY"),
         "MEASURED: whole MOUNTAIN_BERRY phase; run12=1-grape trips, run13=2-grape (grape fix live) -- compare like with like"),
        ("shop_roundtrip_f", base.shop_roundtrip_f, stats.get("BUY_SEEDS"),
         "MEASURED: confirms default"),
        ("establish_first_f/replant_f", (base.establish_first_f, base.establish_replant_f),
         stats.get("CROP_ESTABLISH"),
         "MEASURED (partial, excludes NAV_CROP walk, buffered for nav-stall tail)"),
        ("water composite (ring_f+marginal_f+refill_f)",
         base.water_ring_f + base.water_ring_marginal_f + base.refill_f,
         stats.get("CROP_WATER"),
         "MEASURED composite only (2 rings + 1 refill); 3 terms scaled uniformly"),
        ("harvest_ring_f (per ~8-tile ring)", base.harvest_ring_f, stats.get("HARVEST_ROUTE"),
         "MEASURED n=1 (7 tiles, scaled to 8) — low confidence"),
        ("home_sleep_f", base.home_sleep_f, None,
         f"MEASURED via return_home->day-change ({len(return_home_deltas)} samples), not a DAY_PLAN phase"),
        ("grape_marginal_f", base.grape_marginal_f, None, "GUESS — unmeasured post-fix"),
        ("spa_refill_f / tool_uses_per_spa", (base.spa_refill_f, base.tool_uses_per_spa), None,
         "GUESS — no HOT_SPRING_STAMINA phase observed in calibration logs"),
    ]
    for name, default, s, note in mapping:
        measured = f"n={s.n} median={s.median_f:.0f} ({s.min_f:.0f}-{s.max_f:.0f})" if s else "n/a"
        lines.append(f"  {name:42s} default={default!s:>16s}  measured={measured:26s}  {note}")

    return "\n".join(lines)


__all__ = [
    "PhaseSample", "PhaseStats",
    "parse_phase_samples", "parse_daychange_frames",
    "parse_return_home_to_daychange", "aggregate", "build_report",
]

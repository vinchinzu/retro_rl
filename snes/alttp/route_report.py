"""Shared route-phase / segment reporting for ALTTP live route segments.

Used by ``castle_to_sword`` and ``secret_entrance_clear`` so both stacks share one
result shape and JSON report layout.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from alttp.primitives import CombatResult, PrimitiveResult
from alttp.ram import AlttpSnapshot, snapshot_to_diag


@dataclass
class RoutePhaseResult:
    """Outcome of one named route phase."""

    phase: str
    ok: bool
    frames: int
    snapshot: AlttpSnapshot
    detail: str = ""
    diag: dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentResult:
    """Full multi-phase segment result from a known predecessor.

    ``acceptance`` holds keys that belong to this segment's exit / progress
    contract. ``diagnostics`` is optional log-only state (e.g. later-route
    flags that must not be read as segment success). Prefer one shared type
    with ``report_kind`` over thin per-segment subclasses.
    """

    ok: bool
    phase: str
    frames: int
    snapshot: AlttpSnapshot
    phases: list[RoutePhaseResult] = field(default_factory=list)
    source: str = "unknown"  # natural_boot | state_load_dev | ...
    acceptance: dict[str, bool] = field(default_factory=dict)
    diagnostics: dict[str, bool] = field(default_factory=dict)
    blocker: str = ""
    notes: list[str] = field(default_factory=list)
    report_kind: str = "alttp_segment_report"

    def to_report(self, kind: str | None = None) -> dict[str, Any]:
        report: dict[str, Any] = {
            "kind": kind if kind is not None else self.report_kind,
            "ok": self.ok,
            "phase": self.phase,
            "frames": self.frames,
            "source": self.source,
            "clean_chain": self.source == "natural_boot" and self.ok,
            "development_only": self.source != "natural_boot",
            "acceptance": dict(self.acceptance),
            "blocker": self.blocker,
            "notes": list(self.notes),
            "final": snapshot_to_diag(self.snapshot),
            "phases": [
                {
                    "phase": p.phase,
                    "ok": p.ok,
                    "frames": p.frames,
                    "detail": p.detail,
                    "diag": p.diag or snapshot_to_diag(p.snapshot),
                }
                for p in self.phases
            ],
        }
        if self.diagnostics:
            report["diagnostics"] = dict(self.diagnostics)
        return report


def segment_result_factory(kind: str) -> Callable[..., SegmentResult]:
    """Build a ``result_factory`` that stamps ``report_kind`` (for run_phases)."""

    def _factory(**kwargs: Any) -> SegmentResult:
        kwargs.setdefault("report_kind", kind)
        return SegmentResult(**kwargs)

    return _factory


def phase_from_primitive(
    phase: str,
    result: PrimitiveResult | CombatResult,
    *,
    detail: str | None = None,
) -> RoutePhaseResult:
    """Build a :class:`RoutePhaseResult` from a primitive/combat outcome."""
    return RoutePhaseResult(
        phase=phase,
        ok=result.ok,
        frames=result.frames,
        snapshot=result.snapshot,
        detail=result.reason if detail is None else detail,
        diag=snapshot_to_diag(result.snapshot),
    )

"""Shared route-phase / segment reporting for ALTTP live route segments.

Used by ``castle_to_sword`` and ``sword_to_zelda`` so both stacks share one
result shape and JSON report layout.
"""

from __future__ import annotations

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
    """Full multi-phase segment result from a known predecessor."""

    ok: bool
    phase: str
    frames: int
    snapshot: AlttpSnapshot
    phases: list[RoutePhaseResult] = field(default_factory=list)
    source: str = "unknown"  # natural_boot | state_load_dev | ...
    acceptance: dict[str, bool] = field(default_factory=dict)
    blocker: str = ""
    notes: list[str] = field(default_factory=list)

    def to_report(self, kind: str) -> dict[str, Any]:
        return {
            "kind": kind,
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

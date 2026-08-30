"""Fail-closed one-frame path boundary for unobserved Level 7 stages.

Concrete navigation policies replace these blockers one internal stage at a
time.  A source hypothesis must never press a direction in the cumulative
spine, silently consume a timeout budget, or become route-eligible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_idle_action
from zelda_i.ram import ZeldaSnapshot


class Level7PathController(Protocol):
    """Minimal one-frame controller contract consumed by chapter stages."""

    max_frames: int
    frames: int
    success: bool
    failed: bool

    def step(self, snap: ZeldaSnapshot) -> FrameAction: ...

    def report(self) -> dict[str, object]: ...


@dataclass
class UnverifiedLevel7PathController:
    """Stop immediately when a chapter has no live one-frame policy."""

    stage_id: str
    missing_evidence: str
    max_frames: int = 1
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)

    def report(self) -> dict[str, object]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "spec_id": self.stage_id,
            "evidence": "hypothesis",
            "route_eligible": False,
            "missing_evidence": self.missing_evidence,
            "notes": list(self.notes),
        }

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.failed = True
        note = (
            f"blocked_unverified:{self.stage_id}:"
            f"L{snap.level}:0x{snap.screen:02x}:m{snap.mode}:"
            f"xy={snap.link_x},{snap.link_y}"
        )
        if not self.notes:
            self.notes.append(note)
        return FrameAction(nes_idle_action(), "blocked_unverified")


def unverified_path_controller(
    stage_id: str, missing_evidence: str
) -> UnverifiedLevel7PathController:
    """Return a fresh blocker; controller instances are never shared."""
    return UnverifiedLevel7PathController(stage_id, missing_evidence)


__all__ = [
    "Level7PathController",
    "UnverifiedLevel7PathController",
    "unverified_path_controller",
]

"""Fail-closed one-frame path boundary for unobserved Level 7 stages.

Concrete navigation policies replace these blockers one internal stage at a
time.  A source hypothesis must never press a direction in the cumulative
spine, silently consume a timeout budget, or become route-eligible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_idle_action
from zelda_i.level7.graph import ledger_notes
from zelda_i.ram import ADDR_CANDLE, ADDR_FOOD, ZeldaSnapshot, read_u8


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
    stage_id: str, missing_evidence: str, *, notes: list[str] | None = None
) -> UnverifiedLevel7PathController:
    """Return a fresh blocker; controller instances are never shared."""
    controller = UnverifiedLevel7PathController(stage_id, missing_evidence)
    if notes:
        controller.notes.extend(notes)
    return controller


@dataclass
class HungryGoriyaGateController:
    """Food is a RAM gate; the room itself is still unobserved."""

    stage_id: str = "level7_entry_to_hungry_goriya"
    max_frames: int = 1
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    _env: Any = field(default=None, init=False, repr=False)

    def bind_env(self, env: Any) -> None:
        self._env = env

    def _fail(self, reason: str) -> FrameAction:
        self.failed = True
        if not self.notes:
            self.notes.extend(ledger_notes())
            self.notes.append(reason)
        return FrameAction(nes_idle_action(), reason)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self._env is None:
            return self._fail("hungry_goriya_env_not_bound")
        food = int(read_u8(self._env.get_ram(), ADDR_FOOD))
        if food < 1:
            return self._fail("hungry_goriya_requires_food")
        note = (
            f"blocked_unverified:{self.stage_id}:"
            f"L{snap.level}:0x{snap.screen:02x}:m{snap.mode}"
        )
        return self._fail(note)

    def report(self) -> dict[str, object]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "spec_id": self.stage_id,
            "evidence": "hypothesis",
            "route_eligible": False,
            "writes": 0,
            "notes": list(self.notes),
        }


@dataclass
class RedCandlePickupController:
    """ADDR_CANDLE 1→2 must happen naturally; room id is still unknown."""

    stage_id: str = "level7_red_candle_pickup"
    max_frames: int = 1
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    _env: Any = field(default=None, init=False, repr=False)

    def bind_env(self, env: Any) -> None:
        self._env = env

    def _fail(self, reason: str) -> FrameAction:
        self.failed = True
        if not self.notes:
            self.notes.append(reason)
        return FrameAction(nes_idle_action(), reason)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self._env is None:
            return self._fail("red_candle_env_not_bound")
        candle = int(read_u8(self._env.get_ram(), ADDR_CANDLE))
        if candle >= 2:
            return self._fail("red_candle_room_unobserved")
        return self._fail(
            f"red_candle_still_{candle}:L{snap.level}:0x{snap.screen:02x}"
        )

    def report(self) -> dict[str, object]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "spec_id": self.stage_id,
            "evidence": "hypothesis",
            "route_eligible": False,
            "writes": 0,
            "notes": list(self.notes),
        }


__all__ = [
    "HungryGoriyaGateController",
    "Level7PathController",
    "RedCandlePickupController",
    "UnverifiedLevel7PathController",
    "unverified_path_controller",
]

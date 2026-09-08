"""Idle through a dungeon Triforce fanfare until overworld play.

One controller; each level is a ``TriforceSettleSpec`` row. Success matches
``spine.hops.play_ready`` (OW play, TF bit, optional raft).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_idle_action
from zelda_i.anchors import (
    SCREEN_LEVEL3_ENTRANCE,
    SCREEN_LEVEL4_ENTRANCE,
    SCREEN_LEVEL5_ENTRANCE,
    TF_BIT_L1,
    TF_BIT_L2,
    TF_BIT_L3,
    TF_BIT_L4,
    TF_BIT_L5,
)
from zelda_i.ram import (
    PLAY_MODE,
    SCREEN_LEVEL1_ENTRANCE,
    SCREEN_LEVEL2_ENTRANCE,
    ZeldaSnapshot,
)

__all__ = [
    "POST_L1_SETTLE",
    "POST_L2_SETTLE",
    "POST_L3_SETTLE",
    "POST_L4_SETTLE",
    "POST_L5_SETTLE",
    "POST_L1_SETTLE_MAX_FRAMES",
    "POST_L2_SETTLE_MAX_FRAMES",
    "POST_L3_SETTLE_MAX_FRAMES",
    "POST_L4_SETTLE_MAX_FRAMES",
    "POST_L5_SETTLE_MAX_FRAMES",
    "SETTLE_MAX_FRAMES",
    "TRIFORCE_SETTLES",
    "PostL2SettlePhase",
    "PostL2TriforceSettleController",
    "PostL3SettlePhase",
    "PostL3TriforceSettleController",
    "PostL4SettlePhase",
    "PostL4TriforceSettleController",
    "PostL5SettlePhase",
    "PostL5TriforceSettleController",
    "PostTriforceSettleController",
    "SettlePhase",
    "TriforceSettleController",
    "TriforceSettleSpec",
    "settle_ready",
]


class SettlePhase(Enum):
    WAIT = auto()
    DONE = auto()
    FAILED = auto()


@dataclass(frozen=True)
class TriforceSettleSpec:
    """Per-level leftover: return screen, TF bit, optional raft."""

    spec_id: str
    require_screen: int
    tf_bit: int
    max_frames: int = 2500
    item: str | None = None
    done_note: str = "ow_ready"


def settle_ready(spec: TriforceSettleSpec, snap: ZeldaSnapshot) -> bool:
    """OW play on ``spec.require_screen`` with the shard bit (and raft if set)."""
    if snap.level != 0 or snap.mode != PLAY_MODE or snap.transitioning:
        return False
    if snap.screen != spec.require_screen:
        return False
    if not (snap.triforce & spec.tf_bit):
        return False
    if spec.item is not None and int(getattr(snap, spec.item, 0)) < 1:
        return False
    return True


POST_L1_SETTLE = TriforceSettleSpec(
    "settle_l1_tf",
    SCREEN_LEVEL1_ENTRANCE,
    TF_BIT_L1,
    max_frames=1500,
    done_note="overworld_after_triforce",
)
POST_L2_SETTLE = TriforceSettleSpec(
    "settle_l2_tf",
    SCREEN_LEVEL2_ENTRANCE,
    TF_BIT_L2,
    done_note="overworld_after_l2_triforce",
)
POST_L3_SETTLE = TriforceSettleSpec(
    "settle_l3_tf",
    SCREEN_LEVEL3_ENTRANCE,
    TF_BIT_L3,
    item="raft",
    done_note="post_l3_ow_ready",
)
POST_L4_SETTLE = TriforceSettleSpec(
    "settle_l4_tf",
    SCREEN_LEVEL4_ENTRANCE,
    TF_BIT_L4,
    item="raft",
    done_note="post_l4_ow_ready",
)
POST_L5_SETTLE = TriforceSettleSpec(
    "settle_l5_tf",
    SCREEN_LEVEL5_ENTRANCE,
    TF_BIT_L5,
    done_note="post_l5_ow_ready",
)
TRIFORCE_SETTLES: tuple[TriforceSettleSpec, ...] = (
    POST_L1_SETTLE,
    POST_L2_SETTLE,
    POST_L3_SETTLE,
    POST_L4_SETTLE,
    POST_L5_SETTLE,
)

SETTLE_MAX_FRAMES = POST_L1_SETTLE.max_frames
POST_L1_SETTLE_MAX_FRAMES = POST_L1_SETTLE.max_frames
POST_L2_SETTLE_MAX_FRAMES = POST_L2_SETTLE.max_frames
POST_L3_SETTLE_MAX_FRAMES = POST_L3_SETTLE.max_frames
POST_L4_SETTLE_MAX_FRAMES = POST_L4_SETTLE.max_frames
POST_L5_SETTLE_MAX_FRAMES = POST_L5_SETTLE.max_frames


@dataclass
class TriforceSettleController:
    """Idle until ``settle_ready(spec)``. No walk, no RAM write."""

    spec: TriforceSettleSpec
    phase: SettlePhase = SettlePhase.WAIT
    frames: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.phase = SettlePhase.WAIT
        self.frames = 0
        self.success = False
        self.notes.clear()

    def step(self, snap: ZeldaSnapshot, **_: Any) -> FrameAction:
        self.frames += 1
        if settle_ready(self.spec, snap):
            self.success = True
            if self.phase is not SettlePhase.DONE:
                self.phase = SettlePhase.DONE
                self.notes.append(self.spec.done_note)
            return FrameAction(nes_idle_action(), "settle_done")
        if self.frames >= self.spec.max_frames:
            self.phase = SettlePhase.FAILED
            self.notes.append("timeout")
            return FrameAction(nes_idle_action(), "timeout")
        return FrameAction(nes_idle_action(), "settle_wait")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec.spec_id,
            "require_screen": f"0x{self.spec.require_screen:02x}",
        }


@dataclass
class PostTriforceSettleController(TriforceSettleController):
    spec: TriforceSettleSpec = POST_L1_SETTLE


@dataclass
class PostL2TriforceSettleController(TriforceSettleController):
    spec: TriforceSettleSpec = POST_L2_SETTLE


@dataclass
class PostL3TriforceSettleController(TriforceSettleController):
    spec: TriforceSettleSpec = POST_L3_SETTLE


@dataclass
class PostL4TriforceSettleController(TriforceSettleController):
    spec: TriforceSettleSpec = POST_L4_SETTLE


@dataclass
class PostL5TriforceSettleController(TriforceSettleController):
    spec: TriforceSettleSpec = POST_L5_SETTLE


PostL2SettlePhase = SettlePhase
PostL3SettlePhase = SettlePhase
PostL4SettlePhase = SettlePhase
PostL5SettlePhase = SettlePhase

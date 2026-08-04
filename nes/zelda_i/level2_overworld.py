"""Overworld routing: Level 1 Triforce settle → Level 2 (Moon) approach.

After shard 1, the game holds mode 18 (fanfare) then returns Link to the Level 1
overworld mouth (screen 0x37). That return is **engine-driven**, not a save-state
warp; the controller only idles through it. From 0x37 the agent **walks**.

Probe-stable walk prefix (2026-07-28)::

    0x37 E@y140 → 0x38 S@x120 → 0x48 S@x112 → 0x58
    E@y148–162 → 0x59 N@x112 → 0x49 E → 0x4A E@y141 → 0x4B
    S@x48 → 0x5B (extension; health-sensitive)

Verified controller stop is **0x4A**. Walkthrough target for Level 2 is
overworld screen **0x3C** (Moon door). Continuation past 0x4A needs heart-safe
overworld combat.

Traps:

- Screen **0x79** is a rocky dead-end pocket (enterable from 0x78 east@y180 but
  no east exit). Do not use the naive "right four from start" grid path.
- On 0x37 after settle, only **y≈140** opens east; y≈125 re-enters Level 1.
- Bush screens need mid-height y≈150–160 corridors (same lesson as 0x58).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction
from zelda_i.nav_common import (
    align_and_push,
    on_arrival_edge,
    recover_off_edge,
    swing_action,
    track_stuck,
    unstick_wiggle,
    wake_or_wait_mode,
)
from zelda_i.overworld import (
    LEVEL2_PATH_HOPS,
    LEVEL2_PATH_SCREENS,
    ScreenHop,
)
from zelda_i.ram import (
    PLAY_MODE,
    SCREEN_LEVEL1_ENTRANCE,
    SCREEN_LEVEL2_ENTRANCE,
    ZeldaSnapshot,
    read_snapshot,
)

# --- Geometry / budgets ---
SCREEN_LEVEL2 = SCREEN_LEVEL2_ENTRANCE
LEVEL1_TRIFORCE_BIT = 0x01
SETTLE_MAX_FRAMES = 1500
SEGMENT_MAX_FRAMES = 25000
SWORD_SWING_PERIOD = 10
SWORD_SWING_FRAMES = 3
STUCK_THRESHOLD = 50

# Re-export path tables (single source: overworld.LEVEL2_PATH_HOPS).
# Verified health-stable prefix ends on 0x4A.
assert LEVEL2_PATH_SCREENS[-1] == 0x4A
assert LEVEL2_PATH_SCREENS[0] == 0x37


class SettlePhase(Enum):
    WAIT_FANFARE = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class PostTriforceSettleController:
    """Idle through mode-18 Triforce fanfare until overworld play on 0x37.

    Live probe: ~535 frames of mode 18, then modes 2→3→4 and playable overworld
    around frame 704 at screen 0x37 ~(112, 125) with triforce & 0x01.
    Reloading a mid-fanfare save (Level1Complete) can freeze mode 18; prefer a
    live settle after collection or the Level1ExitOverworld checkpoint.
    """

    phase: SettlePhase = SettlePhase.WAIT_FANFARE
    frames: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.phase = SettlePhase.WAIT_FANFARE
        self.frames = 0
        self.success = False
        self.notes.clear()

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.frames >= SETTLE_MAX_FRAMES:
            self.phase = SettlePhase.FAILED
            self.notes.append("timeout")
            return FrameAction(nes_idle_action(), "timeout")

        if (
            snap.level == 0
            and snap.mode == PLAY_MODE
            and snap.screen == SCREEN_LEVEL1_ENTRANCE
            and (snap.triforce & LEVEL1_TRIFORCE_BIT)
        ):
            self.success = True
            if self.phase is not SettlePhase.DONE:
                self.phase = SettlePhase.DONE
                self.notes.append("overworld_after_triforce")
            return FrameAction(nes_idle_action(), "done")

        return FrameAction(nes_idle_action(), f"settle_mode_{snap.mode}")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
        }


class Level2NavPhase(Enum):
    HOP = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class OverworldToLevel2Controller:
    """Walk from post-Triforce overworld 0x37 through the verified Level 2 path.

    Default stop is screen **0x4A** with sword and triforce bit 0 — the current
    health-stable milestone. Set ``require_level2_screen=True`` to continue
    toward 0x3C once the suffix is promoted.
    """

    hop_index: int = 0
    phase: Level2NavPhase = Level2NavPhase.HOP
    frames: int = 0
    phase_frames: int = 0
    stuck: int = 0
    last_x: int = -1
    last_y: int = -1
    last_screen: int = -1
    success: bool = False
    notes: list[str] = field(default_factory=list)
    require_level2_screen: bool = False
    require_dungeon: bool = False
    hops: tuple[ScreenHop, ...] = LEVEL2_PATH_HOPS

    def reset(self) -> None:
        self.hop_index = 0
        self.phase = Level2NavPhase.HOP
        self.frames = 0
        self.phase_frames = 0
        self.stuck = 0
        self.last_x = -1
        self.last_y = -1
        self.last_screen = -1
        self.success = False
        self.notes.clear()

    def _set_phase(self, phase: Level2NavPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            self.stuck = 0
            if note:
                self.notes.append(note)

    def _swing(self, direction: str, reason: str) -> FrameAction:
        return swing_action(
            self.phase_frames,
            direction,
            reason,
            period=SWORD_SWING_PERIOD,
            hold=SWORD_SWING_FRAMES,
        )

    def _at_stop(self, snap: ZeldaSnapshot) -> bool:
        if self.require_dungeon:
            return snap.level == 2
        if self.require_level2_screen:
            return (
                snap.level == 0
                and snap.mode == PLAY_MODE
                and snap.screen == SCREEN_LEVEL2
                and snap.has_sword
                and bool(snap.triforce & LEVEL1_TRIFORCE_BIT)
            )
        end_screen = self.hops[-1].target if self.hops else SCREEN_LEVEL2
        return (
            self.hop_index >= len(self.hops)
            and snap.level == 0
            and snap.mode == PLAY_MODE
            and snap.screen == end_screen
            and snap.has_sword
            and bool(snap.triforce & LEVEL1_TRIFORCE_BIT)
            and 40 < snap.link_y < 210
        )

    def _advance_hop(self, snap: ZeldaSnapshot, hop: ScreenHop) -> FrameAction | None:
        """If arrived off the entry edge, advance hop index. Return action if done."""
        if (
            snap.screen != hop.target
            or snap.mode not in (PLAY_MODE, 8)
            or snap.transitioning
            or on_arrival_edge(hop.direction, snap)
        ):
            return None

        self.notes.append(f"hop_{self.hop_index}_{hop.target:02x}")
        self.hop_index += 1
        self.stuck = 0
        self.phase_frames = 0
        if self.hop_index >= len(self.hops) and not (
            self.require_level2_screen or self.require_dungeon
        ):
            self.success = True
            self._set_phase(Level2NavPhase.DONE, "path_prefix_complete")
            return FrameAction(nes_idle_action(), "done")
        return FrameAction(nes_idle_action(), "hop_advance")

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        self.stuck, self.last_x, self.last_y, self.last_screen = track_stuck(
            snap,
            last_x=self.last_x,
            last_y=self.last_y,
            last_screen=self.last_screen,
            stuck=self.stuck,
        )

        if self.frames >= SEGMENT_MAX_FRAMES:
            self._set_phase(Level2NavPhase.FAILED, "timeout")
            return FrameAction(nes_idle_action(), "timeout")

        if snap.mode == 17:
            self._set_phase(Level2NavPhase.FAILED, "link_death")
            return FrameAction(nes_idle_action(), "link_death")

        if self._at_stop(snap):
            self.success = True
            self._set_phase(Level2NavPhase.DONE, "level2_path_stop")
            return FrameAction(nes_idle_action(), "done")

        if snap.level == 1:
            return self._swing("DOWN", "exit_l1")

        if snap.transitioning:
            if self.hop_index < len(self.hops):
                return FrameAction(
                    nes_action(self.hops[self.hop_index].direction), "scroll"
                )
            return FrameAction(nes_idle_action(), "scroll_idle")

        if snap.mode not in (PLAY_MODE, 8, 11):
            return wake_or_wait_mode(self.phase_frames, snap.mode)

        if self.hop_index >= len(self.hops):
            if self.require_level2_screen or self.require_dungeon:
                return self._swing("UP", "door_hunt")
            self.success = True
            self._set_phase(Level2NavPhase.DONE, "hops_complete")
            return FrameAction(nes_idle_action(), "done")

        hop = self.hops[self.hop_index]
        advanced = self._advance_hop(snap, hop)
        if advanced is not None:
            return advanced

        if self.stuck > STUCK_THRESHOLD:
            action, self.stuck = unstick_wiggle(self.stuck)
            return action

        edge = recover_off_edge(snap, hop.direction, swing=self._swing)
        if edge is not None:
            return edge

        return align_and_push(
            snap,
            direction=hop.direction,
            reason=f"hop{self.hop_index}",
            align_x=hop.align_x,
            align_y=hop.align_y,
            y_band=hop.y_band,
            stuck=0,  # already handled above
            stuck_threshold=STUCK_THRESHOLD,
            swing=self._swing,
        )

    def report(self) -> dict[str, Any]:
        hop = None
        if self.hop_index < len(self.hops):
            current = self.hops[self.hop_index]
            hop = {
                "index": self.hop_index,
                "target": current.target,
                "direction": current.direction,
            }
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "hop_index": self.hop_index,
            "hop": hop,
            "notes": list(self.notes),
            "stuck": self.stuck,
            "require_level2_screen": self.require_level2_screen,
            "require_dungeon": self.require_dungeon,
        }


def level2_path_prefix_success(ram: np.ndarray) -> bool:
    """Stop on 0x4A after the verified post-L1 walk prefix."""
    snap = read_snapshot(ram)
    return (
        snap.level == 0
        and snap.mode == PLAY_MODE
        and snap.screen == 0x4A
        and snap.has_sword
        and bool(snap.triforce & LEVEL1_TRIFORCE_BIT)
    )


def level2_screen_reached(ram: np.ndarray) -> bool:
    snap = read_snapshot(ram)
    return (
        snap.level == 0
        and snap.mode == PLAY_MODE
        and snap.screen == SCREEN_LEVEL2
        and snap.has_sword
        and bool(snap.triforce & LEVEL1_TRIFORCE_BIT)
    )


def level2_entrance_success(ram: np.ndarray) -> bool:
    snap = read_snapshot(ram)
    return snap.level == 2


def post_triforce_overworld_ready(ram: np.ndarray) -> bool:
    snap = read_snapshot(ram)
    return (
        snap.level == 0
        and snap.mode == PLAY_MODE
        and snap.screen == SCREEN_LEVEL1_ENTRANCE
        and bool(snap.triforce & LEVEL1_TRIFORCE_BIT)
    )

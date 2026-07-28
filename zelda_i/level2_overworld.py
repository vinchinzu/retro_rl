"""Overworld routing: Level 1 Triforce settle → Level 2 (Moon) approach.

After shard 1, the game holds mode 18 (fanfare) then returns Link to the Level 1
overworld mouth (screen 0x37). That return is **engine-driven**, not a save-state
warp; the controller only idles through it. From 0x37 the agent **walks**.

Probe-stable walk prefix (2026-07-28)::

    0x37 E@y140 → 0x38 S@x120 → 0x48 S@x112 → 0x58
    E@y148–162 → 0x59 N@x112 → 0x49 E → 0x4A E@y141 → 0x4B
    S@x48 → 0x5B (extension; health-sensitive)

Verified controller stop is **0x4A**. Walkthrough target for Level 2 is
overworld screen **0x3C** (Moon door): right-4 / up-2 / right-2 / up / left / up
from start
(https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-2-the-moon/).
Continuation past 0x4A needs heart-safe overworld combat.

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
from snes_oneshot.primitives import FrameAction
from zelda_i.ram import (
    PLAY_MODE,
    SCREEN_LEVEL1_ENTRANCE,
    ZeldaSnapshot,
    read_snapshot,
)

# --- Geometry ---
SCREEN_LEVEL2 = 0x3C
LEVEL1_TRIFORCE_BIT = 0x01
SETTLE_MAX_FRAMES = 1500
SEGMENT_MAX_FRAMES = 25000
SWORD_SWING_PERIOD = 10
SWORD_SWING_FRAMES = 3
STUCK_THRESHOLD = 50

# Verified hop list: (target_screen, direction, align_x, align_y, y_band_lo, y_band_hi)
# y_band takes precedence over align_y when both set (bush corridors).
# Verified health-stable prefix ends on 0x4A (arrive and stop). Extension to
# 0x4B→0x5B works in ad-hoc probes but dies too often for a 2/2 gate.
LEVEL2_PATH_HOPS: tuple[tuple[int, str, int | None, int | None, int | None, int | None], ...] = (
    (0x38, "RIGHT", None, 140, None, None),
    (0x48, "DOWN", 120, None, None, None),
    (0x58, "DOWN", 112, None, None, None),
    (0x59, "RIGHT", None, None, 148, 162),
    (0x49, "UP", 112, None, None, None),
    (0x4A, "RIGHT", None, 141, None, None),
)

LEVEL2_PATH_SCREENS: tuple[int, ...] = (0x37,) + tuple(h[0] for h in LEVEL2_PATH_HOPS)


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

        # Fanfare / return cutscene: hold idle (no input required).
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
    hops: tuple[
        tuple[int, str, int | None, int | None, int | None, int | None], ...
    ] = LEVEL2_PATH_HOPS

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

    def _track_stuck(self, snap: ZeldaSnapshot) -> None:
        if (
            snap.link_x == self.last_x
            and snap.link_y == self.last_y
            and snap.screen == self.last_screen
            and not snap.transitioning
        ):
            self.stuck += 1
        else:
            self.stuck = 0
        self.last_x = snap.link_x
        self.last_y = snap.link_y
        self.last_screen = snap.screen

    def _swing(self, direction: str, reason: str) -> FrameAction:
        if self.phase_frames % SWORD_SWING_PERIOD < SWORD_SWING_FRAMES:
            return FrameAction(nes_action(direction, "A"), f"{reason}_slash")
        return FrameAction(nes_action(direction), reason)

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
        # Default milestone: settled on path end hop screen (0x4A)
        end_screen = self.hops[-1][0] if self.hops else SCREEN_LEVEL2
        return (
            self.hop_index >= len(self.hops)
            and snap.level == 0
            and snap.mode == PLAY_MODE
            and snap.screen == end_screen
            and snap.has_sword
            and bool(snap.triforce & LEVEL1_TRIFORCE_BIT)
            and 40 < snap.link_y < 210
        )

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        self._track_stuck(snap)

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
            # Accidental re-entry into Eagle — walk out south.
            return self._swing("DOWN", "exit_l1")

        if snap.transitioning:
            if self.hop_index < len(self.hops):
                return FrameAction(nes_action(self.hops[self.hop_index][1]), "scroll")
            return FrameAction(nes_idle_action(), "scroll_idle")

        if snap.mode not in (PLAY_MODE, 8, 11):
            if self.phase_frames % 30 < 3:
                return FrameAction(nes_action("A"), f"wake_mode_{snap.mode}")
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if self.hop_index >= len(self.hops):
            if self.require_level2_screen or self.require_dungeon:
                # Door hunt placeholder on 0x3C
                return self._swing("UP", "door_hunt")
            self.success = True
            self._set_phase(Level2NavPhase.DONE, "hops_complete")
            return FrameAction(nes_idle_action(), "done")

        tgt, direction, ax, ay, y_lo, y_hi = self.hops[self.hop_index]
        if (
            snap.screen == tgt
            and snap.mode in (PLAY_MODE, 8)
            and not snap.transitioning
        ):
            # Still on the arrival edge from the travel direction — keep walking in.
            on_arrival_edge = (
                (direction == "RIGHT" and snap.link_x > 220)
                or (direction == "LEFT" and snap.link_x < 30)
                or (direction == "UP" and snap.link_y < 70)
                or (direction == "DOWN" and snap.link_y > 200)
            )
            if not on_arrival_edge:
                self.notes.append(f"hop_{self.hop_index}_{tgt:02x}")
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

        if self.stuck > STUCK_THRESHOLD:
            wiggle = ["UP", "DOWN", "LEFT", "RIGHT"][self.stuck % 4]
            if self.stuck > 140:
                self.stuck = 0
            return FrameAction(nes_action(wiggle, "A"), "unstick")

        # Edge recovery after transitions
        if snap.link_y >= 212 and direction != "DOWN":
            return self._swing("UP", "off_south")
        if snap.link_y <= 62 and direction != "UP":
            return self._swing("DOWN", "off_north")
        if snap.link_x >= 232 and direction != "RIGHT":
            return self._swing("LEFT", "off_east")
        if snap.link_x <= 14 and direction != "LEFT":
            return self._swing("RIGHT", "off_west")

        if y_lo is not None and y_hi is not None:
            if snap.link_y < y_lo:
                return self._swing("DOWN", "band_down")
            if snap.link_y > y_hi:
                return self._swing("UP", "band_up")
            return self._swing(direction, f"hop{self.hop_index}")

        if ax is not None and abs(snap.link_x - ax) > 5 and 80 < snap.link_y < 205:
            btn = "LEFT" if snap.link_x > ax else "RIGHT"
            return self._swing(btn, "align_x")
        if ay is not None and abs(snap.link_y - ay) > 5 and 25 < snap.link_x < 230:
            # Critical: on 0x37 only y=140 opens east; force RIGHT near west edge.
            if direction == "RIGHT" and snap.link_x <= 18:
                return self._swing("RIGHT", "enter_corridor")
            # After vertical screen entry (y near 220/60), finish leaving the edge
            # before fine y-align so we do not scrape side rocks.
            if direction == "RIGHT" and snap.link_y > 200:
                return self._swing("UP", "climb_entry")
            if direction == "RIGHT" and snap.link_y < 70:
                return self._swing("DOWN", "drop_entry")
            btn = "UP" if snap.link_y > ay else "DOWN"
            return self._swing(btn, "align_y")
        return self._swing(direction, f"hop{self.hop_index}")

    def report(self) -> dict[str, Any]:
        hop = None
        if self.hop_index < len(self.hops):
            hop = {
                "index": self.hop_index,
                "target": self.hops[self.hop_index][0],
                "direction": self.hops[self.hop_index][1],
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

"""Overworld routing: approach Level 5 (Lizard) via Lost Hills.

Live recon (assisted, 2026-08-06)::

    Door screen **0x0B** (UP @x≈112). Lost Hills maze **0x1B**: four consecutive
    UP transitions (three self-wraps, fourth → 0x0B). Entry room **0x76**
    (south mouth ~(120, 205) after mode-16 settle).

    Pocket trap: entering 0x1B from 0x1C west @y≈140 lands on the east ledge;
    DOWN then LEFT frees the main path before the four-up climb.

Assisted path prefix (from mid-east OW ~0x4A; Survival infinite-life)::

    0x4A N → 0x3A E → 0x3B N → 0x2B E → 0x2C N → 0x1C W@y≈140 → 0x1B
    (free pocket) → UP×4 → 0x0B → door UP → level 5 room 0x76.

Can visit L5 without clearing L2–L4 (first quest). Item: Whistle
(``ADDR_WHISTLE=0x065C``). Boss Digdogger (whistle shrinks). TF bit ``0x10``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.nav_common import align_and_push, unstick_wiggle
from zelda_i.overworld import ScreenHop
from zelda_i.ow_path import OverworldPathController
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot, read_snapshot

# --- Geometry (live recon 2026-08-06); screens from anchors ---
from zelda_i.anchors import (
    LEVEL5_ENTRY_ROOM,
    SCREEN_LEVEL5_DOOR,
    SCREEN_LOST_HILLS,
    TF_BIT_L5 as LEVEL5_TRIFORCE_BIT,
)

LEVEL5_DOOR_X = 112
LEVEL5_LEVEL_ID = 5

# Mid-east approach into Lost Hills (stops on 0x1B after pocket free is separate).
LEVEL5_PATH_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x3A, "UP", align_x=112),
    ScreenHop(0x3B, "RIGHT", align_y=140),
    ScreenHop(0x2B, "UP", align_x=48),
    ScreenHop(0x2C, "RIGHT", align_y=85),
    ScreenHop(0x1C, "UP", align_x=48),
    ScreenHop(0x1B, "LEFT", align_y=140),
)

SEGMENT_MAX_FRAMES = 30000
LOST_HILLS_MAX_FRAMES = 12000
SWORD_SWING_PERIOD = 10
SWORD_SWING_FRAMES = 3
STUCK_THRESHOLD = 50
LOST_HILLS_UPS_REQUIRED = 4
POCKET_FREE_Y = 172
POCKET_FREE_X = 100


class Level5NavPhase(Enum):
    HOP = auto()
    FREE_POCKET = auto()
    LOST_HILLS = auto()
    DOOR = auto()
    DUNGEON_SETTLE = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class OverworldToLevel5Controller(OverworldPathController):
    """Walk hops → free 0x1B pocket → four UP Lost Hills → door → room-ready 0x76.

    Default start assumes mid-east OW near 0x4A (first hop target 0x3A). Pass a
    custom ``hops`` suffix when resuming from another screen. Set
    ``require_dungeon=False`` to stop on door screen 0x0B.
    """

    phase: Level5NavPhase = Level5NavPhase.HOP
    hops: tuple[ScreenHop, ...] = LEVEL5_PATH_HOPS
    require_dungeon: bool = True
    hills_ups: int = 0
    in_scroll: bool = False
    pocket_stage: int = 0  # 0=down, 1=left
    entry_level: int | None = LEVEL5_LEVEL_ID
    entry_room: int | None = LEVEL5_ENTRY_ROOM
    door_screen: int | None = SCREEN_LEVEL5_DOOR
    door_x: int | None = LEVEL5_DOOR_X
    max_frames: int = SEGMENT_MAX_FRAMES
    swing_period: int = SWORD_SWING_PERIOD
    swing_hold: int = SWORD_SWING_FRAMES
    stuck_threshold: int = STUCK_THRESHOLD
    # L5 hop loop historically did not wake non-play modes via allowed_modes;
    # keep base wake for safety but special phases handle their own modes.
    allowed_modes: frozenset[int] = field(
        default_factory=lambda: frozenset({PLAY_MODE, 8, 11, 6, 7, 16, 2, 3, 4})
    )

    def reset(self) -> None:
        super().reset()
        self.hills_ups = 0
        self.in_scroll = False
        self.pocket_stage = 0

    def _at_stop(self, snap: ZeldaSnapshot) -> bool:
        if self.require_dungeon:
            return (
                snap.level == LEVEL5_LEVEL_ID
                and snap.mode == PLAY_MODE
                and snap.screen == LEVEL5_ENTRY_ROOM
            )
        return (
            snap.level == 0
            and snap.mode == PLAY_MODE
            and snap.screen == SCREEN_LEVEL5_DOOR
            and 60 < snap.link_y < 210
        )

    def _on_hop_advanced(
        self, snap: ZeldaSnapshot, completed_hop: ScreenHop
    ) -> FrameAction:
        if self.hop_index >= len(self.hops):
            if completed_hop.target == SCREEN_LOST_HILLS:
                self._set_phase(Level5NavPhase.FREE_POCKET, "lost_hills_arrived")
            else:
                self._set_phase(Level5NavPhase.LOST_HILLS, "hops_done")
            return FrameAction(nes_idle_action(), "hop_advance")
        return FrameAction(nes_idle_action(), "hop_advance")

    def _free_pocket(self, snap: ZeldaSnapshot) -> FrameAction:
        """East ledge from 0x1C: DOWN to y≈172 then LEFT into the main path.

        Already-free if x is west of the ledge (x≲120) and not glued to x≈208.
        """
        if snap.screen != SCREEN_LOST_HILLS:
            self._set_phase(Level5NavPhase.LOST_HILLS, "left_pocket_screen")
            return FrameAction(nes_idle_action(), "pocket_sc")
        # Already on the main path (live free corridor x≲120, not east ledge ~208).
        if snap.link_x <= 120 and snap.link_x < 190:
            self.notes.append("pocket_already_free")
            self._set_phase(Level5NavPhase.LOST_HILLS, "pocket_free")
            return FrameAction(nes_idle_action(), "pocket_done")
        if self.pocket_stage == 0:
            if snap.link_y >= POCKET_FREE_Y - 2:
                self.pocket_stage = 1
                self.notes.append("pocket_down")
                return self._swing("LEFT", "pocket_left")
            return self._swing("DOWN", "pocket_down")
        if snap.link_x <= POCKET_FREE_X:
            self.notes.append("pocket_free")
            self._set_phase(Level5NavPhase.LOST_HILLS, "pocket_free")
            return FrameAction(nes_idle_action(), "pocket_done")
        if self.stuck > self.stuck_threshold:
            action, self.stuck = unstick_wiggle(self.stuck, reason="pocket_unstick")
            return action
        return self._swing("LEFT", "pocket_left")

    def _lost_hills(self, snap: ZeldaSnapshot) -> FrameAction:
        """Four consecutive UP transitions on 0x1B; fourth lands on 0x0B."""
        if snap.level == LEVEL5_LEVEL_ID:
            self._set_phase(Level5NavPhase.DUNGEON_SETTLE, "entered_l5")
            return FrameAction(nes_idle_action(), "dungeon_enter")
        if snap.screen == SCREEN_LEVEL5_DOOR and snap.mode == PLAY_MODE:
            self.notes.append(f"hills_ups_{self.hills_ups}_to_door")
            self._set_phase(Level5NavPhase.DOOR, "door_screen")
            return FrameAction(nes_idle_action(), "door_sc")

        if snap.mode in (6, 7) or snap.transitioning:
            self.in_scroll = True
            return FrameAction(nes_action("UP"), "hills_scroll")

        if self.in_scroll and snap.mode == PLAY_MODE:
            self.in_scroll = False
            if snap.screen == SCREEN_LOST_HILLS:
                self.hills_ups += 1
                self.notes.append(f"hills_wrap_{self.hills_ups}")
            elif snap.screen == SCREEN_LEVEL5_DOOR:
                self.hills_ups += 1
                self.notes.append(f"hills_door_{self.hills_ups}")
                self._set_phase(Level5NavPhase.DOOR, "door_screen")
                return FrameAction(nes_idle_action(), "door_sc")

        if self.stuck > self.stuck_threshold:
            action, self.stuck = unstick_wiggle(self.stuck, reason="hills_unstick")
            return action
        # Live wraps reappear at x≈112; north mouth is not open at x≈96.
        target_x = 112
        if abs(snap.link_x - target_x) > 6:
            btn = "RIGHT" if snap.link_x < target_x else "LEFT"
            return self._swing(btn, "hills_ax")
        return align_and_push(
            snap,
            direction="UP",
            reason="hills_up",
            align_x=target_x,
            stuck=0,
            stuck_threshold=self.stuck_threshold,
            swing=self._swing,
        )

    def _door(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.level == LEVEL5_LEVEL_ID:
            self._set_phase(Level5NavPhase.DUNGEON_SETTLE, "entered_l5")
            return FrameAction(nes_idle_action(), "dungeon_enter")
        if abs(snap.link_x - LEVEL5_DOOR_X) > 5:
            btn = "LEFT" if snap.link_x > LEVEL5_DOOR_X else "RIGHT"
            return self._swing(btn, "door_ax")
        return self._swing("UP", "door_hunt")

    def _before_play(self, snap: ZeldaSnapshot) -> FrameAction | None:
        if self.phase is Level5NavPhase.DUNGEON_SETTLE:
            return FrameAction(nes_idle_action(), "dungeon_settle")
        if self.phase is Level5NavPhase.DOOR:
            return self._door(snap)
        if self.phase is Level5NavPhase.LOST_HILLS:
            return self._lost_hills(snap)
        if self.phase is Level5NavPhase.FREE_POCKET:
            return self._free_pocket(snap)
        return None

    def _after_hops(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.screen == SCREEN_LOST_HILLS:
            self._set_phase(Level5NavPhase.FREE_POCKET, "start_pocket")
            return self._free_pocket(snap)
        if snap.screen == SCREEN_LEVEL5_DOOR:
            self._set_phase(Level5NavPhase.DOOR, "start_door")
            return self._door(snap)
        self._set_phase(Level5NavPhase.LOST_HILLS, "start_hills")
        return self._lost_hills(snap)

    def _handle_transition(self, snap: ZeldaSnapshot) -> FrameAction:
        # Special phases manage their own scroll; only HOP uses hop direction.
        if self.phase is Level5NavPhase.HOP and self.hop_index < len(self.hops):
            return FrameAction(
                nes_action(self.hops[self.hop_index].direction), "scroll"
            )
        if self.phase is Level5NavPhase.LOST_HILLS:
            return self._lost_hills(snap)
        if self.phase is Level5NavPhase.FREE_POCKET:
            return self._free_pocket(snap)
        if self.phase is Level5NavPhase.DOOR:
            return self._door(snap)
        return FrameAction(nes_idle_action(), "scroll_idle")

    def _finish(self, note: str = "path_stop") -> FrameAction:
        label = {
            "path_stop": "level5_stop",
        }.get(note, note)
        self.success = True
        self._set_phase(Level5NavPhase.DONE, label)
        return FrameAction(nes_idle_action(), "done")

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
            "hills_ups": self.hills_ups,
            "require_dungeon": self.require_dungeon,
            "notes": list(self.notes),
            "stuck": self.stuck,
        }


def level5_entrance_success(ram: np.ndarray) -> bool:
    """Room-ready inside Lizard entry: level 5, play mode, room 0x76."""
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL5_LEVEL_ID
        and snap.mode == PLAY_MODE
        and snap.screen == LEVEL5_ENTRY_ROOM
    )


def level5_door_screen_reached(ram: np.ndarray) -> bool:
    snap = read_snapshot(ram)
    return (
        snap.level == 0
        and snap.mode == PLAY_MODE
        and snap.screen == SCREEN_LEVEL5_DOOR
    )


def level5_hops_from(screen: int) -> tuple[ScreenHop, ...]:
    """Remaining path hops after ``screen`` (prefix of LEVEL5_PATH_HOPS)."""
    targets = [h.target for h in LEVEL5_PATH_HOPS]
    if screen in targets:
        return LEVEL5_PATH_HOPS[targets.index(screen) + 1 :]
    if screen == 0x4A:
        return LEVEL5_PATH_HOPS
    if screen == SCREEN_LOST_HILLS:
        return ()
    if screen == SCREEN_LEVEL5_DOOR:
        return ()
    return LEVEL5_PATH_HOPS


__all__ = [
    "SCREEN_LOST_HILLS",
    "SCREEN_LEVEL5_DOOR",
    "LEVEL5_ENTRY_ROOM",
    "LEVEL5_DOOR_X",
    "LEVEL5_TRIFORCE_BIT",
    "LEVEL5_LEVEL_ID",
    "LEVEL5_PATH_HOPS",
    "SEGMENT_MAX_FRAMES",
    "LOST_HILLS_UPS_REQUIRED",
    "Level5NavPhase",
    "OverworldToLevel5Controller",
    "level5_entrance_success",
    "level5_door_screen_reached",
    "level5_hops_from",
]

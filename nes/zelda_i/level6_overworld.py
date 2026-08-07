"""Overworld + entry helpers for Level 6 (Dragon).

Live recon (assisted, 2026-08-06)::

    OW door screen **0x22** (west near graveyard). Enter UP @ x≈24–56
    (prefer ~48). Entry room **0x79** (level==6, mode 5, xy≈(120, 205)).
    East of entry **0x7a**: 5× object type 0x24 + RoomItemId 0x19 key.
    RIGHT from entry needs wall-first y≈157 then y≈138 (fire solids at
    center y≈141 stick x≈128).

Full Clean walk hops from start / post-L1 are **planned** — see
``docs/LEVEL6_ROUTE.md``. Bracelet warp on OW 0x79 is optional residual.
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
from zelda_i.overworld import ScreenHop
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot, read_snapshot

# --- Live-verified geometry ---
SCREEN_LEVEL6_ENTRANCE = 0x22  # overworld door (live)
LEVEL6_ENTRY_ROOM = 0x79  # dungeon south mouth after mode-16 settle
LEVEL6_EAST_KEY_ROOM = 0x7A  # RIGHT of entry (type 0x24 ×5 + key 0x19)
# Door mouth is wide: south-path enter works ~x112; mid-screen band ~24–56.
LEVEL6_DOOR_X = 112  # preferred for south-path fixture L6Probe_22
LEVEL6_DOOR_X_LO = 24
LEVEL6_DOOR_X_HI = 120
LEVEL6 = 6
LEVEL6_TRIFORCE_BIT = 0x20
WIZZROBE_ORANGE_TYPE = 0x24  # walkthrough-correlated; live on 0x7a

# Entry RIGHT door (fire-block bypass)
ENTRY_RIGHT_WALL_Y = 157
ENTRY_RIGHT_DOOR_Y = 141  # channel ~136–152 live (wall blocks tighter y)
ENTRY_RIGHT_DOOR_Y_LO = 136
ENTRY_RIGHT_DOOR_Y_HI = 152
ENTRY_RIGHT_WALL_X = 200  # need x≥200 before y-slide; x~192 y-stuck at 149

SEGMENT_MAX_FRAMES = 25000
SWORD_SWING_PERIOD = 10
SWORD_SWING_FRAMES = 3
STUCK_THRESHOLD = 50

# Planned walk hops — not Clean-verified end-to-end. Door hunt is live.
# Bracelet shortcut (source): 0x77 E E → 0x79 stairs → down/left/up → 0x22.
# Scaffold stops at door screen when hops empty and require_level6_screen.
LEVEL6_DOOR_HOPS: tuple[ScreenHop, ...] = (
    # Filled when a live walk path is recorded; empty ⇒ start on 0x22 or tele.
)

LEVEL6_PATH_SCREENS: tuple[int, ...] = (SCREEN_LEVEL6_ENTRANCE,)


class Level6NavPhase(Enum):
    HOP = auto()
    DOOR = auto()
    DUNGEON_SETTLE = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class OverworldToLevel6Controller:
    """Walk optional hops then door-hunt / enter Level 6 on OW 0x22.

    Default: assume already on or near the door screen (recon fixture
    ``L6Probe_22`` / ``Level6Entrance``). Pass ``hops=...`` when a walk
    prefix exists. ``require_dungeon=True`` waits for room-ready 0x79.
    """

    hop_index: int = 0
    phase: Level6NavPhase = Level6NavPhase.HOP
    frames: int = 0
    phase_frames: int = 0
    stuck: int = 0
    last_x: int = -1
    last_y: int = -1
    last_screen: int = -1
    success: bool = False
    notes: list[str] = field(default_factory=list)
    require_level6_screen: bool = False
    require_dungeon: bool = False
    hops: tuple[ScreenHop, ...] = LEVEL6_DOOR_HOPS
    door_x: int = LEVEL6_DOOR_X

    def reset(self) -> None:
        self.hop_index = 0
        self.phase = Level6NavPhase.HOP
        self.frames = 0
        self.phase_frames = 0
        self.stuck = 0
        self.last_x = -1
        self.last_y = -1
        self.last_screen = -1
        self.success = False
        self.notes.clear()

    def _set_phase(self, phase: Level6NavPhase, note: str = "") -> None:
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
            return (
                snap.level == LEVEL6
                and snap.mode == PLAY_MODE
                and snap.screen == LEVEL6_ENTRY_ROOM
            )
        if self.require_level6_screen:
            return (
                snap.level == 0
                and snap.mode == PLAY_MODE
                and snap.screen == SCREEN_LEVEL6_ENTRANCE
            )
        if not self.hops:
            return (
                snap.level == 0
                and snap.mode == PLAY_MODE
                and snap.screen == SCREEN_LEVEL6_ENTRANCE
            )
        end = self.hops[-1].target
        return (
            self.hop_index >= len(self.hops)
            and snap.level == 0
            and snap.mode == PLAY_MODE
            and snap.screen == end
        )

    def _advance_hop(self, snap: ZeldaSnapshot, hop: ScreenHop) -> FrameAction | None:
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
            self.require_level6_screen or self.require_dungeon
        ):
            self.success = True
            self._set_phase(Level6NavPhase.DONE, "hops_complete")
            return FrameAction(nes_idle_action(), "done")
        return FrameAction(nes_idle_action(), "hop_advance")

    def _door_hunt(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.level == LEVEL6:
            self._set_phase(Level6NavPhase.DUNGEON_SETTLE, "entered_l6")
            return FrameAction(nes_idle_action(), "dungeon_settle")
        if snap.screen != SCREEN_LEVEL6_ENTRANCE:
            # Nudge toward door screen if slightly off (caller should load hops).
            return self._swing("UP", "seek_door_screen")
        # South mouth of 0x22: center x≈120 is solid; corridor is slightly west
        # (live: x≈112 climbs; door band at mouth x≈24–56 once mid-screen).
        if snap.link_y > 200:
            if snap.link_x > 112:
                return self._swing("LEFT", "south_lane")
            if snap.link_x < 100:
                return self._swing("RIGHT", "south_lane")
            return self._swing("UP", "door_climb")
        if snap.link_y > 160:
            # Mid-climb: prefer left toward door mouth.
            if snap.link_x > LEVEL6_DOOR_X_HI:
                return self._swing("LEFT", "climb_ax")
            return self._swing("UP", "door_climb")
        if snap.link_x < LEVEL6_DOOR_X_LO or snap.link_x > LEVEL6_DOOR_X_HI:
            btn = "LEFT" if snap.link_x > self.door_x else "RIGHT"
            return self._swing(btn, "door_ax")
        if abs(snap.link_x - self.door_x) > 5:
            btn = "LEFT" if snap.link_x > self.door_x else "RIGHT"
            return self._swing(btn, "door_ax")
        return self._swing("UP", "door_hunt")

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
            self._set_phase(Level6NavPhase.FAILED, "timeout")
            return FrameAction(nes_idle_action(), "timeout")

        if snap.mode == 17:
            self._set_phase(Level6NavPhase.FAILED, "link_death")
            return FrameAction(nes_idle_action(), "link_death")

        if self._at_stop(snap):
            self.success = True
            self._set_phase(Level6NavPhase.DONE, "level6_path_stop")
            return FrameAction(nes_idle_action(), "done")

        if snap.transitioning:
            if self.hop_index < len(self.hops):
                return FrameAction(
                    nes_action(self.hops[self.hop_index].direction), "scroll"
                )
            if self.require_dungeon or snap.level == LEVEL6:
                return FrameAction(nes_idle_action(), "scroll_idle")
            return self._swing("UP", "scroll_door")

        if snap.mode not in (PLAY_MODE, 8, 11, 16):
            return wake_or_wait_mode(self.phase_frames, snap.mode)

        if snap.level == LEVEL6:
            if self.require_dungeon and snap.mode == PLAY_MODE:
                if snap.screen == LEVEL6_ENTRY_ROOM:
                    self.success = True
                    self._set_phase(Level6NavPhase.DONE, "entry_room_ready")
                    return FrameAction(nes_idle_action(), "done")
            return FrameAction(nes_idle_action(), "dungeon_settle")

        if self.hop_index >= len(self.hops):
            if self.require_level6_screen or self.require_dungeon or not self.hops:
                return self._door_hunt(snap)
            self.success = True
            self._set_phase(Level6NavPhase.DONE, "hops_complete")
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
            stuck=0,
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
            "require_level6_screen": self.require_level6_screen,
            "require_dungeon": self.require_dungeon,
        }


class EntryRightPhase(Enum):
    TO_WALL_Y = auto()
    TO_WALL_X = auto()
    HUG_AND_SLIDE = auto()  # x→208 then y→144 then RIGHT
    DONE = auto()
    FAILED = auto()


@dataclass
class Level6EntryRightController:
    """From entry 0x79, wall-first RIGHT into 0x7a (fire-block bypass).

    Live policy (no A): y≈157 → x≈200 → x≈208 → y≈144 → RIGHT → 0x7a.
    """

    phase: EntryRightPhase = EntryRightPhase.TO_WALL_Y
    frames: int = 0
    phase_frames: int = 0
    stuck: int = 0
    last_x: int = -1
    last_y: int = -1
    last_screen: int = -1
    success: bool = False
    notes: list[str] = field(default_factory=list)
    max_frames: int = 4000
    door_y_target: int = 144

    def reset(self) -> None:
        self.phase = EntryRightPhase.TO_WALL_Y
        self.frames = 0
        self.phase_frames = 0
        self.stuck = 0
        self.last_x = -1
        self.last_y = -1
        self.last_screen = -1
        self.success = False
        self.notes.clear()

    def _move(self, direction: str, reason: str) -> FrameAction:
        # No sword pulse: A-frames block the sub-pixel door channel at x≈200.
        return FrameAction(nes_action(direction), reason)

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
        if self.frames >= self.max_frames:
            self.phase = EntryRightPhase.FAILED
            self.notes.append("timeout")
            return FrameAction(nes_idle_action(), "timeout")
        if snap.mode == 17:
            self.phase = EntryRightPhase.FAILED
            return FrameAction(nes_idle_action(), "link_death")
        if snap.level == LEVEL6 and snap.screen == LEVEL6_EAST_KEY_ROOM:
            if snap.mode == PLAY_MODE or snap.transitioning or snap.mode in (2, 3, 4):
                if snap.mode == PLAY_MODE:
                    self.success = True
                    self.phase = EntryRightPhase.DONE
                    self.notes.append("east_key_room")
                    return FrameAction(nes_idle_action(), "done")
                return FrameAction(nes_idle_action(), "east_settle")
        if snap.transitioning or snap.mode not in (PLAY_MODE, 8):
            return FrameAction(nes_idle_action(), "wait")
        if self.stuck > STUCK_THRESHOLD:
            wiggle = ("UP", "DOWN", "LEFT", "RIGHT")[self.stuck % 4]
            self.stuck = 0 if self.stuck > 140 else self.stuck
            return FrameAction(nes_action(wiggle), "unstick")

        if self.phase is EntryRightPhase.TO_WALL_Y:
            if snap.link_y <= ENTRY_RIGHT_WALL_Y + 2:
                self.phase = EntryRightPhase.TO_WALL_X
                self.phase_frames = 0
                self.notes.append("at_wall_y")
            else:
                return self._move("UP", "to_wall_y")

        if self.phase is EntryRightPhase.TO_WALL_X:
            if snap.link_x >= 198:
                self.phase = EntryRightPhase.HUG_AND_SLIDE
                self.phase_frames = 0
                self.notes.append("at_wall_x")
            elif abs(snap.link_y - ENTRY_RIGHT_WALL_Y) > 8:
                btn = "UP" if snap.link_y > ENTRY_RIGHT_WALL_Y else "DOWN"
                return self._move(btn, "hold_wall_y")
            else:
                return self._move("RIGHT", "to_wall_x")

        # HUG_AND_SLIDE: x to ≥206, then RIGHT while y≤152 (channel ~144–149).
        if snap.link_x < 206:
            if abs(snap.link_y - ENTRY_RIGHT_WALL_Y) > 10 and snap.link_x < 190:
                btn = "UP" if snap.link_y > ENTRY_RIGHT_WALL_Y else "DOWN"
                return self._move(btn, "reband")
            return self._move("RIGHT", "hug_wall")
        # Prefer a bit north of 149 when possible, but do not softlock on UP.
        if snap.link_y > 152:
            return self._move("UP", "slide_door_y")
        if snap.link_y < 136:
            return self._move("DOWN", "slide_door_y")
        if snap.link_y > 146 and self.phase_frames < 40:
            return self._move("UP", "nudge_door_y")
        return self._move("RIGHT", "push_right")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "stuck": self.stuck,
        }


def level6_screen_reached(ram: np.ndarray) -> bool:
    snap = read_snapshot(ram)
    return (
        snap.level == 0
        and snap.mode == PLAY_MODE
        and snap.screen == SCREEN_LEVEL6_ENTRANCE
    )


def level6_entrance_success(ram: np.ndarray) -> bool:
    """Room-ready inside Dragon entry: level 6, play mode, room 0x79."""
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and snap.screen == LEVEL6_ENTRY_ROOM
    )


def level6_east_key_room(ram: np.ndarray) -> bool:
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and snap.screen == LEVEL6_EAST_KEY_ROOM
    )


__all__ = [
    "SCREEN_LEVEL6_ENTRANCE",
    "LEVEL6_ENTRY_ROOM",
    "LEVEL6_EAST_KEY_ROOM",
    "LEVEL6_DOOR_X",
    "LEVEL6_DOOR_X_LO",
    "LEVEL6_DOOR_X_HI",
    "LEVEL6",
    "LEVEL6_TRIFORCE_BIT",
    "WIZZROBE_ORANGE_TYPE",
    "LEVEL6_DOOR_HOPS",
    "OverworldToLevel6Controller",
    "Level6EntryRightController",
    "level6_screen_reached",
    "level6_entrance_success",
    "level6_east_key_room",
]

"""Overworld routing: start → Level 3 (Manji) door + entry stop predicates.

Live recon (assisted, 2026-08-06)::

    Door screen **0x74** (exit spawn ~(128, 125)).
    Entry room **0x7c** (level==3, mode 5).
    Source hop path ``77↑67←×4↓→74`` is **blocked**: 0x67 is an enclosed
    tree pocket with no west exit.

Verified walk pieces (assist / Survival)::

    From start spine toward west forest:
      0x77 E@y≈140 → 0x78 N@x≈48 → 0x68 N → 0x58 W@y≈155
      → 0x57 W → 0x56 W@y≈133 → 0x55 S → 0x65
    Door approach (from OW_66 checkpoint or 0x65):
      0x65 W@y≈141 → 0x64 W → 0x63 S → 0x73 E@y≈117 → 0x74
      door hunt UP @x≈128 (may need mid-screen tour on rock maze)

Item: Raft (``ADDR_RAFT=0x0660``). Boss: Manhandla. TF bit ``0x04``.
Track: assisted first-pass only — do **not** promote Clean STATUS.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_idle_action
from zelda_i.overworld import ScreenHop, path_screens_from_hops
from zelda_i.ow_path import OverworldPathController
from zelda_i.ram import PLAY_MODE, SCREEN_START, ZeldaSnapshot, read_snapshot

# --- Live anchors (assisted recon 2026-08-06) ---
SCREEN_LEVEL3_ENTRANCE = 0x74
SCREEN_LEVEL3_ENTRY_ROOM = 0x7C
LEVEL3_DOOR_X = 128  # exit-spawn x; UP re-enter after y>130 approach
LEVEL3_DOOR_APPROACH_Y = 140
LEVEL3 = 3
LEVEL3_TRIFORCE_BIT = 0x04

SEGMENT_MAX_FRAMES = 35000
SWORD_SWING_PERIOD = 10
SWORD_SWING_FRAMES = 3
STUCK_THRESHOLD = 50

# Source arithmetic (NOT walkable end-to-end — 0x67 sealed). Kept for docs.
LEVEL3_SOURCE_PATH_SCREENS: tuple[int, ...] = (
    0x77,
    0x67,
    0x66,
    0x65,
    0x64,
    0x63,
    0x73,
    0x74,
)

# Live door approach hops ending on 0x74 (from west-forest 0x66).
# Start of this chain assumes Link is already on 0x66 (or join mid-table).
# 0x66 rock rows: y≈117 often works first; widen band if stuck (probe residual).
LEVEL3_DOOR_HOPS_FROM_66: tuple[ScreenHop, ...] = (
    ScreenHop(0x65, "LEFT", y_band_lo=110, y_band_hi=150),
    ScreenHop(0x64, "LEFT", y_band_lo=125, y_band_hi=150),
    ScreenHop(0x63, "LEFT", y_band_lo=110, y_band_hi=145),
    ScreenHop(0x73, "DOWN", align_x=112),
    ScreenHop(0x74, "RIGHT", align_y=117),
)

# Prefix from post-sword start toward west forest (through 0x55 → 0x65).
# 0x56 west corridor is narrow: align_y≈133 (not the 0x58 y≈155 bush band).
LEVEL3_PREFIX_HOPS_FROM_START: tuple[ScreenHop, ...] = (
    ScreenHop(0x78, "RIGHT", align_y=140),
    ScreenHop(0x68, "UP", align_x=48),
    ScreenHop(0x58, "UP", align_x=48),
    ScreenHop(0x57, "LEFT", y_band_lo=148, y_band_hi=162),
    ScreenHop(0x56, "LEFT", y_band_lo=148, y_band_hi=162),
    ScreenHop(0x55, "LEFT", align_y=133),
    ScreenHop(0x65, "DOWN", align_x=112),
)

# Full hop list start → door (prefix + continue from 0x65 without re-visiting 0x66).
LEVEL3_PATH_HOPS: tuple[ScreenHop, ...] = LEVEL3_PREFIX_HOPS_FROM_START + (
    ScreenHop(0x64, "LEFT", align_y=141),
    ScreenHop(0x63, "LEFT", align_y=133),
    ScreenHop(0x73, "DOWN", align_x=112),
    ScreenHop(0x74, "RIGHT", align_y=117),
)
LEVEL3_PATH_SCREENS: tuple[int, ...] = path_screens_from_hops(
    SCREEN_START, LEVEL3_PATH_HOPS
)
assert LEVEL3_PATH_SCREENS[0] == SCREEN_START
assert LEVEL3_PATH_SCREENS[-1] == SCREEN_LEVEL3_ENTRANCE


class Level3NavPhase(Enum):
    HOP = auto()
    DOOR = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class OverworldToLevel3Controller(OverworldPathController):
    """Walk ScreenHop path toward Manji door; optional dungeon settle.

    Default stop is door screen **0x74**. Pass ``require_dungeon=True`` to
    hunt UP @ ``door_x`` and idle until level==3 play (entry room 0x7c).
    """

    phase: Level3NavPhase = Level3NavPhase.HOP
    require_level3_screen: bool = False
    require_dungeon: bool = False
    hops: tuple[ScreenHop, ...] = LEVEL3_PATH_HOPS
    door_x: int = LEVEL3_DOOR_X
    entry_room: int = SCREEN_LEVEL3_ENTRY_ROOM
    entry_level: int | None = LEVEL3
    door_screen: int | None = SCREEN_LEVEL3_ENTRANCE
    max_frames: int = SEGMENT_MAX_FRAMES
    swing_period: int = SWORD_SWING_PERIOD
    swing_hold: int = SWORD_SWING_FRAMES
    stuck_threshold: int = STUCK_THRESHOLD
    require_sword: bool = True

    def _wants_post_hop(self) -> bool:
        return self.require_level3_screen or self.require_dungeon

    def _at_stop(self, snap: ZeldaSnapshot) -> bool:
        if self.require_dungeon:
            return (
                snap.level == LEVEL3
                and snap.mode == PLAY_MODE
                and snap.screen == self.entry_room
            )
        if self.require_level3_screen:
            return (
                snap.level == 0
                and snap.mode == PLAY_MODE
                and snap.screen == SCREEN_LEVEL3_ENTRANCE
                and snap.has_sword
            )
        end_screen = self.hops[-1].target if self.hops else SCREEN_LEVEL3_ENTRANCE
        return (
            self.hop_index >= len(self.hops)
            and snap.level == 0
            and snap.mode == PLAY_MODE
            and snap.screen == end_screen
            and snap.has_sword
            and 40 < snap.link_y < 210
        )

    def _before_play(self, snap: ZeldaSnapshot) -> FrameAction | None:
        if snap.level not in (0, LEVEL3) and snap.level > 0:
            return self._swing("DOWN", f"exit_l{snap.level}")
        return None

    def _after_hops(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.require_level3_screen or self.require_dungeon:
            if snap.level == LEVEL3:
                return FrameAction(nes_idle_action(), "dungeon_settle")
            self._set_phase(Level3NavPhase.DOOR, "door_hunt")
            # Approach from south of mouth then align x and push UP.
            if snap.link_y < LEVEL3_DOOR_APPROACH_Y - 10:
                return self._swing("DOWN", "door_south")
            if abs(snap.link_x - self.door_x) > 5:
                btn = "LEFT" if snap.link_x > self.door_x else "RIGHT"
                return self._swing(btn, "door_ax")
            return self._swing("UP", "door_hunt")
        return self._finish("hops_complete")

    def _finish(self, note: str = "path_stop") -> FrameAction:
        # Preserve historical note labels used by probes/logs.
        label = {
            "path_stop": "level3_path_stop",
            "path_complete": "path_complete",
            "hops_complete": "hops_complete",
        }.get(note, note)
        self.success = True
        self._set_phase(Level3NavPhase.DONE, label)
        return FrameAction(nes_idle_action(), "done")

    def report(self) -> dict[str, Any]:
        out = super().report()
        out["require_level3_screen"] = self.require_level3_screen
        out["require_dungeon"] = self.require_dungeon
        out["door_screen"] = SCREEN_LEVEL3_ENTRANCE
        out["door_x"] = self.door_x
        out["entry_room"] = self.entry_room
        out.pop("require_entrance_screen", None)
        return out


def level3_path_success(ram: np.ndarray) -> bool:
    """Stop on door screen 0x74."""
    snap = read_snapshot(ram)
    return (
        snap.level == 0
        and snap.mode == PLAY_MODE
        and snap.screen == SCREEN_LEVEL3_ENTRANCE
        and snap.has_sword
    )


def level3_screen_reached(ram: np.ndarray) -> bool:
    return level3_path_success(ram)


def level3_entrance_success(ram: np.ndarray) -> bool:
    """Room-ready inside Manji entry: level 3, play mode, room 0x7c."""
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL3
        and snap.mode == PLAY_MODE
        and snap.screen == SCREEN_LEVEL3_ENTRY_ROOM
    )

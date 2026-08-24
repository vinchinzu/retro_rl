"""Overworld + entry helpers for Level 6 (Dragon).

Live recon (assisted, 2026-08-06)::

    OW door screen **0x22** (west near graveyard). Enter UP @ x≈24–56
    (prefer ~48). Entry room **0x79** (level==6, mode 5, xy≈(120, 205)).
    East of entry **0x7a**: 5× object type 0x24 + RoomItemId 0x19 key.
    RIGHT from entry needs wall-first y≈157 then y≈138 (fire solids at
    center y≈141 stick x≈128).

Post-L5 walk (source, OVERWORLD_DOORS): from L5 door ``0x0B``
``↓ ←×7 ↓ ← ↓ ← ↑`` onto door ``0x22``. Lost Hills ``0x1B`` only LEFT
exits (UP/RIGHT/DOWN wrap). 0x0B west/east sealed. 0x1B north-edge LEFT
at leftover ``(112,61)`` is solid (v25); inland LEFT at ``(96,141)`` hits
the x≈72 rock (v1). Screenshot occupancy (v1/v17/v25): sand west channel
y=136–151, x<72. SW notch ``(32,165)`` is live but LEFT at y=165 is
mountain. Bracelet warp on OW 0x79 is optional residual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.nav_common import track_stuck
from zelda_i.overworld import ScreenHop, path_screens_from_hops
from zelda_i.ow_path import OverworldPathController
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot, read_snapshot

# --- Live-verified geometry; entrance from anchors ---
from zelda_i.anchors import (
    LEVEL6_ENTRY_ROOM,
    SCREEN_BRACELET_ARMOS,
    SCREEN_LEVEL5_ENTRANCE,
    SCREEN_LEVEL6_ENTRANCE,
    SCREEN_LOST_HILLS,
    TF_BIT_L5,
    TF_BIT_L6 as LEVEL6_TRIFORCE_BIT,
)

LEVEL6_EAST_KEY_ROOM = 0x7A  # RIGHT of entry (type 0x24 ×5 + key 0x19)
LEVEL6_WEST_WIZZROBE_ROOM = 0x78  # LEFT of entry via key door (5× type 0x24)
LEVEL6_COMPASS_ROOM = 0x68  # UP of cleared 0x78; 5× Zol + compass 0x16
LEVEL6_OLD_MAN_ROOM = 0x6A  # UP key door from 0x7a — DO NOT spend first key
# Door mouth is wide: south-path enter works ~x112; mid-screen band ~24–56.
LEVEL6_DOOR_X = 112  # preferred for south-path fixture L6Probe_22
LEVEL6_DOOR_X_LO = 24
LEVEL6_DOOR_X_HI = 120
LEVEL6 = 6
WIZZROBE_ORANGE_TYPE = 0x24  # walkthrough-correlated; live on 0x7a / 0x78

# Entry RIGHT door (fire-block bypass)
ENTRY_RIGHT_WALL_Y = 157
ENTRY_RIGHT_DOOR_Y = 141  # channel ~136–152 live (wall blocks tighter y)
ENTRY_RIGHT_DOOR_Y_LO = 136
ENTRY_RIGHT_DOOR_Y_HI = 152
ENTRY_RIGHT_WALL_X = 200  # need x≥200 before y-slide; x~192 y-stuck at 149

# Entry LEFT key door (fire-block bypass) — same wall y as RIGHT path.
# Naive y≈141 LEFT from east/center sticks on fire solids (x≈208 / x≈112).
ENTRY_LEFT_WALL_Y = 157
ENTRY_LEFT_DOOR_Y = 141  # key-door channel; y≈143 live after slide
ENTRY_LEFT_DOOR_Y_LO = 136
ENTRY_LEFT_DOOR_Y_HI = 152
ENTRY_LEFT_WALL_X = 32

SEGMENT_MAX_FRAMES = 25000
SWORD_SWING_PERIOD = 10
SWORD_SWING_FRAMES = 3
STUCK_THRESHOLD = 50

# Planned walk hops from start — not Clean-verified. Door hunt is live.
# Bracelet shortcut (source): 0x77 E E → 0x79 stairs → down/left/up → 0x22.
# Scaffold stops at door screen when hops empty and require_level6_screen.
LEVEL6_DOOR_HOPS: tuple[ScreenHop, ...] = (
    # Filled when a live walk path is recorded; empty ⇒ start on 0x22 or tele.
)

LEVEL6_PATH_SCREENS: tuple[int, ...] = (SCREEN_LEVEL6_ENTRANCE,)

# Natural predecessor after L5 Triforce fanfare. L5 returns Link to door
# screen 0x0B (west/east sealed). Lost Hills 0x1B north arrival (112,61)
# LEFT is solid (v25). West channel is screenshot sand y=136–151, x<72;
# DOWN around the x≈72 rock (v1 leftover 96,141) then UP from SW/south
# leftovers (v17 32,165 / v12 64,149) and LEFT at door Y. Isolated BFS
# banned. Then source west/south chain onto 0x22.
SCREEN_POST_L5_RETURN = SCREEN_LEVEL5_ENTRANCE
POST_L5_SETTLE_MAX_FRAMES = 2500
POST_L5_PATH_MAX_FRAMES = 40000
# 0x1B west rock at x≈72, y≈141 (v1). South sand y≈185. Notch x≤48 (v28
# leftover 48,189 UPs; v30 leftover 40,181 LEFT is west mountain). Aisle
# x≤32 at y≈165 (v17) UPs to the y=136–151 channel then LEFT.
HILLS_ROCK_X = 72
HILLS_AISLE_X = 32
HILLS_NOTCH_X = 48
HILLS_NOTCH_Y = 168
HILLS_SOUTH_Y = 185
HILLS_CHANNEL_Y_LO = 136
HILLS_CHANNEL_Y_HI = 152
HILLS_NORTH_WALL_Y = 87
HILLS_STALL_FAIL = 180
HILLS_PUSH_STALL = 360
POST_L5_TO_LEVEL6_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(SCREEN_LOST_HILLS, "DOWN", align_x=112),
    ScreenHop(0x1A, "LEFT", align_y=141),
    ScreenHop(0x19, "LEFT", align_y=141),
    ScreenHop(0x18, "LEFT", align_y=141),
    ScreenHop(0x17, "LEFT", align_y=141),
    ScreenHop(0x16, "LEFT", align_y=141),
    ScreenHop(0x15, "LEFT", align_y=141),
    # v37 leftover 0x15 (104,141): Lynels on door Y; south sand around.
    ScreenHop(0x14, "LEFT", y_band_lo=165, y_band_hi=189),
    # v38 leftover 0x14 (112,189): south mouth is the SE blue path x≈154–165,
    # not center x=112. v39 (152,189) is 2px west of the blue; stand on it.
    ScreenHop(SCREEN_BRACELET_ARMOS, "DOWN", align_x=160),
    ScreenHop(0x23, "LEFT", align_y=141),
    # v40 leftover 0x23 (160,141): east pocket. South mouth is the SE blue
    # path x≈202–213, not center x=112 (mountain splitter).
    ScreenHop(0x33, "DOWN", align_x=208),
    ScreenHop(0x32, "LEFT", align_y=141),
    ScreenHop(SCREEN_LEVEL6_ENTRANCE, "UP", align_x=112),
)
LEVEL6_POST_L5_SCREENS: tuple[int, ...] = path_screens_from_hops(
    SCREEN_POST_L5_RETURN, POST_L5_TO_LEVEL6_HOPS
)
assert LEVEL6_POST_L5_SCREENS[0] == SCREEN_POST_L5_RETURN
assert LEVEL6_POST_L5_SCREENS[-1] == SCREEN_LEVEL6_ENTRANCE
assert SCREEN_LOST_HILLS in LEVEL6_POST_L5_SCREENS


def lost_hills_west_dir(x: int, y: int) -> str:
    """Cardinal toward the 0x1B west sand channel. Screenshot occupancy.

    v25 ``(112,61)`` LEFT solid; v1 ``(96,141)`` LEFT hits the x≈72 rock;
    v26 ``(96,165)`` LEFT is the bottom rock row; v27 ``(71,189)`` UP is
    that rock's south face; v28 ``(48,189)`` LEFT is SW mountain; v29
    ``(48,165)`` UP is the bottom-left rock; v17 ``(32,165)`` LEFT is
    mountain. Channel is y=136–151. South sand y≈185, notch x≤48 then
    aisle x≤32 UP and LEFT.
    """
    if y <= HILLS_NORTH_WALL_Y:
        return "DOWN"
    if y >= HILLS_SOUTH_Y:
        return "LEFT" if x > HILLS_NOTCH_X else "UP"
    if (
        x < HILLS_ROCK_X
        and HILLS_CHANNEL_Y_LO <= y <= HILLS_CHANNEL_Y_HI
    ):
        return "LEFT"
    if x > HILLS_NOTCH_X:
        return "DOWN"
    if y > HILLS_NOTCH_Y:
        return "UP"
    if x > HILLS_AISLE_X and y > HILLS_CHANNEL_Y_HI:
        return "LEFT"
    if y > HILLS_CHANNEL_Y_HI:
        return "UP"
    if y < HILLS_CHANNEL_Y_LO:
        return "DOWN"
    return "LEFT"


class Level6NavPhase(Enum):
    HOP = auto()
    DOOR = auto()
    DUNGEON_SETTLE = auto()
    DONE = auto()
    FAILED = auto()


class PostL5SettlePhase(Enum):
    WAIT = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class PostL5TriforceSettleController:
    """Idle through L5 triforce fanfare until OW door 0x0B play.

    Start: leftover L5 TF room 0x14 mode 18. Do not reload a checkpoint
    mid-fanfare (same class as L1–L4 TF settle). Do not grant Whistle.
    """

    phase: PostL5SettlePhase = PostL5SettlePhase.WAIT
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)
    max_frames: int = POST_L5_SETTLE_MAX_FRAMES
    require_screen: int = SCREEN_POST_L5_RETURN

    def reset(self) -> None:
        self.phase = PostL5SettlePhase.WAIT
        self.frames = 0
        self.phase_frames = 0
        self.success = False
        self.notes.clear()

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        if self.frames > self.max_frames:
            self.phase = PostL5SettlePhase.FAILED
            self.notes.append("settle_timeout")
            return FrameAction(nes_idle_action(), "settle_timeout")

        if (
            snap.level == 0
            and snap.mode == PLAY_MODE
            and snap.screen == self.require_screen
            and bool(snap.triforce & TF_BIT_L5)
            and not snap.transitioning
        ):
            self.success = True
            if self.phase is not PostL5SettlePhase.DONE:
                self.phase = PostL5SettlePhase.DONE
                self.notes.append("post_l5_ow_ready")
            return FrameAction(nes_idle_action(), "settle_done")

        return FrameAction(nes_idle_action(), "settle_wait")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "require_screen": f"0x{self.require_screen:02x}",
        }


def post_l5_overworld_ready(snap: ZeldaSnapshot) -> bool:
    """OW play on Lizard door 0x0B with L5 triforce bit."""
    return (
        snap.level == 0
        and snap.mode == PLAY_MODE
        and snap.screen == SCREEN_POST_L5_RETURN
        and bool(snap.triforce & TF_BIT_L5)
        and not snap.transitioning
    )


@dataclass
class OverworldToLevel6Controller(OverworldPathController):
    """Walk optional hops then door-hunt / enter Level 6 on OW 0x22.

    Default: assume already on or near the door screen (recon fixture
    ``L6Probe_22`` / ``Level6Entrance``). Pass ``hops=...`` when a walk
    prefix exists. ``require_dungeon=True`` waits for room-ready 0x79.
    """

    phase: Level6NavPhase = Level6NavPhase.HOP
    require_level6_screen: bool = False
    require_dungeon: bool = False
    hops: tuple[ScreenHop, ...] = LEVEL6_DOOR_HOPS
    door_x: int = LEVEL6_DOOR_X
    entry_level: int | None = LEVEL6
    entry_room: int | None = LEVEL6_ENTRY_ROOM
    door_screen: int | None = SCREEN_LEVEL6_ENTRANCE
    max_frames: int = SEGMENT_MAX_FRAMES
    swing_period: int = SWORD_SWING_PERIOD
    swing_hold: int = SWORD_SWING_FRAMES
    stuck_threshold: int = STUCK_THRESHOLD
    allowed_modes: frozenset[int] = field(
        default_factory=lambda: frozenset({PLAY_MODE, 8, 11, 16, 6, 7, 2, 3, 4})
    )

    def _wants_post_hop(self) -> bool:
        # Empty hops ⇒ always door-hunt (recon fixture path).
        return (
            self.require_level6_screen
            or self.require_dungeon
            or not self.hops
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

    def _before_play(self, snap: ZeldaSnapshot) -> FrameAction | None:
        if snap.level == LEVEL6:
            if self.require_dungeon and snap.mode == PLAY_MODE:
                if snap.screen == LEVEL6_ENTRY_ROOM:
                    return self._finish("entry_room_ready")
            return FrameAction(nes_idle_action(), "dungeon_settle")
        return None

    def _handle_transition(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.hop_index < len(self.hops):
            return FrameAction(
                nes_action(self.hops[self.hop_index].direction), "scroll"
            )
        if self.require_dungeon or snap.level == LEVEL6:
            return FrameAction(nes_idle_action(), "scroll_idle")
        return self._swing("UP", "scroll_door")

    def _extra_hop_action(
        self, snap: ZeldaSnapshot, hop: ScreenHop
    ) -> FrameAction | None:
        # 0x15 Lynels (continuous leftover 232,109 east edge). Inland then
        # south sand (isolated v38) then LEFT to 0x14. No occupancy BFS.
        if hop.target == 0x14 and snap.screen == 0x15:
            x, y = int(snap.link_x), int(snap.link_y)
            if self.stuck > HILLS_STALL_FAIL:
                self.notes.append(f"ow15_solid_({x},{y})")
                return self._fail(f"ow15_solid_{x}_{y}")
            if x >= 200:
                return self._swing("LEFT", "ow15_inland")
            if y < 165:
                return self._swing("DOWN", "ow15_south")
            if y > 189:
                return self._swing("UP", "ow15_south")
            return self._swing("LEFT", "ow15_west")
        # Lost Hills west door. Cardinals around the x≈72 rock onto the
        # y=136–151 sand channel (screenshot v1/v17/v25). No occupancy BFS.
        if hop.target != 0x1A or snap.screen != SCREEN_LOST_HILLS:
            return None
        x, y = int(snap.link_x), int(snap.link_y)
        in_channel = (
            x < HILLS_ROCK_X
            and HILLS_CHANNEL_Y_LO <= y <= HILLS_CHANNEL_Y_HI
        )
        stall_lim = HILLS_PUSH_STALL if in_channel else HILLS_STALL_FAIL
        if self.stuck > stall_lim:
            self.notes.append(f"hills_solid_({x},{y})")
            return self._fail(f"hills_solid_{x}_{y}")
        direction = lost_hills_west_dir(x, y)
        if in_channel:
            # v31 leftover (24,149) LEFT off door Y; v32 LEFT+UP yo-yo;
            # v33 LEFT+DOWN yo-yo at y=151. Hold exact door Y then LEFT.
            if y < 141:
                return FrameAction(nes_action("DOWN"), "hills_west_ay")
            if y > 141:
                return FrameAction(nes_action("UP"), "hills_west_ay")
            return FrameAction(nes_action("LEFT"), "hills_west_left")
        return self._swing(direction, f"hills_{direction.lower()}")

    def _after_hops(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.require_level6_screen or self.require_dungeon or not self.hops:
            return self._door_hunt(snap)
        return self._finish("hops_complete")

    def _finish(self, note: str = "path_stop") -> FrameAction:
        label = {
            "path_stop": "level6_path_stop",
            "path_complete": "hops_complete",
            "hops_complete": "hops_complete",
            "entry_room_ready": "entry_room_ready",
        }.get(note, note)
        self.success = True
        self._set_phase(Level6NavPhase.DONE, label)
        return FrameAction(nes_idle_action(), "done")

    def report(self) -> dict[str, Any]:
        out = super().report()
        out["require_level6_screen"] = self.require_level6_screen
        out["require_dungeon"] = self.require_dungeon
        out.pop("require_entrance_screen", None)
        return out


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


def make_post_l5_level6_controller() -> OverworldToLevel6Controller:
    """L5 door 0x0B → 0x1B y=141 LEFT → Dragon 0x79. Not bracelet warp."""
    return OverworldToLevel6Controller(
        hops=POST_L5_TO_LEVEL6_HOPS,
        require_dungeon=True,
        max_frames=POST_L5_PATH_MAX_FRAMES,
    )


def level6_hops_from(screen: int) -> tuple[ScreenHop, ...]:
    """Remaining post-L5 hops after ``screen``."""
    if screen == SCREEN_POST_L5_RETURN:
        return POST_L5_TO_LEVEL6_HOPS
    targets = [h.target for h in POST_L5_TO_LEVEL6_HOPS]
    if screen in targets:
        return POST_L5_TO_LEVEL6_HOPS[targets.index(screen) + 1 :]
    return POST_L5_TO_LEVEL6_HOPS


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


def level6_west_wizzrobe_room(ram: np.ndarray) -> bool:
    snap = read_snapshot(ram)
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and snap.screen == LEVEL6_WEST_WIZZROBE_ROOM
    )


class EntryLeftKeyPhase(Enum):
    TO_FIRE_X = auto()  # from east edge: LEFT to x≈208
    TO_WALL_Y = auto()  # y → 157 (south of fire row)
    TO_WEST_X = auto()  # LEFT along wall y to x≈32
    TO_DOOR_Y = auto()  # slide y → 141 at west wall
    PUSH = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level6WestKeyDoorController:
    """From 0x79 with keys≥1, fire-bypass LEFT key door into 0x78.

    Live policy (2026-08-06, no A)::

        From east return ~(224,141): x→208 → y→157 → x→32 @ y157 → y→141 → LEFT.
        From south spawn ~(120,205): y→157 → x→32 @ y157 → y→141 → LEFT.

    Consumes 1 key. Trap: UP from 0x7a spends the same key on Old Man 0x6a.
    """

    phase: EntryLeftKeyPhase = EntryLeftKeyPhase.TO_FIRE_X
    frames: int = 0
    phase_frames: int = 0
    stuck: int = 0
    last_x: int = -1
    last_y: int = -1
    last_screen: int = -1
    success: bool = False
    keys_at_start: int = -1
    notes: list[str] = field(default_factory=list)
    max_frames: int = 5000

    def reset(self) -> None:
        self.phase = EntryLeftKeyPhase.TO_FIRE_X
        self.frames = 0
        self.phase_frames = 0
        self.stuck = 0
        self.last_x = -1
        self.last_y = -1
        self.last_screen = -1
        self.success = False
        self.keys_at_start = -1
        self.notes.clear()

    def _move(self, direction: str, reason: str) -> FrameAction:
        return FrameAction(nes_action(direction), reason)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        if self.keys_at_start < 0:
            self.keys_at_start = int(snap.keys)
        self.stuck, self.last_x, self.last_y, self.last_screen = track_stuck(
            snap,
            last_x=self.last_x,
            last_y=self.last_y,
            last_screen=self.last_screen,
            stuck=self.stuck,
        )
        if self.frames >= self.max_frames:
            self.phase = EntryLeftKeyPhase.FAILED
            self.notes.append("timeout")
            return FrameAction(nes_idle_action(), "timeout")
        if snap.mode == 17:
            self.phase = EntryLeftKeyPhase.FAILED
            return FrameAction(nes_idle_action(), "link_death")
        if (
            snap.level == LEVEL6
            and snap.screen == LEVEL6_WEST_WIZZROBE_ROOM
            and snap.mode == PLAY_MODE
        ):
            self.success = True
            self.phase = EntryLeftKeyPhase.DONE
            self.notes.append("west_wizzrobe_room")
            return FrameAction(nes_idle_action(), "done")
        if snap.transitioning or snap.mode not in (PLAY_MODE, 8):
            # Keep pressing LEFT through key-door scroll.
            if snap.mode in (6, 7) or snap.transitioning:
                return self._move("LEFT", "key_scroll")
            return FrameAction(nes_idle_action(), "wait")
        if snap.level != LEVEL6 or snap.screen != LEVEL6_ENTRY_ROOM:
            return FrameAction(nes_idle_action(), f"wait_room_0x{snap.screen:02x}")
        if snap.keys < 1 and self.keys_at_start < 1:
            self.phase = EntryLeftKeyPhase.FAILED
            self.notes.append("no_keys")
            return FrameAction(nes_idle_action(), "no_keys")

        if self.stuck > STUCK_THRESHOLD:
            wiggle = ("UP", "DOWN", "LEFT", "RIGHT")[self.stuck % 4]
            self.stuck = 0 if self.stuck > 140 else self.stuck
            return FrameAction(nes_action(wiggle), "unstick")

        # South mouth: leave south edge first.
        if snap.link_y > 180 and snap.link_x > 48:
            return self._move("UP", "leave_south_mouth")

        # Door plane (x≤34): slide to door y then push LEFT (key).
        if snap.link_x <= 34:
            if abs(snap.link_y - ENTRY_LEFT_DOOR_Y) > 4:
                btn = "UP" if snap.link_y > ENTRY_LEFT_DOOR_Y else "DOWN"
                return self._move(btn, "west_door_y")
            return self._move("LEFT", "push_key_left")

        # East door channel (x>210): vertical blocked — LEFT to fire-wall column.
        if snap.link_x > 210:
            return self._move("LEFT", "leave_east_door_channel")

        # At fire-wall column (x≈198–210): y-adjust to wall band, then cross.
        if snap.link_x >= 198:
            if abs(snap.link_y - ENTRY_LEFT_WALL_Y) > 4:
                btn = "UP" if snap.link_y > ENTRY_LEFT_WALL_Y else "DOWN"
                return self._move(btn, "east_to_wall_y")
            return self._move("LEFT", "cross_on_wall_y")

        # Approach west door on wall y (keep LEFT until x≤34). Do not drop to
        # door y early — y≈149 @ x≈48 sticks on fire solids (live).
        if abs(snap.link_y - ENTRY_LEFT_WALL_Y) > 6:
            btn = "UP" if snap.link_y > ENTRY_LEFT_WALL_Y else "DOWN"
            return self._move(btn, "mid_to_wall_y")
        return self._move("LEFT", "mid_cross_wall_y")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "stuck": self.stuck,
            "keys_at_start": self.keys_at_start,
        }


__all__ = [
    "SCREEN_LEVEL6_ENTRANCE",
    "SCREEN_POST_L5_RETURN",
    "LEVEL6_ENTRY_ROOM",
    "LEVEL6_EAST_KEY_ROOM",
    "LEVEL6_WEST_WIZZROBE_ROOM",
    "LEVEL6_COMPASS_ROOM",
    "LEVEL6_OLD_MAN_ROOM",
    "LEVEL6_DOOR_X",
    "LEVEL6_DOOR_X_LO",
    "LEVEL6_DOOR_X_HI",
    "LEVEL6",
    "LEVEL6_TRIFORCE_BIT",
    "WIZZROBE_ORANGE_TYPE",
    "ENTRY_LEFT_WALL_Y",
    "ENTRY_LEFT_DOOR_Y",
    "LEVEL6_DOOR_HOPS",
    "POST_L5_TO_LEVEL6_HOPS",
    "POST_L5_SETTLE_MAX_FRAMES",
    "POST_L5_PATH_MAX_FRAMES",
    "LEVEL6_POST_L5_SCREENS",
    "HILLS_ROCK_X",
    "HILLS_AISLE_X",
    "HILLS_NOTCH_X",
    "HILLS_NOTCH_Y",
    "HILLS_SOUTH_Y",
    "HILLS_CHANNEL_Y_LO",
    "HILLS_CHANNEL_Y_HI",
    "lost_hills_west_dir",
    "OverworldToLevel6Controller",
    "PostL5SettlePhase",
    "PostL5TriforceSettleController",
    "Level6EntryRightController",
    "Level6WestKeyDoorController",
    "level6_screen_reached",
    "level6_entrance_success",
    "level6_east_key_room",
    "level6_west_wizzrobe_room",
    "level6_hops_from",
    "make_post_l5_level6_controller",
    "post_l5_overworld_ready",
]

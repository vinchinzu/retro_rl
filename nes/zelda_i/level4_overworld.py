"""Overworld routing: post-L3 → Level 4 (Snake) raft dock → island door.

Live recon (assisted, 2026-08-08, rr-0fx)::

    Level3Complete fanfare → OW **0x74** ~(128,125) with ``raft=1`` ``tf=0x04``.
    Walk: 0x74 W@y141 → 0x73 free mid → N → 0x63 free south
      → E@y≈145–155 → 0x64 E@y141 → 0x65 N@x112 → dock **0x55**
      → N@x≈128 (Raft) → island **0x45** → door UP → level 4 room **0x71**.

Gated by real ``ADDR_RAFT`` from L3 (no inventory poke for claims).
Track: assisted first-pass only — do **not** promote Clean STATUS.

See ``docs/LEVEL4_ROUTE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_idle_action
from zelda_i.overworld import ScreenHop, path_screens_from_hops
from zelda_i.ow_path import OverworldPathController
from zelda_i.ram import (
    ADDR_LADDER,
    ADDR_RAFT,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

from zelda_i.anchors import (
    SCREEN_LEVEL4_ENTRANCE,
    SCREEN_LEVEL4_RAFT_DOCK,
    TF_BIT_L3 as LEVEL3_TRIFORCE_BIT,
    TF_BIT_L4 as LEVEL4_TRIFORCE_BIT,
)

# --- Live anchors (assisted recon 2026-08-08) ---
SOURCE_HYPOTHESIS = False
LEVEL4 = 4
LEVEL4_ENTRY_ROOM = 0x71
LEVEL4_DOOR_X = 128
LEVEL4_DOCK_RAFT_X = 128
LEVEL4_DOCK_SCREEN = SCREEN_LEVEL4_RAFT_DOCK  # 0x55 live
LEVEL4_ISLAND_SCREEN = SCREEN_LEVEL4_ENTRANCE  # 0x45 live

# Post-L3 return (Manji mouth); TF bits after L1+L2+L3 shards.
SCREEN_POST_L3_RETURN = 0x74
POST_L3_SETTLE_MAX_FRAMES = 2500
POST_L3_PATH_MAX_FRAMES = 40000
SEGMENT_MAX_FRAMES = 40000
SWORD_SWING_PERIOD = 10
SWORD_SWING_FRAMES = 3
STUCK_THRESHOLD = 50

# Live hop chain from post-L3 OW 0x74 → island door 0x45 (rr-0fx).
# Traps:
#   - 0x73: arrive east edge from 0x74 — free mid (x≈128) before UP
#   - 0x63: east exit only y≈145–155 (y=141 sticks in bush at x≈144)
#   - 0x55 dock: Raft north only near x≈128 (x≤112 never boards)
LEVEL4_HOPS_FROM_POST_L3: tuple[ScreenHop, ...] = (
    ScreenHop(0x73, "LEFT", align_y=141),
    ScreenHop(0x63, "UP", align_x=128),
    # Rock maze east corridor — live probe y≈147–149; band absorbs variance.
    ScreenHop(0x64, "RIGHT", y_band_lo=145, y_band_hi=155),
    ScreenHop(0x65, "RIGHT", align_y=141),
    ScreenHop(LEVEL4_DOCK_SCREEN, "UP", align_x=112),
    # Raft transit: walk north on dock with ADDR_RAFT → island 0x45.
    ScreenHop(LEVEL4_ISLAND_SCREEN, "UP", align_x=LEVEL4_DOCK_RAFT_X),
)
LEVEL4_POST_L3_SCREENS: tuple[int, ...] = path_screens_from_hops(
    SCREEN_POST_L3_RETURN, LEVEL4_HOPS_FROM_POST_L3
)
assert LEVEL4_POST_L3_SCREENS[0] == SCREEN_POST_L3_RETURN
assert LEVEL4_POST_L3_SCREENS[-1] == LEVEL4_ISLAND_SCREEN

# Legacy name kept for planning_report / docs (was start→dock placeholder).
LEVEL4_DOCK_HOPS: tuple[ScreenHop, ...] = LEVEL4_HOPS_FROM_POST_L3


def has_raft(ram) -> bool:
    """True when Raft inventory flag is set (L3 item)."""
    return bool(read_u8(ram, ADDR_RAFT))


def has_ladder(ram) -> bool:
    """True when Stepladder inventory flag is set (L4 dungeon item)."""
    return bool(read_u8(ram, ADDR_LADDER))


def required_caps_for_entry() -> frozenset[str]:
    """Named capabilities required to *enter* L4."""
    return frozenset({"raft"})


def required_caps_for_clear() -> frozenset[str]:
    """Caps expected by end of clear (source planning)."""
    return frozenset({"raft", "ladder"})


def missing_entry_caps(ram) -> list[str]:
    missing: list[str] = []
    if not has_raft(ram):
        missing.append("raft")
    return missing


def on_level4_dock(snap: ZeldaSnapshot) -> bool:
    """Live: mainland raft dock screen 0x55."""
    return (
        snap.level == 0
        and snap.mode == PLAY_MODE
        and snap.screen == LEVEL4_DOCK_SCREEN
    )


# Back-compat alias (old planning name).
on_level4_dock_hyp = on_level4_dock


def on_level4_island(snap: ZeldaSnapshot) -> bool:
    """Live: Snake island door screen 0x45."""
    return (
        snap.level == 0
        and snap.mode == PLAY_MODE
        and snap.screen == LEVEL4_ISLAND_SCREEN
    )


def level4_dungeon_play(snap: ZeldaSnapshot) -> bool:
    """True if snapshot is play mode inside level 4 (any room)."""
    return snap.level == LEVEL4 and snap.mode == PLAY_MODE


def level4_triforce_stop(snap: ZeldaSnapshot) -> bool:
    """Inventory stop: shard 4 bit set. Not a route-success claim by itself."""
    return bool(snap.triforce & LEVEL4_TRIFORCE_BIT)


def level4_entry_stop(snap: ZeldaSnapshot) -> bool:
    """Room-ready inside Snake entry: level 4, play mode, room 0x71."""
    return (
        snap.level == LEVEL4
        and snap.mode == PLAY_MODE
        and snap.screen == LEVEL4_ENTRY_ROOM
    )


def level4_overworld_stop(snap: ZeldaSnapshot) -> bool:
    """OW stop on island door screen (no dungeon enter)."""
    return on_level4_island(snap) and 40 < snap.link_y < 210


def post_l3_overworld_ready(ram: np.ndarray) -> bool:
    """OW play on Manji return screen with L3 triforce bit and raft."""
    snap = read_snapshot(ram)
    return (
        snap.level == 0
        and snap.mode == PLAY_MODE
        and snap.screen == SCREEN_POST_L3_RETURN
        and bool(snap.triforce & LEVEL3_TRIFORCE_BIT)
        and has_raft(ram)
    )


def level4_entrance_success(ram: np.ndarray) -> bool:
    """Room-ready inside Snake entry room 0x71."""
    return level4_entry_stop(read_snapshot(ram))


def planning_report() -> dict[str, Any]:
    """Machine-readable planning / live summary for probes / docs."""
    return {
        "level": LEVEL4,
        "name": "The Snake",
        "status": "live_entry" if not SOURCE_HYPOTHESIS else "planning",
        "source_hypothesis": SOURCE_HYPOTHESIS,
        "required_entry_caps": sorted(required_caps_for_entry()),
        "triforce_bit": LEVEL4_TRIFORCE_BIT,
        "ram": {
            "raft": hex(ADDR_RAFT),
            "ladder": hex(ADDR_LADDER),
        },
        "screens": {
            "post_l3_return": hex(SCREEN_POST_L3_RETURN),
            "dock": hex(LEVEL4_DOCK_SCREEN),
            "island_or_door": hex(LEVEL4_ISLAND_SCREEN),
            "entry_room": hex(LEVEL4_ENTRY_ROOM),
        },
        # Legacy key for run_level4_entry --plan-only.
        "screens_hypothesized": {
            "dock": hex(LEVEL4_DOCK_SCREEN),
            "island_or_door": hex(LEVEL4_ISLAND_SCREEN),
            "raft_heart_dock": hex(0x3F),
        },
        "post_l3_hops": [
            {"target": hex(h.target), "dir": h.direction} for h in LEVEL4_HOPS_FROM_POST_L3
        ],
        "dock_hops_from_start": [
            {"target": hex(h.target), "dir": h.direction} for h in LEVEL4_DOCK_HOPS
        ],
        "live": {
            "door_screen": hex(LEVEL4_ISLAND_SCREEN),
            "entry_room": hex(LEVEL4_ENTRY_ROOM),
            "dock_screen": hex(LEVEL4_DOCK_SCREEN),
            "boss_room": None,
            "verified": True,
            "bead": "rr-0fx",
            "date": "2026-08-08",
        },
        "docs": "nes/zelda_i/docs/LEVEL4_ROUTE.md",
    }


class Level4NavPhase(Enum):
    HOP = auto()
    DOOR = auto()
    DONE = auto()
    FAILED = auto()


class PostL3SettlePhase(Enum):
    WAIT = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class PostL3TriforceSettleController:
    """Idle through L3 triforce fanfare until OW 0x74 play with raft.

    Start: ``Level3Complete`` (mode 18, room 0x3d, raft=1, tf&0x04).
    """

    phase: PostL3SettlePhase = PostL3SettlePhase.WAIT
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)
    max_frames: int = POST_L3_SETTLE_MAX_FRAMES
    require_screen: int = SCREEN_POST_L3_RETURN

    def reset(self) -> None:
        self.phase = PostL3SettlePhase.WAIT
        self.frames = 0
        self.phase_frames = 0
        self.success = False
        self.notes.clear()

    def step(self, snap: ZeldaSnapshot, *, has_raft_flag: bool = True) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        if self.frames > self.max_frames:
            self.phase = PostL3SettlePhase.FAILED
            self.notes.append("settle_timeout")
            return FrameAction(nes_idle_action(), "settle_timeout")

        if (
            snap.level == 0
            and snap.mode == PLAY_MODE
            and snap.screen == self.require_screen
            and bool(snap.triforce & LEVEL3_TRIFORCE_BIT)
            and has_raft_flag
        ):
            self.success = True
            if self.phase is not PostL3SettlePhase.DONE:
                self.phase = PostL3SettlePhase.DONE
                self.notes.append("post_l3_ow_ready")
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


@dataclass
class OverworldToLevel4Controller(OverworldPathController):
    """Post-L3 OW 0x74 → dock 0x55 → raft island 0x45 → L4 room 0x71.

    Default stop is dungeon entry (room 0x71). Pass ``require_dungeon=False``
    to stop on island door screen. Pass ``stop_at_dock=True`` to stop on 0x55.
    """

    phase: Level4NavPhase = Level4NavPhase.HOP
    hops: tuple[ScreenHop, ...] = LEVEL4_HOPS_FROM_POST_L3
    require_dungeon: bool = True
    stop_at_dock: bool = False
    door_x: int | None = LEVEL4_DOOR_X
    door_dir: str = "UP"
    door_screen: int | None = LEVEL4_ISLAND_SCREEN
    entry_level: int | None = LEVEL4
    entry_room: int | None = LEVEL4_ENTRY_ROOM
    max_frames: int = POST_L3_PATH_MAX_FRAMES
    swing_period: int = SWORD_SWING_PERIOD
    swing_hold: int = SWORD_SWING_FRAMES
    stuck_threshold: int = STUCK_THRESHOLD
    require_sword: bool = True
    require_triforce_bit: int | None = LEVEL3_TRIFORCE_BIT
    # Require raft in stop checks when on OW (caller should still refuse without).
    require_raft: bool = True

    def _wants_post_hop(self) -> bool:
        if self.stop_at_dock:
            return False
        return self.require_dungeon or self.require_entrance_screen

    def _at_stop(self, snap: ZeldaSnapshot) -> bool:
        if self.stop_at_dock:
            return (
                self.hop_index >= 5  # through dock hop (index of 0x55)
                and on_level4_dock(snap)
                and 40 < snap.link_y < 210
            )
        if self.require_dungeon:
            return level4_entry_stop(snap)
        return level4_overworld_stop(snap)

    def _before_play(self, snap: ZeldaSnapshot) -> FrameAction | None:
        if snap.level not in (0, LEVEL4) and snap.level > 0:
            return self._swing("DOWN", f"exit_l{snap.level}")
        return None

    def _extra_hop_action(
        self, snap: ZeldaSnapshot, hop: ScreenHop
    ) -> FrameAction | None:
        # 0x73: free from east edge before UP (arrive x=240 from 0x74 LEFT).
        if hop.target == 0x63 and snap.screen == 0x73:
            if snap.link_x > 160:
                return self._swing("LEFT", "73_inland")
            if snap.link_x < 80:
                return self._swing("RIGHT", "73_inland")
            if hop.align_x is not None and abs(snap.link_x - hop.align_x) > 8:
                btn = "RIGHT" if snap.link_x < hop.align_x else "LEFT"
                return self._swing(btn, "73_align_x")
            return None  # default UP push

        # 0x63 east: y-band only; free south edge first (arrival y=221).
        if hop.target == 0x64 and snap.screen == 0x63:
            return self._leave_63_east(snap, hop)

        # 0x55 dock: free south edge, align raft x, push UP (engine rafts north).
        if hop.target == LEVEL4_ISLAND_SCREEN and snap.screen == LEVEL4_DOCK_SCREEN:
            return self._leave_dock_raft(snap, hop)

        return None

    def _leave_63_east(self, snap: ZeldaSnapshot, hop: ScreenHop) -> FrameAction:
        """0x63 → 0x64: east corridor opens only near y≈145–155.

        Live: y=141 sticks in bush around x≈144; y≈147–149 scrolls east.
        """
        lo = hop.y_band_lo if hop.y_band_lo is not None else 145
        hi = hop.y_band_hi if hop.y_band_hi is not None else 155
        target_y = 149
        if self.stuck > self.stuck_threshold:
            seq = ("LEFT", "UP", "RIGHT", "DOWN", "RIGHT", "UP", "LEFT")
            btn = seq[self.stuck % len(seq)]
            if self.stuck > 160:
                self.stuck = 0
            return self._swing(btn, "63_east_wiggle")
        # Free south arrival edge before y-band work.
        if snap.link_y > 180:
            return self._swing("UP", "63_free_south")
        if snap.link_y < lo:
            # Step west of east bush wall if hugging x≈144.
            if snap.link_x > 130:
                return self._swing("LEFT", "63_west_for_band")
            return self._swing("DOWN", "63_band_down")
        if snap.link_y > hi:
            return self._swing("UP", "63_band_up")
        if abs(snap.link_y - target_y) > 6 and snap.link_x < 200:
            return self._swing(
                "DOWN" if snap.link_y < target_y else "UP", "63_fine_y"
            )
        return self._swing("RIGHT", "63_east")

    def _leave_dock_raft(self, snap: ZeldaSnapshot, hop: ScreenHop) -> FrameAction:
        """0x55 → 0x45: walk north onto dock with Raft (x≈128)."""
        raft_x = hop.align_x if hop.align_x is not None else LEVEL4_DOCK_RAFT_X
        if self.stuck > self.stuck_threshold:
            seq = ("LEFT", "UP", "RIGHT", "UP", "DOWN", "UP")
            btn = seq[self.stuck % len(seq)]
            if self.stuck > 160:
                self.stuck = 0
            return self._swing(btn, "dock_raft_wiggle")
        if snap.link_y > 180:
            return self._swing("UP", "dock_free_south")
        if abs(snap.link_x - raft_x) > 6:
            btn = "RIGHT" if snap.link_x < raft_x else "LEFT"
            return self._swing(btn, "dock_align_x")
        return self._swing("UP", "dock_raft_n")

    def _after_hops(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.stop_at_dock and on_level4_dock(snap):
            return self._finish("dock_stop")
        if self.require_dungeon:
            if snap.level == LEVEL4:
                if level4_entry_stop(snap):
                    return self._finish("level4_entry")
                return FrameAction(nes_idle_action(), "dungeon_settle")
            self._set_phase(Level4NavPhase.DOOR, "door_hunt")
            if snap.level == 0 and snap.screen != LEVEL4_ISLAND_SCREEN:
                # Drifted — nudge from dock if still water-adjacent.
                if snap.screen == LEVEL4_DOCK_SCREEN:
                    return self._swing("UP", "door_return_raft")
                return self._swing("UP", "door_return")
            # Approach from south of mouth then align x and push UP.
            if snap.link_y < 100:
                return self._swing("DOWN", "door_south")
            if self.door_x is not None and abs(snap.link_x - self.door_x) > 5:
                btn = "LEFT" if snap.link_x > self.door_x else "RIGHT"
                return self._swing(btn, "door_ax")
            return self._swing("UP", "door_hunt")
        if on_level4_island(snap):
            return self._finish("island_stop")
        return self._finish("hops_complete")

    def report(self) -> dict[str, Any]:
        out = super().report()
        out["path"] = "post_l3"
        out["start_screen"] = f"0x{SCREEN_POST_L3_RETURN:02x}"
        out["dock_screen"] = f"0x{LEVEL4_DOCK_SCREEN:02x}"
        out["island_screen"] = f"0x{LEVEL4_ISLAND_SCREEN:02x}"
        out["entry_room"] = f"0x{LEVEL4_ENTRY_ROOM:02x}"
        out["door_x"] = self.door_x
        out["require_dungeon"] = self.require_dungeon
        out["stop_at_dock"] = self.stop_at_dock
        out["source_hypothesis"] = SOURCE_HYPOTHESIS
        return out

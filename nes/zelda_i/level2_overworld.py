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

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_idle_action
from zelda_i.overworld import (
    LEVEL2_5C_MAZE_WAYPOINTS,
    LEVEL2_DOOR_HOPS,
    LEVEL2_PATH_HOPS,
    LEVEL2_PATH_SCREENS,
    ScreenHop,
    is_5c_maze_hop,
)
from zelda_i.ow_path import OverworldPathController
from zelda_i.ram import (
    PLAY_MODE,
    SCREEN_LEVEL1_ENTRANCE,
    SCREEN_LEVEL2_ENTRANCE,
    SCREEN_LEVEL2_ENTRY_ROOM,
    ZeldaSnapshot,
    read_snapshot,
)

# --- Geometry / budgets ---
SCREEN_LEVEL2 = SCREEN_LEVEL2_ENTRANCE
LEVEL2_ENTRY_ROOM = SCREEN_LEVEL2_ENTRY_ROOM  # 0x7d after mode-16→5 settle
LEVEL2_DOOR_X = 112  # Moon overworld door UP lane on 0x3C
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

# Verified prefix stop 0x4A is *not* on LEVEL2_DOOR_HOPS (door path turns
# east at 0x59→0x5A). Live probe: 0x4A has **no south exit** to 0x5A (rock
# wall). Rejoin west → 0x49 → south → 0x59 → then door hops from 0x5A.
LEVEL2_REJOIN_4A_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x49, "LEFT", align_y=141),
    ScreenHop(0x59, "DOWN", align_x=112),
)
# Back-compat alias (was incorrectly a single south hop to 0x5A).
LEVEL2_REJOIN_4A_TO_5A = LEVEL2_REJOIN_4A_HOPS[0]

# Clean heart-safe suffix after farm on 0x4A (probe 2026-08-06):
# rejoin → east @y≈140 into 0x5A → *corridor clear* → align_y=140 to 0x5B → maze.
# Default LEVEL2_DOOR_HOPS y-bands lose too many hearts for Clean.
LEVEL2_CLEAN_FROM_4A_TO_5A: tuple[ScreenHop, ...] = (
    ScreenHop(0x49, "LEFT", align_y=141),
    ScreenHop(0x59, "DOWN", align_x=112),
    ScreenHop(0x5A, "RIGHT", align_y=140),
)
LEVEL2_CLEAN_FROM_5A_TO_3C: tuple[ScreenHop, ...] = (
    ScreenHop(0x5B, "RIGHT", align_y=140),
    ScreenHop(0x5C, "RIGHT", y_band_lo=80, y_band_hi=95),
    ScreenHop(0x5D, "RIGHT", y_band_lo=120, y_band_hi=140),
    ScreenHop(0x4D, "UP", align_x=52),
    ScreenHop(0x4C, "LEFT", y_band_lo=120, y_band_hi=170),
    ScreenHop(0x3C, "UP", align_x=112),
)


def level2_door_hops_from(screen: int) -> tuple[ScreenHop, ...]:
    """Remaining door-path hops after ``screen`` (maze-aware full table).

    ``LEVEL2_DOOR_HOPS`` runs 0x37→…→0x3C via 0x5A (not via 0x4A/0x4B).
    From the verified prefix stop 0x4A, rejoin west/south via 0x49→0x59
    (0x4A has no south corridor to 0x5A).
    """
    targets = [h.target for h in LEVEL2_DOOR_HOPS]
    if screen in targets:
        return LEVEL2_DOOR_HOPS[targets.index(screen) + 1 :]
    if screen == 0x4A:
        # After 0x59 on door path comes 0x5A…
        return LEVEL2_REJOIN_4A_HOPS + LEVEL2_DOOR_HOPS[targets.index(0x59) + 1 :]
    if screen == 0x49:
        # Prefix intermediate: drop south to 0x59 then resume door path.
        return (ScreenHop(0x59, "DOWN", align_x=112),) + LEVEL2_DOOR_HOPS[
            targets.index(0x59) + 1 :
        ]
    if screen == 0x4B:
        # Wrong north-entry trap screen; west back toward 0x4A then rejoin.
        return (ScreenHop(0x4A, "LEFT", align_y=141),) + level2_door_hops_from(0x4A)
    if screen == 0x37:
        return LEVEL2_DOOR_HOPS
    return LEVEL2_DOOR_HOPS


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
class OverworldToLevel2Controller(OverworldPathController):
    """Walk from post-Triforce overworld 0x37 through the verified Level 2 path.

    Default stop is screen **0x4A** with sword and triforce bit 0 — the current
    health-stable milestone. Pass ``door_path=True`` (or ``hops=LEVEL2_DOOR_HOPS``)
    for the full Moon-door route via 0x5A/0x5C maze to 0x3C. Set
    ``require_level2_screen=True`` / ``require_dungeon=True`` for door hunt or
    dungeon entry after hops complete.
    """

    phase: Level2NavPhase = Level2NavPhase.HOP
    require_level2_screen: bool = False
    require_dungeon: bool = False
    door_path: bool = False
    hops: tuple[ScreenHop, ...] = LEVEL2_PATH_HOPS
    maze_waypoints: tuple[tuple[int, int], ...] = LEVEL2_5C_MAZE_WAYPOINTS
    maze_hop_pred: Any = None  # set in __post_init__ to is_5c_maze_hop
    door_x: int | None = LEVEL2_DOOR_X
    entry_level: int | None = 2
    entry_room: int | None = LEVEL2_ENTRY_ROOM
    door_screen: int | None = SCREEN_LEVEL2
    max_frames: int = SEGMENT_MAX_FRAMES
    swing_period: int = SWORD_SWING_PERIOD
    swing_hold: int = SWORD_SWING_FRAMES
    stuck_threshold: int = STUCK_THRESHOLD
    require_sword: bool = True
    require_triforce_bit: int | None = LEVEL1_TRIFORCE_BIT

    def __post_init__(self) -> None:
        if self.door_path:
            self.hops = LEVEL2_DOOR_HOPS
        if self.maze_hop_pred is None:
            self.maze_hop_pred = is_5c_maze_hop

    def reset(self) -> None:
        super().reset()
        if self.door_path:
            self.hops = LEVEL2_DOOR_HOPS

    def _wants_post_hop(self) -> bool:
        return self.require_level2_screen or self.require_dungeon

    def _at_stop(self, snap: ZeldaSnapshot) -> bool:
        if self.require_dungeon:
            # Door enter is mode 16 with screen still 0x3C; room-ready is
            # level==2, mode 5, entry room 0x7d (~180 idle frames).
            return (
                snap.level == 2
                and snap.mode == PLAY_MODE
                and snap.screen == LEVEL2_ENTRY_ROOM
            )
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

    def _before_play(self, snap: ZeldaSnapshot) -> FrameAction | None:
        if snap.level == 1:
            return self._swing("DOWN", "exit_l1")
        return None

    def _after_hops(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.require_level2_screen or self.require_dungeon:
            # Mid-dungeon-enter settle (mode 16/2/3/4) — idle until room-ready.
            if snap.level == 2:
                return FrameAction(nes_idle_action(), "dungeon_settle")
            # Moon door on 0x3C: align x≈112 then push UP (same as L1 mouth).
            if abs(snap.link_x - LEVEL2_DOOR_X) > 5:
                btn = "LEFT" if snap.link_x > LEVEL2_DOOR_X else "RIGHT"
                return self._swing(btn, "door_ax")
            return self._swing("UP", "door_hunt")
        return self._finish("hops_complete")

    def _finish(self, note: str = "path_stop") -> FrameAction:
        label = {
            "path_stop": "level2_path_stop",
            "path_complete": "path_prefix_complete",
            "hops_complete": "hops_complete",
        }.get(note, note)
        self.success = True
        self._set_phase(Level2NavPhase.DONE, label)
        return FrameAction(nes_idle_action(), "done")

    def _on_hop_advanced(
        self, snap: ZeldaSnapshot, completed_hop: ScreenHop
    ) -> FrameAction:
        if self.hop_index >= len(self.hops) and not self._wants_post_hop():
            return self._finish("path_prefix_complete")
        return FrameAction(nes_idle_action(), "hop_advance")

    def report(self) -> dict[str, Any]:
        hop = None
        if self.hop_index < len(self.hops):
            current = self.hops[self.hop_index]
            hop = {
                "index": self.hop_index,
                "target": current.target,
                "direction": current.direction,
                "maze": is_5c_maze_hop(current),
            }
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "hop_index": self.hop_index,
            "hop": hop,
            "maze_wp_index": self.maze_wp_index,
            "door_path": self.door_path,
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
    """Room-ready inside Moon entry: level 2, play mode, room 0x7d."""
    snap = read_snapshot(ram)
    return (
        snap.level == 2
        and snap.mode == PLAY_MODE
        and snap.screen == LEVEL2_ENTRY_ROOM
    )


def post_triforce_overworld_ready(ram: np.ndarray) -> bool:
    snap = read_snapshot(ram)
    return (
        snap.level == 0
        and snap.mode == PLAY_MODE
        and snap.screen == SCREEN_LEVEL1_ENTRANCE
        and bool(snap.triforce & LEVEL1_TRIFORCE_BIT)
    )

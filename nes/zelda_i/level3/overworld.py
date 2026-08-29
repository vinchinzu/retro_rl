"""Overworld routing: start / post-L2 → Level 3 (Manji) door + entry.

Live recon (assisted, 2026-08-06 / 2026-08-07)::

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

Post-L2 TF settle (live 2026-08-07)::

    After Moon triforce bit ``0x02``, mode 18 fanfare ~800f → OW **0x3C**
    ~(112, 125) with ``tf==0x03``. Checkpoint ``Level2ExitOverworld``.
    Path: reverse L2 door corridor (5C maze west) → 5B bush leave →
    west forest join → Manji door (see ``LEVEL3_HOPS_FROM_POST_L2``).

Item: Raft (``ADDR_RAFT=0x0660``). Boss: Manhandla. TF bit ``0x04``.
Track: assisted first-pass only — do **not** promote Clean STATUS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_idle_action
from zelda_i.overworld.common import unstick_wiggle
from zelda_i.overworld.graph import (
    LEVEL2_5C_MAZE_WAYPOINTS,
    SCREEN_5C_MAZE,
    ScreenHop,
    path_screens_from_hops,
)
from zelda_i.overworld.path import OverworldPathController
from zelda_i.ram import PLAY_MODE, SCREEN_START, ZeldaSnapshot, read_snapshot

# --- Live anchors (assisted recon 2026-08-06); screens from anchors ---
from zelda_i.anchors import (
    SCREEN_LEVEL3_ENTRANCE,
    SCREEN_LEVEL3_ENTRY_ROOM,
    TF_BIT_L2 as LEVEL2_TRIFORCE_BIT,
    TF_BIT_L3 as LEVEL3_TRIFORCE_BIT,
)

LEVEL3_DOOR_X = 128  # exit-spawn x; UP re-enter after y>130 approach
LEVEL3_DOOR_APPROACH_Y = 140
LEVEL3 = 3
# Post-L2 return (Moon mouth); TF bits after L1+L2 shards.
SCREEN_POST_L2_RETURN = 0x3C
POST_L2_SETTLE_MAX_FRAMES = 2500
POST_L2_PATH_MAX_FRAMES = 45000

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

# --- Post-L2 (Moon TF) → Manji door (live assisted 2026-08-07, rr-rnx) ---
# Reverse L2 door corridor from OW 0x3C, reverse 0x5C maze west, leave 0x5B
# north-bush via inland→south→west, then L3 west forest → 0x74.
# Traps (must match AGENTS / LEVEL3_ROUTE):
#   - 0x4C east: y∈[133,145] only (y=149 solid forever; y≈125 sticks on rock)
#   - 0x5C reverse: denser channel waypoints; NO y_band on 0x5B hop
#   - 0x64 west: y≈125–150 (y≈109 wall-hug sticks) — `_leave_64_west`
#   - 0x59→0x58 west is y≈120–145 (NOT L2 east band 148–162)
#   - 0x63→0x73 needs free south-band (align_x=112 alone fails from east entry)
# Reverse of L2 east maze (live 2026-08-07): enter east ~y132, west to
# channel x192, UP denser steps, then west along y≈92 to 0x5B.
# Denser vertical channel than plain reverse — sparse (192,108)→(192,84)
# stuck around y117 in bushes.
LEVEL3_5C_MAZE_REVERSE_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (240, 132),
    (224, 132),
    (208, 132),
    (200, 132),
    (192, 132),
    (192, 120),
    (192, 108),
    (192, 100),
    (192, 92),
    (192, 84),
    (184, 92),
    (168, 92),
    (144, 92),
    (120, 92),
    (104, 92),
    (80, 92),
    (56, 92),
    (40, 92),
    (20, 92),
)


def is_5c_maze_reverse_hop(hop: ScreenHop) -> bool:
    """True for the 0x5C→0x5B west hop that needs reverse maze waypoints."""
    return hop.target == 0x5B and hop.direction == "LEFT"


LEVEL3_HOPS_FROM_POST_L2: tuple[ScreenHop, ...] = (
    ScreenHop(0x4C, "DOWN", align_x=112),
    # 0x4c east: live probe 2026-08-07 — y∈[133,145] OK; y=149 solid forever.
    ScreenHop(0x4D, "RIGHT", y_band_lo=133, y_band_hi=145),
    ScreenHop(0x5D, "DOWN", align_x=52),
    ScreenHop(0x5C, "LEFT", y_band_lo=120, y_band_hi=140),
    # Reverse 0x5C maze (waypoints only — NO y_band: east-edge band force
    # was fighting reverse path y≈132 → channel → y≈92 west exit).
    ScreenHop(0x5B, "LEFT"),
    # 0x5B north-bush leave is multi-phase (see controller _extra_hop_action).
    ScreenHop(0x5A, "LEFT", y_band_lo=130, y_band_hi=150),
    ScreenHop(0x59, "LEFT", y_band_lo=125, y_band_hi=145),
    ScreenHop(0x58, "LEFT", y_band_lo=120, y_band_hi=145),
    ScreenHop(0x57, "LEFT", y_band_lo=148, y_band_hi=162),
    ScreenHop(0x56, "LEFT", y_band_lo=148, y_band_hi=162),
    ScreenHop(0x55, "LEFT", align_y=133),
    ScreenHop(0x65, "DOWN", align_x=112),
    ScreenHop(0x64, "LEFT", align_y=141),
    # 0x64→0x63: west exit needs mid-south band (y≈125–150). Single align_y=133
    # flaked 1/2 at ~(24,109) — use y_band + controller leave_64.
    ScreenHop(0x63, "LEFT", y_band_lo=125, y_band_hi=150),
    # 0x63 rock maze: free south then DOWN (controller override).
    ScreenHop(0x73, "DOWN", align_x=112),
    ScreenHop(0x74, "RIGHT", align_y=117),
)
LEVEL3_POST_L2_SCREENS: tuple[int, ...] = path_screens_from_hops(
    SCREEN_POST_L2_RETURN, LEVEL3_HOPS_FROM_POST_L2
)
assert LEVEL3_POST_L2_SCREENS[0] == SCREEN_POST_L2_RETURN
assert LEVEL3_POST_L2_SCREENS[-1] == SCREEN_LEVEL3_ENTRANCE


class Level3NavPhase(Enum):
    HOP = auto()
    DOOR = auto()
    DONE = auto()
    FAILED = auto()


class PostL2SettlePhase(Enum):
    WAIT = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class PostL2TriforceSettleController:
    """Idle through L2 mode-18 fanfare until overworld play on 0x3C.

    Live: ~800 idle frames after TF collect → mode 5 OW **0x3C** ~(112, 125)
    with triforce & 0x02 (and usually & 0x01 from L1). Prefer live settle or
    ``Level2ExitOverworld``; mid-fanfare ``Level2Complete`` reload can stick.
    """

    phase: PostL2SettlePhase = PostL2SettlePhase.WAIT
    frames: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)
    max_frames: int = POST_L2_SETTLE_MAX_FRAMES
    require_screen: int = SCREEN_POST_L2_RETURN

    def reset(self) -> None:
        self.phase = PostL2SettlePhase.WAIT
        self.frames = 0
        self.success = False
        self.notes.clear()

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.frames >= self.max_frames:
            self.phase = PostL2SettlePhase.FAILED
            self.notes.append("timeout")
            return FrameAction(nes_idle_action(), "timeout")

        if (
            snap.level == 0
            and snap.mode == PLAY_MODE
            and snap.screen == self.require_screen
            and (snap.triforce & LEVEL2_TRIFORCE_BIT)
        ):
            self.success = True
            if self.phase is not PostL2SettlePhase.DONE:
                self.phase = PostL2SettlePhase.DONE
                self.notes.append("overworld_after_l2_triforce")
            return FrameAction(nes_idle_action(), "done")

        return FrameAction(nes_idle_action(), f"settle_mode_{snap.mode}")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "require_screen": f"0x{self.require_screen:02x}",
        }


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
    # Door tour after simple hunt stalls (rock maze on 0x74).
    door_tour_period: int = 2500

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
            if snap.level == 0 and snap.screen != SCREEN_LEVEL3_ENTRANCE:
                # Drifted off door screen — nudge back from 0x73.
                btn = "RIGHT" if snap.screen == 0x73 else "LEFT"
                return self._swing(btn, "door_return")
            # Light rock-maze tour if simple hunt stalls.
            if self.phase_frames > 0 and self.phase_frames % self.door_tour_period >= (
                self.door_tour_period * 3 // 5
            ):
                wps = (
                    (40, 140),
                    (100, 140),
                    (160, 140),
                    (200, 140),
                    (200, 100),
                    (128, 100),
                    (128, 160),
                    (80, 160),
                )
                wp = wps[(self.phase_frames // 50) % len(wps)]
                if abs(snap.link_x - wp[0]) > 6:
                    btn = "RIGHT" if snap.link_x < wp[0] else "LEFT"
                    return self._swing(btn, "door_tour_x")
                if abs(snap.link_y - wp[1]) > 6:
                    btn = "DOWN" if snap.link_y < wp[1] else "UP"
                    return self._swing(btn, "door_tour_y")
                return self._swing("UP", "door_tour_up")
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


@dataclass
class OverworldPostL2ToLevel3Controller(OverworldToLevel3Controller):
    """Post-L2 OW 0x3C → Manji door / entry (assisted path, rr-rnx).

    Special screens handled in ``_extra_hop_action`` / maze reverse:
      - 0x5C→0x5B reverse maze waypoints (exit LEFT)
      - 0x5B north-bush inland → south → west → 0x5A
      - 0x63 free south-band → 0x73
    """

    hops: tuple[ScreenHop, ...] = LEVEL3_HOPS_FROM_POST_L2
    maze_waypoints: tuple[tuple[int, int], ...] = LEVEL3_5C_MAZE_REVERSE_WAYPOINTS
    maze_hop_pred: Any = None  # set in __post_init__
    maze_screen: int = SCREEN_5C_MAZE
    max_frames: int = POST_L2_PATH_MAX_FRAMES
    require_sword: bool = True
    require_triforce_bit: int | None = LEVEL2_TRIFORCE_BIT
    # 0x5B leave sub-phase: INLAND | SOUTH | WEST
    leave_5b_phase: str = "INLAND"

    def __post_init__(self) -> None:
        if self.maze_hop_pred is None:
            self.maze_hop_pred = is_5c_maze_reverse_hop

    def _follow_maze(self, snap: ZeldaSnapshot) -> FrameAction:
        """Reverse 0x5C maze — policy matched to live 3/3 probe.

        nearest-wp snap on start; advance tol 8; x-then-y toward wp; stuck
        cycles UP/LEFT/DOWN (not random unstick that leaves the corridor).
        """
        if not self.maze_waypoints:
            return self._swing("LEFT", "maze_no_waypoints")

        if "maze_start" not in self.notes:
            self.notes.append("maze_start")
            best_i = 0
            best_d = 10**9
            for i, (wx, wy) in enumerate(self.maze_waypoints):
                d = abs(snap.link_x - wx) + abs(snap.link_y - wy)
                if d < best_d:
                    best_d = d
                    best_i = i
            self.maze_wp_index = best_i

        if self.maze_wp_index >= len(self.maze_waypoints):
            return self._swing("LEFT", "maze_exit")

        tx, ty = self.maze_waypoints[self.maze_wp_index]
        if abs(snap.link_x - tx) <= 8 and abs(snap.link_y - ty) <= 8:
            self.maze_wp_index += 1
            self.stuck = 0
            if self.maze_wp_index >= len(self.maze_waypoints):
                return self._swing("LEFT", "maze_exit")
            tx, ty = self.maze_waypoints[self.maze_wp_index]

        if self.stuck > 40:
            btn = ("UP", "LEFT", "DOWN")[self.stuck % 3]
            if self.stuck > 120:
                self.stuck = 0
            return self._swing(btn, "maze_stuck")

        if abs(snap.link_x - tx) > 6:
            direction = "RIGHT" if snap.link_x < tx else "LEFT"
        elif abs(snap.link_y - ty) > 6:
            direction = "DOWN" if snap.link_y < ty else "UP"
        else:
            direction = "LEFT"
        return self._swing(direction, f"maze_rev_wp{self.maze_wp_index}")

    def _extra_hop_action(
        self, snap: ZeldaSnapshot, hop: ScreenHop
    ) -> FrameAction | None:
        # Maze owns 0x5C while reverse hop is active — do not steal frames.
        if self._in_maze_phase(snap, hop) or is_5c_maze_reverse_hop(hop):
            return None
        # 0x5B: north bush arrival → inland → south corridor → west 0x5A
        if hop.target == 0x5A and snap.screen == 0x5B:
            return self._leave_5b(snap)
        if hop.target == 0x5A and snap.screen == 0x5C:
            # Bounced back into maze corridor — push west.
            return self._swing("LEFT", "5b_reenter_fix")
        # 0x64 → 0x63: west corridor only y≈125–150 (specialized leave).
        if hop.target == 0x63 and snap.screen == 0x64:
            return self._leave_64_west(snap)
        # 0x63 rock maze: free south then DOWN (pure align_x=112 fails east entry)
        if hop.target == 0x73 and snap.screen == 0x63:
            return self._leave_63_south(snap)
        # LEFT hop y-corridor: align height before west push.
        # West wall + wrong y → RIGHT inland first (other west rocks).
        # East edge + wrong y → vertical then LEFT (0x58/0x56).
        if hop.direction == "LEFT":
            need_y = False
            y_btn = "DOWN"
            if hop.align_y is not None and abs(snap.link_y - hop.align_y) > 5:
                need_y = True
                y_btn = "DOWN" if snap.link_y < hop.align_y else "UP"
            elif hop.y_band_lo is not None and hop.y_band_hi is not None:
                if snap.link_y < hop.y_band_lo:
                    need_y = True
                    y_btn = "DOWN"
                elif snap.link_y > hop.y_band_hi:
                    need_y = True
                    y_btn = "UP"
            if need_y:
                if snap.link_x < 48:
                    return self._swing("RIGHT", "west_inland_for_y")
                if snap.link_x > 180:
                    return self._swing(y_btn, "east_align_y")
                return self._swing(y_btn, "align_y_left")
            if snap.link_x > 200:
                return self._swing("LEFT", "inland_from_east")
        return None

    def _leave_5b(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.stuck > self.stuck_threshold:
            seq = ("LEFT", "DOWN", "LEFT", "DOWN", "RIGHT", "DOWN", "LEFT", "UP")
            btn = seq[self.stuck % len(seq)]
            if self.stuck > 140:
                self.stuck = 0
            return self._swing(btn, "5b_leave_wiggle")

        phase = self.leave_5b_phase
        if phase == "INLAND":
            if snap.link_x > 130:
                return self._swing("LEFT", "5b_inland")
            if snap.link_x < 90:
                return self._swing("RIGHT", "5b_inland")
            self.leave_5b_phase = "SOUTH"
            self.notes.append("5b_south")
            return self._swing("DOWN", "5b_south")
        if phase == "SOUTH":
            if snap.link_y >= 130:
                self.leave_5b_phase = "WEST"
                self.notes.append("5b_west")
                return self._swing("LEFT", "5b_west")
            if self.stuck > 20:
                tx = 100 + ((self.stuck // 5) % 8) * 8
                if abs(snap.link_x - tx) > 4:
                    btn = "RIGHT" if snap.link_x < tx else "LEFT"
                    return self._swing(btn, "5b_south_hunt")
            return self._swing("DOWN", "5b_south")
        # WEST
        if snap.link_y < 128:
            return self._swing("DOWN", "5b_west_y")
        if snap.link_y > 155:
            return self._swing("UP", "5b_west_y")
        return self._swing("LEFT", "5b_west")

    def _leave_64_west(self, snap: ZeldaSnapshot) -> FrameAction:
        """0x64 → 0x63: west corridor only opens near y≈125–150.

        Flake: wall-hug LEFT at y≈109 never scrolls. Step inland if on the
        west rock face, align y-band, then pure LEFT.
        """
        lo, hi = 125, 150
        target_y = 141
        if self.stuck > self.stuck_threshold:
            seq = ("RIGHT", "DOWN", "LEFT", "DOWN", "RIGHT", "UP", "LEFT")
            btn = seq[self.stuck % len(seq)]
            if self.stuck > 160:
                self.stuck = 0
            return self._swing(btn, "64_west_wiggle")
        # On west wall with wrong y: step right into open sand first.
        if snap.link_x < 48 and not (lo <= snap.link_y <= hi):
            return self._swing("RIGHT", "64_inland_for_y")
        if snap.link_y < lo:
            if abs(snap.link_x - 80) > 12 and snap.link_x < 60:
                return self._swing("RIGHT", "64_inland_down")
            return self._swing("DOWN", "64_band_down")
        if snap.link_y > hi:
            return self._swing("UP", "64_band_up")
        # In band: micro-align then LEFT.
        if abs(snap.link_y - target_y) > 8 and snap.link_x > 40:
            return self._swing(
                "DOWN" if snap.link_y < target_y else "UP", "64_fine_y"
            )
        return self._swing("LEFT", "64_west")

    def _leave_63_south(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.stuck > self.stuck_threshold:
            action, self.stuck = unstick_wiggle(self.stuck, reason="63_south")
            return action
        if snap.link_y < 170:
            if self.stuck > 25:
                return self._swing(("LEFT", "RIGHT", "DOWN")[self.stuck % 3], "63_gap")
            if snap.link_x >= 200:
                return self._swing("LEFT", "63_inland")
            if snap.link_x < 40:
                return self._swing("RIGHT", "63_inland")
            return self._swing("DOWN", "63_south")
        if self.stuck > 15:
            btn = "LEFT" if (self.stuck // 10) % 2 == 0 else "RIGHT"
            return self._swing(btn, "63_south_edge")
        return self._swing("DOWN", "63_south_exit")

    def report(self) -> dict[str, Any]:
        out = super().report()
        out["path"] = "post_l2"
        out["leave_5b_phase"] = self.leave_5b_phase
        out["start_screen"] = f"0x{SCREEN_POST_L2_RETURN:02x}"
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


def post_l2_overworld_ready(ram: np.ndarray) -> bool:
    """OW play on Moon return screen with L2 triforce bit."""
    snap = read_snapshot(ram)
    return (
        snap.level == 0
        and snap.mode == PLAY_MODE
        and snap.screen == SCREEN_POST_L2_RETURN
        and bool(snap.triforce & LEVEL2_TRIFORCE_BIT)
    )

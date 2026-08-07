"""Overworld routing: start → Level 8 (Lion) candle-bush screen 0x6D.

Level 8 mouth is under a **lone bush** on overworld **0x6D**, revealed only
with Blue/Red Candle (``ADDR_CANDLE=0x065B``). Source walkthrough (Zelda
Dungeon)::

    From start: right 4, up 2, right, down, right; burn lone bush.

Naive grid decode ``0x77→…→0x6D`` via ``0x79`` hits the rocky dead-end pocket
(same trap as L2). **Live assisted path (2026-08-06)** detours L1-style north
then east along the L2 door corridor + 0x5C maze, then south into the bush
dead-end::

    0x77 E@y≈140 → 0x78 N@x≈48 → 0x68 N@x≈48 → 0x58
      E@y≈155 → 0x59 E → 0x5A E → 0x5B E@y≈88 → 0x5C
      [maze east] → 0x5D S@x≈48 → **0x6D** (bush pocket)

Burn/enter requires candle (shop Blue ~60R or L7 Red). Assist contract forbids
inventory poke — without candle, stop at ``Level8BushOW`` / ``OW_6D``.

Items (source): Book of Magic ``0x0661``, Magical Key ``0x0664`` (optional for
credits). Boss Gleeok 4-head. Triforce bit ``0x80``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
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
    LEVEL2_5C_MAZE_WAYPOINTS,
    ScreenHop,
    path_screens_from_hops,
)
from zelda_i.ram import (
    ADDR_CANDLE,
    PLAY_MODE,
    SCREEN_START,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

# --- Anchors ---
LEVEL_8 = 8
TRIFORCE_BIT_L8 = 0x80
# Live: bush dead-end pocket south of 0x5D @ x≈48.
SCREEN_LEVEL8_BUSH = 0x6D
SCREEN_LEVEL8_BUSH_PLANNED = SCREEN_LEVEL8_BUSH  # alias
# Unknown until live dungeon settle (mode 16→5, level==8) after burn.
SCREEN_LEVEL8_ENTRY_ROOM: int | None = None

ADDR_CANDLE_ITEM = ADDR_CANDLE  # 0x065B

SEGMENT_MAX_FRAMES = 50000
SWORD_SWING_PERIOD = 10
SWORD_SWING_FRAMES = 3
STUCK_THRESHOLD = 50
MAZE_WAYPOINT_TOL = 6
SCREEN_5C_MAZE = 0x5C
MAZE_HOP_TARGET = 0x5D

# Live-verified bush approach (assisted 2026-08-06). 0x5C→0x5D needs maze
# waypoints (same BFS path as L2 door). Final hop 0x5D→0x6D south @x≈48.
LEVEL8_BUSH_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x78, "RIGHT", align_y=140),
    ScreenHop(0x68, "UP", align_x=48),
    ScreenHop(0x58, "UP", align_x=48),
    ScreenHop(0x59, "RIGHT", y_band_lo=148, y_band_hi=162),
    ScreenHop(0x5A, "RIGHT", y_band_lo=120, y_band_hi=145),
    # North bush corridor into 0x5C (y≈80–95), not south pocket on 0x5B.
    ScreenHop(0x5B, "RIGHT", y_band_lo=130, y_band_hi=150),
    ScreenHop(0x5C, "RIGHT", y_band_lo=80, y_band_hi=95),
    ScreenHop(0x5D, "RIGHT", y_band_lo=120, y_band_hi=140),
    ScreenHop(0x6D, "DOWN", align_x=48),
)
LEVEL8_BUSH_SCREENS: tuple[int, ...] = path_screens_from_hops(
    SCREEN_START, LEVEL8_BUSH_HOPS
)

# Shared maze geometry with L2 door path (east @y≈88 → channel → east @y≈128).
LEVEL8_5C_MAZE_WAYPOINTS: tuple[tuple[int, int], ...] = LEVEL2_5C_MAZE_WAYPOINTS

# Source shop planning (Blue Candle ≈60 rupees). Early IGN path (N then W of
# start) did not yield a live west exit off 0x67 in this recon — shop screen
# still source-only. See LEVEL8_ROUTE.md.
CANDLE_SHOP_PRICE_SOURCE = 60
CANDLE_SHOP_SCREEN_LIVE: int | None = None

# Back-compat names used by probe --path
LEVEL8_BUSH_HOPS_VIA_6B_EAST = LEVEL8_BUSH_HOPS
LEVEL8_BUSH_HOPS_VIA_58 = LEVEL8_BUSH_HOPS


class Level8NavPhase(Enum):
    HOP = auto()
    BURN = auto()
    ENTER = auto()
    DONE = auto()
    FAILED = auto()


def is_5c_maze_hop(hop: ScreenHop) -> bool:
    return hop.target == MAZE_HOP_TARGET and hop.direction == "RIGHT"


@dataclass
class OverworldToLevel8Controller:
    """Walk hop table from start toward L8 bush 0x6D; optional burn/enter.

    Does **not** require triforce bits. Candle acquisition is external: stop on
    bush screen if ``ADDR_CANDLE`` is 0. Burn/enter are best-effort (B-item
    must already be candle).
    """

    hop_index: int = 0
    phase: Level8NavPhase = Level8NavPhase.HOP
    frames: int = 0
    phase_frames: int = 0
    stuck: int = 0
    last_x: int = -1
    last_y: int = -1
    last_screen: int = -1
    success: bool = False
    notes: list[str] = field(default_factory=list)
    hops: tuple[ScreenHop, ...] = LEVEL8_BUSH_HOPS
    maze_waypoints: tuple[tuple[int, int], ...] = LEVEL8_5C_MAZE_WAYPOINTS
    maze_wp_index: int = 0
    burn_bush: bool = False
    enter_dungeon: bool = False
    bush_x: int = 120
    bush_y: int = 120
    burn_frames: int = 0
    burn_budget: int = 400

    def reset(self) -> None:
        self.hop_index = 0
        self.phase = Level8NavPhase.HOP
        self.frames = 0
        self.phase_frames = 0
        self.stuck = 0
        self.last_x = -1
        self.last_y = -1
        self.last_screen = -1
        self.success = False
        self.notes.clear()
        self.maze_wp_index = 0
        self.burn_frames = 0

    def _set_phase(self, phase: Level8NavPhase, note: str = "") -> None:
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

    def end_screen(self) -> int:
        return self.hops[-1].target if self.hops else SCREEN_LEVEL8_BUSH

    def _at_bush_screen(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.level == 0
            and snap.mode == PLAY_MODE
            and snap.screen == self.end_screen()
            and 40 < snap.link_y < 210
        )

    def _in_level8(self, snap: ZeldaSnapshot) -> bool:
        return snap.level == LEVEL_8 and snap.mode == PLAY_MODE

    def _advance_hop(self, snap: ZeldaSnapshot, hop: ScreenHop) -> FrameAction | None:
        if (
            snap.screen != hop.target
            or snap.mode not in (PLAY_MODE, 8)
            or snap.transitioning
            or on_arrival_edge(hop.direction, snap)
        ):
            return None
        self.notes.append(f"hop_{self.hop_index}_{hop.target:02x}")
        if is_5c_maze_hop(hop):
            self.notes.append("maze_complete")
        self.hop_index += 1
        self.stuck = 0
        self.phase_frames = 0
        self.maze_wp_index = 0
        if self.hop_index >= len(self.hops):
            if self.burn_bush or self.enter_dungeon:
                self._set_phase(Level8NavPhase.BURN, "at_bush_screen")
                return FrameAction(nes_idle_action(), "bush_ready")
            self.success = True
            self._set_phase(Level8NavPhase.DONE, "bush_screen_reached")
            return FrameAction(nes_idle_action(), "done")
        return FrameAction(nes_idle_action(), "hop_advance")

    def _follow_5c_maze(self, snap: ZeldaSnapshot) -> FrameAction:
        if not self.maze_waypoints:
            return self._swing("RIGHT", "maze_no_waypoints")
        if "maze_start" not in self.notes:
            self.notes.append("maze_start")
        if self.maze_wp_index >= len(self.maze_waypoints):
            return self._swing("RIGHT", "maze_exit")
        tx, ty = self.maze_waypoints[self.maze_wp_index]
        if (
            abs(snap.link_x - tx) <= MAZE_WAYPOINT_TOL
            and abs(snap.link_y - ty) <= MAZE_WAYPOINT_TOL
        ):
            self.maze_wp_index += 1
            self.stuck = 0
            if self.maze_wp_index >= len(self.maze_waypoints):
                return self._swing("RIGHT", "maze_exit")
            tx, ty = self.maze_waypoints[self.maze_wp_index]
        if self.stuck > STUCK_THRESHOLD:
            action, self.stuck = unstick_wiggle(self.stuck, reason="maze_unstick")
            return action
        dx = tx - snap.link_x
        dy = ty - snap.link_y
        if abs(dx) > MAZE_WAYPOINT_TOL:
            direction = "RIGHT" if dx > 0 else "LEFT"
        elif abs(dy) > MAZE_WAYPOINT_TOL:
            direction = "DOWN" if dy > 0 else "UP"
        else:
            direction = "RIGHT"
        return self._swing(direction, f"maze_wp{self.maze_wp_index}")

    def _in_maze_phase(self, snap: ZeldaSnapshot, hop: ScreenHop) -> bool:
        if not is_5c_maze_hop(hop) or not self.maze_waypoints:
            return False
        return snap.screen == SCREEN_5C_MAZE

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
            self._set_phase(Level8NavPhase.FAILED, "timeout")
            return FrameAction(nes_idle_action(), "timeout")

        if snap.mode == 17:
            self._set_phase(Level8NavPhase.FAILED, "link_death")
            return FrameAction(nes_idle_action(), "link_death")

        if self._in_level8(snap):
            self.success = True
            self._set_phase(Level8NavPhase.DONE, "level8_entered")
            return FrameAction(nes_idle_action(), "done")

        if snap.transitioning:
            if self.phase is Level8NavPhase.ENTER or (
                self.hop_index < len(self.hops) and self.phase is Level8NavPhase.HOP
            ):
                direction = (
                    "UP"
                    if self.phase is Level8NavPhase.ENTER
                    else self.hops[self.hop_index].direction
                )
                return FrameAction(nes_action(direction), "scroll")
            return FrameAction(nes_idle_action(), "scroll_idle")

        if snap.mode not in (PLAY_MODE, 8, 11):
            return wake_or_wait_mode(self.phase_frames, snap.mode)

        if self.phase is Level8NavPhase.BURN:
            return self._burn_step(snap)
        if self.phase is Level8NavPhase.ENTER:
            return self._enter_step(snap)

        if self.hop_index >= len(self.hops):
            if self.burn_bush or self.enter_dungeon:
                self._set_phase(Level8NavPhase.BURN, "hops_done_burn")
                return FrameAction(nes_idle_action(), "bush_ready")
            if self._at_bush_screen(snap):
                self.success = True
                self._set_phase(Level8NavPhase.DONE, "bush_screen_reached")
                return FrameAction(nes_idle_action(), "done")
            self._set_phase(Level8NavPhase.FAILED, "hops_exhausted_off_screen")
            return FrameAction(nes_idle_action(), "fail")

        hop = self.hops[self.hop_index]
        advanced = self._advance_hop(snap, hop)
        if advanced is not None:
            return advanced

        if self._in_maze_phase(snap, hop):
            return self._follow_5c_maze(snap)

        # On 0x5B heading to 0x5C: climb to north corridor before pushing east.
        if (
            hop.target == 0x5C
            and hop.direction == "RIGHT"
            and snap.screen == 0x5B
            and snap.link_y > 100
        ):
            return self._swing("UP", "5b_north_corridor")

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

    def _burn_step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.burn_frames += 1
        if self.burn_frames > self.burn_budget:
            if self._at_bush_screen(snap):
                self.success = True
                self._set_phase(Level8NavPhase.DONE, "burn_budget_on_bush_screen")
                return FrameAction(nes_idle_action(), "done")
            self._set_phase(Level8NavPhase.FAILED, "burn_timeout")
            return FrameAction(nes_idle_action(), "burn_timeout")

        if snap.mode == 16 or snap.level == LEVEL_8:
            self._set_phase(Level8NavPhase.ENTER, "mouth_open")
            return FrameAction(nes_action("UP"), "enter_mouth")

        dx = self.bush_x - snap.link_x
        dy = self.bush_y - snap.link_y
        if abs(dx) > 6:
            return self._swing("RIGHT" if dx > 0 else "LEFT", "bush_ax")
        if abs(dy) > 6:
            return self._swing("DOWN" if dy > 0 else "UP", "bush_ay")
        if self.phase_frames % 20 < 4:
            return FrameAction(nes_action("B"), "candle_fire")
        orbit = ("UP", "RIGHT", "DOWN", "LEFT")[(self.phase_frames // 30) % 4]
        return FrameAction(nes_action(orbit, "B"), "candle_orbit")

    def _enter_step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self._in_level8(snap):
            self.success = True
            self._set_phase(Level8NavPhase.DONE, "level8_entered")
            return FrameAction(nes_idle_action(), "done")
        if self.phase_frames > 600:
            self._set_phase(Level8NavPhase.FAILED, "enter_timeout")
            return FrameAction(nes_idle_action(), "enter_timeout")
        if abs(snap.link_x - self.bush_x) > 8:
            btn = "LEFT" if snap.link_x > self.bush_x else "RIGHT"
            return self._swing(btn, "enter_ax")
        return self._swing("UP", "enter_up")

    def report(self) -> dict[str, Any]:
        hop = None
        if self.hop_index < len(self.hops):
            h = self.hops[self.hop_index]
            hop = {
                "index": self.hop_index,
                "target": h.target,
                "direction": h.direction,
            }
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "hop_index": self.hop_index,
            "hop": hop,
            "end_screen": self.end_screen(),
            "notes": list(self.notes),
        }


def has_candle(ram) -> bool:
    return read_u8(ram, ADDR_CANDLE) != 0


def level8_bush_screen_reached(ram, *, screen: int | None = None) -> bool:
    snap = read_snapshot(ram)
    target = screen if screen is not None else SCREEN_LEVEL8_BUSH
    return (
        snap.level == 0
        and snap.mode == PLAY_MODE
        and snap.screen == target
        and snap.has_sword
    )


def level8_entered(ram) -> bool:
    snap = read_snapshot(ram)
    return snap.level == LEVEL_8 and snap.mode == PLAY_MODE


__all__ = [
    "LEVEL_8",
    "TRIFORCE_BIT_L8",
    "SCREEN_LEVEL8_BUSH",
    "SCREEN_LEVEL8_BUSH_PLANNED",
    "SCREEN_LEVEL8_ENTRY_ROOM",
    "LEVEL8_BUSH_HOPS",
    "LEVEL8_BUSH_SCREENS",
    "LEVEL8_BUSH_HOPS_VIA_6B_EAST",
    "LEVEL8_BUSH_HOPS_VIA_58",
    "LEVEL8_5C_MAZE_WAYPOINTS",
    "CANDLE_SHOP_PRICE_SOURCE",
    "CANDLE_SHOP_SCREEN_LIVE",
    "SEGMENT_MAX_FRAMES",
    "OverworldToLevel8Controller",
    "Level8NavPhase",
    "has_candle",
    "level8_bush_screen_reached",
    "level8_entered",
]

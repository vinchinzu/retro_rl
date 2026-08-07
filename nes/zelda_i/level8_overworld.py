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

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.overworld import (
    LEVEL2_5C_MAZE_WAYPOINTS,
    ScreenHop,
    is_5c_maze_hop,
    path_screens_from_hops,
)
from zelda_i.ow_path import OverworldPathController
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


@dataclass
class OverworldToLevel8Controller(OverworldPathController):
    """Walk hop table from start toward L8 bush 0x6D; optional burn/enter.

    Does **not** require triforce bits. Candle acquisition is external: stop on
    bush screen if ``ADDR_CANDLE`` is 0. Burn/enter are best-effort (B-item
    must already be candle).
    """

    phase: Level8NavPhase = Level8NavPhase.HOP
    hops: tuple[ScreenHop, ...] = LEVEL8_BUSH_HOPS
    maze_waypoints: tuple[tuple[int, int], ...] = LEVEL8_5C_MAZE_WAYPOINTS
    maze_hop_pred: Any = None
    burn_bush: bool = False
    enter_dungeon: bool = False
    bush_x: int = 120
    bush_y: int = 120
    burn_frames: int = 0
    burn_budget: int = 400
    max_frames: int = SEGMENT_MAX_FRAMES
    swing_period: int = SWORD_SWING_PERIOD
    swing_hold: int = SWORD_SWING_FRAMES
    stuck_threshold: int = STUCK_THRESHOLD
    door_screen: int | None = SCREEN_LEVEL8_BUSH

    def __post_init__(self) -> None:
        if self.maze_hop_pred is None:
            self.maze_hop_pred = is_5c_maze_hop

    def reset(self) -> None:
        super().reset()
        self.burn_frames = 0

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

    def _wants_post_hop(self) -> bool:
        return self.burn_bush or self.enter_dungeon

    def _at_stop(self, snap: ZeldaSnapshot) -> bool:
        # Dungeon entry is success; hop-complete bush screen handled in after_hops.
        return self._in_level8(snap)

    def _before_play(self, snap: ZeldaSnapshot) -> FrameAction | None:
        # Transition handling matches original: only ENTER/HOP push a direction.
        if snap.transitioning:
            return None
        if self.phase is Level8NavPhase.BURN:
            return self._burn_step(snap)
        if self.phase is Level8NavPhase.ENTER:
            return self._enter_step(snap)
        return None

    def _handle_transition(self, snap: ZeldaSnapshot) -> FrameAction:
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

    def _on_hop_advanced(
        self, snap: ZeldaSnapshot, completed_hop: ScreenHop
    ) -> FrameAction:
        if self.hop_index >= len(self.hops):
            if self.burn_bush or self.enter_dungeon:
                self._set_phase(Level8NavPhase.BURN, "at_bush_screen")
                return FrameAction(nes_idle_action(), "bush_ready")
            self.success = True
            self._set_phase(Level8NavPhase.DONE, "bush_screen_reached")
            return FrameAction(nes_idle_action(), "done")
        return FrameAction(nes_idle_action(), "hop_advance")

    def _after_hops(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.burn_bush or self.enter_dungeon:
            self._set_phase(Level8NavPhase.BURN, "hops_done_burn")
            return FrameAction(nes_idle_action(), "bush_ready")
        if self._at_bush_screen(snap):
            self.success = True
            self._set_phase(Level8NavPhase.DONE, "bush_screen_reached")
            return FrameAction(nes_idle_action(), "done")
        self._set_phase(Level8NavPhase.FAILED, "hops_exhausted_off_screen")
        return FrameAction(nes_idle_action(), "fail")

    def _extra_hop_action(
        self, snap: ZeldaSnapshot, hop: ScreenHop
    ) -> FrameAction | None:
        # On 0x5B heading to 0x5C: climb to north corridor before pushing east.
        if (
            hop.target == 0x5C
            and hop.direction == "RIGHT"
            and snap.screen == 0x5B
            and snap.link_y > 100
        ):
            return self._swing("UP", "5b_north_corridor")
        return None

    def _finish(self, note: str = "path_stop") -> FrameAction:
        label = {
            "path_stop": "level8_entered",
        }.get(note, note)
        self.success = True
        self._set_phase(Level8NavPhase.DONE, label)
        return FrameAction(nes_idle_action(), "done")

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
    "is_5c_maze_hop",
]

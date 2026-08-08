"""Overworld routing: Level 8 (Lion) bush 0x6D + Blue Candle shop 0x5E.

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


Blue Candle shop (first-quest O-6) continues **east** from the same 0x5D::

    … → 0x5D E@y≈141 → **0x5E** → cave UP@x≈112 → right pedestal 60R

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
from zelda_i.anchors import (
    SCREEN_CANDLE_SHOP,
    SCREEN_LEVEL8_BUSH,
    TF_BIT_L8 as TRIFORCE_BIT_L8,
)

LEVEL_8 = 8
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

# Blue Candle shop (first-quest O-6 / screen **0x5E**). Live cave fixture
# ``CandleShop5E`` is mode-11 @ xy≈(112,213). Inventory: Magical Shield 160R
# (left x≈72), Key 100R (mid x≈120), **Blue Candle 60R (right, touch ≈152,149)**.
# OW path assisted-verified 2026-08-06 (rr-ccx). Natural 60R farm + buy residual.
# False lead: IGN “N of start then W” → 0x67 has no west corridor (G-7/0x66
# also sells candle but is a longer detour from the L8 bush corridor).
CANDLE_SHOP_PRICE_SOURCE = 60
CANDLE_SHOP_PRICE = CANDLE_SHOP_PRICE_SOURCE
CANDLE_SHOP_SCREEN_LIVE: int = SCREEN_CANDLE_SHOP
CANDLE_SHOP_STATE = "CandleShop5E"
# Cave mouth on OW 0x5E (mode 16→11). Enter UP @ x≈112 from mid-screen.
CANDLE_SHOP_CAVE_X = 112
CANDLE_SHOP_CAVE_Y = 77
# Buy contact (live): after stairs, RIGHT along y≈149 into right pedestal zone.
CANDLE_BUY_X = 152
CANDLE_BUY_Y = 149
# Pedestal object x (type 0x40); merchant type 0x78 @ (120,128).
CANDLE_SHOP_ITEM_LEFT_X = 72
CANDLE_SHOP_ITEM_MID_X = 120
CANDLE_SHOP_ITEM_RIGHT_X = 168
CANDLE_SHOP_MERCHANT_TYPE = 0x78
CANDLE_SHOP_ITEM_TYPE = 0x40

# Hop table: start → shop OW 0x5E (reuses L8 bush corridor through 0x5D maze,
# then east instead of south into 0x6D). Live assisted 2026-08-06 ~2.7k frames
# PostSwordStart → (0,141) on 0x5E. 0x5D→0x5E east open @ y≈141 (L2_ROUTE).
CANDLE_SHOP_HOPS: tuple[ScreenHop, ...] = LEVEL8_BUSH_HOPS[:-1] + (
    ScreenHop(0x5E, "RIGHT", y_band_lo=130, y_band_hi=150),
)
CANDLE_SHOP_SCREENS: tuple[int, ...] = path_screens_from_hops(
    SCREEN_START, CANDLE_SHOP_HOPS
)

# Rupee farm sketch (residual natural policy — not automated this bead):
# path screens 0x59–0x5E host Octoroks (type 0x03); farm ≥60R before buy.
# CandleShop5E fixture has 0R. Do not RAM-poke rupees for Clean / published.
RUPEE_FARM_SCREENS_SKETCH: tuple[int, ...] = (0x59, 0x5A, 0x5B, 0x5E)

# B-item cursor (Data Crystal ``$0656`` selected-item pos). Live: with candle
# owned, game/HUD use **4** for Blue/Red Candle (bomb=1). Once-per-screen flag
# ``$0513`` (0=ready, 1=used this screen) — leave+reenter or red candle resets.
ADDR_SELECTED_ITEM = 0x0656
ADDR_CANDLE_USED = 0x0513
B_ITEM_CANDLE = 0x04
CANDLE_BLUE = 1
CANDLE_RED = 2

# Walkable pocket on 0x6D (assisted recon): left corridor x≈32–56 + mid
# horizontal sand channel y≈88–96 out to x≈144. Default burn aim sits on the
# east end of that channel (“lone bush blocking pathway” source).
DEFAULT_BUSH_X = 136
DEFAULT_BUSH_Y = 93

# Back-compat names used by probe --path
LEVEL8_BUSH_HOPS_VIA_6B_EAST = LEVEL8_BUSH_HOPS
LEVEL8_BUSH_HOPS_VIA_58 = LEVEL8_BUSH_HOPS


class Level8NavPhase(Enum):
    HOP = auto()
    BURN = auto()
    ENTER = auto()
    DONE = auto()
    FAILED = auto()


class CandleShopNavPhase(Enum):
    """Phases for start → 0x5E shop cave (+ optional natural buy)."""

    HOP = auto()
    DOOR = auto()  # required name for OverworldPathController post-hop door
    BUY = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class OverworldToCandleShopController(OverworldPathController):
    """Walk hop table from start (or mid-path) to Blue Candle shop cave 0x5E.

    Assisted-verified OW hops (rr-ccx). Optional cave enter (UP @ x≈112) and
    optional natural buy when ``snap.rupees >= CANDLE_SHOP_PRICE`` (no inventory
    poke). Rupee farm is external — fixture ``CandleShop5E`` starts at 0R.
    """

    phase: CandleShopNavPhase = CandleShopNavPhase.HOP
    hops: tuple[ScreenHop, ...] = CANDLE_SHOP_HOPS
    maze_waypoints: tuple[tuple[int, int], ...] = LEVEL8_5C_MAZE_WAYPOINTS
    maze_hop_pred: Any = None
    enter_cave: bool = True
    buy_candle: bool = False
    door_x: int | None = CANDLE_SHOP_CAVE_X
    door_dir: str = "UP"
    door_screen: int | None = SCREEN_CANDLE_SHOP
    require_sword: bool = True
    max_frames: int = SEGMENT_MAX_FRAMES
    swing_period: int = SWORD_SWING_PERIOD
    swing_hold: int = SWORD_SWING_FRAMES
    stuck_threshold: int = STUCK_THRESHOLD
    buy_x: int = CANDLE_BUY_X
    buy_y: int = CANDLE_BUY_Y
    buy_frames: int = 0
    buy_budget: int = 900
    # Stairs / dialog settle at cave bottom before lateral move.
    cave_dialog_idle: int = 120
    # Runner sets each frame from ``ADDR_CANDLE`` (snapshot omits candle).
    candle_value: int = 0
    _rupees_at_buy: int | None = None

    def __post_init__(self) -> None:
        if self.maze_hop_pred is None:
            self.maze_hop_pred = is_5c_maze_hop

    def reset(self) -> None:
        super().reset()
        self.buy_frames = 0
        self.candle_value = 0
        self._rupees_at_buy = None

    def end_screen(self) -> int:
        return self.hops[-1].target if self.hops else SCREEN_CANDLE_SHOP

    def _wants_post_hop(self) -> bool:
        return self.enter_cave or self.buy_candle

    def _in_shop_cave(self, snap: ZeldaSnapshot) -> bool:
        return snap.mode == 11 and snap.level == 0 and snap.screen == SCREEN_CANDLE_SHOP

    def _owns_candle(self) -> bool:
        return int(self.candle_value) != 0

    def _at_stop(self, snap: ZeldaSnapshot) -> bool:
        if self.buy_candle:
            return self._owns_candle()
        if self.enter_cave:
            return self._in_shop_cave(snap)
        return (
            snap.level == 0
            and snap.mode == PLAY_MODE
            and snap.screen == self.end_screen()
            and self.stop_y_lo < snap.link_y < self.stop_y_hi
            and (not self.require_sword or snap.has_sword)
        )

    def _before_play(self, snap: ZeldaSnapshot) -> FrameAction | None:
        if snap.transitioning:
            if self.phase is CandleShopNavPhase.DOOR:
                return FrameAction(nes_action("UP"), "cave_transition")
            return None
        if self.phase is CandleShopNavPhase.BUY:
            return self._buy_step(snap)
        if self.phase is CandleShopNavPhase.DOOR and self._in_shop_cave(snap):
            if self.buy_candle:
                self._rupees_at_buy = snap.rupees
                self._set_phase(CandleShopNavPhase.BUY, "in_shop_cave")
                return FrameAction(nes_idle_action(), "shop_ready")
            self.success = True
            self._set_phase(CandleShopNavPhase.DONE, "shop_cave_entered")
            return FrameAction(nes_idle_action(), "done")
        return None

    def _simple_door_hunt(self, snap: ZeldaSnapshot) -> FrameAction:
        if self._in_shop_cave(snap):
            if self.buy_candle:
                self._rupees_at_buy = snap.rupees
                self._set_phase(CandleShopNavPhase.BUY, "in_shop_cave")
                return FrameAction(nes_idle_action(), "shop_ready")
            self.success = True
            self._set_phase(CandleShopNavPhase.DONE, "shop_cave_entered")
            return FrameAction(nes_idle_action(), "done")
        if self.phase_frames > 1200:
            # OW screen reached is still useful progress.
            if snap.screen == SCREEN_CANDLE_SHOP and snap.mode == PLAY_MODE:
                self.success = True
                self._set_phase(CandleShopNavPhase.DONE, "shop_ow_no_cave")
                return FrameAction(nes_idle_action(), "done")
            self._set_phase(CandleShopNavPhase.FAILED, "cave_timeout")
            return FrameAction(nes_idle_action(), "cave_timeout")
        return super()._simple_door_hunt(snap)

    def _extra_hop_action(
        self, snap: ZeldaSnapshot, hop: ScreenHop
    ) -> FrameAction | None:
        # Same 0x5B north-corridor climb as L8 bush path into 0x5C maze.
        if (
            hop.target == 0x5C
            and hop.direction == "RIGHT"
            and snap.screen == 0x5B
            and snap.link_y > 100
        ):
            return self._swing("UP", "5b_north_corridor")
        return None

    def _buy_step(self, snap: ZeldaSnapshot) -> FrameAction:
        """Natural buy: stairs → RIGHT along y≈149 → touch right pedestal."""
        self.buy_frames += 1
        if self._owns_candle():
            self.success = True
            self._set_phase(CandleShopNavPhase.DONE, "candle_bought")
            return FrameAction(nes_idle_action(), "done")
        if self.buy_frames > self.buy_budget:
            self._set_phase(CandleShopNavPhase.FAILED, "buy_timeout")
            return FrameAction(nes_idle_action(), "buy_timeout")
        if not self._in_shop_cave(snap):
            self._set_phase(CandleShopNavPhase.DOOR, "left_cave")
            return FrameAction(nes_idle_action(), "reenter")
        if self._rupees_at_buy is None:
            self._rupees_at_buy = snap.rupees
        if (
            self._rupees_at_buy < CANDLE_SHOP_PRICE
            and self.buy_frames > self.cave_dialog_idle
        ):
            # Cannot afford — path evidence stops in cave.
            self.success = True
            self._set_phase(CandleShopNavPhase.DONE, "shop_cave_need_rupees")
            return FrameAction(nes_idle_action(), "done")
        # Idle through short merchant dialog / stairs settle.
        if self.buy_frames < self.cave_dialog_idle and snap.link_y > 200:
            return FrameAction(nes_idle_action(), "shop_dialog")
        # Live path: UP until y≤150, then RIGHT until x≥buy_x, contact ~(152,149).
        if snap.link_y > self.buy_y + 1:
            return FrameAction(nes_action("UP"), "shop_up_stairs")
        if snap.link_x < self.buy_x:
            return FrameAction(nes_action("RIGHT"), "shop_right_candle")
        return FrameAction(nes_action("UP"), "shop_touch_candle")

    def report(self) -> dict[str, Any]:
        base = super().report()
        base["end_screen"] = self.end_screen()
        base["enter_cave"] = self.enter_cave
        base["buy_candle"] = self.buy_candle
        base["buy_frames"] = self.buy_frames
        base["candle_value"] = self.candle_value
        return base


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
    bush_x: int = DEFAULT_BUSH_X
    bush_y: int = DEFAULT_BUSH_Y
    burn_frames: int = 0
    burn_budget: int = 800
    # After B fire, push into facing/bush for this many frames before re-aim.
    post_fire_push: int = 40
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
                # Still on bush OW — path worked; candle/mouth residual.
                self.success = True
                self._set_phase(Level8NavPhase.DONE, "burn_budget_on_bush_screen")
                return FrameAction(nes_idle_action(), "done")
            self._set_phase(Level8NavPhase.FAILED, "burn_timeout")
            return FrameAction(nes_idle_action(), "burn_timeout")

        if snap.mode == 16 or snap.level == LEVEL_8:
            self._set_phase(Level8NavPhase.ENTER, "mouth_open")
            return FrameAction(nes_action("UP"), "enter_mouth")

        # Approach default bush aim (east channel on 0x6D).
        dx = self.bush_x - snap.link_x
        dy = self.bush_y - snap.link_y
        if abs(dx) > 6:
            return self._swing("RIGHT" if dx > 0 else "LEFT", "bush_ax")
        if abs(dy) > 6:
            return self._swing("DOWN" if dy > 0 else "UP", "bush_ay")

        # At aim: face bush-ward (prefer RIGHT into blocking bush), B-fire,
        # then push into flame for stairs.
        cycle = self.phase_frames % (12 + self.post_fire_push)
        if cycle < 4:
            return FrameAction(nes_action("RIGHT"), "bush_face")
        if cycle < 12:
            return FrameAction(nes_action("B"), "candle_fire")
        # Push into fire / potential stairs; orbit if mouth still closed.
        push_i = cycle - 12
        if push_i < self.post_fire_push // 2:
            return FrameAction(nes_action("RIGHT"), "bush_push")
        orbit = ("UP", "RIGHT", "DOWN", "LEFT")[(push_i // 8) % 4]
        return FrameAction(nes_action(orbit), "bush_orbit")

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


def candle_selected(ram) -> bool:
    """True when B-item cursor is on candle (live value ``B_ITEM_CANDLE``)."""
    return read_u8(ram, ADDR_SELECTED_ITEM) == B_ITEM_CANDLE


def poke_candle_for_recon(env, *, candle: int = CANDLE_BLUE, selected: int = B_ITEM_CANDLE) -> list[str]:
    """RECON-ONLY inventory poke: set candle + B-cursor.

    Not allowed under ``docs/ASSIST_CONTRACT.md`` for Clean or published
    assisted STATUS. Use for geometry/entrance probes when natural shop buy
    is still residual (rr-ccx). Records notes for the run report.
    """
    notes: list[str] = []
    data = env.unwrapped.data
    for name, addr, val in (
        ("candle", ADDR_CANDLE, int(candle) & 0xFF),
        ("selected_item", ADDR_SELECTED_ITEM, int(selected) & 0xFF),
        ("candle_used", ADDR_CANDLE_USED, 0),
    ):
        try:
            data.set_variable(name, {"address": addr, "type": "|u1"})
            data.set_value(name, val)
            notes.append(f"{name}={val}")
        except Exception as exc:  # noqa: BLE001 — recon best-effort
            try:
                data.memory.assign(addr, "|u1", val)
                notes.append(f"{name}_assign={val}")
            except Exception as exc2:  # noqa: BLE001
                notes.append(f"{name}_fail={exc!r}/{exc2!r}")
    notes.append("RECON_POKE_NOT_CLEAN")
    return notes


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


def candle_shop_screen_reached(ram, *, screen: int | None = None) -> bool:
    """True on OW play mode on the candle shop screen (not necessarily in cave)."""
    snap = read_snapshot(ram)
    target = screen if screen is not None else SCREEN_CANDLE_SHOP
    return (
        snap.level == 0
        and snap.mode == PLAY_MODE
        and snap.screen == target
        and snap.has_sword
    )


def candle_shop_cave_entered(ram) -> bool:
    snap = read_snapshot(ram)
    return snap.mode == 11 and snap.level == 0 and snap.screen == SCREEN_CANDLE_SHOP


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
    "CANDLE_SHOP_PRICE",
    "SCREEN_CANDLE_SHOP",
    "CANDLE_SHOP_SCREEN_LIVE",
    "CANDLE_SHOP_STATE",
    "CANDLE_SHOP_CAVE_X",
    "CANDLE_SHOP_CAVE_Y",
    "CANDLE_BUY_X",
    "CANDLE_BUY_Y",
    "CANDLE_SHOP_ITEM_LEFT_X",
    "CANDLE_SHOP_ITEM_MID_X",
    "CANDLE_SHOP_ITEM_RIGHT_X",
    "CANDLE_SHOP_HOPS",
    "CANDLE_SHOP_SCREENS",
    "RUPEE_FARM_SCREENS_SKETCH",
    "ADDR_SELECTED_ITEM",
    "ADDR_CANDLE_USED",
    "B_ITEM_CANDLE",
    "CANDLE_BLUE",
    "CANDLE_RED",
    "DEFAULT_BUSH_X",
    "DEFAULT_BUSH_Y",
    "SEGMENT_MAX_FRAMES",
    "OverworldToLevel8Controller",
    "OverworldToCandleShopController",
    "Level8NavPhase",
    "CandleShopNavPhase",
    "has_candle",
    "candle_selected",
    "poke_candle_for_recon",
    "level8_bush_screen_reached",
    "level8_entered",
    "candle_shop_screen_reached",
    "candle_shop_cave_entered",
    "is_5c_maze_hop",
]

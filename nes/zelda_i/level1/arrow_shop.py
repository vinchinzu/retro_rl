"""L1 Survival side-branch: first-quest wooden arrows at OW 0x4A.

Live K-5 cave (2026-09-01 scratch probe): Magical Shield 130 / Bombs 20 /
Arrows 80. Cave mouth mode-16 ``(176,77)``. Spawn mode-11 ``(112,213)``.
Buy: settle, UP stairs to y=165, RIGHT to x=152, UP touch ``(152,157)``
until ``ADDR_ARROWS`` 0→1. Mid bombs buy on y=149 contact — stay south.
Enemy drops are ammo, not ownership.

Dedicated ``--through level1-arrows`` only. Do not splice into
``level1_survival_tf_stages`` (that would change the L6 tape). Clean M5
skips. No rupee poke on the spine; farm Octoroks on 0x4A if short of 80R.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.overworld.common import swing_action
from zelda_i.overworld.graph import LEVEL2_PATH_HOPS, ScreenHop
from zelda_i.overworld.path import OverworldPathController
from zelda_i.ram import CAVE_MODE, PLAY_MODE, ZeldaSnapshot

__all__ = [
    "ARROW_SHOP_CAVE_X",
    "ARROW_SHOP_MAX_FRAMES",
    "ARROW_SHOP_PRICE",
    "ARROW_SHOP_SCREEN",
    "ArrowShopNavPhase",
    "OverworldToArrowShopController",
    "level1_arrows_stages",
    "level1_arrows_success",
    "make_arrow_shop_controller",
]

ARROW_SHOP_SCREEN = 0x4A
ARROW_SHOP_PRICE = 80
ARROW_SHOP_CAVE_X = 176
ARROW_SHOP_CAVE_Y = 77
ARROW_BUY_X = 152
ARROW_BUY_Y = 165
ARROW_SHOP_HOPS: tuple[ScreenHop, ...] = LEVEL2_PATH_HOPS
ARROW_SHOP_MAX_FRAMES = 50000
FARM_MAX_FRAMES = 36000
BUY_BUDGET = 900
CAVE_DIALOG_IDLE = 120
FARM_Y_LO = 120
SWORD_SWING_PERIOD = 8
SWORD_SWING_HOLD = 3
STUCK_THRESHOLD = 50


class ArrowShopNavPhase(Enum):
    HOP = auto()
    FARM = auto()
    DOOR = auto()
    BUY = auto()
    DONE = auto()
    FAILED = auto()


def make_arrow_shop_controller() -> "OverworldToArrowShopController":
    """Post-L1-TF leftover 0x37 → 0x4A cave → wooden arrows. No poke."""
    return OverworldToArrowShopController()


def level1_arrows_success(snap: ZeldaSnapshot) -> bool:
    """Stop when ADDR_ARROWS is wooden. Cave leftover is allowed."""
    return int(snap.arrows) >= 1


def level1_arrows_stages():
    """Bow detour + L1 TF + settle + 0x4A buy. Dedicated through only."""
    from zelda_i.level1.bow_pickup import level1_survival_tf_stages
    from zelda_i.level2.overworld import (
        SETTLE_MAX_FRAMES,
        PostTriforceSettleController,
    )

    return (
        *level1_survival_tf_stages(),
        ("settle_l1_tf", PostTriforceSettleController(), SETTLE_MAX_FRAMES),
        ("level1_arrows", make_arrow_shop_controller(), ARROW_SHOP_MAX_FRAMES),
    )


@dataclass
class OverworldToArrowShopController(OverworldPathController):
    """Walk L2 prefix 0x37→0x4A, farm to 80R, enter NE cave, buy arrows."""

    phase: ArrowShopNavPhase = ArrowShopNavPhase.HOP
    hops: tuple[ScreenHop, ...] = ARROW_SHOP_HOPS
    enter_cave: bool = True
    door_x: int | None = ARROW_SHOP_CAVE_X
    door_dir: str = "UP"
    door_screen: int | None = ARROW_SHOP_SCREEN
    require_sword: bool = True
    max_frames: int = ARROW_SHOP_MAX_FRAMES
    swing_period: int = SWORD_SWING_PERIOD
    swing_hold: int = SWORD_SWING_HOLD
    stuck_threshold: int = STUCK_THRESHOLD
    buy_x: int = ARROW_BUY_X
    buy_y: int = ARROW_BUY_Y
    buy_frames: int = 0
    farm_frames: int = 0
    empty_frames: int = 0
    _rupees_at_buy: int | None = None
    leftover: dict[str, Any] | None = None

    def reset(self) -> None:
        super().reset()
        self.buy_frames = 0
        self.farm_frames = 0
        self.empty_frames = 0
        self._rupees_at_buy = None
        self.leftover = None

    def end_screen(self) -> int:
        return ARROW_SHOP_SCREEN

    def _wants_post_hop(self) -> bool:
        return True

    def _in_shop_cave(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.mode == CAVE_MODE
            and snap.level == 0
            and snap.screen == ARROW_SHOP_SCREEN
        )

    def _at_stop(self, snap: ZeldaSnapshot) -> bool:
        return level1_arrows_success(snap)

    def _record(self, snap: ZeldaSnapshot) -> None:
        self.leftover = {
            "mode": int(snap.mode),
            "screen": int(snap.screen),
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "rupees": int(snap.rupees),
            "arrows": int(snap.arrows),
            "bow": int(snap.bow),
            "bombs": int(snap.bombs),
            "phase": self.phase.name,
        }

    def _before_play(self, snap: ZeldaSnapshot) -> FrameAction | None:
        self._record(snap)
        if snap.level == 1:
            return self._swing("DOWN", "exit_l1")
        if snap.transitioning:
            if self.phase is ArrowShopNavPhase.DOOR:
                return FrameAction(nes_action("UP"), "cave_transition")
            if self.phase is ArrowShopNavPhase.FARM:
                going_east = snap.screen == 0x49 or snap.next_screen == ARROW_SHOP_SCREEN
                btn = "RIGHT" if going_east else "LEFT"
                return FrameAction(nes_action(btn), "farm_scroll")
            return None
        if self.phase is ArrowShopNavPhase.BUY:
            return self._buy_step(snap)
        if self.phase is ArrowShopNavPhase.FARM:
            return self._farm_step(snap)
        if self.phase is ArrowShopNavPhase.DOOR and self._in_shop_cave(snap):
            self._rupees_at_buy = snap.rupees
            self._set_phase(ArrowShopNavPhase.BUY, "in_shop_cave")
            return FrameAction(nes_idle_action(), "shop_ready")
        return None

    def _after_hops(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.phase is ArrowShopNavPhase.HOP:
            if snap.rupees < ARROW_SHOP_PRICE:
                self._set_phase(
                    ArrowShopNavPhase.FARM,
                    f"farm_need_{ARROW_SHOP_PRICE}_have_{snap.rupees}",
                )
                return self._farm_step(snap)
            self._set_phase(ArrowShopNavPhase.DOOR, "door_hunt")
        if self.phase is ArrowShopNavPhase.FARM:
            return self._farm_step(snap)
        if self.phase is ArrowShopNavPhase.BUY:
            return self._buy_step(snap)
        return self._simple_door_hunt(snap)

    def _simple_door_hunt(self, snap: ZeldaSnapshot) -> FrameAction:
        if self._in_shop_cave(snap):
            self._rupees_at_buy = snap.rupees
            self._set_phase(ArrowShopNavPhase.BUY, "in_shop_cave")
            return FrameAction(nes_idle_action(), "shop_ready")
        if self.phase_frames > 1800:
            return self._fail(
                f"cave_timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            )
        if snap.screen != ARROW_SHOP_SCREEN:
            return super()._simple_door_hunt(snap)
        # North gap @x=112 y<90 UP-exits to 0x3A. Cave is NE (176,77).
        if snap.link_y < FARM_Y_LO and abs(snap.link_x - ARROW_SHOP_CAVE_X) > 8:
            return self._swing("DOWN", "cave_drop")
        if self.door_x is not None and abs(snap.link_x - self.door_x) > 5:
            btn = "LEFT" if snap.link_x > self.door_x else "RIGHT"
            return self._swing(btn, "door_ax")
        return self._swing(self.door_dir, "door_hunt")

    def _farm_step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.farm_frames += 1
        if snap.rupees >= ARROW_SHOP_PRICE:
            if snap.screen != ARROW_SHOP_SCREEN:
                return self._swing("RIGHT", "farm_return")
            self._set_phase(
                ArrowShopNavPhase.DOOR,
                f"farm_ok_{snap.rupees}",
            )
            return FrameAction(nes_idle_action(), "farm_done")
        if self.farm_frames > FARM_MAX_FRAMES:
            return self._fail(f"farm_timeout_rupees_{snap.rupees}")
        if snap.level != 0:
            return self._fail(f"farm_left_level_{snap.level}")
        # OW enemies do not respawn while we stay. 0x4A←0x49 restock.
        if snap.screen == 0x49:
            return self._swing("RIGHT", "farm_respawn")
        if snap.screen != ARROW_SHOP_SCREEN:
            return self._fail(f"farm_left_{snap.screen:02x}")
        if snap.link_y < FARM_Y_LO:
            return self._swing("DOWN", "farm_south")
        drops = [
            obj
            for obj in snap.objects
            if obj.slot >= 1 and obj.type_id == 0x60
        ]
        prey = drops or [
            obj
            for obj in snap.objects
            if obj.slot >= 1
            and obj.type_id not in (0, 0xFF, 0x60)
            and obj.hp > 0
            and FARM_Y_LO < obj.y < 210
            and 16 < obj.x < 240
        ]
        if not prey:
            self.empty_frames += 1
            if self.empty_frames < 90:
                return swing_action(
                    self.frames,
                    "RIGHT" if snap.link_x < 160 else "LEFT",
                    "farm_wait",
                    period=self.swing_period,
                    hold=self.swing_hold,
                )
            self.empty_frames = 0
            return self._swing("LEFT", "farm_leave")
        self.empty_frames = 0
        nearest = min(
            prey,
            key=lambda obj: abs(obj.x - snap.link_x) + abs(obj.y - snap.link_y),
        )
        dx = nearest.x - snap.link_x
        dy = nearest.y - snap.link_y
        if abs(dx) >= abs(dy) and abs(dx) > 4:
            direction = "RIGHT" if dx > 0 else "LEFT"
        elif abs(dy) > 4:
            direction = "DOWN" if dy > 0 else "UP"
        else:
            direction = "RIGHT" if dx >= 0 else "LEFT"
        if direction == "UP" and snap.link_y < FARM_Y_LO + 16:
            direction = "DOWN"
        reason = "farm_rupee" if drops else "farm_chase"
        return swing_action(
            self.frames,
            direction,
            reason,
            period=self.swing_period,
            hold=self.swing_hold,
        )

    def _buy_step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.buy_frames += 1
        if level1_arrows_success(snap):
            return self._finish("arrows_bought")
        if self.buy_frames > BUY_BUDGET:
            return self._fail(
                f"buy_timeout_{snap.link_x}_{snap.link_y}_r{snap.rupees}"
            )
        if not self._in_shop_cave(snap):
            self._set_phase(ArrowShopNavPhase.DOOR, "left_cave")
            return FrameAction(nes_idle_action(), "reenter")
        if self._rupees_at_buy is None:
            self._rupees_at_buy = snap.rupees
        if (
            self._rupees_at_buy < ARROW_SHOP_PRICE
            and self.buy_frames > CAVE_DIALOG_IDLE
        ):
            return self._fail(f"shop_need_{ARROW_SHOP_PRICE}_have_{self._rupees_at_buy}")
        if self.buy_frames < CAVE_DIALOG_IDLE and snap.link_y > 200:
            return FrameAction(nes_idle_action(), "shop_dialog")
        # y=165 lateral avoids mid bombs (20R) at ~y=149.
        if snap.link_y > self.buy_y + 1:
            return FrameAction(nes_action("UP"), "shop_up_stairs")
        if snap.link_x < self.buy_x:
            return FrameAction(nes_action("RIGHT"), "shop_right_arrows")
        return FrameAction(nes_action("UP"), "shop_touch_arrows")

    def report(self) -> dict[str, Any]:
        base = super().report()
        base.update(
            {
                "end_screen": self.end_screen(),
                "buy_frames": self.buy_frames,
                "farm_frames": self.farm_frames,
                "rupees_at_buy": self._rupees_at_buy,
                "leftover": dict(self.leftover or {}),
                "policy": (
                    "0x37 L2 prefix to 0x4A; farm Octoroks to 80R; UP cave "
                    "(176,77); stairs; RIGHT y=165 x=152; UP touch; no poke"
                ),
            }
        )
        return base

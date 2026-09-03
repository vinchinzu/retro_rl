"""L1 Survival side-branch: first-quest wooden arrows at OW 0x4A.

Live K-5 cave (2026-09-01 scratch probe): Magical Shield 130 / Bombs 20 /
Arrows 80. Cave mouth mode-16 ``(176,77)``. Spawn mode-11 ``(112,213)``.
Buy: settle, UP stairs to y=165, RIGHT to x=152, UP touch ``(152,157)``
until ``ADDR_ARROWS`` 0→1. Mid bombs buy on y=149 contact — stay south.
Enemy drops are ammo, not ownership.

Dedicated ``--through level1-arrows`` only. Do not splice into
``level1_survival_tf_stages`` (that would change the L6 tape). Clean M5
skips. No rupee poke on the spine; farm Octoroks on 0x4A if short of 80R.

The buy state machine underneath is ``zelda_i.overworld.cave_shop
.CaveShopBuyController`` (rr-ps7.2) — this module is now just the
0x4A-specific instantiation (screen/cave/pedestal geometry + the
wooden-arrows success read) plus the level-1-detour exit hook the L1 TF
splice needs before handing off to the generic controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from retro_harness.input_script import FrameAction
from zelda_i.overworld.cave_shop import CaveShopBuyController, CaveShopBuyPhase
from zelda_i.overworld.graph import LEVEL2_PATH_HOPS, ScreenHop
from zelda_i.overworld.rupee_farm import RupeeFarmController
from zelda_i.ram import ADDR_ARROWS, ZeldaSnapshot

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
# 0x4A<->0x49 restock: leave west to force overworld respawns, come back east.
ARROW_SHOP_RESTOCK_SCREEN = 0x49
ARROW_SHOP_RESTOCK_DIRECTION = "LEFT"
FARM_MAX_FRAMES = 36000
SWORD_SWING_PERIOD = 8
SWORD_SWING_HOLD = 3
STUCK_THRESHOLD = 50
# North gap @x=112 y<90 UP-exits to 0x3A. Cave is NE (176,77).
ARROW_SHOP_NORTH_GAP_Y_HI = 120

# Back-compat alias: the generic engine's phase enum, under the name this
# module (and its tests) have always used. Same members: HOP/FARM/DOOR/BUY/
# DONE/FAILED.
ArrowShopNavPhase = CaveShopBuyPhase


def _arrows_value(snap: ZeldaSnapshot) -> int:
    return int(snap.arrows)


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
class OverworldToArrowShopController(CaveShopBuyController):
    """Walk L2 prefix 0x37→0x4A, farm to 80R, enter NE cave, buy arrows.

    Thin 0x4A instantiation of the generic ``CaveShopBuyController``
    (rr-ps7.2). The only thing this subclass adds over the generic engine
    is the level-1-dungeon exit hook: the L1 TF splice can hand off control
    while Link is still inside dungeon level 1 (bow-pickup detour), and
    this controller has to walk him out (``DOWN``) before the generic
    hop/farm/door/buy machine sees an overworld snapshot.
    """

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

    shop_screen: int = ARROW_SHOP_SCREEN
    cave_x: int = ARROW_SHOP_CAVE_X
    cave_y: int = ARROW_SHOP_CAVE_Y
    buy_x: int = ARROW_BUY_X
    buy_y: int = ARROW_BUY_Y
    price: int = ARROW_SHOP_PRICE
    success_getter: Callable[[ZeldaSnapshot], int] = _arrows_value
    success_addr: int | None = ADDR_ARROWS
    success_note: str = "arrows_bought"
    north_gap_x: int | None = ARROW_SHOP_CAVE_X
    north_gap_y_hi: int = ARROW_SHOP_NORTH_GAP_Y_HI

    farm: RupeeFarmController = field(
        default_factory=lambda: RupeeFarmController(
            target_rupees=ARROW_SHOP_PRICE,
            farm_screen=ARROW_SHOP_SCREEN,
            restock_neighbor_screen=ARROW_SHOP_RESTOCK_SCREEN,
            restock_direction=ARROW_SHOP_RESTOCK_DIRECTION,
            leftover_screen=ARROW_SHOP_SCREEN,
            max_frames=FARM_MAX_FRAMES,
            swing_period=SWORD_SWING_PERIOD,
            swing_hold=SWORD_SWING_HOLD,
        )
    )

    def _before_play(self, snap: ZeldaSnapshot) -> FrameAction | None:
        if snap.level == 1:
            self._record(snap)
            return self._swing("DOWN", "exit_l1")
        return super()._before_play(snap)

    def report(self) -> dict[str, Any]:
        base = super().report()
        base["policy"] = (
            "0x37 L2 prefix to 0x4A; farm Octoroks to 80R; UP cave "
            "(176,77); stairs; RIGHT y=165 x=152; UP touch; no poke"
        )
        return base

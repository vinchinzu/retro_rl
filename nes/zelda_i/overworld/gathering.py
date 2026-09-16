"""Composer row; walk is shop_p7; bomb_shop.py is later 0x4A."""

from __future__ import annotations

from zelda_i.overworld.shop_p7 import (
    SHOP_P7_BUY_MAX_FRAMES,
    SHOP_P7_PRICE,
    SHOP_P7_SCREEN,
    SHOP_P7_WALK_MAX_FRAMES,
    make_shop_p7_buy_controller,
    make_shop_p7_walk_controller,
)
from zelda_i.overworld.sword_cave import (
    SEGMENT_MAX_FRAMES as SWORD_MAX,
    SwordCaveController,
)
from zelda_i.overworld.topup import TOPUP_MAX_FRAMES, make_shop_p7_topup_controller
from zelda_i.ram import ZeldaSnapshot


def pre_l1_bomb_shop_success(snap: ZeldaSnapshot) -> bool:
    """Bombs in inventory. Arrival on 0x6F is not the errand."""
    return snap.level == 0 and snap.has_sword and int(snap.bombs) >= 1


def pre_l1_stages() -> tuple[tuple[str, object, int], ...]:
    """Wooden sword, the hunting walk, a top-up if it arrived short, the pack.

    ``bomb_topup`` is unconditional in the list and a no-op in the run: it
    stops on its first frame when Link is already on 0x6F with the price
    (``RupeeTopUpController._at_stop``). It is a stage rather than a branch
    inside the walk because "arrived short" is a different errand with a
    different stop — the walk's stop is the shop screen, the top-up's is the
    money, and the buy's is ``ADDR_BOMBS``.
    """
    return (
        ("sword_cave", SwordCaveController(), SWORD_MAX),
        ("bomb_walk", make_shop_p7_walk_controller(), SHOP_P7_WALK_MAX_FRAMES),
        (
            "bomb_topup",
            make_shop_p7_topup_controller(
                shop_screen=SHOP_P7_SCREEN, price=SHOP_P7_PRICE
            ),
            TOPUP_MAX_FRAMES,
        ),
        ("bomb_buy", make_shop_p7_buy_controller(), SHOP_P7_BUY_MAX_FRAMES),
    )


__all__ = [
    "SHOP_P7_BUY_MAX_FRAMES",
    "SHOP_P7_PRICE",
    "SHOP_P7_SCREEN",
    "SHOP_P7_WALK_MAX_FRAMES",
    "TOPUP_MAX_FRAMES",
    "SWORD_MAX",
    "SwordCaveController",
    "make_shop_p7_buy_controller",
    "make_shop_p7_topup_controller",
    "make_shop_p7_walk_controller",
    "pre_l1_bomb_shop_success",
    "pre_l1_stages",
]

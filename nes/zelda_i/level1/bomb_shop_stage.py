"""L1 Survival side-branch: first-quest bombs at OW 0x4A mid pedestal.

Same cave as ``level1.arrow_shop`` (K-5, 0x4A: Magical Shield 130R / Bombs
20R mid pedestal y=149 / Arrows 80R south pedestal y=165). The buy engine
itself (``CaveShopBuyController`` instantiation, hops, pedestal geometry)
lives in ``zelda_i.overworld.bomb_shop`` — it needs no L1-dungeon exit hook,
unlike ``level1.arrow_shop``'s ``OverworldToArrowShopController``. This
module only supplies the L1-TF-stages composition (``level1_bombs_stages``)
so ``spine/survival.py`` can wire a dedicated ``--through level1-bombs``
stop the same way it wires ``level1-arrows`` — that composition needs
``level1.bow_pickup``/``level2.overworld`` imports, which is why it can't
live under ``overworld/`` (overworld/ never imports level*/).

Dedicated ``--through level1-bombs`` only. Do not splice into
``level1_survival_tf_stages`` (that would change the L6 tape). Clean M5
skips. No rupee poke on the spine; farm Octoroks on 0x4A if short of 20R.
"""

from __future__ import annotations

from zelda_i.overworld.bomb_shop import (
    BOMB_SHOP_MAX_FRAMES,
    bomb_shop_success,
    make_bomb_shop_controller,
)
from zelda_i.ram import ZeldaSnapshot

__all__ = [
    "level1_bombs_stages",
    "level1_bombs_success",
]


def level1_bombs_success(snap: ZeldaSnapshot) -> bool:
    """Stop when ADDR_BOMBS is >=1. Cave leftover is allowed."""
    return bomb_shop_success(snap)


def level1_bombs_stages():
    """Bow detour + L1 TF + settle + 0x4A mid-pedestal buy. Dedicated only."""
    from zelda_i.level1.bow_pickup import level1_survival_tf_stages
    from zelda_i.overworld.settle import PostTriforceSettleController
    from zelda_i.overworld.settle import SETTLE_MAX_FRAMES

    return (
        *level1_survival_tf_stages(),
        ("settle_l1_tf", PostTriforceSettleController(), SETTLE_MAX_FRAMES),
        ("level1_bombs", make_bomb_shop_controller(), BOMB_SHOP_MAX_FRAMES),
    )

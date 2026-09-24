"""OW bomb packs (20R): the 0x4A cave, and between-dungeon restocks.

``BombRestockController`` buys only when the carried count is short of the
next dungeon's ``want`` (0x4A before L2; 0x44 on the L4 and L8 walks), which
is what retired the Survival bomb top-ups through L8 (rr-doua).

Same K-5 cave as ``level1.arrow_shop`` (rr-doua.1, live 2026-09-01 probe:
Magical Shield 130R / Bombs 20R mid pedestal y≈149 / Arrows 80R south
pedestal y=165). ``OverworldToArrowShopController`` deliberately walks PAST
y=149 ("stay south") on its way to the arrows pedestal to avoid buying
bombs by accident — this module targets that same mid pedestal on purpose.

This is a thin factory around ``zelda_i.overworld.cave_shop
.CaveShopBuyController`` (rr-ps7.2): same 0x37->0x4A hops
(``LEVEL2_PATH_HOPS``), same cave entry ``(176,77)`` UP, same ``x=152``
touch corridor as the arrow shop, but ``buy_y=149`` (mid pedestal, not
south) and ``price=20`` (not 80). Unlike ``level1.arrow_shop``, this
controller has no L1-dungeon-detour exit hook to add, so it lives directly
under ``overworld/`` next to the generic engine — mirroring how
``level8.overworld.make_candle_shop_buy_controller`` wires the same engine
for the 0x5E cave. Never writes ``ADDR_BOMBS``/``ADDR_MAX_BOMBS``; success
is read via ``snap.bombs`` only, cost is read via ``snap.rupees`` only, and
a rupee shortfall is closed by farming (never a poke) via
``RupeeFarmController``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from retro_harness.input_script import FrameAction
from zelda_i.overworld.cave_shop import CaveShopBuyController
from zelda_i.overworld.graph import LEVEL2_PATH_HOPS, ScreenHop
from zelda_i.overworld.locations import restock_for
from zelda_i.overworld.rupee_farm import RupeeFarmController
from zelda_i.ram import ADDR_BOMBS, ZeldaSnapshot

__all__ = [
    "BOMB_BUY_X",
    "BOMB_BUY_Y",
    "BOMB_SHOP_CAVE_X",
    "BOMB_SHOP_CAVE_Y",
    "BOMB_SHOP_HOPS",
    "BOMB_SHOP_MAX_FRAMES",
    "BOMB_SHOP_PRICE",
    "BOMB_SHOP_SCREEN",
    "BombRestockController",
    "SHOP_E5_CAVE_X",
    "SHOP_E5_SCREEN",
    "bomb_restock_stages",
    "bomb_shop_restock",
    "bomb_shop_success",
    "make_bomb_restock_controller",
    "make_bomb_shop_controller",
]

BOMB_SHOP_SCREEN = 0x4A
BOMB_SHOP_PRICE = 20
BOMB_SHOP_CAVE_X = 176
BOMB_SHOP_CAVE_Y = 77
BOMB_BUY_X = 152
BOMB_BUY_Y = 149
BOMB_SHOP_HOPS: tuple[ScreenHop, ...] = LEVEL2_PATH_HOPS
BOMB_SHOP_MAX_FRAMES = 50000
FARM_MAX_FRAMES = 36000
SWORD_SWING_PERIOD = 8
SWORD_SWING_HOLD = 3
STUCK_THRESHOLD = 50
# North gap @x=112 y<90 UP-exits to 0x3A. Cave is NE (176,77).
BOMB_SHOP_NORTH_GAP_Y_HI = 105
# 0x44 (E5) is the same 0x1D cave (shield 130 / bombs 20 / arrows 80), so the
# pedestal touch is 0x4A's. Mouth measured off $6530 on the gather's ring
# walk (2026-09-24): the 0x24 pair at x=64-79, y=88, above a 16 px column
# between two bush pairs; the lake fills x>=144 and the 0x34 gap is x=128.
SHOP_E5_SCREEN = 0x44
SHOP_E5_CAVE_X = 64
SHOP_E5_CAVE_Y = 77
# Lattice row under the mouth. Link comes up the south gap (x 112-143),
# walled in to the west, so the walk to the mouth is ``mouth_step``'s.
SHOP_E5_APPROACH_Y = 93
SHOP_E5_MAX_FRAMES = 12000


def _bombs_value(snap: ZeldaSnapshot) -> int:
    return int(snap.bombs)


def bomb_shop_success(snap: ZeldaSnapshot) -> bool:
    """Stop when ``ADDR_BOMBS`` shows at least one bomb. Read-only."""
    return int(snap.bombs) >= 1


def bomb_shop_restock() -> tuple[int, str]:
    """0x4A<->0x49 restock pair: leave west to respawn, come back east.

    Read from the farm catalog rather than hand-written, and checked here
    rather than at module scope: this module sits on an unconditional import
    chain (``spine.survival``), so a catalog gap must fail the one caller
    that needs the pair, not every import of the package.
    """
    pair = restock_for(BOMB_SHOP_SCREEN)
    if pair is None:
        raise RuntimeError("0x4A farm catalog is missing a restock pair")
    neighbor, direction = pair
    return int(neighbor), str(direction)


def make_bomb_shop_controller(
    *,
    hops: tuple[ScreenHop, ...] = BOMB_SHOP_HOPS,
    restock_farm: bool = True,
    want: int | None = None,
) -> CaveShopBuyController:
    """0x37 (or leftover) -> 0x4A cave -> bombs mid pedestal (20R). No poke.

    With ``want`` it is a restock (``BombRestockController``): it ends on its
    first frame when the carried bombs already cover ``want``.

    ``restock_farm=False`` drops the 0x4A<->0x49 ``RupeeFarmController``, so a
    shortfall fails closed with ``shop_need_20_have_N`` on the spot instead of
    spending ``FARM_MAX_FRAMES`` in it. Overworld waves are one-shot at depth
    1-2 (AGENTS.md), so that loop is a give-up detector, not a rupee supply —
    on a leg where the walk is the farm, a stalled restock only hides how
    short the walk came.
    """
    farm: RupeeFarmController | None = None
    if restock_farm:
        restock_screen, restock_direction = bomb_shop_restock()
        farm = RupeeFarmController(
            target_rupees=BOMB_SHOP_PRICE,
            farm_screen=BOMB_SHOP_SCREEN,
            restock_neighbor_screen=restock_screen,
            restock_direction=restock_direction,
            leftover_screen=BOMB_SHOP_SCREEN,
            max_frames=FARM_MAX_FRAMES,
            swing_period=SWORD_SWING_PERIOD,
            swing_hold=SWORD_SWING_HOLD,
        )
    kind: type[CaveShopBuyController] = CaveShopBuyController
    extra: dict[str, Any] = {"min_headroom": 4}
    if want is not None:
        kind, extra = BombRestockController, {"want": int(want), "min_headroom": 1}
    return kind(
        hops=hops,
        enter_cave=True,
        door_x=BOMB_SHOP_CAVE_X,
        door_dir="UP",
        door_screen=BOMB_SHOP_SCREEN,
        farm_below_hearts=0,
        max_farm_attempts=0,
        require_sword=True,
        max_frames=BOMB_SHOP_MAX_FRAMES,
        swing_period=SWORD_SWING_PERIOD,
        swing_hold=SWORD_SWING_HOLD,
        stuck_threshold=STUCK_THRESHOLD,
        shop_screen=BOMB_SHOP_SCREEN,
        cave_x=BOMB_SHOP_CAVE_X,
        cave_y=BOMB_SHOP_CAVE_Y,
        buy_x=BOMB_BUY_X,
        buy_y=BOMB_BUY_Y,
        price=BOMB_SHOP_PRICE,
        success_getter=_bombs_value,
        min_item_gain=1,
        success_addr=ADDR_BOMBS,
        success_note="bombs_bought",
        north_gap_x=BOMB_SHOP_CAVE_X,
        north_gap_y_hi=BOMB_SHOP_NORTH_GAP_Y_HI,
        farm=farm,
        **extra,
    )


@dataclass
class BombRestockController(CaveShopBuyController):
    """Buy the 20R pack between dungeons only when bombs are short of ``want``.

    Mirrors the potion restock: on its first frame it ends at once, walking
    nowhere, when the carried count already covers ``want``. Short of it, a
    pack that tops out the bag is still a buy (``min_headroom=1``).
    """

    want: int = 0

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.frames == 0 and int(snap.bombs) >= self.want:
            self.frames += 1
            return self._finish("bomb_restock_enough")
        return super().step(snap)


def make_bomb_restock_controller(
    *, hops: tuple[ScreenHop, ...], want: int
) -> BombRestockController:
    """0x44 bombs (20R) on a walk that ends on 0x44. Fails closed when short."""
    return BombRestockController(
        hops=tuple(hops),
        resume_on_screen=True,
        want=int(want),
        enter_cave=True,
        door_x=SHOP_E5_CAVE_X,
        door_dir="UP",
        door_screen=SHOP_E5_SCREEN,
        farm_below_hearts=0,
        max_farm_attempts=0,
        require_sword=True,
        max_frames=SHOP_E5_MAX_FRAMES,
        swing_period=SWORD_SWING_PERIOD,
        swing_hold=SWORD_SWING_HOLD,
        stuck_threshold=STUCK_THRESHOLD,
        shop_screen=SHOP_E5_SCREEN,
        cave_x=SHOP_E5_CAVE_X,
        cave_y=SHOP_E5_CAVE_Y,
        mouth_approach_y=SHOP_E5_APPROACH_Y,
        buy_x=BOMB_BUY_X,
        buy_y=BOMB_BUY_Y,
        price=BOMB_SHOP_PRICE,
        success_getter=_bombs_value,
        min_item_gain=1,
        min_headroom=1,
        success_addr=ADDR_BOMBS,
        success_note="bombs_bought",
    )


def bomb_restock_stages(
    hops: tuple[ScreenHop, ...], tag: str, *, want: int
) -> tuple[tuple[str, Any, int], ...]:
    """Spine stages for a 0x44 bomb buy on a walk that crosses it: buy, exit.

    ``hops`` is the whole walk; the buy walks it as far as 0x44. The buy
    resumes from the screen Link is on (0x64 after a potion restock there),
    and the walk after these stages should too: Link is on 0x44 after a
    buy and still where he started after a skip.
    """
    from zelda_i.overworld.gather_segments import CaveExitController

    to_shop = hops[: [hop.target for hop in hops].index(SHOP_E5_SCREEN) + 1]
    buy = make_bomb_restock_controller(hops=to_shop, want=want)
    return (
        (f"bomb_restock_{tag}", buy, buy.max_frames),
        (f"exit_bomb_restock_{tag}", CaveExitController(clear=0), 600),
    )

"""OW 0x4A cave mid pedestal: buy bombs (20R) via CaveShopBuyController.

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
    "bomb_shop_restock",
    "bomb_shop_success",
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
BOMB_SHOP_NORTH_GAP_Y_HI = 120


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
) -> CaveShopBuyController:
    """0x37 (or leftover) -> 0x4A cave -> bombs mid pedestal (20R). No poke.

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
    return CaveShopBuyController(
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
        success_addr=ADDR_BOMBS,
        success_note="bombs_bought",
        north_gap_x=BOMB_SHOP_CAVE_X,
        north_gap_y_hi=BOMB_SHOP_NORTH_GAP_Y_HI,
        farm=farm,
    )

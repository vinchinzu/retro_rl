"""OW wooden arrows (80R): the 0x4A and 0x44 caves, and between-dungeon restocks.

``ArrowRestockController`` buys wooden arrows only when the carried count is
short of ``want`` (default 1, e.g. at 0x4A before L2, or at 0x44 between
dungeons), retiring the Survival wooden-arrow poke at Level 6 Gohma (rr-ps7.7).

Same K-5 / E-5 caves (cave type 0x1D: Magical Shield 130R / Bombs 20R mid
pedestal y≈149 / Arrows 80R south pedestal y=165). This module targets the south
pedestal at ``buy_y=165``, ``buy_x=152`` with ``price=80``.

Mirrors ``zelda_i.overworld.bomb_shop``: thin factory around
``zelda_i.overworld.cave_shop.CaveShopBuyController`` (rr-ps7.2). Never writes
``ADDR_ARROWS``; success is read via ``snap.arrows`` only, cost is read via
``snap.rupees`` only, and a rupee shortfall either fails closed (between dungeons)
or is closed by farming via ``RupeeFarmController``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action
from zelda_i.overworld.cave_shop import CaveShopBuyController, NORTH_GAP_Y_HI
from zelda_i.overworld.graph import LEVEL2_PATH_HOPS, ScreenHop
from zelda_i.overworld.locations import restock_for
from zelda_i.overworld.rupee_farm import RupeeFarmController
from zelda_i.ram import ADDR_ARROWS, ZeldaSnapshot

__all__ = [
    "ARROW_BUY_X",
    "ARROW_BUY_Y",
    "ARROW_SHOP_CAVE_X",
    "ARROW_SHOP_CAVE_Y",
    "ARROW_SHOP_HOPS",
    "ARROW_SHOP_MAX_FRAMES",
    "ARROW_SHOP_PRICE",
    "ARROW_SHOP_SCREEN",
    "ArrowRestockController",
    "SHOP_E5_APPROACH_Y",
    "SHOP_E5_CAVE_X",
    "SHOP_E5_CAVE_Y",
    "SHOP_E5_MAX_FRAMES",
    "SHOP_E5_SCREEN",
    "arrow_restock_stages",
    "arrow_shop_restock",
    "arrow_shop_success",
    "make_arrow_restock_controller",
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
SWORD_SWING_PERIOD = 8
SWORD_SWING_HOLD = 3
STUCK_THRESHOLD = 50
ARROW_SHOP_NORTH_GAP_Y_HI = 120

SHOP_E5_SCREEN = 0x44
SHOP_E5_CAVE_X = 64
SHOP_E5_CAVE_Y = 77
SHOP_E5_APPROACH_Y = 93
SHOP_E5_MAX_FRAMES = 12000


def _arrows_value(snap: ZeldaSnapshot) -> int:
    return int(snap.arrows)


def arrow_shop_success(snap: ZeldaSnapshot) -> bool:
    """Stop when ``ADDR_ARROWS`` shows at least wooden arrows (>= 1). Read-only."""
    return int(snap.arrows) >= 1


def arrow_shop_restock() -> tuple[int, str]:
    """0x4A<->0x49 restock pair: leave west to respawn, come back east.

    Read from the farm catalog rather than hand-written, and checked here
    rather than at module scope: this module sits on an unconditional import
    chain, so a catalog gap must fail the one caller that needs the pair,
    not every import of the package.
    """
    pair = restock_for(ARROW_SHOP_SCREEN)
    if pair is None:
        raise RuntimeError("0x4A farm catalog is missing a restock pair")
    neighbor, direction = pair
    return int(neighbor), str(direction)


def make_arrow_shop_controller(
    *,
    hops: tuple[ScreenHop, ...] = ARROW_SHOP_HOPS,
    restock_farm: bool = True,
    want: int | None = None,
) -> CaveShopBuyController:
    """0x37 (or leftover) -> 0x4A cave -> wooden arrows south pedestal (80R). No poke.

    With ``want`` it is a restock (``ArrowRestockController``): it ends on its
    first frame when the carried arrows already cover ``want``.

    ``restock_farm=False`` drops the 0x4A<->0x49 ``RupeeFarmController``, so a
    shortfall fails closed with ``shop_need_80_have_N`` on the spot instead of
    spending ``FARM_MAX_FRAMES`` in it. Overworld waves are one-shot at depth
    1-2 (AGENTS.md), so that loop is a give-up detector, not a rupee supply —
    on a leg where the walk is the farm, a stalled restock only hides how
    short the walk came.
    """
    farm: RupeeFarmController | None = None
    if restock_farm:
        restock_screen, restock_direction = arrow_shop_restock()
        farm = RupeeFarmController(
            target_rupees=ARROW_SHOP_PRICE,
            farm_screen=ARROW_SHOP_SCREEN,
            restock_neighbor_screen=restock_screen,
            restock_direction=restock_direction,
            leftover_screen=ARROW_SHOP_SCREEN,
            max_frames=FARM_MAX_FRAMES,
            swing_period=SWORD_SWING_PERIOD,
            swing_hold=SWORD_SWING_HOLD,
        )
    kind: type[CaveShopBuyController] = CaveShopBuyController
    extra: dict[str, Any] = {}
    if want is not None:
        kind, extra = ArrowRestockController, {"want": int(want)}
    return kind(
        hops=hops,
        enter_cave=True,
        door_x=ARROW_SHOP_CAVE_X,
        door_dir="UP",
        door_screen=ARROW_SHOP_SCREEN,
        farm_below_hearts=0,
        max_farm_attempts=0,
        require_sword=True,
        max_frames=ARROW_SHOP_MAX_FRAMES,
        swing_period=SWORD_SWING_PERIOD,
        swing_hold=SWORD_SWING_HOLD,
        stuck_threshold=STUCK_THRESHOLD,
        shop_screen=ARROW_SHOP_SCREEN,
        cave_x=ARROW_SHOP_CAVE_X,
        cave_y=ARROW_SHOP_CAVE_Y,
        buy_x=ARROW_BUY_X,
        buy_y=ARROW_BUY_Y,
        price=ARROW_SHOP_PRICE,
        success_getter=_arrows_value,
        success_addr=ADDR_ARROWS,
        success_note="arrows_bought",
        north_gap_x=ARROW_SHOP_CAVE_X,
        north_gap_y_hi=ARROW_SHOP_NORTH_GAP_Y_HI,
        farm=farm,
        **extra,
    )


@dataclass
class ArrowRestockController(CaveShopBuyController):
    """Buy the 80R wooden arrows between dungeons only when arrows are short of ``want``.

    Mirrors ``BombRestockController``: on its first frame it ends at once,
    walking nowhere, when the carried arrows already cover ``want`` (default 1).
    When short (arrows == 0), it buys wooden arrows at the 80R pedestal.
    """

    want: int = 1

    def _extra_hop_action(
        self, snap: ZeldaSnapshot, hop: ScreenHop
    ) -> FrameAction | None:
        # The post-L4 0x55 raft reverses at y=128. Reach the land at y=141
        # before threat facing or the next east hop can take control.
        if (snap.level == 0 and snap.screen == 0x55 and hop.target == 0x56
                and 100 <= snap.link_x <= 152 and snap.link_y < 141):
            return FrameAction(nes_action("DOWN"), "raft_dismount")
        return None

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.frames == 0 and int(snap.arrows) >= self.want:
            self.frames += 1
            return self._finish("arrow_restock_enough")
        return super().step(snap)


def make_arrow_restock_controller(
    *,
    hops: tuple[ScreenHop, ...],
    want: int = 1,
    screen: int | None = None,
) -> ArrowRestockController:
    """Buy 80R wooden arrows on a walk that crosses/ends on ``screen`` (0x44 or 0x4A).

    Fails closed when short of 80R. Defaults to 0x44 (SHOP_E5_SCREEN), but also
    supports 0x4A (ARROW_SHOP_SCREEN) or infers from ``hops[-1].target``.
    """
    if screen is None:
        screen = (
            hops[-1].target
            if hops and hops[-1].target in (ARROW_SHOP_SCREEN, SHOP_E5_SCREEN)
            else SHOP_E5_SCREEN
        )

    if screen == SHOP_E5_SCREEN:
        cave_x = SHOP_E5_CAVE_X
        cave_y = SHOP_E5_CAVE_Y
        door_x = SHOP_E5_CAVE_X
        mouth_approach_y: int | None = SHOP_E5_APPROACH_Y
        north_gap_x: int | None = None
        north_gap_y_hi: int = NORTH_GAP_Y_HI
        max_frames = SHOP_E5_MAX_FRAMES
    elif screen == ARROW_SHOP_SCREEN:
        cave_x = ARROW_SHOP_CAVE_X
        cave_y = ARROW_SHOP_CAVE_Y
        door_x = ARROW_SHOP_CAVE_X
        mouth_approach_y = None
        north_gap_x = ARROW_SHOP_CAVE_X
        north_gap_y_hi = ARROW_SHOP_NORTH_GAP_Y_HI
        max_frames = ARROW_SHOP_MAX_FRAMES
    else:
        raise ValueError(f"unsupported arrow shop screen: 0x{screen:02X}")

    return ArrowRestockController(
        hops=tuple(hops),
        resume_on_screen=True,
        want=int(want),
        enter_cave=True,
        door_x=door_x,
        door_dir="UP",
        door_screen=screen,
        farm_below_hearts=0,
        max_farm_attempts=0,
        require_sword=True,
        max_frames=max_frames,
        swing_period=SWORD_SWING_PERIOD,
        swing_hold=SWORD_SWING_HOLD,
        stuck_threshold=STUCK_THRESHOLD,
        shop_screen=screen,
        cave_x=cave_x,
        cave_y=cave_y,
        mouth_approach_y=mouth_approach_y,
        north_gap_x=north_gap_x,
        north_gap_y_hi=north_gap_y_hi,
        buy_x=ARROW_BUY_X,
        buy_y=ARROW_BUY_Y,
        price=ARROW_SHOP_PRICE,
        success_getter=_arrows_value,
        success_addr=ADDR_ARROWS,
        success_note="arrows_bought",
    )


def arrow_restock_stages(
    hops: tuple[ScreenHop, ...],
    tag: str,
    *,
    want: int = 1,
    screen: int = SHOP_E5_SCREEN,
) -> tuple[tuple[str, Any, int], ...]:
    """Spine stages for an arrow buy on a walk that crosses it: buy, exit.

    ``hops`` is the whole walk; the buy walks it as far as ``screen``. The buy
    resumes from the screen Link is on, and the walk after these stages
    should too: Link is on the shop screen after a buy and still where he
    started after a skip.
    """
    from zelda_i.overworld.gather_segments import CaveExitController

    targets = [hop.target for hop in hops]
    if screen in targets:
        to_shop = hops[: targets.index(screen) + 1]
    else:
        to_shop = hops
    buy = make_arrow_restock_controller(hops=to_shop, want=want, screen=screen)
    return (
        (f"arrow_restock_{tag}", buy, buy.max_frames),
        (f"exit_arrow_restock_{tag}", CaveExitController(clear=0), 600),
    )

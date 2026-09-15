"""Pre-L1 bomb shop walk: sword leftover on 0x77 -> play leftover on 0x4A.

CLEANUP_PLAN 4.5.1 first half (rr-ps7.4). The original plan attempted to
reach 0x6F via row 6, but physical measurements revealed:
- 0x68 east is a bush wall (DEAD_68_EAST_Y141).
- 0x6C east is a walled west pocket with vertical bush column (DEAD_6C_EAST_BUSH).
- 0x6D has no west entrance, and 0x5E east is a tree wall.

However, 0x4A belongs to the identical ``CAVE_SHOP_ARROWS`` shop family in ROM
(AttrsB >> 2 == 0x1D), selling Bombs 4-pack for 20R on the mid pedestal.
Screen 0x4A is reached via the proven live corridor:

    0x77 --RIGHT align_y=140--> 0x78     [nav.START_EAST_Y]
    0x78 --UP    align_x=48 --> 0x68     [nav.COL_NORTH_X]
    0x68 --UP    align_x=48 --> 0x58     [nav.COL_NORTH_X]
    0x58 --RIGHT y148-162  --> 0x59
    0x59 --UP    align_x=112--> 0x49
    0x49 --RIGHT align_y=141--> 0x4A

Traps this table bypasses:
- 0x79 ``rocky_deadend_east_of_78`` (naive "right 8 from start")
- 0x77 north onto 0x67 (dead-end from start)
- 0x68 door-repair cave (``OPEN_SECRET``) — walk across, do not enter
- 0x6C west pocket bush wall
- Lost Hills 0x1B — later; not this hop
"""

from __future__ import annotations

from dataclasses import dataclass

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_idle_action
from zelda_i.combat import (
    BOMB_DROP_OBJECT_TYPE,
    BOMB_DROP_STATES,
    HEART_OR_FAIRY_STATES,
    RUPEE_DROP_STATES,
)
from zelda_i.dungeon.ids import HEART_DROP_OBJECT_TYPE, RUPEE_DROP_OBJECT_TYPE
from zelda_i.overworld.bomb_shop import (
    BOMB_SHOP_MAX_FRAMES as BOMB_BUY_MAX_FRAMES,
    make_bomb_shop_controller,
)
from zelda_i.overworld.cave_shop import CaveShopBuyController
from zelda_i.overworld.common import scoop_floor_drop
from zelda_i.overworld.heart_farm import owns_bombs
from zelda_i.overworld.graph import SCREEN_START, ScreenHop, path_screens_from_hops
from zelda_i.overworld.path import OverworldPathController
from zelda_i.overworld.sword_cave import (
    SEGMENT_MAX_FRAMES as SWORD_MAX,
    SwordCaveController,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

SOURCE_HYPOTHESIS = True  # live through 0x68/0x58/0x59; 0x68 west-column east is dead.
DEAD_68_EAST_Y141 = True  # t1b: x=48 y=141 is a bush, not a corridor.
DEAD_6C_EAST_BUSH = True  # row 6 blocked at 0x6C west pocket by vertical bush column.
DEAD_ROW6_EAST_TO_6F = True  # row 6 to 0x6F is unroutable; 0x4A is identical shop family

BOMB_SHOP_SCREEN = 0x4A  # CAVE_SHOP_ARROWS family (AttrsB >> 2 == 0x1D), bombs 20R mid pedestal
BOMB_SHOP_PRICE = 20
BOMB_SHOP_WALK_MAX_FRAMES = 30000

# Backward-compatibility aliases
SHOP_P7_SCREEN = BOMB_SHOP_SCREEN
SHOP_P7_PRICE = BOMB_SHOP_PRICE
SHOP_P7_WALK_MAX_FRAMES = BOMB_SHOP_WALK_MAX_FRAMES

# Proven live route to 0x4A: 0x77 -> 0x78 -> 0x68 -> 0x58 -> 0x59 -> 0x49 -> 0x4A
#
# **0x48 is not on it, and its four leevers are not worth taking.** On paper
# they are the second-richest screen in reach — drop-table row 1, 3.6R against
# the whole five-screen octorok corridor's 2.9R — and a there-and-back detour
# off 0x58 uses only hops ``LEVEL2_PATH_HOPS`` already proves. Live it killed
# the run twice (2026-09-15 ``fixG``/``fixH``, both mode 17 with three
# containers): Link scrolls onto 0x48 at y≈205, *below* ``HUNT_BOX``, so the
# hunt cannot fight there at all, and he arrives with one heart left because
# 0x58 comes first. Three containers is not enough health to buy a fifth
# screen. Revisit it after the 0x7B / 0x2C heart containers, not before.
PRE_L1_BOMB_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x78, "RIGHT", align_y=140),
    ScreenHop(0x68, "UP", align_x=48),
    ScreenHop(0x58, "UP", align_x=48),
    ScreenHop(0x59, "RIGHT", y_band_lo=148, y_band_hi=162),
    ScreenHop(0x49, "UP", align_x=112),
    ScreenHop(0x4A, "RIGHT", align_y=141),
)

# **The lap is the rupee supply, and it has to be a lap.** One pass of this
# corridor pays ~12R of random drops against a 20R pack, so the money has to
# come from a second wave. ``overworld.respawn`` has the ROM rule: a screen's
# kill flags are cleared — the wave comes back whole — only when the screen is
# absent from the six-entry ``RoomHistory``, and a room is appended to that
# history **only when it is not already in it**. An out-and-back therefore
# evicts nothing (every screen on the way back is already in the history),
# which is why the ``0x4A <-> 0x49`` restock in ``rupee_farm`` never was a
# farm. Eviction needs new rooms, and this corridor has exactly seven distinct
# screens against six slots: walking back onto 0x77 evicts 0x78, and from
# there each screen Link enters evicts the next one in front of him.
#
# Each row mirrors the ``PRE_L1_BOMB_HOPS`` row that made the crossing
# eastward, on the same alignment — the corridor is one lane wide and the hop
# engine aligns before it pushes. Live to 0x59 (2026-09-15 ``lap1``); the lap
# itself is gated on health, not on the hops (see ``docs/PRE_L1.md``).
PRE_L1_LAP_WEST_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x49, "LEFT", align_y=141),
    ScreenHop(0x59, "DOWN", align_x=112),
    ScreenHop(0x58, "LEFT", y_band_lo=148, y_band_hi=162),
    ScreenHop(0x68, "DOWN", align_x=48),
    ScreenHop(0x78, "DOWN", align_x=48),
    ScreenHop(0x77, "LEFT", align_y=140),
)
PRE_L1_LAP_HOPS: tuple[ScreenHop, ...] = PRE_L1_LAP_WEST_HOPS + PRE_L1_BOMB_HOPS


def pre_l1_walk_hops(laps: int = 0) -> tuple[ScreenHop, ...]:
    """The walk to 0x4A, plus ``laps`` full laps of the corridor behind it."""
    return PRE_L1_BOMB_HOPS + PRE_L1_LAP_HOPS * max(int(laps), 0)


BOMB_SHOP_HOPS: tuple[ScreenHop, ...] = PRE_L1_BOMB_HOPS
SHOP_P7_HOPS: tuple[ScreenHop, ...] = BOMB_SHOP_HOPS
SHOP_P7_HOPS_LIVE_PREFIX: tuple[ScreenHop, ...] = BOMB_SHOP_HOPS[:5]


def bomb_shop_screens() -> tuple[int, ...]:
    return path_screens_from_hops(SCREEN_START, BOMB_SHOP_HOPS)


shop_p7_screens = bomb_shop_screens

_BOMB_SHOP_SCREENS = bomb_shop_screens()
_SHOP_P7_SCREENS = _BOMB_SHOP_SCREENS
assert _BOMB_SHOP_SCREENS[0] == SCREEN_START == 0x77
assert _BOMB_SHOP_SCREENS[-1] == BOMB_SHOP_SCREEN == 0x4A
assert 0x79 not in _BOMB_SHOP_SCREENS
assert 0x67 not in _BOMB_SHOP_SCREENS
assert 0x1B not in _BOMB_SHOP_SCREENS
assert 0x6C not in _BOMB_SHOP_SCREENS


def bomb_shop_arrived(snap: ZeldaSnapshot) -> bool:
    """Play leftover on 0x4A with the sword. Arrival, not the errand."""
    return (
        snap.level == 0
        and snap.mode == PLAY_MODE
        and snap.screen == BOMB_SHOP_SCREEN
        and snap.has_sword
    )


shop_p7_arrived = bomb_shop_arrived


def pre_l1_bomb_shop_success(snap: ZeldaSnapshot) -> bool:
    """Bombs in inventory, read from ``ADDR_BOMBS``. Never a poke.

    Arrival on 0x4A used to be the stop. It is not the errand: the errand is
    the 4-pack, and a stop that greens on arrival cannot tell a walk that
    banked 20R from one that banked one rupee.
    """
    return snap.level == 0 and snap.has_sword and int(snap.bombs) >= 1


@dataclass
class ShopP7WalkController(OverworldPathController):
    """0x77 leftover -> 0x4A play leftover. Walk only; cave mouth is (176,77).

    Hunts every screen on the way (``overworld.hunt.ScreenHunter``): the walk
    is the rupee farm. ``need_rupees=0`` keeps the old restock-farm loops off —
    overworld waves are one-shot, so a restock stall was never a rupee supply.
    """

    hops: tuple[ScreenHop, ...] = BOMB_SHOP_HOPS
    require_sword: bool = True
    farm_below_hearts: int = 0  # OW waves are one-shot
    need_rupees: int = 0  # 0 disables restock-farm loops; transit scoop collects
    max_farm_attempts: int = 0  # transit farming only; no restock stalls
    evade: bool = True
    occupied_lane: bool = True  # LEFT/RIGHT peel; path.py does not peel UP/DOWN
    # Clear each screen on the way. The errand is money and the only money on
    # this leg is what the wave drops: crossing 0x78/0x68/0x58/0x59/0x49 on one
    # lane arrived at the shop with 0-1 rupees against a 20R price.
    hunt: bool = True
    # 0x4A's own six blue tektites are drop-table row 1 (0.891 R/kill): 5.3R
    # of the corridor's 8.4R from 6 of its 28 bodies, and the hop table ends
    # the frame Link scrolls onto them. Without this the walk's best screen is
    # the one screen it never fights.
    hunt_destination: bool = True
    # 0x59 is crossed, not cleared. Its wave is four peahats and a Zora —
    # ROM drop row 3, 0.081 R/kill, the cheapest table in the game — and the
    # ROM spawn table does not even list the Zora, so nothing knew it was
    # there. One measured pass (2026-09-15 ``tables1``) spent 533 hunt
    # frames, the walk's only whole heart and a 5-kill streak on it for one
    # kill and no rupees; the streak Link carried in was worth more than
    # every body on the screen. A peahat cannot be hit while it flies and a
    # Zora is not a kill at all (``prey.SKIP_TYPES``), so this is not a
    # fight the wooden sword can win faster — it is one with nothing in it.
    # The hop still crosses the screen and the blade still answers contact.
    hunt_transit_screens: frozenset[int] = frozenset({0x59})
    # Laps of the corridor to walk before the shop. 0 is one pass, which is
    # all the drop tables can pay for; see ``PRE_L1_LAP_HOPS`` for why a lap
    # and not an out-and-back. Build the table with ``pre_l1_walk_hops``.
    laps: int = 0
    max_frames: int = BOMB_SHOP_WALK_MAX_FRAMES

    def _at_stop(self, snap: ZeldaSnapshot) -> bool:
        if bomb_shop_arrived(snap) and int(snap.rupees) >= BOMB_SHOP_PRICE:
            # The errand is funded. Another lap is only more chances to be hit.
            return True
        if self.laps and self.hop_index < len(self.hops):
            # 0x4A is the end of *every* lap, not only the last one, so a
            # lapped table cannot stop on arrival. Gated on ``laps`` so the
            # one-pass walk keeps the stop it greened on.
            return False
        return bomb_shop_arrived(snap) and self.destination_hunted(snap)

    def _after_hops(self, snap: ZeldaSnapshot) -> FrameAction:
        final = self._final_hunt(snap)
        if final is not None:
            return final
        if not self.destination_hunted(snap):
            # The hunt is between bodies (settle / spawn wait). Idling here is
            # cheap and bounded: it clears on ``HUNT_SETTLE_FRAMES`` or retires
            # on the screen budget, and there is no next hop to drive anyway.
            return FrameAction(nes_idle_action(), "bomb_shop_hunt_settle")
        if bomb_shop_arrived(snap):
            return self._finish("shop_p7_arrived")
        return self._fail("hops_complete_not_bomb_shop")

    def _rupee_scoop(self, snap: ZeldaSnapshot, hop: ScreenHop) -> FrameAction | None:
        """Transit drop scooping: hearts when hurt, rupees under the shop price,
        then bombs — and only once Link owns bombs at all.

        Order matters. A bomb drop is ROM item code ``0x00``, the same ObjState
        a cleared object slot reads as, and the bomb scoop used to sit ahead of
        the rupee scoop behind ``want=snap.bombs < 4`` — always true pre-L1,
        because Link has no bomb item yet. On this very leg that made a phantom
        slot outrank every real rupee, and rupees are what buys the bombs.
        """
        if snap.mode != PLAY_MODE or snap.level != 0:
            return None
        # 1. Hearts / fairies when damaged
        heart = scoop_floor_drop(
            snap,
            types=(HEART_DROP_OBJECT_TYPE, RUPEE_DROP_OBJECT_TYPE),
            states=HEART_OR_FAIRY_STATES,
            travel_dir=hop.direction,
            radius=self.scoop_radius,
            reason="scoop_heart",
            want=snap.filled_hearts < snap.heart_containers,
        )
        if heart is not None:
            return heart
        # 2. Rupees while short of the shop price (this is the pre-L1 errand)
        rupee = scoop_floor_drop(
            snap,
            types=(RUPEE_DROP_OBJECT_TYPE,),
            states=RUPEE_DROP_STATES,
            travel_dir=hop.direction,
            radius=self.scoop_radius,
            reason="scoop_rupee",
            # Not ``< BOMB_SHOP_PRICE``: the bombs are the first purchase on a
            # list of fourteen, and a rupee left on the floor at 20 is one the
            # candle at 0x0C still needs.
            want=True,
        )
        if rupee is not None:
            return rupee
        # 3. Bombs — only when Link owns bombs and is short of a pack
        return scoop_floor_drop(
            snap,
            types=(BOMB_DROP_OBJECT_TYPE,),
            states=BOMB_DROP_STATES,
            travel_dir=hop.direction,
            radius=self.scoop_radius,
            reason="scoop_bomb",
            want=owns_bombs(snap) and int(snap.bombs) < 4,
        )


def make_shop_p7_walk_controller(*, laps: int = 0) -> OverworldPathController:
    """0x77 leftover -> 0x4A play leftover, 0x4A hunted. No door_x / cave enter.

    ``laps`` walks the corridor again behind the first pass. A re-entered
    screen is a fresh fight, so the hunt has to forget it cleared one
    (``hunt_reopen``) or the lap is a walk with no wave in it.
    """
    return ShopP7WalkController(
        hops=pre_l1_walk_hops(laps), laps=int(laps), hunt_reopen=bool(laps)
    )


def make_pre_l1_bomb_buy_controller() -> CaveShopBuyController:
    """0x4A play leftover -> cave mouth (176,77) -> mid pedestal, bombs 20R.

    No hops: the walk stage already stands on 0x4A, so the buy engine starts
    in ``_after_hops``. No restock farm either — overworld waves are one-shot
    at depth 1-2, so the 0x4A<->0x49 loop is a give-up detector rather than a
    rupee supply, and 36,000 frames of it would only hide how short the walk
    came. Short of 20R this fails closed with ``shop_need_20_have_N``.
    """
    return make_bomb_shop_controller(hops=(), restock_farm=False)


def pre_l1_stages() -> tuple[tuple[str, object, int], ...]:
    """Dedicated spine hop: wooden sword, the hunting walk, then the 4-pack."""
    return (
        ("sword_cave", SwordCaveController(), SWORD_MAX),
        ("bomb_walk", make_shop_p7_walk_controller(), BOMB_SHOP_WALK_MAX_FRAMES),
        ("bomb_buy", make_pre_l1_bomb_buy_controller(), BOMB_BUY_MAX_FRAMES),
    )


__all__ = [
    "BOMB_SHOP_HOPS",
    "BOMB_SHOP_PRICE",
    "BOMB_SHOP_SCREEN",
    "BOMB_SHOP_WALK_MAX_FRAMES",
    "DEAD_68_EAST_Y141",
    "DEAD_6C_EAST_BUSH",
    "DEAD_ROW6_EAST_TO_6F",
    "PRE_L1_BOMB_HOPS",
    "PRE_L1_LAP_HOPS",
    "PRE_L1_LAP_WEST_HOPS",
    "SHOP_P7_HOPS",
    "SHOP_P7_HOPS_LIVE_PREFIX",
    "SHOP_P7_PRICE",
    "SHOP_P7_SCREEN",
    "SHOP_P7_WALK_MAX_FRAMES",
    "SOURCE_HYPOTHESIS",
    "BOMB_BUY_MAX_FRAMES",
    "SWORD_MAX",
    "ShopP7WalkController",
    "bomb_shop_arrived",
    "bomb_shop_screens",
    "make_pre_l1_bomb_buy_controller",
    "make_shop_p7_walk_controller",
    "pre_l1_bomb_shop_success",
    "pre_l1_walk_hops",
    "pre_l1_stages",
    "shop_p7_arrived",
    "shop_p7_screens",
]

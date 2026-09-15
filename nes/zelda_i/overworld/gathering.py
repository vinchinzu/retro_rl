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
from zelda_i.combat import (
    BOMB_DROP_OBJECT_TYPE,
    BOMB_DROP_STATES,
    HEART_OR_FAIRY_STATES,
    RUPEE_DROP_STATES,
)
from zelda_i.dungeon.ids import HEART_DROP_OBJECT_TYPE, RUPEE_DROP_OBJECT_TYPE
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
PRE_L1_BOMB_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x78, "RIGHT", align_y=140),
    ScreenHop(0x68, "UP", align_x=48),
    ScreenHop(0x58, "UP", align_x=48),
    ScreenHop(0x59, "RIGHT", y_band_lo=148, y_band_hi=162),
    ScreenHop(0x49, "UP", align_x=112),
    ScreenHop(0x4A, "RIGHT", align_y=141),
)

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
    """Play leftover on 0x4A with the sword."""
    return (
        snap.level == 0
        and snap.mode == PLAY_MODE
        and snap.screen == BOMB_SHOP_SCREEN
        and snap.has_sword
    )


shop_p7_arrived = bomb_shop_arrived


def pre_l1_bomb_shop_success(snap: ZeldaSnapshot) -> bool:
    """4.5.1: arrival on 0x4A bomb shop screen with sword (or bombs acquired)."""
    return (
        snap.level == 0
        and snap.has_sword
        and (
            (snap.screen == BOMB_SHOP_SCREEN and snap.mode in (PLAY_MODE, 11))
            or int(snap.bombs) >= 1
        )
    )


@dataclass
class ShopP7WalkController(OverworldPathController):
    """0x77 leftover -> 0x4A play leftover. Walk only; cave mouth is (176,77)."""

    hops: tuple[ScreenHop, ...] = BOMB_SHOP_HOPS
    require_sword: bool = True
    farm_below_hearts: int = 0  # OW waves are one-shot
    need_rupees: int = 0  # 0 disables restock-farm loops; transit scoop collects
    max_farm_attempts: int = 0  # transit farming only; no restock stalls
    evade: bool = True
    occupied_lane: bool = True  # LEFT/RIGHT peel; path.py does not peel UP/DOWN
    max_frames: int = BOMB_SHOP_WALK_MAX_FRAMES

    def _at_stop(self, snap: ZeldaSnapshot) -> bool:
        return bomb_shop_arrived(snap)

    def _after_hops(self, snap: ZeldaSnapshot) -> FrameAction:
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
            want=snap.rupees < BOMB_SHOP_PRICE,
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


def make_shop_p7_walk_controller() -> OverworldPathController:
    """0x77 leftover -> 0x4A play leftover. No door_x / cave enter."""
    return ShopP7WalkController()


def pre_l1_stages() -> tuple[tuple[str, object, int], ...]:
    """Dedicated spine hop: wooden sword, then the 0x6F walk."""
    return (
        ("sword_cave", SwordCaveController(), SWORD_MAX),
        ("shop_p7_walk", make_shop_p7_walk_controller(), SHOP_P7_WALK_MAX_FRAMES),
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
    "SHOP_P7_HOPS",
    "SHOP_P7_HOPS_LIVE_PREFIX",
    "SHOP_P7_PRICE",
    "SHOP_P7_SCREEN",
    "SHOP_P7_WALK_MAX_FRAMES",
    "SOURCE_HYPOTHESIS",
    "SWORD_MAX",
    "ShopP7WalkController",
    "bomb_shop_arrived",
    "bomb_shop_screens",
    "make_shop_p7_walk_controller",
    "pre_l1_bomb_shop_success",
    "pre_l1_stages",
    "shop_p7_arrived",
    "shop_p7_screens",
]

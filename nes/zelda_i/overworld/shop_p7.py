"""Coast bomb shop (catalog ``shop_p7``): Map-1 south coast 0x77 → 0x6F.

Map-1.png gives the screens and the directions. It does not give a row that
walks: three of the nine hops take a measured live lane instead of the
painted centre one (``scratch/probe_coast_lane.py``). 0x79 east is the south
beach (y≈165), not the overlay centre lane (y≈130), which is the rocky bowl.
``overworld.bomb_shop`` is the later 0x4A cave (same ROM shop family, arrows
and a 20R pack after L1). Inland joins (0x68 / 0x5C maze / 0x5E candle) stay
off this table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from retro_harness.input_script import FrameAction
from zelda_i.overworld.cave_shop import CaveShopBuyController
from zelda_i.overworld.graph import SCREEN_START, ScreenHop, path_screens_from_hops
from zelda_i.overworld.hunt import (
    HUNT_DESTINATION_FRAMES,
    HUNT_SCREEN_MAX_FRAMES,
    ScreenHunter,
)
from zelda_i.overworld.path import OverworldPathController
from zelda_i.overworld.zd_map import mirror_screen_hops
from zelda_i.ram import ADDR_BOMBS, PLAY_MODE, ZeldaSnapshot

SHOP_P7_SCREEN = 0x6F
SHOP_P7_PRICE = 20
SHOP_P7_WALK_MAX_FRAMES = 30000
SHOP_P7_BUY_MAX_FRAMES = 8000
SHOP_P7_CAVE_X = 48
SHOP_P7_CAVE_Y = 77
SHOP_P7_BUY_X = 152
SHOP_P7_BUY_Y = 149
# South of the 0x79 rock column. Overlay paints ~130 (the bowl).
SCREEN_79_BEACH_Y = 165
# The row east of x=192 that reaches 0x7A (133 and 141 are open there).
SCREEN_79_EXIT_Y = 133
COAST_TEKTITE_SCREEN = 0x7A
# Measured east lanes (``scratch/probe_coast_lane.py``, tag ``l1``, one boot,
# emulator state restored per row, 17 rows a screen). A row is a lane only if
# holding EAST from it scrolled the screen:
#
#   0x79  165 only                 0x7A  133, 141 only     0x7B  every row
#   0x7C  every row                0x7D  every row         0x7E  117/125/141+
#   0x7F  no east lane at all (the hop out of 0x7F is UP into 0x6F)
#
# Two of those are why the walk timed out. ``align_y`` carries ``y_tol=5``
# (``overworld.common.align_and_push``), so 0x7A's painted 131 accepts y=126
# — a dead row — and the retest sat 27501f on it pressing RIGHT at y~126 with
# 53 occupancy misses. 0x7E's painted 130 accepts 133, which is also dead.
# A band is the honest shape for both: it is the measured corridor, not a
# point plus a tolerance that reaches outside it.
SCREEN_7A_EAST_BAND = (133, 141)
SCREEN_7E_EAST_BAND = (137, 145)
# 0x7B, 0x7C and 0x7D scrolled east from *every* row the sweep tried, which
# was ``range(77, 206, 8)``. Carrying an ``align_y`` across 0x7B/0x7C is
# therefore not a lane, it is a vertical shuffle in a leever swarm, and the
# frame census priced it: 394 of 0x7B's 1863 frames were ``hop_ay`` — Link
# walking up and down to reach a row that was never required
# (``pre_l1_census1``). 0x7D is the same shape for *its own* exit, but 0x7E
# is not: want_y=133 stood at 131 and never scrolled (``l1``).
# ``pre_l1_topup_live`` entered 0x7E at that dead row and died ``(40,131)``.
# The 0x7E east band therefore sits on the hop that *leaves* 0x7D, so the
# drift is corrected before the scroll, not after. 0x7B/0x7C keep the
# sweep's own extent so the push never corrects.
SCREEN_ANY_ROW_BAND = (77, 205)

# Screens the hunt crosses instead of clearing. See ScreenHunter.transit_screens.
SHOP_P7_TRANSIT_SCREENS: frozenset[int] = frozenset({0x7B, 0x7D})

# Inland / later-shop screens. The coast table must not grow these back in.
SHOP_P7_NOT_ON_WALK: frozenset[int] = frozenset(
    {
        0x4A,  # later arrows cave
        0x5C,  # L2/L8 maze
        0x5E,  # candle shop
        0x68,
        0x58,
        0x59,
        0x49,  # L2 prefix inland
        0x6A,
        0x6B,
        0x6C,  # row-6 / south-skirt dead
        0x67,
        0x1B,
    }
)


# Painted Map-1 hops with the three measured live lanes. Not a PNG decode.
SHOP_P7_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x78, "RIGHT", align_y=133),
    ScreenHop(0x79, "RIGHT", align_y=130),
    ScreenHop(0x7A, "RIGHT", align_y=SCREEN_79_BEACH_Y),
    ScreenHop(0x7B, "RIGHT", y_band_lo=SCREEN_7A_EAST_BAND[0], y_band_hi=SCREEN_7A_EAST_BAND[1]),
    ScreenHop(0x7C, "RIGHT", y_band_lo=SCREEN_ANY_ROW_BAND[0], y_band_hi=SCREEN_ANY_ROW_BAND[1]),
    ScreenHop(0x7D, "RIGHT", y_band_lo=SCREEN_ANY_ROW_BAND[0], y_band_hi=SCREEN_ANY_ROW_BAND[1]),
    ScreenHop(0x7E, "RIGHT", y_band_lo=SCREEN_7E_EAST_BAND[0], y_band_hi=SCREEN_7E_EAST_BAND[1]),
    ScreenHop(0x7F, "RIGHT", y_band_lo=SCREEN_7E_EAST_BAND[0], y_band_hi=SCREEN_7E_EAST_BAND[1]),
    ScreenHop(0x6F, "UP", align_x=82),
)
PRE_L1_BOMB_HOPS: tuple[ScreenHop, ...] = SHOP_P7_HOPS
PRE_L1_LAP_WEST_HOPS: tuple[ScreenHop, ...] = mirror_screen_hops(
    SCREEN_START, SHOP_P7_HOPS
)
PRE_L1_LAP_HOPS: tuple[ScreenHop, ...] = PRE_L1_LAP_WEST_HOPS + SHOP_P7_HOPS

_SCREENS = path_screens_from_hops(SCREEN_START, SHOP_P7_HOPS)
assert _SCREENS[0] == SCREEN_START == 0x77
assert _SCREENS[-1] == SHOP_P7_SCREEN == 0x6F
assert _SCREENS == (0x77, 0x78, 0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E, 0x7F, 0x6F)
assert SHOP_P7_NOT_ON_WALK.isdisjoint(_SCREENS)
assert SHOP_P7_HOPS[2] == ScreenHop(0x7A, "RIGHT", align_y=SCREEN_79_BEACH_Y)
assert SHOP_P7_HOPS[3].y_band == SCREEN_7A_EAST_BAND
assert SHOP_P7_HOPS[6].y_band == SCREEN_7E_EAST_BAND
assert SHOP_P7_HOPS[7].y_band == SCREEN_7E_EAST_BAND
# 0x7B / 0x7C still leave on every row. 0x7D does too, but the hop that
# leaves it carries 0x7E's live corridor so the next screen is not entered
# on the dead 133 row. Picking that drift up *after* the scroll is the
# ``(40,131)`` death.
assert all(h.y_band == SCREEN_ANY_ROW_BAND for h in SHOP_P7_HOPS[4:6])
assert all(h.align_y is None for h in SHOP_P7_HOPS[4:7])


def shop_p7_screens() -> tuple[int, ...]:
    return _SCREENS


def pre_l1_walk_hops(laps: int = 0) -> tuple[ScreenHop, ...]:
    """The walk to 0x6F, plus ``laps`` full laps of the corridor behind it."""
    return SHOP_P7_HOPS + PRE_L1_LAP_HOPS * max(int(laps), 0)


def shop_p7_arrived(snap: ZeldaSnapshot) -> bool:
    """Play leftover on the coast shop 0x6F with the sword. Arrival, not the buy."""
    return (
        snap.level == 0
        and snap.mode == PLAY_MODE
        and snap.screen == SHOP_P7_SCREEN
        and snap.has_sword
    )


@dataclass
class ShopP7WalkController(OverworldPathController):
    """0x77 leftover → Map-1 south coast → 0x6F. Walk only; cave is the buy stage.

    Hunts every screen on the way (``overworld.hunt.ScreenHunter``): the walk
    is the rupee farm. ``need_rupees=0`` keeps restock-farm loops off;
    ``scoop_rupees`` / ``scoop_bombs`` still bank floor drops.
    0x79 east is the beach skirt around the centre rocks into 0x7A.
    """

    hops: tuple[ScreenHop, ...] = SHOP_P7_HOPS
    require_sword: bool = True
    farm_below_hearts: int = 0
    need_rupees: int = 0
    scoop_rupees: bool = True
    scoop_bombs: bool = True
    max_farm_attempts: int = 0
    evade: bool = True
    occupied_lane: bool = True
    # The coast's whole damage bill is the Zora (``hits_by_cause``: 4 of 6
    # hits, twice running). Its spit sits on the muzzle for ~16 frames, so a
    # six-sample velocity reads 0.4 px/frame on the first moving frame and
    # only tells the truth once the shot has closed 10 px of a 40 px lane.
    shot_history: int = 2
    # The 0x79 skirt keeps the default ``HOP_RUNG_EXTRA`` — the top of the
    # hop ladder, which is where the hook has always been called from. Its
    # gate is clearance, not precedence; ``_extra_hop_action`` carries the
    # measurement that says why those are different.
    hunter: ScreenHunter | None = field(
        default_factory=lambda: ScreenHunter(
            transit_screens=SHOP_P7_TRANSIT_SCREENS
        )
    )
    hunt_destination: bool = True
    laps: int = 0
    max_frames: int = SHOP_P7_WALK_MAX_FRAMES

    def _leave_79_east(self, snap: ZeldaSnapshot) -> FrameAction:
        x, y = snap.link_x, snap.link_y
        if x < 35:
            if y < 120:
                return self._swing("DOWN", "79_mouth_drop")
            return self._swing("RIGHT", "79_mouth_enter")
        if x < 192:
            if y < SCREEN_79_BEACH_Y and x < 120:
                return self._swing("DOWN", "79_skirt_south")
            return self._swing("RIGHT", "79_skirt_beach")
        # Past x=192 only rows 133/141 reach the east edge: come to 133
        # from either side (from (192, 109) RIGHT was rock for 28184 frames
        # after a rupee scoop, run 19). A RIGHT press within 4 px slides
        # onto the row.
        if y > SCREEN_79_EXIT_Y:
            return self._swing("UP", "79_skirt_exit_up")
        if y < SCREEN_79_EXIT_Y - 4:
            return self._swing("DOWN", "79_skirt_exit_down")
        return self._swing("RIGHT", "79_skirt_exit_east")

    def _leave_79_west(self, snap: ZeldaSnapshot) -> FrameAction:
        x, y = snap.link_x, snap.link_y
        if x > 190:
            return self._swing("LEFT", "79_west_enter")
        if y < SCREEN_79_BEACH_Y and x > 40:
            return self._swing("DOWN", "79_west_south")
        if y > 125 and x <= 40:
            return self._swing("UP", "79_west_mouth")
        return self._swing("LEFT", "79_west_exit")

    def _extra_hop_action(
        self, snap: ZeldaSnapshot, hop: ScreenHop
    ) -> FrameAction | None:
        """The 0x79 skirt route, once the wave on 0x79 is off the chase list.

        C2 read the opening ``0x79 not in self.hunter.done`` as a precedence
        edit — the hook declining so the hunt rung below could have the frame
        — and the fix for that is ``extra_hop_priority``. Wiring the ladder
        proved the reading wrong here, and the counter-example is worth
        keeping: a *completion* gate and a *decline* are not the same set.

        ``not in done`` opens once and stays open. A rung sitting under the
        hunt opens on every frame the hunt declines, of which there are many
        while the wave is alive — so moving this hook below ``hop_hunt``
        changes behaviour in both directions. It hands the skirt frames that
        used to push east (waking ``_leave_79_east``'s ``x < 35`` branch,
        which was dead code), and it lets the beam, the scoop and the hunt's
        own ``_lane_return`` take frames the skirt used to own outright.

        So the gate stays and the hook keeps the top of the ladder, which is
        where it has always been. What changed is that its place is now the
        number ``extra_hop_priority`` rather than the line ``_do_hop`` calls
        it from, and the gate asks :meth:`ScreenHunter.chase_finished` — a
        declared query — instead of reaching into the ``done`` set. Pre-L1 is
        frame-perfect (``AGENTS.md``: M5 Clean 18909f is live, re-measure
        after walker changes), so whether the skirt *should* run before the
        wave dies is a policy question that deserves its own measured card,
        not a side effect of a refactor.
        """
        if self.hunter is not None and not self.hunter.chase_finished(0x79):
            return None
        if snap.screen == 0x79 and hop.target == 0x7A:
            return self._leave_79_east(snap)
        if snap.screen == 0x79 and hop.target == 0x78:
            return self._leave_79_west(snap)
        return None

    def _before_play(self, snap: ZeldaSnapshot) -> FrameAction | None:
        """Keep the coast wave open until the pack is banked.

        600f retires a screen with the drop still on the floor, which is how
        a pass reaches 0x6F a rupee or two short. Once the wallet can pay,
        the cap drops back so the rest of the walk is travel.
        """
        if self.hunter is not None:
            self.hunter.screen_max_frames = (
                HUNT_DESTINATION_FRAMES
                if int(snap.rupees) < SHOP_P7_PRICE
                else HUNT_SCREEN_MAX_FRAMES
            )
        return None

    def _at_stop(self, snap: ZeldaSnapshot) -> bool:
        """Play 0x6F after the hops. Short of the pack is ``bomb_topup``.

        Fighting the shop wave until ``destination_hunted`` deadlocked
        ``pre_l1_c3_melee1``: 18R, cave mode 11, 22403f of
        ``shop_p7_hunt_settle``. Arrival-short must leave and fight next
        door, not idle in the cave.
        """
        if not shop_p7_arrived(snap):
            return False
        if int(snap.rupees) >= SHOP_P7_PRICE:
            return True
        if self.laps and self.hop_index < len(self.hops):
            return False
        return True

    def _after_hops(self, snap: ZeldaSnapshot) -> FrameAction:
        if shop_p7_arrived(snap):
            return self._finish("shop_p7_arrived")
        return self._fail("hops_complete_not_shop_p7")


def make_shop_p7_walk_controller(*, laps: int = 0) -> OverworldPathController:
    """0x77 leftover → 0x6F along the south coast. No door_x / cave enter."""
    return ShopP7WalkController(
        hops=pre_l1_walk_hops(laps),
        laps=int(laps),
        hunter=ScreenHunter(
            transit_screens=SHOP_P7_TRANSIT_SCREENS,
            # A lap walks every screen twice and a screen re-entered is a fresh
            # fight; the one-pass walk must not reopen. See respawn.
            reopen_on_enter=bool(laps),
        ),
    )


@dataclass
class ShopP7BuyController(CaveShopBuyController):
    """Buy engine for the 4-pack at the 0x6F coast cave."""

    def _simple_door_hunt(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.screen == self.shop_screen and snap.link_y > 120:
            return self._swing("UP", "door_up_first")
        return super()._simple_door_hunt(snap)


def make_shop_p7_buy_controller() -> CaveShopBuyController:
    """Buy engine for the 4-pack at the 0x6F coast cave mouth (48, 77)."""
    return ShopP7BuyController(
        hops=(),
        enter_cave=True,
        door_x=SHOP_P7_CAVE_X,
        door_dir="UP",
        door_screen=SHOP_P7_SCREEN,
        shop_screen=SHOP_P7_SCREEN,
        cave_x=SHOP_P7_CAVE_X,
        cave_y=SHOP_P7_CAVE_Y,
        buy_x=SHOP_P7_BUY_X,
        buy_y=SHOP_P7_BUY_Y,
        price=SHOP_P7_PRICE,
        success_getter=lambda s: int(s.bombs),
        success_threshold=1,
        success_addr=ADDR_BOMBS,
        success_note="bombs_bought",
        farm=None,
    )


__all__ = [
    "COAST_TEKTITE_SCREEN",
    "SCREEN_ANY_ROW_BAND",
    "SCREEN_7A_EAST_BAND",
    "SCREEN_7E_EAST_BAND",
    "PRE_L1_BOMB_HOPS",
    "PRE_L1_LAP_HOPS",
    "PRE_L1_LAP_WEST_HOPS",
    "SCREEN_79_BEACH_Y",
    "SHOP_P7_BUY_MAX_FRAMES",
    "SHOP_P7_HOPS",
    "SHOP_P7_NOT_ON_WALK",
    "SHOP_P7_PRICE",
    "SHOP_P7_SCREEN",
    "SHOP_P7_TRANSIT_SCREENS",
    "SHOP_P7_WALK_MAX_FRAMES",
    "ShopP7BuyController",
    "ShopP7WalkController",
    "make_shop_p7_buy_controller",
    "make_shop_p7_walk_controller",
    "pre_l1_walk_hops",
    "shop_p7_arrived",
    "shop_p7_screens",
]

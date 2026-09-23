"""Arrive short at a shop, go fight next door, come back and buy.

The coast walk banks the pack's price on a good pass and one rupee short of it
on a bad one (``pre_l1_beam4``: 19R of 20R, 24 kills). Arrival is therefore
not the errand — ``ADDR_BOMBS >= 1`` is — and a walk that reaches ``0x6F``
short has exactly one honest move left: fight somewhere the ROM will still
spawn a wave.

**Not the corridor behind it.** ``overworld.respawn`` is the ROM's own rule:
``ModifyObjCountByHistoryOW`` reopens a screen's wave only when the screen is
absent from ``RoomHistory`` ($621, six slots) *and* its kill flags already
read 7, and ``RunCrossRoomTasksAndBeginUpdateMode`` appends a room only if it
is not already in the ring. Walking 0x77…0x6F leaves the ring holding the last
six of those screens, so every screen within five hops west of the shop is
still in it and an out-and-back evicts nothing at any depth. Turning round
buys frames and damage and no enemies at all.

What *is* fresh is a screen the walk has never entered. Those are the shop
screen's own neighbours, which is why this module is a table of one-hop
out-and-back excursions rather than a lap: one hop out is one hop of exposure,
and the wave on the far side has never been touched.

``SHOP_P7_TOPUP_EXCURSIONS`` is measured, not painted —
``scratch/probe_6f_neighbours.py`` sweeps each exit off 0x6F row by row from a
restored arrival state, pushes, counts what spawned, and pushes back, which is
the same shape ``probe_coast_lane.py`` used for the east lanes. An empty
neighbour table is not a short finish. The fallback then hunts the nearest
coast screen ``RoomHistory`` has dropped and comes home. Inland screens are
not that hunt. The buy runs only once the wallet can pay.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_idle_action
from zelda_i.overworld.common import recover_off_edge
from zelda_i.overworld.graph import ScreenHop
from zelda_i.overworld.hunt import (
    HUNT_DESTINATION_FRAMES,
    HUNT_SCREEN_MAX_FRAMES,
    ScreenHunter,
    hop_lane,
)
from zelda_i.overworld.path import OverworldPathController
from zelda_i.overworld.respawn import RoomHistory
from zelda_i.overworld.shop_p7 import (
    SHOP_P7_HOPS,
    SHOP_P7_NOT_ON_WALK,
    SHOP_P7_SCREEN,
    SHOP_P7_TRANSIT_SCREENS,
    shop_p7_screens,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

# Back hop travel → the way into the neighbour. An out-and-back lands on
# the reverse hop's arrival edge, which is the scroll line home.
_INWARD = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

__all__ = [
    "SCREEN_6E_WEST_BAND",
    "SCREEN_6F_NORTH_X",
    "SHOP_P7_TOPUP_EXCURSIONS",
    "TOPUP_MAX_FRAMES",
    "Excursion",
    "RupeeTopUpController",
    "coast_fallback_hops",
    "excursion_hops",
    "make_shop_p7_topup_controller",
]

# Neighbours first, then each coast screen the walk has already left behind,
# hunted out to the destination cap while the wallet is short. 600f is how a
# screen used to retire with the drop still on it.
TOPUP_MAX_FRAMES = 30000


@dataclass(frozen=True)
class Excursion:
    """One out-and-back off the shop screen, as the two hops that walk it.

    ``out.target`` is the neighbour and ``back.target`` must be the shop
    screen: an excursion that does not come home strands the buy stage on the
    wrong screen, and that is a table error, not a runtime one, so it is
    checked at construction.
    """

    name: str
    out: ScreenHop
    back: ScreenHop

    def __post_init__(self) -> None:
        if self.out.target == self.back.target:
            raise ValueError(f"{self.name}: out and back name the same screen")


def excursion_hops(
    shop_screen: int, excursions: tuple[Excursion, ...]
) -> tuple[ScreenHop, ...]:
    """Flatten the table into one hop list, checking every leg comes home."""
    hops: list[ScreenHop] = []
    for trip in excursions:
        if trip.back.target != shop_screen:
            raise ValueError(
                f"{trip.name}: back hop targets {trip.back.target:#04x}, "
                f"not the shop screen {shop_screen:#04x}"
            )
        hops.extend((trip.out, trip.back))
    return tuple(hops)


def _mirror_lane(fwd: ScreenHop, target: int, direction: str) -> ScreenHop:
    """Same measured lane, opposite push. ``align_y`` alone accepts a dead row."""
    if direction in ("LEFT", "RIGHT"):
        return ScreenHop(
            target,
            direction,
            align_y=fwd.align_y,
            y_band_lo=fwd.y_band_lo,
            y_band_hi=fwd.y_band_hi,
        )
    return ScreenHop(target, direction, align_x=fwd.align_x)


def coast_fallback_hops(
    visited: tuple[int, ...], *, skip: frozenset[int] = frozenset()
) -> tuple[int | None, tuple[ScreenHop, ...]]:
    """Nearest coast screen history would reopen, then back to the shop.

    ``visited`` is the coast walk plus whatever this stage already entered.
    Transit screens were crossed, not cleared, so a history miss there does
    not bring a wave back. Inland screens are not candidates. ``skip`` is
    the set this stage already walked out to hunt: entering one evicts
    another, and chasing that rotation is not a farm.
    """
    history = RoomHistory()
    for screen in visited:
        history.enter(int(screen))
    screens = shop_p7_screens()
    target: int | None = None
    for screen in reversed(screens[:-1]):
        if screen in history.slots or screen in skip:
            continue
        if screen in SHOP_P7_TRANSIT_SCREENS or screen in SHOP_P7_NOT_ON_WALK:
            continue
        target = int(screen)
        break
    if target is None:
        return None, ()
    shop_i = len(screens) - 1
    target_i = screens.index(target)
    out: list[ScreenHop] = []
    for i in range(shop_i, target_i, -1):
        fwd = SHOP_P7_HOPS[i - 1]
        out.append(_mirror_lane(fwd, screens[i - 1], _INWARD[fwd.direction]))
    out.extend(SHOP_P7_HOPS[target_i:shop_i])
    return target, tuple(out)


# Measured off the coast shop (``scratch/probe_6f_neighbours.py``, tag ``n4``:
# one boot, the real hop table to 0x6F, the emulator state saved on arrival,
# then every candidate row/column restored, walked to, pushed, censused and
# pushed back). Both neighbours cross **and come home** from the whole band
# below; neither had a lane before this.
#
#   0x6F UP   -> 0x5F   every column 72..232, 193-240f out, 85f home.
#               Wave: 3 octorok_fast, 2 octorok_blue, 1 zora (peak 6).
#               Link cannot align past x=128 on 0x6F; the push works anyway.
#   0x6F LEFT -> 0x6E   y 93..197, 155-377f out, 106f home. y 77 / 85 / 205
#               are dead. Wave: 3 moblin, 2 octorok_blue_fast,
#               1 moblin_blue (peak 6).
#
# 0x5F is first because its wave is octoroks — two wooden hits each, and the
# ROM's drop table pays them — where 0x6E is four moblins. Both are six
# bodies, which is the same size wave as the corridor screens that cost the
# walk its hearts, so this is a real fight and not a lap of an empty screen.
SCREEN_6F_NORTH_X = 122  # cheapest measured column out (193f)
SCREEN_6E_WEST_BAND = (109, 189)  # every row here crossed in a flat 168f
SHOP_P7_TOPUP_EXCURSIONS: tuple[Excursion, ...] = (
    Excursion(
        "north_5f",
        ScreenHop(0x5F, "UP", align_x=SCREEN_6F_NORTH_X),
        ScreenHop(SHOP_P7_SCREEN, "DOWN", align_x=SCREEN_6F_NORTH_X),
    ),
    Excursion(
        "west_6e",
        ScreenHop(
            0x6E, "LEFT",
            y_band_lo=SCREEN_6E_WEST_BAND[0], y_band_hi=SCREEN_6E_WEST_BAND[1],
        ),
        ScreenHop(
            SHOP_P7_SCREEN, "RIGHT",
            y_band_lo=SCREEN_6E_WEST_BAND[0], y_band_hi=SCREEN_6E_WEST_BAND[1],
        ),
    ),
)


@dataclass
class RupeeTopUpController(OverworldPathController):
    """Hunt the shop's neighbours until the price is banked, then come home.

    Stop is ``rupees >= price`` **on the shop screen**, never just the rupee
    count: the stage after this one enters a cave mouth, so finishing one
    screen north with the money is the same failure as not having it.

    A walk that already has the price finishes on its first frame, which is
    what makes this safe to leave in the stage list unconditionally — it is a
    no-op on every pass that did not need it. A walk that does not is not
    finished here. The neighbours are hunted first; still short, the nearest
    coast screen the ring has dropped is next. The buy does not run until
    this stage is home with the price.
    """

    shop_screen: int = 0
    price: int = 0
    excursions: tuple[Excursion, ...] = ()
    require_sword: bool = True
    scoop_rupees: bool = True
    scoop_bombs: bool = True
    need_rupees: int = 0
    farm_below_hearts: int = 0
    max_farm_attempts: int = 0
    evade: bool = True
    occupied_lane: bool = True
    shot_history: int = 2
    hunt_destination: bool = True
    max_frames: int = TOPUP_MAX_FRAMES
    _shortfall_targets: set[int] = field(default_factory=set, repr=False)

    def reset(self) -> None:
        super().reset()
        self._shortfall_targets.clear()

    def _short(self, snap: ZeldaSnapshot) -> bool:
        return int(snap.rupees) < int(self.price)

    def _visited(self) -> tuple[int, ...]:
        return shop_p7_screens() + tuple(int(hop.target) for hop in self.hops)

    def _before_play(self, snap: ZeldaSnapshot) -> FrameAction | None:
        """While the pack is short, do not retire the wave at 600f."""
        if self.hunter is not None:
            self.hunter.screen_max_frames = (
                HUNT_DESTINATION_FRAMES
                if self._short(snap)
                else HUNT_SCREEN_MAX_FRAMES
            )
        return None

    def _home(self, snap: ZeldaSnapshot) -> bool:
        return (
            int(snap.level) == 0
            and int(snap.mode) == PLAY_MODE
            and int(snap.screen) == int(self.shop_screen)
        )

    def _at_stop(self, snap: ZeldaSnapshot) -> bool:
        return self._home(snap) and not self._short(snap)

    def _extra_hop_action(
        self, snap: ZeldaSnapshot, hop: ScreenHop
    ) -> FrameAction | None:
        """Fight the neighbour before the back hop retraces.

        Live ``scratch/probe_topup.py`` t1: hop_index advanced onto the DOWN
        home hop the first play frame of 0x5F (``on_arrival_edge`` names the
        *travel* edge, and an UP arrival is the south). Hunt then skipped
        because y>200 is the DOWN arrival edge, ``recover_off_edge`` allowed
        DOWN, and the hop retraced in 87f with ``peak_live`` 0. 0x6E was the
        same one-frame visit. Both neighbours measured a six-body wave;
        neither got a hunt frame. The same hold applies to a coast screen
        the shortfall walked out to. A wallet that can already pay skips it
        and takes the hop home. Transit screens are crossed.
        """
        if self.hunter is None or not self._short(snap):
            return None
        screen = int(snap.screen)
        if screen == int(self.shop_screen) or screen in self.hunter.transit_screens:
            return None
        hunted = None
        if screen not in self.hunter.done:
            hunted = self.hunter.step(
                snap, self.frames, lane=hop_lane(hop), y_band=hop.y_band
            )
            # ``FarmOccupancy`` stands when the 1px grid has no path. On 0x6E
            # that is the bush maze, and extra returning the stand left Link
            # on the east scroll line for 214f (live t2). The hold's inward
            # step is the sand corridor the neighbour probe already walked.
            if hunted is not None and hunted.reason != "occupancy_stand":
                return hunted
        if screen in self.hunter.done:
            return None
        inward = _INWARD.get(hop.direction)
        if inward is not None:
            rec = recover_off_edge(
                snap,
                inward,
                swing=lambda d, _r: self._swing(d, "topup_hold"),
            )
            if rec is not None:
                return rec
            if hunted is not None and hunted.reason == "occupancy_stand":
                return self._swing(inward, "topup_hold")
        return FrameAction(nes_idle_action(), "topup_hold")

    def _keep_hunting(self, snap: ZeldaSnapshot) -> FrameAction:
        """Home, still short. Leave again. Do not hand the buy a short wallet."""
        target, extra = coast_fallback_hops(
            self._visited(), skip=frozenset(self._shortfall_targets)
        )
        if target is None or not extra:
            return self._fail(f"topup_short_{int(snap.rupees)}")
        self._shortfall_targets.add(int(target))
        self.hops = tuple(self.hops) + extra
        self.notes.append(f"topup_hunt_{target:02x}")
        return FrameAction(nes_idle_action(), f"topup_hunt_{target:02x}")

    def _after_hops(self, snap: ZeldaSnapshot) -> FrameAction:
        """Home with the price is the stop. Home without it keeps hunting.

        Fighting the shop screen itself is the cave-mode deadlock. The
        neighbours, then a coast screen the ring has dropped, are the hunt.
        """
        if self._home(snap) and not self._short(snap):
            return self._finish("topup_done")
        if self._home(snap):
            return self._keep_hunting(snap)
        final = self._final_hunt(snap)
        if final is not None:
            return final
        if not self.destination_hunted(snap):
            return FrameAction(nes_idle_action(), "topup_hunt_settle")
        return self._fail("topup_not_home")


def make_shop_p7_topup_controller(
    *, shop_screen: int, price: int,
    excursions: tuple[Excursion, ...] = SHOP_P7_TOPUP_EXCURSIONS,
) -> RupeeTopUpController:
    """Top-up for the 0x6F coast pack: fight the neighbours, come back to 0x6F.

    The hunter reopens on entry. The shop screen is re-entered once per
    excursion and the ROM decides whether its wave is back (absent from
    ``RoomHistory`` *and* flags 7); leaving it in ``done`` would make the
    hunt decline a wave that did respawn, which is the one thing the walk is
    here for.
    """
    return RupeeTopUpController(
        hops=excursion_hops(shop_screen, excursions),
        shop_screen=int(shop_screen),
        price=int(price),
        excursions=excursions,
        hunter=ScreenHunter(
            reopen_on_enter=True,
            transit_screens=SHOP_P7_TRANSIT_SCREENS,
        ),
    )

"""Generic parameterized cave-shop buy controller (no inventory/rupee poke).

Templated from two live buy state machines:

- ``OverworldToArrowShopController`` (``zelda_i.level1.arrow_shop``) — the
  0x4A wooden-arrow cave: settle, UP stairs to a lateral ``buy_y``, walk to
  ``buy_x``, touch/hold until ``ADDR_ARROWS`` flips 0→1.
- ``OverworldToCandleShopController`` (``zelda_i.level8.overworld``) — the
  ``CandleShop5E`` (0x5E) cave with the same stairs/lateral/touch shape but
  a different pedestal (Blue Candle, right pedestal, mid = Key 100R).

Both shops share one buy shape: dialog idle wait, walk UP the stairs to a
lateral ``buy_y``, walk to ``buy_x``, then touch/hold until purchase is
confirmed (item at threshold plus a debit, free price, or already owned) —
with a ``buy_budget`` timeout and a rupee-insufficiency fail-close from
``snap.rupees`` (never a poke). This module extracts that shape as
``CaveShopBuyController`` so any shop chapter can instantiate it instead of
writing a bespoke buy state machine.

Cost check: the cost is read from ``snap.rupees`` only. When short, the
controller either calls the supplied ``RupeeFarmController`` (``farm``) to
raise rupees to ``price``, or — with no farm configured — fails closed with
a ``shop_need_{price}_have_{rupees}`` note. Never a rupee poke either way.

Success reading: since ``ZeldaSnapshot`` does not carry every purchasable
item (arrows/bombs are fields; candle/food are not), the success value is
read via a caller-supplied ``success_getter(snap) -> int`` rather than a
raw address poke/peek baked into this module. ``success_addr`` is kept only
for documentation/``report()`` — it is never itself read or written here.

Pedestal spacing: adjacent pedestals sit close together (0x4A mid Bombs 20R
at y≈149 vs Arrows at y≈165; 0x5E mid Key 100R vs Candle 60R at y≈149) — the
per-shop ``buy_y``/``buy_x`` must be precise enough that the UP-stairs +
lateral walk touches the intended pedestal, not its neighbor.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.overworld.graph import ScreenHop
from zelda_i.overworld.path import OverworldPathController
from zelda_i.overworld.rupee_farm import RupeeFarmController, RupeeFarmPhase
from zelda_i.ram import CAVE_MODE, ZeldaSnapshot

__all__ = [
    "BUY_BUDGET",
    "CAVE_DIALOG_IDLE",
    "CaveShopBuyController",
    "CaveShopBuyPhase",
    "DOOR_HUNT_TIMEOUT",
    "NORTH_GAP_Y_HI",
]

BUY_BUDGET = 900
CAVE_DIALOG_IDLE = 120
DOOR_HUNT_TIMEOUT = 1800
NORTH_GAP_Y_HI = 120


class CaveShopBuyPhase(Enum):
    HOP = auto()
    FARM = auto()
    DOOR = auto()
    BUY = auto()
    DONE = auto()
    FAILED = auto()


def _default_success_getter(snap: ZeldaSnapshot) -> int:
    return 0


@dataclass
class CaveShopBuyController(OverworldPathController):
    """Walk a hop table to an OW cave mouth, enter, farm if short, buy.

    Parameters (bead-literal): OW screen (``shop_screen``), cave mouth
    (``cave_x``/``cave_y`` — ``cave_x`` also doubles as the door-align x),
    dialog idle frames (``cave_dialog_idle``), stairs-UP lateral y
    (``buy_y``), touch point (``buy_x``/``buy_y``), success read
    (``success_getter``/``success_threshold``, ``success_addr`` for
    reporting only), and ``price`` — checked only against ``snap.rupees``,
    calling ``farm`` (a ``RupeeFarmController``) when short. Never pokes
    rupees or the item address; only real gameplay contact changes them.
    """

    phase: CaveShopBuyPhase = CaveShopBuyPhase.HOP
    hops: tuple[ScreenHop, ...] = ()
    enter_cave: bool = True
    door_dir: str = "UP"
    require_sword: bool = True

    # --- Bead-literal shop parameters ---
    shop_screen: int = 0
    cave_x: int = 0
    cave_y: int = 0
    buy_x: int = 0
    buy_y: int = 0
    price: int = 0
    success_getter: Callable[[ZeldaSnapshot], int] = _default_success_getter
    success_threshold: int = 1
    success_addr: int | None = None  # documentation / report() only
    success_note: str = "item_bought"

    cave_dialog_idle: int = CAVE_DIALOG_IDLE
    buy_budget: int = BUY_BUDGET
    door_hunt_timeout: int = DOOR_HUNT_TIMEOUT

    # Optional cave-mouth vs. north-gap x-alignment guard (0x4A style: a
    # north exit gap sits near the cave-mouth x but leads off-screen). Unset
    # (None) disables the check — most shops don't need it.
    north_gap_x: int | None = None
    north_gap_y_hi: int = NORTH_GAP_Y_HI
    # A south arrival may have a solid west edge; climb to this open row
    # before aligning with the cave's x column (0x34 Armos shop).
    door_approach_y: int | None = None
    # On an Armos shop, pushing UP wakes the statue and carries Link past the
    # newly exposed stairs. Turn back onto those stairs from above.
    door_reverse_y: int | None = None
    _door_returning: bool = False

    # Rupee farm — call into it (never poke) when short of ``price``. None
    # means "no farm available": short-of-price fails closed instead.
    farm: RupeeFarmController | None = None

    buy_frames: int = 0
    farm_frames: int = 0
    empty_frames: int = 0
    _rupees_at_buy: int | None = None
    _item_owned_at_start: bool | None = None
    leftover: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.door_x is None:
            self.door_x = self.cave_x
        if self.door_screen is None:
            self.door_screen = self.shop_screen
        if self.need_rupees <= 0 and self.price > 0:
            self.need_rupees = int(self.price)

    def reset(self) -> None:
        super().reset()
        self.buy_frames = 0
        self.farm_frames = 0
        self.empty_frames = 0
        self._rupees_at_buy = None
        self._item_owned_at_start = None
        self.leftover = None
        self._door_returning = False
        if self.farm is not None:
            self.farm.reset()

    def end_screen(self) -> int:
        return self.shop_screen

    def _wants_post_hop(self) -> bool:
        return True

    def _in_shop_cave(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.mode == CAVE_MODE
            and snap.level == 0
            and snap.screen == self.shop_screen
        )

    def _purchase_done(self, snap: ZeldaSnapshot) -> bool:
        if self._item_owned_at_start is None:
            self._item_owned_at_start = (
                self.success_getter(snap) >= self.success_threshold
            )
        if self.success_getter(snap) < self.success_threshold:
            return False
        if self.price <= 0 or self._item_owned_at_start:
            return True
        return (
            self._rupees_at_buy is not None
            and snap.rupees <= self._rupees_at_buy - self.price
        )

    def _at_stop(self, snap: ZeldaSnapshot) -> bool:
        return self._purchase_done(snap)

    def _record(self, snap: ZeldaSnapshot) -> None:
        self.leftover = {
            "mode": int(snap.mode),
            "screen": int(snap.screen),
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "rupees": int(snap.rupees),
            "item_value": int(self.success_getter(snap)),
            "phase": self.phase.name,
        }

    def _before_play(self, snap: ZeldaSnapshot) -> FrameAction | None:
        self._record(snap)
        if self.phase is CaveShopBuyPhase.FARM:
            # RupeeFarmController owns its own transition (scroll) handling.
            return self._farm_step(snap)
        if snap.transitioning:
            if self.phase is CaveShopBuyPhase.DOOR:
                return FrameAction(nes_action("UP"), "cave_transition")
            return None
        if self.phase is CaveShopBuyPhase.BUY:
            return self._buy_step(snap)
        if self.phase is CaveShopBuyPhase.DOOR and self._in_shop_cave(snap):
            self._rupees_at_buy = snap.rupees
            self._set_phase(CaveShopBuyPhase.BUY, "in_shop_cave")
            return FrameAction(nes_idle_action(), "shop_ready")
        return None

    def _after_hops(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.phase is CaveShopBuyPhase.HOP:
            if snap.rupees < self.price:
                self._set_phase(
                    CaveShopBuyPhase.FARM,
                    f"farm_need_{self.price}_have_{snap.rupees}",
                )
                return self._farm_step(snap)
            self._set_phase(CaveShopBuyPhase.DOOR, "door_hunt")
        if self.phase is CaveShopBuyPhase.FARM:
            return self._farm_step(snap)
        if self.phase is CaveShopBuyPhase.BUY:
            return self._buy_step(snap)
        return self._simple_door_hunt(snap)

    def _simple_door_hunt(self, snap: ZeldaSnapshot) -> FrameAction:
        if self._in_shop_cave(snap):
            self._rupees_at_buy = snap.rupees
            self._set_phase(CaveShopBuyPhase.BUY, "in_shop_cave")
            return FrameAction(nes_idle_action(), "shop_ready")
        if self.phase_frames > self.door_hunt_timeout:
            return self._fail(
                f"cave_timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            )
        if snap.screen != self.shop_screen:
            return super()._simple_door_hunt(snap)
        if self.door_reverse_y is not None:
            if snap.link_y < self.door_reverse_y:
                self._door_returning = True
            if self._door_returning:
                if (
                    self.door_approach_y is not None
                    and snap.link_y > self.door_approach_y
                ):
                    self._door_returning = False
                else:
                    return self._swing("DOWN", "door_exposed_stairs")
        if (
            self.door_approach_y is not None
            and self.door_x is not None
            and abs(snap.link_x - self.door_x) > 5
            and snap.link_y > self.door_approach_y
        ):
            return self._swing("UP", "door_approach_row")
        if (
            self.north_gap_x is not None
            and snap.link_y < self.north_gap_y_hi
            and abs(snap.link_x - self.north_gap_x) > 8
        ):
            return self._swing("DOWN", "cave_drop")
        if self.door_x is not None and abs(snap.link_x - self.door_x) > 5:
            btn = "LEFT" if snap.link_x > self.door_x else "RIGHT"
            return self._swing(btn, "door_ax")
        return self._swing(self.door_dir, "door_hunt")

    def _farm_step(self, snap: ZeldaSnapshot) -> FrameAction:
        """Call the generic ``RupeeFarmController`` to close a rupee shortfall.

        Never pokes ``$066D``: the farm only reads ``snap`` and returns
        walk/swing ``FrameAction``s. With no ``farm`` configured, a
        shortfall fails closed immediately instead of stalling. A farm
        failure (timeout, death, left overworld, unreachable leftover) is
        surfaced as this controller's own ``FAILED`` phase with the same
        note.
        """
        if self.farm is None:
            return self._fail(f"shop_need_{self.price}_have_{snap.rupees}")
        action = self.farm.step(snap)
        self.farm_frames = self.farm.frames
        self.empty_frames = self.farm.empty_frames
        if self.farm.phase is RupeeFarmPhase.FAILED:
            note = self.farm.notes[-1] if self.farm.notes else "farm_failed"
            return self._fail(note)
        if self.farm.phase is RupeeFarmPhase.DONE:
            note = self.farm.notes[-1] if self.farm.notes else "farm_ok"
            self._set_phase(CaveShopBuyPhase.DOOR, note)
            return FrameAction(nes_idle_action(), "farm_done")
        return action

    def _buy_step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.buy_frames += 1
        if self._purchase_done(snap):
            return self._finish(self.success_note)
        if self.buy_frames > self.buy_budget:
            return self._fail(
                f"buy_timeout_{snap.link_x}_{snap.link_y}_r{snap.rupees}"
            )
        if not self._in_shop_cave(snap):
            self._set_phase(CaveShopBuyPhase.DOOR, "left_cave")
            return FrameAction(nes_idle_action(), "reenter")
        if self._rupees_at_buy is None:
            self._rupees_at_buy = snap.rupees
        if (
            self._rupees_at_buy < self.price
            and self.buy_frames > self.cave_dialog_idle
        ):
            return self._fail(f"shop_need_{self.price}_have_{self._rupees_at_buy}")
        if self.buy_frames < self.cave_dialog_idle and snap.link_y > 200:
            return FrameAction(nes_idle_action(), "shop_dialog")
        # Lateral y avoids the adjacent pedestal (0x4A mid bombs @y≈149 vs
        # arrows @y=165; 0x5E mid Key 100R vs candle @y≈149) — stay off it.
        if snap.link_y > self.buy_y + 1:
            return FrameAction(nes_action("UP"), "shop_up_stairs")
        if snap.link_x < self.buy_x:
            return FrameAction(nes_action("RIGHT"), "shop_right")
        return FrameAction(nes_action("UP"), "shop_touch")

    def report(self) -> dict[str, Any]:
        base = super().report()
        base.update(
            {
                "end_screen": self.end_screen(),
                "shop_screen": self.shop_screen,
                "price": self.price,
                "success_addr": (
                    hex(self.success_addr) if self.success_addr is not None else None
                ),
                "buy_frames": self.buy_frames,
                "farm_frames": self.farm_frames,
                "farm": self.farm.report() if self.farm is not None else None,
                "rupees_at_buy": self._rupees_at_buy,
                "leftover": dict(self.leftover or {}),
            }
        )
        return base

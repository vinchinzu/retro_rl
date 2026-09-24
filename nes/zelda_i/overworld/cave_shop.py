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

``PotionShopBuyController`` is the one subclass: a potion shop sells only
after the 0x0E letter has been shown inside it once, and waits on its
keeper's text before the wares can be touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.hop_controller import mouth_step
from zelda_i.dungeon.pause_select import (
    B_SLOT_LETTER,
    PauseSelectController,
    b_slot_owned,
    pause_dropped,
)
from zelda_i.overworld.graph import ScreenHop
from zelda_i.overworld.path import OverworldPathController
from zelda_i.overworld.rupee_farm import RupeeFarmController, RupeeFarmPhase
from zelda_i.ram import (
    ADDR_POTION,
    ADDR_SELECTED_ITEM,
    CAVE_MODE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_u8,
)

__all__ = [
    "BLUE_POTION_PRICE",
    "BUY_BUDGET",
    "CAVE_DIALOG_IDLE",
    "CaveShopBuyController",
    "CaveShopBuyPhase",
    "DOOR_HUNT_TIMEOUT",
    "NORTH_GAP_Y_HI",
    "POTION_SHOP_SCREEN",
    "PotionShopBuyController",
    "RED_POTION_PRICE",
    "make_potion_buy_controller",
    "make_potion_restock_controller",
    "potion_restock_stages",
    "restock_item",
]

BUY_BUDGET = 900
CAVE_DIALOG_IDLE = 120
DOOR_HUNT_TIMEOUT = 1800
NORTH_GAP_Y_HI = 120
# A pedestal left of the stairs (x=112) is walked to from the right. UP
# slides Link to the NEAREST 8 px column, not onward (measured: UP at x=95
# slid to 96, then LEFT to 95, a tug-of-war beside the 88 blue potion), so
# LEFT runs until the pedestal's column is the nearest. A right-hand
# pedestal overshot by a pixel or two must not turn back.
PEDESTAL_SLACK = 3

# Potion shop, cave type 0x1A (ROM cave table, file offset 0x18610): blue
# potion (item 0x1F) 40R on the left, red (0x20) 68R on the right. Measured
# 2026-09-23 at 0x64: an OPEN mouth at x=112 (no slot-11 tile object; UP
# at x=112 from y=93 enters) although ``locations`` lists it as a secret.
# Keeper 0x74 at (120, 128). Its ObjState is 0 before the letter (no text,
# no wares, Link free), 1 while its text types (Link halted, $40), 2 once
# the wares are up. B on the letter (slot 15) inside sets $0666 1 -> 2 on
# that frame. Walking over a pedestal while the keeper is at 1 buys
# nothing. Touch rows: blue (88, 157), red (152, 157) from y=165.
POTION_SHOP_SCREEN = 0x64
POTION_MOUTH_X = 112
POTION_MOUTH_Y = 77
POTION_KEEPER = 0x74
KEEPER_WARES_UP = 2
LETTER_TAKEN = 1
LETTER_SHOWN = 2
BLUE_POTION_PRICE = 40
RED_POTION_PRICE = 68
BLUE_POTION_X = 88
RED_POTION_X = 152
POTION_BUY_Y = 165
POTION_BUY_BUDGET = 1500
# Lattice row under the mouth; ``mouth_step`` routes there from any side
# (the ring return arrives at (60, 61), boxed in by trees to the east).
POTION_APPROACH_Y = 93
# START is dropped for ~40 frames after a cave's mode 11 (entry walk, mode
# init, a 14-frame halt). Link free this long means the menu will open.
CAVE_SETTLE_FRAMES = 8
LETTER_PRESS_TRIES = 3
LETTER_PRESS_GAP = 4


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
    min_item_gain: int = 0  # consumable shops: a held item is not a new buy
    min_headroom: int = 0  # refuse a pack that would waste capacity
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
    # Lattice row under the mouth. Set, ``mouth_step`` routes there from any
    # side and pushes ``door_dir`` (a bomb or potion mouth above a walled
    # south gap); unset keeps the blind align-then-push hunt.
    mouth_approach_y: int | None = None

    # Rupee farm — call into it (never poke) when short of ``price``. None
    # means "no farm available": short-of-price fails closed instead.
    farm: RupeeFarmController | None = None

    buy_frames: int = 0
    farm_frames: int = 0
    empty_frames: int = 0
    _rupees_at_buy: int | None = None
    _item_owned_at_start: bool | None = None
    _item_at_start: int | None = None
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
        self._item_at_start = None
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
        item = int(self.success_getter(snap))
        if self._item_at_start is None:
            self._item_at_start = item
        if self._item_owned_at_start is None:
            self._item_owned_at_start = item >= self.success_threshold
        if item < self.success_threshold:
            return False
        if self.min_item_gain and item < self._item_at_start + self.min_item_gain:
            return False
        if self.min_item_gain:
            return self.price <= 0 or (
                self._rupees_at_buy is not None
                and snap.rupees <= self._rupees_at_buy - self.price
            )
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
        if (
            self.mouth_approach_y is not None
            and snap.level == 0
            and snap.mode == PLAY_MODE
        ):
            direction = mouth_step(
                snap,
                int(self.door_x),
                int(self.mouth_approach_y),
                direction=self.door_dir,
                env=self._env,
            )
            return self._swing(direction, "cave_mouth")
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
        capacity = int(getattr(snap, "max_bombs", 0))
        item_base = (
            self._item_at_start
            if self._item_at_start is not None
            else int(self.success_getter(snap))
        )
        if (
            self.min_headroom
            and capacity
            and capacity - item_base < self.min_headroom
        ):
            return self._fail(
                f"shop_headroom_{capacity - item_base}_need_{self.min_headroom}"
            )
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
        if snap.link_x > self.buy_x + PEDESTAL_SLACK:
            # Inside the slack, UP slides on to the pedestal's column.
            return FrameAction(nes_action("LEFT"), "shop_left")
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
                "item_at_start": self._item_at_start,
                "min_item_gain": self.min_item_gain,
                "leftover": dict(self.leftover or {}),
            }
        )
        return base


def _potion(snap: ZeldaSnapshot) -> int:
    return int(snap.potion)


def restock_item(snap: ZeldaSnapshot, *, reserve: int = 0) -> str | None:
    """The potion a restock buys, keeping ``reserve`` rupees, else ``None``.

    A buy must add a drink: a red is full, and a blue over a blue is still
    one drink, so a held blue only upgrades to red. ``reserve`` is what the
    route owes next (the 60R Bait before L7).
    """
    potion, spare = int(snap.potion), int(snap.rupees) - int(reserve)
    if potion >= 2:
        return None
    if spare >= RED_POTION_PRICE:
        return "red"
    if potion == 0 and spare >= BLUE_POTION_PRICE:
        return "blue"
    return None


@dataclass
class PotionShopBuyController(CaveShopBuyController):
    """Buy a potion: show the letter once, wait for the wares, buy, restore B.

    ``item`` is ``"red"`` (68R, right), ``"blue"`` (40R, left) or ``"auto"``:
    red when the wallet holds 68 on arriving at the shop screen, else blue.
    Short of the chosen price, the base class fails closed (or farms, with a
    ``farm``). The B item held on entering the cave is put back after the
    purchase, so a later stage's B is what it was (showing the letter and the
    buy both move the cursor). Stops inside the cave at the pedestal.
    """

    item: str = "auto"
    # Between dungeons: pick with ``restock_item`` (keeping ``reserve``
    # rupees) on the first frame, or end at once when nothing adds a drink.
    restock: bool = False
    reserve: int = 0
    keeper: int = POTION_KEEPER
    letter_shows: int = 0
    chosen: str = ""
    _prior_b: int | None = None
    _select: PauseSelectController | None = None
    _letter_presses: int = 0
    _letter_gap: int = 0
    _restore_done: bool = False
    _settled: int = 0

    def __post_init__(self) -> None:
        self._configure("blue" if self.item == "auto" else self.item)
        super().__post_init__()

    def _configure(self, item: str) -> None:
        if item not in ("red", "blue"):
            raise ValueError(f"potion item must be red, blue or auto: {item!r}")
        red = item == "red"
        self.chosen = item
        self.price = RED_POTION_PRICE if red else BLUE_POTION_PRICE
        self.buy_x = RED_POTION_X if red else BLUE_POTION_X
        self.success_threshold = 2 if red else 1
        self.success_note = f"{item}_potion_bought"

    def reset(self) -> None:
        super().reset()
        self._prior_b = None
        self._select = None
        self._letter_presses = 0
        self._letter_gap = 0
        self._restore_done = False
        self._settled = 0

    def _at_stop(self, snap: ZeldaSnapshot) -> bool:
        return self._purchase_done(snap) and self._restore_done

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.restock and self.frames == 0:
            item = restock_item(snap, reserve=self.reserve)
            if item is None:
                self.frames += 1
                return self._finish("potion_restock_nothing_to_buy")
            self.item = item
            self._configure(item)
        return super().step(snap)

    def _after_hops(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.phase is CaveShopBuyPhase.HOP:
            if int(snap.letter) < LETTER_TAKEN:
                return self._fail("potion_shop_needs_letter")
            if self.item == "auto":
                self._configure("red" if snap.rupees >= RED_POTION_PRICE else "blue")
        return super()._after_hops(snap)

    def _selector(self, want: int) -> PauseSelectController:
        ctl = PauseSelectController(want=want)
        ctl.bind_env(self._env)
        return ctl

    def _cave_settled(self, snap: ZeldaSnapshot) -> bool:
        """Link free and the mode past its init for ``CAVE_SETTLE_FRAMES``."""
        free = int(snap.is_updating_mode) != 0 and (
            not snap.objects or int(snap.objects[0].state) == 0
        )
        self._settled = self._settled + 1 if free else 0
        return self._settled >= CAVE_SETTLE_FRAMES

    def _buy_step(self, snap: ZeldaSnapshot) -> FrameAction:
        if not self._in_shop_cave(snap):
            return super()._buy_step(snap)
        if self._env is None:
            return self._fail("potion_shop_env_not_bound")
        ram = self._env.get_ram()
        settled = self._cave_settled(snap)
        if self._prior_b is None:
            self._prior_b = int(read_u8(ram, ADDR_SELECTED_ITEM))
        if self._purchase_done(snap):
            act = self._restore_b(snap, ram, settled)
            return act if act is not None else super()._buy_step(snap)
        keeper = next((o for o in snap.objects if int(o.type_id) == self.keeper), None)
        if keeper is not None and int(keeper.state) == KEEPER_WARES_UP:
            return super()._buy_step(snap)
        self.buy_frames += 1
        if self.buy_frames > self.buy_budget:
            return self._fail(f"potion_wares_timeout_letter{int(snap.letter)}")
        if keeper is not None and int(keeper.state) == 0 and int(snap.letter) == LETTER_TAKEN:
            return self._show_letter(snap, settled)
        return FrameAction(nes_idle_action(), "potion_shop_text")

    def _show_letter(self, snap: ZeldaSnapshot, settled: bool) -> FrameAction:
        """Letter on B, one press: ``$0666`` 1 -> 2 and the keeper talks."""
        link_free = not snap.objects or int(snap.objects[0].state) == 0
        if self._select is None:
            if not settled:
                return FrameAction(nes_idle_action(), "potion_letter_wait")
            self._select = self._selector(B_SLOT_LETTER)
        if pause_dropped(self._select, self._env.get_ram()):
            self._select, self._settled = None, 0
            return FrameAction(nes_idle_action(), "potion_pause_dropped")
        act = self._select.drive(snap)
        if self._select.failed:
            return self._fail(f"letter_{self._select.fail_reason}")
        if act is not None:
            return act
        self._letter_gap += 1
        if self._letter_presses >= LETTER_PRESS_TRIES and self._letter_gap > LETTER_PRESS_GAP:
            return self._fail("letter_not_shown")
        if link_free and (self._letter_presses == 0 or self._letter_gap > LETTER_PRESS_GAP):
            self._letter_presses += 1
            self._letter_gap = 0
            if self._letter_presses == 1:
                self.letter_shows += 1
            return FrameAction(nes_action("B"), "potion_show_letter")
        return FrameAction(nes_idle_action(), "potion_letter_press_wait")

    def _restore_b(self, snap: ZeldaSnapshot, ram: Any, settled: bool) -> FrameAction | None:
        """Put the B item from the cave entry back. None once it is there."""
        if self._restore_done:
            return None
        prior = self._prior_b
        if self._select is not None and self._select.want != prior:
            self._select = None  # the letter select is finished
        if self._select is None:
            selected = int(read_u8(ram, ADDR_SELECTED_ITEM))
            if prior is None or prior == selected or not b_slot_owned(ram, prior):
                self._restore_done = True
                return None
            if not settled:
                return FrameAction(nes_idle_action(), "potion_restore_wait")
            self._select = self._selector(prior)
        if pause_dropped(self._select, ram):
            self._select, self._settled = None, 0
            return FrameAction(nes_idle_action(), "potion_pause_dropped")
        act = self._select.drive(snap)
        if self._select.failed:
            return self._fail(f"restore_b_{self._select.fail_reason}")
        if act is None:
            self._restore_done = True
        return act

    def report(self) -> dict[str, Any]:
        base = super().report()
        base.update(
            {
                "item": self.item,
                "chosen": self.chosen,
                "letter_shows": self.letter_shows,
                "prior_b": self._prior_b,
                "b_restored": self._restore_done,
            }
        )
        return base


def make_potion_buy_controller(
    *, hops: tuple[ScreenHop, ...] | None = None, item: str = "auto", restock: bool = False
) -> PotionShopBuyController:
    """Potion buy at 0x64. Zero-arg: ``stage_replay.py`` target.

    ``hops`` default is the ring return's leg to 0x64 (``RING_RETURN_HOPS``
    through its 0x64 row): the chain's pose after ``exit_ring`` on 0x34,
    arriving on 0x64 from the north at x=60. Any other arrival works too
    (``mouth_step`` routes to the mouth). Needs the letter taken (``$0666``
    >= 1) and the price in the wallet on arrival (68 red, 40 blue).
    """
    if hops is None:
        from zelda_i.overworld.gather_segments import RING_RETURN_HOPS

        last = next(
            i for i, hop in enumerate(RING_RETURN_HOPS) if hop.target == POTION_SHOP_SCREEN
        )
        hops = RING_RETURN_HOPS[: last + 1]
    return PotionShopBuyController(
        hops=tuple(hops),
        item=item,
        restock=restock,
        shop_screen=POTION_SHOP_SCREEN,
        cave_x=POTION_MOUTH_X,
        cave_y=POTION_MOUTH_Y,
        mouth_approach_y=POTION_APPROACH_Y,
        buy_y=POTION_BUY_Y,
        buy_budget=POTION_BUY_BUDGET,
        success_getter=_potion,
        success_addr=ADDR_POTION,
        farm_below_hearts=0,
        evade=False,
        max_frames=12000,
        require_sword=True,
    )


def make_potion_restock_controller(
    *, hops: tuple[ScreenHop, ...], reserve: int = 0
) -> PotionShopBuyController:
    """0x64 buy between dungeons, skipped on its first frame when not wanted."""
    ctl = make_potion_buy_controller(hops=hops, restock=True)
    ctl.reserve = int(reserve)
    return ctl


def potion_restock_stages(
    hops: tuple[ScreenHop, ...], tag: str, *, reserve: int = 0
) -> tuple[tuple[str, Any, int], ...]:
    """Spine stages for a restock on a walk that crosses 0x64: buy, then exit.

    The walk after them should set ``resume_on_screen``: Link is on 0x64
    after a buy and still where he started after a skip. ``clear=0`` on the
    exit: neither case walks anywhere.
    """
    from zelda_i.overworld.gather_segments import CaveExitController

    to_shop = hops[: [hop.target for hop in hops].index(POTION_SHOP_SCREEN) + 1]
    buy = make_potion_restock_controller(hops=to_shop, reserve=reserve)
    return (
        (f"potion_restock_{tag}", buy, buy.max_frames),
        (f"exit_potion_{tag}", CaveExitController(clear=0), 600),
    )

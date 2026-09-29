"""Animal-shop sale menu classification and counter-menu input tapes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from harvest.core.ram_catalog import field_spec, read_ram_u16, read_ram_value
from harvest.tasks.nav import get_pos_from_ram, make_action
from harvest.tasks.primitives import press_a_sequence, press_button_sequence


SHOP_MENU_TEXT_ID = 0x0305
SELL_CHICKEN_TEXT_ID = 0x030B
# bank_81 animal-shop script. Displayed gold is storage × 10.
COW_CONFIRM_TEXT_ID = 0x0307
CHICKEN_CONFIRM_TEXT_ID = 0x030A
COW_BOUGHT_TEXT_ID = 0x0308
CHICKEN_BOUGHT_TEXT_ID = 0x0311
WEATHER_BLOCK_TEXT_ID = 0x03D0
BOUGHT_TODAY_TEXT_ID = 0x03B6
ANIMAL_CAP_TEXT_ID = 0x03B7
GRASS_SHORT_TEXT_ID = 0x0306
MONEY_SHORT_TEXT_ID = 0x0304
PURCHASE_DECLINED_TEXT_ID = 0x0303
WEATHER_BLOCK_MASK = 0x000A
BOUGHT_TODAY_MASK = 0x0020
ANIMAL_CAP = 12
GRASS_PER_ANIMAL = 16
COW_PRICE = 5000
CHICKEN_PRICE = 1000

_REJECT_TEXT = {
    WEATHER_BLOCK_TEXT_ID: "weather blocks delivery",
    BOUGHT_TODAY_TEXT_ID: "already bought an animal today",
    ANIMAL_CAP_TEXT_ID: "barn or coop is full",
    GRASS_SHORT_TEXT_ID: "not enough grass",
    MONEY_SHORT_TEXT_ID: "not enough money",
    PURCHASE_DECLINED_TEXT_ID: "purchase declined",
}

_SPECIES: Dict[str, Dict[str, int | str]] = {
    "cow": {
        "price": COW_PRICE,
        "choice": 0,
        "confirm": COW_CONFIRM_TEXT_ID,
        "count_key": "num_cows",
    },
    "chicken": {
        "price": CHICKEN_PRICE,
        "choice": 1,
        "confirm": CHICKEN_CONFIRM_TEXT_ID,
        "count_key": "num_chickens",
    },
}


def dialog_text_id(ram: np.ndarray) -> int:
    return read_ram_u16(ram, field_spec("dialog_text_id").address, live_offset=False)


def near_px(
    ram: np.ndarray, target: Tuple[int, int], tolerance: int
) -> bool:
    pos = get_pos_from_ram(ram)
    return abs(pos.x - target[0]) <= tolerance and abs(pos.y - target[1]) <= tolerance


def sale_request_ready(saw_shop_menu: bool, ram: np.ndarray, request_text_id: int) -> bool:
    """True once the four-option shop menu has advanced to the sale prompt."""
    return saw_shop_menu and dialog_text_id(ram) == request_text_id


def open_counter_menu_actions() -> List[np.ndarray]:
    return list(
        press_a_sequence(
            "right",
            face_frames=4,
            pre_press_settle_frames=0,
            hold_frames=12,
            settle_frames=12,
            hold_face_with_a=True,
        )
    )


def sell_chicken_choice_actions() -> List[np.ndarray]:
    """Select "sell chicken" (dialog 0x030B) from the four-option shop menu."""
    actions: List[np.ndarray] = []
    actions.extend(make_action(right=True, a=True) for _ in range(6))
    actions.extend(make_action(right=True) for _ in range(7))
    actions.extend(make_action(right=True, a=True) for _ in range(3))
    actions.extend(make_action(a=True) for _ in range(4))
    actions.extend(make_action() for _ in range(12))
    actions.extend(press_button_sequence("a", hold_frames=8, settle_frames=19))
    actions.extend(press_button_sequence("a", hold_frames=8, settle_frames=141))
    actions.extend(press_button_sequence("down", hold_frames=11, settle_frames=7))
    actions.extend(press_button_sequence("down", hold_frames=7, settle_frames=16))
    actions.extend(press_button_sequence("a", hold_frames=10, settle_frames=30))
    return actions


def payout_press_actions() -> List[np.ndarray]:
    return list(
        press_button_sequence(
            "a",
            face="down",
            face_frames=8,
            pre_press_settle_frames=0,
            hold_frames=10,
            settle_frames=45,
            hold_face_with_button=True,
        )
    )


def species_spec(species: str) -> Dict[str, int | str]:
    try:
        return _SPECIES[species]
    except KeyError as exc:
        raise ValueError(f"unknown animal species {species!r}") from exc


def weather_blocks_animal_delivery(ram: np.ndarray) -> bool:
    """Shopkeeper refuses delivery when $0196 has rain or snow (mask 0x000A)."""
    return (read_ram_u16(ram, 0x0196) & WEATHER_BLOCK_MASK) != 0


def animal_bought_today(ram: np.ndarray) -> bool:
    """SRAM 0x11F5A bit 0x20 is set after either animal is bought that day."""
    return (read_ram_u16(ram, 0x11F5A) & BOUGHT_TODAY_MASK) != 0


def purchase_grass_required(cows: int, chickens: int, species: str) -> int:
    """Planted grass tiles the shop script compares before AddMoney."""
    species_spec(species)
    if species == "cow":
        return chickens * GRASS_PER_ANIMAL + (cows + 1) * GRASS_PER_ANIMAL
    return (chickens + 1) * GRASS_PER_ANIMAL + cows * GRASS_PER_ANIMAL


def purchase_block_reason(ram: np.ndarray, species: str) -> Optional[str]:
    """Why this species cannot be bought, or None when the gates are open.

    The live task still talks to the shopkeeper. These predicates are the
    planner and the no-ROM contract: weather, one-animal-per-day, cap, wallet,
    and grass.
    """
    spec = species_spec(species)
    if weather_blocks_animal_delivery(ram):
        return "weather"
    if animal_bought_today(ram):
        return "bought_today"
    count = int(read_ram_value(ram, str(spec["count_key"])))
    if count >= ANIMAL_CAP:
        return "full"
    if int(read_ram_value(ram, "money")) < int(spec["price"]):
        return "money"
    cows = int(read_ram_value(ram, "num_cows"))
    chickens = int(read_ram_value(ram, "num_chickens"))
    grass = int(read_ram_value(ram, "planted_grass"))
    if grass < purchase_grass_required(cows, chickens, species):
        return "grass"
    return None


@dataclass
class PurchaseDecision:
    status: str
    reason: str
    action: Optional[np.ndarray] = None


@dataclass
class AnimalPurchaseSession:
    """Counter dialogue for one cow or one chicken. One animal per reload.

    Success is the species count rising by one and the displayed wallet
    falling by at least that animal's price. A menu, a reject line, or the
    name screen is not a buy.
    """

    species: str
    timeout: int = 3600

    _price: int = field(init=False)
    _count_key: str = field(init=False)
    _confirm: int = field(init=False)
    _choice: int = field(init=False)
    _start_money: int = field(default=0, init=False)
    _start_count: int = field(default=0, init=False)
    _frames: int = field(default=0, init=False)
    _queue: Deque[np.ndarray] = field(default_factory=deque, init=False)
    _opened: bool = field(default=False, init=False)
    _on_choice_frames: int = field(default=0, init=False)
    _nudge_hold: int = field(default=0, init=False)
    _nudges_done: int = field(default=0, init=False)
    _last_menu_a: int = field(default=-1000, init=False)
    _last_confirm_a: int = field(default=-1000, init=False)
    _last_advance_a: int = field(default=-1000, init=False)

    def start(self, ram: np.ndarray) -> None:
        spec = species_spec(self.species)
        self._price = int(spec["price"])
        self._count_key = str(spec["count_key"])
        self._confirm = int(spec["confirm"])
        self._choice = int(spec["choice"])
        self._start_money = int(read_ram_value(ram, "money"))
        self._start_count = int(read_ram_value(ram, self._count_key))
        self._frames = 0
        self._queue = deque()
        self._opened = False
        self._on_choice_frames = 0
        self._nudge_hold = 0
        self._nudges_done = 0
        self._last_menu_a = -1000
        self._last_confirm_a = -1000
        self._last_advance_a = -1000

    def _registered(self, ram: np.ndarray) -> bool:
        count = int(read_ram_value(ram, self._count_key))
        money = int(read_ram_value(ram, "money"))
        return count >= self._start_count + 1 and money <= self._start_money - self._price

    def _success_reason(self, ram: np.ndarray) -> str:
        count = int(read_ram_value(ram, self._count_key))
        money = int(read_ram_value(ram, "money"))
        return (
            f"{self.species} {self._start_count}->{count} "
            f"money {self._start_money}->{money}"
        )

    def step(self, ram: np.ndarray) -> PurchaseDecision:
        self._frames += 1
        if self._registered(ram):
            return PurchaseDecision("success", self._success_reason(ram))
        text = dialog_text_id(ram)
        if text in _REJECT_TEXT:
            return PurchaseDecision("failure", _REJECT_TEXT[text])
        other = CHICKEN_CONFIRM_TEXT_ID if self.species == "cow" else COW_CONFIRM_TEXT_ID
        if text == other:
            # Leave the box on No. Failing while Yes is highlighted lets the
            # recovery press confirm the other animal.
            if self._frames > self.timeout + 800:
                return PurchaseDecision(
                    "failure",
                    f"could not decline other animal text={text:#06x} "
                    f"cursor={self._menu_cursor(ram)}",
                )
            return self._decline_other(ram)
        if self._frames > self.timeout:
            return PurchaseDecision(
                "failure",
                f"{self.species} purchase timeout text={text:#06x} "
                f"cursor={self._menu_cursor(ram)} nudges={self._nudges_done} "
                f"money={int(read_ram_value(ram, 'money'))} "
                f"count={int(read_ram_value(ram, self._count_key))}",
            )
        if self._queue:
            return PurchaseDecision("run", "open counter menu", self._queue.popleft())
        if text == SHOP_MENU_TEXT_ID:
            return self._choose_menu(ram)
        if text == self._confirm:
            return self._pulse(self._frames - self._last_confirm_a, 20, "confirm yes", a=True, slot="confirm")
        lock = int(read_ram_value(ram, "input_lock"))
        if not self._opened and lock == 1:
            self._opened = True
            self._queue.extend(open_counter_menu_actions())
            return PurchaseDecision("run", "face counter", make_action())
        if lock != 1:
            return self._pulse(
                self._frames - self._last_advance_a,
                12,
                "advance shop text",
                a=True,
                slot="advance",
            )
        return PurchaseDecision("run", "wait for shopkeeper", make_action())

    def _menu_cursor(self, ram: np.ndarray) -> int:
        # $018A is the text offset. The ROM reads $018F for the chosen row.
        return int(read_ram_value(ram, "dialog_choice_result", raw=True))

    def _choose_menu(self, ram: np.ndarray) -> PurchaseDecision:
        """Confirm the species row. A one-frame Down does not move this menu.

        The sell-chicken tape holds Down for 11 frames per row. The highlight
        byte does not move on a one-frame tap, so a tap then A buys the cow.
        """
        if not int(ram[0x019B]) & 0x10:
            return self._pulse(self._frames - self._last_advance_a, 20,
                               "advance to shop choices", a=True, slot="advance")
        cursor = self._menu_cursor(ram)
        # A held Down that leaves the cursor byte on the cow row is still the
        # cow row. Confirming anyway buys the cow (live chicken reload).
        if cursor != self._choice:
            self._on_choice_frames = 0
            if self._nudge_hold < 11:
                self._nudge_hold += 1
                return PurchaseDecision(
                    "run",
                    f"menu hold down {self._nudge_hold}/11 cursor={cursor}",
                    make_action(down=True),
                )
            if self._nudge_hold < 19:
                self._nudge_hold += 1
                return PurchaseDecision("run", f"menu nudge settle cursor={cursor}", make_action())
            self._nudges_done += 1
            self._nudge_hold = 0
            return PurchaseDecision("run", f"menu row {self._nudges_done}", make_action())
        self._on_choice_frames += 1
        if self._on_choice_frames < 8:
            return PurchaseDecision("run", f"menu cursor {cursor} settle", make_action())
        return self._pulse(
            self._frames - self._last_menu_a,
            16,
            "confirm menu choice",
            a=True,
            slot="menu",
        )

    def _decline_other(self, ram: np.ndarray) -> PurchaseDecision:
        """Move a Yes/No box to No and confirm. Never press A on Yes."""
        if not int(ram[0x019B]) & 0x10:
            return self._pulse(self._frames - self._last_advance_a, 20,
                               "advance to yes/no", a=True, slot="advance")
        cursor = self._menu_cursor(ram)
        if cursor != 1:
            return self._pulse(
                self._frames - self._last_menu_a,
                6,
                "decline move to no",
                slot="menu",
                down=True,
            )
        return self._pulse(
            self._frames - self._last_confirm_a,
            12,
            "decline other animal",
            a=True,
            slot="confirm",
        )

    def _pulse(self, since: int, gap: int, reason: str, *, slot: str, **buttons: bool) -> PurchaseDecision:
        if since < gap:
            return PurchaseDecision("run", f"wait {reason}", make_action())
        if slot == "menu":
            self._last_menu_a = self._frames
        elif slot == "confirm":
            self._last_confirm_a = self._frames
        elif slot == "advance":
            self._last_advance_a = self._frames
        return PurchaseDecision("run", reason, make_action(**buttons))


def chicken_count_reason(
    start_chickens: int,
    current_chickens: int,
    start_money: int,
    current_money: int,
    *,
    money_label: str = "shipping",
    extra: Optional[str] = None,
) -> str:
    text = (
        f"chickens {start_chickens}->{current_chickens} "
        f"{money_label} {start_money}->{current_money}"
    )
    if extra:
        return f"{text} {extra}"
    return text


__all__ = [
    "ANIMAL_CAP",
    "CHICKEN_PRICE",
    "COW_PRICE",
    "SELL_CHICKEN_TEXT_ID",
    "SHOP_MENU_TEXT_ID",
    "AnimalPurchaseSession",
    "PurchaseDecision",
    "animal_bought_today",
    "chicken_count_reason",
    "dialog_text_id",
    "near_px",
    "open_counter_menu_actions",
    "payout_press_actions",
    "purchase_block_reason",
    "purchase_grass_required",
    "sale_request_ready",
    "sell_chicken_choice_actions",
    "weather_blocks_animal_delivery",
]

"""Shipping-credit facts for harvest → 5pm → overnight wallet settle.

ROM (HM-Decomp bank_82):

- Bin drop adds to ``shipping_money`` immediately (not wallet ``money``).
- At hour 17 on farm (tilemap < 4) ``ShippingScene`` runs dialogue.
- Wallet credit is ``AddMoney(shipping_money)`` during overnight/morning
  settle; ``shipping_money`` is then zeroed.

rr-53g acceptance: ``shipped_count > 0`` and wallet money rises after the
5pm shipping window (verified next morning). Pre/post-5pm checkpoints are
saved around the farm shipping scene, not as the money-credit moment.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from harvest.core.ram_catalog import read_ram_value


SHIPPING_SCENE_HOUR = 17
# bank_82 ShippingScene: tilemap < 4. Text_Daily_Shipping / _Nothing.
SHIPPING_DIALOG_TEXT_IDS = (0x031A, 0x031B)
# $7F1F5A bit set when hour hits 17 on farm; UpdateTime freezes until cleared.
SHIPPING_SCENE_PENDING_FLAG = 0x0400
SHIPPED_CROP_FIELDS = {"corn": "shipped_corn", "tomato": "shipped_tomatoes",
                       "turnip": "shipped_turnips", "potato": "shipped_potatoes"}


class ShippingLedger:
    """Read-only frame observer: count deposits once, across night resets.

    Opening bin contents and lifetime crop counts belong to the predecessor,
    so they establish the baseline. A bin decrease is a payout, not a sale;
    a crop-counter decrease is an anomaly, never a wrapped positive sale.
    """

    def __init__(self) -> None:
        self._previous: tuple[int, ...] | None = None
        self._seasons: dict[tuple[int, int], dict[str, Any]] = {}
        self.start: dict[str, int] | None = None
        self.end: dict[str, int] | None = None
        self.opening_bin_gold = 0
        self.observations = 0
        self.counter_decreases: list[dict[str, Any]] = []

    def observe(self, ram) -> None:
        date = {key: int(read_ram_value(ram, key)) for key in ("year", "season", "day")}
        values = tuple(int(read_ram_value(ram, key)) for key in
                       ("shipping_money", *SHIPPED_CROP_FIELDS.values()))
        self.observations += 1
        self.end = date
        row = self._seasons.setdefault((date["year"], date["season"]), {
            "year": date["year"], "season": date["season"], "bin_gold": 0,
            "crop_shipped": dict.fromkeys(SHIPPED_CROP_FIELDS, 0),
        })
        if self._previous is None:
            self.start, self.opening_bin_gold = date, values[0]
        else:
            row["bin_gold"] += max(0, values[0] - self._previous[0])
            for index, kind in enumerate(SHIPPED_CROP_FIELDS, 1):
                delta = values[index] - self._previous[index]
                if delta < 0:
                    self.counter_decreases.append({**date, "crop": kind, "delta": delta})
                else:
                    row["crop_shipped"][kind] += delta
        self._previous = values

    def report(self) -> dict[str, Any]:
        return {
            "source": "per_frame_ram", "start": self.start, "end": self.end,
            "observations": self.observations, "opening_bin_gold": self.opening_bin_gold,
            "counter_decreases": list(self.counter_decreases),
            "seasons": [{**row, "crop_shipped": dict(row["crop_shipped"])}
                        for row in self._seasons.values()],
        }


def shipping_scene_on_farm(tilemap: int) -> bool:
    """ROM: ``CMP.B #$04 / BCS ShippingReturn`` — farm maps only."""
    return int(tilemap) < 4


def shipping_scene_needs_dismiss(ram) -> bool:
    """True when 5pm ShippingScene is waiting for an A *edge*.

    The CC script (bank_81 ``$096F=2``) idles while ``inputstate==2``. Holding
    A from before the box opens never counts — StartTextBox needs a press.

    ShippingScene runs at hour 17 on farm, but farm+hour>=17 is not enough:
    ``input_lock==0`` is a tool swing / lift / menu and must not pulse A.
    """
    lock = int(read_ram_value(ram, "input_lock", raw=True))
    # Text_Daily_Shipping (0x031A) sticks in dialog_text_id after the box
    # closes. Only pulse while input is actually taken.
    if lock == 1:
        return False
    # lock 0 is the scene still starting (the pending flag is already up, the box is
    # not): an A there sets game_state & 0x0800 for the rest of the day, the bed then
    # ignores every press, and the 8 rescues that follow end the run (Y2 Spring D10).
    # The scene opens its own box (lock 2) a few frames later and takes the A edge.
    if lock == 0:
        return False
    text = int(read_ram_value(ram, "dialog_text_id", raw=True))
    if text in SHIPPING_DIALOG_TEXT_IDS:
        return True
    flags = int(read_ram_value(ram, "event_flags_1f5a", raw=True))
    if flags & SHIPPING_SCENE_PENDING_FLAG:
        return True
    # CC inputstate==2 can be the box before dialog_text_id is written.
    # lock==0 is not that state.
    if lock != 2:
        return False
    tilemap = int(read_ram_value(ram, "tilemap", raw=True))
    hour = int(read_ram_value(ram, "hour"))
    return shipping_scene_on_farm(tilemap) and hour >= SHIPPING_SCENE_HOUR


def money_rose_after_shipping_window(
    *,
    money_pre: int,
    money_post: int,
    shipped_count: int,
    shipping_money_pre: int = 0,
) -> bool:
    """True when bin work shipped and wallet rose after the shipping window."""
    if int(shipped_count) <= 0 and int(shipping_money_pre) <= 0:
        return False
    return int(money_post) > int(money_pre)


def shipping_credit_journal_row(
    *,
    shipped_count: int,
    harvested_count: int = 0,
    money_pre_5pm: int,
    money_post_5pm: int,
    money_post_sleep: int,
    shipping_money_pre_5pm: int = 0,
    shipping_money_post_5pm: int = 0,
    shipping_money_post_sleep: int = 0,
    hour_pre_5pm: Optional[int] = None,
    hour_post_5pm: Optional[int] = None,
    day_pre: Optional[int] = None,
    day_post_sleep: Optional[int] = None,
    pre_5pm_state: str = "",
    post_5pm_state: str = "",
    post_sleep_state: str = "",
    notes: str = "",
) -> dict[str, Any]:
    """Build a journal row documenting harvest ship + post-5pm money settle."""
    rose = money_rose_after_shipping_window(
        money_pre=money_pre_5pm,
        money_post=money_post_sleep,
        shipped_count=shipped_count,
        shipping_money_pre=shipping_money_pre_5pm,
    )
    return {
        "kind": "harvest_ship_5pm_credit",
        "shipped_count": int(shipped_count),
        "harvested_count": int(harvested_count),
        "money_pre_5pm": int(money_pre_5pm),
        "money_post_5pm": int(money_post_5pm),
        "money_post_sleep": int(money_post_sleep),
        "money_delta": int(money_post_sleep) - int(money_pre_5pm),
        "shipping_money_pre_5pm": int(shipping_money_pre_5pm),
        "shipping_money_post_5pm": int(shipping_money_post_5pm),
        "shipping_money_post_sleep": int(shipping_money_post_sleep),
        "hour_pre_5pm": hour_pre_5pm,
        "hour_post_5pm": hour_post_5pm,
        "day_pre": day_pre,
        "day_post_sleep": day_post_sleep,
        "shipping_scene_hour": SHIPPING_SCENE_HOUR,
        "pre_5pm_state": pre_5pm_state,
        "post_5pm_state": post_5pm_state,
        "post_sleep_state": post_sleep_state,
        "money_rose_after_5pm_window": rose,
        "notes": notes
        or (
            "Wallet credits overnight after farm 5pm ShippingScene; "
            "bin drop only bumps shipping_money same-day."
        ),
    }


def acceptance_ok(row: Mapping[str, Any]) -> bool:
    """rr-53g: shipped_count>0 and money rises after 5pm window."""
    shipped = int(row.get("shipped_count") or 0)
    rose = bool(row.get("money_rose_after_5pm_window"))
    if not rose:
        # Recompute if caller omitted the flag.
        rose = money_rose_after_shipping_window(
            money_pre=int(row.get("money_pre_5pm") or 0),
            money_post=int(row.get("money_post_sleep") or 0),
            shipped_count=shipped,
            shipping_money_pre=int(row.get("shipping_money_pre_5pm") or 0),
        )
    return shipped > 0 and rose

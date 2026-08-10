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


SHIPPING_SCENE_HOUR = 17


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
